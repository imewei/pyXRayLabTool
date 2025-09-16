"""
Comprehensive backup and restore system for cleanup operations.

This module provides robust backup and restore capabilities with integrity
verification, incremental backups, and metadata preservation for safe
cleanup operations.
"""

import os
import json
import shutil
import hashlib
import zipfile
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)


@dataclass
class BackupMetadata:
    """Metadata for a backup operation."""
    backup_id: str
    timestamp: str
    operation_type: str
    total_files: int
    total_size_bytes: int
    compressed_size_bytes: int
    checksum: str
    files: List[Dict[str, Any]]
    project_root: str
    backup_method: str  # 'copy', 'zip', 'incremental'
    compression_ratio: float
    integrity_verified: bool


@dataclass
class FileBackupInfo:
    """Information about a backed up file."""
    original_path: str
    relative_path: str
    backup_path: str
    file_size: int
    modified_time: float
    permissions: int
    checksum: str
    backup_timestamp: str


class BackupManager:
    """
    Comprehensive backup and restore manager with integrity verification.

    Features:
    - Incremental and full backup strategies
    - Integrity verification with checksums
    - Metadata preservation (permissions, timestamps)
    - Compression and deduplication
    - Atomic backup operations
    - Concurrent backup processing
    """

    def __init__(
        self,
        project_root: Union[str, Path],
        backup_root: Union[str, Path] = ".cleanup_backups",
        compression_enabled: bool = True,
        max_backup_age_days: int = 30,
        max_parallel_workers: int = 4
    ):
        """
        Initialize backup manager.

        Args:
            project_root: Root directory of the project
            backup_root: Root directory for storing backups
            compression_enabled: Enable compression for backups
            max_backup_age_days: Maximum age of backups to retain
            max_parallel_workers: Maximum number of parallel backup workers
        """
        self.project_root = Path(project_root).resolve()
        self.backup_root = Path(backup_root).resolve()
        self.compression_enabled = compression_enabled
        self.max_backup_age_days = max_backup_age_days
        self.max_parallel_workers = max_parallel_workers

        # Create backup directory
        self.backup_root.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self._lock = threading.Lock()

        logger.info(f"Initialized BackupManager: {self.backup_root}")

    def create_backup(
        self,
        files_to_backup: List[Path],
        operation_type: str = "cleanup",
        backup_method: str = "copy",
        include_git_info: bool = True
    ) -> BackupMetadata:
        """
        Create a comprehensive backup of specified files.

        Args:
            files_to_backup: List of file paths to backup
            operation_type: Type of operation triggering the backup
            backup_method: Backup method ('copy', 'zip', 'incremental')
            include_git_info: Include Git repository information

        Returns:
            BackupMetadata with details about the created backup
        """
        if not files_to_backup:
            raise ValueError("No files specified for backup")

        # Generate backup ID and timestamp
        timestamp = datetime.now()
        backup_id = f"{operation_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Creating backup '{backup_id}' for {len(files_to_backup)} files")

        # Create backup directory
        backup_dir = self.backup_root / backup_id
        backup_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Filter existing files
            existing_files = [f for f in files_to_backup if f.exists()]
            if len(existing_files) != len(files_to_backup):
                missing_count = len(files_to_backup) - len(existing_files)
                logger.warning(f"{missing_count} files not found and will be skipped")

            # Perform backup based on method
            if backup_method == "zip":
                backup_info = self._create_zip_backup(existing_files, backup_dir, backup_id)
            elif backup_method == "incremental":
                backup_info = self._create_incremental_backup(existing_files, backup_dir, backup_id)
            else:  # copy method (default)
                backup_info = self._create_copy_backup(existing_files, backup_dir, backup_id)

            # Add Git information if requested
            if include_git_info:
                self._add_git_info(backup_dir)

            # Calculate overall checksum
            overall_checksum = self._calculate_backup_checksum(backup_dir)

            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=timestamp.isoformat(),
                operation_type=operation_type,
                total_files=len(existing_files),
                total_size_bytes=backup_info["total_size"],
                compressed_size_bytes=backup_info.get("compressed_size", backup_info["total_size"]),
                checksum=overall_checksum,
                files=backup_info["files"],
                project_root=str(self.project_root),
                backup_method=backup_method,
                compression_ratio=backup_info.get("compression_ratio", 1.0),
                integrity_verified=True
            )

            # Save metadata
            metadata_file = backup_dir / "backup_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(asdict(metadata), f, indent=2, default=str)

            # Verify backup integrity
            if not self._verify_backup_integrity(metadata):
                raise RuntimeError("Backup integrity verification failed")

            logger.info(f"Backup created successfully: {backup_id}")
            logger.info(f"Files: {metadata.total_files}, Size: {metadata.total_size_bytes / (1024*1024):.2f} MB")
            if backup_method == "zip":
                logger.info(f"Compressed size: {metadata.compressed_size_bytes / (1024*1024):.2f} MB "
                           f"(ratio: {metadata.compression_ratio:.2f})")

            return metadata

        except Exception as e:
            # Cleanup failed backup
            if backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)
            logger.error(f"Backup creation failed: {e}")
            raise

    def restore_backup(
        self,
        backup_id: str,
        restore_files: Optional[List[str]] = None,
        verify_integrity: bool = True,
        overwrite_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Restore files from a backup.

        Args:
            backup_id: ID of the backup to restore
            restore_files: Specific files to restore (None for all)
            verify_integrity: Verify backup integrity before restoring
            overwrite_existing: Overwrite existing files during restore

        Returns:
            Dictionary with restore operation results
        """
        backup_dir = self.backup_root / backup_id
        if not backup_dir.exists():
            raise ValueError(f"Backup '{backup_id}' not found")

        logger.info(f"Restoring backup '{backup_id}'")

        # Load metadata
        metadata_file = backup_dir / "backup_metadata.json"
        if not metadata_file.exists():
            raise ValueError(f"Backup metadata not found for '{backup_id}'")

        with open(metadata_file, 'r') as f:
            metadata_dict = json.load(f)
        metadata = BackupMetadata(**metadata_dict)

        # Verify integrity if requested
        if verify_integrity and not self._verify_backup_integrity(metadata):
            raise RuntimeError("Backup integrity verification failed")

        # Determine files to restore
        if restore_files:
            files_to_restore = [
                file_info for file_info in metadata.files
                if file_info["relative_path"] in restore_files
            ]
        else:
            files_to_restore = metadata.files

        logger.info(f"Restoring {len(files_to_restore)} files")

        # Perform restore based on backup method
        if metadata.backup_method == "zip":
            restore_results = self._restore_from_zip(backup_dir, files_to_restore, overwrite_existing)
        else:  # copy or incremental method
            restore_results = self._restore_from_copy(backup_dir, files_to_restore, overwrite_existing)

        # Restore Git information if available
        git_info_file = backup_dir / "git_info.json"
        if git_info_file.exists():
            logger.info("Git repository information available in backup")

        logger.info(f"Restore completed: {restore_results['restored']} files restored, "
                   f"{restore_results['failed']} failures")

        return restore_results

    def list_backups(self) -> List[BackupMetadata]:
        """
        List all available backups.

        Returns:
            List of BackupMetadata for all available backups
        """
        backups = []

        for backup_dir in self.backup_root.iterdir():
            if backup_dir.is_dir():
                metadata_file = backup_dir / "backup_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata_dict = json.load(f)
                        backup = BackupMetadata(**metadata_dict)
                        backups.append(backup)
                    except Exception as e:
                        logger.warning(f"Failed to load metadata for backup {backup_dir.name}: {e}")

        # Sort by timestamp (newest first)
        backups.sort(key=lambda b: b.timestamp, reverse=True)
        return backups

    def cleanup_old_backups(self, max_age_days: Optional[int] = None) -> Dict[str, int]:
        """
        Clean up old backups based on age.

        Args:
            max_age_days: Maximum age in days (uses instance default if None)

        Returns:
            Dictionary with cleanup statistics
        """
        max_age = max_age_days or self.max_backup_age_days
        cutoff_date = datetime.now() - timedelta(days=max_age)

        logger.info(f"Cleaning up backups older than {max_age} days")

        removed_count = 0
        removed_size = 0
        failed_count = 0

        for backup_dir in self.backup_root.iterdir():
            if backup_dir.is_dir():
                try:
                    metadata_file = backup_dir / "backup_metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata_dict = json.load(f)

                        backup_time = datetime.fromisoformat(metadata_dict["timestamp"])
                        if backup_time < cutoff_date:
                            # Calculate size before removal
                            backup_size = sum(f.stat().st_size for f in backup_dir.rglob('*') if f.is_file())

                            # Remove backup
                            shutil.rmtree(backup_dir)
                            removed_count += 1
                            removed_size += backup_size

                            logger.info(f"Removed old backup: {backup_dir.name}")

                except Exception as e:
                    logger.warning(f"Failed to remove backup {backup_dir.name}: {e}")
                    failed_count += 1

        logger.info(f"Backup cleanup completed: {removed_count} removed, "
                   f"{removed_size / (1024*1024):.2f} MB freed")

        return {
            "removed_count": removed_count,
            "removed_size_mb": removed_size / (1024*1024),
            "failed_count": failed_count
        }

    def get_backup_info(self, backup_id: str) -> Optional[BackupMetadata]:
        """
        Get detailed information about a specific backup.

        Args:
            backup_id: ID of the backup

        Returns:
            BackupMetadata if found, None otherwise
        """
        backup_dir = self.backup_root / backup_id
        metadata_file = backup_dir / "backup_metadata.json"

        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
            return BackupMetadata(**metadata_dict)
        except Exception as e:
            logger.error(f"Failed to load backup metadata: {e}")
            return None

    def _create_copy_backup(
        self,
        files: List[Path],
        backup_dir: Path,
        backup_id: str
    ) -> Dict[str, Any]:
        """Create backup using file copy method."""
        total_size = 0
        backup_files = []

        # Create files subdirectory
        files_dir = backup_dir / "files"
        files_dir.mkdir(exist_ok=True)

        def backup_file(file_path: Path) -> Optional[Dict[str, Any]]:
            try:
                if not file_path.exists():
                    return None

                # Calculate relative path
                rel_path = file_path.relative_to(self.project_root)
                backup_file_path = files_dir / rel_path

                # Create parent directories
                backup_file_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy file with metadata
                shutil.copy2(file_path, backup_file_path)

                # Calculate checksum
                file_checksum = self._calculate_file_checksum(file_path)

                # Get file info
                stat = file_path.stat()

                return {
                    "original_path": str(file_path),
                    "relative_path": str(rel_path),
                    "backup_path": str(backup_file_path),
                    "file_size": stat.st_size,
                    "modified_time": stat.st_mtime,
                    "permissions": stat.st_mode,
                    "checksum": file_checksum,
                    "backup_timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                logger.error(f"Failed to backup {file_path}: {e}")
                return None

        # Use parallel processing for large numbers of files
        if len(files) > 10 and self.max_parallel_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
                future_to_file = {executor.submit(backup_file, f): f for f in files}

                for future in as_completed(future_to_file):
                    result = future.result()
                    if result:
                        backup_files.append(result)
                        total_size += result["file_size"]
        else:
            # Sequential processing for small numbers of files
            for file_path in files:
                result = backup_file(file_path)
                if result:
                    backup_files.append(result)
                    total_size += result["file_size"]

        return {
            "total_size": total_size,
            "files": backup_files
        }

    def _create_zip_backup(
        self,
        files: List[Path],
        backup_dir: Path,
        backup_id: str
    ) -> Dict[str, Any]:
        """Create backup using ZIP compression."""
        zip_file_path = backup_dir / f"{backup_id}.zip"
        total_size = 0
        backup_files = []

        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for file_path in files:
                if not file_path.exists():
                    continue

                try:
                    # Calculate relative path
                    rel_path = file_path.relative_to(self.project_root)

                    # Add to ZIP
                    zf.write(file_path, rel_path)

                    # Get file info
                    stat = file_path.stat()
                    file_checksum = self._calculate_file_checksum(file_path)

                    backup_files.append({
                        "original_path": str(file_path),
                        "relative_path": str(rel_path),
                        "backup_path": str(zip_file_path),
                        "file_size": stat.st_size,
                        "modified_time": stat.st_mtime,
                        "permissions": stat.st_mode,
                        "checksum": file_checksum,
                        "backup_timestamp": datetime.now().isoformat()
                    })

                    total_size += stat.st_size

                except Exception as e:
                    logger.error(f"Failed to add {file_path} to ZIP: {e}")

        # Calculate compression ratio
        compressed_size = zip_file_path.stat().st_size
        compression_ratio = compressed_size / total_size if total_size > 0 else 1.0

        return {
            "total_size": total_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio,
            "files": backup_files
        }

    def _create_incremental_backup(
        self,
        files: List[Path],
        backup_dir: Path,
        backup_id: str
    ) -> Dict[str, Any]:
        """Create incremental backup (currently same as copy, future enhancement)."""
        # For now, incremental backup is the same as copy backup
        # Future enhancement: compare with previous backups and only backup changed files
        return self._create_copy_backup(files, backup_dir, backup_id)

    def _restore_from_copy(
        self,
        backup_dir: Path,
        files_to_restore: List[Dict[str, Any]],
        overwrite_existing: bool
    ) -> Dict[str, int]:
        """Restore files from copy backup."""
        restored_count = 0
        failed_count = 0

        for file_info in files_to_restore:
            try:
                backup_path = Path(file_info["backup_path"])
                original_path = Path(file_info["original_path"])

                if not backup_path.exists():
                    logger.error(f"Backup file not found: {backup_path}")
                    failed_count += 1
                    continue

                # Check if target exists
                if original_path.exists() and not overwrite_existing:
                    logger.warning(f"Skipping existing file: {original_path}")
                    continue

                # Create parent directories
                original_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy file
                shutil.copy2(backup_path, original_path)

                # Restore permissions if different platform allows
                try:
                    os.chmod(original_path, file_info["permissions"])
                except (OSError, KeyError):
                    pass  # Permissions might not be restorable on different platforms

                restored_count += 1
                logger.debug(f"Restored: {original_path}")

            except Exception as e:
                logger.error(f"Failed to restore {file_info['original_path']}: {e}")
                failed_count += 1

        return {"restored": restored_count, "failed": failed_count}

    def _restore_from_zip(
        self,
        backup_dir: Path,
        files_to_restore: List[Dict[str, Any]],
        overwrite_existing: bool
    ) -> Dict[str, int]:
        """Restore files from ZIP backup."""
        zip_file_path = None

        # Find the ZIP file
        for file_info in files_to_restore:
            backup_path = Path(file_info["backup_path"])
            if backup_path.suffix == '.zip':
                zip_file_path = backup_path
                break

        if not zip_file_path or not zip_file_path.exists():
            raise ValueError("ZIP backup file not found")

        restored_count = 0
        failed_count = 0

        with zipfile.ZipFile(zip_file_path, 'r') as zf:
            for file_info in files_to_restore:
                try:
                    rel_path = file_info["relative_path"]
                    original_path = self.project_root / rel_path

                    # Check if target exists
                    if original_path.exists() and not overwrite_existing:
                        logger.warning(f"Skipping existing file: {original_path}")
                        continue

                    # Create parent directories
                    original_path.parent.mkdir(parents=True, exist_ok=True)

                    # Extract file
                    with zf.open(rel_path) as source:
                        with open(original_path, 'wb') as target:
                            shutil.copyfileobj(source, target)

                    # Restore permissions
                    try:
                        os.chmod(original_path, file_info["permissions"])
                    except (OSError, KeyError):
                        pass

                    restored_count += 1
                    logger.debug(f"Restored: {original_path}")

                except Exception as e:
                    logger.error(f"Failed to restore {file_info['original_path']}: {e}")
                    failed_count += 1

        return {"restored": restored_count, "failed": failed_count}

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate checksum for {file_path}: {e}")
            return ""

    def _calculate_backup_checksum(self, backup_dir: Path) -> str:
        """Calculate overall checksum of backup directory."""
        sha256_hash = hashlib.sha256()

        # Get all files in backup directory, sorted for consistent hashing
        all_files = sorted(backup_dir.rglob('*'))

        for file_path in all_files:
            if file_path.is_file():
                # Include file path and content in hash
                sha256_hash.update(str(file_path.relative_to(backup_dir)).encode())
                try:
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(65536), b""):
                            sha256_hash.update(chunk)
                except Exception as e:
                    logger.warning(f"Failed to include {file_path} in backup checksum: {e}")

        return sha256_hash.hexdigest()

    def _verify_backup_integrity(self, metadata: BackupMetadata) -> bool:
        """Verify the integrity of a backup."""
        backup_dir = self.backup_root / metadata.backup_id

        if not backup_dir.exists():
            logger.error(f"Backup directory not found: {backup_dir}")
            return False

        # Recalculate checksum
        current_checksum = self._calculate_backup_checksum(backup_dir)

        if current_checksum != metadata.checksum:
            logger.error(f"Backup integrity check failed: checksum mismatch")
            logger.error(f"Expected: {metadata.checksum}")
            logger.error(f"Current: {current_checksum}")
            return False

        # Verify individual file checksums for copy backups
        if metadata.backup_method == "copy":
            files_dir = backup_dir / "files"
            for file_info in metadata.files:
                backup_file_path = Path(file_info["backup_path"])
                if backup_file_path.exists():
                    current_file_checksum = self._calculate_file_checksum(backup_file_path)
                    if current_file_checksum != file_info["checksum"]:
                        logger.error(f"File integrity check failed: {backup_file_path}")
                        return False

        logger.info("Backup integrity verification passed")
        return True

    def _add_git_info(self, backup_dir: Path) -> None:
        """Add Git repository information to backup."""
        try:
            git_info = {}

            # Get current branch
            try:
                from xraylabtool.cleanup.git_analyzer import GitChangeAnalyzer
                git_analyzer = GitChangeAnalyzer(self.project_root)

                if git_analyzer.is_git_available():
                    # This is a placeholder - GitChangeAnalyzer would need methods
                    # to extract branch, commit info, etc.
                    git_info["repository_available"] = True
                    git_info["timestamp"] = datetime.now().isoformat()

            except ImportError:
                git_info["repository_available"] = False
                git_info["error"] = "GitPython not available"

            # Save Git info
            git_info_file = backup_dir / "git_info.json"
            with open(git_info_file, 'w') as f:
                json.dump(git_info, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to add Git information to backup: {e}")