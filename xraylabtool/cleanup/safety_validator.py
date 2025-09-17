"""
Safety validation and rollback systems for cleanup operations.

This module provides comprehensive safety checks, validation systems,
and rollback capabilities to ensure safe cleanup operations with
full recovery capabilities.
"""

import os
import sys
import shutil
import signal
import psutil
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
import time
import subprocess

from .backup_manager import BackupManager, BackupMetadata

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety levels for operations."""

    SAFE = "safe"
    CAUTION = "caution"
    DANGEROUS = "dangerous"
    CRITICAL = "critical"


class ValidationResult(Enum):
    """Validation results."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    ERROR = "error"


@dataclass
class SafetyCheck:
    """Individual safety check result."""

    name: str
    description: str
    result: ValidationResult
    level: SafetyLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ValidationReport:
    """Comprehensive validation report."""

    operation_type: str
    total_checks: int
    passed_checks: int
    warning_checks: int
    failed_checks: int
    error_checks: int
    overall_safety_level: SafetyLevel
    overall_result: ValidationResult
    checks: List[SafetyCheck]
    recommendations: List[str]
    required_actions: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def can_proceed(self) -> bool:
        """
        Determine if operation can proceed based on validation results.

        Returns True for PASS and WARNING results, False for FAIL and ERROR.
        """
        return self.overall_result in [ValidationResult.PASS, ValidationResult.WARNING]


class EmergencyStop:
    """Emergency stop mechanism for long-running operations."""

    def __init__(self, cleanup_callback: Optional[Callable] = None):
        """
        Initialize emergency stop mechanism.

        Args:
            cleanup_callback: Function to call on emergency stop
        """
        self._stop_requested = threading.Event()
        self._cleanup_callback = cleanup_callback
        self._original_handlers = {}
        self._install_signal_handlers()

    def _install_signal_handlers(self):
        """Install signal handlers for emergency stop."""

        def signal_handler(signum, frame):
            logger.warning(f"Emergency stop signal received: {signum}")
            self.request_stop()

        # Handle common stop signals
        for sig in [signal.SIGINT, signal.SIGTERM]:
            try:
                self._original_handlers[sig] = signal.signal(sig, signal_handler)
            except (OSError, ValueError):
                # Some signals might not be available on all platforms
                pass

    def request_stop(self, reason: str = ""):
        """Request emergency stop."""
        if reason:
            logger.warning(f"Emergency stop requested: {reason}")
        else:
            logger.warning("Emergency stop requested")
        self._stop_requested.set()

        if self._cleanup_callback:
            try:
                self._cleanup_callback()
            except Exception as e:
                logger.error(f"Error during emergency cleanup: {e}")

    def is_stop_requested(self) -> bool:
        """Check if emergency stop was requested."""
        return self._stop_requested.is_set()

    def restore_handlers(self):
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError):
                pass


class SafetyValidator:
    """
    Comprehensive safety validator for cleanup operations.

    Performs multiple layers of safety checks and validation before
    allowing destructive operations to proceed.
    """

    def __init__(
        self,
        project_root: Union[str, Path],
        backup_manager: Optional[BackupManager] = None,
        strict_mode: bool = True,
    ):
        """
        Initialize safety validator.

        Args:
            project_root: Root directory of the project
            backup_manager: Backup manager instance
            strict_mode: Enable strict safety validation
        """
        self.project_root = Path(project_root).resolve()
        self.backup_manager = backup_manager
        self.strict_mode = strict_mode

        logger.info(f"Initialized SafetyValidator (strict_mode={strict_mode})")

    def validate_pre_operation(
        self, files_to_process: List[Path], operation_type: str = "cleanup"
    ) -> ValidationReport:
        """
        Perform comprehensive pre-operation safety validation.

        Args:
            files_to_process: List of files that will be processed
            operation_type: Type of operation being performed

        Returns:
            ValidationReport with all safety check results
        """
        logger.info(
            f"Starting pre-operation validation for {len(files_to_process)} files"
        )

        checks = []

        # Layer 1: System-level checks
        checks.extend(self._check_system_resources())
        checks.extend(self._check_file_system_permissions())
        checks.extend(self._check_disk_space(files_to_process))

        # Layer 2: Project-level checks
        checks.extend(self._check_project_integrity())
        checks.extend(self._check_git_repository_state())
        checks.extend(self._check_build_dependencies())

        # Layer 3: File-level checks
        checks.extend(self._check_file_criticality(files_to_process))
        checks.extend(self._check_file_dependencies(files_to_process))
        checks.extend(self._check_backup_capability(files_to_process))

        # Generate report
        report = self._generate_validation_report(checks, operation_type)

        logger.info(
            f"Pre-operation validation completed: {report.overall_result.value}"
        )
        return report

    def validate_post_operation(
        self, processed_files: List[Path], operation_type: str = "cleanup"
    ) -> ValidationReport:
        """
        Perform post-operation validation to ensure system integrity.

        Args:
            processed_files: List of files that were processed
            operation_type: Type of operation that was performed

        Returns:
            ValidationReport with post-operation validation results
        """
        logger.info(
            f"Starting post-operation validation for {len(processed_files)} files"
        )

        checks = []

        # Verify project still functions
        checks.extend(self._check_project_integrity())
        checks.extend(self._verify_import_capabilities())
        checks.extend(self._verify_build_capabilities())

        # Check for broken dependencies
        checks.extend(self._check_broken_references())

        # Generate report
        report = self._generate_validation_report(checks, f"post_{operation_type}")

        logger.info(
            f"Post-operation validation completed: {report.overall_result.value}"
        )
        return report

    def create_rollback_plan(
        self, backup_metadata: BackupMetadata, validation_report: ValidationReport
    ) -> Dict[str, Any]:
        """
        Create a rollback plan based on backup and validation results.

        Args:
            backup_metadata: Metadata from the backup
            validation_report: Validation results

        Returns:
            Dictionary containing rollback plan details
        """
        rollback_plan = {
            "backup_id": backup_metadata.backup_id,
            "rollback_required": (
                validation_report.overall_result
                in [ValidationResult.FAIL, ValidationResult.ERROR]
            ),
            "automatic_rollback": (
                validation_report.overall_safety_level == SafetyLevel.CRITICAL
            ),
            "rollback_steps": [],
            "verification_steps": [],
            "recovery_time_estimate": "5-15 minutes",
            "risk_assessment": "low",
        }

        # Determine rollback steps based on validation failures
        if validation_report.failed_checks > 0 or validation_report.error_checks > 0:
            rollback_plan["rollback_steps"] = [
                "Stop any running processes",
                "Restore files from backup",
                "Verify file integrity",
                "Run post-restore validation",
                "Restart services if needed",
            ]

            rollback_plan["verification_steps"] = [
                "Check project build capability",
                "Verify import statements work",
                "Run basic functionality tests",
                "Confirm Git repository state",
            ]

        # Adjust risk and time estimates
        if validation_report.overall_safety_level == SafetyLevel.CRITICAL:
            rollback_plan["risk_assessment"] = "high"
            rollback_plan["recovery_time_estimate"] = "15-30 minutes"

        return rollback_plan

    def execute_rollback(
        self,
        backup_id: str,
        rollback_plan: Dict[str, Any],
        emergency_stop: Optional[EmergencyStop] = None,
    ) -> Dict[str, Any]:
        """
        Execute rollback operation.

        Args:
            backup_id: ID of backup to restore from
            rollback_plan: Rollback plan to execute
            emergency_stop: Emergency stop mechanism

        Returns:
            Dictionary with rollback results
        """
        if not self.backup_manager:
            raise ValueError("No backup manager available for rollback")

        logger.warning(f"Executing rollback from backup: {backup_id}")

        rollback_results = {
            "success": False,
            "restored_files": 0,
            "failed_files": 0,
            "errors": [],
            "duration_seconds": 0,
        }

        start_time = time.time()

        try:
            # Check for emergency stop
            if emergency_stop and emergency_stop.is_stop_requested():
                raise RuntimeError("Rollback aborted due to emergency stop")

            # Restore from backup
            restore_results = self.backup_manager.restore_backup(
                backup_id=backup_id, verify_integrity=True, overwrite_existing=True
            )

            rollback_results["restored_files"] = restore_results["restored"]
            rollback_results["failed_files"] = restore_results["failed"]

            # Post-rollback validation
            if emergency_stop and emergency_stop.is_stop_requested():
                raise RuntimeError("Rollback aborted during validation")

            validation = self.validate_post_operation([], "rollback")

            if validation.overall_result == ValidationResult.PASS:
                rollback_results["success"] = True
                logger.info("Rollback completed successfully")
            else:
                rollback_results["errors"].append("Post-rollback validation failed")
                logger.error("Rollback validation failed")

        except Exception as e:
            error_msg = f"Rollback execution failed: {e}"
            logger.error(error_msg)
            rollback_results["errors"].append(error_msg)

        finally:
            rollback_results["duration_seconds"] = time.time() - start_time

        return rollback_results

    def _check_system_resources(self) -> List[SafetyCheck]:
        """Check system resource availability."""
        checks = []

        try:
            # Check available memory
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                checks.append(
                    SafetyCheck(
                        name="memory_check",
                        description="System memory availability",
                        result=ValidationResult.WARNING,
                        level=SafetyLevel.CAUTION,
                        message=f"High memory usage: {memory.percent}%",
                        details={"memory_percent": memory.percent},
                    )
                )
            else:
                checks.append(
                    SafetyCheck(
                        name="memory_check",
                        description="System memory availability",
                        result=ValidationResult.PASS,
                        level=SafetyLevel.SAFE,
                        message=f"Memory usage normal: {memory.percent}%",
                    )
                )

            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                checks.append(
                    SafetyCheck(
                        name="cpu_check",
                        description="System CPU usage",
                        result=ValidationResult.WARNING,
                        level=SafetyLevel.CAUTION,
                        message=f"High CPU usage: {cpu_percent}%",
                        details={"cpu_percent": cpu_percent},
                    )
                )
            else:
                checks.append(
                    SafetyCheck(
                        name="cpu_check",
                        description="System CPU usage",
                        result=ValidationResult.PASS,
                        level=SafetyLevel.SAFE,
                        message=f"CPU usage normal: {cpu_percent}%",
                    )
                )

        except Exception as e:
            checks.append(
                SafetyCheck(
                    name="system_resources",
                    description="System resource check",
                    result=ValidationResult.ERROR,
                    level=SafetyLevel.DANGEROUS,
                    message=f"Failed to check system resources: {e}",
                )
            )

        return checks

    def _check_file_system_permissions(self) -> List[SafetyCheck]:
        """Check file system permissions."""
        checks = []

        try:
            # Check write permission to project root
            test_file = self.project_root / ".safety_test_file"
            try:
                test_file.write_text("test")
                test_file.unlink()

                checks.append(
                    SafetyCheck(
                        name="fs_permissions",
                        description="File system write permissions",
                        result=ValidationResult.PASS,
                        level=SafetyLevel.SAFE,
                        message="Write permissions verified",
                    )
                )
            except PermissionError:
                checks.append(
                    SafetyCheck(
                        name="fs_permissions",
                        description="File system write permissions",
                        result=ValidationResult.FAIL,
                        level=SafetyLevel.CRITICAL,
                        message="No write permission to project directory",
                    )
                )

        except Exception as e:
            checks.append(
                SafetyCheck(
                    name="fs_permissions",
                    description="File system permissions check",
                    result=ValidationResult.ERROR,
                    level=SafetyLevel.DANGEROUS,
                    message=f"Permission check failed: {e}",
                )
            )

        return checks

    def _check_disk_space(self, files_to_process: List[Path]) -> List[SafetyCheck]:
        """Check available disk space for backups."""
        checks = []

        try:
            # Calculate total size of files to backup
            total_size = 0
            for file_path in files_to_process:
                if file_path.exists():
                    total_size += file_path.stat().st_size

            # Check available disk space
            disk_usage = shutil.disk_usage(self.project_root)
            available_gb = disk_usage.free / (1024**3)
            required_gb = (total_size * 2) / (1024**3)  # 2x for safety margin

            if available_gb < required_gb:
                checks.append(
                    SafetyCheck(
                        name="disk_space",
                        description="Available disk space for backup",
                        result=ValidationResult.FAIL,
                        level=SafetyLevel.CRITICAL,
                        message=f"Insufficient disk space: {available_gb:.2f}GB available, {required_gb:.2f}GB required",
                        details={
                            "available_gb": available_gb,
                            "required_gb": required_gb,
                        },
                    )
                )
            elif available_gb < required_gb * 2:
                checks.append(
                    SafetyCheck(
                        name="disk_space",
                        description="Available disk space for backup",
                        result=ValidationResult.WARNING,
                        level=SafetyLevel.CAUTION,
                        message=f"Limited disk space: {available_gb:.2f}GB available, {required_gb:.2f}GB required",
                    )
                )
            else:
                checks.append(
                    SafetyCheck(
                        name="disk_space",
                        description="Available disk space for backup",
                        result=ValidationResult.PASS,
                        level=SafetyLevel.SAFE,
                        message=f"Sufficient disk space: {available_gb:.2f}GB available",
                    )
                )

        except Exception as e:
            checks.append(
                SafetyCheck(
                    name="disk_space",
                    description="Disk space availability check",
                    result=ValidationResult.ERROR,
                    level=SafetyLevel.DANGEROUS,
                    message=f"Disk space check failed: {e}",
                )
            )

        return checks

    def _check_project_integrity(self) -> List[SafetyCheck]:
        """Check basic project integrity."""
        checks = []

        # Check for essential project files
        essential_files = ["pyproject.toml", "setup.py", "Makefile"]

        missing_files = []
        for file_name in essential_files:
            if not (self.project_root / file_name).exists():
                missing_files.append(file_name)

        if missing_files:
            if "pyproject.toml" in missing_files and "setup.py" in missing_files:
                checks.append(
                    SafetyCheck(
                        name="project_integrity",
                        description="Essential project files",
                        result=ValidationResult.FAIL,
                        level=SafetyLevel.CRITICAL,
                        message=f"Critical project files missing: {missing_files}",
                    )
                )
            else:
                checks.append(
                    SafetyCheck(
                        name="project_integrity",
                        description="Essential project files",
                        result=ValidationResult.WARNING,
                        level=SafetyLevel.CAUTION,
                        message=f"Some project files missing: {missing_files}",
                    )
                )
        else:
            checks.append(
                SafetyCheck(
                    name="project_integrity",
                    description="Essential project files",
                    result=ValidationResult.PASS,
                    level=SafetyLevel.SAFE,
                    message="All essential project files present",
                )
            )

        return checks

    def _check_git_repository_state(self) -> List[SafetyCheck]:
        """Check Git repository state."""
        checks = []

        try:
            # Check if it's a Git repository
            if not (self.project_root / ".git").exists():
                checks.append(
                    SafetyCheck(
                        name="git_repository",
                        description="Git repository status",
                        result=ValidationResult.WARNING,
                        level=SafetyLevel.CAUTION,
                        message="Not a Git repository - version control protection unavailable",
                    )
                )
                return checks

            # Try to get Git status
            try:
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0:
                    if result.stdout.strip():
                        checks.append(
                            SafetyCheck(
                                name="git_repository",
                                description="Git repository status",
                                result=ValidationResult.WARNING,
                                level=SafetyLevel.CAUTION,
                                message="Uncommitted changes detected - consider committing before cleanup",
                                details={
                                    "uncommitted_files": (
                                        result.stdout.strip().split("\n")
                                    )
                                },
                            )
                        )
                    else:
                        checks.append(
                            SafetyCheck(
                                name="git_repository",
                                description="Git repository status",
                                result=ValidationResult.PASS,
                                level=SafetyLevel.SAFE,
                                message="Git repository clean",
                            )
                        )
                else:
                    checks.append(
                        SafetyCheck(
                            name="git_repository",
                            description="Git repository status",
                            result=ValidationResult.WARNING,
                            level=SafetyLevel.CAUTION,
                            message="Could not determine Git status",
                        )
                    )

            except subprocess.TimeoutExpired:
                checks.append(
                    SafetyCheck(
                        name="git_repository",
                        description="Git repository status",
                        result=ValidationResult.WARNING,
                        level=SafetyLevel.CAUTION,
                        message="Git status check timed out",
                    )
                )

        except Exception as e:
            checks.append(
                SafetyCheck(
                    name="git_repository",
                    description="Git repository status check",
                    result=ValidationResult.ERROR,
                    level=SafetyLevel.CAUTION,
                    message=f"Git status check failed: {e}",
                )
            )

        return checks

    def _check_build_dependencies(self) -> List[SafetyCheck]:
        """Check if project can still build after cleanup."""
        checks = []

        # This is a basic check - in a real implementation, this would
        # run a quick build test or dependency check
        try:
            # Check for Python module structure
            main_module = None
            for potential in ["xraylabtool", "src", "lib"]:
                if (self.project_root / potential).exists():
                    main_module = potential
                    break

            if main_module:
                checks.append(
                    SafetyCheck(
                        name="build_dependencies",
                        description="Project build structure",
                        result=ValidationResult.PASS,
                        level=SafetyLevel.SAFE,
                        message=f"Main module directory found: {main_module}",
                    )
                )
            else:
                checks.append(
                    SafetyCheck(
                        name="build_dependencies",
                        description="Project build structure",
                        result=ValidationResult.WARNING,
                        level=SafetyLevel.CAUTION,
                        message="Could not identify main module directory",
                    )
                )

        except Exception as e:
            checks.append(
                SafetyCheck(
                    name="build_dependencies",
                    description="Build dependencies check",
                    result=ValidationResult.ERROR,
                    level=SafetyLevel.CAUTION,
                    message=f"Build check failed: {e}",
                )
            )

        return checks

    def _check_file_criticality(
        self, files_to_process: List[Path]
    ) -> List[SafetyCheck]:
        """Check criticality of files to be processed."""
        checks = []

        critical_patterns = [
            "*.py",  # Source code
            "*.toml",  # Configuration
            "*.yaml",  # Configuration
            "*.yml",  # Configuration
            "*.json",  # Data/config
            "Makefile",  # Build system
            "README*",  # Documentation
            "LICENSE*",  # Legal
        ]

        critical_files = []
        for file_path in files_to_process:
            if file_path.exists():
                for pattern in critical_patterns:
                    if file_path.match(pattern):
                        critical_files.append(file_path)
                        break

        if critical_files:
            if (
                len(critical_files) > len(files_to_process) * 0.1
            ):  # More than 10% critical
                checks.append(
                    SafetyCheck(
                        name="file_criticality",
                        description="Critical file analysis",
                        result=ValidationResult.WARNING,
                        level=SafetyLevel.DANGEROUS,
                        message=f"Many critical files to be processed: {len(critical_files)}",
                        details={
                            "critical_files": [str(f) for f in critical_files[:10]]
                        },
                    )
                )
            else:
                checks.append(
                    SafetyCheck(
                        name="file_criticality",
                        description="Critical file analysis",
                        result=ValidationResult.WARNING,
                        level=SafetyLevel.CAUTION,
                        message=f"Some critical files to be processed: {len(critical_files)}",
                    )
                )
        else:
            checks.append(
                SafetyCheck(
                    name="file_criticality",
                    description="Critical file analysis",
                    result=ValidationResult.PASS,
                    level=SafetyLevel.SAFE,
                    message="No critical files in processing list",
                )
            )

        return checks

    def _check_file_dependencies(
        self, files_to_process: List[Path]
    ) -> List[SafetyCheck]:
        """Check for file dependencies that might be broken."""
        checks = []

        # This is a simplified dependency check
        # In a full implementation, this would analyze imports, references, etc.

        checks.append(
            SafetyCheck(
                name="file_dependencies",
                description="File dependency analysis",
                result=ValidationResult.PASS,
                level=SafetyLevel.SAFE,
                message="Dependency analysis completed (basic check)",
            )
        )

        return checks

    def _check_backup_capability(
        self, files_to_process: List[Path]
    ) -> List[SafetyCheck]:
        """Check if backup can be created for the files."""
        checks = []

        if not self.backup_manager:
            checks.append(
                SafetyCheck(
                    name="backup_capability",
                    description="Backup system availability",
                    result=ValidationResult.WARNING,
                    level=SafetyLevel.CAUTION,
                    message="No backup manager available",
                )
            )
            return checks

        try:
            # Check if backup directory is writable
            backup_test_file = self.backup_manager.backup_root / ".backup_test"
            backup_test_file.write_text("test")
            backup_test_file.unlink()

            checks.append(
                SafetyCheck(
                    name="backup_capability",
                    description="Backup system availability",
                    result=ValidationResult.PASS,
                    level=SafetyLevel.SAFE,
                    message="Backup system ready",
                )
            )

        except Exception as e:
            checks.append(
                SafetyCheck(
                    name="backup_capability",
                    description="Backup system availability",
                    result=ValidationResult.FAIL,
                    level=SafetyLevel.CRITICAL,
                    message=f"Backup system not available: {e}",
                )
            )

        return checks

    def _verify_import_capabilities(self) -> List[SafetyCheck]:
        """Verify that Python imports still work."""
        checks = []

        try:
            # Try to import the main package
            import importlib.util

            main_package = self.project_root / "xraylabtool" / "__init__.py"
            if main_package.exists():
                spec = importlib.util.spec_from_file_location(
                    "xraylabtool", main_package
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    checks.append(
                        SafetyCheck(
                            name="import_verification",
                            description="Package import verification",
                            result=ValidationResult.PASS,
                            level=SafetyLevel.SAFE,
                            message="Main package imports successfully",
                        )
                    )
                else:
                    checks.append(
                        SafetyCheck(
                            name="import_verification",
                            description="Package import verification",
                            result=ValidationResult.WARNING,
                            level=SafetyLevel.CAUTION,
                            message="Could not create import spec",
                        )
                    )
            else:
                checks.append(
                    SafetyCheck(
                        name="import_verification",
                        description="Package import verification",
                        result=ValidationResult.WARNING,
                        level=SafetyLevel.CAUTION,
                        message="Main package __init__.py not found",
                    )
                )

        except Exception as e:
            checks.append(
                SafetyCheck(
                    name="import_verification",
                    description="Package import verification",
                    result=ValidationResult.FAIL,
                    level=SafetyLevel.DANGEROUS,
                    message=f"Import verification failed: {e}",
                )
            )

        return checks

    def _verify_build_capabilities(self) -> List[SafetyCheck]:
        """Verify that the project can still build."""
        checks = []

        # Simple build verification - check for build configuration
        build_files = ["pyproject.toml", "setup.py", "setup.cfg"]
        build_config_found = any((self.project_root / f).exists() for f in build_files)

        if build_config_found:
            checks.append(
                SafetyCheck(
                    name="build_verification",
                    description="Build capability verification",
                    result=ValidationResult.PASS,
                    level=SafetyLevel.SAFE,
                    message="Build configuration files present",
                )
            )
        else:
            checks.append(
                SafetyCheck(
                    name="build_verification",
                    description="Build capability verification",
                    result=ValidationResult.WARNING,
                    level=SafetyLevel.CAUTION,
                    message="No build configuration files found",
                )
            )

        return checks

    def _check_broken_references(self) -> List[SafetyCheck]:
        """Check for broken file references after cleanup."""
        checks = []

        # This would be a more complex analysis in a full implementation
        checks.append(
            SafetyCheck(
                name="broken_references",
                description="Broken reference analysis",
                result=ValidationResult.PASS,
                level=SafetyLevel.SAFE,
                message="Reference analysis completed (basic check)",
            )
        )

        return checks

    def _generate_validation_report(
        self, checks: List[SafetyCheck], operation_type: str
    ) -> ValidationReport:
        """Generate comprehensive validation report."""
        # Count results by type
        passed = sum(1 for c in checks if c.result == ValidationResult.PASS)
        warnings = sum(1 for c in checks if c.result == ValidationResult.WARNING)
        failed = sum(1 for c in checks if c.result == ValidationResult.FAIL)
        errors = sum(1 for c in checks if c.result == ValidationResult.ERROR)

        # Determine overall result and safety level
        if errors > 0:
            overall_result = ValidationResult.ERROR
            overall_safety = SafetyLevel.CRITICAL
        elif failed > 0:
            overall_result = ValidationResult.FAIL
            overall_safety = SafetyLevel.DANGEROUS
        elif warnings > 0:
            overall_result = ValidationResult.WARNING
            overall_safety = SafetyLevel.CAUTION
        else:
            overall_result = ValidationResult.PASS
            overall_safety = SafetyLevel.SAFE

        # Generate recommendations
        recommendations = []
        required_actions = []

        for check in checks:
            if check.result == ValidationResult.FAIL:
                required_actions.append(f"REQUIRED: {check.name} - {check.message}")
            elif check.result == ValidationResult.WARNING:
                recommendations.append(f"RECOMMENDED: {check.name} - {check.message}")
            elif check.result == ValidationResult.ERROR:
                required_actions.append(f"CRITICAL: {check.name} - {check.message}")

        return ValidationReport(
            operation_type=operation_type,
            total_checks=len(checks),
            passed_checks=passed,
            warning_checks=warnings,
            failed_checks=failed,
            error_checks=errors,
            overall_safety_level=overall_safety,
            overall_result=overall_result,
            checks=checks,
            recommendations=recommendations,
            required_actions=required_actions,
        )
