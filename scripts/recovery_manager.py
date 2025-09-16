#!/usr/bin/env python3
"""
Recovery and restoration manager for cleanup operations.

This script provides comprehensive recovery capabilities for cleanup operations,
including backup restoration, emergency recovery, and system health verification.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Set, Optional, Union, Any
from datetime import datetime, timedelta
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from xraylabtool.cleanup.backup_manager import BackupManager, BackupMetadata
    from xraylabtool.cleanup.safety_validator import SafetyValidator, ValidationReport
    from xraylabtool.cleanup.safety_integration import SafetyIntegratedCleanup
    from xraylabtool.cleanup.config import CleanupConfig
except ImportError as e:
    print(f"Warning: Could not import cleanup modules: {e}")
    print("Some functionality may be limited.")


class RecoveryManager:
    """
    Comprehensive recovery manager for cleanup operations.

    Provides tools for backup restoration, emergency recovery,
    system health checks, and guided recovery procedures.
    """

    def __init__(
        self,
        project_root: Union[str, Path],
        config: Optional[CleanupConfig] = None
    ):
        """
        Initialize recovery manager.

        Args:
            project_root: Root directory of the project
            config: Cleanup configuration
        """
        self.project_root = Path(project_root).resolve()
        self.config = config or CleanupConfig()

        # Initialize components
        self.backup_manager = BackupManager(
            project_root=self.project_root,
            backup_root=self.project_root / self.config.safety.backup_directory
        )

        self.safety_validator = SafetyValidator(
            project_root=self.project_root,
            backup_manager=self.backup_manager
        )

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.reporting.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        print(f"🔧 RecoveryManager initialized for: {self.project_root}")

    def list_available_backups(self, show_details: bool = False) -> List[BackupMetadata]:
        """
        List all available backups with optional details.

        Args:
            show_details: Show detailed information about each backup

        Returns:
            List of available backup metadata
        """
        backups = self.backup_manager.list_backups()

        if not backups:
            print("📭 No backups found")
            return []

        print(f"📦 Found {len(backups)} backup(s):")
        print("=" * 80)

        for i, backup in enumerate(backups):
            age_hours = (datetime.now() - datetime.fromisoformat(backup.timestamp)).total_seconds() / 3600

            print(f"{i+1:2d}. {backup.backup_id}")
            print(f"    📅 Created: {backup.timestamp} ({age_hours:.1f} hours ago)")
            print(f"    📁 Files: {backup.total_files}, Size: {backup.total_size_bytes / (1024*1024):.2f} MB")
            print(f"    🗜️  Method: {backup.backup_method}, Compression: {backup.compression_ratio:.2f}")
            print(f"    ✅ Verified: {'Yes' if backup.integrity_verified else 'No'}")

            if show_details:
                print(f"    🔍 Operation: {backup.operation_type}")
                print(f"    📊 Checksum: {backup.checksum[:16]}...")
                if backup.files:
                    print(f"    📄 Sample files: {', '.join([f['relative_path'] for f in backup.files[:3]])}")
                    if len(backup.files) > 3:
                        print(f"         ... and {len(backup.files) - 3} more files")

            print()

        return backups

    def inspect_backup(self, backup_id: str) -> Optional[Dict[str, Any]]:
        """
        Inspect a specific backup in detail.

        Args:
            backup_id: ID of the backup to inspect

        Returns:
            Dictionary with detailed backup information
        """
        backup = self.backup_manager.get_backup_info(backup_id)

        if not backup:
            print(f"❌ Backup '{backup_id}' not found")
            return None

        print(f"🔍 Detailed inspection of backup: {backup_id}")
        print("=" * 80)

        # Basic information
        print(f"📅 Created: {backup.timestamp}")
        print(f"🏷️  Operation Type: {backup.operation_type}")
        print(f"📁 Total Files: {backup.total_files}")
        print(f"📊 Total Size: {backup.total_size_bytes / (1024*1024):.2f} MB")

        if backup.backup_method == "zip":
            print(f"🗜️  Compressed Size: {backup.compressed_size_bytes / (1024*1024):.2f} MB")
            print(f"📉 Compression Ratio: {backup.compression_ratio:.2f}")

        print(f"🔐 Checksum: {backup.checksum}")
        print(f"✅ Integrity Verified: {'Yes' if backup.integrity_verified else 'No'}")

        # File breakdown by extension
        if backup.files:
            print(f"\n📄 File Breakdown:")
            extensions = {}
            for file_info in backup.files:
                ext = Path(file_info["relative_path"]).suffix or "no_extension"
                extensions[ext] = extensions.get(ext, 0) + 1

            for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True):
                print(f"    {ext}: {count} files")

        # Largest files
        if backup.files and len(backup.files) > 1:
            print(f"\n📈 Largest Files:")
            sorted_files = sorted(backup.files, key=lambda f: f.get("file_size", 0), reverse=True)
            for file_info in sorted_files[:5]:
                size_mb = file_info.get("file_size", 0) / (1024*1024)
                print(f"    {file_info['relative_path']} ({size_mb:.2f} MB)")

        return {
            "backup_id": backup_id,
            "metadata": backup,
            "file_extensions": extensions if backup.files else {},
            "total_size_mb": backup.total_size_bytes / (1024*1024)
        }

    def verify_backup_integrity(self, backup_id: str) -> bool:
        """
        Verify the integrity of a backup.

        Args:
            backup_id: ID of the backup to verify

        Returns:
            True if backup integrity is verified, False otherwise
        """
        print(f"🔍 Verifying integrity of backup: {backup_id}")

        backup = self.backup_manager.get_backup_info(backup_id)
        if not backup:
            print(f"❌ Backup '{backup_id}' not found")
            return False

        # Perform integrity verification
        try:
            integrity_ok = self.backup_manager._verify_backup_integrity(backup)

            if integrity_ok:
                print("✅ Backup integrity verification PASSED")
                return True
            else:
                print("❌ Backup integrity verification FAILED")
                return False

        except Exception as e:
            print(f"❌ Integrity verification error: {e}")
            return False

    def restore_from_backup(
        self,
        backup_id: str,
        specific_files: Optional[List[str]] = None,
        verify_first: bool = True,
        dry_run: bool = True,
        interactive: bool = False
    ) -> Dict[str, Any]:
        """
        Restore files from a backup.

        Args:
            backup_id: ID of the backup to restore from
            specific_files: Specific files to restore (None for all)
            verify_first: Verify backup integrity before restoring
            dry_run: Show what would be restored without doing it
            interactive: Ask for confirmation before each file

        Returns:
            Dictionary with restoration results
        """
        print(f"🔄 {'[DRY RUN] ' if dry_run else ''}Restoring from backup: {backup_id}")

        # Verify backup integrity if requested
        if verify_first:
            if not self.verify_backup_integrity(backup_id):
                return {"success": False, "error": "Backup integrity verification failed"}

        # Get backup information
        backup = self.backup_manager.get_backup_info(backup_id)
        if not backup:
            return {"success": False, "error": f"Backup '{backup_id}' not found"}

        # Show what will be restored
        files_to_restore = specific_files or [f["relative_path"] for f in backup.files]
        print(f"📁 Files to restore: {len(files_to_restore)}")

        if interactive or dry_run:
            print("📋 Restoration preview:")
            for i, file_path in enumerate(files_to_restore[:10]):
                print(f"  {i+1:2d}. {file_path}")
            if len(files_to_restore) > 10:
                print(f"      ... and {len(files_to_restore) - 10} more files")

        if dry_run:
            print("🧪 [DRY RUN] No files were actually restored")
            return {
                "success": True,
                "dry_run": True,
                "files_to_restore": len(files_to_restore),
                "backup_id": backup_id
            }

        # Interactive confirmation
        if interactive:
            response = input(f"\n🤔 Restore {len(files_to_restore)} files? This will overwrite existing files! (y/N): ")
            if response.lower() != 'y':
                print("❌ Restoration cancelled by user")
                return {"success": False, "cancelled": True}

        # Perform restoration
        try:
            restore_results = self.backup_manager.restore_backup(
                backup_id=backup_id,
                restore_files=specific_files,
                verify_integrity=verify_first,
                overwrite_existing=True
            )

            print(f"✅ Restoration completed:")
            print(f"   📁 Files restored: {restore_results['restored']}")
            print(f"   ❌ Files failed: {restore_results['failed']}")

            return {
                "success": restore_results['failed'] == 0,
                "restored_files": restore_results['restored'],
                "failed_files": restore_results['failed'],
                "backup_id": backup_id
            }

        except Exception as e:
            error_msg = f"Restoration failed: {e}"
            print(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}

    def emergency_recovery(
        self,
        auto_select_backup: bool = False,
        verify_after_recovery: bool = True
    ) -> Dict[str, Any]:
        """
        Perform emergency recovery using the most recent backup.

        Args:
            auto_select_backup: Automatically select the most recent backup
            verify_after_recovery: Verify system health after recovery

        Returns:
            Dictionary with emergency recovery results
        """
        print("🚨 EMERGENCY RECOVERY MODE")
        print("=" * 50)

        # List available backups
        backups = self.backup_manager.list_backups()
        if not backups:
            print("❌ No backups available for emergency recovery")
            return {"success": False, "error": "No backups available"}

        # Select backup for recovery
        if auto_select_backup:
            selected_backup = backups[0]  # Most recent
            print(f"🔄 Auto-selected most recent backup: {selected_backup.backup_id}")
        else:
            print(f"📦 Available backups:")
            for i, backup in enumerate(backups[:5]):  # Show top 5
                age_hours = (datetime.now() - datetime.fromisoformat(backup.timestamp)).total_seconds() / 3600
                print(f"  {i+1}. {backup.backup_id} ({age_hours:.1f} hours ago, {backup.total_files} files)")

            try:
                choice = int(input(f"Select backup (1-{min(5, len(backups))}): ")) - 1
                if choice < 0 or choice >= min(5, len(backups)):
                    raise ValueError("Invalid selection")
                selected_backup = backups[choice]
            except (ValueError, KeyboardInterrupt):
                print("❌ Emergency recovery cancelled")
                return {"success": False, "cancelled": True}

        # Perform emergency restoration
        print(f"🚨 Performing emergency recovery from: {selected_backup.backup_id}")

        recovery_results = self.restore_from_backup(
            backup_id=selected_backup.backup_id,
            verify_first=True,
            dry_run=False,
            interactive=False
        )

        if not recovery_results.get("success"):
            print("❌ Emergency recovery failed")
            return recovery_results

        # Post-recovery validation if requested
        if verify_after_recovery:
            print("🔍 Verifying system health after recovery...")
            validation = self.safety_validator.validate_post_operation([], "emergency_recovery")

            if validation.overall_result.value in ["pass", "warning"]:
                print("✅ Post-recovery validation passed")
                recovery_results["post_validation"] = "passed"
            else:
                print("⚠️ Post-recovery validation detected issues")
                recovery_results["post_validation"] = "issues_detected"
                recovery_results["validation_warnings"] = validation.recommendations

        print("🎉 Emergency recovery completed")
        return recovery_results

    def cleanup_old_backups(
        self,
        max_age_days: int = 30,
        max_backups: int = 10,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Clean up old backups based on age and count.

        Args:
            max_age_days: Maximum age of backups to keep
            max_backups: Maximum number of backups to keep
            dry_run: Show what would be deleted without doing it

        Returns:
            Dictionary with cleanup results
        """
        print(f"🧹 {'[DRY RUN] ' if dry_run else ''}Cleaning up old backups")
        print(f"    📅 Max age: {max_age_days} days")
        print(f"    📦 Max count: {max_backups}")

        backups = self.backup_manager.list_backups()
        if not backups:
            print("📭 No backups found")
            return {"removed_count": 0, "error": "No backups found"}

        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        backups_to_remove = []

        # Find backups to remove by age
        for backup in backups:
            backup_date = datetime.fromisoformat(backup.timestamp)
            if backup_date < cutoff_date:
                backups_to_remove.append(backup)

        # Find backups to remove by count (keep newest)
        if len(backups) > max_backups:
            excess_backups = backups[max_backups:]
            for backup in excess_backups:
                if backup not in backups_to_remove:
                    backups_to_remove.append(backup)

        if not backups_to_remove:
            print("✨ No old backups to clean up")
            return {"removed_count": 0}

        print(f"🗑️ Backups to remove: {len(backups_to_remove)}")
        for backup in backups_to_remove:
            age_days = (datetime.now() - datetime.fromisoformat(backup.timestamp)).days
            size_mb = backup.total_size_bytes / (1024*1024)
            print(f"  • {backup.backup_id} ({age_days} days old, {size_mb:.1f} MB)")

        if dry_run:
            print("🧪 [DRY RUN] No backups were actually removed")
            return {"dry_run": True, "would_remove": len(backups_to_remove)}

        # Perform actual cleanup
        try:
            cleanup_results = self.backup_manager.cleanup_old_backups(max_age_days)
            print(f"✅ Cleanup completed: {cleanup_results['removed_count']} backups removed")
            print(f"💾 Space freed: {cleanup_results['removed_size_mb']:.2f} MB")

            return cleanup_results

        except Exception as e:
            error_msg = f"Backup cleanup failed: {e}"
            print(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}

    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive system health check.

        Returns:
            Dictionary with health check results
        """
        print("🔍 Performing system health check")
        print("=" * 40)

        health_results = {
            "overall_health": "unknown",
            "checks": {},
            "recommendations": [],
            "critical_issues": []
        }

        # Check 1: Backup system health
        try:
            backups = self.backup_manager.list_backups()
            backup_health = {
                "available_backups": len(backups),
                "backup_directory_writable": True,
                "most_recent_backup": backups[0].timestamp if backups else None
            }

            if len(backups) == 0:
                health_results["recommendations"].append("No backups available - consider creating a backup")
            elif len(backups) > 20:
                health_results["recommendations"].append("Many backups detected - consider cleanup")

            health_results["checks"]["backup_system"] = backup_health
            print(f"✅ Backup system: {len(backups)} backups available")

        except Exception as e:
            health_results["checks"]["backup_system"] = {"error": str(e)}
            health_results["critical_issues"].append(f"Backup system error: {e}")
            print(f"❌ Backup system: {e}")

        # Check 2: Project integrity
        try:
            validation = self.safety_validator.validate_post_operation([], "health_check")
            health_results["checks"]["project_integrity"] = {
                "validation_result": validation.overall_result.value,
                "safety_level": validation.overall_safety_level.value,
                "passed_checks": validation.passed_checks,
                "failed_checks": validation.failed_checks
            }

            if validation.failed_checks > 0:
                health_results["critical_issues"].extend(validation.required_actions)
                print(f"❌ Project integrity: {validation.failed_checks} failed checks")
            else:
                print("✅ Project integrity: All checks passed")

        except Exception as e:
            health_results["checks"]["project_integrity"] = {"error": str(e)}
            health_results["critical_issues"].append(f"Project integrity check failed: {e}")
            print(f"❌ Project integrity: {e}")

        # Check 3: Disk space
        try:
            import shutil
            disk_usage = shutil.disk_usage(self.project_root)
            available_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            used_percent = ((disk_usage.total - disk_usage.free) / disk_usage.total) * 100

            disk_health = {
                "available_gb": available_gb,
                "total_gb": total_gb,
                "used_percent": used_percent
            }

            if used_percent > 90:
                health_results["critical_issues"].append(f"Low disk space: {used_percent:.1f}% used")
                print(f"❌ Disk space: {used_percent:.1f}% used ({available_gb:.1f}GB free)")
            elif used_percent > 80:
                health_results["recommendations"].append(f"Consider freeing disk space: {used_percent:.1f}% used")
                print(f"⚠️ Disk space: {used_percent:.1f}% used ({available_gb:.1f}GB free)")
            else:
                print(f"✅ Disk space: {available_gb:.1f}GB free ({used_percent:.1f}% used)")

            health_results["checks"]["disk_space"] = disk_health

        except Exception as e:
            health_results["checks"]["disk_space"] = {"error": str(e)}
            print(f"❌ Disk space check: {e}")

        # Determine overall health
        if health_results["critical_issues"]:
            health_results["overall_health"] = "critical"
            print("\n🚨 CRITICAL ISSUES DETECTED")
        elif health_results["recommendations"]:
            health_results["overall_health"] = "warning"
            print("\n⚠️ Some issues detected")
        else:
            health_results["overall_health"] = "good"
            print("\n✅ System health is good")

        return health_results


def main():
    """Main entry point for the recovery manager."""
    parser = argparse.ArgumentParser(
        description="Recovery and restoration manager for cleanup operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available backups
  python recovery_manager.py --list-backups

  # Inspect a specific backup
  python recovery_manager.py --inspect-backup backup_id_here

  # Emergency recovery (interactive)
  python recovery_manager.py --emergency-recovery

  # Restore from specific backup
  python recovery_manager.py --restore backup_id_here --dry-run

  # System health check
  python recovery_manager.py --health-check

  # Clean up old backups
  python recovery_manager.py --cleanup-backups --max-age 7 --dry-run
        """
    )

    parser.add_argument("--list-backups", action="store_true",
                       help="List all available backups")
    parser.add_argument("--list-backups-detailed", action="store_true",
                       help="List backups with detailed information")
    parser.add_argument("--inspect-backup", type=str,
                       help="Inspect a specific backup by ID")
    parser.add_argument("--verify-backup", type=str,
                       help="Verify integrity of a specific backup")
    parser.add_argument("--restore", type=str,
                       help="Restore from backup by ID")
    parser.add_argument("--restore-files", nargs='+',
                       help="Specific files to restore (relative paths)")
    parser.add_argument("--emergency-recovery", action="store_true",
                       help="Perform emergency recovery")
    parser.add_argument("--auto-select", action="store_true",
                       help="Auto-select most recent backup for emergency recovery")
    parser.add_argument("--cleanup-backups", action="store_true",
                       help="Clean up old backups")
    parser.add_argument("--max-age", type=int, default=30,
                       help="Maximum age of backups to keep (days)")
    parser.add_argument("--max-count", type=int, default=10,
                       help="Maximum number of backups to keep")
    parser.add_argument("--health-check", action="store_true",
                       help="Perform system health check")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Show what would happen without doing it (default)")
    parser.add_argument("--execute", action="store_true",
                       help="Actually perform operations (overrides --dry-run)")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive mode with confirmations")
    parser.add_argument("--project-root", type=Path, default=".",
                       help="Project root directory (default: current directory)")

    args = parser.parse_args()

    # Handle execution mode
    dry_run = not args.execute if args.execute else args.dry_run

    # Initialize recovery manager
    try:
        recovery = RecoveryManager(project_root=args.project_root)
    except Exception as e:
        print(f"❌ Failed to initialize recovery manager: {e}")
        return 1

    try:
        # Execute requested operation
        if args.list_backups or args.list_backups_detailed:
            recovery.list_available_backups(show_details=args.list_backups_detailed)

        elif args.inspect_backup:
            recovery.inspect_backup(args.inspect_backup)

        elif args.verify_backup:
            result = recovery.verify_backup_integrity(args.verify_backup)
            return 0 if result else 1

        elif args.restore:
            result = recovery.restore_from_backup(
                backup_id=args.restore,
                specific_files=args.restore_files,
                dry_run=dry_run,
                interactive=args.interactive
            )
            return 0 if result.get("success", False) else 1

        elif args.emergency_recovery:
            result = recovery.emergency_recovery(
                auto_select_backup=args.auto_select
            )
            return 0 if result.get("success", False) else 1

        elif args.cleanup_backups:
            result = recovery.cleanup_old_backups(
                max_age_days=args.max_age,
                max_backups=args.max_count,
                dry_run=dry_run
            )
            return 0 if result.get("removed_count", 0) >= 0 else 1

        elif args.health_check:
            result = recovery.health_check()
            return 0 if result["overall_health"] != "critical" else 1

        else:
            print("❓ No operation specified. Use --help for available options.")
            parser.print_help()
            return 1

        return 0

    except KeyboardInterrupt:
        print("\n❌ Operation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Operation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())