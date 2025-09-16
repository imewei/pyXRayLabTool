#!/usr/bin/env python3
"""
Intelligent Cleanup Manager for XRayLabTool

This script provides comprehensive cleanup capabilities integrated with the
codebase cleanup detection system. It supports dry-run mode, safety checks,
backup creation, and detailed reporting.
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime
import tempfile

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from xraylabtool.cleanup.file_detector import ObsoleteFileDetector, FileCategory
    from xraylabtool.cleanup.safety_classifier import SafetyClassifier
    from xraylabtool.cleanup.config import CleanupConfig
    from xraylabtool.cleanup.reporting import ReportGenerator, CleanupLogger, OperationStats
    from xraylabtool.cleanup.metadata_analyzer import MetadataAnalyzer
except ImportError as e:
    print(f"Warning: Could not import cleanup modules: {e}")
    print("Some functionality may be limited.")


class CleanupManager:
    """Comprehensive cleanup manager with safety and intelligence."""

    def __init__(
        self,
        project_root: Path,
        config: Optional[CleanupConfig] = None,
        dry_run: bool = True,
        verbose: bool = False
    ):
        """
        Initialize cleanup manager.

        Args:
            project_root: Root directory of the project
            config: Cleanup configuration to use
            dry_run: If True, only preview operations without making changes
            verbose: Enable verbose output
        """
        self.project_root = Path(project_root).resolve()
        self.config = config or CleanupConfig()
        self.dry_run = dry_run
        self.verbose = verbose

        # Initialize components
        self.detector = ObsoleteFileDetector(
            root_path=self.project_root,
            recursive=self.config.detection.recursive_scan,
            use_git_context=self.config.detection.use_git_context,
            max_file_size_mb=self.config.safety.max_file_size_mb
        )
        self.classifier = SafetyClassifier(
            custom_rules=self.config.get_file_category_mapping(),
            strict_mode=self.config.safety.strict_mode
        )
        self.logger = CleanupLogger(
            log_level=self.config.reporting.log_level,
            console_output=verbose
        )
        self.report_generator = ReportGenerator(
            self.config.reporting.report_directory
        )

        # Backup management
        self.backup_dir = self.project_root / self.config.safety.backup_directory
        self.backup_created = False

    def detect_files(self, category_filter: Optional[List[str]] = None) -> List:
        """
        Detect obsolete files using the detection system.

        Args:
            category_filter: Optional list of categories to filter by

        Returns:
            List of DetectionResult objects
        """
        print(f"🔍 Analyzing project directory: {self.project_root}")

        with self.logger.operation_context("file_detection"):
            results = self.detector.detect_all()

            # Filter by category if specified
            if category_filter:
                category_enums = []
                for cat_name in category_filter:
                    try:
                        category_enums.append(FileCategory[cat_name.upper()])
                    except KeyError:
                        print(f"Warning: Unknown category '{cat_name}'")

                if category_enums:
                    results = [r for r in results if r.category in category_enums]

            self.logger.log_detection_results(results)
            return results

    def classify_files(self, file_paths: List[Path]) -> Dict:
        """
        Classify files for safety assessment.

        Args:
            file_paths: List of file paths to classify

        Returns:
            Dictionary mapping file paths to classification results
        """
        print(f"🛡️  Performing safety classification on {len(file_paths)} files...")

        classifications = self.classifier.classify_batch(file_paths)
        self.logger.log_safety_analysis(classifications)

        return classifications

    def create_backup(self, files_to_backup: List[Path]) -> bool:
        """
        Create backup of files before cleanup.

        Args:
            files_to_backup: List of file paths to backup

        Returns:
            True if backup was successful, False otherwise
        """
        if not self.config.safety.create_backup or not files_to_backup:
            return True

        print(f"💾 Creating backup of {len(files_to_backup)} files...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = self.backup_dir / f"backup_{timestamp}"

        try:
            backup_subdir.mkdir(parents=True, exist_ok=True)

            # Create backup manifest
            manifest = {
                "timestamp": timestamp,
                "files": [],
                "project_root": str(self.project_root)
            }

            for file_path in files_to_backup:
                if file_path.exists():
                    # Calculate relative path for backup structure
                    rel_path = file_path.relative_to(self.project_root)
                    backup_file_path = backup_subdir / rel_path

                    # Create parent directories in backup
                    backup_file_path.parent.mkdir(parents=True, exist_ok=True)

                    # Copy file to backup
                    if self.dry_run:
                        print(f"  [DRY RUN] Would backup: {rel_path}")
                    else:
                        shutil.copy2(file_path, backup_file_path)
                        print(f"  ✓ Backed up: {rel_path}")

                    manifest["files"].append({
                        "original": str(file_path),
                        "relative": str(rel_path),
                        "backup": str(backup_file_path),
                        "size": file_path.stat().st_size if file_path.exists() else 0
                    })

            # Save manifest
            manifest_path = backup_subdir / "backup_manifest.json"
            if not self.dry_run:
                with open(manifest_path, 'w') as f:
                    json.dump(manifest, f, indent=2)

            print(f"✅ Backup created: {backup_subdir}")
            self.backup_created = True
            return True

        except Exception as e:
            print(f"❌ Backup creation failed: {e}")
            return False

    def remove_files(self, files_to_remove: List[Path]) -> Dict[str, int]:
        """
        Remove files with safety checks and logging.

        Args:
            files_to_remove: List of file paths to remove

        Returns:
            Dictionary with removal statistics
        """
        if not files_to_remove:
            return {"removed": 0, "failed": 0, "size_freed_mb": 0.0}

        print(f"🗑️  Removing {len(files_to_remove)} files...")

        removed_count = 0
        failed_count = 0
        total_size_bytes = 0

        for file_path in files_to_remove:
            try:
                if file_path.exists():
                    file_size = file_path.stat().st_size

                    if self.dry_run:
                        print(f"  [DRY RUN] Would remove: {file_path}")
                        removed_count += 1
                        total_size_bytes += file_size
                    else:
                        if file_path.is_dir():
                            shutil.rmtree(file_path)
                        else:
                            file_path.unlink()

                        print(f"  ✓ Removed: {file_path}")
                        removed_count += 1
                        total_size_bytes += file_size

                        self.logger.log_file_operation("remove", file_path, True)
                else:
                    print(f"  ⚠️  File not found: {file_path}")

            except Exception as e:
                print(f"  ❌ Failed to remove {file_path}: {e}")
                failed_count += 1
                self.logger.log_file_operation("remove", file_path, False, str(e))

        size_freed_mb = total_size_bytes / (1024 * 1024)

        return {
            "removed": removed_count,
            "failed": failed_count,
            "size_freed_mb": size_freed_mb
        }

    def generate_report(
        self,
        detection_results: List,
        classifications: Dict,
        removal_stats: Dict[str, int]
    ) -> Path:
        """
        Generate comprehensive cleanup report.

        Args:
            detection_results: Results from file detection
            classifications: File safety classifications
            removal_stats: Statistics from file removal

        Returns:
            Path to the generated report
        """
        print("📊 Generating cleanup report...")

        operation_stats = OperationStats(
            files_scanned=len(detection_results),
            files_detected=len(detection_results),
            files_removed=removal_stats.get("removed", 0),
            files_skipped=removal_stats.get("failed", 0),
            total_size_removed_mb=removal_stats.get("size_freed_mb", 0.0),
            errors_encountered=removal_stats.get("failed", 0)
        )

        from xraylabtool.cleanup.reporting import ReportFormat

        report_path = self.report_generator.generate_cleanup_report(
            operation_type="makefile_cleanup",
            root_directory=self.project_root,
            detection_results=detection_results,
            classifications=classifications,
            operation_stats=operation_stats,
            configuration=self.config.to_dict(),
            format=ReportFormat.JSON
        )

        print(f"📄 Report saved: {report_path}")
        return report_path


def main():
    """Main entry point for the cleanup manager."""
    parser = argparse.ArgumentParser(
        description="Intelligent cleanup manager for XRayLabTool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run analysis of all obsolete files
  python cleanup_manager.py --detect-only

  # Clean only safe files (dry-run by default)
  python cleanup_manager.py --category safe_to_remove --dry-run

  # Interactive cleanup with backup
  python cleanup_manager.py --interactive --backup

  # Report current cleanup status
  python cleanup_manager.py --status-only
        """
    )

    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Preview operations without making changes (default)")
    parser.add_argument("--execute", action="store_true",
                       help="Actually perform cleanup operations (overrides --dry-run)")
    parser.add_argument("--category", action="append",
                       choices=["safe_to_remove", "legacy", "system_generated",
                               "build_artifact", "temporary"],
                       help="Filter by file category (can be used multiple times)")
    parser.add_argument("--backup", action="store_true",
                       help="Create backup before cleanup")
    parser.add_argument("--no-backup", action="store_true",
                       help="Skip backup creation")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive mode with confirmations")
    parser.add_argument("--detect-only", action="store_true",
                       help="Only detect and report, no cleanup")
    parser.add_argument("--status-only", action="store_true",
                       help="Show cleanup status and recommendations")
    parser.add_argument("--config", type=Path,
                       help="Path to cleanup configuration file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--project-root", type=Path, default=".",
                       help="Project root directory (default: current directory)")

    args = parser.parse_args()

    # Handle execution mode
    dry_run = not args.execute if args.execute else args.dry_run

    # Load configuration
    config = None
    if args.config:
        config = CleanupConfig.from_file(args.config)
    else:
        config = CleanupConfig()

    # Override backup setting from args
    if args.no_backup:
        config.safety.create_backup = False
    elif args.backup:
        config.safety.create_backup = True

    # Initialize cleanup manager
    manager = CleanupManager(
        project_root=args.project_root,
        config=config,
        dry_run=dry_run,
        verbose=args.verbose
    )

    print("🧹 XRayLabTool Intelligent Cleanup Manager")
    print("=" * 45)
    print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    print(f"Project: {manager.project_root}")
    print(f"Backup: {'Enabled' if config.safety.create_backup else 'Disabled'}")
    print()

    try:
        # Detect obsolete files
        detection_results = manager.detect_files(args.category)

        if args.status_only:
            # Status report only
            print(f"📋 Found {len(detection_results)} potentially obsolete files")
            category_counts = {}
            total_size_mb = 0.0

            for result in detection_results:
                cat_name = result.category.name
                category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
                total_size_mb += result.size_bytes / (1024 * 1024)

            print(f"📊 Total size: {total_size_mb:.2f} MB")
            print("📂 By category:")
            for category, count in category_counts.items():
                print(f"  • {category}: {count} files")

            print("\n💡 Run with --detect-only for detailed analysis")
            return 0

        if not detection_results:
            print("✅ No obsolete files detected. Project is clean!")
            return 0

        # Classify files for safety
        file_paths = [result.file_path for result in detection_results]
        classifications = manager.classify_files(file_paths)

        # Show summary
        safe_count = sum(1 for c in classifications.values()
                        if c.category == FileCategory.SAFE_TO_REMOVE)
        review_count = sum(1 for c in classifications.values()
                          if c.category == FileCategory.REVIEW_NEEDED)

        print(f"📊 Analysis Summary:")
        print(f"  • Total files detected: {len(detection_results)}")
        print(f"  • Safe to remove: {safe_count}")
        print(f"  • Need review: {review_count}")
        print(f"  • Critical (keep): {len(classifications) - safe_count - review_count}")

        if args.detect_only:
            print("\n📄 Detection complete. Use --execute to perform cleanup.")
            return 0

        # Determine files to remove based on safety classification
        files_to_remove = []
        for file_path, classification in classifications.items():
            if classification.category == FileCategory.SAFE_TO_REMOVE:
                files_to_remove.append(file_path)

        if not files_to_remove:
            print("⚠️  No files classified as safe to remove automatically.")
            print("💡 Use --detect-only to see what requires manual review.")
            return 0

        # Interactive confirmation if requested
        if args.interactive and not dry_run:
            print(f"\n🤔 About to remove {len(files_to_remove)} files.")
            response = input("Continue? (y/N): ").lower().strip()
            if response != 'y':
                print("❌ Cleanup cancelled by user.")
                return 0

        # Create backup if enabled
        if config.safety.create_backup:
            backup_success = manager.create_backup(files_to_remove)
            if not backup_success and not dry_run:
                print("❌ Backup creation failed. Aborting cleanup for safety.")
                return 1

        # Perform cleanup
        removal_stats = manager.remove_files(files_to_remove)

        # Generate report
        report_path = manager.generate_report(
            detection_results,
            classifications,
            removal_stats
        )

        # Summary
        mode_str = "[DRY RUN] " if dry_run else ""
        print(f"\n✅ {mode_str}Cleanup completed successfully!")
        print(f"📈 Files processed: {removal_stats['removed']}")
        print(f"💾 Space freed: {removal_stats['size_freed_mb']:.2f} MB")

        if removal_stats['failed'] > 0:
            print(f"⚠️  Failed operations: {removal_stats['failed']}")

        if manager.backup_created:
            print(f"💾 Backup location: {manager.backup_dir}")

        return 0

    except KeyboardInterrupt:
        print("\n❌ Cleanup interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Cleanup failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())