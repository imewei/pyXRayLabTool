"""
Comprehensive logging and reporting system for codebase cleanup operations.

This module provides detailed logging, progress tracking, and comprehensive
reporting capabilities for cleanup operations, including safety analysis
and operation summaries.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Union, Any, TextIO
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from contextlib import contextmanager
from enum import Enum

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .file_detector import DetectionResult, FileCategory
from .safety_classifier import ClassificationResult
from .git_analyzer import FileHistory, AgeAnalysis

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Supported report formats."""
    JSON = 'json'
    YAML = 'yaml'
    TEXT = 'text'
    HTML = 'html'


@dataclass
class OperationStats:
    """Statistics for cleanup operations."""
    files_scanned: int = 0
    files_detected: int = 0
    files_removed: int = 0
    files_skipped: int = 0
    total_size_removed_mb: float = 0.0
    total_time_seconds: float = 0.0
    errors_encountered: int = 0


@dataclass
class SafetySummary:
    """Summary of safety analysis results."""
    total_files: int
    safe_to_remove: int
    review_needed: int
    critical_keep: int
    low_confidence: int
    warnings_count: int
    recommendation: str


@dataclass
class CleanupReport:
    """Comprehensive cleanup operation report."""
    timestamp: str
    operation_type: str
    root_directory: str
    configuration_used: Dict[str, Any]
    operation_stats: OperationStats
    safety_summary: SafetySummary
    category_breakdown: Dict[str, int]
    largest_files_found: List[Dict[str, Any]]
    oldest_files_found: List[Dict[str, Any]]
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]
    detailed_results: List[Dict[str, Any]]


class ProgressTracker:
    """Progress tracking for long-running operations."""

    def __init__(
        self,
        total_items: int,
        operation_name: str = "Processing",
        log_interval: int = 100
    ):
        """
        Initialize progress tracker.

        Args:
            total_items: Total number of items to process
            operation_name: Name of the operation being tracked
            log_interval: Log progress every N items
        """
        self.total_items = total_items
        self.operation_name = operation_name
        self.log_interval = log_interval

        self.processed_items = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_log_count = 0

    def update(self, count: int = 1) -> None:
        """Update progress counter."""
        self.processed_items += count

        # Log progress at intervals
        if (self.processed_items - self.last_log_count) >= self.log_interval:
            self._log_progress()
            self.last_log_time = time.time()
            self.last_log_count = self.processed_items

    def complete(self) -> None:
        """Mark operation as complete and log final statistics."""
        self.processed_items = self.total_items
        self._log_progress()

        total_time = time.time() - self.start_time
        logger.info(f"{self.operation_name} completed in {total_time:.2f} seconds")

    def _log_progress(self) -> None:
        """Log current progress."""
        if self.total_items > 0:
            percentage = (self.processed_items / self.total_items) * 100
            elapsed = time.time() - self.start_time

            if elapsed > 0 and self.processed_items > 0:
                rate = self.processed_items / elapsed
                eta_seconds = (self.total_items - self.processed_items) / rate
                eta_str = f", ETA: {timedelta(seconds=int(eta_seconds))}"
            else:
                eta_str = ""

            logger.info(
                f"{self.operation_name}: {self.processed_items}/{self.total_items} "
                f"({percentage:.1f}%){eta_str}"
            )


class CleanupLogger:
    """Enhanced logging system for cleanup operations."""

    def __init__(
        self,
        log_file: Optional[Union[str, Path]] = None,
        log_level: str = 'INFO',
        console_output: bool = True
    ):
        """
        Initialize cleanup logger.

        Args:
            log_file: Optional log file path
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            console_output: Whether to output to console
        """
        self.log_file = Path(log_file) if log_file else None
        self.log_level = getattr(logging, log_level.upper())
        self.console_output = console_output

        # Set up logging configuration
        self._setup_logging()

        # Operation tracking
        self.operation_start_time = None
        self.current_operation = None
        self.operation_stats = OperationStats()

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Get cleanup logger
        cleanup_logger = logging.getLogger('xraylabtool.cleanup')
        cleanup_logger.setLevel(self.log_level)

        # Clear existing handlers
        cleanup_logger.handlers.clear()

        # Add console handler if requested
        if self.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            cleanup_logger.addHandler(console_handler)

        # Add file handler if log file specified
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            cleanup_logger.addHandler(file_handler)

    @contextmanager
    def operation_context(self, operation_name: str):
        """Context manager for tracking operations."""
        self.start_operation(operation_name)
        try:
            yield self
        finally:
            self.end_operation()

    def start_operation(self, operation_name: str) -> None:
        """Start tracking a cleanup operation."""
        self.current_operation = operation_name
        self.operation_start_time = time.time()
        self.operation_stats = OperationStats()

        logger.info(f"Starting operation: {operation_name}")

    def end_operation(self) -> None:
        """End the current operation tracking."""
        if self.operation_start_time:
            self.operation_stats.total_time_seconds = time.time() - self.operation_start_time

        logger.info(f"Completed operation: {self.current_operation}")
        logger.info(f"Operation statistics: {asdict(self.operation_stats)}")

        self.current_operation = None
        self.operation_start_time = None

    def log_detection_results(
        self,
        results: List[DetectionResult]
    ) -> None:
        """Log detection results summary."""
        if not results:
            logger.info("No files detected for cleanup")
            return

        # Count by category
        category_counts = {}
        total_size_mb = 0.0

        for result in results:
            cat_name = result.category.name
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
            total_size_mb += result.size_bytes / (1024 * 1024)

        logger.info(f"Detection completed: {len(results)} files found")
        logger.info(f"Total size: {total_size_mb:.2f} MB")

        for category, count in category_counts.items():
            logger.info(f"  {category}: {count} files")

        # Update stats
        self.operation_stats.files_detected = len(results)

    def log_safety_analysis(
        self,
        classifications: Dict[Path, ClassificationResult]
    ) -> None:
        """Log safety classification results."""
        if not classifications:
            logger.info("No files classified for safety")
            return

        # Count by category
        category_counts = {}
        low_confidence_count = 0
        warnings_count = 0

        for file_path, result in classifications.items():
            cat_name = result.category.name
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

            if result.confidence < 0.6:
                low_confidence_count += 1

            if result.warnings:
                warnings_count += len(result.warnings)

        logger.info(f"Safety analysis completed: {len(classifications)} files classified")

        for category, count in category_counts.items():
            logger.info(f"  {category}: {count} files")

        if low_confidence_count > 0:
            logger.warning(f"Low confidence classifications: {low_confidence_count}")

        if warnings_count > 0:
            logger.warning(f"Classification warnings: {warnings_count}")

    def log_file_operation(
        self,
        operation: str,
        file_path: Path,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Log individual file operations."""
        if success:
            logger.debug(f"{operation} successful: {file_path}")
            if operation.lower() == 'remove':
                self.operation_stats.files_removed += 1
        else:
            logger.error(f"{operation} failed for {file_path}: {error}")
            self.operation_stats.errors_encountered += 1


class ReportGenerator:
    """Comprehensive report generator for cleanup operations."""

    def __init__(
        self,
        report_directory: Union[str, Path] = '.cleanup_reports'
    ):
        """
        Initialize report generator.

        Args:
            report_directory: Directory to store reports
        """
        self.report_directory = Path(report_directory)
        self.report_directory.mkdir(parents=True, exist_ok=True)

    def generate_cleanup_report(
        self,
        operation_type: str,
        root_directory: Union[str, Path],
        detection_results: List[DetectionResult],
        classifications: Dict[Path, ClassificationResult],
        operation_stats: OperationStats,
        configuration: Dict[str, Any],
        errors: List[str] = None,
        warnings: List[str] = None,
        format: ReportFormat = ReportFormat.JSON
    ) -> Path:
        """
        Generate comprehensive cleanup report.

        Args:
            operation_type: Type of cleanup operation
            root_directory: Root directory that was analyzed
            detection_results: File detection results
            classifications: Safety classification results
            operation_stats: Operation statistics
            configuration: Configuration used
            errors: List of errors encountered
            warnings: List of warnings
            format: Report format to generate

        Returns:
            Path to the generated report file
        """
        timestamp = datetime.now().isoformat()

        # Build safety summary
        safety_summary = self._build_safety_summary(classifications)

        # Build category breakdown
        category_breakdown = self._build_category_breakdown(detection_results)

        # Find largest and oldest files
        largest_files = self._find_largest_files(detection_results)
        oldest_files = self._find_oldest_files(detection_results)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            detection_results, classifications, operation_stats
        )

        # Build detailed results
        detailed_results = self._build_detailed_results(detection_results, classifications)

        # Create report object
        report = CleanupReport(
            timestamp=timestamp,
            operation_type=operation_type,
            root_directory=str(root_directory),
            configuration_used=configuration,
            operation_stats=operation_stats,
            safety_summary=safety_summary,
            category_breakdown=category_breakdown,
            largest_files_found=largest_files,
            oldest_files_found=oldest_files,
            errors=errors or [],
            warnings=warnings or [],
            recommendations=recommendations,
            detailed_results=detailed_results
        )

        # Generate report file
        report_filename = f"cleanup_report_{timestamp.replace(':', '-')}.{format.value}"
        report_path = self.report_directory / report_filename

        if format == ReportFormat.JSON:
            self._write_json_report(report, report_path)
        elif format == ReportFormat.YAML:
            self._write_yaml_report(report, report_path)
        elif format == ReportFormat.TEXT:
            self._write_text_report(report, report_path)
        elif format == ReportFormat.HTML:
            self._write_html_report(report, report_path)

        logger.info(f"Generated cleanup report: {report_path}")
        return report_path

    def _build_safety_summary(
        self,
        classifications: Dict[Path, ClassificationResult]
    ) -> SafetySummary:
        """Build safety analysis summary."""
        if not classifications:
            return SafetySummary(0, 0, 0, 0, 0, 0, "No files analyzed")

        total_files = len(classifications)
        safe_to_remove = 0
        review_needed = 0
        critical_keep = 0
        low_confidence = 0
        warnings_count = 0

        for result in classifications.values():
            if result.category == FileCategory.SAFE_TO_REMOVE:
                safe_to_remove += 1
            elif result.category == FileCategory.REVIEW_NEEDED:
                review_needed += 1
            elif result.category == FileCategory.CRITICAL_KEEP:
                critical_keep += 1

            if result.confidence < 0.6:
                low_confidence += 1

            warnings_count += len(result.warnings or [])

        # Generate recommendation
        safe_ratio = safe_to_remove / total_files if total_files > 0 else 0
        if safe_ratio > 0.8 and warnings_count < total_files * 0.1:
            recommendation = "SAFE: Most files can be removed automatically"
        elif safe_ratio > 0.6 and warnings_count < total_files * 0.2:
            recommendation = "MOSTLY_SAFE: Proceed with caution"
        elif review_needed > safe_to_remove:
            recommendation = "REVIEW_REQUIRED: Manual review needed"
        else:
            recommendation = "CAUTION: High risk - manual review essential"

        return SafetySummary(
            total_files=total_files,
            safe_to_remove=safe_to_remove,
            review_needed=review_needed,
            critical_keep=critical_keep,
            low_confidence=low_confidence,
            warnings_count=warnings_count,
            recommendation=recommendation
        )

    def _build_category_breakdown(
        self,
        results: List[DetectionResult]
    ) -> Dict[str, int]:
        """Build breakdown by file category."""
        breakdown = {}
        for result in results:
            category = result.category.name
            breakdown[category] = breakdown.get(category, 0) + 1
        return breakdown

    def _find_largest_files(
        self,
        results: List[DetectionResult],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find the largest files in detection results."""
        sorted_results = sorted(results, key=lambda r: r.size_bytes, reverse=True)
        return [
            {
                'path': str(result.file_path),
                'size_mb': round(result.size_bytes / (1024 * 1024), 2),
                'category': result.category.name
            }
            for result in sorted_results[:limit]
        ]

    def _find_oldest_files(
        self,
        results: List[DetectionResult],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find the oldest files in detection results."""
        sorted_results = sorted(results, key=lambda r: r.last_modified)
        return [
            {
                'path': str(result.file_path),
                'last_modified': datetime.fromtimestamp(result.last_modified).isoformat(),
                'category': result.category.name
            }
            for result in sorted_results[:limit]
        ]

    def _generate_recommendations(
        self,
        detection_results: List[DetectionResult],
        classifications: Dict[Path, ClassificationResult],
        stats: OperationStats
    ) -> List[str]:
        """Generate cleanup recommendations."""
        recommendations = []

        # File count recommendations
        if len(detection_results) > 100:
            recommendations.append("Large number of files detected - consider batch processing")

        # Safety recommendations
        safe_files = sum(1 for c in classifications.values() if c.category == FileCategory.SAFE_TO_REMOVE)
        if safe_files > 50:
            recommendations.append(f"{safe_files} files appear safe to remove - consider automated cleanup")

        review_files = sum(1 for c in classifications.values() if c.category == FileCategory.REVIEW_NEEDED)
        if review_files > 0:
            recommendations.append(f"{review_files} files need manual review before removal")

        # Performance recommendations
        if stats.errors_encountered > 0:
            recommendations.append(f"{stats.errors_encountered} errors encountered - investigate before proceeding")

        # Size recommendations
        total_size_mb = sum(r.size_bytes for r in detection_results) / (1024 * 1024)
        if total_size_mb > 100:
            recommendations.append(f"Cleanup would free {total_size_mb:.1f} MB of disk space")

        return recommendations

    def _build_detailed_results(
        self,
        detection_results: List[DetectionResult],
        classifications: Dict[Path, ClassificationResult]
    ) -> List[Dict[str, Any]]:
        """Build detailed results for each file."""
        detailed = []

        for result in detection_results:
            classification = classifications.get(result.file_path)

            file_info = {
                'path': str(result.file_path),
                'size_bytes': result.size_bytes,
                'size_mb': round(result.size_bytes / (1024 * 1024), 3),
                'last_modified': datetime.fromtimestamp(result.last_modified).isoformat(),
                'detection_category': result.category.name,
                'detection_reason': result.reason,
                'detection_confidence': result.confidence
            }

            if classification:
                file_info.update({
                    'safety_category': classification.category.name,
                    'safety_confidence': classification.confidence,
                    'safety_reasons': classification.reasons,
                    'safety_warnings': classification.warnings or []
                })

            detailed.append(file_info)

        return detailed

    def _write_json_report(self, report: CleanupReport, report_path: Path) -> None:
        """Write report in JSON format."""
        with open(report_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)

    def _write_yaml_report(self, report: CleanupReport, report_path: Path) -> None:
        """Write report in YAML format."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML reports")

        with open(report_path, 'w') as f:
            yaml.dump(asdict(report), f, default_flow_style=False, default=str)

    def _write_text_report(self, report: CleanupReport, report_path: Path) -> None:
        """Write report in human-readable text format."""
        with open(report_path, 'w') as f:
            f.write(f"Cleanup Report - {report.timestamp}\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Operation: {report.operation_type}\n")
            f.write(f"Directory: {report.root_directory}\n")
            f.write(f"Duration: {report.operation_stats.total_time_seconds:.2f} seconds\n\n")

            f.write("Summary Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Files Scanned: {report.operation_stats.files_scanned}\n")
            f.write(f"Files Detected: {report.operation_stats.files_detected}\n")
            f.write(f"Files Removed: {report.operation_stats.files_removed}\n")
            f.write(f"Total Size Freed: {report.operation_stats.total_size_removed_mb:.2f} MB\n")
            f.write(f"Errors: {report.operation_stats.errors_encountered}\n\n")

            f.write("Safety Analysis:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Total Files: {report.safety_summary.total_files}\n")
            f.write(f"Safe to Remove: {report.safety_summary.safe_to_remove}\n")
            f.write(f"Review Needed: {report.safety_summary.review_needed}\n")
            f.write(f"Must Keep: {report.safety_summary.critical_keep}\n")
            f.write(f"Recommendation: {report.safety_summary.recommendation}\n\n")

            if report.recommendations:
                f.write("Recommendations:\n")
                f.write("-" * 15 + "\n")
                for rec in report.recommendations:
                    f.write(f"• {rec}\n")
                f.write("\n")

            if report.errors:
                f.write("Errors:\n")
                f.write("-" * 7 + "\n")
                for error in report.errors:
                    f.write(f"• {error}\n")
                f.write("\n")

            if report.warnings:
                f.write("Warnings:\n")
                f.write("-" * 9 + "\n")
                for warning in report.warnings:
                    f.write(f"• {warning}\n")

    def _write_html_report(self, report: CleanupReport, report_path: Path) -> None:
        """Write report in HTML format."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Cleanup Report - {report.timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .stats {{ display: flex; gap: 20px; }}
        .stat-box {{ border: 1px solid #ccc; padding: 10px; border-radius: 5px; }}
        .recommendations {{ background-color: #e6f3ff; padding: 10px; border-radius: 5px; }}
        .errors {{ background-color: #ffebee; padding: 10px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
        th {{ background-color: #f0f0f0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Cleanup Report</h1>
        <p>Generated: {report.timestamp}</p>
        <p>Operation: {report.operation_type}</p>
        <p>Directory: {report.root_directory}</p>
    </div>

    <div class="section">
        <h2>Summary Statistics</h2>
        <div class="stats">
            <div class="stat-box">
                <h3>Files</h3>
                <p>Scanned: {report.operation_stats.files_scanned}</p>
                <p>Detected: {report.operation_stats.files_detected}</p>
                <p>Removed: {report.operation_stats.files_removed}</p>
            </div>
            <div class="stat-box">
                <h3>Performance</h3>
                <p>Duration: {report.operation_stats.total_time_seconds:.2f}s</p>
                <p>Size Freed: {report.operation_stats.total_size_removed_mb:.2f} MB</p>
                <p>Errors: {report.operation_stats.errors_encountered}</p>
            </div>
            <div class="stat-box">
                <h3>Safety</h3>
                <p>Safe: {report.safety_summary.safe_to_remove}</p>
                <p>Review: {report.safety_summary.review_needed}</p>
                <p>Keep: {report.safety_summary.critical_keep}</p>
            </div>
        </div>
    </div>

    <div class="section recommendations">
        <h2>Recommendations</h2>
        <ul>
        {''.join(f'<li>{rec}</li>' for rec in report.recommendations)}
        </ul>
    </div>

    {'<div class="section errors"><h2>Errors</h2><ul>' + ''.join(f'<li>{error}</li>' for error in report.errors) + '</ul></div>' if report.errors else ''}

</body>
</html>"""

        report_path.write_text(html_content)