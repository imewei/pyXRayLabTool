"""
Obsolete file detection system for XRayLabTool codebase cleanup.

This module provides intelligent detection of obsolete files that can be safely
removed from the repository, including performance artifacts, legacy scripts,
build artifacts, and system-generated files.
"""

import os
import re
import fnmatch
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional, Iterator, Union
from dataclasses import dataclass
from enum import Enum, auto

# Set up logging
logger = logging.getLogger(__name__)


class FileCategory(Enum):
    """Categories for file classification during cleanup analysis."""
    SAFE_TO_REMOVE = auto()      # Files that are definitely safe to remove
    LEGACY = auto()              # Legacy files that are likely obsolete
    SYSTEM_GENERATED = auto()    # System-generated files (OS, IDE)
    BUILD_ARTIFACT = auto()      # Build and compilation artifacts
    TEMPORARY = auto()           # Temporary files and caches
    REVIEW_NEEDED = auto()       # Files that need manual review
    CRITICAL_KEEP = auto()       # Files that must never be removed


@dataclass
class DetectionResult:
    """Result of file detection with metadata and safety classification."""
    file_path: Path
    category: FileCategory
    reason: str
    confidence: float
    size_bytes: int
    last_modified: float
    metadata: Dict[str, any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ObsoleteFileDetector:
    """
    Intelligent detector for obsolete files in the XRayLabTool codebase.

    This class uses pattern matching, file analysis, and safety classification
    to identify files that can be safely removed during cleanup operations.
    """

    # Known obsolete file patterns for XRayLabTool
    OBSOLETE_PATTERNS = {
        'performance_artifacts': [
            'performance_baseline_summary.json',
            'baseline_ci_report.json',
            'test_persistence.json',
            'performance_history.json',
            'benchmark_report.json',
            'benchmark.json',
            'benchmark-results.json'
        ],
        'legacy_completion': [
            '_xraylabtool_completion.bash',
            'install_completion.py'
        ],
        'system_files': [
            '.DS_Store',
            '._*',
            'Thumbs.db',
            'ehthumbs.db',
            'Desktop.ini'
        ],
        'temporary_patterns': [
            '*.tmp',
            '*.bak',
            '*~',
            '*.swp',
            '*.swo',
            'core.*',
            '.#*'
        ],
        'build_artifacts': [
            'build/**/*',
            'dist/**/*',
            '*.egg-info/**/*',
            '**/__pycache__/**/*',
            '**/*.pyc',
            '**/*.pyo',
            '.pytest_cache/**/*',
            'htmlcov/**/*',
            '.coverage*',
            'coverage.xml',
            'coverage.json',
            '.tox/**/*',
            '.mypy_cache/**/*',
            '.ruff_cache/**/*',
            '.benchmarks/**/*'
        ],
        'log_files': [
            '*.log',
            'docs_build.log',
            'test_results.log'
        ],
        'security_reports': [
            'bandit-report.json',
            'bandit_report.json',
            'bandit-claude-report.json',
            'consistency_report.json',
            'CLAUDE_QUALITY_SUMMARY.json',
            'coverage-claude.json',
            'CODE_QUALITY_REPORT.md'
        ]
    }

    def __init__(
        self,
        root_path: Union[str, Path],
        recursive: bool = True,
        use_git_context: bool = False,
        max_file_size_mb: Optional[float] = None,
        exclude_patterns: Optional[List[str]] = None
    ):
        """
        Initialize the obsolete file detector.

        Args:
            root_path: Root directory to scan for obsolete files
            recursive: Whether to scan subdirectories recursively
            use_git_context: Whether to consider Git repository context
            max_file_size_mb: Maximum file size to consider (None for no limit)
            exclude_patterns: Additional patterns to exclude from detection
        """
        self.root_path = Path(root_path).resolve()
        self.recursive = recursive
        self.use_git_context = use_git_context
        self.max_file_size_mb = max_file_size_mb
        self.exclude_patterns = exclude_patterns or []

        # Initialize components
        self._compiled_patterns = self._compile_patterns()

        logger.info(f"Initialized ObsoleteFileDetector for {self.root_path}")

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile file patterns into regex objects for efficient matching."""
        compiled = {}

        for category, patterns in self.OBSOLETE_PATTERNS.items():
            compiled[category] = []
            for pattern in patterns:
                # Convert glob patterns to regex
                if '*' in pattern or '?' in pattern:
                    regex_pattern = fnmatch.translate(pattern)
                    compiled[category].append(re.compile(regex_pattern, re.IGNORECASE))
                else:
                    # Exact filename match
                    escaped = re.escape(pattern)
                    compiled[category].append(re.compile(f'^{escaped}$', re.IGNORECASE))

        return compiled

    def detect_all(self) -> List[DetectionResult]:
        """
        Detect all categories of obsolete files.

        Returns:
            List of detection results for all found obsolete files
        """
        results = []

        logger.info("Starting comprehensive obsolete file detection")

        # Run all detection methods
        results.extend(self.detect_performance_artifacts())
        results.extend(self.detect_legacy_files())
        results.extend(self.detect_system_files())
        results.extend(self.detect_build_artifacts())
        results.extend(self.detect_temporary_files())
        results.extend(self.detect_log_files())
        results.extend(self.detect_security_reports())

        # Remove duplicates and sort by path
        unique_results = self._deduplicate_results(results)
        unique_results.sort(key=lambda r: str(r.file_path))

        logger.info(f"Found {len(unique_results)} obsolete files")
        return unique_results

    def detect_performance_artifacts(self) -> List[DetectionResult]:
        """Detect performance tracking and benchmark artifacts."""
        return self._detect_by_category('performance_artifacts', FileCategory.SAFE_TO_REMOVE)

    def detect_legacy_files(self) -> List[DetectionResult]:
        """Detect legacy completion and installation scripts."""
        return self._detect_by_category('legacy_completion', FileCategory.LEGACY)

    def detect_system_files(self) -> List[DetectionResult]:
        """Detect system-generated files (OS and IDE artifacts)."""
        return self._detect_by_category('system_files', FileCategory.SYSTEM_GENERATED)

    def detect_build_artifacts(self) -> List[DetectionResult]:
        """Detect build artifacts and compilation byproducts."""
        return self._detect_by_category('build_artifacts', FileCategory.BUILD_ARTIFACT)

    def detect_temporary_files(self) -> List[DetectionResult]:
        """Detect temporary files and editor backups."""
        return self._detect_by_category('temporary_patterns', FileCategory.TEMPORARY)

    def detect_log_files(self) -> List[DetectionResult]:
        """Detect log files and debug output."""
        return self._detect_by_category('log_files', FileCategory.SAFE_TO_REMOVE)

    def detect_security_reports(self) -> List[DetectionResult]:
        """Detect security scan reports and quality analysis files."""
        return self._detect_by_category('security_reports', FileCategory.SAFE_TO_REMOVE)

    def _detect_by_category(
        self,
        category: str,
        file_category: FileCategory
    ) -> List[DetectionResult]:
        """
        Detect files matching patterns for a specific category.

        Args:
            category: Pattern category to match against
            file_category: FileCategory to assign to matches

        Returns:
            List of detection results for matching files
        """
        results = []
        patterns = self._compiled_patterns.get(category, [])

        if not patterns:
            logger.warning(f"No patterns found for category: {category}")
            return results

        for file_path in self._scan_files():
            # Check if file matches any pattern in this category
            relative_path = file_path.relative_to(self.root_path)

            for pattern in patterns:
                if pattern.search(str(relative_path)) or pattern.search(file_path.name):
                    if self._should_include_file(file_path):
                        result = self._create_detection_result(
                            file_path,
                            file_category,
                            f"Matches {category} pattern: {pattern.pattern}",
                            confidence=0.9
                        )
                        results.append(result)
                        break  # Only add once per file

        logger.debug(f"Found {len(results)} files in category: {category}")
        return results

    def _scan_files(self) -> Iterator[Path]:
        """
        Scan the root directory for files to analyze.

        Yields:
            Path objects for files to be analyzed
        """
        if self.recursive:
            # Recursive scan
            for root, dirs, files in os.walk(self.root_path):
                # Skip hidden directories and common exclusions
                dirs[:] = [d for d in dirs if not d.startswith('.') or d in ['.git']]

                for filename in files:
                    file_path = Path(root) / filename
                    yield file_path
        else:
            # Non-recursive scan (current directory only)
            for file_path in self.root_path.iterdir():
                if file_path.is_file():
                    yield file_path

    def _should_include_file(self, file_path: Path) -> bool:
        """
        Determine if a file should be included in detection results.

        Args:
            file_path: Path to the file to check

        Returns:
            True if file should be included, False otherwise
        """
        # Check file size limits
        if self.max_file_size_mb is not None:
            try:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb > self.max_file_size_mb:
                    return False
            except (OSError, IOError):
                logger.warning(f"Could not check size of {file_path}")
                return False

        # Check exclude patterns
        relative_path = str(file_path.relative_to(self.root_path))
        for exclude_pattern in self.exclude_patterns:
            if fnmatch.fnmatch(relative_path, exclude_pattern):
                return False

        # Basic safety checks - never include critical files
        critical_names = {
            'pyproject.toml', 'setup.py', 'requirements.txt',
            'README.md', 'LICENSE', 'CHANGELOG.md',
            '__init__.py', '.gitignore', '.gitattributes'
        }

        if file_path.name in critical_names:
            return False

        return True

    def _create_detection_result(
        self,
        file_path: Path,
        category: FileCategory,
        reason: str,
        confidence: float
    ) -> DetectionResult:
        """
        Create a DetectionResult with file metadata.

        Args:
            file_path: Path to the detected file
            category: Classification category
            reason: Human-readable reason for detection
            confidence: Confidence score (0.0 to 1.0)

        Returns:
            DetectionResult with populated metadata
        """
        try:
            stat = file_path.stat()
            size_bytes = stat.st_size
            last_modified = stat.st_mtime
        except (OSError, IOError) as e:
            logger.warning(f"Could not stat {file_path}: {e}")
            size_bytes = 0
            last_modified = 0.0

        return DetectionResult(
            file_path=file_path,
            category=category,
            reason=reason,
            confidence=confidence,
            size_bytes=size_bytes,
            last_modified=last_modified,
            metadata={
                'detector_version': '1.0',
                'relative_path': str(file_path.relative_to(self.root_path))
            }
        )

    def _deduplicate_results(self, results: List[DetectionResult]) -> List[DetectionResult]:
        """
        Remove duplicate detection results based on file path.

        Args:
            results: List of detection results

        Returns:
            Deduplicated list of results
        """
        seen_paths = set()
        unique_results = []

        for result in results:
            if result.file_path not in seen_paths:
                seen_paths.add(result.file_path)
                unique_results.append(result)
            else:
                logger.debug(f"Skipping duplicate detection: {result.file_path}")

        return unique_results

    def get_detection_summary(self, results: List[DetectionResult]) -> Dict[str, any]:
        """
        Generate a summary of detection results.

        Args:
            results: List of detection results

        Returns:
            Summary dictionary with statistics and breakdowns
        """
        if not results:
            return {
                'total_files': 0,
                'total_size_mb': 0.0,
                'categories': {},
                'largest_files': [],
                'oldest_files': []
            }

        # Calculate statistics
        total_files = len(results)
        total_size_bytes = sum(r.size_bytes for r in results)
        total_size_mb = total_size_bytes / (1024 * 1024)

        # Group by category
        categories = {}
        for result in results:
            cat_name = result.category.name
            if cat_name not in categories:
                categories[cat_name] = {'count': 0, 'size_mb': 0.0}
            categories[cat_name]['count'] += 1
            categories[cat_name]['size_mb'] += result.size_bytes / (1024 * 1024)

        # Find largest and oldest files
        largest_files = sorted(results, key=lambda r: r.size_bytes, reverse=True)[:5]
        oldest_files = sorted(results, key=lambda r: r.last_modified)[:5]

        return {
            'total_files': total_files,
            'total_size_mb': round(total_size_mb, 2),
            'categories': categories,
            'largest_files': [
                {
                    'path': str(f.file_path.relative_to(self.root_path)),
                    'size_mb': round(f.size_bytes / (1024 * 1024), 2)
                } for f in largest_files
            ],
            'oldest_files': [
                {
                    'path': str(f.file_path.relative_to(self.root_path)),
                    'last_modified': f.last_modified
                } for f in oldest_files
            ]
        }