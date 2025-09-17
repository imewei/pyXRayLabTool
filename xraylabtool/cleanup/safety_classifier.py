"""
Safety classification system for codebase cleanup operations.

This module provides intelligent classification of files based on safety levels,
helping to prevent accidental removal of important files during cleanup operations.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Union, Pattern
from dataclasses import dataclass
from enum import Enum

from .file_detector import FileCategory

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of file safety classification."""

    category: FileCategory
    confidence: float
    reasons: List[str]
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class SafetyClassifier:
    """
    Intelligent safety classifier for file removal decisions.

    This classifier uses multiple heuristics to determine the safety level
    of removing files, helping prevent accidental deletion of important data.
    """

    # Critical files that should never be removed
    CRITICAL_PATTERNS = {
        "project_files": [
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "requirements*.txt",
            "Pipfile",
            "poetry.lock",
            "package.json",
            "yarn.lock",
            "package-lock.json",
        ],
        "documentation": [
            "README*",
            "LICENSE*",
            "CHANGELOG*",
            "CONTRIBUTING*",
            "CODE_OF_CONDUCT*",
            "SECURITY*",
            "AUTHORS*",
        ],
        "configuration": [
            ".gitignore",
            ".gitattributes",
            ".editorconfig",
            "pyproject.toml",
            "setup.cfg",
            "tox.ini",
            "Makefile",
            "makefile",
            "CMakeLists.txt",
        ],
        "source_code": [
            "*.py",
            "*.pyx",
            "*.pxd",
            "*.c",
            "*.cpp",
            "*.cc",
            "*.cxx",
            "*.h",
            "*.hpp",
            "*.js",
            "*.ts",
            "*.jsx",
            "*.tsx",
            "*.java",
            "*.kt",
            "*.go",
            "*.rs",
            "*.rb",
            "*.php",
            "*.pl",
            "*.sh",
        ],
        "data_files": [
            "*.json",
            "*.yaml",
            "*.yml",
            "*.toml",
            "*.ini",
            "*.cfg",
            "*.xml",
            "*.csv",
            "*.db",
            "*.sqlite*",
        ],
    }

    # Files that are generally safe to remove
    SAFE_PATTERNS = {
        "build_artifacts": [
            "*.pyc",
            "*.pyo",
            "*.pyd",
            "*.so",
            "*.dll",
            "*.o",
            "*.obj",
            "*.a",
            "*.lib",
            "*.class",
            "*.jar",
        ],
        "cache_files": ["*.cache", "*.pid", "*.lock"],
        "temporary_files": ["*.tmp", "*.temp", "*.bak", "*.backup", "*~", "#*#", ".#*"],
        "system_files": [".DS_Store", "Thumbs.db", "ehthumbs.db", "._*", "Desktop.ini"],
        "editor_files": ["*.swp", "*.swo", "*.orig", "*.rej"],
    }

    # Files that need review before removal
    REVIEW_PATTERNS = {
        "config_like": [
            "*.conf",
            "*.config",
            "*.properties",
            "*.env",
            ".env*",
            "*.secret*",
        ],
        "potential_data": ["*.txt", "*.md", "*.rst", "*.log", "*.out", "*.err"],
    }

    # Suspicious patterns that might indicate important files
    SUSPICIOUS_PATTERNS = [
        r"(secret|password|key|token|credential)",
        r"(production|prod|live)",
        r"(backup|archive)",
        r"(config|configuration|settings)",
        r"(important|critical|do[_\-]not[_\-]delete)",
    ]

    def __init__(
        self,
        custom_rules: Optional[Dict[str, FileCategory]] = None,
        strict_mode: bool = True,
    ):
        """
        Initialize the safety classifier.

        Args:
            custom_rules: Custom classification rules (pattern -> category)
            strict_mode: If True, err on the side of caution
        """
        self.custom_rules = custom_rules or {}
        self.strict_mode = strict_mode

        # Compile patterns for efficient matching
        self._compiled_patterns = self._compile_all_patterns()
        self._compiled_suspicious = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.SUSPICIOUS_PATTERNS
        ]

        logger.info(f"Initialized SafetyClassifier (strict_mode={strict_mode})")

    def classify(self, file_path: Path) -> ClassificationResult:
        """
        Classify a file for removal safety.

        Args:
            file_path: Path to the file to classify

        Returns:
            ClassificationResult with safety assessment
        """
        file_name = file_path.name
        file_path_str = str(file_path).lower()
        relative_path_str = str(file_path).lower()

        reasons = []
        warnings = []
        confidence = 0.5  # Default neutral confidence

        # 1. Check custom rules first
        for pattern, category in self.custom_rules.items():
            if self._matches_pattern(file_name, pattern):
                return ClassificationResult(
                    category=category,
                    confidence=0.95,
                    reasons=[f"Matches custom rule: {pattern}"],
                    warnings=[],
                )

        # 2. Check critical patterns (never remove)
        critical_match = self._check_pattern_groups(
            file_path, self._compiled_patterns["critical"]
        )
        if critical_match:
            return ClassificationResult(
                category=FileCategory.CRITICAL_KEEP,
                confidence=0.99,
                reasons=[f"Critical file type: {critical_match}"],
                warnings=[],
            )

        # 3. Check for suspicious content in filename
        suspicious_matches = []
        for pattern in self._compiled_suspicious:
            if pattern.search(file_path_str):
                suspicious_matches.append(pattern.pattern)

        if suspicious_matches:
            warnings.append(
                f"Suspicious patterns found: {', '.join(suspicious_matches)}"
            )
            confidence = max(0.1, confidence - 0.3)  # Reduce confidence

        # 4. Check safe patterns
        safe_match = self._check_pattern_groups(
            file_path, self._compiled_patterns["safe"]
        )
        if safe_match:
            category = FileCategory.SAFE_TO_REMOVE
            confidence = 0.9 - (0.2 if suspicious_matches else 0)
            reasons.append(f"Safe file type: {safe_match}")

        # 5. Check review patterns
        elif self._check_pattern_groups(file_path, self._compiled_patterns["review"]):
            category = FileCategory.REVIEW_NEEDED
            confidence = 0.6
            reasons.append("File type requires manual review")

        # 6. Default classification based on file characteristics
        else:
            category, conf, reason = self._classify_by_characteristics(file_path)
            confidence = conf
            reasons.append(reason)

        # 7. Apply strict mode adjustments
        if (
            self.strict_mode
            and confidence > 0.8
            and category == FileCategory.SAFE_TO_REMOVE
        ):
            if suspicious_matches or self._has_complex_content(file_path):
                category = FileCategory.REVIEW_NEEDED
                confidence = 0.7
                warnings.append("Strict mode: downgraded to review needed")

        return ClassificationResult(
            category=category, confidence=confidence, reasons=reasons, warnings=warnings
        )

    def classify_batch(
        self, file_paths: List[Path]
    ) -> Dict[Path, ClassificationResult]:
        """
        Classify multiple files for safety.

        Args:
            file_paths: List of file paths to classify

        Returns:
            Dictionary mapping file paths to classification results
        """
        results = {}

        for file_path in file_paths:
            try:
                results[file_path] = self.classify(file_path)
            except Exception as e:
                logger.error(f"Failed to classify {file_path}: {e}")
                # Fail safe - mark as needs review
                results[file_path] = ClassificationResult(
                    category=FileCategory.REVIEW_NEEDED,
                    confidence=0.0,
                    reasons=[f"Classification failed: {str(e)}"],
                    warnings=["Failed to classify - manual review required"],
                )

        return results

    def get_safety_summary(
        self, classifications: Dict[Path, ClassificationResult]
    ) -> Dict[str, any]:
        """
        Generate a summary of safety classifications.

        Args:
            classifications: Dictionary of classification results

        Returns:
            Summary statistics and safety analysis
        """
        if not classifications:
            return {"total_files": 0, "categories": {}, "warnings": []}

        # Count by category
        category_counts = {}
        total_warnings = []
        low_confidence_count = 0

        for file_path, result in classifications.items():
            cat_name = result.category.name
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

            if result.warnings:
                total_warnings.extend(
                    [f"{file_path.name}: {warning}" for warning in result.warnings]
                )

            if result.confidence < 0.6:
                low_confidence_count += 1

        # Calculate safety statistics
        safe_count = category_counts.get("SAFE_TO_REMOVE", 0)
        review_count = category_counts.get("REVIEW_NEEDED", 0)
        critical_count = category_counts.get("CRITICAL_KEEP", 0)

        return {
            "total_files": len(classifications),
            "categories": category_counts,
            "safety_stats": {
                "safe_to_remove": safe_count,
                "needs_review": review_count,
                "critical_keep": critical_count,
                "low_confidence": low_confidence_count,
            },
            "warnings": total_warnings[:10],  # Limit to first 10 warnings
            "recommendation": self._get_safety_recommendation(
                safe_count, review_count, critical_count, len(total_warnings)
            ),
        }

    def _compile_all_patterns(self) -> Dict[str, Dict[str, List[Pattern]]]:
        """Compile all pattern groups into regex objects."""
        compiled = {"critical": {}, "safe": {}, "review": {}}

        # Compile critical patterns
        for group_name, patterns in self.CRITICAL_PATTERNS.items():
            compiled["critical"][group_name] = self._compile_pattern_list(patterns)

        # Compile safe patterns
        for group_name, patterns in self.SAFE_PATTERNS.items():
            compiled["safe"][group_name] = self._compile_pattern_list(patterns)

        # Compile review patterns
        for group_name, patterns in self.REVIEW_PATTERNS.items():
            compiled["review"][group_name] = self._compile_pattern_list(patterns)

        return compiled

    def _compile_pattern_list(self, patterns: List[str]) -> List[Pattern]:
        """Compile a list of glob patterns into regex objects."""
        compiled = []

        for pattern in patterns:
            if "*" in pattern or "?" in pattern:
                # Convert glob to regex
                regex_pattern = pattern.replace(".", r"\.")
                regex_pattern = regex_pattern.replace("*", ".*")
                regex_pattern = regex_pattern.replace("?", ".")
                regex_pattern = f"^{regex_pattern}$"
            else:
                # Exact match
                regex_pattern = f"^{re.escape(pattern)}$"

            compiled.append(re.compile(regex_pattern, re.IGNORECASE))

        return compiled

    def _check_pattern_groups(
        self, file_path: Path, pattern_groups: Dict[str, List[Pattern]]
    ) -> Optional[str]:
        """Check if file matches any pattern in the given groups."""
        file_name = file_path.name

        for group_name, patterns in pattern_groups.items():
            for pattern in patterns:
                if pattern.match(file_name):
                    return group_name

        return None

    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches a pattern."""
        if "*" in pattern or "?" in pattern:
            regex_pattern = pattern.replace(".", r"\.")
            regex_pattern = regex_pattern.replace("*", ".*")
            regex_pattern = regex_pattern.replace("?", ".")
            return bool(re.match(f"^{regex_pattern}$", filename, re.IGNORECASE))
        else:
            return filename.lower() == pattern.lower()

    def _classify_by_characteristics(self, file_path: Path) -> tuple:
        """Classify file based on general characteristics."""
        file_name = file_path.name

        # Check for hidden files
        if file_name.startswith(".") and len(file_name) > 1:
            if file_name in [".gitignore", ".gitattributes"]:
                return FileCategory.CRITICAL_KEEP, 0.95, "Essential git configuration"
            else:
                return FileCategory.REVIEW_NEEDED, 0.6, "Hidden file - needs review"

        # Check for files without extensions
        if "." not in file_name:
            return FileCategory.REVIEW_NEEDED, 0.5, "No extension - unclear type"

        # Check file extension patterns
        extension = file_path.suffix.lower()

        # Known safe extensions
        safe_extensions = {".tmp", ".bak", ".cache", ".pid", ".lock"}
        if extension in safe_extensions:
            return FileCategory.SAFE_TO_REMOVE, 0.8, f"Safe extension: {extension}"

        # Known critical extensions
        critical_extensions = {".py", ".js", ".json", ".toml", ".yaml", ".yml"}
        if extension in critical_extensions:
            return FileCategory.REVIEW_NEEDED, 0.4, f"Important extension: {extension}"

        # Default to review needed
        return FileCategory.REVIEW_NEEDED, 0.5, "Unknown file type"

    def _has_complex_content(self, file_path: Path) -> bool:
        """Check if file might have complex content worth preserving."""
        try:
            # Check file size - very small files are likely artifacts
            if file_path.stat().st_size < 100:  # Less than 100 bytes
                return False

            # Check if it's a text file with meaningful content
            if file_path.suffix.lower() in [".txt", ".md", ".log"]:
                return True  # Text files might have important content

            return False

        except (OSError, IOError):
            return True  # If we can't check, assume it might be complex

    def _get_safety_recommendation(
        self,
        safe_count: int,
        review_count: int,
        critical_count: int,
        warning_count: int,
    ) -> str:
        """Generate a safety recommendation based on classification results."""
        total = safe_count + review_count + critical_count

        if total == 0:
            return "No files to process"

        safe_ratio = safe_count / total
        warning_ratio = warning_count / total if total > 0 else 0

        if safe_ratio > 0.8 and warning_ratio < 0.1:
            return "SAFE: Most files appear safe to remove automatically"
        elif safe_ratio > 0.6 and warning_ratio < 0.2:
            return "MOSTLY_SAFE: Proceed with caution, review warnings"
        elif review_count > safe_count:
            return "REVIEW_REQUIRED: Manual review recommended before removal"
        else:
            return "CAUTION: High risk detected, careful manual review essential"
