"""
File metadata analysis for intelligent cleanup operations.

This module provides comprehensive analysis of file metadata, content patterns,
and relationships to help make informed decisions about file removal safety.
"""

import os
import re
import ast
import json
import logging
import mimetypes
from pathlib import Path
from typing import Dict, List, Set, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class FileMetadata:
    """Comprehensive metadata information for a file."""

    path: Path
    size_bytes: int
    created_time: datetime
    modified_time: datetime
    accessed_time: datetime
    mime_type: Optional[str]
    encoding: Optional[str]
    is_text: bool
    is_binary: bool
    is_executable: bool
    file_type: str
    content_hash: Optional[str] = None
    line_count: Optional[int] = None
    char_count: Optional[int] = None
    dependencies: Set[str] = field(default_factory=set)
    imports: Set[str] = field(default_factory=set)
    references: Set[str] = field(default_factory=set)
    content_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrphanAnalysis:
    """Analysis result for orphaned files."""

    file_path: Path
    is_orphaned: bool
    confidence: float
    reasons: List[str]
    related_files: Set[Path] = field(default_factory=set)
    potential_references: Set[str] = field(default_factory=set)


class MetadataAnalyzer:
    """
    Comprehensive file metadata analyzer for cleanup operations.

    This analyzer examines file properties, content patterns, dependencies,
    and relationships to provide deep insights for cleanup decisions.
    """

    # Known temporary file patterns
    TEMP_FILE_PATTERNS = [
        r".*\.tmp$",
        r".*\.temp$",
        r".*\.bak$",
        r".*\.backup$",
        r".*~$",
        r"\.#.*",
        r"#.*#$",
        r".*\.swp$",
        r".*\.swo$",
        r"core\.\d+$",
        r".*\.orig$",
        r".*\.rej$",
    ]

    # Build artifact patterns
    BUILD_ARTIFACT_PATTERNS = [
        r".*\.pyc$",
        r".*\.pyo$",
        r".*\.pyd$",
        r".*\.o$",
        r".*\.obj$",
        r".*\.so$",
        r".*\.dll$",
        r".*\.class$",
        r".*\.jar$",
        r".*\.war$",
    ]

    # Configuration file patterns
    CONFIG_FILE_PATTERNS = [
        r".*\.conf$",
        r".*\.config$",
        r".*\.ini$",
        r".*\.cfg$",
        r".*\.properties$",
        r".*\.env$",
        r"\.env.*$",
    ]

    def __init__(
        self,
        root_path: Union[str, Path],
        max_file_size_mb: float = 100.0,
        enable_content_analysis: bool = True,
    ):
        """
        Initialize the metadata analyzer.

        Args:
            root_path: Root directory for analysis
            max_file_size_mb: Maximum file size to analyze content
            enable_content_analysis: Whether to perform deep content analysis
        """
        self.root_path = Path(root_path).resolve()
        self.max_file_size_mb = max_file_size_mb
        self.enable_content_analysis = enable_content_analysis

        # Compile patterns for efficient matching
        self._temp_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.TEMP_FILE_PATTERNS
        ]
        self._build_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.BUILD_ARTIFACT_PATTERNS
        ]
        self._config_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.CONFIG_FILE_PATTERNS
        ]

        logger.info(f"Initialized MetadataAnalyzer for {self.root_path}")

    def analyze_file(self, file_path: Path) -> FileMetadata:
        """
        Perform comprehensive analysis of a single file.

        Args:
            file_path: Path to the file to analyze

        Returns:
            FileMetadata with comprehensive information
        """
        try:
            stat = file_path.stat()

            # Basic file information
            size_bytes = stat.st_size
            created_time = datetime.fromtimestamp(stat.st_ctime)
            modified_time = datetime.fromtimestamp(stat.st_mtime)
            accessed_time = datetime.fromtimestamp(stat.st_atime)

            # MIME type detection
            mime_type, encoding = mimetypes.guess_type(str(file_path))

            # File type classification
            is_text = self._is_text_file(file_path, mime_type)
            is_binary = not is_text and file_path.is_file()
            is_executable = os.access(file_path, os.X_OK)

            file_type = self._classify_file_type(file_path, mime_type)

            # Create base metadata
            metadata = FileMetadata(
                path=file_path,
                size_bytes=size_bytes,
                created_time=created_time,
                modified_time=modified_time,
                accessed_time=accessed_time,
                mime_type=mime_type,
                encoding=encoding,
                is_text=is_text,
                is_binary=is_binary,
                is_executable=is_executable,
                file_type=file_type,
            )

            # Perform content analysis if enabled and file is not too large
            if (
                self.enable_content_analysis
                and size_bytes < self.max_file_size_mb * 1024 * 1024
            ):
                self._analyze_content(metadata)

            return metadata

        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            # Return minimal metadata on error
            return FileMetadata(
                path=file_path,
                size_bytes=0,
                created_time=datetime.now(),
                modified_time=datetime.now(),
                accessed_time=datetime.now(),
                mime_type=None,
                encoding=None,
                is_text=False,
                is_binary=True,
                is_executable=False,
                file_type="unknown",
            )

    def find_orphaned_files(
        self, file_paths: List[Path], reference_extensions: Optional[Set[str]] = None
    ) -> List[OrphanAnalysis]:
        """
        Find files that appear to be orphaned (no references from other files).

        Args:
            file_paths: List of file paths to analyze
            reference_extensions: File extensions to scan for references

        Returns:
            List of orphan analysis results
        """
        if reference_extensions is None:
            reference_extensions = {
                ".py",
                ".js",
                ".ts",
                ".json",
                ".yaml",
                ".yml",
                ".toml",
            }

        # Build reference database
        reference_db = self._build_reference_database(file_paths, reference_extensions)

        orphan_analyses = []

        for file_path in file_paths:
            analysis = self._analyze_file_references(file_path, reference_db)
            orphan_analyses.append(analysis)

        return orphan_analyses

    def analyze_duplicate_files(
        self, file_paths: List[Path], compare_content: bool = True
    ) -> Dict[str, List[Path]]:
        """
        Find duplicate files based on size and optionally content.

        Args:
            file_paths: List of file paths to analyze
            compare_content: Whether to compare file content for duplicates

        Returns:
            Dictionary mapping hash/size to list of duplicate files
        """
        duplicates = defaultdict(list)
        processed_files = {}

        for file_path in file_paths:
            try:
                stat = file_path.stat()
                size = stat.st_size

                # Group by size first
                size_key = f"size_{size}"

                if compare_content and size < self.max_file_size_mb * 1024 * 1024:
                    # Calculate content hash for small files
                    content_hash = self._calculate_file_hash(file_path)
                    key = f"hash_{content_hash}"
                else:
                    key = size_key

                duplicates[key].append(file_path)

            except Exception as e:
                logger.warning(f"Failed to analyze {file_path} for duplicates: {e}")
                continue

        # Filter out non-duplicates
        return {k: v for k, v in duplicates.items() if len(v) > 1}

    def analyze_file_relationships(
        self, file_paths: List[Path]
    ) -> Dict[Path, Set[Path]]:
        """
        Analyze relationships between files (imports, includes, references).

        Args:
            file_paths: List of file paths to analyze

        Returns:
            Dictionary mapping files to their dependencies
        """
        relationships = {}

        for file_path in file_paths:
            try:
                metadata = self.analyze_file(file_path)
                dependencies = set()

                # Find related files based on imports and references
                for imp in metadata.imports:
                    related_files = self._find_files_by_import(imp, file_paths)
                    dependencies.update(related_files)

                for ref in metadata.references:
                    related_files = self._find_files_by_reference(ref, file_paths)
                    dependencies.update(related_files)

                relationships[file_path] = dependencies

            except Exception as e:
                logger.error(f"Failed to analyze relationships for {file_path}: {e}")
                relationships[file_path] = set()

        return relationships

    def get_cleanup_recommendations(
        self, file_paths: List[Path]
    ) -> Dict[str, List[Path]]:
        """
        Generate cleanup recommendations based on metadata analysis.

        Args:
            file_paths: List of file paths to analyze

        Returns:
            Dictionary categorizing files by cleanup recommendation
        """
        recommendations = {
            "safe_to_remove": [],
            "likely_temporary": [],
            "build_artifacts": [],
            "duplicates": [],
            "orphaned": [],
            "review_needed": [],
            "keep": [],
        }

        # Analyze duplicates
        duplicates = self.analyze_duplicate_files(file_paths)
        duplicate_files = set()
        for file_list in duplicates.values():
            # Keep first file, mark others as duplicates
            duplicate_files.update(file_list[1:])

        # Analyze orphaned files
        orphan_analyses = self.find_orphaned_files(file_paths)
        orphaned_files = {
            analysis.file_path
            for analysis in orphan_analyses
            if analysis.is_orphaned and analysis.confidence > 0.7
        }

        # Categorize files
        for file_path in file_paths:
            try:
                metadata = self.analyze_file(file_path)

                # Check for temporary files
                if self._matches_temp_patterns(file_path):
                    recommendations["likely_temporary"].append(file_path)
                    continue

                # Check for build artifacts
                if self._matches_build_patterns(file_path):
                    recommendations["build_artifacts"].append(file_path)
                    continue

                # Check for duplicates
                if file_path in duplicate_files:
                    recommendations["duplicates"].append(file_path)
                    continue

                # Check for orphaned files
                if file_path in orphaned_files:
                    recommendations["orphaned"].append(file_path)
                    continue

                # Check for critical files
                if self._is_critical_file(file_path, metadata):
                    recommendations["keep"].append(file_path)
                    continue

                # Check for safe removal candidates
                if self._is_safe_to_remove(file_path, metadata):
                    recommendations["safe_to_remove"].append(file_path)
                    continue

                # Default to review needed
                recommendations["review_needed"].append(file_path)

            except Exception as e:
                logger.error(f"Failed to categorize {file_path}: {e}")
                recommendations["review_needed"].append(file_path)

        return recommendations

    def _is_text_file(self, file_path: Path, mime_type: Optional[str]) -> bool:
        """Determine if a file is a text file."""
        if mime_type and mime_type.startswith("text/"):
            return True

        # Check common text file extensions
        text_extensions = {
            ".txt",
            ".md",
            ".rst",
            ".py",
            ".js",
            ".ts",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
            ".conf",
            ".log",
            ".csv",
            ".xml",
            ".html",
            ".css",
            ".sql",
            ".sh",
            ".bash",
            ".fish",
            ".zsh",
        }

        return file_path.suffix.lower() in text_extensions

    def _classify_file_type(self, file_path: Path, mime_type: Optional[str]) -> str:
        """Classify file type based on extension and MIME type."""
        extension = file_path.suffix.lower()

        type_mapping = {
            ".py": "python_source",
            ".js": "javascript",
            ".ts": "typescript",
            ".json": "json_data",
            ".yaml": "yaml_config",
            ".yml": "yaml_config",
            ".toml": "toml_config",
            ".ini": "ini_config",
            ".cfg": "config_file",
            ".conf": "config_file",
            ".log": "log_file",
            ".tmp": "temporary",
            ".bak": "backup",
            ".pyc": "python_bytecode",
            ".so": "shared_library",
            ".dll": "dynamic_library",
        }

        return type_mapping.get(extension, "unknown")

    def _analyze_content(self, metadata: FileMetadata) -> None:
        """Analyze file content and populate content-related metadata."""
        if not metadata.is_text:
            return

        try:
            content = metadata.path.read_text(encoding="utf-8", errors="ignore")

            # Basic content statistics
            metadata.line_count = len(content.split("\n"))
            metadata.char_count = len(content)

            # Content-specific analysis based on file type
            if metadata.file_type == "python_source":
                self._analyze_python_content(content, metadata)
            elif metadata.file_type in ["json_data"]:
                self._analyze_json_content(content, metadata)
            elif metadata.file_type in ["yaml_config", "toml_config"]:
                self._analyze_config_content(content, metadata)

            # General pattern analysis
            self._analyze_general_patterns(content, metadata)

        except Exception as e:
            logger.warning(f"Failed to analyze content of {metadata.path}: {e}")

    def _analyze_python_content(self, content: str, metadata: FileMetadata) -> None:
        """Analyze Python source code content."""
        try:
            tree = ast.parse(content)

            imports = set()
            references = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)

            metadata.imports = imports
            metadata.references = references

            # Additional Python-specific analysis
            metadata.content_summary.update(
                {
                    "has_main": "__main__" in content,
                    "has_classes": any(
                        isinstance(n, ast.ClassDef) for n in ast.walk(tree)
                    ),
                    "has_functions": any(
                        isinstance(n, ast.FunctionDef) for n in ast.walk(tree)
                    ),
                    "import_count": len(imports),
                }
            )

        except SyntaxError:
            logger.debug(f"Syntax error in Python file: {metadata.path}")
        except Exception as e:
            logger.warning(f"Failed to parse Python content: {e}")

    def _analyze_json_content(self, content: str, metadata: FileMetadata) -> None:
        """Analyze JSON file content."""
        try:
            data = json.loads(content)

            metadata.content_summary.update(
                {
                    "json_type": type(data).__name__,
                    "key_count": len(data) if isinstance(data, dict) else None,
                    "is_empty": not bool(data),
                }
            )

            # Look for common patterns
            if isinstance(data, dict):
                keys = set(data.keys())
                metadata.references.update(str(k) for k in keys if isinstance(k, str))

        except json.JSONDecodeError:
            logger.debug(f"Invalid JSON in file: {metadata.path}")

    def _analyze_config_content(self, content: str, metadata: FileMetadata) -> None:
        """Analyze configuration file content."""
        # Look for key-value patterns
        config_patterns = [
            r"(\w+)\s*[=:]\s*(.+)",  # key=value or key: value
            r"\[(\w+)\]",  # [section]
        ]

        references = set()
        for pattern in config_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                references.add(match.group(1))

        metadata.references.update(references)

    def _analyze_general_patterns(self, content: str, metadata: FileMetadata) -> None:
        """Analyze general patterns in text content."""
        # Look for file references
        file_patterns = [
            r'["\']([^"\']+\.[a-zA-Z0-9]+)["\']',  # Quoted file names
            r"([a-zA-Z0-9_/.-]+\.[a-zA-Z0-9]+)",  # File-like paths
        ]

        references = set()
        for pattern in file_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                ref = match.group(1)
                if "." in ref and len(ref) > 2:  # Basic validation
                    references.add(ref)

        metadata.references.update(references)

    def _build_reference_database(
        self, file_paths: List[Path], reference_extensions: Set[str]
    ) -> Dict[str, Set[Path]]:
        """Build a database of file references."""
        reference_db = defaultdict(set)

        for file_path in file_paths:
            if file_path.suffix.lower() in reference_extensions:
                try:
                    metadata = self.analyze_file(file_path)

                    # Add all references from this file
                    for ref in metadata.references:
                        reference_db[ref].add(file_path)

                except Exception as e:
                    logger.warning(f"Failed to build references for {file_path}: {e}")

        return dict(reference_db)

    def _analyze_file_references(
        self, file_path: Path, reference_db: Dict[str, Set[Path]]
    ) -> OrphanAnalysis:
        """Analyze if a file appears to be orphaned."""
        file_name = file_path.name
        reasons = []
        related_files = set()
        potential_references = set()

        # Check if file is referenced by name
        if file_name in reference_db:
            related_files.update(reference_db[file_name])
            potential_references.add(file_name)

        # Check if file stem is referenced (without extension)
        file_stem = file_path.stem
        if file_stem in reference_db:
            related_files.update(reference_db[file_stem])
            potential_references.add(file_stem)

        # Determine orphan status
        is_orphaned = len(related_files) == 0

        if is_orphaned:
            reasons.append("No references found in analyzed files")
            confidence = 0.8
        else:
            reasons.append(f"Referenced by {len(related_files)} files")
            confidence = 0.2

        # Adjust confidence based on file type
        if self._matches_temp_patterns(file_path):
            confidence += 0.1
            reasons.append("Matches temporary file pattern")

        return OrphanAnalysis(
            file_path=file_path,
            is_orphaned=is_orphaned,
            confidence=min(1.0, confidence),
            reasons=reasons,
            related_files=related_files,
            potential_references=potential_references,
        )

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate hash of file content."""
        import hashlib

        try:
            content = file_path.read_bytes()
            return hashlib.md5(content).hexdigest()
        except Exception:
            return f"error_{file_path.stat().st_size}"

    def _find_files_by_import(
        self, import_name: str, file_paths: List[Path]
    ) -> Set[Path]:
        """Find files that might correspond to an import."""
        matches = set()

        for file_path in file_paths:
            if (
                file_path.stem == import_name
                or file_path.name == f"{import_name}.py"
                or import_name in str(file_path)
            ):
                matches.add(file_path)

        return matches

    def _find_files_by_reference(
        self, reference: str, file_paths: List[Path]
    ) -> Set[Path]:
        """Find files that might correspond to a reference."""
        matches = set()

        for file_path in file_paths:
            if reference in str(file_path) or file_path.name == reference:
                matches.add(file_path)

        return matches

    def _matches_temp_patterns(self, file_path: Path) -> bool:
        """Check if file matches temporary file patterns."""
        file_name = file_path.name
        return any(pattern.match(file_name) for pattern in self._temp_patterns)

    def _matches_build_patterns(self, file_path: Path) -> bool:
        """Check if file matches build artifact patterns."""
        file_name = file_path.name
        return any(pattern.match(file_name) for pattern in self._build_patterns)

    def _is_critical_file(self, file_path: Path, metadata: FileMetadata) -> bool:
        """Determine if a file is critical and should be kept."""
        critical_names = {
            "README.md",
            "LICENSE",
            "setup.py",
            "pyproject.toml",
            "requirements.txt",
            ".gitignore",
            "Makefile",
        }

        return (
            file_path.name in critical_names
            or metadata.file_type in ["python_source"]
            or file_path.suffix.lower() in [".py"]
        )

    def _is_safe_to_remove(self, file_path: Path, metadata: FileMetadata) -> bool:
        """Determine if a file appears safe to remove."""
        return (
            self._matches_temp_patterns(file_path)
            or self._matches_build_patterns(file_path)
            or metadata.file_type in ["temporary", "backup", "python_bytecode"]
        )
