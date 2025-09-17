"""
Configuration system for codebase cleanup operations.

This module provides a flexible configuration system that allows customization
of cleanup behavior, safety settings, and detection patterns while maintaining
sensible defaults for the XRayLabTool project.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Union, Any
from dataclasses import dataclass, field, asdict
from copy import deepcopy

from .file_detector import FileCategory

logger = logging.getLogger(__name__)


@dataclass
class SafetySettings:
    """Safety-related configuration options."""

    require_confirmation: bool = True
    create_backup: bool = True
    backup_directory: str = ".cleanup_backup"
    max_file_size_mb: float = 100.0
    strict_mode: bool = True
    dry_run_by_default: bool = True


@dataclass
class DetectionSettings:
    """File detection configuration options."""

    recursive_scan: bool = True
    follow_symlinks: bool = False
    include_hidden_files: bool = False
    use_git_context: bool = True
    enable_content_analysis: bool = True
    stale_threshold_days: int = 90


@dataclass
class PatternSettings:
    """Custom pattern configuration for file detection."""

    obsolete_patterns: List[str] = field(default_factory=list)
    safe_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    custom_rules: Dict[str, str] = field(default_factory=dict)  # pattern -> category


@dataclass
class ReportingSettings:
    """Configuration for logging and reporting."""

    log_level: str = "INFO"
    detailed_logging: bool = True
    create_summary_report: bool = True
    report_format: str = "json"  # json, yaml, text
    report_directory: str = ".cleanup_reports"


class CleanupConfig:
    """
    Comprehensive configuration management for codebase cleanup.

    This class provides a centralized configuration system with validation,
    defaults management, and flexible loading from various sources.
    """

    # Default configuration for XRayLabTool
    DEFAULT_CONFIG = {
        "safety": {
            "require_confirmation": True,
            "create_backup": True,
            "backup_directory": ".cleanup_backup",
            "max_file_size_mb": 100.0,
            "strict_mode": True,
            "dry_run_by_default": True,
        },
        "detection": {
            "recursive_scan": True,
            "follow_symlinks": False,
            "include_hidden_files": False,
            "use_git_context": True,
            "enable_content_analysis": True,
            "stale_threshold_days": 90,
        },
        "patterns": {
            "obsolete_patterns": [
                "performance_baseline_summary.json",
                "baseline_ci_report.json",
                "test_persistence.json",
                "performance_history.json",
                "_xraylabtool_completion.bash",
                "install_completion.py",
                "*.pyc",
                "*.pyo",
                "**/__pycache__/**",
                ".DS_Store",
                "*.tmp",
                "*.bak",
            ],
            "safe_patterns": [
                "*.pyc",
                "*.pyo",
                "**/__pycache__/**/*",
                "build/**/*",
                "dist/**/*",
                "*.egg-info/**/*",
                ".pytest_cache/**/*",
                "htmlcov/**/*",
                ".coverage*",
                ".DS_Store",
                "._*",
                "Thumbs.db",
            ],
            "exclude_patterns": [
                "pyproject.toml",
                "setup.py",
                "requirements*.txt",
                "README*",
                "LICENSE*",
                "CHANGELOG*",
                ".gitignore",
                ".gitattributes",
            ],
            "custom_rules": {
                "*.secret": "CRITICAL_KEEP",
                "*.key": "CRITICAL_KEEP",
                "config_*": "REVIEW_NEEDED",
                "important_*": "CRITICAL_KEEP",
            },
        },
        "reporting": {
            "log_level": "INFO",
            "detailed_logging": True,
            "create_summary_report": True,
            "report_format": "json",
            "report_directory": ".cleanup_reports",
        },
    }

    def __init__(
        self,
        config_dict: Optional[Dict[str, Any]] = None,
        config_file: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize cleanup configuration.

        Args:
            config_dict: Configuration dictionary to use
            config_file: Path to configuration file to load

        Raises:
            ValueError: If configuration validation fails
        """
        # Start with default configuration
        self._config = deepcopy(self.DEFAULT_CONFIG)

        # Load from file if specified
        if config_file:
            file_config = self._load_from_file(config_file)
            self._merge_config(file_config)

        # Override with provided dictionary
        if config_dict:
            self._merge_config(config_dict)

        # Validate the final configuration
        self._validate_config()

        # Create typed settings objects
        self.safety = SafetySettings(**self._config["safety"])
        self.detection = DetectionSettings(**self._config["detection"])
        self.patterns = PatternSettings(**self._config["patterns"])
        self.reporting = ReportingSettings(**self._config["reporting"])

        logger.info("Initialized cleanup configuration")

    @classmethod
    def from_file(cls, config_file: Union[str, Path]) -> "CleanupConfig":
        """
        Create configuration from a file.

        Args:
            config_file: Path to configuration file

        Returns:
            CleanupConfig instance loaded from file
        """
        return cls(config_file=config_file)

    @classmethod
    def create_default_config_file(
        cls, output_path: Union[str, Path], format: str = "json"
    ) -> Path:
        """
        Create a default configuration file.

        Args:
            output_path: Path where to create the configuration file
            format: Format to use ('json' or 'yaml')

        Returns:
            Path to the created configuration file
        """
        output_path = Path(output_path)

        if format.lower() == "json":
            output_path = (
                output_path.with_suffix(".json")
                if not output_path.suffix
                else output_path
            )
            with open(output_path, "w") as f:
                json.dump(cls.DEFAULT_CONFIG, f, indent=2, sort_keys=True)

        elif format.lower() == "yaml":
            try:
                import yaml

                output_path = (
                    output_path.with_suffix(".yaml")
                    if not output_path.suffix
                    else output_path
                )
                with open(output_path, "w") as f:
                    yaml.dump(
                        cls.DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=True
                    )
            except ImportError:
                raise ImportError("PyYAML is required for YAML configuration files")

        else:
            raise ValueError(f"Unsupported configuration format: {format}")

        logger.info(f"Created default configuration file: {output_path}")
        return output_path

    def save_to_file(
        self, output_path: Union[str, Path], format: Optional[str] = None
    ) -> Path:
        """
        Save current configuration to a file.

        Args:
            output_path: Path where to save the configuration
            format: Format to use (auto-detected from extension if None)

        Returns:
            Path to the saved configuration file
        """
        output_path = Path(output_path)

        if format is None:
            format = output_path.suffix.lstrip(".").lower() or "json"

        config_dict = self.to_dict()

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(config_dict, f, indent=2, sort_keys=True)

        elif format == "yaml":
            try:
                import yaml

                with open(output_path, "w") as f:
                    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)
            except ImportError:
                raise ImportError("PyYAML is required for YAML configuration files")

        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved configuration to: {output_path}")
        return output_path

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return {
            "safety": asdict(self.safety),
            "detection": asdict(self.detection),
            "patterns": asdict(self.patterns),
            "reporting": asdict(self.reporting),
        }

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of updates to apply

        Raises:
            ValueError: If updates would create invalid configuration
        """
        # Create a temporary configuration for validation
        temp_config = deepcopy(self._config)
        self._merge_config(updates, temp_config)

        # Validate the updated configuration
        original_config = self._config
        self._config = temp_config
        try:
            self._validate_config()
        except ValueError:
            self._config = original_config  # Restore original
            raise

        # Update the settings objects
        self.safety = SafetySettings(**self._config["safety"])
        self.detection = DetectionSettings(**self._config["detection"])
        self.patterns = PatternSettings(**self._config["patterns"])
        self.reporting = ReportingSettings(**self._config["reporting"])

        logger.info("Updated configuration")

    def get_file_category_mapping(self) -> Dict[str, FileCategory]:
        """
        Get mapping of custom patterns to file categories.

        Returns:
            Dictionary mapping patterns to FileCategory enums
        """
        mapping = {}
        category_map = {
            "SAFE_TO_REMOVE": FileCategory.SAFE_TO_REMOVE,
            "LEGACY": FileCategory.LEGACY,
            "SYSTEM_GENERATED": FileCategory.SYSTEM_GENERATED,
            "BUILD_ARTIFACT": FileCategory.BUILD_ARTIFACT,
            "TEMPORARY": FileCategory.TEMPORARY,
            "REVIEW_NEEDED": FileCategory.REVIEW_NEEDED,
            "CRITICAL_KEEP": FileCategory.CRITICAL_KEEP,
        }

        for pattern, category_name in self.patterns.custom_rules.items():
            if category_name in category_map:
                mapping[pattern] = category_map[category_name]
            else:
                logger.warning(f"Unknown category in custom rules: {category_name}")

        return mapping

    def validate_patterns(self) -> List[str]:
        """
        Validate pattern syntax and report issues.

        Returns:
            List of validation warnings
        """
        import re
        import fnmatch

        warnings = []

        # Check obsolete patterns
        for pattern in self.patterns.obsolete_patterns:
            try:
                # Try to compile as regex or glob
                if "*" in pattern or "?" in pattern:
                    fnmatch.translate(pattern)
                else:
                    re.compile(pattern)
            except Exception as e:
                warnings.append(f"Invalid obsolete pattern '{pattern}': {e}")

        # Check safe patterns
        for pattern in self.patterns.safe_patterns:
            try:
                if "*" in pattern or "?" in pattern:
                    fnmatch.translate(pattern)
                else:
                    re.compile(pattern)
            except Exception as e:
                warnings.append(f"Invalid safe pattern '{pattern}': {e}")

        # Check exclude patterns
        for pattern in self.patterns.exclude_patterns:
            try:
                if "*" in pattern or "?" in pattern:
                    fnmatch.translate(pattern)
                else:
                    re.compile(pattern)
            except Exception as e:
                warnings.append(f"Invalid exclude pattern '{pattern}': {e}")

        return warnings

    def _load_from_file(self, config_file: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file."""
        config_path = Path(config_file)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                if config_path.suffix.lower() == ".json":
                    return json.load(f)
                elif config_path.suffix.lower() in [".yaml", ".yml"]:
                    try:
                        import yaml

                        return yaml.safe_load(f)
                    except ImportError:
                        raise ImportError(
                            "PyYAML is required for YAML configuration files"
                        )
                else:
                    # Try JSON first, then YAML
                    content = f.read()
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        try:
                            import yaml

                            return yaml.safe_load(content)
                        except ImportError:
                            raise ValueError(
                                f"Cannot determine format for {config_path}"
                            )

        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {e}")

    def _merge_config(
        self, source: Dict[str, Any], target: Optional[Dict[str, Any]] = None
    ) -> None:
        """Recursively merge configuration dictionaries."""
        if target is None:
            target = self._config

        for key, value in source.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                self._merge_config(value, target[key])
            else:
                target[key] = deepcopy(value)

    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Validate safety settings
        safety = self._config.get("safety", {})
        if "max_file_size_mb" in safety:
            if (
                not isinstance(safety["max_file_size_mb"], (int, float))
                or safety["max_file_size_mb"] < 0
            ):
                raise ValueError("max_file_size_mb must be a non-negative number")

        # Validate detection settings
        detection = self._config.get("detection", {})
        if "stale_threshold_days" in detection:
            if (
                not isinstance(detection["stale_threshold_days"], int)
                or detection["stale_threshold_days"] < 1
            ):
                raise ValueError("stale_threshold_days must be a positive integer")

        # Validate reporting settings
        reporting = self._config.get("reporting", {})
        if "log_level" in reporting:
            valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            if reporting["log_level"] not in valid_levels:
                raise ValueError(f"log_level must be one of: {valid_levels}")

        if "report_format" in reporting:
            valid_formats = {"json", "yaml", "text"}
            if reporting["report_format"] not in valid_formats:
                raise ValueError(f"report_format must be one of: {valid_formats}")

        # Validate custom rules
        patterns = self._config.get("patterns", {})
        if "custom_rules" in patterns:
            valid_categories = {
                "SAFE_TO_REMOVE",
                "LEGACY",
                "SYSTEM_GENERATED",
                "BUILD_ARTIFACT",
                "TEMPORARY",
                "REVIEW_NEEDED",
                "CRITICAL_KEEP",
            }
            for pattern, category in patterns["custom_rules"].items():
                if category not in valid_categories:
                    raise ValueError(
                        f"Invalid category '{category}' for pattern '{pattern}'. "
                        f"Valid categories: {valid_categories}"
                    )

        logger.debug("Configuration validation passed")


def load_project_config(project_root: Union[str, Path]) -> CleanupConfig:
    """
    Load project-specific cleanup configuration.

    This function looks for configuration files in standard locations
    within the project directory.

    Args:
        project_root: Root directory of the project

    Returns:
        CleanupConfig instance with project-specific settings
    """
    project_root = Path(project_root)

    # Standard configuration file locations
    config_locations = [
        project_root / ".cleanup.json",
        project_root / ".cleanup.yaml",
        project_root / ".cleanup.yml",
        project_root / "cleanup.json",
        project_root / "cleanup.yaml",
        project_root / "cleanup.yml",
        project_root / ".config" / "cleanup.json",
        project_root / ".config" / "cleanup.yaml",
    ]

    # Try to find and load configuration file
    for config_path in config_locations:
        if config_path.exists():
            logger.info(f"Loading project configuration from: {config_path}")
            return CleanupConfig.from_file(config_path)

    # No configuration file found, use defaults
    logger.info("No project configuration found, using defaults")
    return CleanupConfig()


def create_project_config(
    project_root: Union[str, Path], config_name: str = ".cleanup.json"
) -> Path:
    """
    Create a project-specific configuration file with defaults.

    Args:
        project_root: Root directory of the project
        config_name: Name of the configuration file to create

    Returns:
        Path to the created configuration file
    """
    project_root = Path(project_root)
    config_path = project_root / config_name

    return CleanupConfig.create_default_config_file(config_path)
