"""
Basic file cleanup utilities.

Simplified cleanup functionality for removing common build artifacts
and cache files.
"""

import os
from pathlib import Path
import shutil


def clean_build_artifacts(project_root: Path) -> list[str]:
    """
    Remove common build artifacts and cache files.

    Args:
        project_root: Root directory of the project

    Returns:
        List of removed items
    """
    removed_items = []

    # Common patterns to remove
    patterns = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/.pytest_cache",
        "**/build",
        "**/dist",
        "**/*.egg-info",
        "**/.mypy_cache",
        "**/.ruff_cache",
        "**/.xraylabtool_cache",
    ]

    for pattern in patterns:
        for path in project_root.glob(pattern):
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                removed_items.append(str(path))
            except (OSError, PermissionError):
                continue

    return removed_items


__all__ = ["clean_build_artifacts"]
