#!/usr/bin/env python3
"""
End-to-end workflow validation tests for codebase cleanup system.

This module validates complete user workflows from initial analysis
through cleanup execution to final verification, simulating real-world
usage patterns and scenarios.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

# Import cleanup components
from xraylabtool.cleanup.config import CleanupConfig
from xraylabtool.cleanup.safety_integration import SafetyIntegratedCleanup
from xraylabtool.cleanup.backup_manager import BackupManager
from xraylabtool.cleanup.audit_logger import AuditLogger
from xraylabtool.cleanup.emergency_manager import EmergencyStopManager


class EndToEndWorkflowBase(unittest.TestCase):
    """Base class for end-to-end workflow tests"""

    def setUp(self):
        """Setup realistic project environment for workflow testing"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.project_root = self.temp_dir / "realistic_project"
        self.project_root.mkdir(parents=True)

        # Store original working directory
        self.original_cwd = Path.cwd()

        # Create comprehensive project structure
        self._create_realistic_project()

        # Change to project directory for realistic testing
        os.chdir(self.project_root)

        # Initialize workflow tracking
        self.workflow_steps = []
        self.workflow_start_time = time.time()

    def tearDown(self):
        """Cleanup workflow test environment"""
        # Restore original working directory
        os.chdir(self.original_cwd)

        # Generate workflow report
        self._generate_workflow_report()

        # Clean up temporary directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _create_realistic_project(self):
        """Create a realistic Python project for testing"""
        # Python package structure
        package_dir = self.project_root / "mypackage"
        package_dir.mkdir(parents=True)

        # Core modules
        modules = {
            "__init__.py": "# MyPackage\n__version__ = '1.0.0'\n",
            "core.py": (
                """
# Core functionality
import logging
import numpy as np

logger = logging.getLogger(__name__)

class CoreCalculator:
    def __init__(self):
        self.data = np.array([1, 2, 3, 4, 5])

    def calculate(self, x):
        return np.sum(self.data * x)
"""
            ),
            "utils.py": (
                """
# Utility functions
import os
import json
from pathlib import Path

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def save_results(data, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
"""
            ),
            "cli.py": (
                """
# Command line interface
import argparse
import sys
from .core import CoreCalculator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    calc = CoreCalculator()
    result = calc.calculate([1, 2, 3, 4, 5])
    print(f"Result: {result}")

if __name__ == '__main__':
    main()
"""
            ),
        }

        for filename, content in modules.items():
            (package_dir / filename).write_text(content)

        # Tests directory
        tests_dir = self.project_root / "tests"
        tests_dir.mkdir()

        test_files = {
            "__init__.py": "",
            "test_core.py": (
                """
import unittest
from mypackage.core import CoreCalculator

class TestCore(unittest.TestCase):
    def setUp(self):
        self.calc = CoreCalculator()

    def test_calculate(self):
        result = self.calc.calculate([1, 1, 1, 1, 1])
        self.assertEqual(result, 15)

if __name__ == '__main__':
    unittest.main()
"""
            ),
            "test_utils.py": (
                """
import unittest
import tempfile
import json
from pathlib import Path
from mypackage.utils import load_config, save_results

class TestUtils(unittest.TestCase):
    def test_save_and_load_config(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = {'test': 'data'}
            json.dump(test_data, f)
            config_path = f.name

        loaded = load_config(config_path)
        self.assertEqual(loaded, test_data)

        Path(config_path).unlink()

if __name__ == '__main__':
    unittest.main()
"""
            ),
        }

        for filename, content in test_files.items():
            (tests_dir / filename).write_text(content)

        # Documentation
        docs_dir = self.project_root / "docs"
        docs_dir.mkdir()

        doc_files = {
            "README.md": (
                """
# MyPackage

A sample Python package for testing.

## Installation

```bash
pip install -e .
```

## Usage

```python
from mypackage.core import CoreCalculator
calc = CoreCalculator()
result = calc.calculate([1, 2, 3, 4, 5])
```
"""
            ),
            "api.rst": (
                """
API Documentation
=================

Core Module
-----------

.. automodule:: mypackage.core
   :members:

Utils Module
------------

.. automodule:: mypackage.utils
   :members:
"""
            ),
            "changelog.md": (
                """
# Changelog

## v1.0.0
- Initial release
- Core calculation functionality
- CLI interface
"""
            ),
        }

        for filename, content in doc_files.items():
            (docs_dir / filename).write_text(content)

        # Configuration files
        config_files = {
            "pyproject.toml": (
                """
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm"]
build-backend = "setuptools.build_meta"

[project]
name = "mypackage"
version = "1.0.0"
description = "A sample package for testing"
authors = [{name = "Test Author", email = "test@example.com"}]
dependencies = ["numpy>=1.20.0"]

[project.optional-dependencies]
dev = ["pytest>=6.0", "black", "flake8", "mypy"]
docs = ["sphinx", "sphinx-rtd-theme"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]

[tool.black]
line-length = 88
target-version = ['py38']

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
"""
            ),
            "setup.py": (
                """
from setuptools import setup, find_packages

setup(
    name="mypackage",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["numpy>=1.20.0"],
    entry_points={
        'console_scripts': [
            'mypackage=mypackage.cli:main',
        ],
    },
)
"""
            ),
            ".gitignore": (
                """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Documentation
docs/_build/
"""
            ),
            "requirements.txt": (
                """
numpy>=1.20.0
"""
            ),
            "requirements-dev.txt": (
                """
pytest>=6.0
black
flake8
mypy
coverage
sphinx
sphinx-rtd-theme
"""
            ),
            "tox.ini": (
                """
[tox]
envlist = py38,py39,py310,py311

[testenv]
deps =
    pytest
    coverage
commands =
    coverage run -m pytest
    coverage report
"""
            ),
            "Makefile": (
                """
.PHONY: test clean build install docs lint format

# Test commands
test:
\tpytest tests/ -v

test-coverage:
\tcoverage run -m pytest tests/
\tcoverage report -m
\tcoverage html

# Build commands
build:
\tpython -m build

install:
\tpip install -e .

install-dev:
\tpip install -e .[dev]

# Documentation
docs:
\tsphinx-build -b html docs docs/_build/html

docs-clean:
\trm -rf docs/_build/

# Code quality
lint:
\tflake8 mypackage/ tests/
\tmypy mypackage/

format:
\tblack mypackage/ tests/

format-check:
\tblack --check mypackage/ tests/

# Cleanup commands
clean:
\trm -rf build/ dist/ *.egg-info/
\tfind . -name "*.pyc" -delete
\tfind . -name "__pycache__" -type d -exec rm -rf {} +
\trm -rf .pytest_cache/ .coverage htmlcov/
\trm -rf .tox/ .mypy_cache/

clean-all: clean docs-clean
\trm -rf venv/ .venv/

# Development helpers
dev-setup: install-dev
\tpre-commit install

release-test:
\tpython -m build
\ttwine check dist/*

release:
\tpython -m build
\ttwine upload dist/*
"""
            ),
        }

        for filename, content in config_files.items():
            (self.project_root / filename).write_text(content)

        # Create build artifacts and cache directories
        self._create_build_artifacts()

        # Initialize git repository
        self._initialize_git_repo()

    def _create_build_artifacts(self):
        """Create realistic build artifacts and cache files"""
        # Build directory
        build_dir = self.project_root / "build"
        build_dir.mkdir()

        (build_dir / "lib").mkdir()
        (build_dir / "lib" / "mypackage").mkdir()
        (build_dir / "lib" / "mypackage" / "core.so").write_bytes(
            b"fake compiled module"
        )
        (build_dir / "bdist.win-amd64").mkdir()
        (build_dir / "temp.win-amd64-3.9").mkdir()

        # Dist directory
        dist_dir = self.project_root / "dist"
        dist_dir.mkdir()
        (dist_dir / "mypackage-1.0.0.tar.gz").write_bytes(b"fake source distribution")
        (dist_dir / "mypackage-1.0.0-py3-none-any.whl").write_bytes(b"fake wheel")

        # Egg info
        egg_info_dir = self.project_root / "mypackage.egg-info"
        egg_info_dir.mkdir()
        (egg_info_dir / "PKG-INFO").write_text("Name: mypackage\nVersion: 1.0.0\n")
        (egg_info_dir / "dependency_links.txt").write_text("")
        (egg_info_dir / "requires.txt").write_text("numpy>=1.20.0\n")

        # Python cache directories
        cache_dirs = [
            "mypackage/__pycache__",
            "tests/__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".tox",
        ]

        for cache_dir in cache_dirs:
            cache_path = self.project_root / cache_dir
            cache_path.mkdir(parents=True, exist_ok=True)
            (cache_path / "cache_file.pyc").write_bytes(b"fake cache data")

        # Coverage files
        (self.project_root / ".coverage").write_text("fake coverage data")
        htmlcov_dir = self.project_root / "htmlcov"
        htmlcov_dir.mkdir()
        (htmlcov_dir / "index.html").write_text("<html>Coverage Report</html>")

        # IDE files
        vscode_dir = self.project_root / ".vscode"
        vscode_dir.mkdir()
        (vscode_dir / "settings.json").write_text(
            '{"python.defaultInterpreterPath": "./venv/bin/python"}'
        )

        idea_dir = self.project_root / ".idea"
        idea_dir.mkdir()
        (idea_dir / "workspace.xml").write_text(
            "<?xml version='1.0' encoding='UTF-8'?>\n<project></project>"
        )

        # Temporary and swap files
        temp_files = [
            "temp_file.tmp",
            "backup_20241201.bak",
            ".DS_Store",
            "Thumbs.db",
            "core.py.swp",
            "utils.py.swo",
        ]

        for temp_file in temp_files:
            (self.project_root / temp_file).write_text("temporary data")

        # Virtual environment simulation
        venv_dir = self.project_root / "venv"
        venv_dir.mkdir()
        (venv_dir / "lib").mkdir()
        (venv_dir / "lib" / "python3.9").mkdir()
        (venv_dir / "lib" / "python3.9" / "site-packages").mkdir()

    def _initialize_git_repo(self):
        """Initialize git repository with realistic history"""
        try:
            subprocess.run(
                ["git", "init"], cwd=self.project_root, capture_output=True, check=True
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                cwd=self.project_root,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=self.project_root,
                capture_output=True,
            )

            # Add initial files
            subprocess.run(
                [
                    "git",
                    "add",
                    "mypackage/",
                    "tests/",
                    "docs/",
                    "*.py",
                    "*.toml",
                    "*.txt",
                    "*.md",
                    "Makefile",
                ],
                cwd=self.project_root,
                capture_output=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "Initial project setup"],
                cwd=self.project_root,
                capture_output=True,
            )

            # Create some additional commits
            (self.project_root / "mypackage" / "new_feature.py").write_text(
                "# New feature\npass\n"
            )
            subprocess.run(
                ["git", "add", "mypackage/new_feature.py"],
                cwd=self.project_root,
                capture_output=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "Add new feature"],
                cwd=self.project_root,
                capture_output=True,
            )

        except (subprocess.CalledProcessError, FileNotFoundError):
            # Git may not be available in test environment
            pass

    def _log_workflow_step(self, step_name: str, details: Dict[str, Any] = None):
        """Log a workflow step for reporting"""
        step_info = {
            "step": step_name,
            "timestamp": time.time(),
            "elapsed": time.time() - self.workflow_start_time,
            "details": details or {},
        }
        self.workflow_steps.append(step_info)

    def _generate_workflow_report(self):
        """Generate workflow execution report"""
        total_time = time.time() - self.workflow_start_time

        report = {
            "workflow_duration": total_time,
            "steps_completed": len(self.workflow_steps),
            "steps": self.workflow_steps,
        }

        # Save report for analysis
        report_path = self.temp_dir / "workflow_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)


class TestDeveloperWorkflows(EndToEndWorkflowBase):
    """Test workflows that developers would typically perform"""

    def test_daily_development_cleanup_workflow(self):
        """Test typical daily development cleanup workflow"""
        self._log_workflow_step(
            "workflow_start", {"workflow_type": "daily_development"}
        )

        # Step 1: Developer analyzes current project state
        self._log_workflow_step("project_analysis")

        # Check what cleanup targets exist
        cache_files = list(self.project_root.glob("**/__pycache__"))
        build_files = list(self.project_root.glob("build/**/*"))
        temp_files = list(self.project_root.glob("**/*.tmp"))

        self.assertGreater(len(cache_files), 0)
        self.assertGreater(len(build_files), 0)
        self.assertGreater(len(temp_files), 0)

        # Step 2: Initialize safety-integrated cleanup
        self._log_workflow_step("safety_system_init")

        safety_cleanup = SafetyIntegratedCleanup(
            project_root=self.project_root,
            config=CleanupConfig(),
            dry_run=True,  # Safe default for daily workflow
        )

        # Step 3: Perform dry-run analysis
        self._log_workflow_step("dry_run_analysis")

        all_cleanup_targets = cache_files + build_files + temp_files

        dry_run_result = safety_cleanup.execute_safe_cleanup(
            files_to_cleanup=all_cleanup_targets,
            operation_type="daily_cleanup",
            force_backup=False,
            user_confirmation=False,
        )

        self.assertTrue(dry_run_result.get("dry_run", False))
        self.assertIn("files_would_be_processed", dry_run_result)

        # Step 4: Developer reviews results and proceeds with actual cleanup
        self._log_workflow_step("actual_cleanup")

        # Switch to actual cleanup for less risky files (cache only)
        safety_cleanup.dry_run = False

        cache_cleanup_result = safety_cleanup.execute_safe_cleanup(
            files_to_cleanup=cache_files,
            operation_type="daily_cache_cleanup",
            force_backup=False,
            user_confirmation=False,
        )

        self.assertFalse(cache_cleanup_result.get("dry_run", True))

        # Step 5: Verify cleanup was successful
        self._log_workflow_step("verification")

        remaining_cache_files = list(self.project_root.glob("**/__pycache__"))
        self.assertEqual(len(remaining_cache_files), 0)

        # Step 6: Check audit logs
        self._log_workflow_step("audit_review")

        audit_dir = self.project_root / ".cleanup_audit"
        self.assertTrue(audit_dir.exists())

        json_logs = list(audit_dir.glob("json/audit_*.json"))
        self.assertGreater(len(json_logs), 0)

        self._log_workflow_step("workflow_complete")

    def test_pre_commit_cleanup_workflow(self):
        """Test cleanup workflow before committing code"""
        self._log_workflow_step("workflow_start", {"workflow_type": "pre_commit"})

        # Step 1: Developer wants to clean up before commit
        self._log_workflow_step("pre_commit_analysis")

        # Identify cleanup targets that are safe to remove before commit
        safe_cleanup_targets = []

        # Python cache files
        safe_cleanup_targets.extend(self.project_root.glob("**/__pycache__"))
        safe_cleanup_targets.extend(self.project_root.glob("**/*.pyc"))

        # Test artifacts
        safe_cleanup_targets.extend(self.project_root.glob(".pytest_cache/**/*"))
        safe_cleanup_targets.extend(self.project_root.glob("htmlcov/**/*"))
        if (self.project_root / ".coverage").exists():
            safe_cleanup_targets.append(self.project_root / ".coverage")

        # Temporary files
        safe_cleanup_targets.extend(self.project_root.glob("**/*.tmp"))
        safe_cleanup_targets.extend(self.project_root.glob("**/*.swp"))
        safe_cleanup_targets.extend(self.project_root.glob("**/*.swo"))

        self.assertGreater(len(safe_cleanup_targets), 0)

        # Step 2: Initialize with stricter safety settings
        self._log_workflow_step("safety_system_init")

        config = CleanupConfig()
        config.safety.strict_mode = True

        safety_cleanup = SafetyIntegratedCleanup(
            project_root=self.project_root, config=config, dry_run=False
        )

        # Step 3: Execute cleanup with backup for safety
        self._log_workflow_step("safe_cleanup")

        cleanup_result = safety_cleanup.execute_safe_cleanup(
            files_to_cleanup=safe_cleanup_targets,
            operation_type="pre_commit_cleanup",
            force_backup=True,  # Always backup before commits
            user_confirmation=False,
        )

        # Step 4: Verify backup was created
        self._log_workflow_step("backup_verification")

        self.assertIn("backup_metadata", cleanup_result)
        backup_metadata = cleanup_result["backup_metadata"]
        if backup_metadata:
            self.assertTrue(backup_metadata.backup_path.exists())
            self.assertGreater(backup_metadata.files_count, 0)

        # Step 5: Verify git status is clean for cache files
        self._log_workflow_step("git_status_check")

        try:
            git_status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            # Should not show any cache files in git status
            status_lines = (
                git_status.stdout.strip().split("\n")
                if git_status.stdout.strip()
                else []
            )
            cache_files_in_status = [
                line for line in status_lines if "__pycache__" in line or ".pyc" in line
            ]
            self.assertEqual(len(cache_files_in_status), 0)

        except (subprocess.CalledProcessError, FileNotFoundError):
            pass  # Git not available

        self._log_workflow_step("workflow_complete")

    def test_release_preparation_workflow(self):
        """Test comprehensive cleanup workflow for release preparation"""
        self._log_workflow_step(
            "workflow_start", {"workflow_type": "release_preparation"}
        )

        # Step 1: Comprehensive project analysis
        self._log_workflow_step("comprehensive_analysis")

        # Identify all cleanup targets for release
        cleanup_categories = {
            "build_artifacts": (
                list(self.project_root.glob("build/**/*"))
                + list(self.project_root.glob("dist/**/*"))
                + list(self.project_root.glob("*.egg-info/**/*"))
            ),
            "cache_files": (
                list(self.project_root.glob("**/__pycache__"))
                + list(self.project_root.glob("**/*.pyc"))
                + list(self.project_root.glob(".pytest_cache/**/*"))
                + list(self.project_root.glob(".mypy_cache/**/*"))
            ),
            "test_artifacts": (
                list(self.project_root.glob("htmlcov/**/*"))
                + (
                    [self.project_root / ".coverage"]
                    if (self.project_root / ".coverage").exists()
                    else []
                )
            ),
            "temp_files": (
                list(self.project_root.glob("**/*.tmp"))
                + list(self.project_root.glob("**/*.bak"))
                + list(self.project_root.glob("**/*.swp"))
                + list(self.project_root.glob("**/*.swo"))
            ),
            "ide_files": (
                list(self.project_root.glob(".vscode/**/*"))
                + list(self.project_root.glob(".idea/**/*"))
            ),
            "os_files": (
                list(self.project_root.glob("**/.DS_Store"))
                + list(self.project_root.glob("**/Thumbs.db"))
            ),
        }

        total_files = sum(len(files) for files in cleanup_categories.values())
        self.assertGreater(total_files, 0)

        # Step 2: Initialize with maximum safety for release
        self._log_workflow_step("max_safety_init")

        config = CleanupConfig()
        config.safety.strict_mode = True
        config.safety.require_git_clean = True

        safety_cleanup = SafetyIntegratedCleanup(
            project_root=self.project_root,
            config=config,
            dry_run=True,  # Start with dry-run for release preparation
        )

        # Step 3: Perform comprehensive dry-run
        self._log_workflow_step("comprehensive_dry_run")

        all_cleanup_targets = []
        for category_files in cleanup_categories.values():
            all_cleanup_targets.extend(category_files)

        dry_run_result = safety_cleanup.execute_safe_cleanup(
            files_to_cleanup=all_cleanup_targets,
            operation_type="release_preparation_dry_run",
            force_backup=True,
            user_confirmation=False,
        )

        self.assertTrue(dry_run_result.get("dry_run", False))

        # Step 4: Execute cleanup by category with verification
        self._log_workflow_step("categorized_cleanup")

        safety_cleanup.dry_run = False

        for category, files in cleanup_categories.items():
            if not files:
                continue

            category_result = safety_cleanup.execute_safe_cleanup(
                files_to_cleanup=files,
                operation_type=f"release_cleanup_{category}",
                force_backup=True,
                user_confirmation=False,
            )

            # Verify each category cleanup
            self.assertIn("operation_id", category_result)
            if category_result.get("backup_metadata"):
                self.assertTrue(category_result["backup_metadata"].backup_path.exists())

        # Step 5: Final verification
        self._log_workflow_step("final_verification")

        # Verify critical files remain
        critical_files = [
            "mypackage/__init__.py",
            "setup.py",
            "pyproject.toml",
            "README.md",
            "Makefile",
        ]

        for critical_file in critical_files:
            self.assertTrue(
                (self.project_root / critical_file).exists(),
                f"Critical file {critical_file} was removed",
            )

        # Verify cleanup targets are gone
        remaining_cache = list(self.project_root.glob("**/__pycache__"))
        remaining_build = list(self.project_root.glob("build/**/*"))

        self.assertEqual(len(remaining_cache), 0)
        self.assertEqual(len(remaining_build), 0)

        # Step 6: Generate release report
        self._log_workflow_step("release_report")

        audit_summary = safety_cleanup.audit_logger.get_audit_summary()
        self.assertIn("active_operations", audit_summary)

        self._log_workflow_step("workflow_complete")


class TestMaintenanceWorkflows(EndToEndWorkflowBase):
    """Test maintenance and administrative workflows"""

    def test_scheduled_maintenance_workflow(self):
        """Test scheduled maintenance cleanup workflow"""
        self._log_workflow_step(
            "workflow_start", {"workflow_type": "scheduled_maintenance"}
        )

        # Step 1: Initialize maintenance cleanup
        self._log_workflow_step("maintenance_init")

        config = CleanupConfig()
        config.safety.strict_mode = False  # More permissive for maintenance

        safety_cleanup = SafetyIntegratedCleanup(
            project_root=self.project_root, config=config, dry_run=False
        )

        # Step 2: Comprehensive cleanup including old backups
        self._log_workflow_step("comprehensive_cleanup")

        # Create some old backup files to simulate maintenance scenario
        old_backup_dir = self.project_root / ".cleanup_backups"
        old_backup_dir.mkdir(exist_ok=True)

        # Create old backup files
        old_backups = []
        for i in range(3):
            old_backup = old_backup_dir / f"old_backup_{i}.zip"
            old_backup.write_bytes(b"old backup data")
            old_backups.append(old_backup)

        # Identify all maintenance targets
        maintenance_targets = []

        # Standard cleanup targets
        maintenance_targets.extend(self.project_root.glob("**/__pycache__"))
        maintenance_targets.extend(self.project_root.glob("**/*.pyc"))
        maintenance_targets.extend(self.project_root.glob("build/**/*"))
        maintenance_targets.extend(self.project_root.glob("dist/**/*"))
        maintenance_targets.extend(self.project_root.glob("**/*.tmp"))
        maintenance_targets.extend(self.project_root.glob("**/*.bak"))

        # Old backup files
        maintenance_targets.extend(old_backups)

        # Execute maintenance cleanup
        maintenance_result = safety_cleanup.execute_safe_cleanup(
            files_to_cleanup=maintenance_targets,
            operation_type="scheduled_maintenance",
            force_backup=True,
            user_confirmation=False,
        )

        # Step 3: Verify maintenance completed
        self._log_workflow_step("maintenance_verification")

        self.assertIn("operation_id", maintenance_result)
        self.assertIn("files_processed", maintenance_result)

        # Step 4: Cleanup old audit logs
        self._log_workflow_step("audit_cleanup")

        # Simulate old audit logs cleanup
        audit_dir = self.project_root / ".cleanup_audit"
        if audit_dir.exists():
            old_log_count = safety_cleanup.audit_logger.cleanup_old_logs()
            # Should not fail even if no old logs to clean

        self._log_workflow_step("workflow_complete")

    def test_emergency_recovery_workflow(self):
        """Test emergency recovery workflow"""
        self._log_workflow_step(
            "workflow_start", {"workflow_type": "emergency_recovery"}
        )

        # Step 1: Setup emergency scenario
        self._log_workflow_step("emergency_scenario_setup")

        # Create backup first
        backup_manager = BackupManager(
            project_root=self.project_root,
            backup_root=self.project_root / ".emergency_backups",
        )

        important_files = [
            self.project_root / "mypackage" / "core.py",
            self.project_root / "setup.py",
            self.project_root / "pyproject.toml",
        ]

        backup_metadata = backup_manager.create_backup(
            files_to_backup=important_files, operation_type="emergency_backup"
        )

        # Step 2: Simulate emergency (accidental deletion)
        self._log_workflow_step("emergency_simulation")

        # "Accidentally" delete important files
        deleted_files = []
        for file in important_files:
            if file.exists():
                file_content = file.read_text()
                file.unlink()
                deleted_files.append((file, file_content))

        # Verify files are gone
        for file, _ in deleted_files:
            self.assertFalse(file.exists())

        # Step 3: Emergency recovery initialization
        self._log_workflow_step("emergency_recovery_init")

        safety_cleanup = SafetyIntegratedCleanup(
            project_root=self.project_root, config=CleanupConfig(), dry_run=False
        )

        # Step 4: Execute emergency recovery
        self._log_workflow_step("emergency_recovery_execution")

        recovery_result = backup_manager.restore_backup(
            backup_id=backup_metadata.backup_id, verify_integrity=True
        )

        # Step 5: Verify recovery
        self._log_workflow_step("recovery_verification")

        self.assertTrue(recovery_result.get("success", False))
        self.assertEqual(recovery_result.get("files_restored", 0), len(important_files))

        # Verify files are restored
        for file, original_content in deleted_files:
            self.assertTrue(file.exists())
            restored_content = file.read_text()
            self.assertEqual(restored_content, original_content)

        # Step 6: Post-recovery audit
        self._log_workflow_step("post_recovery_audit")

        # Log recovery event
        from xraylabtool.cleanup.audit_logger import (
            AuditEvent,
            AuditLevel,
            AuditCategory,
        )

        recovery_event = AuditEvent(
            level=AuditLevel.CRITICAL,
            category=AuditCategory.EMERGENCY,
            message=f"Emergency recovery completed for {len(important_files)} files",
            operation_id="emergency_recovery",
            details={
                "backup_id": backup_metadata.backup_id,
                "files_recovered": len(important_files),
                "recovery_method": "backup_restore",
            },
        )

        safety_cleanup.audit_logger.log_event(recovery_event)

        self._log_workflow_step("workflow_complete")

    def test_makefile_integration_workflow(self):
        """Test Makefile integration and enhancement workflow"""
        self._log_workflow_step(
            "workflow_start", {"workflow_type": "makefile_integration"}
        )

        # Step 1: Analyze existing Makefile
        self._log_workflow_step("makefile_analysis")

        makefile_integration = MakefileCleanupIntegration(
            project_root=self.project_root, config=CleanupConfig()
        )

        analysis_result = makefile_integration.analyze_cleanup_commands()

        self.assertIn("existing_commands", analysis_result)
        self.assertIn("clean", analysis_result["existing_commands"])

        # Step 2: Backup original Makefile
        self._log_workflow_step("makefile_backup")

        original_makefile = (self.project_root / "Makefile").read_text()

        # Step 3: Enhance Makefile with safety features
        self._log_workflow_step("makefile_enhancement")

        with patch("builtins.input", return_value="y"):
            enhancement_result = makefile_integration.enhance_makefile_cleanup(
                dry_run=False, backup_original=True
            )

        # Step 4: Verify enhancement
        self._log_workflow_step("enhancement_verification")

        self.assertFalse(enhancement_result.get("dry_run", True))

        enhanced_makefile = (self.project_root / "Makefile").read_text()

        # Verify safety features were added
        expected_features = ["clean-safe", "clean-comprehensive", "backup-before-clean"]

        for feature in expected_features:
            self.assertIn(feature, enhanced_makefile)

        # Verify backup was created
        makefile_backups = list(self.project_root.glob("Makefile.backup.*"))
        self.assertGreater(len(makefile_backups), 0)

        # Step 5: Test enhanced Makefile commands
        self._log_workflow_step("enhanced_makefile_testing")

        # Test dry-run of enhanced clean command
        try:
            result = subprocess.run(
                ["make", "clean-safe", "DRY_RUN=1"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30,
            )
            # Should not fail (though make may not be available)
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ):
            pass  # Make may not be available in test environment

        self._log_workflow_step("workflow_complete")


class TestErrorRecoveryWorkflows(EndToEndWorkflowBase):
    """Test error recovery and edge case workflows"""

    def test_interrupted_operation_recovery(self):
        """Test recovery from interrupted cleanup operation"""
        self._log_workflow_step(
            "workflow_start", {"workflow_type": "interrupted_operation_recovery"}
        )

        # Step 1: Setup operation that will be interrupted
        self._log_workflow_step("operation_setup")

        safety_cleanup = SafetyIntegratedCleanup(
            project_root=self.project_root, config=CleanupConfig(), dry_run=False
        )

        cleanup_targets = list(self.project_root.glob("**/__pycache__"))

        # Step 2: Start operation and interrupt it
        self._log_workflow_step("operation_interruption")

        def interrupt_after_delay():
            time.sleep(0.1)  # Allow operation to start
            safety_cleanup.emergency_manager.trigger_emergency_stop(
                reason="system_shutdown", message="Simulated system interruption"
            )

        import threading

        interrupt_thread = threading.Thread(target=interrupt_after_delay)
        interrupt_thread.start()

        # Execute operation that will be interrupted
        try:
            cleanup_result = safety_cleanup.execute_safe_cleanup(
                files_to_cleanup=cleanup_targets,
                operation_type="interrupted_operation",
                force_backup=True,
                user_confirmation=False,
            )
        except Exception:
            pass  # Expected due to interruption

        interrupt_thread.join()

        # Step 3: Verify interruption was logged
        self._log_workflow_step("interruption_verification")

        emergency_report = safety_cleanup.emergency_manager.get_emergency_report()
        self.assertEqual(emergency_report.get("status"), "emergency_stop")

        # Step 4: Recovery from interruption
        self._log_workflow_step("interruption_recovery")

        # Reset emergency manager for recovery
        safety_cleanup.emergency_manager.reset()

        # Complete the interrupted operation
        recovery_result = safety_cleanup.execute_safe_cleanup(
            files_to_cleanup=cleanup_targets,
            operation_type="recovery_operation",
            force_backup=True,
            user_confirmation=False,
        )

        # Step 5: Verify recovery completed
        self._log_workflow_step("recovery_verification")

        self.assertIn("operation_id", recovery_result)

        # Verify emergency manager is reset
        post_recovery_report = safety_cleanup.emergency_manager.get_emergency_report()
        self.assertEqual(post_recovery_report.get("status"), "no_emergency")

        self._log_workflow_step("workflow_complete")

    def test_corrupted_backup_recovery(self):
        """Test recovery when backup is corrupted"""
        self._log_workflow_step(
            "workflow_start", {"workflow_type": "corrupted_backup_recovery"}
        )

        # Step 1: Create backup
        self._log_workflow_step("backup_creation")

        backup_manager = BackupManager(
            project_root=self.project_root,
            backup_root=self.project_root / ".test_backups",
        )

        test_files = list(self.project_root.glob("**/*.tmp"))

        backup_metadata = backup_manager.create_backup(
            files_to_backup=test_files, operation_type="corruption_test"
        )

        # Step 2: Verify backup integrity initially
        self._log_workflow_step("initial_integrity_check")

        initial_integrity = backup_manager.verify_backup_integrity(
            backup_metadata.backup_id
        )
        self.assertTrue(initial_integrity)

        # Step 3: Simulate backup corruption
        self._log_workflow_step("backup_corruption_simulation")

        # Corrupt backup by modifying files
        if backup_metadata.backup_method == "copy":
            backup_files = list(backup_metadata.backup_path.rglob("*.tmp"))
            if backup_files:
                # Corrupt one backup file
                backup_files[0].write_text("CORRUPTED DATA")

        # Step 4: Detect corruption
        self._log_workflow_step("corruption_detection")

        corrupted_integrity = backup_manager.verify_backup_integrity(
            backup_metadata.backup_id
        )
        self.assertFalse(corrupted_integrity)

        # Step 5: Recovery from corruption
        self._log_workflow_step("corruption_recovery")

        # Create new backup as recovery
        recovery_backup = backup_manager.create_backup(
            files_to_backup=test_files, operation_type="corruption_recovery"
        )

        # Verify new backup integrity
        recovery_integrity = backup_manager.verify_backup_integrity(
            recovery_backup.backup_id
        )
        self.assertTrue(recovery_integrity)

        self._log_workflow_step("workflow_complete")


if __name__ == "__main__":
    # Create comprehensive end-to-end test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all workflow test classes
    test_classes = [
        TestDeveloperWorkflows,
        TestMaintenanceWorkflows,
        TestErrorRecoveryWorkflows,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests with detailed reporting
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)

    # Print comprehensive workflow summary
    print(f"\n{'='*80}")
    print(f"END-TO-END WORKFLOW TESTS SUMMARY")
    print(f"{'='*80}")
    print(f"Total workflows tested: {result.testsRun}")
    print(f"Workflow failures: {len(result.failures)}")
    print(f"Workflow errors: {len(result.errors)}")
    print(
        f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%"
    )

    # Analyze workflow types
    workflow_types = [
        "daily_development",
        "pre_commit",
        "release_preparation",
        "scheduled_maintenance",
        "emergency_recovery",
        "makefile_integration",
        "interrupted_operation_recovery",
        "corrupted_backup_recovery",
    ]

    print(f"\nWorkflow Types Tested:")
    for workflow_type in workflow_types:
        print(f"  ✓ {workflow_type.replace('_', ' ').title()}")

    if result.failures:
        print(f"\n{'*'*40} WORKFLOW FAILURES {'*'*40}")
        for test, traceback in result.failures:
            print(f"\nFAILED WORKFLOW: {test}")
            print(f"Details:\n{traceback}")

    if result.errors:
        print(f"\n{'*'*40} WORKFLOW ERRORS {'*'*40}")
        for test, traceback in result.errors:
            print(f"\nERROR IN WORKFLOW: {test}")
            print(f"Details:\n{traceback}")

    print(f"\n{'='*80}")

    if len(result.failures) + len(result.errors) == 0:
        print("🎉 ALL END-TO-END WORKFLOWS PASSED!")
        print("✅ Developer workflows validated")
        print("✅ Maintenance workflows validated")
        print("✅ Error recovery workflows validated")
    else:
        print(f"⚠️  {len(result.failures) + len(result.errors)} workflow tests failed")

    print(f"{'='*80}")
