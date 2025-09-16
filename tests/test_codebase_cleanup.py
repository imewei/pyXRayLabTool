"""
Comprehensive tests for the codebase cleanup and file detection system.

This module tests the file detection algorithms, safety classification,
and cleanup operations for maintaining a clean repository.
"""

import pytest
import os
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Set

try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

# Import the modules we'll be testing
from xraylabtool.cleanup.file_detector import (
    ObsoleteFileDetector,
    FileCategory,
    DetectionResult
)
from xraylabtool.cleanup.safety_classifier import SafetyClassifier
from xraylabtool.cleanup.git_analyzer import GitChangeAnalyzer
from xraylabtool.cleanup.config import CleanupConfig


class TestObsoleteFileDetector:
    """Test suite for obsolete file detection algorithms."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.detector = None  # Will be set when module is implemented

        # Create test files for detection
        self.test_files = {
            'obsolete': [
                'performance_baseline_summary.json',
                'baseline_ci_report.json',
                'test_persistence.json',
                'performance_history.json',
                '_xraylabtool_completion.bash',
                'install_completion.py',
                '.DS_Store'
            ],
            'safe_to_keep': [
                'requirements.txt',
                'pyproject.toml',
                'setup.py',
                'README.md',
                'CHANGELOG.md'
            ],
            'build_artifacts': [
                'build/lib/module.so',
                'dist/package-1.0.tar.gz',
                'package.egg-info/PKG-INFO',
                '__pycache__/module.cpython-312.pyc',
                '.pytest_cache/v/cache/nodeids',
                'htmlcov/index.html',
                '.coverage'
            ],
            'temporary': [
                'temp_file.tmp',
                'backup_file.bak',
                'file~',
                'core.12345',
                '.swp'
            ]
        }

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_files(self, file_dict: Dict[str, List[str]]) -> Dict[str, List[Path]]:
        """Create test files in temporary directory."""
        created_files = {}

        for category, files in file_dict.items():
            created_files[category] = []
            for file_name in files:
                file_path = Path(self.temp_dir) / file_name

                # Create parent directories if needed
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # Create file with some content
                file_path.write_text(f"Test content for {file_name}")
                created_files[category].append(file_path)

        return created_files

    def test_detect_obsolete_performance_files(self):
        """Test detection of obsolete performance tracking files."""
        created_files = self.create_test_files({'obsolete': self.test_files['obsolete'][:4]})

        detector = ObsoleteFileDetector(self.temp_dir)
        results = detector.detect_performance_artifacts()

        # Should detect all performance files
        detected_files = [r.file_path for r in results]
        assert len(detected_files) == 4

        # Check specific files are detected
        file_names = [f.name for f in detected_files]
        assert 'performance_baseline_summary.json' in file_names
        assert 'baseline_ci_report.json' in file_names
        assert 'test_persistence.json' in file_names
        assert 'performance_history.json' in file_names

        # All should be classified as safe to remove
        for result in results:
            assert result.category == FileCategory.SAFE_TO_REMOVE

    def test_detect_legacy_completion_files(self):
        """Test detection of legacy shell completion files."""
        legacy_files = self.test_files['obsolete'][4:6]  # completion files
        created_files = self.create_test_files({'obsolete': legacy_files})

        detector = ObsoleteFileDetector(self.temp_dir)
        results = detector.detect_legacy_files()

        detected_files = [r.file_path for r in results]
        assert len(detected_files) == 2

        file_names = [f.name for f in detected_files]
        assert '_xraylabtool_completion.bash' in file_names
        assert 'install_completion.py' in file_names

        # Should be classified as legacy
        for result in results:
            assert result.category == FileCategory.LEGACY

    def test_detect_system_files(self):
        """Test detection of system-generated files."""
        system_files = ['.DS_Store']
        created_files = self.create_test_files({'obsolete': system_files})

        detector = ObsoleteFileDetector(self.temp_dir)
        results = detector.detect_system_files()

        assert len(results) == 1
        assert results[0].file_path.name == '.DS_Store'
        assert results[0].category == FileCategory.SYSTEM_GENERATED

    @pytest.mark.skip(reason="Implementation pending")
    def test_detect_build_artifacts(self):
        """Test detection of build artifacts and cache files."""
        created_files = self.create_test_files({'build_artifacts': self.test_files['build_artifacts']})

        detector = ObsoleteFileDetector(self.temp_dir)
        results = detector.detect_build_artifacts()

        detected_files = [r.file_path for r in results]
        assert len(detected_files) == len(self.test_files['build_artifacts'])

        # Check for specific build patterns
        file_names = [f.name for f in detected_files]
        assert any('__pycache__' in str(f) for f in detected_files)
        assert any('.coverage' in name for name in file_names)
        assert any('htmlcov' in str(f) for f in detected_files)

    @pytest.mark.skip(reason="Implementation pending")
    def test_false_positive_prevention(self):
        """Test that important files are never marked for removal."""
        created_files = self.create_test_files({'safe_to_keep': self.test_files['safe_to_keep']})

        detector = ObsoleteFileDetector(self.temp_dir)
        results = detector.detect_all()

        # Get all detected files
        detected_paths = {str(r.file_path) for r in results}

        # None of the important files should be detected
        for category_files in created_files.values():
            for file_path in category_files:
                assert str(file_path) not in detected_paths, f"Important file {file_path} was incorrectly marked for removal"

    @pytest.mark.skip(reason="Implementation pending")
    def test_recursive_detection(self):
        """Test recursive detection in subdirectories."""
        # Create nested directory structure
        nested_obsolete = {
            'nested': [
                'subdir/__pycache__/module.pyc',
                'deep/nested/.DS_Store',
                'build/temp/file.o'
            ]
        }
        created_files = self.create_test_files(nested_obsolete)

        detector = ObsoleteFileDetector(self.temp_dir, recursive=True)
        results = detector.detect_all()

        # Should find files in subdirectories
        assert len(results) >= 3

        # Check that nested paths are detected
        detected_paths = [str(r.file_path) for r in results]
        assert any('__pycache__' in path for path in detected_paths)
        assert any('.DS_Store' in path for path in detected_paths)

    @pytest.mark.skip(reason="Implementation pending")
    def test_size_based_filtering(self):
        """Test filtering based on file size thresholds."""
        # Create files of different sizes
        large_file = Path(self.temp_dir) / 'large_temp.tmp'
        small_file = Path(self.temp_dir) / 'small_temp.tmp'

        large_file.write_text('x' * 10000)  # 10KB
        small_file.write_text('x' * 100)    # 100B

        detector = ObsoleteFileDetector(self.temp_dir, max_file_size_mb=0.001)  # 1KB limit
        results = detector.detect_temporary_files()

        # Should only detect small file
        detected_names = [r.file_path.name for r in results]
        assert 'small_temp.tmp' in detected_names
        assert 'large_temp.tmp' not in detected_names


class TestSafetyClassifier:
    """Test suite for file safety classification."""

    def test_classify_safe_files(self):
        """Test classification of files that are safe to remove."""
        classifier = SafetyClassifier(strict_mode=False)

        safe_files = [
            '__pycache__/module.pyc',
            '.DS_Store',
            'build/temp.tmp',
            'cache.bak'
        ]

        for file_path in safe_files:
            result = classifier.classify(Path(file_path))
            assert result.category == FileCategory.SAFE_TO_REMOVE
            assert result.confidence > 0.8

    @pytest.mark.skip(reason="Implementation pending")
    def test_classify_risky_files(self):
        """Test classification of files that might be risky to remove."""
        classifier = SafetyClassifier()

        risky_files = [
            'requirements.txt',  # Could be modified
            'config.ini',        # Configuration file
            'data.db'           # Database file
        ]

        for file_path in risky_files:
            result = classifier.classify(Path(file_path))
            assert result.category == FileCategory.REVIEW_NEEDED
            assert result.confidence < 0.9

    @pytest.mark.skip(reason="Implementation pending")
    def test_classify_critical_files(self):
        """Test classification of files that should never be removed."""
        classifier = SafetyClassifier()

        critical_files = [
            'pyproject.toml',
            'setup.py',
            'README.md',
            'LICENSE',
            '__init__.py'
        ]

        for file_path in critical_files:
            result = classifier.classify(Path(file_path))
            assert result.category == FileCategory.CRITICAL_KEEP

    @pytest.mark.skip(reason="Implementation pending")
    def test_custom_classification_rules(self):
        """Test custom classification rules."""
        custom_rules = {
            '*.custom': FileCategory.SAFE_TO_REMOVE,
            'important_*': FileCategory.CRITICAL_KEEP
        }

        classifier = SafetyClassifier(custom_rules=custom_rules)

        # Test custom extension
        result1 = classifier.classify(Path('test.custom'))
        assert result1.category == FileCategory.SAFE_TO_REMOVE

        # Test custom pattern
        result2 = classifier.classify(Path('important_data.txt'))
        assert result2.category == FileCategory.CRITICAL_KEEP


class TestGitChangeAnalyzer:
    """Test suite for Git integration and change analysis."""

    def setup_method(self):
        """Set up Git repository for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)

        if GIT_AVAILABLE:
            # Initialize git repository
            self.repo = git.Repo.init(self.repo_path)

            # Configure git
            self.repo.config_writer().set_value("user", "name", "Test User").release()
            self.repo.config_writer().set_value("user", "email", "test@example.com").release()
        else:
            self.repo = None

    def teardown_method(self):
        """Clean up test repository."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.skipif(not GIT_AVAILABLE, reason="GitPython not available")
    def test_track_file_history(self):
        """Test tracking file creation and modification history."""
        # Create and commit a file
        test_file = self.repo_path / 'test.json'
        test_file.write_text('{"test": "data"}')

        self.repo.index.add(['test.json'])
        commit1 = self.repo.index.commit("Add test file")

        # Modify and commit again
        test_file.write_text('{"test": "modified"}')
        self.repo.index.add(['test.json'])
        commit2 = self.repo.index.commit("Modify test file")

        analyzer = GitChangeAnalyzer(self.repo_path)
        history = analyzer.get_file_history(test_file)

        assert len(history) == 2
        assert history[0].message == "Add test file"
        assert history[1].message == "Modify test file"

    @pytest.mark.skipif(not GIT_AVAILABLE, reason="GitPython not available")
    def test_detect_untracked_files(self):
        """Test detection of untracked files."""
        # Create untracked file
        untracked_file = self.repo_path / 'untracked.tmp'
        untracked_file.write_text('temporary content')

        analyzer = GitChangeAnalyzer(self.repo_path)
        untracked_files = analyzer.get_untracked_files()

        assert len(untracked_files) == 1
        assert untracked_files[0].name == 'untracked.tmp'

    @pytest.mark.skipif(not GIT_AVAILABLE, reason="GitPython not available")
    def test_analyze_file_age(self):
        """Test analysis of file creation and last modification times."""
        # Create file with specific timestamp
        old_file = self.repo_path / 'old_file.json'
        old_file.write_text('{"old": true}')

        self.repo.index.add(['old_file.json'])
        self.repo.index.commit("Add old file")

        analyzer = GitChangeAnalyzer(self.repo_path)
        age_analysis = analyzer.analyze_file_age(old_file)

        assert age_analysis.days_since_creation >= 0
        assert age_analysis.days_since_last_commit >= 0
        assert age_analysis.is_stale is not None


class TestCleanupConfig:
    """Test suite for cleanup configuration system."""

    def test_default_configuration(self):
        """Test default cleanup configuration."""
        config = CleanupConfig()

        # Test default patterns
        assert len(config.patterns.obsolete_patterns) > 0
        assert '*.pyc' in config.patterns.safe_patterns
        assert '.DS_Store' in config.patterns.safe_patterns

        # Test safety settings
        assert config.safety.require_confirmation is True
        assert config.safety.create_backup is True

    @pytest.mark.skip(reason="Implementation pending")
    def test_custom_configuration(self):
        """Test custom configuration loading."""
        config_data = {
            'obsolete_patterns': ['*.custom', 'temp_*'],
            'safety': {
                'require_confirmation': False,
                'max_file_size_mb': 100
            }
        }

        config = CleanupConfig(config_data)

        assert '*.custom' in config.obsolete_patterns
        assert config.require_confirmation is False
        assert config.max_file_size_mb == 100

    @pytest.mark.skip(reason="Implementation pending")
    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid configuration
        invalid_config = {
            'safety': {
                'max_file_size_mb': -1  # Invalid negative size
            }
        }

        with pytest.raises(ValueError):
            CleanupConfig(invalid_config)

    @pytest.mark.skip(reason="Implementation pending")
    def test_config_file_loading(self):
        """Test loading configuration from file."""
        config_file = Path(self.temp_dir) / 'cleanup_config.json'
        config_data = {
            'obsolete_patterns': ['test_*.json'],
            'safety': {'require_confirmation': False}
        }

        config_file.write_text(json.dumps(config_data))

        config = CleanupConfig.from_file(config_file)
        assert 'test_*.json' in config.obsolete_patterns
        assert config.require_confirmation is False


class TestIntegration:
    """Integration tests for complete detection workflow."""

    def setup_method(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()

        # Create realistic project structure
        self.project_structure = {
            'root_files': [
                'performance_baseline_summary.json',
                'requirements.txt',
                'pyproject.toml',
                '.DS_Store'
            ],
            'build_artifacts': [
                'build/lib/module.so',
                'dist/package.tar.gz',
                '__pycache__/module.pyc'
            ],
            'source_code': [
                'src/__init__.py',
                'src/main.py',
                'tests/test_main.py'
            ]
        }

    def teardown_method(self):
        """Clean up integration test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.skip(reason="Implementation pending")
    def test_complete_detection_workflow(self):
        """Test complete file detection and classification workflow."""
        # Create project structure
        for category, files in self.project_structure.items():
            for file_path in files:
                full_path = Path(self.temp_dir) / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(f"Content for {file_path}")

        # Run complete detection
        detector = ObsoleteFileDetector(self.temp_dir)
        all_results = detector.detect_all()

        # Verify detection results
        assert len(all_results) > 0

        # Check that safe files are detected
        detected_names = [r.file_path.name for r in all_results]
        assert 'performance_baseline_summary.json' in detected_names
        assert '.DS_Store' in detected_names

        # Check that important files are NOT detected
        assert 'pyproject.toml' not in detected_names
        assert 'requirements.txt' not in detected_names
        assert '__init__.py' not in detected_names

    @pytest.mark.skipif(not GIT_AVAILABLE, reason="GitPython not available")
    def test_detection_with_git_context(self):
        """Test detection with Git repository context."""
        # Initialize git repository
        repo = git.Repo.init(self.temp_dir)
        repo.config_writer().set_value("user", "name", "Test User").release()
        repo.config_writer().set_value("user", "email", "test@example.com").release()

        # Create and commit some files
        important_file = Path(self.temp_dir) / 'important.py'
        important_file.write_text('print("important")')

        temp_file = Path(self.temp_dir) / 'temp.json'
        temp_file.write_text('{"temporary": true}')

        repo.index.add(['important.py'])
        repo.index.commit("Add important file")

        # Run detection with Git context
        detector = ObsoleteFileDetector(self.temp_dir, use_git_context=True)
        results = detector.detect_all()

        # Untracked temp file should be detected
        detected_names = [r.file_path.name for r in results]
        assert 'temp.json' in detected_names

        # Tracked important file should not be detected
        assert 'important.py' not in detected_names


if __name__ == '__main__':
    pytest.main([__file__, '-v'])