#!/usr/bin/env python3
"""
Comprehensive integration tests for codebase cleanup system.

This module provides end-to-end integration testing for all cleanup
components working together, including Makefile integration, safety
mechanisms, and user workflows.
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import cleanup components
from xraylabtool.cleanup.config import CleanupConfig
from xraylabtool.cleanup.safety_integration import SafetyIntegratedCleanup
from xraylabtool.cleanup.makefile_integration import MakefileCleanupIntegration
from xraylabtool.cleanup.backup_manager import BackupManager
from xraylabtool.cleanup.audit_logger import AuditLogger


class BaseIntegrationTest(unittest.TestCase):
    """Base class for integration tests with common setup"""

    def setUp(self):
        """Setup test environment with realistic project structure"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.project_root = self.temp_dir / "test_project"
        self.project_root.mkdir(parents=True)

        # Create realistic project structure
        self._create_test_project_structure()

        # Initialize cleanup config
        self.config = CleanupConfig()

        # Store original working directory
        self.original_cwd = Path.cwd()

        # Change to test project directory
        os.chdir(self.project_root)

    def tearDown(self):
        """Cleanup test environment"""
        # Restore original working directory
        os.chdir(self.original_cwd)

        # Clean up temporary directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _create_test_project_structure(self):
        """Create realistic project structure for testing"""
        # Source code
        src_dir = self.project_root / "xraylabtool"
        src_dir.mkdir(parents=True)

        for module in ["core.py", "utils.py", "cli.py"]:
            (src_dir / module).write_text(f"# {module}\nprint('module {module}')\n")

        # Tests
        tests_dir = self.project_root / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_core.py").write_text("# Test file\nimport unittest\n")

        # Documentation
        docs_dir = self.project_root / "docs"
        docs_dir.mkdir()
        (docs_dir / "README.md").write_text("# Documentation\n")
        (docs_dir / "api.rst").write_text("API Documentation\n")

        # Configuration files
        (self.project_root / "pyproject.toml").write_text("""
[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.black]
line-length = 88
""")

        (self.project_root / "setup.py").write_text("from setuptools import setup\nsetup(name='test-project')\n")
        (self.project_root / ".gitignore").write_text("__pycache__/\n*.pyc\n.pytest_cache/\n")

        # Build artifacts (to be cleaned)
        build_dir = self.project_root / "build"
        build_dir.mkdir()
        (build_dir / "lib").mkdir()
        (build_dir / "lib" / "compiled.so").write_text("binary data")

        dist_dir = self.project_root / "dist"
        dist_dir.mkdir()
        (dist_dir / "package-1.0.tar.gz").write_text("distribution package")

        # Cache directories
        cache_dirs = [
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            "node_modules"
        ]

        for cache_dir in cache_dirs:
            cache_path = self.project_root / cache_dir
            cache_path.mkdir()
            (cache_path / "cached_file").write_text("cached data")

        # Temporary files
        temp_files = [
            "temp_file.tmp",
            "backup_20241201.bak",
            ".DS_Store",
            "Thumbs.db"
        ]

        for temp_file in temp_files:
            (self.project_root / temp_file).write_text("temporary data")

        # IDE files
        vscode_dir = self.project_root / ".vscode"
        vscode_dir.mkdir()
        (vscode_dir / "settings.json").write_text('{"python.defaultInterpreterPath": "./venv/bin/python"}')

        # Create Makefile
        makefile_content = """
.PHONY: clean test build

clean:
\t@echo "Cleaning project..."
\trm -rf build/ dist/ *.egg-info/
\tfind . -name "*.pyc" -delete
\tfind . -name "__pycache__" -delete

test:
\tpytest tests/

build:
\tpython setup.py build
"""
        (self.project_root / "Makefile").write_text(makefile_content)

        # Initialize git repository
        try:
            subprocess.run(["git", "init"], cwd=self.project_root, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=self.project_root, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.project_root, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=self.project_root, capture_output=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=self.project_root, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Git may not be available in test environment
            pass


class TestBasicCleanupIntegration(BaseIntegrationTest):
    """Test basic cleanup functionality integration"""

    def test_safety_integrated_cleanup_dry_run(self):
        """Test dry-run cleanup with safety integration"""
        safety_cleanup = SafetyIntegratedCleanup(
            project_root=self.project_root,
            config=self.config,
            dry_run=True
        )

        # Identify files to cleanup
        files_to_cleanup = []
        for pattern in ["**/__pycache__", "**/*.pyc", "build", "dist"]:
            files_to_cleanup.extend(self.project_root.glob(pattern))

        # Execute dry-run cleanup
        result = safety_cleanup.execute_safe_cleanup(
            files_to_cleanup=files_to_cleanup,
            operation_type="basic_cleanup",
            force_backup=False,
            user_confirmation=False
        )

        # Verify dry-run results
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("dry_run", False))
        self.assertIn("operation_id", result)
        self.assertIn("files_processed", result)

        # Verify no files were actually deleted in dry-run
        self.assertTrue((self.project_root / "build").exists())
        self.assertTrue((self.project_root / "dist").exists())
        self.assertTrue(list(self.project_root.glob("**/__pycache__")))

    def test_safety_integrated_cleanup_with_backup(self):
        """Test cleanup with backup creation"""
        safety_cleanup = SafetyIntegratedCleanup(
            project_root=self.project_root,
            config=self.config,
            dry_run=False
        )

        # Files to cleanup (less critical ones for this test)
        files_to_cleanup = list(self.project_root.glob("**/__pycache__"))

        # Execute cleanup with backup
        result = safety_cleanup.execute_safe_cleanup(
            files_to_cleanup=files_to_cleanup,
            operation_type="backup_test",
            force_backup=True,
            user_confirmation=False
        )

        # Verify backup was created
        self.assertIn("backup_metadata", result)
        if result["backup_metadata"]:
            backup_metadata = result["backup_metadata"]
            self.assertTrue(backup_metadata.backup_path.exists())
            self.assertGreater(backup_metadata.files_count, 0)

    def test_audit_logging_integration(self):
        """Test audit logging during cleanup operations"""
        safety_cleanup = SafetyIntegratedCleanup(
            project_root=self.project_root,
            config=self.config,
            dry_run=True
        )

        # Execute cleanup to generate audit logs
        files_to_cleanup = list(self.project_root.glob("**/*.tmp"))

        result = safety_cleanup.execute_safe_cleanup(
            files_to_cleanup=files_to_cleanup,
            operation_type="audit_test",
            force_backup=False,
            user_confirmation=False
        )

        # Verify audit logs were created
        audit_dir = self.project_root / ".cleanup_audit"
        self.assertTrue(audit_dir.exists())

        # Check JSON logs
        json_logs = list(audit_dir.glob("json/audit_*.json"))
        self.assertGreater(len(json_logs), 0)

        # Verify operation was logged
        json_content = json_logs[0].read_text()
        self.assertIn("audit_test", json_content)
        self.assertIn("Operation started", json_content)

        # Check human-readable logs
        human_logs = list(audit_dir.glob("human/audit_*.log"))
        self.assertGreater(len(human_logs), 0)

    def test_emergency_stop_integration(self):
        """Test emergency stop mechanism integration"""
        safety_cleanup = SafetyIntegratedCleanup(
            project_root=self.project_root,
            config=self.config,
            dry_run=True
        )

        # Trigger emergency stop during operation
        def trigger_stop():
            time.sleep(0.1)  # Allow operation to start
            safety_cleanup.emergency_manager.trigger_emergency_stop(
                reason="user_abort",
                message="Integration test emergency stop"
            )

        import threading
        stop_thread = threading.Thread(target=trigger_stop)
        stop_thread.start()

        # Execute operation that should be stopped
        files_to_cleanup = list(self.project_root.rglob("*"))  # Many files

        result = safety_cleanup.execute_safe_cleanup(
            files_to_cleanup=files_to_cleanup,
            operation_type="emergency_test",
            force_backup=False,
            user_confirmation=False
        )

        stop_thread.join()

        # Verify emergency stop was triggered
        emergency_report = safety_cleanup.emergency_manager.get_emergency_report()
        self.assertEqual(emergency_report.get("status"), "emergency_stop")


class TestMakefileIntegration(BaseIntegrationTest):
    """Test Makefile-based cleanup integration"""

    def test_makefile_cleanup_detection(self):
        """Test detection of Makefile cleanup commands"""
        makefile_integration = MakefileCleanupIntegration(
            project_root=self.project_root,
            config=self.config
        )

        # Detect existing cleanup commands
        cleanup_info = makefile_integration.analyze_cleanup_commands()

        self.assertIsInstance(cleanup_info, dict)
        self.assertIn("existing_commands", cleanup_info)
        self.assertIn("clean", cleanup_info["existing_commands"])

    def test_makefile_enhancement_dry_run(self):
        """Test Makefile enhancement in dry-run mode"""
        makefile_integration = MakefileCleanupIntegration(
            project_root=self.project_root,
            config=self.config
        )

        # Read original Makefile
        original_makefile = (self.project_root / "Makefile").read_text()

        # Enhance Makefile
        enhancement_result = makefile_integration.enhance_makefile_cleanup(
            dry_run=True,
            backup_original=True
        )

        self.assertIsInstance(enhancement_result, dict)
        self.assertTrue(enhancement_result.get("dry_run", False))

        # Verify original Makefile unchanged in dry-run
        current_makefile = (self.project_root / "Makefile").read_text()
        self.assertEqual(original_makefile, current_makefile)

    @patch('builtins.input', return_value='y')
    def test_makefile_enhancement_execution(self):
        """Test actual Makefile enhancement"""
        makefile_integration = MakefileCleanupIntegration(
            project_root=self.project_root,
            config=self.config
        )

        # Enhance Makefile
        enhancement_result = makefile_integration.enhance_makefile_cleanup(
            dry_run=False,
            backup_original=True
        )

        self.assertIsInstance(enhancement_result, dict)
        self.assertFalse(enhancement_result.get("dry_run", True))

        # Verify backup was created
        makefile_backups = list(self.project_root.glob("Makefile.backup.*"))
        self.assertGreater(len(makefile_backups), 0)

        # Verify Makefile was enhanced
        enhanced_makefile = (self.project_root / "Makefile").read_text()
        self.assertIn("clean-comprehensive", enhanced_makefile)
        self.assertIn("clean-safe", enhanced_makefile)


class TestWorkflowIntegration(BaseIntegrationTest):
    """Test complete workflow integration"""

    def test_complete_cleanup_workflow(self):
        """Test complete cleanup workflow from start to finish"""
        # Step 1: Initialize systems
        safety_cleanup = SafetyIntegratedCleanup(
            project_root=self.project_root,
            config=self.config,
            dry_run=False  # Actual cleanup for workflow test
        )

        makefile_integration = MakefileCleanupIntegration(
            project_root=self.project_root,
            config=self.config
        )

        # Step 2: Analyze project
        analysis_result = makefile_integration.analyze_cleanup_commands()
        self.assertIn("existing_commands", analysis_result)

        # Step 3: Identify cleanup targets
        cleanup_targets = []

        # Add cache directories
        for cache_pattern in ["**/__pycache__", "**/.pytest_cache", "**/.mypy_cache"]:
            cleanup_targets.extend(self.project_root.glob(cache_pattern))

        # Add temporary files
        for temp_pattern in ["**/*.tmp", "**/*.bak"]:
            cleanup_targets.extend(self.project_root.glob(temp_pattern))

        self.assertGreater(len(cleanup_targets), 0)

        # Step 4: Execute safety-wrapped cleanup
        cleanup_result = safety_cleanup.execute_safe_cleanup(
            files_to_cleanup=cleanup_targets,
            operation_type="complete_workflow",
            force_backup=True,
            user_confirmation=False
        )

        # Step 5: Verify workflow completion
        self.assertIsInstance(cleanup_result, dict)
        self.assertIn("operation_id", cleanup_result)

        # Verify backup was created
        if cleanup_result.get("backup_metadata"):
            self.assertTrue(cleanup_result["backup_metadata"].backup_path.exists())

        # Step 6: Verify audit logging
        audit_dir = self.project_root / ".cleanup_audit"
        self.assertTrue(audit_dir.exists())

        audit_summary = safety_cleanup.audit_logger.get_audit_summary()
        self.assertIsInstance(audit_summary, dict)

    def test_error_recovery_workflow(self):
        """Test error recovery and rollback workflow"""
        safety_cleanup = SafetyIntegratedCleanup(
            project_root=self.project_root,
            config=self.config,
            dry_run=False
        )

        # Create backup first
        backup_manager = BackupManager(
            project_root=self.project_root,
            backup_root=self.project_root / ".cleanup_backups"
        )

        test_files = list(self.project_root.glob("**/*.tmp"))
        backup_metadata = backup_manager.create_backup(
            files_to_backup=test_files,
            operation_type="error_recovery_test"
        )

        # Simulate error during cleanup by creating a scenario that should trigger rollback
        # For this test, we'll use the emergency stop mechanism

        def trigger_emergency():
            time.sleep(0.1)
            safety_cleanup.emergency_manager.trigger_emergency_stop(
                reason="critical_error",
                message="Simulated critical error for recovery test"
            )

        import threading
        emergency_thread = threading.Thread(target=trigger_emergency)
        emergency_thread.start()

        # Attempt cleanup that will be interrupted
        try:
            cleanup_result = safety_cleanup.execute_safe_cleanup(
                files_to_cleanup=test_files,
                operation_type="error_recovery_test",
                force_backup=True,
                user_confirmation=False
            )
        except Exception:
            pass  # Expected due to emergency stop

        emergency_thread.join()

        # Verify emergency stop was triggered
        emergency_report = safety_cleanup.emergency_manager.get_emergency_report()
        self.assertEqual(emergency_report.get("status"), "emergency_stop")

        # Test recovery
        restore_result = backup_manager.restore_backup(
            backup_id=backup_metadata.backup_id,
            verify_integrity=True
        )

        self.assertTrue(restore_result.get("success", False))

    def test_concurrent_operations_safety(self):
        """Test safety mechanisms with concurrent operations"""
        safety_cleanup = SafetyIntegratedCleanup(
            project_root=self.project_root,
            config=self.config,
            dry_run=True
        )

        # Create multiple file sets for concurrent operations
        file_sets = []
        base_files = list(self.project_root.rglob("*"))

        # Split files into chunks for concurrent processing
        chunk_size = max(1, len(base_files) // 3)
        for i in range(0, len(base_files), chunk_size):
            file_sets.append(base_files[i:i + chunk_size])

        results = []
        errors = []

        def run_cleanup(file_set, operation_id):
            try:
                result = safety_cleanup.execute_safe_cleanup(
                    files_to_cleanup=file_set,
                    operation_type=f"concurrent_test_{operation_id}",
                    force_backup=False,
                    user_confirmation=False
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Run concurrent operations
        import threading
        threads = []

        for i, file_set in enumerate(file_sets[:2]):  # Limit to 2 concurrent operations
            thread = threading.Thread(target=run_cleanup, args=(file_set, i))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify operations completed
        self.assertEqual(len(errors), 0, f"Concurrent operations failed: {errors}")
        self.assertGreater(len(results), 0)

        # Verify each operation was logged separately
        audit_dir = self.project_root / ".cleanup_audit"
        if audit_dir.exists():
            json_logs = list(audit_dir.glob("json/audit_*.json"))
            if json_logs:
                json_content = json_logs[0].read_text()
                self.assertIn("concurrent_test", json_content)


class TestPerformanceIntegration(BaseIntegrationTest):
    """Test performance characteristics of integrated system"""

    def test_large_project_cleanup_performance(self):
        """Test cleanup performance with large number of files"""
        # Create many files for performance testing
        large_cache_dir = self.project_root / "large_cache"
        large_cache_dir.mkdir()

        # Create 100 cache files
        cache_files = []
        for i in range(100):
            cache_file = large_cache_dir / f"cache_file_{i:03d}.pyc"
            cache_file.write_text(f"cached data {i}")
            cache_files.append(cache_file)

        safety_cleanup = SafetyIntegratedCleanup(
            project_root=self.project_root,
            config=self.config,
            dry_run=True  # Dry run for performance test
        )

        # Measure performance
        start_time = time.time()

        result = safety_cleanup.execute_safe_cleanup(
            files_to_cleanup=cache_files,
            operation_type="performance_test",
            force_backup=False,
            user_confirmation=False
        )

        end_time = time.time()
        duration = end_time - start_time

        # Verify reasonable performance (should complete in under 10 seconds)
        self.assertLess(duration, 10.0, f"Cleanup took too long: {duration:.2f} seconds")

        # Verify all files were processed
        self.assertIn("files_processed", result)
        self.assertEqual(len(result.get("files_processed", [])), 100)

    def test_backup_performance_with_large_files(self):
        """Test backup performance with large files"""
        # Create large files for backup testing
        large_files = []
        for i in range(5):
            large_file = self.project_root / f"large_file_{i}.dat"
            # Create 1MB file
            large_file.write_text("x" * (1024 * 1024))
            large_files.append(large_file)

        backup_manager = BackupManager(
            project_root=self.project_root,
            backup_root=self.project_root / ".performance_backups",
            compression_enabled=True
        )

        # Measure backup performance
        start_time = time.time()

        backup_metadata = backup_manager.create_backup(
            files_to_backup=large_files,
            operation_type="performance_backup_test",
            backup_method="zip"
        )

        end_time = time.time()
        duration = end_time - start_time

        # Verify backup completed
        self.assertTrue(backup_metadata.backup_path.exists())
        self.assertEqual(backup_metadata.files_count, 5)

        # Verify reasonable performance
        self.assertLess(duration, 30.0, f"Backup took too long: {duration:.2f} seconds")

    def test_audit_logging_performance(self):
        """Test audit logging performance with many events"""
        audit_logger = AuditLogger(
            audit_dir=self.project_root / ".performance_audit",
            max_file_size_mb=10
        )

        # Log many events to test performance
        start_time = time.time()

        with audit_logger.operation_context("perf_test", "performance_testing"):
            for i in range(1000):
                audit_logger.log_file_operation(
                    operation_id="perf_test",
                    file_path=Path(f"test_file_{i}.py"),
                    operation="process",
                    success=True,
                    duration_ms=1.0
                )

        end_time = time.time()
        duration = end_time - start_time

        # Verify reasonable performance
        self.assertLess(duration, 5.0, f"Audit logging took too long: {duration:.2f} seconds")

        # Verify logs were created
        json_logs = list((self.project_root / ".performance_audit").glob("json/audit_*.json"))
        self.assertGreater(len(json_logs), 0)


class TestErrorHandlingIntegration(BaseIntegrationTest):
    """Test error handling across integrated components"""

    def test_invalid_file_handling(self):
        """Test handling of invalid or missing files"""
        safety_cleanup = SafetyIntegratedCleanup(
            project_root=self.project_root,
            config=self.config,
            dry_run=True
        )

        # Try to cleanup non-existent files
        invalid_files = [
            self.project_root / "non_existent_file.py",
            self.project_root / "missing" / "nested_file.txt"
        ]

        result = safety_cleanup.execute_safe_cleanup(
            files_to_cleanup=invalid_files,
            operation_type="invalid_file_test",
            force_backup=False,
            user_confirmation=False
        )

        # Verify operation handled invalid files gracefully
        self.assertIsInstance(result, dict)
        # Should not crash, but may report issues

    def test_permission_denied_handling(self):
        """Test handling of permission denied errors"""
        if os.name == 'nt':  # Skip on Windows due to different permission model
            self.skipTest("Permission test not applicable on Windows")

        # Create a file with restricted permissions
        restricted_file = self.project_root / "restricted_file.txt"
        restricted_file.write_text("restricted content")
        restricted_file.chmod(0o000)  # No permissions

        try:
            safety_cleanup = SafetyIntegratedCleanup(
                project_root=self.project_root,
                config=self.config,
                dry_run=True
            )

            result = safety_cleanup.execute_safe_cleanup(
                files_to_cleanup=[restricted_file],
                operation_type="permission_test",
                force_backup=False,
                user_confirmation=False
            )

            # Should handle permission errors gracefully
            self.assertIsInstance(result, dict)

        finally:
            # Restore permissions for cleanup
            try:
                restricted_file.chmod(0o644)
            except:
                pass

    def test_disk_space_error_handling(self):
        """Test handling of disk space issues"""
        # This is a mock test since we can't easily simulate disk space issues
        safety_cleanup = SafetyIntegratedCleanup(
            project_root=self.project_root,
            config=self.config,
            dry_run=True
        )

        # Mock insufficient disk space
        with patch('shutil.disk_usage') as mock_disk_usage:
            mock_disk_usage.return_value = (1000, 900, 50)  # Very low free space

            files_to_cleanup = list(self.project_root.glob("**/*.tmp"))

            result = safety_cleanup.execute_safe_cleanup(
                files_to_cleanup=files_to_cleanup,
                operation_type="disk_space_test",
                force_backup=False,
                user_confirmation=False
            )

            # Should detect low disk space and handle appropriately
            self.assertIsInstance(result, dict)

    def test_interrupted_operation_recovery(self):
        """Test recovery from interrupted operations"""
        safety_cleanup = SafetyIntegratedCleanup(
            project_root=self.project_root,
            config=self.config,
            dry_run=False
        )

        # Start operation and interrupt it
        def interrupt_operation():
            time.sleep(0.05)  # Allow operation to start
            safety_cleanup.emergency_manager.trigger_emergency_stop(
                reason="system_shutdown",
                message="Simulated system shutdown"
            )

        import threading
        interrupt_thread = threading.Thread(target=interrupt_operation)
        interrupt_thread.start()

        files_to_cleanup = list(self.project_root.glob("**/*.tmp"))

        try:
            result = safety_cleanup.execute_safe_cleanup(
                files_to_cleanup=files_to_cleanup,
                operation_type="interruption_test",
                force_backup=True,
                user_confirmation=False
            )
        except Exception:
            pass  # Expected due to interruption

        interrupt_thread.join()

        # Verify emergency stop was triggered
        emergency_report = safety_cleanup.emergency_manager.get_emergency_report()
        self.assertEqual(emergency_report.get("status"), "emergency_stop")
        self.assertEqual(emergency_report.get("reason"), "system_shutdown")

        # Verify audit logs captured the interruption
        audit_dir = self.project_root / ".cleanup_audit"
        if audit_dir.exists():
            json_logs = list(audit_dir.glob("json/audit_*.json"))
            if json_logs:
                json_content = json_logs[0].read_text()
                self.assertIn("emergency", json_content.lower())


if __name__ == "__main__":
    # Create comprehensive integration test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all integration test classes
    test_classes = [
        TestBasicCleanupIntegration,
        TestMakefileIntegration,
        TestWorkflowIntegration,
        TestPerformanceIntegration,
        TestErrorHandlingIntegration
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)

    # Print comprehensive summary
    print(f"\n{'='*80}")
    print(f"INTEGRATION TESTS SUMMARY")
    print(f"{'='*80}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")

    if result.failures:
        print(f"\n{'*'*40} FAILURES {'*'*40}")
        for test, traceback in result.failures:
            print(f"\nFAILED: {test}")
            print(f"Traceback:\n{traceback}")

    if result.errors:
        print(f"\n{'*'*40} ERRORS {'*'*40}")
        for test, traceback in result.errors:
            print(f"\nERROR: {test}")
            print(f"Traceback:\n{traceback}")

    print(f"\n{'='*80}")

    if len(result.failures) + len(result.errors) == 0:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
    else:
        print(f"⚠️  {len(result.failures) + len(result.errors)} integration tests failed")

    print(f"{'='*80}")