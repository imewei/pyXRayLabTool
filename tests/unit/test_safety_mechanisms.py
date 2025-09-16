#!/usr/bin/env python3
"""
Comprehensive tests for safety mechanisms and validation systems.

This test suite validates all safety components including backup manager,
safety validator, emergency stop manager, audit logger, and safety integration.
"""

import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the safety components
from xraylabtool.cleanup.backup_manager import BackupManager, BackupMetadata
from xraylabtool.cleanup.safety_validator import SafetyValidator, ValidationReport, SafetyLevel, EmergencyStop
from xraylabtool.cleanup.emergency_manager import EmergencyStopManager, EmergencyStopReason, EmergencyContext
from xraylabtool.cleanup.audit_logger import AuditLogger, AuditEvent, AuditLevel, AuditCategory
from xraylabtool.cleanup.safety_integration import SafetyIntegratedCleanup
from xraylabtool.cleanup.config import CleanupConfig


class TestBackupManager(unittest.TestCase):
    """Test backup manager functionality"""

    def setUp(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.project_root = self.temp_dir / "project"
        self.backup_root = self.temp_dir / "backups"

        self.project_root.mkdir(parents=True)
        self.backup_root.mkdir(parents=True)

        # Create test files
        self.test_files = []
        for i in range(3):
            test_file = self.project_root / f"test_file_{i}.py"
            test_file.write_text(f"# Test file {i}\nprint('test {i}')\n")
            self.test_files.append(test_file)

        self.backup_manager = BackupManager(
            project_root=self.project_root,
            backup_root=self.backup_root,
            compression_enabled=True,
            max_backup_age_days=30
        )

    def tearDown(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir)

    def test_backup_creation(self):
        """Test backup creation with different methods"""
        # Test copy backup
        backup_metadata = self.backup_manager.create_backup(
            files_to_backup=self.test_files,
            operation_type="test_cleanup",
            backup_method="copy"
        )

        self.assertIsInstance(backup_metadata, BackupMetadata)
        self.assertTrue(backup_metadata.backup_path.exists())
        self.assertEqual(backup_metadata.files_count, 3)
        self.assertTrue(backup_metadata.backup_id.startswith("test_cleanup_"))

        # Test zip backup
        backup_metadata_zip = self.backup_manager.create_backup(
            files_to_backup=self.test_files,
            operation_type="test_cleanup",
            backup_method="zip"
        )

        self.assertIsInstance(backup_metadata_zip, BackupMetadata)
        self.assertTrue(backup_metadata_zip.backup_path.exists())
        self.assertTrue(backup_metadata_zip.backup_path.suffix == ".zip")

    def test_backup_integrity_verification(self):
        """Test backup integrity verification"""
        backup_metadata = self.backup_manager.create_backup(
            files_to_backup=self.test_files,
            operation_type="test_cleanup",
            backup_method="copy"
        )

        # Verify integrity
        is_valid = self.backup_manager.verify_backup_integrity(backup_metadata.backup_id)
        self.assertTrue(is_valid)

        # Test with corrupted backup
        # Modify one of the backed-up files to simulate corruption
        if backup_metadata.backup_method == "copy":
            backup_files = list(backup_metadata.backup_path.rglob("*.py"))
            if backup_files:
                backup_files[0].write_text("corrupted content")
                is_valid_corrupted = self.backup_manager.verify_backup_integrity(backup_metadata.backup_id)
                self.assertFalse(is_valid_corrupted)

    def test_backup_restoration(self):
        """Test backup restoration"""
        backup_metadata = self.backup_manager.create_backup(
            files_to_backup=self.test_files,
            operation_type="test_cleanup",
            backup_method="copy"
        )

        # Delete original files
        for file in self.test_files:
            file.unlink()

        # Restore from backup
        restore_result = self.backup_manager.restore_backup(
            backup_id=backup_metadata.backup_id,
            verify_integrity=True
        )

        self.assertTrue(restore_result["success"])
        self.assertEqual(restore_result["files_restored"], 3)

        # Verify restored files exist and have correct content
        for i, file in enumerate(self.test_files):
            self.assertTrue(file.exists())
            content = file.read_text()
            self.assertIn(f"Test file {i}", content)

    def test_backup_cleanup(self):
        """Test backup cleanup for old backups"""
        # Create backups with different ages
        old_backup = self.backup_manager.create_backup(
            files_to_backup=self.test_files,
            operation_type="old_cleanup",
            backup_method="copy"
        )

        # Simulate old backup by modifying timestamp
        old_time = time.time() - (32 * 24 * 3600)  # 32 days ago
        os.utime(old_backup.backup_path, (old_time, old_time))

        recent_backup = self.backup_manager.create_backup(
            files_to_backup=self.test_files,
            operation_type="recent_cleanup",
            backup_method="copy"
        )

        # Cleanup old backups
        removed_count = self.backup_manager.cleanup_old_backups()

        self.assertEqual(removed_count, 1)
        self.assertFalse(old_backup.backup_path.exists())
        self.assertTrue(recent_backup.backup_path.exists())


class TestSafetyValidator(unittest.TestCase):
    """Test safety validator functionality"""

    def setUp(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.project_root = self.temp_dir / "project"
        self.project_root.mkdir(parents=True)

        # Create mock backup manager
        self.backup_manager = Mock()

        self.safety_validator = SafetyValidator(
            project_root=self.project_root,
            backup_manager=self.backup_manager,
            strict_mode=False
        )

        # Create test files
        self.test_files = []
        for i in range(3):
            test_file = self.project_root / f"test_file_{i}.py"
            test_file.write_text(f"# Test file {i}\nprint('test {i}')\n")
            self.test_files.append(test_file)

    def tearDown(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir)

    def test_pre_operation_validation(self):
        """Test pre-operation validation"""
        validation_report = self.safety_validator.validate_pre_operation(
            files_to_process=self.test_files,
            operation_type="cleanup"
        )

        self.assertIsInstance(validation_report, ValidationReport)
        self.assertIsInstance(validation_report.overall_safety_level, SafetyLevel)
        self.assertIsInstance(validation_report.can_proceed, bool)

    def test_post_operation_validation(self):
        """Test post-operation validation"""
        validation_report = self.safety_validator.validate_post_operation(
            files_processed=self.test_files,
            operation_type="cleanup"
        )

        self.assertIsInstance(validation_report, ValidationReport)

    def test_system_resource_validation(self):
        """Test system resource validation"""
        # Test with sufficient resources
        resource_report = self.safety_validator._validate_system_resources()
        self.assertTrue(resource_report.is_valid)

        # Mock insufficient disk space
        with patch('shutil.disk_usage') as mock_disk_usage:
            mock_disk_usage.return_value = (1000, 500, 100)  # total, used, free (very low free space)
            resource_report_low = self.safety_validator._validate_system_resources()
            self.assertFalse(resource_report_low.is_valid)

    def test_emergency_stop_functionality(self):
        """Test emergency stop mechanism"""
        stop_called = False

        def cleanup_callback():
            nonlocal stop_called
            stop_called = True

        emergency_stop = EmergencyStop(cleanup_callback=cleanup_callback)

        self.assertFalse(emergency_stop.is_stop_requested())

        emergency_stop.request_stop("Test stop")
        self.assertTrue(emergency_stop.is_stop_requested())
        self.assertTrue(stop_called)


class TestEmergencyStopManager(unittest.TestCase):
    """Test emergency stop manager functionality"""

    def setUp(self):
        """Setup test environment"""
        self.emergency_manager = EmergencyStopManager()

    def tearDown(self):
        """Cleanup"""
        self.emergency_manager.reset()

    def test_emergency_stop_trigger(self):
        """Test emergency stop triggering"""
        self.assertFalse(self.emergency_manager.is_stop_requested())

        self.emergency_manager.trigger_emergency_stop(
            reason=EmergencyStopReason.USER_ABORT,
            message="Test emergency stop"
        )

        self.assertTrue(self.emergency_manager.is_stop_requested())

        emergency_report = self.emergency_manager.get_emergency_report()
        self.assertEqual(emergency_report["status"], "emergency_stop")
        self.assertEqual(emergency_report["reason"], "user_abort")

    def test_operation_context(self):
        """Test operation context management"""
        test_files = [Path("test1.py"), Path("test2.py")]

        with self.emergency_manager.operation_context(
            operation_id="test_op",
            phase="testing",
            timeout=30.0,
            files_in_progress=test_files
        ) as context:
            self.assertIsNotNone(context)
            self.assertEqual(self.emergency_manager.current_operation_id, "test_op")
            self.assertEqual(self.emergency_manager.current_phase, "testing")

        # Context should be reset after exiting
        self.assertEqual(self.emergency_manager.current_operation_id, "")

    def test_callback_registration(self):
        """Test callback registration and execution"""
        cleanup_called = False
        rollback_called = False

        def test_cleanup(context):
            nonlocal cleanup_called
            cleanup_called = True

        def test_rollback(context):
            nonlocal rollback_called
            rollback_called = True

        self.emergency_manager.register_cleanup_callback(test_cleanup)
        self.emergency_manager.register_rollback_callback(test_rollback)

        self.emergency_manager.trigger_emergency_stop(
            reason=EmergencyStopReason.CRITICAL_ERROR,
            message="Test callbacks"
        )

        # Allow time for callbacks to execute
        time.sleep(0.1)

        self.assertTrue(cleanup_called)
        self.assertTrue(rollback_called)

    def test_timeout_handling(self):
        """Test operation timeout handling"""
        self.emergency_manager.set_operation_context(
            operation_id="timeout_test",
            phase="testing",
            timeout=0.1  # Very short timeout
        )

        # Wait longer than timeout
        time.sleep(0.2)

        # Check timeout should trigger emergency stop
        is_timeout = self.emergency_manager.check_timeout()
        self.assertTrue(is_timeout)
        self.assertTrue(self.emergency_manager.is_stop_requested())


class TestAuditLogger(unittest.TestCase):
    """Test audit logger functionality"""

    def setUp(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.audit_logger = AuditLogger(
            audit_dir=self.temp_dir,
            retention_days=30,
            max_file_size_mb=1,  # Small for testing
            enable_json_logs=True,
            enable_csv_logs=True,
            enable_human_readable=True
        )

    def tearDown(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir)

    def test_event_logging(self):
        """Test basic event logging"""
        event = AuditEvent(
            level=AuditLevel.INFO,
            category=AuditCategory.OPERATION,
            message="Test event",
            operation_id="test_op_001"
        )

        self.audit_logger.log_event(event)

        # Check that log files were created
        json_files = list(self.temp_dir.glob("json/audit_*.json"))
        csv_files = list(self.temp_dir.glob("csv/audit_*.csv"))
        human_files = list(self.temp_dir.glob("human/audit_*.log"))

        self.assertTrue(len(json_files) > 0)
        self.assertTrue(len(csv_files) > 0)
        self.assertTrue(len(human_files) > 0)

        # Verify JSON log content
        json_file = json_files[0]
        json_content = json_file.read_text()
        self.assertIn("Test event", json_content)
        self.assertIn("test_op_001", json_content)

    def test_operation_context(self):
        """Test operation context logging"""
        with self.audit_logger.operation_context(
            operation_id="context_test",
            operation_type="test_operation"
        ) as context:
            self.assertEqual(context.operation_id, "context_test")
            self.assertEqual(context.operation_type, "test_operation")

        # Verify operation start/end events were logged
        json_files = list(self.temp_dir.glob("json/audit_*.json"))
        self.assertTrue(len(json_files) > 0)

        json_content = json_files[0].read_text()
        self.assertIn("Operation started", json_content)
        self.assertIn("Operation ended", json_content)

    def test_file_operation_logging(self):
        """Test file operation logging"""
        test_file = Path("test_file.py")

        self.audit_logger.log_file_operation(
            operation_id="file_test",
            file_path=test_file,
            operation="delete",
            success=True,
            hash_before="abc123",
            hash_after=None,
            size_before=1024,
            size_after=0,
            duration_ms=15.5
        )

        # Verify file operation was logged
        json_files = list(self.temp_dir.glob("json/audit_*.json"))
        json_content = json_files[0].read_text()
        self.assertIn("File delete", json_content)
        self.assertIn("test_file.py", json_content)
        self.assertIn("abc123", json_content)

    def test_integrity_verification(self):
        """Test audit log integrity verification"""
        # Log several events
        for i in range(5):
            event = AuditEvent(
                level=AuditLevel.INFO,
                category=AuditCategory.OPERATION,
                message=f"Test event {i}",
                operation_id=f"test_op_{i:03d}"
            )
            self.audit_logger.log_event(event)

        # Verify integrity
        integrity_result = self.audit_logger.verify_integrity()
        self.assertTrue(integrity_result["integrity_verified"])
        self.assertEqual(integrity_result["events_verified"], 4)  # Chain verification count


class TestSafetyIntegration(unittest.TestCase):
    """Test integrated safety system"""

    def setUp(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.project_root = self.temp_dir / "project"
        self.project_root.mkdir(parents=True)

        # Create test files
        self.test_files = []
        for i in range(3):
            test_file = self.project_root / f"test_file_{i}.py"
            test_file.write_text(f"# Test file {i}\nprint('test {i}')\n")
            self.test_files.append(test_file)

        self.config = CleanupConfig()
        self.safety_system = SafetyIntegratedCleanup(
            project_root=self.project_root,
            config=self.config,
            dry_run=True
        )

    def tearDown(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir)

    def test_dry_run_operation(self):
        """Test dry-run safety operation"""
        result = self.safety_system.execute_safe_cleanup(
            files_to_cleanup=self.test_files,
            operation_type="test_cleanup",
            force_backup=False,
            user_confirmation=False  # Skip confirmation for tests
        )

        self.assertIsInstance(result, dict)
        self.assertIn("operation_id", result)
        self.assertIn("dry_run", result)
        self.assertTrue(result["dry_run"])

        # Files should still exist in dry-run mode
        for file in self.test_files:
            self.assertTrue(file.exists())

    @patch('builtins.input', return_value='y')  # Mock user confirmation
    def test_backup_creation_integration(self):
        """Test integrated backup creation"""
        result = self.safety_system.execute_safe_cleanup(
            files_to_cleanup=self.test_files,
            operation_type="test_cleanup",
            force_backup=True,
            user_confirmation=True
        )

        self.assertIsInstance(result, dict)
        if "backup_metadata" in result and result["backup_metadata"]:
            backup_metadata = result["backup_metadata"]
            self.assertTrue(backup_metadata.backup_path.exists())

    def test_emergency_stop_integration(self):
        """Test emergency stop integration"""
        # Start operation in separate thread
        def run_operation():
            self.safety_system.execute_safe_cleanup(
                files_to_cleanup=self.test_files,
                operation_type="long_cleanup",
                force_backup=False,
                user_confirmation=False
            )

        operation_thread = threading.Thread(target=run_operation)
        operation_thread.start()

        # Trigger emergency stop
        time.sleep(0.1)  # Allow operation to start
        self.safety_system.emergency_manager.trigger_emergency_stop(
            reason=EmergencyStopReason.USER_ABORT,
            message="Test emergency stop"
        )

        operation_thread.join(timeout=5)
        self.assertFalse(operation_thread.is_alive())

    def test_validation_integration(self):
        """Test validation system integration"""
        # Test with non-existent files (should fail validation)
        non_existent_files = [Path("non_existent.py"), Path("also_missing.py")]

        result = self.safety_system.execute_safe_cleanup(
            files_to_cleanup=non_existent_files,
            operation_type="test_cleanup",
            force_backup=False,
            user_confirmation=False
        )

        # Operation should abort due to validation failure
        self.assertIn("aborted", result.get("status", "").lower())

    def test_audit_logging_integration(self):
        """Test audit logging integration"""
        result = self.safety_system.execute_safe_cleanup(
            files_to_cleanup=self.test_files,
            operation_type="audit_test",
            force_backup=False,
            user_confirmation=False
        )

        # Check that audit logs were created
        audit_dir = self.project_root / ".cleanup_audit"
        self.assertTrue(audit_dir.exists())

        json_logs = list(audit_dir.glob("json/audit_*.json"))
        self.assertTrue(len(json_logs) > 0)

        # Verify operation was logged
        json_content = json_logs[0].read_text()
        self.assertIn("audit_test", json_content)

    def test_full_safety_cycle(self):
        """Test complete safety cycle with all mechanisms"""
        # Create a more comprehensive test
        result = self.safety_system.execute_safe_cleanup(
            files_to_cleanup=self.test_files,
            operation_type="comprehensive_test",
            force_backup=True,
            user_confirmation=False
        )

        # Verify all safety components were involved
        self.assertIsInstance(result, dict)

        # Check audit logs exist
        audit_dir = self.project_root / ".cleanup_audit"
        self.assertTrue(audit_dir.exists())

        # Check no emergency stop occurred
        emergency_report = self.safety_system.emergency_manager.get_emergency_report()
        self.assertEqual(emergency_report.get("status"), "no_emergency")

        # Get audit summary
        audit_summary = self.safety_system.audit_logger.get_audit_summary()
        self.assertIsInstance(audit_summary, dict)
        self.assertIn("integrity_chain_length", audit_summary)


class TestSafetyMechanismsStressTest(unittest.TestCase):
    """Stress tests for safety mechanisms under load"""

    def setUp(self):
        """Setup stress test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.project_root = self.temp_dir / "project"
        self.project_root.mkdir(parents=True)

        # Create many test files
        self.test_files = []
        for i in range(50):  # More files for stress testing
            test_file = self.project_root / f"test_file_{i:03d}.py"
            content = f"# Test file {i}\n" + "print('test data')\n" * 10  # Larger files
            test_file.write_text(content)
            self.test_files.append(test_file)

    def tearDown(self):
        """Cleanup stress test environment"""
        shutil.rmtree(self.temp_dir)

    def test_large_backup_operations(self):
        """Test backup operations with many files"""
        backup_manager = BackupManager(
            project_root=self.project_root,
            backup_root=self.temp_dir / "backups",
            compression_enabled=True
        )

        backup_metadata = backup_manager.create_backup(
            files_to_backup=self.test_files,
            operation_type="stress_test",
            backup_method="zip"
        )

        self.assertEqual(backup_metadata.files_count, 50)
        self.assertTrue(backup_metadata.backup_path.exists())

        # Test integrity
        is_valid = backup_manager.verify_backup_integrity(backup_metadata.backup_id)
        self.assertTrue(is_valid)

    def test_concurrent_audit_logging(self):
        """Test audit logging under concurrent load"""
        audit_logger = AuditLogger(
            audit_dir=self.temp_dir / "audit",
            max_file_size_mb=5
        )

        def log_events(thread_id):
            for i in range(20):
                event = AuditEvent(
                    level=AuditLevel.INFO,
                    category=AuditCategory.OPERATION,
                    message=f"Concurrent event from thread {thread_id}, iteration {i}",
                    operation_id=f"thread_{thread_id}_op_{i}"
                )
                audit_logger.log_event(event)

        # Run multiple threads logging simultaneously
        threads = []
        for t in range(5):
            thread = threading.Thread(target=log_events, args=(t,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all events were logged
        json_files = list((self.temp_dir / "audit").glob("json/audit_*.json"))
        self.assertTrue(len(json_files) > 0)

        total_events = 0
        for json_file in json_files:
            content = json_file.read_text()
            total_events += content.count("Concurrent event")

        self.assertEqual(total_events, 100)  # 5 threads * 20 events each


if __name__ == "__main__":
    # Create comprehensive test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestBackupManager,
        TestSafetyValidator,
        TestEmergencyStopManager,
        TestAuditLogger,
        TestSafetyIntegration,
        TestSafetyMechanismsStressTest
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Safety Mechanisms Test Summary")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")

    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    print(f"{'='*60}")
    print("Safety mechanisms testing completed!")
    print(f"{'='*60}")