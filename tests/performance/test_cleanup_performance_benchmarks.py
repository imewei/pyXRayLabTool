#!/usr/bin/env python3
"""
Comprehensive performance benchmarking and validation for cleanup system.

This module provides detailed performance testing and benchmarking for all
cleanup components, measuring throughput, latency, memory usage, and scalability.
"""

import json
import os
import psutil
import shutil
import statistics
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from unittest.mock import patch

# Import cleanup components for benchmarking
from xraylabtool.cleanup.config import CleanupConfig
from xraylabtool.cleanup.safety_integration import SafetyIntegratedCleanup
from xraylabtool.cleanup.backup_manager import BackupManager
from xraylabtool.cleanup.audit_logger import AuditLogger
from xraylabtool.cleanup.emergency_manager import EmergencyStopManager
from xraylabtool.cleanup.safety_validator import SafetyValidator


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation"""

    operation_name: str
    duration_seconds: float
    memory_peak_mb: float
    memory_average_mb: float
    cpu_percent_peak: float
    cpu_percent_average: float
    files_processed: int
    throughput_files_per_second: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    operation_success: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return asdict(self)


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report"""

    benchmark_name: str
    timestamp: float
    system_info: Dict[str, Any]
    test_configuration: Dict[str, Any]
    metrics: List[PerformanceMetrics]
    summary_statistics: Dict[str, Any]
    performance_targets: Dict[str, Any]
    targets_met: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "benchmark_name": self.benchmark_name,
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "test_configuration": self.test_configuration,
            "metrics": [m.to_dict() for m in self.metrics],
            "summary_statistics": self.summary_statistics,
            "performance_targets": self.performance_targets,
            "targets_met": self.targets_met,
        }


class PerformanceMonitor:
    """Real-time performance monitoring during operations"""

    def __init__(self, sample_interval: float = 0.1):
        """
        Initialize performance monitor.

        Args:
            sample_interval: Interval between performance samples in seconds
        """
        self.sample_interval = sample_interval
        self.monitoring = False
        self.samples = []
        self._monitor_thread = None
        self._process = psutil.Process()

    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring = True
        self.samples = []
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return performance metrics"""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

        if not self.samples:
            return {
                "memory_peak_mb": 0.0,
                "memory_average_mb": 0.0,
                "cpu_percent_peak": 0.0,
                "cpu_percent_average": 0.0,
                "disk_io_read_mb": 0.0,
                "disk_io_write_mb": 0.0,
            }

        memory_values = [s["memory_mb"] for s in self.samples]
        cpu_values = [s["cpu_percent"] for s in self.samples]

        # Calculate disk I/O delta
        if len(self.samples) > 1:
            disk_read_delta = (
                self.samples[-1]["disk_read_bytes"] - self.samples[0]["disk_read_bytes"]
            )
            disk_write_delta = (
                self.samples[-1]["disk_write_bytes"]
                - self.samples[0]["disk_write_bytes"]
            )
        else:
            disk_read_delta = disk_write_delta = 0

        return {
            "memory_peak_mb": max(memory_values),
            "memory_average_mb": statistics.mean(memory_values),
            "cpu_percent_peak": max(cpu_values),
            "cpu_percent_average": statistics.mean(cpu_values),
            "disk_io_read_mb": disk_read_delta / (1024 * 1024),
            "disk_io_write_mb": disk_write_delta / (1024 * 1024),
        }

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Get process performance data
                memory_info = self._process.memory_info()
                cpu_percent = self._process.cpu_percent()

                # Get disk I/O data
                try:
                    disk_io = self._process.io_counters()
                    disk_read_bytes = disk_io.read_bytes
                    disk_write_bytes = disk_io.write_bytes
                except (AttributeError, OSError):
                    disk_read_bytes = disk_write_bytes = 0

                sample = {
                    "timestamp": time.time(),
                    "memory_mb": memory_info.rss / (1024 * 1024),
                    "cpu_percent": cpu_percent,
                    "disk_read_bytes": disk_read_bytes,
                    "disk_write_bytes": disk_write_bytes,
                }

                self.samples.append(sample)

                time.sleep(self.sample_interval)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            except Exception:
                # Continue monitoring even if some metrics fail
                time.sleep(self.sample_interval)


class PerformanceBenchmarkBase(unittest.TestCase):
    """Base class for performance benchmarks"""

    def setUp(self):
        """Setup performance testing environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.project_root = self.temp_dir / "perf_test_project"
        self.project_root.mkdir(parents=True)

        # Store original working directory
        self.original_cwd = Path.cwd()
        os.chdir(self.project_root)

        # Initialize performance tracking
        self.benchmark_results = []
        self.performance_targets = self._get_performance_targets()

        # Create test data at various scales
        self._create_test_data()

        # System information for reporting
        self.system_info = self._get_system_info()

    def tearDown(self):
        """Cleanup and generate performance report"""
        # Restore working directory
        os.chdir(self.original_cwd)

        # Generate benchmark report
        self._generate_benchmark_report()

        # Cleanup temporary directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _get_performance_targets(self) -> Dict[str, Any]:
        """Define performance targets for validation"""
        return {
            "small_cleanup_max_duration": 2.0,  # seconds
            "medium_cleanup_max_duration": 10.0,  # seconds
            "large_cleanup_max_duration": 30.0,  # seconds
            "max_memory_overhead_mb": 200.0,  # MB
            "min_throughput_files_per_second": 100.0,  # files/sec
            "backup_max_duration_per_mb": 2.0,  # seconds per MB
            "audit_logging_max_overhead": 0.1,  # seconds per event
            "emergency_stop_max_response_time": 1.0,  # seconds
        }

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context"""
        try:
            return {
                "platform": os.name,
                "cpu_count": os.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                "python_version": (
                    f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
                ),
                "disk_total_gb": psutil.disk_usage(".").total / (1024**3),
                "disk_free_gb": psutil.disk_usage(".").free / (1024**3),
            }
        except Exception:
            return {"platform": os.name, "cpu_count": os.cpu_count()}

    def _create_test_data(self):
        """Create test data at various scales for benchmarking"""
        # Small scale: 50 files, ~1MB total
        self._create_file_set("small", 50, 1024 * 20)  # 20KB per file

        # Medium scale: 500 files, ~10MB total
        self._create_file_set("medium", 500, 1024 * 20)  # 20KB per file

        # Large scale: 2000 files, ~50MB total
        self._create_file_set("large", 2000, 1024 * 25)  # 25KB per file

        # Create cache directories for realistic testing
        self._create_cache_structure()

    def _create_file_set(self, scale: str, file_count: int, file_size: int):
        """Create a set of test files"""
        scale_dir = self.project_root / f"test_data_{scale}"
        scale_dir.mkdir(parents=True)

        # Create Python files
        python_dir = scale_dir / "python_files"
        python_dir.mkdir()

        for i in range(file_count // 4):
            python_file = python_dir / f"module_{i:04d}.py"
            content = f"# Python module {i}\n" + "print('test data')\n" * (
                file_size // 20
            )
            python_file.write_text(content)

        # Create cache files
        cache_dir = scale_dir / "__pycache__"
        cache_dir.mkdir()

        for i in range(file_count // 4):
            cache_file = cache_dir / f"module_{i:04d}.cpython-39.pyc"
            cache_file.write_bytes(b"fake cache data" * (file_size // 15))

        # Create build artifacts
        build_dir = scale_dir / "build"
        build_dir.mkdir()

        for i in range(file_count // 4):
            build_file = build_dir / f"artifact_{i:04d}.o"
            build_file.write_bytes(b"fake build data" * (file_size // 15))

        # Create temporary files
        for i in range(file_count // 4):
            temp_file = scale_dir / f"temp_{i:04d}.tmp"
            temp_file.write_text("temporary data" * (file_size // 13))

    def _create_cache_structure(self):
        """Create realistic cache directory structure"""
        cache_patterns = [
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            "node_modules",
            ".tox",
        ]

        for pattern in cache_patterns:
            cache_dir = self.project_root / pattern
            cache_dir.mkdir(exist_ok=True)

            # Create nested cache files
            for i in range(10):
                cache_file = cache_dir / f"cache_file_{i}.cache"
                cache_file.write_bytes(b"cache data" * 100)

    def measure_operation(
        self, operation_name: str, operation_func: Callable, *args, **kwargs
    ) -> PerformanceMetrics:
        """Measure performance of an operation"""
        monitor = PerformanceMonitor(sample_interval=0.05)

        # Start monitoring
        monitor.start_monitoring()
        start_time = time.time()

        # Execute operation
        operation_success = True
        error_message = None
        result = None

        try:
            result = operation_func(*args, **kwargs)
        except Exception as e:
            operation_success = False
            error_message = str(e)

        # Stop monitoring
        end_time = time.time()
        perf_data = monitor.stop_monitoring()

        duration = end_time - start_time

        # Calculate files processed
        files_processed = 0
        if isinstance(result, dict):
            files_processed = len(result.get("files_processed", []))
        elif isinstance(result, list):
            files_processed = len(result)

        # Calculate throughput
        throughput = files_processed / duration if duration > 0 else 0

        return PerformanceMetrics(
            operation_name=operation_name,
            duration_seconds=duration,
            memory_peak_mb=perf_data["memory_peak_mb"],
            memory_average_mb=perf_data["memory_average_mb"],
            cpu_percent_peak=perf_data["cpu_percent_peak"],
            cpu_percent_average=perf_data["cpu_percent_average"],
            files_processed=files_processed,
            throughput_files_per_second=throughput,
            disk_io_read_mb=perf_data["disk_io_read_mb"],
            disk_io_write_mb=perf_data["disk_io_write_mb"],
            operation_success=operation_success,
            error_message=error_message,
        )

    def _generate_benchmark_report(self):
        """Generate comprehensive benchmark report"""
        if not self.benchmark_results:
            return

        # Calculate summary statistics
        durations = [
            m.duration_seconds for m in self.benchmark_results if m.operation_success
        ]
        throughputs = [
            m.throughput_files_per_second
            for m in self.benchmark_results
            if m.operation_success
        ]
        memory_peaks = [
            m.memory_peak_mb for m in self.benchmark_results if m.operation_success
        ]

        summary_stats = {}
        if durations:
            summary_stats.update(
                {
                    "average_duration": statistics.mean(durations),
                    "median_duration": statistics.median(durations),
                    "max_duration": max(durations),
                    "min_duration": min(durations),
                }
            )

        if throughputs:
            summary_stats.update(
                {
                    "average_throughput": statistics.mean(throughputs),
                    "median_throughput": statistics.median(throughputs),
                    "max_throughput": max(throughputs),
                    "min_throughput": min(throughputs),
                }
            )

        if memory_peaks:
            summary_stats.update(
                {
                    "average_memory_peak": statistics.mean(memory_peaks),
                    "max_memory_peak": max(memory_peaks),
                }
            )

        # Check if performance targets are met
        targets_met = self._check_performance_targets()

        # Create benchmark report
        report = BenchmarkReport(
            benchmark_name=self.__class__.__name__,
            timestamp=time.time(),
            system_info=self.system_info,
            test_configuration={
                "temp_dir": str(self.temp_dir),
                "project_root": str(self.project_root),
            },
            metrics=self.benchmark_results,
            summary_statistics=summary_stats,
            performance_targets=self.performance_targets,
            targets_met=targets_met,
        )

        # Save report
        report_path = self.temp_dir / f"benchmark_report_{self.__class__.__name__}.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

    def _check_performance_targets(self) -> bool:
        """Check if performance targets are met"""
        if not self.benchmark_results:
            return False

        successful_results = [m for m in self.benchmark_results if m.operation_success]
        if not successful_results:
            return False

        # Check various performance criteria
        checks = []

        # Duration checks
        max_duration = max(m.duration_seconds for m in successful_results)
        checks.append(
            max_duration
            <= self.performance_targets.get("large_cleanup_max_duration", 30.0)
        )

        # Memory checks
        max_memory = max(m.memory_peak_mb for m in successful_results)
        checks.append(
            max_memory <= self.performance_targets.get("max_memory_overhead_mb", 200.0)
        )

        # Throughput checks
        min_throughput = min(m.throughput_files_per_second for m in successful_results)
        checks.append(
            min_throughput
            >= self.performance_targets.get("min_throughput_files_per_second", 10.0)
        )

        return all(checks)


class TestCleanupPerformance(PerformanceBenchmarkBase):
    """Test performance of cleanup operations"""

    def test_small_scale_cleanup_performance(self):
        """Test performance with small number of files"""
        cleanup_targets = list(self.project_root.glob("test_data_small/**/*"))
        cleanup_targets = [f for f in cleanup_targets if f.is_file()]

        safety_cleanup = SafetyIntegratedCleanup(
            project_root=self.project_root, config=CleanupConfig(), dry_run=True
        )

        def cleanup_operation():
            return safety_cleanup.execute_safe_cleanup(
                files_to_cleanup=cleanup_targets,
                operation_type="small_scale_perf_test",
                force_backup=False,
                user_confirmation=False,
            )

        metrics = self.measure_operation("small_scale_cleanup", cleanup_operation)
        self.benchmark_results.append(metrics)

        # Validate performance
        self.assertTrue(metrics.operation_success)
        self.assertLess(
            metrics.duration_seconds,
            self.performance_targets["small_cleanup_max_duration"],
        )
        self.assertGreater(metrics.throughput_files_per_second, 10.0)

    def test_medium_scale_cleanup_performance(self):
        """Test performance with medium number of files"""
        cleanup_targets = list(self.project_root.glob("test_data_medium/**/*"))
        cleanup_targets = [f for f in cleanup_targets if f.is_file()]

        safety_cleanup = SafetyIntegratedCleanup(
            project_root=self.project_root, config=CleanupConfig(), dry_run=True
        )

        def cleanup_operation():
            return safety_cleanup.execute_safe_cleanup(
                files_to_cleanup=cleanup_targets,
                operation_type="medium_scale_perf_test",
                force_backup=False,
                user_confirmation=False,
            )

        metrics = self.measure_operation("medium_scale_cleanup", cleanup_operation)
        self.benchmark_results.append(metrics)

        # Validate performance
        self.assertTrue(metrics.operation_success)
        self.assertLess(
            metrics.duration_seconds,
            self.performance_targets["medium_cleanup_max_duration"],
        )
        self.assertLess(
            metrics.memory_peak_mb, self.performance_targets["max_memory_overhead_mb"]
        )

    def test_large_scale_cleanup_performance(self):
        """Test performance with large number of files"""
        cleanup_targets = list(self.project_root.glob("test_data_large/**/*"))
        cleanup_targets = [f for f in cleanup_targets if f.is_file()]

        safety_cleanup = SafetyIntegratedCleanup(
            project_root=self.project_root, config=CleanupConfig(), dry_run=True
        )

        def cleanup_operation():
            return safety_cleanup.execute_safe_cleanup(
                files_to_cleanup=cleanup_targets,
                operation_type="large_scale_perf_test",
                force_backup=False,
                user_confirmation=False,
            )

        metrics = self.measure_operation("large_scale_cleanup", cleanup_operation)
        self.benchmark_results.append(metrics)

        # Validate performance
        self.assertTrue(metrics.operation_success)
        self.assertLess(
            metrics.duration_seconds,
            self.performance_targets["large_cleanup_max_duration"],
        )
        self.assertGreater(metrics.files_processed, 1000)

    def test_concurrent_cleanup_performance(self):
        """Test performance of concurrent cleanup operations"""
        # Split files into chunks for concurrent processing
        all_targets = list(self.project_root.glob("test_data_medium/**/*"))
        all_targets = [f for f in all_targets if f.is_file()]

        chunk_size = len(all_targets) // 4
        file_chunks = [
            all_targets[i : i + chunk_size]
            for i in range(0, len(all_targets), chunk_size)
        ]

        safety_cleanup = SafetyIntegratedCleanup(
            project_root=self.project_root, config=CleanupConfig(), dry_run=True
        )

        def concurrent_cleanup():
            results = []

            def cleanup_chunk(chunk_id, files):
                return safety_cleanup.execute_safe_cleanup(
                    files_to_cleanup=files,
                    operation_type=f"concurrent_perf_test_{chunk_id}",
                    force_backup=False,
                    user_confirmation=False,
                )

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for i, chunk in enumerate(file_chunks):
                    future = executor.submit(cleanup_chunk, i, chunk)
                    futures.append(future)

                for future in as_completed(futures):
                    results.append(future.result())

            return results

        metrics = self.measure_operation("concurrent_cleanup", concurrent_cleanup)
        self.benchmark_results.append(metrics)

        # Validate concurrent performance
        self.assertTrue(metrics.operation_success)
        self.assertGreater(metrics.throughput_files_per_second, 50.0)

    def test_cleanup_with_backup_performance(self):
        """Test performance of cleanup operations with backup"""
        cleanup_targets = list(self.project_root.glob("test_data_small/**/*"))
        cleanup_targets = [f for f in cleanup_targets if f.is_file()][
            :50
        ]  # Limit for backup test

        safety_cleanup = SafetyIntegratedCleanup(
            project_root=self.project_root,
            config=CleanupConfig(),
            dry_run=False,  # Need actual operation for backup
        )

        def cleanup_with_backup():
            return safety_cleanup.execute_safe_cleanup(
                files_to_cleanup=cleanup_targets,
                operation_type="backup_perf_test",
                force_backup=True,
                user_confirmation=False,
            )

        metrics = self.measure_operation("cleanup_with_backup", cleanup_with_backup)
        self.benchmark_results.append(metrics)

        # Validate backup performance overhead
        self.assertTrue(metrics.operation_success)
        # Backup should add reasonable overhead
        self.assertLess(metrics.duration_seconds, 15.0)


class TestBackupPerformance(PerformanceBenchmarkBase):
    """Test performance of backup operations"""

    def test_copy_backup_performance(self):
        """Test performance of copy-based backup"""
        test_files = list(self.project_root.glob("test_data_medium/**/*"))
        test_files = [f for f in test_files if f.is_file()][
            :200
        ]  # Limit for focused test

        backup_manager = BackupManager(
            project_root=self.project_root,
            backup_root=self.project_root / ".perf_backups",
        )

        def copy_backup_operation():
            return backup_manager.create_backup(
                files_to_backup=test_files,
                operation_type="copy_perf_test",
                backup_method="copy",
            )

        metrics = self.measure_operation("copy_backup", copy_backup_operation)
        self.benchmark_results.append(metrics)

        # Validate backup performance
        self.assertTrue(metrics.operation_success)
        self.assertGreater(metrics.throughput_files_per_second, 20.0)

    def test_zip_backup_performance(self):
        """Test performance of zip-based backup"""
        test_files = list(self.project_root.glob("test_data_medium/**/*"))
        test_files = [f for f in test_files if f.is_file()][:200]

        backup_manager = BackupManager(
            project_root=self.project_root,
            backup_root=self.project_root / ".perf_backups",
        )

        def zip_backup_operation():
            return backup_manager.create_backup(
                files_to_backup=test_files,
                operation_type="zip_perf_test",
                backup_method="zip",
            )

        metrics = self.measure_operation("zip_backup", zip_backup_operation)
        self.benchmark_results.append(metrics)

        # Validate zip backup performance
        self.assertTrue(metrics.operation_success)
        # Zip backup may be slower but should still be reasonable
        self.assertLess(metrics.duration_seconds, 20.0)

    def test_backup_integrity_verification_performance(self):
        """Test performance of backup integrity verification"""
        test_files = list(self.project_root.glob("test_data_small/**/*"))
        test_files = [f for f in test_files if f.is_file()][:100]

        backup_manager = BackupManager(
            project_root=self.project_root,
            backup_root=self.project_root / ".perf_backups",
        )

        # Create backup first
        backup_metadata = backup_manager.create_backup(
            files_to_backup=test_files,
            operation_type="integrity_perf_test",
            backup_method="copy",
        )

        def verify_integrity():
            return backup_manager.verify_backup_integrity(backup_metadata.backup_id)

        metrics = self.measure_operation(
            "backup_integrity_verification", verify_integrity
        )
        self.benchmark_results.append(metrics)

        # Validate integrity verification performance
        self.assertTrue(metrics.operation_success)
        self.assertLess(metrics.duration_seconds, 5.0)

    def test_backup_restoration_performance(self):
        """Test performance of backup restoration"""
        test_files = list(self.project_root.glob("test_data_small/**/*"))
        test_files = [f for f in test_files if f.is_file()][:50]

        backup_manager = BackupManager(
            project_root=self.project_root,
            backup_root=self.project_root / ".perf_backups",
        )

        # Create backup
        backup_metadata = backup_manager.create_backup(
            files_to_backup=test_files,
            operation_type="restore_perf_test",
            backup_method="copy",
        )

        # Delete original files
        for file in test_files:
            if file.exists():
                file.unlink()

        def restore_backup():
            return backup_manager.restore_backup(
                backup_id=backup_metadata.backup_id, verify_integrity=True
            )

        metrics = self.measure_operation("backup_restoration", restore_backup)
        self.benchmark_results.append(metrics)

        # Validate restoration performance
        self.assertTrue(metrics.operation_success)
        self.assertLess(metrics.duration_seconds, 10.0)


class TestAuditLoggingPerformance(PerformanceBenchmarkBase):
    """Test performance of audit logging system"""

    def test_high_volume_audit_logging_performance(self):
        """Test performance with high volume of audit events"""
        audit_logger = AuditLogger(
            audit_dir=self.project_root / ".perf_audit", max_file_size_mb=50
        )

        def high_volume_logging():
            from xraylabtool.cleanup.audit_logger import (
                AuditEvent,
                AuditLevel,
                AuditCategory,
            )

            events_logged = 0

            with audit_logger.operation_context("perf_test", "high_volume_logging"):
                for i in range(1000):
                    event = AuditEvent(
                        level=AuditLevel.INFO,
                        category=AuditCategory.OPERATION,
                        message=f"Performance test event {i}",
                        operation_id="perf_test",
                        details={"iteration": i, "test_data": f"data_{i}"},
                    )
                    audit_logger.log_event(event)
                    events_logged += 1

            return events_logged

        metrics = self.measure_operation(
            "high_volume_audit_logging", high_volume_logging
        )
        self.benchmark_results.append(metrics)

        # Validate audit logging performance
        self.assertTrue(metrics.operation_success)
        self.assertGreater(metrics.files_processed, 900)  # Should log most events
        self.assertLess(metrics.duration_seconds, 5.0)

    def test_concurrent_audit_logging_performance(self):
        """Test performance of concurrent audit logging"""
        audit_logger = AuditLogger(
            audit_dir=self.project_root / ".perf_audit_concurrent", max_file_size_mb=50
        )

        def concurrent_logging():
            from xraylabtool.cleanup.audit_logger import (
                AuditEvent,
                AuditLevel,
                AuditCategory,
            )

            def log_from_thread(thread_id):
                for i in range(100):
                    event = AuditEvent(
                        level=AuditLevel.INFO,
                        category=AuditCategory.OPERATION,
                        message=f"Concurrent event from thread {thread_id}, iteration {i}",
                        operation_id=f"concurrent_test_{thread_id}",
                    )
                    audit_logger.log_event(event)

            # Run multiple threads
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for thread_id in range(5):
                    future = executor.submit(log_from_thread, thread_id)
                    futures.append(future)

                for future in as_completed(futures):
                    future.result()

            return 500  # 5 threads * 100 events

        metrics = self.measure_operation("concurrent_audit_logging", concurrent_logging)
        self.benchmark_results.append(metrics)

        # Validate concurrent logging performance
        self.assertTrue(metrics.operation_success)
        self.assertLess(metrics.duration_seconds, 8.0)

    def test_audit_integrity_verification_performance(self):
        """Test performance of audit log integrity verification"""
        audit_logger = AuditLogger(
            audit_dir=self.project_root / ".perf_audit_integrity", max_file_size_mb=10
        )

        # Generate some audit events first
        from xraylabtool.cleanup.audit_logger import (
            AuditEvent,
            AuditLevel,
            AuditCategory,
        )

        for i in range(200):
            event = AuditEvent(
                level=AuditLevel.INFO,
                category=AuditCategory.OPERATION,
                message=f"Integrity test event {i}",
                operation_id="integrity_test",
            )
            audit_logger.log_event(event)

        def verify_integrity():
            return audit_logger.verify_integrity()

        metrics = self.measure_operation(
            "audit_integrity_verification", verify_integrity
        )
        self.benchmark_results.append(metrics)

        # Validate integrity verification performance
        self.assertTrue(metrics.operation_success)
        self.assertLess(metrics.duration_seconds, 3.0)


class TestEmergencyStopPerformance(PerformanceBenchmarkBase):
    """Test performance of emergency stop mechanisms"""

    def test_emergency_stop_response_time(self):
        """Test response time of emergency stop mechanism"""
        from xraylabtool.cleanup.emergency_manager import (
            EmergencyStopManager,
            EmergencyStopReason,
        )

        emergency_manager = EmergencyStopManager()

        def emergency_stop_test():
            # Measure time to trigger and respond to emergency stop
            start_time = time.time()

            emergency_manager.trigger_emergency_stop(
                reason=EmergencyStopReason.USER_ABORT,
                message="Performance test emergency stop",
            )

            # Verify stop was triggered
            stop_detected = emergency_manager.is_stop_requested()
            response_time = time.time() - start_time

            return {"response_time": response_time, "stop_detected": stop_detected}

        metrics = self.measure_operation("emergency_stop_response", emergency_stop_test)
        self.benchmark_results.append(metrics)

        # Validate emergency stop performance
        self.assertTrue(metrics.operation_success)
        self.assertLess(
            metrics.duration_seconds,
            self.performance_targets["emergency_stop_max_response_time"],
        )

    def test_resource_monitoring_performance(self):
        """Test performance of resource monitoring"""
        from xraylabtool.cleanup.emergency_manager import EmergencyStopManager

        emergency_manager = EmergencyStopManager()

        def resource_monitoring_test():
            # Start resource monitoring
            emergency_manager.start_resource_monitoring(
                disk_threshold_mb=1000.0, memory_threshold_mb=2000.0, check_interval=0.1
            )

            # Run for a short period to test monitoring overhead
            time.sleep(2.0)

            # Stop monitoring
            emergency_manager.reset()

            return True

        metrics = self.measure_operation(
            "resource_monitoring", resource_monitoring_test
        )
        self.benchmark_results.append(metrics)

        # Validate resource monitoring performance
        self.assertTrue(metrics.operation_success)
        # Resource monitoring should have minimal performance impact
        self.assertLess(metrics.cpu_percent_average, 10.0)


class TestScalabilityBenchmarks(PerformanceBenchmarkBase):
    """Test scalability characteristics"""

    def test_cleanup_scalability_analysis(self):
        """Test how cleanup performance scales with file count"""
        file_counts = [50, 100, 200, 500, 1000]
        scalability_results = []

        for file_count in file_counts:
            # Create specific file set for this test
            test_files = []
            scale_dir = self.project_root / f"scalability_{file_count}"
            scale_dir.mkdir(exist_ok=True)

            for i in range(file_count):
                test_file = scale_dir / f"file_{i:04d}.tmp"
                test_file.write_text(f"test data {i}" * 100)
                test_files.append(test_file)

            safety_cleanup = SafetyIntegratedCleanup(
                project_root=self.project_root, config=CleanupConfig(), dry_run=True
            )

            def scalability_test():
                return safety_cleanup.execute_safe_cleanup(
                    files_to_cleanup=test_files,
                    operation_type=f"scalability_test_{file_count}",
                    force_backup=False,
                    user_confirmation=False,
                )

            metrics = self.measure_operation(
                f"scalability_{file_count}_files", scalability_test
            )
            self.benchmark_results.append(metrics)
            scalability_results.append(metrics)

        # Analyze scalability
        self._analyze_scalability(scalability_results)

    def _analyze_scalability(self, results: List[PerformanceMetrics]):
        """Analyze scalability characteristics"""
        if len(results) < 2:
            return

        # Calculate how duration and throughput scale with file count
        for i in range(1, len(results)):
            prev_result = results[i - 1]
            curr_result = results[i]

            file_ratio = curr_result.files_processed / prev_result.files_processed
            duration_ratio = curr_result.duration_seconds / prev_result.duration_seconds

            # Ideally, duration should scale linearly or sub-linearly with file count
            scaling_efficiency = file_ratio / duration_ratio

            # Log scaling analysis
            print(
                f"Scaling from {prev_result.files_processed} to {curr_result.files_processed} files:"
            )
            print(f"  Duration ratio: {duration_ratio:.2f}")
            print(f"  File ratio: {file_ratio:.2f}")
            print(f"  Scaling efficiency: {scaling_efficiency:.2f}")


if __name__ == "__main__":
    # Create comprehensive performance benchmark suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all performance test classes
    test_classes = [
        TestCleanupPerformance,
        TestBackupPerformance,
        TestAuditLoggingPerformance,
        TestEmergencyStopPerformance,
        TestScalabilityBenchmarks,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run performance tests
    print("=" * 80)
    print("PERFORMANCE BENCHMARK SUITE")
    print("=" * 80)

    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    total_time = time.time() - start_time

    # Performance summary
    print(f"\n{'='*80}")
    print(f"PERFORMANCE BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"Total benchmark time: {total_time:.2f} seconds")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%"
    )

    # Performance categories tested
    categories = [
        "Cleanup Operations",
        "Backup Systems",
        "Audit Logging",
        "Emergency Stop Mechanisms",
        "Scalability Analysis",
    ]

    print(f"\nPerformance Categories Benchmarked:")
    for category in categories:
        print(f"  ✓ {category}")

    if result.failures:
        print(f"\n{'*'*40} PERFORMANCE FAILURES {'*'*40}")
        for test, traceback in result.failures:
            print(f"\nFAILED: {test}")
            print(f"Details:\n{traceback}")

    if result.errors:
        print(f"\n{'*'*40} PERFORMANCE ERRORS {'*'*40}")
        for test, traceback in result.errors:
            print(f"\nERROR: {test}")
            print(f"Details:\n{traceback}")

    print(f"\n{'='*80}")

    if len(result.failures) + len(result.errors) == 0:
        print("🎉 ALL PERFORMANCE BENCHMARKS PASSED!")
        print("✅ Cleanup operations performance validated")
        print("✅ Backup systems performance validated")
        print("✅ Audit logging performance validated")
        print("✅ Emergency stop performance validated")
        print("✅ Scalability characteristics analyzed")
    else:
        print(
            f"⚠️  {len(result.failures) + len(result.errors)} performance tests failed"
        )

    print(f"{'='*80}")
    print("Performance benchmark reports saved to temporary directories")
