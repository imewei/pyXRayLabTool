#!/usr/bin/env python3
"""
Emergency Stop Manager for Codebase Cleanup Operations

This module provides comprehensive emergency stop mechanisms that integrate
with all safety systems to ensure operations can be safely terminated at any point.
"""

import logging
import signal
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class EmergencyStopReason(Enum):
    """Emergency stop reasons"""
    USER_ABORT = "user_abort"
    KEYBOARD_INTERRUPT = "keyboard_interrupt"
    SIGNAL_TERMINATION = "signal_termination"
    RESOURCE_THRESHOLD = "resource_threshold"
    OPERATION_TIMEOUT = "operation_timeout"
    CRITICAL_ERROR = "critical_error"
    SAFETY_VIOLATION = "safety_violation"
    SYSTEM_SHUTDOWN = "system_shutdown"


@dataclass
class EmergencyContext:
    """Emergency stop context information"""
    reason: EmergencyStopReason
    message: str
    timestamp: float
    operation_id: str
    current_phase: str
    cleanup_required: bool = True
    rollback_required: bool = True
    backup_available: bool = False
    files_in_progress: List[Path] = None

    def __post_init__(self):
        if self.files_in_progress is None:
            self.files_in_progress = []


class EmergencyStopManager:
    """
    Comprehensive emergency stop manager that coordinates with all safety systems.

    Provides multiple mechanisms for safely terminating operations:
    - Signal handling (SIGINT, SIGTERM)
    - Resource monitoring with automatic thresholds
    - User-initiated abort through interactive prompts
    - Operation timeout handling
    - Critical error response
    """

    def __init__(self):
        self.stop_requested = False
        self.emergency_context: Optional[EmergencyContext] = None
        self.signal_handlers_installed = False
        self.cleanup_callbacks: List[Callable] = []
        self.rollback_callbacks: List[Callable] = []
        self.resource_monitor: Optional[ResourceMonitor] = None
        self.operation_timeout: Optional[float] = None
        self.operation_start_time: Optional[float] = None
        self.current_operation_id: str = ""
        self.current_phase: str = ""
        self.abort_lock = threading.Lock()

        logger.debug("EmergencyStopManager initialized")

    def install_signal_handlers(self):
        """Install signal handlers for graceful shutdown"""
        if self.signal_handlers_installed:
            return

        try:
            # Install SIGINT handler (Ctrl+C)
            signal.signal(signal.SIGINT, self._signal_handler)

            # Install SIGTERM handler (process termination)
            signal.signal(signal.SIGTERM, self._signal_handler)

            # Install SIGUSR1 for emergency stop (Unix only)
            if hasattr(signal, 'SIGUSR1'):
                signal.signal(signal.SIGUSR1, self._emergency_signal_handler)

            self.signal_handlers_installed = True
            logger.info("Emergency stop signal handlers installed")

        except Exception as e:
            logger.warning(f"Failed to install signal handlers: {e}")

    def _signal_handler(self, signum: int, frame):
        """Handle termination signals"""
        reason = EmergencyStopReason.KEYBOARD_INTERRUPT if signum == signal.SIGINT else EmergencyStopReason.SIGNAL_TERMINATION
        message = f"Received signal {signum}"

        logger.warning(f"Emergency stop triggered by signal {signum}")
        self.trigger_emergency_stop(reason, message)

    def _emergency_signal_handler(self, signum: int, frame):
        """Handle emergency signals (SIGUSR1)"""
        logger.critical(f"Emergency signal {signum} received - immediate stop")
        self.trigger_emergency_stop(EmergencyStopReason.SIGNAL_TERMINATION,
                                   f"Emergency signal {signum}")

    def register_cleanup_callback(self, callback: Callable):
        """Register callback for cleanup operations during emergency stop"""
        self.cleanup_callbacks.append(callback)
        logger.debug(f"Registered cleanup callback: {callback.__name__}")

    def register_rollback_callback(self, callback: Callable):
        """Register callback for rollback operations during emergency stop"""
        self.rollback_callbacks.append(callback)
        logger.debug(f"Registered rollback callback: {callback.__name__}")

    def set_operation_context(self, operation_id: str, phase: str, timeout: Optional[float] = None):
        """Set current operation context for emergency tracking"""
        self.current_operation_id = operation_id
        self.current_phase = phase
        self.operation_timeout = timeout
        self.operation_start_time = time.time()

        logger.debug(f"Operation context set: {operation_id} - {phase}")

    def start_resource_monitoring(self,
                                disk_threshold_mb: float = 1000.0,
                                memory_threshold_mb: float = 2000.0,
                                check_interval: float = 5.0):
        """Start resource monitoring with automatic emergency stop on threshold breach"""
        if self.resource_monitor:
            self.resource_monitor.stop()

        self.resource_monitor = ResourceMonitor(
            disk_threshold_mb=disk_threshold_mb,
            memory_threshold_mb=memory_threshold_mb,
            check_interval=check_interval,
            emergency_callback=self._resource_threshold_breach
        )

        self.resource_monitor.start()
        logger.info(f"Resource monitoring started (disk: {disk_threshold_mb}MB, memory: {memory_threshold_mb}MB)")

    def _resource_threshold_breach(self, resource_type: str, current_value: float, threshold: float):
        """Handle resource threshold breach"""
        message = f"{resource_type} threshold breached: {current_value:.1f}MB >= {threshold:.1f}MB"
        logger.critical(message)
        self.trigger_emergency_stop(EmergencyStopReason.RESOURCE_THRESHOLD, message)

    def check_timeout(self) -> bool:
        """Check if operation has timed out"""
        if not self.operation_timeout or not self.operation_start_time:
            return False

        elapsed = time.time() - self.operation_start_time
        if elapsed > self.operation_timeout:
            message = f"Operation timeout: {elapsed:.1f}s > {self.operation_timeout:.1f}s"
            logger.error(message)
            self.trigger_emergency_stop(EmergencyStopReason.OPERATION_TIMEOUT, message)
            return True

        return False

    def trigger_emergency_stop(self,
                             reason: EmergencyStopReason,
                             message: str,
                             files_in_progress: Optional[List[Path]] = None):
        """Trigger emergency stop with specified reason"""
        with self.abort_lock:
            if self.stop_requested:
                return  # Already stopping

            self.stop_requested = True
            self.emergency_context = EmergencyContext(
                reason=reason,
                message=message,
                timestamp=time.time(),
                operation_id=self.current_operation_id,
                current_phase=self.current_phase,
                files_in_progress=files_in_progress or []
            )

            logger.critical(f"EMERGENCY STOP TRIGGERED: {reason.value} - {message}")

            # Execute emergency procedures
            self._execute_emergency_procedures()

    def _execute_emergency_procedures(self):
        """Execute emergency stop procedures"""
        if not self.emergency_context:
            return

        logger.info("Executing emergency stop procedures...")

        try:
            # Stop resource monitoring
            if self.resource_monitor:
                self.resource_monitor.stop()

            # Execute cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    logger.debug(f"Executing cleanup callback: {callback.__name__}")
                    callback(self.emergency_context)
                except Exception as e:
                    logger.error(f"Cleanup callback failed: {callback.__name__}: {e}")

            # Execute rollback callbacks if needed
            if self.emergency_context.rollback_required:
                for callback in self.rollback_callbacks:
                    try:
                        logger.debug(f"Executing rollback callback: {callback.__name__}")
                        callback(self.emergency_context)
                    except Exception as e:
                        logger.error(f"Rollback callback failed: {callback.__name__}: {e}")

            logger.info("Emergency stop procedures completed")

        except Exception as e:
            logger.critical(f"Emergency procedures failed: {e}")

    def is_stop_requested(self) -> bool:
        """Check if emergency stop has been requested"""
        if self.stop_requested:
            return True

        # Check for timeout
        if self.check_timeout():
            return True

        return False

    def wait_for_user_abort(self, prompt_interval: float = 1.0) -> bool:
        """
        Non-blocking check for user abort input.
        Returns True if user wants to abort, False otherwise.
        """
        try:
            # Use select on Unix systems for non-blocking input
            import select
            import tty
            import termios

            if select.select([sys.stdin], [], [], 0.1)[0]:
                # Input is available
                old_settings = termios.tcgetattr(sys.stdin)
                try:
                    tty.cbreak(sys.stdin.fileno())
                    char = sys.stdin.read(1)
                    if char.lower() in ['q', '\x03', '\x1b']:  # 'q', Ctrl+C, ESC
                        return True
                finally:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        except (ImportError, OSError):
            # Fallback for Windows or when termios is not available
            pass

        return False

    @contextmanager
    def operation_context(self,
                         operation_id: str,
                         phase: str,
                         timeout: Optional[float] = None,
                         files_in_progress: Optional[List[Path]] = None):
        """Context manager for safe operation execution with emergency stop"""
        self.set_operation_context(operation_id, phase, timeout)

        try:
            yield self

        except KeyboardInterrupt:
            self.trigger_emergency_stop(
                EmergencyStopReason.KEYBOARD_INTERRUPT,
                "Keyboard interrupt during operation",
                files_in_progress
            )
            raise

        except Exception as e:
            if not self.stop_requested:
                self.trigger_emergency_stop(
                    EmergencyStopReason.CRITICAL_ERROR,
                    f"Critical error during operation: {e}",
                    files_in_progress
                )
            raise

        finally:
            if self.resource_monitor:
                self.resource_monitor.stop()

    def reset(self):
        """Reset emergency stop state for new operation"""
        with self.abort_lock:
            self.stop_requested = False
            self.emergency_context = None
            self.current_operation_id = ""
            self.current_phase = ""
            self.operation_start_time = None
            self.operation_timeout = None

        if self.resource_monitor:
            self.resource_monitor.stop()
            self.resource_monitor = None

        logger.debug("Emergency stop manager reset")

    def get_emergency_report(self) -> Dict[str, Any]:
        """Get detailed emergency stop report"""
        if not self.emergency_context:
            return {"status": "no_emergency"}

        return {
            "status": "emergency_stop",
            "reason": self.emergency_context.reason.value,
            "message": self.emergency_context.message,
            "timestamp": self.emergency_context.timestamp,
            "operation_id": self.emergency_context.operation_id,
            "current_phase": self.emergency_context.current_phase,
            "cleanup_required": self.emergency_context.cleanup_required,
            "rollback_required": self.emergency_context.rollback_required,
            "files_in_progress": [str(f) for f in self.emergency_context.files_in_progress]
        }


class ResourceMonitor:
    """Resource usage monitor with threshold-based emergency stop"""

    def __init__(self,
                 disk_threshold_mb: float,
                 memory_threshold_mb: float,
                 check_interval: float,
                 emergency_callback: Callable[[str, float, float], None]):
        self.disk_threshold_mb = disk_threshold_mb
        self.memory_threshold_mb = memory_threshold_mb
        self.check_interval = check_interval
        self.emergency_callback = emergency_callback
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

    def start(self):
        """Start resource monitoring"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        logger.debug("Resource monitoring started")

    def stop(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.debug("Resource monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        import psutil

        while self.monitoring:
            try:
                # Check available disk space
                disk_usage = psutil.disk_usage('.')
                available_mb = disk_usage.free / (1024 * 1024)

                if available_mb < self.disk_threshold_mb:
                    self.emergency_callback("disk_space", self.disk_threshold_mb - available_mb, self.disk_threshold_mb)
                    break

                # Check available memory
                memory = psutil.virtual_memory()
                available_mb = memory.available / (1024 * 1024)

                if available_mb < self.memory_threshold_mb:
                    self.emergency_callback("memory", self.memory_threshold_mb - available_mb, self.memory_threshold_mb)
                    break

                time.sleep(self.check_interval)

            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                time.sleep(self.check_interval)

    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        try:
            import psutil

            disk_usage = psutil.disk_usage('.')
            memory = psutil.virtual_memory()

            return {
                "disk_available_mb": disk_usage.free / (1024 * 1024),
                "disk_used_mb": disk_usage.used / (1024 * 1024),
                "memory_available_mb": memory.available / (1024 * 1024),
                "memory_used_mb": memory.used / (1024 * 1024),
                "memory_percent": memory.percent
            }

        except ImportError:
            logger.warning("psutil not available - resource monitoring disabled")
            return {}
        except Exception as e:
            logger.error(f"Failed to get resource usage: {e}")
            return {}


def create_emergency_manager() -> EmergencyStopManager:
    """Factory function to create configured emergency stop manager"""
    manager = EmergencyStopManager()
    manager.install_signal_handlers()
    return manager


# Global emergency manager instance
_global_emergency_manager: Optional[EmergencyStopManager] = None


def get_emergency_manager() -> EmergencyStopManager:
    """Get or create global emergency stop manager"""
    global _global_emergency_manager

    if _global_emergency_manager is None:
        _global_emergency_manager = create_emergency_manager()

    return _global_emergency_manager


def emergency_stop_wrapper(func):
    """Decorator to wrap functions with emergency stop protection"""
    def wrapper(*args, **kwargs):
        manager = get_emergency_manager()

        if manager.is_stop_requested():
            raise InterruptedError("Operation aborted due to emergency stop")

        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            manager.trigger_emergency_stop(
                EmergencyStopReason.KEYBOARD_INTERRUPT,
                f"Keyboard interrupt in {func.__name__}"
            )
            raise
        except Exception as e:
            if not manager.stop_requested:
                manager.trigger_emergency_stop(
                    EmergencyStopReason.CRITICAL_ERROR,
                    f"Critical error in {func.__name__}: {e}"
                )
            raise

    return wrapper


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    manager = create_emergency_manager()

    def test_cleanup(context):
        print(f"Cleanup callback executed: {context.reason}")

    def test_rollback(context):
        print(f"Rollback callback executed: {context.reason}")

    manager.register_cleanup_callback(test_cleanup)
    manager.register_rollback_callback(test_rollback)

    print("Emergency stop manager test - press Ctrl+C to test")

    try:
        with manager.operation_context("test_op", "testing", timeout=10.0):
            for i in range(100):
                if manager.is_stop_requested():
                    print("Stop requested - breaking")
                    break
                print(f"Processing {i}...")
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("Caught keyboard interrupt")

    report = manager.get_emergency_report()
    print(f"Emergency report: {report}")