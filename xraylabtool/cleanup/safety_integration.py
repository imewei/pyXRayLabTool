"""
Safety integration module for comprehensive cleanup operations.

This module integrates all safety mechanisms into a unified system
providing comprehensive protection during cleanup operations.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional, Union, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import threading
import time

from .backup_manager import BackupManager, BackupMetadata
from .safety_validator import (
    SafetyValidator,
    ValidationReport,
    EmergencyStop,
    SafetyLevel,
    ValidationResult,
)
from .emergency_manager import (
    EmergencyStopManager,
    EmergencyStopReason,
    EmergencyContext,
)
from .audit_logger import (
    AuditLogger,
    AuditEvent,
    AuditLevel,
    AuditCategory,
    create_audit_logger,
)
from .config import CleanupConfig

logger = logging.getLogger(__name__)


@dataclass
class SafetyOperation:
    """Comprehensive safety-wrapped operation."""

    operation_id: str
    operation_type: str
    files_to_process: List[Path]
    backup_metadata: Optional[BackupMetadata] = None
    pre_validation: Optional[ValidationReport] = None
    post_validation: Optional[ValidationReport] = None
    emergency_stop: Optional[EmergencyStop] = None
    rollback_executed: bool = False
    operation_successful: bool = False


class SafetyIntegratedCleanup:
    """
    Safety-integrated cleanup system with comprehensive protection.

    This class provides a complete safety wrapper around cleanup operations,
    ensuring maximum protection through multiple layers of safety mechanisms.
    """

    def __init__(
        self,
        project_root: Union[str, Path],
        config: Optional[CleanupConfig] = None,
        dry_run: bool = True,
    ):
        """
        Initialize safety-integrated cleanup system.

        Args:
            project_root: Root directory of the project
            config: Cleanup configuration
            dry_run: Run in dry-run mode for safety
        """
        self.project_root = Path(project_root).resolve()
        self.config = config or CleanupConfig()
        self.dry_run = dry_run

        # Initialize safety components
        self.backup_manager = BackupManager(
            project_root=self.project_root,
            backup_root=self.project_root / self.config.safety.backup_directory,
            compression_enabled=True,
            max_backup_age_days=30,
        )

        self.safety_validator = SafetyValidator(
            project_root=self.project_root,
            backup_manager=self.backup_manager,
            strict_mode=self.config.safety.strict_mode,
        )

        # Initialize emergency stop manager
        self.emergency_manager = EmergencyStopManager()
        self.emergency_manager.install_signal_handlers()
        self._setup_emergency_callbacks()

        # Initialize audit logger
        self.audit_logger = create_audit_logger(
            project_root=self.project_root, audit_subdir=".cleanup_audit"
        )

        # Log system initialization
        init_event = AuditEvent(
            level=AuditLevel.INFO,
            category=AuditCategory.SYSTEM,
            message="SafetyIntegratedCleanup initialized",
            details={
                "project_root": str(self.project_root),
                "dry_run": self.dry_run,
                "strict_mode": self.config.safety.strict_mode,
            },
        )
        self.audit_logger.log_event(init_event)

        # Operation tracking
        self._active_operations: Dict[str, SafetyOperation] = {}
        self._lock = threading.Lock()

        logger.info(f"Initialized SafetyIntegratedCleanup (dry_run={dry_run})")

    def _setup_emergency_callbacks(self):
        """Setup emergency stop callbacks for cleanup and rollback"""

        def emergency_cleanup(context: EmergencyContext):
            """Emergency cleanup callback"""
            logger.warning(f"Emergency cleanup triggered: {context.reason.value}")

            # Stop any active operations
            with self._lock:
                for op_id, operation in self._active_operations.items():
                    if (
                        not operation.operation_successful
                        and not operation.rollback_executed
                    ):
                        logger.info(f"Marking operation {op_id} for emergency cleanup")

            # Perform emergency cleanup tasks
            self._emergency_cleanup_tasks(context)

        def emergency_rollback(context: EmergencyContext):
            """Emergency rollback callback"""
            logger.warning(f"Emergency rollback triggered: {context.reason.value}")

            # Perform rollback for operations in progress
            with self._lock:
                for op_id, operation in self._active_operations.items():
                    if (
                        not operation.operation_successful
                        and not operation.rollback_executed
                    ):
                        try:
                            self._emergency_rollback_operation(operation, context)
                            operation.rollback_executed = True
                            logger.info(
                                f"Emergency rollback completed for operation {op_id}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Emergency rollback failed for operation {op_id}: {e}"
                            )

        # Register callbacks with emergency manager
        self.emergency_manager.register_cleanup_callback(emergency_cleanup)
        self.emergency_manager.register_rollback_callback(emergency_rollback)

    def _emergency_cleanup_tasks(self, context: EmergencyContext):
        """Perform emergency cleanup tasks"""
        try:
            # Ensure any temporary files are cleaned up
            # This would include partial operation files, temp directories, etc.
            logger.info("Performing emergency cleanup tasks")

            # Stop any background processes
            # Clean temporary files
            # Restore file permissions if needed

        except Exception as e:
            logger.error(f"Emergency cleanup tasks failed: {e}")

    def _emergency_rollback_operation(
        self, operation: SafetyOperation, context: EmergencyContext
    ):
        """Perform emergency rollback for a specific operation"""
        if operation.backup_metadata:
            try:
                logger.info(
                    f"Attempting emergency rollback for operation {operation.operation_id}"
                )

                # Use backup manager to restore from backup
                restore_result = self.backup_manager.restore_backup(
                    operation.backup_metadata.backup_id, verify_integrity=True
                )

                if restore_result.get("success", False):
                    logger.info(
                        f"Emergency rollback successful for operation {operation.operation_id}"
                    )
                else:
                    logger.error(
                        f"Emergency rollback failed for operation {operation.operation_id}: {restore_result.get('error')}"
                    )

            except Exception as e:
                logger.error(
                    f"Emergency rollback exception for operation {operation.operation_id}: {e}"
                )
        else:
            logger.warning(
                f"No backup available for emergency rollback of operation {operation.operation_id}"
            )

    def execute_safe_cleanup(
        self,
        files_to_cleanup: List[Path],
        operation_type: str = "cleanup",
        force_backup: bool = False,
        user_confirmation: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a comprehensive safety-wrapped cleanup operation.

        Args:
            files_to_cleanup: List of files to clean up
            operation_type: Type of cleanup operation
            force_backup: Force backup creation even for low-risk operations
            user_confirmation: Require user confirmation before proceeding

        Returns:
            Dictionary with operation results and safety information
        """
        operation_id = f"{operation_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting safe cleanup operation: {operation_id}")
        logger.info(f"Files to process: {len(files_to_cleanup)}")
        logger.info(f"Dry run mode: {self.dry_run}")

        # Create safety operation tracking
        safety_op = SafetyOperation(
            operation_id=operation_id,
            operation_type=operation_type,
            files_to_process=files_to_cleanup,
        )

        with self._lock:
            self._active_operations[operation_id] = safety_op

        # Use emergency manager context for the entire operation
        operation_timeout = (
            self.config.safety.operation_timeout_seconds
            if hasattr(self.config.safety, "operation_timeout_seconds")
            else 3600
        )  # 1 hour default

        try:
            # Use audit logger context for comprehensive logging
            with self.audit_logger.operation_context(
                operation_id=operation_id, operation_type=operation_type
            ) as audit_context:

                with self.emergency_manager.operation_context(
                    operation_id=operation_id,
                    phase="initialization",
                    timeout=operation_timeout,
                    files_in_progress=files_to_cleanup,
                ):
                    # Start resource monitoring
                    self.emergency_manager.start_resource_monitoring(
                        disk_threshold_mb=1000.0,
                        memory_threshold_mb=2000.0,
                        check_interval=5.0,
                    )

                # Step 1: Pre-operation safety validation
                self.emergency_manager.set_operation_context(
                    operation_id, "pre_validation"
                )
                logger.info("🔍 Phase 1: Pre-operation safety validation")

                # Log validation start
                validation_event = AuditEvent(
                    level=AuditLevel.INFO,
                    category=AuditCategory.VALIDATION,
                    operation_id=operation_id,
                    message=f"Starting pre-operation validation for {len(files_to_cleanup)} files",
                    details={
                        "files_count": len(files_to_cleanup),
                        "operation_type": operation_type,
                    },
                )
                self.audit_logger.log_event(validation_event)

                # Check for emergency stop before validation
                if self.emergency_manager.is_stop_requested():
                    abort_event = AuditEvent(
                        level=AuditLevel.WARNING,
                        category=AuditCategory.EMERGENCY,
                        operation_id=operation_id,
                        message="Emergency stop requested during initialization",
                        success=False,
                    )
                    self.audit_logger.log_event(abort_event)
                    return self._create_abort_result(
                        safety_op, "Emergency stop requested during initialization"
                    )

                pre_validation = self.safety_validator.validate_pre_operation(
                    files_to_cleanup, operation_type
                )
                safety_op.pre_validation = pre_validation

                # Log validation results
                validation_result_event = AuditEvent(
                    level=(
                        AuditLevel.INFO
                        if pre_validation.overall_safety_level != SafetyLevel.CRITICAL
                        else AuditLevel.WARNING
                    ),
                    category=AuditCategory.VALIDATION,
                    operation_id=operation_id,
                    message=f"Pre-validation completed: {pre_validation.overall_safety_level.value}",
                    success=pre_validation.can_proceed,
                    details={
                        "safety_level": pre_validation.overall_safety_level.value,
                        "can_proceed": pre_validation.can_proceed,
                        "issues_count": len(pre_validation.required_actions),
                        "warnings_count": len(pre_validation.recommendations),
                    },
                )
                self.audit_logger.log_event(validation_result_event)

                # Check if operation should proceed based on validation
                if not self._should_proceed_with_operation(
                    pre_validation, user_confirmation
                ):
                    return self._create_abort_result(
                        safety_op, "Pre-validation failed or user declined"
                    )

                # Step 2: Create backup if needed
                self.emergency_manager.set_operation_context(
                    operation_id, "backup_creation"
                )
                backup_created = False
                if self._should_create_backup(pre_validation, force_backup):
                    logger.info("💾 Phase 2: Creating safety backup")

                    # Log backup start
                    backup_start_event = AuditEvent(
                        level=AuditLevel.INFO,
                        category=AuditCategory.BACKUP,
                        operation_id=operation_id,
                        message=f"Starting backup creation for {len(files_to_cleanup)} files",
                        details={
                            "backup_reason": "safety_backup",
                            "files_count": len(files_to_cleanup),
                        },
                    )
                    self.audit_logger.log_event(backup_start_event)

                    # Check for emergency stop before backup
                    if self.emergency_manager.is_stop_requested():
                        abort_event = AuditEvent(
                            level=AuditLevel.WARNING,
                            category=AuditCategory.EMERGENCY,
                            operation_id=operation_id,
                            message="Emergency stop requested during backup preparation",
                            success=False,
                        )
                        self.audit_logger.log_event(abort_event)
                        return self._create_abort_result(
                            safety_op,
                            "Emergency stop requested during backup preparation",
                        )

                    backup_metadata = self._create_safety_backup(
                        files_to_cleanup, operation_type
                    )
                    safety_op.backup_metadata = backup_metadata
                    backup_created = True

                    # Log backup completion
                    backup_complete_event = AuditEvent(
                        level=AuditLevel.INFO,
                        category=AuditCategory.BACKUP,
                        operation_id=operation_id,
                        message=f"Backup created successfully: {backup_metadata.backup_id}",
                        success=True,
                        details={
                            "backup_id": backup_metadata.backup_id,
                            "backup_path": str(backup_metadata.backup_path),
                            "files_backed_up": backup_metadata.files_count,
                            "backup_size_mb": (
                                backup_metadata.total_size_bytes / (1024 * 1024)
                                if backup_metadata.total_size_bytes
                                else 0
                            ),
                            "backup_method": backup_metadata.backup_method,
                        },
                    )
                    self.audit_logger.log_event(backup_complete_event)

                else:
                    logger.info("💾 Phase 2: Backup not required for this operation")

                    # Log backup skipped
                    backup_skip_event = AuditEvent(
                        level=AuditLevel.INFO,
                        category=AuditCategory.BACKUP,
                        operation_id=operation_id,
                        message="Backup skipped - not required for this operation",
                        details={
                            "reason": "low_risk_operation",
                            "force_backup": force_backup,
                        },
                    )
                    self.audit_logger.log_event(backup_skip_event)

                # Step 3: Setup emergency stop mechanism
                self.emergency_manager.set_operation_context(
                    operation_id, "emergency_setup"
                )
                logger.info("🛡️ Phase 3: Setting up emergency stop mechanism")
                emergency_stop = EmergencyStop(
                    cleanup_callback=lambda: self._emergency_cleanup(operation_id)
                )
                safety_op.emergency_stop = emergency_stop

                # Step 4: Execute cleanup operation with monitoring
                self.emergency_manager.set_operation_context(
                    operation_id, "cleanup_execution"
                )
                logger.info("🧹 Phase 4: Executing cleanup operation")

                # Check for emergency stop before cleanup
                if self.emergency_manager.is_stop_requested():
                    return self._create_abort_result(
                        safety_op, "Emergency stop requested before cleanup execution"
                    )

                cleanup_results = self._execute_monitored_cleanup(
                    files_to_cleanup, emergency_stop, operation_type
                )

            # Step 5: Post-operation validation
            logger.info("✅ Phase 5: Post-operation validation")
            post_validation = self.safety_validator.validate_post_operation(
                files_to_cleanup, operation_type
            )
            safety_op.post_validation = post_validation

            # Step 6: Check if rollback is needed
            if self._should_rollback(post_validation):
                logger.warning(
                    "🔄 Phase 6: Post-validation failed - executing rollback"
                )
                rollback_results = self._execute_rollback(safety_op)
                return self._create_rollback_result(safety_op, rollback_results)

            # Step 7: Operation successful
            safety_op.operation_successful = True
            logger.info("🎉 Phase 7: Operation completed successfully")

            return self._create_success_result(safety_op, cleanup_results)

        except Exception as e:
            logger.error(f"Safety cleanup operation failed: {e}")

            # Attempt emergency rollback if backup exists
            if safety_op.backup_metadata:
                logger.warning("Attempting emergency rollback due to operation failure")
                rollback_results = self._execute_rollback(safety_op)
                return self._create_emergency_result(
                    safety_op, str(e), rollback_results
                )

            return self._create_failure_result(safety_op, str(e))

        finally:
            # Cleanup emergency stop handlers
            if safety_op.emergency_stop:
                safety_op.emergency_stop.restore_handlers()

            # Remove from active operations
            with self._lock:
                self._active_operations.pop(operation_id, None)

    def list_active_operations(self) -> List[SafetyOperation]:
        """List all currently active safety operations."""
        with self._lock:
            return list(self._active_operations.values())

    def abort_operation(self, operation_id: str) -> bool:
        """Abort an active operation."""
        with self._lock:
            if operation_id in self._active_operations:
                safety_op = self._active_operations[operation_id]
                if safety_op.emergency_stop:
                    safety_op.emergency_stop.request_stop()
                    return True
        return False

    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety system status."""
        return {
            "active_operations": len(self._active_operations),
            "backup_manager_available": self.backup_manager is not None,
            "validator_available": self.safety_validator is not None,
            "dry_run_mode": self.dry_run,
            "strict_mode": self.config.safety.strict_mode,
            "backup_directory": str(self.backup_manager.backup_root),
            "available_backups": len(self.backup_manager.list_backups()),
        }

    def _should_proceed_with_operation(
        self, validation: ValidationReport, user_confirmation: bool
    ) -> bool:
        """Determine if operation should proceed based on validation results."""

        # Never proceed if there are critical errors
        if validation.overall_result == ValidationResult.ERROR:
            logger.error("Operation blocked: Critical validation errors detected")
            return False

        # Block critical safety level operations in strict mode
        if (
            validation.overall_safety_level == SafetyLevel.CRITICAL
            and self.config.safety.strict_mode
        ):
            logger.error("Operation blocked: Critical safety level in strict mode")
            return False

        # Require confirmation for dangerous operations
        if (
            validation.overall_safety_level == SafetyLevel.DANGEROUS
            and user_confirmation
        ):
            if not self.dry_run:
                logger.warning(
                    "Dangerous operation detected - would require user confirmation"
                )
                # In a real implementation, this would prompt the user
                return False
            else:
                logger.info("Dangerous operation in dry-run mode - proceeding")

        # Block failed validations unless it's a dry run
        if validation.overall_result == ValidationResult.FAIL:
            if not self.dry_run:
                logger.error("Operation blocked: Validation failed")
                return False
            else:
                logger.warning("Validation failed but continuing in dry-run mode")

        return True

    def _should_create_backup(
        self, validation: ValidationReport, force_backup: bool
    ) -> bool:
        """Determine if backup should be created."""

        # Always create backup if forced
        if force_backup:
            return True

        # Create backup if enabled in config
        if not self.config.safety.create_backup:
            return False

        # Always create backup for dangerous or critical operations
        if validation.overall_safety_level in [
            SafetyLevel.DANGEROUS,
            SafetyLevel.CRITICAL,
        ]:
            return True

        # Create backup for operations with failures or warnings
        if validation.overall_result in [
            ValidationResult.FAIL,
            ValidationResult.WARNING,
        ]:
            return True

        # Don't create backup for dry runs unless forced
        if self.dry_run and not force_backup:
            return False

        return True

    def _create_safety_backup(
        self, files_to_backup: List[Path], operation_type: str
    ) -> BackupMetadata:
        """Create comprehensive safety backup."""

        if self.dry_run:
            logger.info("🧪 [DRY RUN] Would create backup for safety")
            # Return a mock backup metadata for dry run
            return BackupMetadata(
                backup_id=f"dry_run_{operation_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now().isoformat(),
                operation_type=operation_type,
                total_files=len(files_to_backup),
                total_size_bytes=sum(
                    f.stat().st_size for f in files_to_backup if f.exists()
                ),
                compressed_size_bytes=0,
                checksum="dry_run_checksum",
                files=[],
                project_root=str(self.project_root),
                backup_method="dry_run",
                compression_ratio=1.0,
                integrity_verified=True,
            )

        return self.backup_manager.create_backup(
            files_to_backup=files_to_backup,
            operation_type=operation_type,
            backup_method="copy",  # Use copy method for safety
            include_git_info=True,
        )

    def _execute_monitored_cleanup(
        self,
        files_to_cleanup: List[Path],
        emergency_stop: EmergencyStop,
        operation_type: str,
    ) -> Dict[str, Any]:
        """Execute cleanup operation with emergency stop monitoring."""

        cleanup_results = {
            "processed_files": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_size_freed": 0,
            "operation_details": [],
        }

        for i, file_path in enumerate(files_to_cleanup):
            # Check for emergency stop
            if emergency_stop.is_stop_requested():
                logger.warning("Emergency stop requested during cleanup")
                break

            try:
                if file_path.exists():
                    file_size = file_path.stat().st_size

                    if self.dry_run:
                        logger.debug(f"🧪 [DRY RUN] Would remove: {file_path}")
                        cleanup_results["successful_operations"] += 1
                        cleanup_results["total_size_freed"] += file_size
                    else:
                        # Actual file removal
                        if file_path.is_dir():
                            import shutil

                            shutil.rmtree(file_path)
                        else:
                            file_path.unlink()

                        cleanup_results["successful_operations"] += 1
                        cleanup_results["total_size_freed"] += file_size
                        logger.debug(f"✅ Removed: {file_path}")

                    cleanup_results["operation_details"].append(
                        {
                            "file": str(file_path),
                            "operation": "remove",
                            "success": True,
                            "size_bytes": file_size,
                        }
                    )

                cleanup_results["processed_files"] += 1

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                cleanup_results["failed_operations"] += 1
                cleanup_results["operation_details"].append(
                    {
                        "file": str(file_path),
                        "operation": "remove",
                        "success": False,
                        "error": str(e),
                    }
                )

        return cleanup_results

    def _should_rollback(self, post_validation: ValidationReport) -> bool:
        """Determine if rollback is needed based on post-validation."""

        # Always rollback on critical errors
        if post_validation.overall_result == ValidationResult.ERROR:
            return True

        # Rollback on failures in strict mode
        if (
            post_validation.overall_result == ValidationResult.FAIL
            and self.config.safety.strict_mode
        ):
            return True

        # Rollback if critical safety level detected
        if post_validation.overall_safety_level == SafetyLevel.CRITICAL:
            return True

        return False

    def _execute_rollback(self, safety_op: SafetyOperation) -> Dict[str, Any]:
        """Execute rollback operation."""

        if not safety_op.backup_metadata:
            return {"success": False, "error": "No backup available for rollback"}

        if self.dry_run:
            logger.info("🧪 [DRY RUN] Would execute rollback from backup")
            return {"success": True, "restored_files": len(safety_op.files_to_process)}

        logger.warning(
            f"Executing rollback from backup: {safety_op.backup_metadata.backup_id}"
        )

        rollback_plan = self.safety_validator.create_rollback_plan(
            safety_op.backup_metadata, safety_op.post_validation
        )

        rollback_results = self.safety_validator.execute_rollback(
            backup_id=safety_op.backup_metadata.backup_id,
            rollback_plan=rollback_plan,
            emergency_stop=safety_op.emergency_stop,
        )

        safety_op.rollback_executed = True
        return rollback_results

    def _emergency_cleanup(self, operation_id: str) -> None:
        """Emergency cleanup callback."""
        logger.warning(f"Emergency cleanup initiated for operation: {operation_id}")

        with self._lock:
            if operation_id in self._active_operations:
                safety_op = self._active_operations[operation_id]
                # Mark operation as requiring emergency handling
                logger.warning("Operation marked for emergency handling")

    def _create_success_result(
        self, safety_op: SafetyOperation, cleanup_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create success result dictionary."""
        return {
            "success": True,
            "operation_id": safety_op.operation_id,
            "operation_type": safety_op.operation_type,
            "dry_run": self.dry_run,
            "files_processed": cleanup_results["processed_files"],
            "successful_operations": cleanup_results["successful_operations"],
            "failed_operations": cleanup_results["failed_operations"],
            "total_size_freed_mb": cleanup_results["total_size_freed"] / (1024 * 1024),
            "backup_created": safety_op.backup_metadata is not None,
            "backup_id": (
                safety_op.backup_metadata.backup_id
                if safety_op.backup_metadata
                else None
            ),
            "pre_validation_result": safety_op.pre_validation.overall_result.value,
            "post_validation_result": safety_op.post_validation.overall_result.value,
            "safety_level": safety_op.post_validation.overall_safety_level.value,
            "rollback_executed": safety_op.rollback_executed,
        }

    def _create_failure_result(
        self, safety_op: SafetyOperation, error_message: str
    ) -> Dict[str, Any]:
        """Create failure result dictionary."""
        return {
            "success": False,
            "operation_id": safety_op.operation_id,
            "operation_type": safety_op.operation_type,
            "dry_run": self.dry_run,
            "error": error_message,
            "backup_created": safety_op.backup_metadata is not None,
            "backup_id": (
                safety_op.backup_metadata.backup_id
                if safety_op.backup_metadata
                else None
            ),
            "rollback_executed": safety_op.rollback_executed,
        }

    def _create_abort_result(
        self, safety_op: SafetyOperation, abort_reason: str
    ) -> Dict[str, Any]:
        """Create abort result dictionary."""
        return {
            "success": False,
            "operation_id": safety_op.operation_id,
            "operation_type": safety_op.operation_type,
            "aborted": True,
            "abort_reason": abort_reason,
            "pre_validation_result": (
                safety_op.pre_validation.overall_result.value
                if safety_op.pre_validation
                else None
            ),
            "safety_recommendations": (
                safety_op.pre_validation.recommendations
                if safety_op.pre_validation
                else []
            ),
        }

    def _create_rollback_result(
        self, safety_op: SafetyOperation, rollback_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create rollback result dictionary."""
        return {
            "success": rollback_results.get("success", False),
            "operation_id": safety_op.operation_id,
            "operation_type": safety_op.operation_type,
            "dry_run": self.dry_run,
            "rollback_executed": True,
            "rollback_success": rollback_results.get("success", False),
            "restored_files": rollback_results.get("restored_files", 0),
            "rollback_errors": rollback_results.get("errors", []),
            "backup_id": (
                safety_op.backup_metadata.backup_id
                if safety_op.backup_metadata
                else None
            ),
        }

    def _create_emergency_result(
        self,
        safety_op: SafetyOperation,
        error_message: str,
        rollback_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create emergency result dictionary."""
        return {
            "success": False,
            "operation_id": safety_op.operation_id,
            "operation_type": safety_op.operation_type,
            "emergency": True,
            "error": error_message,
            "rollback_executed": True,
            "rollback_success": rollback_results.get("success", False),
            "restored_files": rollback_results.get("restored_files", 0),
            "emergency_recovery": "Emergency rollback attempted",
            "backup_id": (
                safety_op.backup_metadata.backup_id
                if safety_op.backup_metadata
                else None
            ),
        }
