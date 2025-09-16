# Codebase Cleanup System Guide

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Safety Mechanisms](#safety-mechanisms)
4. [API Reference](#api-reference)
5. [Configuration](#configuration)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Overview

The XRayLabTool codebase cleanup system provides enterprise-grade cleanup capabilities with comprehensive safety mechanisms, audit logging, and performance optimization. It's designed to safely and efficiently clean up build artifacts, cache files, temporary files, and other cleanup targets while protecting critical project files.

### Key Features

- **🛡️ Safety First**: Multi-layer validation with defense-in-depth architecture
- **💾 Automatic Backup**: Intelligent backup creation with integrity verification
- **🚨 Emergency Stop**: Real-time emergency stop mechanisms with signal handling
- **📊 Comprehensive Logging**: Tamper-evident audit trails with multiple output formats
- **⚡ High Performance**: Vectorized operations supporting 150,000+ files/second
- **🔧 Developer Friendly**: CLI and API interfaces with Makefile integration

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Safety Integration Layer                         │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │   Backup    │ │   Safety    │ │  Emergency  │ │    Audit    │   │
│  │  Manager    │ │ Validator   │ │   Manager   │ │   Logger    │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                       Core Cleanup Engine                          │
├─────────────────────────────────────────────────────────────────────┤
│          CLI Interface          │         API Interface            │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

The cleanup system is included with XRayLabTool:

```bash
pip install xraylabtool[cleanup]
```

### Basic Usage

#### Python API

```python
from xraylabtool.cleanup import SafetyIntegratedCleanup

# Initialize with safety defaults
cleanup = SafetyIntegratedCleanup(
    project_root=".",
    dry_run=True  # Safe default - preview changes first
)

# Execute cleanup with comprehensive protection
result = cleanup.execute_safe_cleanup(
    files_to_cleanup=["**/__pycache__", "**/*.pyc", "build/", "dist/"],
    operation_type="development_cleanup",
    force_backup=True,
    user_confirmation=True
)

print(f"Operation: {result['operation_id']}")
print(f"Files processed: {len(result.get('files_processed', []))}")
print(f"Backup created: {result.get('backup_metadata') is not None}")
```

#### Command Line (via Makefile)

```bash
# Dry run to preview changes
make clean-safe DRY_RUN=1

# Execute with backup
make clean-comprehensive BACKUP=1

# Quick cache cleanup
make clean-cache
```

### Your First Cleanup

1. **Start with a dry run** to understand what will be cleaned:

```python
cleanup = SafetyIntegratedCleanup(project_root=".", dry_run=True)
result = cleanup.execute_safe_cleanup(
    files_to_cleanup=["**/__pycache__"],
    operation_type="first_cleanup"
)
# Review result['files_would_be_processed']
```

2. **Execute with backup protection**:

```python
cleanup.dry_run = False
result = cleanup.execute_safe_cleanup(
    files_to_cleanup=["**/__pycache__"],
    operation_type="first_cleanup",
    force_backup=True
)
```

3. **Verify the results**:

```python
# Check audit logs
audit_dir = Path(".cleanup_audit")
if audit_dir.exists():
    print("Audit logs created successfully")

# Verify backup if created
if result.get('backup_metadata'):
    print(f"Backup: {result['backup_metadata'].backup_path}")
```

## Safety Mechanisms

The cleanup system implements a comprehensive 5-layer safety architecture:

### Layer 1: Pre-Operation Validation

- **System Resource Checks**: Validates available disk space and memory
- **File System Permissions**: Ensures proper access rights
- **Project Integrity**: Validates git repository state and critical files
- **Risk Assessment**: Calculates operation risk score with confidence metrics

```python
# Configure validation strictness
config = CleanupConfig()
config.safety.strict_mode = True
config.safety.require_git_clean = True
config.safety.min_disk_space_mb = 1000

cleanup = SafetyIntegratedCleanup(project_root=".", config=config)
```

### Layer 2: Backup and Restore

- **Automatic Backup Creation**: Creates backups based on risk assessment
- **Multiple Backup Methods**: Copy, ZIP, and incremental backups
- **Integrity Verification**: SHA-256 checksums for tamper detection
- **Atomic Operations**: All-or-nothing backup transactions

```python
# Force backup for critical operations
result = cleanup.execute_safe_cleanup(
    files_to_cleanup=sensitive_files,
    force_backup=True,
    backup_method="zip"  # Options: "copy", "zip", "incremental"
)

# Restore from backup if needed
if backup_metadata := result.get('backup_metadata'):
    restore_result = cleanup.backup_manager.restore_backup(
        backup_metadata.backup_id,
        verify_integrity=True
    )
```

### Layer 3: Emergency Stop

- **Signal Handling**: Graceful handling of SIGINT, SIGTERM, SIGUSR1
- **Resource Monitoring**: Automatic abort on disk/memory thresholds
- **User Abort**: Interactive abort capabilities during long operations
- **Operation Timeout**: Configurable timeouts with graceful termination

```python
# Configure emergency stop thresholds
cleanup.emergency_manager.start_resource_monitoring(
    disk_threshold_mb=500.0,    # Stop if disk space < 500MB
    memory_threshold_mb=1000.0, # Stop if available memory < 1GB
    check_interval=5.0          # Check every 5 seconds
)

# Set operation timeout
with cleanup.emergency_manager.operation_context(
    operation_id="long_cleanup",
    phase="processing",
    timeout=3600  # 1 hour timeout
):
    # Your cleanup operation here
    pass
```

### Layer 4: Real-Time Monitoring

- **Progress Tracking**: Real-time operation progress reporting
- **Performance Metrics**: Throughput, memory usage, and timing data
- **Error Detection**: Immediate error detection and handling
- **Health Checks**: Continuous system health validation

### Layer 5: Post-Operation Audit

- **Comprehensive Logging**: Every operation logged with full context
- **Tamper-Evident Trails**: Cryptographic integrity chains
- **Multiple Formats**: JSON, CSV, and human-readable logs
- **Compliance Ready**: Audit trails suitable for compliance requirements

## API Reference

### SafetyIntegratedCleanup

Primary interface for safe cleanup operations.

#### Constructor

```python
SafetyIntegratedCleanup(
    project_root: Union[str, Path],
    config: Optional[CleanupConfig] = None,
    dry_run: bool = True
)
```

**Parameters:**
- `project_root`: Root directory of the project
- `config`: Configuration object (optional)
- `dry_run`: If True, only simulate operations without making changes

#### Methods

##### execute_safe_cleanup()

```python
execute_safe_cleanup(
    files_to_cleanup: List[Path],
    operation_type: str = "cleanup",
    force_backup: bool = False,
    user_confirmation: bool = True
) -> Dict[str, Any]
```

Execute a comprehensive safety-wrapped cleanup operation.

**Parameters:**
- `files_to_cleanup`: List of files/patterns to clean up
- `operation_type`: Description of the operation type
- `force_backup`: Force backup creation regardless of risk assessment
- `user_confirmation`: Require user confirmation before proceeding

**Returns:**
- Dictionary containing operation results, backup metadata, and audit information

**Example:**
```python
result = cleanup.execute_safe_cleanup(
    files_to_cleanup=[
        Path("build/"),
        Path("dist/"),
        *Path(".").glob("**/__pycache__")
    ],
    operation_type="release_preparation",
    force_backup=True,
    user_confirmation=False  # For automated workflows
)

# Access results
operation_id = result["operation_id"]
files_processed = result.get("files_processed", [])
backup_info = result.get("backup_metadata")
dry_run_mode = result.get("dry_run", False)
```

### BackupManager

Handles backup creation, verification, and restoration.

#### Constructor

```python
BackupManager(
    project_root: Union[str, Path],
    backup_root: Union[str, Path],
    compression_enabled: bool = True,
    max_backup_age_days: int = 30
)
```

#### Key Methods

##### create_backup()

```python
create_backup(
    files_to_backup: List[Path],
    operation_type: str = "cleanup",
    backup_method: str = "copy",
    include_git_info: bool = True
) -> BackupMetadata
```

**Backup Methods:**
- `"copy"`: Direct file copying (fastest, larger size)
- `"zip"`: ZIP compression (slower, smaller size)
- `"incremental"`: Only changed files (efficient for large projects)

##### restore_backup()

```python
restore_backup(
    backup_id: str,
    verify_integrity: bool = True
) -> Dict[str, Any]
```

##### verify_backup_integrity()

```python
verify_backup_integrity(backup_id: str) -> bool
```

### EmergencyStopManager

Provides emergency stop and resource monitoring capabilities.

#### Key Methods

##### trigger_emergency_stop()

```python
trigger_emergency_stop(
    reason: EmergencyStopReason,
    message: str,
    files_in_progress: Optional[List[Path]] = None
)
```

##### start_resource_monitoring()

```python
start_resource_monitoring(
    disk_threshold_mb: float = 1000.0,
    memory_threshold_mb: float = 2000.0,
    check_interval: float = 5.0
)
```

### AuditLogger

Comprehensive audit logging with tamper-evident integrity.

#### Key Methods

##### log_event()

```python
log_event(event: AuditEvent)
```

##### log_file_operation()

```python
log_file_operation(
    operation_id: str,
    file_path: Path,
    operation: str,
    success: bool = True,
    hash_before: Optional[str] = None,
    hash_after: Optional[str] = None
)
```

## Configuration

### CleanupConfig

Comprehensive configuration for all cleanup operations.

```python
from xraylabtool.cleanup import CleanupConfig

config = CleanupConfig()

# Safety settings
config.safety.strict_mode = True
config.safety.require_git_clean = True
config.safety.min_disk_space_mb = 1000
config.safety.backup_directory = ".cleanup_backups"

# Performance settings
config.performance.parallel_workers = 4
config.performance.chunk_size = 1000
config.performance.memory_limit_mb = 2000

# Audit settings
config.audit.enable_json_logs = True
config.audit.enable_csv_logs = True
config.audit.retention_days = 365
config.audit.max_log_size_mb = 100
```

### Environment Variables

Control behavior through environment variables:

```bash
# Safety settings
export CLEANUP_SAFETY_STRICT=true
export CLEANUP_MIN_DISK_SPACE=1000

# Performance settings
export CLEANUP_PARALLEL_WORKERS=4
export CLEANUP_MEMORY_LIMIT=2000

# Audit settings
export CLEANUP_AUDIT_RETENTION_DAYS=365
export CLEANUP_ENABLE_DEBUG_LOGS=false

# Development settings
export CLEANUP_DRY_RUN=true
export CLEANUP_FORCE_BACKUP=false
```

## Advanced Usage

### Custom Cleanup Patterns

Create custom cleanup patterns for specific use cases:

```python
def create_python_cleanup_pattern():
    """Create cleanup pattern for Python projects"""
    return [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.pyd",
        ".pytest_cache/",
        ".coverage",
        "htmlcov/",
        ".tox/",
        ".mypy_cache/",
        "build/",
        "dist/",
        "*.egg-info/",
    ]

def create_node_cleanup_pattern():
    """Create cleanup pattern for Node.js projects"""
    return [
        "node_modules/",
        "npm-debug.log*",
        "yarn-debug.log*",
        "yarn-error.log*",
        ".npm",
        ".yarn-integrity",
        "coverage/",
        ".nyc_output/",
    ]

# Use custom patterns
python_files = [Path(p) for pattern in create_python_cleanup_pattern()
                for p in Path(".").glob(pattern)]

result = cleanup.execute_safe_cleanup(
    files_to_cleanup=python_files,
    operation_type="python_project_cleanup"
)
```

### Integration with CI/CD

Integrate with continuous integration pipelines:

```python
import os

def ci_safe_cleanup():
    """CI-safe cleanup with appropriate settings"""

    # Detect CI environment
    is_ci = any(var in os.environ for var in [
        'CI', 'GITHUB_ACTIONS', 'GITLAB_CI', 'JENKINS_URL'
    ])

    config = CleanupConfig()

    if is_ci:
        # CI-specific settings
        config.safety.strict_mode = False
        config.safety.user_confirmation = False
        config.audit.enable_debug_logs = True

    cleanup = SafetyIntegratedCleanup(
        project_root=".",
        config=config,
        dry_run=False  # CI can run actual cleanup
    )

    # Standard CI cleanup patterns
    ci_patterns = [
        "**/__pycache__",
        "**/*.pyc",
        ".pytest_cache/",
        "htmlcov/",
        "test-results/",
    ]

    files_to_cleanup = []
    for pattern in ci_patterns:
        files_to_cleanup.extend(Path(".").glob(pattern))

    return cleanup.execute_safe_cleanup(
        files_to_cleanup=files_to_cleanup,
        operation_type="ci_cleanup",
        force_backup=False,  # CI doesn't need backups
        user_confirmation=False
    )

# Usage in CI
if __name__ == "__main__":
    result = ci_safe_cleanup()
    print(f"CI cleanup completed: {result['operation_id']}")
```

### Performance Optimization

Optimize for large-scale operations:

```python
def optimize_for_large_projects():
    """Optimize cleanup for projects with many files"""

    config = CleanupConfig()

    # Performance optimizations
    config.performance.parallel_workers = os.cpu_count()
    config.performance.chunk_size = 5000
    config.performance.use_threading = True
    config.performance.memory_limit_mb = 4000

    # Disable expensive operations for large scale
    config.audit.enable_file_hashing = False
    config.safety.detailed_validation = False

    cleanup = SafetyIntegratedCleanup(
        project_root=".",
        config=config
    )

    # Use efficient backup method for large operations
    result = cleanup.execute_safe_cleanup(
        files_to_cleanup=get_large_file_list(),
        operation_type="large_scale_cleanup",
        force_backup=True,
        backup_method="incremental"  # Most efficient for large operations
    )

    return result
```

### Custom Validation Rules

Implement custom validation logic:

```python
from xraylabtool.cleanup import SafetyValidator, ValidationResult

class CustomSafetyValidator(SafetyValidator):
    """Custom safety validator with project-specific rules"""

    def validate_custom_rules(self, files_to_process: List[Path]) -> ValidationResult:
        """Implement custom validation rules"""

        issues = []
        warnings = []

        for file_path in files_to_process:
            # Custom rule: Never delete files with certain extensions
            forbidden_extensions = {'.config', '.key', '.pem'}
            if file_path.suffix in forbidden_extensions:
                issues.append(f"Cannot delete sensitive file: {file_path}")

            # Custom rule: Warn about large files
            if file_path.exists() and file_path.stat().st_size > 100 * 1024 * 1024:
                warnings.append(f"Large file deletion: {file_path} ({file_path.stat().st_size // 1024 // 1024}MB)")

            # Custom rule: Check for active file locks (Windows)
            if os.name == 'nt' and self._is_file_locked(file_path):
                issues.append(f"File is locked and cannot be deleted: {file_path}")

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            safety_level=SafetyLevel.HIGH_RISK if issues else SafetyLevel.MEDIUM_RISK
        )

    def _is_file_locked(self, file_path: Path) -> bool:
        """Check if file is locked (Windows-specific)"""
        try:
            if file_path.exists():
                with open(file_path, 'a'):
                    pass
                return False
        except (PermissionError, OSError):
            return True
        return False

# Use custom validator
cleanup = SafetyIntegratedCleanup(project_root=".")
cleanup.safety_validator = CustomSafetyValidator(
    project_root=cleanup.project_root,
    backup_manager=cleanup.backup_manager
)
```

## Troubleshooting

### Common Issues

#### 1. Permission Denied Errors

**Symptom:** `PermissionError` when trying to delete files

**Solutions:**
```python
# Check file permissions before cleanup
def check_permissions(files_to_cleanup):
    permission_issues = []
    for file_path in files_to_cleanup:
        if file_path.exists() and not os.access(file_path, os.W_OK):
            permission_issues.append(file_path)

    if permission_issues:
        print(f"Permission issues found for {len(permission_issues)} files")
        # Fix permissions or run with elevated privileges

# Enable detailed permission validation
config = CleanupConfig()
config.safety.validate_permissions = True
config.safety.strict_mode = True
```

#### 2. Emergency Stop Not Responding

**Symptom:** Emergency stop mechanism doesn't respond to signals

**Solutions:**
```python
# Ensure signal handlers are properly installed
cleanup.emergency_manager.install_signal_handlers()

# Test emergency stop mechanism
def test_emergency_stop():
    import signal
    import time

    # Send SIGUSR1 to trigger emergency stop
    os.kill(os.getpid(), signal.SIGUSR1)

    # Check if stop was registered
    time.sleep(0.1)
    assert cleanup.emergency_manager.is_stop_requested()

# Configure more aggressive monitoring
cleanup.emergency_manager.start_resource_monitoring(
    disk_threshold_mb=100.0,    # Lower threshold
    memory_threshold_mb=500.0,  # Lower threshold
    check_interval=1.0          # Check more frequently
)
```

#### 3. Backup Integrity Failures

**Symptom:** Backup integrity verification fails

**Solutions:**
```python
# Enable detailed backup logging
config = CleanupConfig()
config.backup.enable_integrity_logging = True
config.backup.verify_on_creation = True

# Manual integrity verification
backup_manager = cleanup.backup_manager
backup_id = "your_backup_id"

is_valid = backup_manager.verify_backup_integrity(backup_id)
if not is_valid:
    # Get detailed integrity report
    integrity_report = backup_manager.get_integrity_report(backup_id)
    print(f"Integrity issues: {integrity_report}")

    # Create new backup
    new_backup = backup_manager.create_backup(
        files_to_backup=files_list,
        operation_type="integrity_recovery",
        backup_method="copy"  # Most reliable method
    )
```

#### 4. High Memory Usage

**Symptom:** Cleanup operations consume excessive memory

**Solutions:**
```python
# Configure memory limits
config = CleanupConfig()
config.performance.memory_limit_mb = 1000  # Limit to 1GB
config.performance.chunk_size = 500        # Process smaller chunks
config.performance.parallel_workers = 2    # Reduce parallelism

# Enable memory monitoring
cleanup.emergency_manager.start_resource_monitoring(
    memory_threshold_mb=800.0,  # Stop before hitting limit
    check_interval=2.0
)

# Use streaming operations for large file sets
def process_large_fileset(large_file_list):
    chunk_size = 1000
    for i in range(0, len(large_file_list), chunk_size):
        chunk = large_file_list[i:i + chunk_size]

        result = cleanup.execute_safe_cleanup(
            files_to_cleanup=chunk,
            operation_type=f"chunk_{i // chunk_size}",
            force_backup=False  # Reduce memory usage
        )

        # Process results and free memory
        del result
```

### Debugging Tools

#### Enable Debug Logging

```python
import logging

# Enable detailed debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable cleanup-specific debug logs
cleanup_logger = logging.getLogger('xraylabtool.cleanup')
cleanup_logger.setLevel(logging.DEBUG)

# Enable audit trail debugging
os.environ['CLEANUP_ENABLE_DEBUG_LOGS'] = 'true'
```

#### Performance Profiling

```python
import cProfile
import pstats

def profile_cleanup_operation():
    """Profile cleanup operation for performance analysis"""

    profiler = cProfile.Profile()
    profiler.enable()

    # Your cleanup operation
    result = cleanup.execute_safe_cleanup(
        files_to_cleanup=your_file_list,
        operation_type="profiling_test"
    )

    profiler.disable()

    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

    return result
```

#### Audit Trail Analysis

```python
def analyze_audit_logs():
    """Analyze audit logs for debugging"""

    audit_dir = Path(".cleanup_audit")

    # Load JSON logs
    json_logs = list(audit_dir.glob("json/audit_*.json"))

    for log_file in json_logs:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    event = json.loads(line)

                    # Look for error events
                    if not event.get("success", True):
                        print(f"Error event: {event}")

                    # Look for performance issues
                    if event.get("duration_ms", 0) > 5000:  # >5 seconds
                        print(f"Slow operation: {event}")

                except json.JSONDecodeError:
                    continue
```

## Best Practices

### 1. Always Start with Dry Run

```python
# GOOD: Always preview changes first
cleanup = SafetyIntegratedCleanup(project_root=".", dry_run=True)
result = cleanup.execute_safe_cleanup(files_to_cleanup=files)

# Review the results
if result["dry_run"]:
    print(f"Would process {len(result['files_would_be_processed'])} files")
    # Only proceed after review
    cleanup.dry_run = False
    actual_result = cleanup.execute_safe_cleanup(files_to_cleanup=files)
```

### 2. Use Appropriate Safety Levels

```python
# Development cleanup - high safety
config = CleanupConfig()
config.safety.strict_mode = True
config.safety.require_backup = True

# CI cleanup - balanced safety
config.safety.strict_mode = False
config.safety.require_backup = False

# Production cleanup - maximum safety
config.safety.strict_mode = True
config.safety.require_backup = True
config.safety.require_git_clean = True
config.safety.user_confirmation = True
```

### 3. Implement Proper Error Handling

```python
try:
    result = cleanup.execute_safe_cleanup(
        files_to_cleanup=files_to_cleanup,
        operation_type="safe_cleanup"
    )

    if not result.get("success", True):
        logger.error(f"Cleanup failed: {result.get('error_message')}")

        # Attempt recovery if backup exists
        if backup_metadata := result.get("backup_metadata"):
            logger.info("Attempting recovery from backup...")
            recovery_result = cleanup.backup_manager.restore_backup(
                backup_metadata.backup_id
            )

            if recovery_result.get("success"):
                logger.info("Recovery successful")
            else:
                logger.error("Recovery failed - manual intervention required")

except Exception as e:
    logger.error(f"Cleanup operation failed: {e}")

    # Check for emergency stop
    emergency_report = cleanup.emergency_manager.get_emergency_report()
    if emergency_report.get("status") == "emergency_stop":
        logger.warning(f"Emergency stop triggered: {emergency_report['reason']}")
```

### 4. Regular Maintenance

```python
def perform_maintenance():
    """Regular maintenance of cleanup system"""

    # Clean old backups
    removed_backups = cleanup.backup_manager.cleanup_old_backups()
    logger.info(f"Removed {removed_backups} old backups")

    # Clean old audit logs
    removed_logs = cleanup.audit_logger.cleanup_old_logs()
    logger.info(f"Removed {removed_logs} old log files")

    # Verify backup integrity
    all_backups = cleanup.backup_manager.list_backups()
    for backup in all_backups:
        is_valid = cleanup.backup_manager.verify_backup_integrity(backup.backup_id)
        if not is_valid:
            logger.warning(f"Backup integrity issue: {backup.backup_id}")

# Schedule maintenance
import schedule

schedule.every().week.do(perform_maintenance)
```

### 5. Monitor and Alert

```python
def setup_monitoring():
    """Setup monitoring and alerting for cleanup operations"""

    def cleanup_monitor(result):
        """Monitor cleanup operation results"""

        # Check for failures
        if not result.get("success", True):
            send_alert(f"Cleanup failed: {result.get('error_message')}")

        # Check for performance issues
        duration = result.get("duration_seconds", 0)
        if duration > 300:  # 5 minutes
            send_alert(f"Cleanup took {duration:.1f} seconds - performance issue?")

        # Check backup health
        if backup_metadata := result.get("backup_metadata"):
            if not cleanup.backup_manager.verify_backup_integrity(backup_metadata.backup_id):
                send_alert(f"Backup integrity failure: {backup_metadata.backup_id}")

    def send_alert(message):
        """Send alert (implement with your preferred method)"""
        logger.critical(f"ALERT: {message}")
        # Implement: Slack, email, PagerDuty, etc.

    return cleanup_monitor

# Use monitoring
monitor = setup_monitoring()

result = cleanup.execute_safe_cleanup(files_to_cleanup=files)
monitor(result)
```

This comprehensive guide provides everything needed to safely and effectively use the codebase cleanup system. Remember to always prioritize safety, start with dry runs, and maintain proper backups and audit trails for all cleanup operations.