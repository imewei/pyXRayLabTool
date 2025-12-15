"""
Detailed memory usage profiling tools for XRayLabTool optimization.

This module provides comprehensive memory profiling capabilities to identify
memory allocation patterns, track memory usage over time, and detect potential
memory leaks in scientific computing workloads.
"""

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
import gc
import threading
import time
import tracemalloc
from typing import TYPE_CHECKING, Any

import numpy as np
import psutil

if TYPE_CHECKING:
    from collections.abc import Generator

# Global profiling state - disabled by default for performance
import os

_profiling_active = os.getenv("XRAYLABTOOL_MEMORY_PROFILING", "false").lower() == "true"
_profiling_lock = None
_memory_snapshots = None
_allocation_tracking = None


def _ensure_profiling_structures():
    """Lazy initialization of profiling data structures."""
    global _profiling_lock, _memory_snapshots, _allocation_tracking
    if _profiling_lock is None:
        _profiling_lock = threading.RLock()
        _memory_snapshots = deque(maxlen=1000)
        _allocation_tracking = defaultdict(list)


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a specific point in time."""

    timestamp: float
    rss_mb: float
    vms_mb: float
    percent: float
    python_memory_mb: float
    numpy_arrays_mb: float
    gc_objects: int
    malloc_peak_mb: float = 0.0
    malloc_current_mb: float = 0.0
    context: str = ""


@dataclass
class AllocationProfile:
    """Profile of memory allocations during a specific operation."""

    operation_name: str
    start_memory_mb: float
    peak_memory_mb: float
    end_memory_mb: float
    duration_seconds: float
    allocations_count: int
    total_allocated_mb: float
    numpy_arrays_created: int
    numpy_memory_mb: float
    gc_collections: int
    context_data: dict[str, Any] = field(default_factory=dict)


class MemoryProfiler:
    """Comprehensive memory profiler for scientific computing workloads."""

    def __init__(self, enable_tracemalloc: bool = True):
        """
        Initialize memory profiler.

        Args:
            enable_tracemalloc: Whether to enable Python tracemalloc for detailed tracking
        """
        self.process = psutil.Process()
        self.enable_tracemalloc = enable_tracemalloc
        self.baseline_memory = None
        self.profiles: list[AllocationProfile] = []

        if enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()

    def get_current_memory_info(self, context: str = "") -> MemorySnapshot:
        """
        Get comprehensive current memory information.

        Args:
            context: Optional context description for this snapshot

        Returns:
            MemorySnapshot with current memory usage details
        """
        # System memory info
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()

        # Python-specific memory tracking
        python_memory_mb = 0.0
        malloc_current_mb = 0.0
        malloc_peak_mb = 0.0

        if self.enable_tracemalloc and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            malloc_current_mb = current / 1024 / 1024
            malloc_peak_mb = peak / 1024 / 1024
            python_memory_mb = malloc_current_mb

        # NumPy array memory estimation
        numpy_memory_mb = self._estimate_numpy_memory()

        # Garbage collection info
        gc_objects = len(gc.get_objects())

        return MemorySnapshot(
            timestamp=time.time(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=memory_percent,
            python_memory_mb=python_memory_mb,
            numpy_arrays_mb=numpy_memory_mb,
            gc_objects=gc_objects,
            malloc_current_mb=malloc_current_mb,
            malloc_peak_mb=malloc_peak_mb,
            context=context,
        )

    def _estimate_numpy_memory(self) -> float:
        """Estimate memory usage of NumPy arrays in current process."""
        total_memory = 0.0

        # Get all objects and filter for numpy arrays
        for obj in gc.get_objects():
            if isinstance(obj, np.ndarray):
                try:
                    total_memory += obj.nbytes
                except (AttributeError, ValueError):
                    continue

        return total_memory / 1024 / 1024  # Convert to MB

    def set_baseline(self, context: str = "baseline") -> MemorySnapshot:
        """
        Set memory baseline for future comparisons.

        Args:
            context: Description of when baseline was taken

        Returns:
            MemorySnapshot of baseline memory usage
        """
        # Force garbage collection before baseline
        gc.collect()

        self.baseline_memory = self.get_current_memory_info(context)
        return self.baseline_memory

    @contextmanager
    def profile_operation(
        self, operation_name: str, **context_data
    ) -> Generator[AllocationProfile, None, None]:
        """
        Context manager to profile memory usage during an operation.

        Args:
            operation_name: Name of the operation being profiled
            **context_data: Additional context data to store with profile

        Yields:
            AllocationProfile object that will be populated during execution
        """
        # Reset tracemalloc peak if enabled
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.reset_peak()

        # Get starting conditions
        start_snapshot = self.get_current_memory_info(f"{operation_name}_start")
        start_time = time.perf_counter()
        start_gc_count = sum(gc.get_count())
        start_numpy_arrays = self._count_numpy_arrays()

        # Create profile object
        profile = AllocationProfile(
            operation_name=operation_name,
            start_memory_mb=start_snapshot.rss_mb,
            peak_memory_mb=start_snapshot.rss_mb,
            end_memory_mb=0.0,
            duration_seconds=0.0,
            allocations_count=0,
            total_allocated_mb=0.0,
            numpy_arrays_created=0,
            numpy_memory_mb=0.0,
            gc_collections=0,
            context_data=context_data,
        )

        try:
            # Monitor peak memory during execution
            peak_memory = start_snapshot.rss_mb

            def memory_monitor() -> None:
                nonlocal peak_memory
                while True:
                    try:
                        current_memory = self.process.memory_info().rss / 1024 / 1024
                        peak_memory = max(peak_memory, current_memory)
                        time.sleep(0.01)  # Check every 10ms
                    except:
                        break

            monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
            monitor_thread.start()

            yield profile

        finally:
            # Get ending conditions
            end_time = time.perf_counter()
            end_snapshot = self.get_current_memory_info(f"{operation_name}_end")
            end_gc_count = sum(gc.get_count())
            end_numpy_arrays = self._count_numpy_arrays()

            # Calculate allocations if tracemalloc enabled
            total_allocated_mb = 0.0
            allocations_count = 0
            if self.enable_tracemalloc and tracemalloc.is_tracing():
                _current, peak = tracemalloc.get_traced_memory()
                total_allocated_mb = peak / 1024 / 1024

            # Update profile
            profile.peak_memory_mb = peak_memory
            profile.end_memory_mb = end_snapshot.rss_mb
            profile.duration_seconds = end_time - start_time
            profile.allocations_count = allocations_count
            profile.total_allocated_mb = total_allocated_mb
            profile.numpy_arrays_created = end_numpy_arrays - start_numpy_arrays
            profile.numpy_memory_mb = (
                end_snapshot.numpy_arrays_mb - start_snapshot.numpy_arrays_mb
            )
            profile.gc_collections = end_gc_count - start_gc_count

            # Store profile
            self.profiles.append(profile)

    def _count_numpy_arrays(self) -> int:
        """Count current number of NumPy arrays in memory."""
        count = 0
        for obj in gc.get_objects():
            if isinstance(obj, np.ndarray):
                count += 1
        return count

    def analyze_memory_leaks(self, threshold_mb: float = 10.0) -> dict[str, Any]:
        """
        Analyze potential memory leaks from profiling data.

        Args:
            threshold_mb: Memory growth threshold to consider a potential leak

        Returns:
            Dictionary with leak analysis results
        """
        if not self.profiles:
            return {"status": "no_data", "message": "No profiling data available"}

        leak_analysis = {
            "potential_leaks": [],
            "memory_growth_operations": [],
            "numpy_array_growth": [],
            "gc_efficiency": {},
            "overall_trend": {},
        }

        # Analyze each operation for memory growth
        for profile in self.profiles:
            memory_growth = profile.end_memory_mb - profile.start_memory_mb

            if memory_growth > threshold_mb:
                leak_info = {
                    "operation": profile.operation_name,
                    "memory_growth_mb": memory_growth,
                    "peak_growth_mb": profile.peak_memory_mb - profile.start_memory_mb,
                    "numpy_arrays_created": profile.numpy_arrays_created,
                    "numpy_memory_mb": profile.numpy_memory_mb,
                    "gc_collections": profile.gc_collections,
                    "duration_seconds": profile.duration_seconds,
                }

                if profile.numpy_memory_mb > memory_growth * 0.8:
                    # Likely explained by NumPy arrays
                    leak_analysis["numpy_array_growth"].append(leak_info)
                else:
                    # Potential leak
                    leak_analysis["potential_leaks"].append(leak_info)

                leak_analysis["memory_growth_operations"].append(leak_info)

        # Analyze garbage collection efficiency
        if self.profiles:
            total_gc_collections = sum(p.gc_collections for p in self.profiles)
            total_memory_growth = sum(
                max(0, p.end_memory_mb - p.start_memory_mb) for p in self.profiles
            )

            leak_analysis["gc_efficiency"] = {
                "total_gc_collections": total_gc_collections,
                "total_memory_growth_mb": total_memory_growth,
                "gc_per_mb_growth": total_gc_collections / max(1, total_memory_growth),
            }

        # Overall memory trend
        if len(self.profiles) > 1:
            first_profile = self.profiles[0]
            last_profile = self.profiles[-1]

            leak_analysis["overall_trend"] = {
                "total_operations": len(self.profiles),
                "start_memory_mb": first_profile.start_memory_mb,
                "end_memory_mb": last_profile.end_memory_mb,
                "net_growth_mb": (
                    last_profile.end_memory_mb - first_profile.start_memory_mb
                ),
                "average_growth_per_operation": (
                    (last_profile.end_memory_mb - first_profile.start_memory_mb)
                    / len(self.profiles)
                ),
            }

        return leak_analysis

    def get_memory_efficiency_report(self) -> dict[str, Any]:
        """
        Generate comprehensive memory efficiency report.

        Returns:
            Dictionary with memory efficiency analysis
        """
        if not self.profiles:
            return {"status": "no_data", "message": "No profiling data available"}

        # Calculate statistics
        total_operations = len(self.profiles)
        total_duration = sum(p.duration_seconds for p in self.profiles)
        total_memory_allocated = sum(p.total_allocated_mb for p in self.profiles)
        total_numpy_memory = sum(p.numpy_memory_mb for p in self.profiles)

        # Memory allocation efficiency
        allocation_rates = [
            p.total_allocated_mb / p.duration_seconds
            for p in self.profiles
            if p.duration_seconds > 0
        ]

        # Peak memory usage patterns
        peak_memory_usage = [
            p.peak_memory_mb - p.start_memory_mb for p in self.profiles
        ]

        # NumPy usage patterns
        numpy_ratios = [
            p.numpy_memory_mb / max(1, p.total_allocated_mb) for p in self.profiles
        ]

        efficiency_report = {
            "summary": {
                "total_operations": total_operations,
                "total_duration_seconds": total_duration,
                "total_memory_allocated_mb": total_memory_allocated,
                "total_numpy_memory_mb": total_numpy_memory,
                "numpy_percentage": (
                    (total_numpy_memory / max(1, total_memory_allocated)) * 100
                ),
            },
            "allocation_efficiency": {
                "average_allocation_rate_mb_per_sec": (
                    np.mean(allocation_rates) if allocation_rates else 0
                ),
                "peak_allocation_rate_mb_per_sec": (
                    np.max(allocation_rates) if allocation_rates else 0
                ),
                "allocation_rate_std": (
                    np.std(allocation_rates) if allocation_rates else 0
                ),
            },
            "peak_memory_patterns": {
                "average_peak_growth_mb": (
                    np.mean(peak_memory_usage) if peak_memory_usage else 0
                ),
                "max_peak_growth_mb": (
                    np.max(peak_memory_usage) if peak_memory_usage else 0
                ),
                "peak_growth_std": (
                    np.std(peak_memory_usage) if peak_memory_usage else 0
                ),
            },
            "numpy_usage_patterns": {
                "average_numpy_ratio": np.mean(numpy_ratios) if numpy_ratios else 0,
                "numpy_dominated_operations": sum(1 for r in numpy_ratios if r > 0.8),
                "numpy_ratio_std": np.std(numpy_ratios) if numpy_ratios else 0,
            },
            "optimization_recommendations": (
                self._generate_optimization_recommendations()
            ),
        }

        return efficiency_report

    def _generate_optimization_recommendations(self) -> list[str]:
        """Generate optimization recommendations based on profiling data."""
        recommendations = []

        if not self.profiles:
            return ["Insufficient profiling data for recommendations"]

        # Analyze memory allocation patterns
        high_allocation_ops = [
            p
            for p in self.profiles
            if p.total_allocated_mb / max(0.001, p.duration_seconds) > 100
        ]

        if high_allocation_ops:
            recommendations.append(
                "High memory allocation rate detected in"
                f" {len(high_allocation_ops)} operations. Consider object pooling or"
                " pre-allocation strategies."
            )

        # Analyze peak memory usage
        high_peak_ops = [
            p for p in self.profiles if (p.peak_memory_mb - p.start_memory_mb) > 50
        ]

        if high_peak_ops:
            recommendations.append(
                f"High peak memory usage in {len(high_peak_ops)} operations. Consider"
                " streaming processing or memory-mapped arrays for large datasets."
            )

        # Analyze NumPy array creation
        high_numpy_ops = [p for p in self.profiles if p.numpy_arrays_created > 100]

        if high_numpy_ops:
            recommendations.append(
                f"High NumPy array creation in {len(high_numpy_ops)} operations. "
                "Consider array reuse or in-place operations."
            )

        # Analyze garbage collection efficiency
        low_gc_ops = [
            p
            for p in self.profiles
            if p.gc_collections == 0 and p.total_allocated_mb > 10
        ]

        if low_gc_ops:
            recommendations.append(
                f"Low garbage collection activity in {len(low_gc_ops)} memory-intensive"
                " operations. Consider explicit gc.collect() calls or object lifecycle"
                " management."
            )

        if not recommendations:
            recommendations.append(
                "Memory usage patterns appear efficient. No major optimizations needed."
            )

        return recommendations

    def export_profiling_data(self, filename: str) -> None:
        """
        Export profiling data to JSON file for analysis.

        Args:
            filename: Output filename for profiling data
        """
        import json

        export_data = {
            "baseline": self.baseline_memory.__dict__ if self.baseline_memory else None,
            "profiles": [
                {
                    "operation_name": p.operation_name,
                    "start_memory_mb": p.start_memory_mb,
                    "peak_memory_mb": p.peak_memory_mb,
                    "end_memory_mb": p.end_memory_mb,
                    "duration_seconds": p.duration_seconds,
                    "allocations_count": p.allocations_count,
                    "total_allocated_mb": p.total_allocated_mb,
                    "numpy_arrays_created": p.numpy_arrays_created,
                    "numpy_memory_mb": p.numpy_memory_mb,
                    "gc_collections": p.gc_collections,
                    "context_data": p.context_data,
                }
                for p in self.profiles
            ],
            "analysis": {
                "memory_leaks": self.analyze_memory_leaks(),
                "efficiency_report": self.get_memory_efficiency_report(),
            },
        }

        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2)


# Global profiler instance
_global_profiler: MemoryProfiler | None = None


def get_memory_profiler() -> MemoryProfiler:
    """Get or create global memory profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = MemoryProfiler()
    return _global_profiler


@contextmanager
def profile_memory(
    operation_name: str, **context
) -> Generator[AllocationProfile, None, None]:
    """
    Convenience context manager for memory profiling.

    Args:
        operation_name: Name of operation being profiled
        **context: Additional context data

    Yields:
        AllocationProfile for the operation
    """
    profiler = get_memory_profiler()
    with profiler.profile_operation(operation_name, **context) as profile:
        yield profile


def get_current_memory_usage() -> dict[str, float]:
    """
    Get current memory usage summary.

    Returns:
        Dictionary with current memory usage metrics
    """
    profiler = get_memory_profiler()
    snapshot = profiler.get_current_memory_info()

    return {
        "rss_mb": snapshot.rss_mb,
        "vms_mb": snapshot.vms_mb,
        "percent": snapshot.percent,
        "python_memory_mb": snapshot.python_memory_mb,
        "numpy_arrays_mb": snapshot.numpy_arrays_mb,
        "gc_objects": snapshot.gc_objects,
    }


def analyze_memory_efficiency() -> dict[str, Any]:
    """
    Analyze memory efficiency based on current profiling data.

    Returns:
        Dictionary with efficiency analysis and recommendations
    """
    profiler = get_memory_profiler()
    return profiler.get_memory_efficiency_report()


def reset_memory_profiling() -> None:
    """Reset global memory profiling data."""
    if _global_profiler:
        _global_profiler.profiles.clear()
        _global_profiler.baseline_memory = None
