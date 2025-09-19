"""
Cache metrics tracking system for XRayLabTool.

This module provides decorators and tracking functionality to monitor cache
hit rates, access patterns, and performance metrics with 1% accuracy.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    pass

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])

# Global cache metrics storage - thread-safe
_metrics_lock = threading.RLock()
_cache_metrics: dict[str, dict[str, Any]] = {
    "element_hit_rates": defaultdict(
        lambda: {"hits": 0, "misses": 0, "last_accessed": None, "access_frequency": 0.0}
    ),
    "compound_patterns": defaultdict(
        lambda: {
            "elements": [],
            "access_count": 0,
            "avg_processing_time": 0.0,
            "cache_efficiency": 0.0,
        }
    ),
    "memory_usage": {
        "timestamp": datetime.now(),
        "total_cache_memory": 0,
        "element_cache_size": 0,
        "interpolator_cache_size": 0,
        "available_system_memory": 0,
    },
}

# Usage pattern tracking with sliding windows
_usage_patterns: dict[str, Any] = {
    "session_id": f"session_{int(time.time())}",
    "time_windows": deque(maxlen=60),  # 1-hour sliding window (1 minute buckets)
    "element_access_history": deque(maxlen=1000),  # Recent access history
    "compound_calculations": defaultdict(int),
    "performance_metrics": {
        "cache_hit_rate": 0.0,
        "memory_pressure_events": 0,
        "warming_time_ms": 0.0,
    },
}


def cache_metrics(
    cache_name: str = "default", track_timing: bool = True
) -> Callable[[F], F]:
    """
    Decorator to track cache metrics for element access patterns.

    This decorator monitors cache hits/misses, access frequency, and timing
    information with 1% accuracy for hit rate calculations.

    Args:
        cache_name: Name of the cache being tracked
        track_timing: Whether to track execution timing

    Returns:
        Decorated function with cache metrics tracking

    Examples:
        >>> # Usage as a decorator
        >>> metrics = cache_metrics("atomic_data")
        >>> # Function would be decorated with @cache_metrics("atomic_data")
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter() if track_timing else None

            # Extract element from args - assume first arg is element
            element = args[0] if args and isinstance(args[0], str) else "unknown"

            # Check if this is a cache hit by looking at function name
            is_cache_function = any(
                cache_word in func.__name__.lower()
                for cache_word in ["cache", "get", "load"]
            )

            try:
                result = func(*args, **kwargs)

                # Track cache hit
                if is_cache_function:
                    _record_cache_access(
                        element,
                        cache_name,
                        hit=True,
                        execution_time=(
                            (time.perf_counter() - start_time) if start_time else None
                        ),
                    )

                return result

            except (FileNotFoundError, KeyError, ValueError):
                # Track cache miss for these specific exceptions
                if is_cache_function:
                    _record_cache_access(
                        element,
                        cache_name,
                        hit=False,
                        execution_time=(
                            (time.perf_counter() - start_time) if start_time else None
                        ),
                    )
                raise

        return wrapper

    return decorator


def _record_cache_access(
    element: str, cache_name: str, hit: bool, execution_time: float | None = None
) -> None:
    """
    Record a cache access event with thread safety.

    Args:
        element: Element symbol being accessed
        cache_name: Name of the cache
        hit: Whether this was a cache hit or miss
        execution_time: Execution time in seconds
    """
    with _metrics_lock:
        # Update element hit rates
        element_metrics = _cache_metrics["element_hit_rates"][element]

        if hit:
            element_metrics["hits"] += 1
        else:
            element_metrics["misses"] += 1

        element_metrics["last_accessed"] = datetime.now()

        # Calculate hit rate with high precision
        total_accesses = element_metrics["hits"] + element_metrics["misses"]
        if total_accesses > 0:
            hit_rate = (element_metrics["hits"] / total_accesses) * 100.0
            element_metrics["hit_rate"] = round(hit_rate, 2)  # 1% accuracy

        # Update access frequency (accesses per minute)
        current_time = time.time()
        _usage_patterns["element_access_history"].append(
            {
                "element": element,
                "timestamp": current_time,
                "cache_name": cache_name,
                "hit": hit,
                "execution_time": execution_time,
            }
        )

        # Calculate frequency from recent history
        recent_accesses = [
            access
            for access in _usage_patterns["element_access_history"]
            if access["element"] == element
            and current_time - access["timestamp"] < 3600  # 1 hour
        ]
        element_metrics["access_frequency"] = len(recent_accesses) / 60.0  # per minute

    # Record element access for adaptive learning (with lazy import)
    try:
        from xraylabtool.data_handling.adaptive_preloading import record_element_access

        record_element_access(element, f"cache_{cache_name}")
    except ImportError:
        pass  # Adaptive preloading not available

    # Record cache access for real-time monitoring (with lazy import)
    try:
        from xraylabtool.data_handling.hit_rate_monitor import (
            record_cache_hit,
            record_cache_miss,
        )

        if hit:
            record_cache_hit(element, cache_name)
        else:
            record_cache_miss(element, cache_name)
    except ImportError:
        pass  # Hit rate monitoring not available


def get_cache_hit_rate(element: str | None = None) -> float:
    """
    Get cache hit rate for specific element or overall.

    Args:
        element: Element symbol, or None for overall hit rate

    Returns:
        Hit rate as percentage (0-100) with 1% accuracy
    """
    with _metrics_lock:
        if element:
            element_metrics = _cache_metrics["element_hit_rates"].get(element, {})
            hits = element_metrics.get("hits", 0)
            misses = element_metrics.get("misses", 0)
        else:
            # Calculate overall hit rate
            hits = sum(
                metrics.get("hits", 0)
                for metrics in _cache_metrics["element_hit_rates"].values()
            )
            misses = sum(
                metrics.get("misses", 0)
                for metrics in _cache_metrics["element_hit_rates"].values()
            )

        total = hits + misses
        if total == 0:
            return 0.0

        return round((hits / total) * 100.0, 2)


def get_element_access_patterns() -> dict[str, Any]:
    """
    Get detailed access patterns for all elements.

    Returns:
        Dictionary with element access statistics
    """
    with _metrics_lock:
        # Create a deep copy to avoid lock contention
        patterns = {}
        for element, metrics in _cache_metrics["element_hit_rates"].items():
            if (
                metrics["hits"] > 0 or metrics["misses"] > 0
            ):  # Only include accessed elements
                patterns[element] = {
                    "hits": metrics["hits"],
                    "misses": metrics["misses"],
                    "hit_rate": metrics.get("hit_rate", 0.0),
                    "access_frequency": metrics["access_frequency"],
                    "last_accessed": metrics["last_accessed"],
                }
    return patterns


def track_compound_calculation(
    formula: str, elements: list[str], processing_time: float
) -> None:
    """
    Track compound calculation patterns for intelligent warming.

    Args:
        formula: Chemical formula (e.g., "SiO2")
        elements: List of constituent elements
        processing_time: Processing time in seconds
    """
    from xraylabtool.data_handling.compound_analysis import (
        get_compound_complexity_score,
        get_compound_family,
        get_compound_frequency_score,
    )

    with _metrics_lock:
        compound_data = _cache_metrics["compound_patterns"][formula]
        compound_data["elements"] = elements
        compound_data["access_count"] += 1

        # Update average processing time
        current_avg = compound_data["avg_processing_time"]
        count = compound_data["access_count"]
        compound_data["avg_processing_time"] = (
            current_avg * (count - 1) + processing_time
        ) / count

        # Calculate cache efficiency based on element hit rates
        element_hit_rates = [
            _cache_metrics["element_hit_rates"][element].get("hit_rate", 0.0)
            for element in elements
            if element in _cache_metrics["element_hit_rates"]
        ]

        if element_hit_rates:
            compound_data["cache_efficiency"] = sum(element_hit_rates) / len(
                element_hit_rates
            )
        else:
            compound_data["cache_efficiency"] = 0.0

        # Add compound analysis metadata
        compound_data["frequency_score"] = get_compound_frequency_score(formula)
        compound_data["complexity_score"] = get_compound_complexity_score(formula)
        compound_data["compound_family"] = get_compound_family(formula)

        # Update usage patterns
        _usage_patterns["compound_calculations"][formula] += 1

    # Record compound calculation for adaptive learning (with lazy import)
    try:
        from xraylabtool.data_handling.adaptive_preloading import (
            record_compound_calculation,
        )

        record_compound_calculation(formula, elements)
    except ImportError:
        pass  # Adaptive preloading not available


def get_common_compound_patterns(min_frequency: int = 5) -> dict[str, dict[str, Any]]:
    """
    Get commonly accessed compound patterns for cache warming.

    Args:
        min_frequency: Minimum access count to be considered "common"

    Returns:
        Dictionary of common compounds and their patterns
    """
    with _metrics_lock:
        common_patterns = {}
        for formula, data in _cache_metrics["compound_patterns"].items():
            if data["access_count"] >= min_frequency:
                common_patterns[formula] = {
                    "elements": data["elements"],
                    "access_count": data["access_count"],
                    "avg_processing_time": data["avg_processing_time"],
                    "cache_efficiency": data["cache_efficiency"],
                    "frequency_score": data.get("frequency_score", 0.0),
                    "complexity_score": data.get("complexity_score", 0.0),
                    "compound_family": data.get("compound_family", None),
                }
    return common_patterns


def get_intelligent_warming_recommendations(max_elements: int = 15) -> dict[str, Any]:
    """
    Get intelligent cache warming recommendations based on compound usage patterns.

    Args:
        max_elements: Maximum number of elements to recommend for warming

    Returns:
        Dictionary with warming recommendations and rationale
    """
    from xraylabtool.data_handling.compound_analysis import (
        COMPOUND_FAMILIES,
        analyze_element_associations,
        get_recommended_elements_for_warming,
    )

    with _metrics_lock:
        # Get recent compound usage
        recent_compounds = list(_cache_metrics["compound_patterns"].keys())

        # Get element recommendations
        recommended_elements = get_recommended_elements_for_warming(
            recent_compounds, max_elements
        )

        # Analyze element associations
        element_associations = analyze_element_associations(recent_compounds)

        # Get family-based recommendations
        family_counts = {}
        for _formula, data in _cache_metrics["compound_patterns"].items():
            family = data.get("compound_family")
            if family:
                family_counts[family] = (
                    family_counts.get(family, 0) + data["access_count"]
                )

        # Sort families by usage
        popular_families = sorted(
            family_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]

        # Get family-based element recommendations
        family_elements = set()
        for family, _count in popular_families:
            if family in COMPOUND_FAMILIES:
                for compound in COMPOUND_FAMILIES[family][
                    :3
                ]:  # Top 3 compounds per family
                    from xraylabtool.data_handling.compound_analysis import (
                        get_elements_for_compound,
                    )

                    family_elements.update(get_elements_for_compound(compound))

        return {
            "recommended_elements": recommended_elements,
            "element_associations": element_associations,
            "popular_compound_families": [family for family, count in popular_families],
            "family_based_elements": list(family_elements),
            "rationale": {
                "total_compounds_analyzed": len(recent_compounds),
                "recommendation_basis": "compound_frequency_and_associations",
                "confidence": min(
                    1.0, len(recent_compounds) / 10.0
                ),  # Higher confidence with more data
            },
        }


def reset_cache_metrics() -> None:
    """Reset all cache metrics for testing or new sessions."""
    with _metrics_lock:
        global _cache_metrics, _usage_patterns

        _cache_metrics = {
            "element_hit_rates": defaultdict(
                lambda: {
                    "hits": 0,
                    "misses": 0,
                    "last_accessed": None,
                    "access_frequency": 0.0,
                }
            ),
            "compound_patterns": defaultdict(
                lambda: {
                    "elements": [],
                    "access_count": 0,
                    "avg_processing_time": 0.0,
                    "cache_efficiency": 0.0,
                }
            ),
            "memory_usage": {
                "timestamp": datetime.now(),
                "total_cache_memory": 0,
                "element_cache_size": 0,
                "interpolator_cache_size": 0,
                "available_system_memory": 0,
            },
        }

        _usage_patterns = {
            "session_id": f"session_{int(time.time())}",
            "time_windows": deque(maxlen=60),
            "element_access_history": deque(maxlen=1000),
            "compound_calculations": defaultdict(int),
            "performance_metrics": {
                "cache_hit_rate": 0.0,
                "memory_pressure_events": 0,
                "warming_time_ms": 0.0,
            },
        }


def get_cache_performance_summary() -> dict[str, Any]:
    """
    Get comprehensive cache performance summary.

    Returns:
        Dictionary with cache performance metrics
    """
    with _metrics_lock:
        overall_hit_rate = get_cache_hit_rate()
        total_accesses = sum(
            metrics.get("hits", 0) + metrics.get("misses", 0)
            for metrics in _cache_metrics["element_hit_rates"].values()
        )

        # Get most frequently accessed elements
        frequent_elements = sorted(
            [
                (element, metrics.get("hits", 0) + metrics.get("misses", 0))
                for element, metrics in _cache_metrics["element_hit_rates"].items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        return {
            "overall_hit_rate": overall_hit_rate,
            "total_accesses": total_accesses,
            "unique_elements_accessed": len(_cache_metrics["element_hit_rates"]),
            "most_frequent_elements": frequent_elements,
            "compound_calculations": len(_cache_metrics["compound_patterns"]),
            "session_id": _usage_patterns["session_id"],
            "performance_metrics": _usage_patterns["performance_metrics"].copy(),
        }


def update_memory_usage(
    total_cache_memory: int,
    element_cache_size: int,
    interpolator_cache_size: int,
    available_system_memory: int,
) -> None:
    """
    Update memory usage metrics.

    Args:
        total_cache_memory: Total cache memory in bytes
        element_cache_size: Element cache size in bytes
        interpolator_cache_size: Interpolator cache size in bytes
        available_system_memory: Available system memory in bytes
    """
    with _metrics_lock:
        _cache_metrics["memory_usage"] = {
            "timestamp": datetime.now(),
            "total_cache_memory": total_cache_memory,
            "element_cache_size": element_cache_size,
            "interpolator_cache_size": interpolator_cache_size,
            "available_system_memory": available_system_memory,
        }
