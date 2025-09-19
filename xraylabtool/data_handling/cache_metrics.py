"""
Lightweight cache metrics tracking system for XRayLabTool.

This module provides minimal overhead cache metrics tracking that can be
optionally enabled via environment variable for performance debugging.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])

# Simplified cache metrics - disabled by default for maximum performance
_CACHE_METRICS_ENABLED = (
    os.getenv("XRAYLABTOOL_CACHE_METRICS", "false").lower() == "true"
)

# Simple lightweight counters
_cache_hits = 0
_cache_misses = 0


def cache_metrics(
    cache_name: str = "default", track_timing: bool = False
) -> Callable[[F], F]:
    """
    Lightweight cache metrics decorator with minimal overhead.

    Only tracks basic hit/miss counts when metrics are enabled via environment variable.
    Disabled by default for maximum performance.

    Args:
        cache_name: Name of the cache being tracked (unused in simplified version)
        track_timing: Whether to track timing (disabled for performance)

    Returns:
        Decorated function with optional lightweight metrics tracking
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Skip all metrics tracking if disabled (default)
            if not _CACHE_METRICS_ENABLED:
                return func(*args, **kwargs)

            # Lightweight tracking when enabled
            global _cache_hits, _cache_misses
            try:
                result = func(*args, **kwargs)
                _cache_hits += 1
                return result
            except Exception:
                _cache_misses += 1
                raise

        return wrapper

    return decorator


def get_cache_stats() -> dict[str, int]:
    """
    Get simple cache statistics.

    Returns:
        Dictionary with hit/miss counts if metrics enabled, empty dict otherwise
    """
    if not _CACHE_METRICS_ENABLED:
        return {}

    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "total": _cache_hits + _cache_misses,
        "hit_rate": (
            _cache_hits / (_cache_hits + _cache_misses)
            if (_cache_hits + _cache_misses) > 0
            else 0.0
        ),
    }


def reset_cache_stats() -> None:
    """Reset cache statistics counters."""
    global _cache_hits, _cache_misses
    _cache_hits = 0
    _cache_misses = 0


# Legacy compatibility functions (no-op for performance)
def track_element_access(*args, **kwargs) -> None:
    """Legacy function - no-op for performance."""
    pass


def track_compound_pattern(*args, **kwargs) -> None:
    """Legacy function - no-op for performance."""
    pass


def get_memory_usage() -> dict[str, int]:
    """Legacy function - returns empty dict for performance."""
    return {}


def get_usage_patterns() -> dict[str, Any]:
    """Legacy function - returns empty dict for performance."""
    return {}
