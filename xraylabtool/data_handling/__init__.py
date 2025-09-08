"""
XRayLabTool Data Handling Module.

This module contains data management, caching, and atomic data utilities.
"""

from xraylabtool.data_handling.atomic_cache import (
    get_atomic_data_fast,
    get_bulk_atomic_data_fast,
    get_cache_stats,
    is_element_preloaded,
    warm_up_cache,
)
from xraylabtool.data_handling.batch_processing import (
    BatchConfig,
    MemoryMonitor,
    calculate_batch_properties,
    load_batch_input,
    save_batch_results,
)

__all__ = [
    # Atomic data cache
    "get_atomic_data_fast",
    "get_bulk_atomic_data_fast",
    "warm_up_cache",
    "get_cache_stats",
    "is_element_preloaded",
    # Batch processing
    "BatchConfig",
    "MemoryMonitor",
    "calculate_batch_properties",
    "load_batch_input",
    "save_batch_results",
]
