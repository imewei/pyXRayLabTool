"""
XRayLabTool Data Handling Module.

This module contains data management, caching, and atomic data utilities.
"""


# Lazy import heavy modules to improve startup time
def __getattr__(name):
    if name in [
        "BatchConfig",
        "MemoryMonitor",
        "calculate_batch_properties",
        "load_batch_input",
        "save_batch_results",
    ]:
        from xraylabtool.data_handling.batch_processing import (
            BatchConfig,
            MemoryMonitor,
            calculate_batch_properties,
            load_batch_input,
            save_batch_results,
        )

        globals().update(
            {
                "BatchConfig": BatchConfig,
                "MemoryMonitor": MemoryMonitor,
                "calculate_batch_properties": calculate_batch_properties,
                "load_batch_input": load_batch_input,
                "save_batch_results": save_batch_results,
            }
        )
        return globals()[name]
    elif name in [
        "get_atomic_data_fast",
        "get_bulk_atomic_data_fast",
        "get_cache_stats",
        "is_element_preloaded",
        "warm_up_cache",
    ]:
        from xraylabtool.data_handling.atomic_cache import (
            get_atomic_data_fast,
            get_bulk_atomic_data_fast,
            get_cache_stats,
            is_element_preloaded,
            warm_up_cache,
        )

        globals().update(
            {
                "get_atomic_data_fast": get_atomic_data_fast,
                "get_bulk_atomic_data_fast": get_bulk_atomic_data_fast,
                "get_cache_stats": get_cache_stats,
                "is_element_preloaded": is_element_preloaded,
                "warm_up_cache": warm_up_cache,
            }
        )
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Batch processing
    "BatchConfig",
    "MemoryMonitor",
    "calculate_batch_properties",
    # Atomic data cache
    "get_atomic_data_fast",
    "get_bulk_atomic_data_fast",
    "get_cache_stats",
    "is_element_preloaded",
    "load_batch_input",
    "save_batch_results",
    "warm_up_cache",
]
