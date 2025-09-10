Data Handling Module
====================

The data_handling module provides high-performance atomic data caching and batch processing capabilities.

.. currentmodule:: xraylabtool.data_handling

Atomic Data Cache
-----------------

.. automodule:: xraylabtool.data_handling.atomic_cache
   :members:
   :undoc-members:
   :show-inheritance:

High-Performance Features
~~~~~~~~~~~~~~~~~~~~~~~~~

The atomic data cache provides several performance optimizations:

1. **Preloaded Common Elements**: 92 elements are preloaded at startup
2. **LRU Caching**: Least Recently Used cache for computed scattering factors
3. **Vectorized Operations**: NumPy-based calculations for energy arrays
4. **Memory Management**: Efficient data structures and automatic cleanup

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   from xraylabtool.data_handling.atomic_cache import get_atomic_scattering_factors

   # Get scattering factors for silicon at 8 keV
   f1, f2 = get_atomic_scattering_factors("Si", 8000)

   print(f"f1 (real): {f1}")
   print(f"f2 (imaginary): {f2}")

Cache Statistics
~~~~~~~~~~~~~~~~

.. code-block:: python

   from xraylabtool.data_handling.atomic_cache import get_cache_info

   stats = get_cache_info()
   print(f"Cache hits: {stats['hits']}")
   print(f"Cache misses: {stats['misses']}")
   print(f"Cache size: {stats['current_size']}")

Batch Processing
----------------

.. automodule:: xraylabtool.data_handling.batch_processing
   :members:
   :undoc-members:
   :show-inheritance:

Batch Processing Features
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Memory Management**: Automatic chunking for large datasets
- **Progress Tracking**: Built-in progress bars with tqdm
- **Error Handling**: Robust error recovery and reporting
- **Parallel Processing**: Multi-core support for independent calculations

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   from xraylabtool.data_handling.batch_processing import process_batch

   materials = [
       {"formula": "Si", "density": 2.33},
       {"formula": "Al", "density": 2.70},
       {"formula": "Cu", "density": 8.96}
   ]

   energies = [5000, 8000, 10000, 12000]

   results = process_batch(materials, energies, show_progress=True)

Performance Benchmarks
-----------------------

Typical performance characteristics:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Operation
     - Cold Cache
     - Warm Cache
   * - Single element lookup
     - ~0.5 ms
     - ~0.05 ms
   * - Complex formula (SiO₂)
     - ~1.2 ms
     - ~0.1 ms
   * - Batch 1000 materials
     - ~50 ms
     - ~8 ms
   * - Energy array (100 points)
     - ~15 ms
     - ~1.5 ms

Memory Usage
~~~~~~~~~~~~

The atomic cache uses approximately:

- **Startup**: ~10 MB for preloaded elements
- **Per element**: ~50 KB for full energy range
- **Peak usage**: Scales with number of unique elements used

Cache Management
----------------

.. code-block:: python

   from xraylabtool.data_handling.atomic_cache import clear_cache, preload_elements

   # Clear all cached data
   clear_cache()

   # Preload specific elements for better performance
   preload_elements(["Si", "O", "Al", "Fe"])

Data Sources
------------

Atomic scattering factor data is sourced from:

1. **CXRO Database**: Center for X-ray Optics, Lawrence Berkeley National Laboratory
2. **NIST Database**: National Institute of Standards and Technology
3. **Henke Tables**: Widely used X-ray optical constants

The data files are in Henke format (.nff files) and cover the energy range from ~10 eV to ~100 keV with high precision interpolation between tabulated values.
