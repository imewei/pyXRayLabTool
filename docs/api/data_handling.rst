Data Handling Module
====================

The data_handling module provides atomic data caching and batch processing capabilities.

.. currentmodule:: xraylabtool.data_handling

Atomic Data Cache
-----------------

.. automodule:: xraylabtool.data_handling.atomic_cache
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   from xraylabtool.data_handling.atomic_cache import (
       get_atomic_data_fast,
       get_cache_stats,
       is_element_preloaded,
       warm_up_cache,
   )

   si = get_atomic_data_fast("Si")          # read-only mapping
   print(si["atomic_number"], si["atomic_weight"])

   print(is_element_preloaded("Si"))        # True; all 92 elements load at import
   warm_up_cache(["Si", "O", "Al"])          # build interpolators ahead of first use
   print(get_cache_stats())
   # {'preloaded_elements': 92, 'runtime_cached_elements': 0, 'total_cached_elements': 92}

Compound Analysis
-----------------

.. automodule:: xraylabtool.data_handling.compound_analysis
   :members:
   :undoc-members:
   :show-inheritance:

Drives formula-aware cache warming: groups compounds into families (silicates, oxides, ...)
and recommends which elements to preload for a set of recently used formulas.

.. code-block:: python

   from xraylabtool.data_handling.atomic_cache import warm_cache_for_compounds
   from xraylabtool.data_handling.compound_analysis import (
       get_compound_family,
       get_recommended_elements_for_warming,
   )

   get_compound_family("SiO2")                                # 'silicates'
   get_recommended_elements_for_warming(["SiO2", "Al2O3"])    # ['O', 'Si', 'Al', 'N', ...]
   warm_cache_for_compounds(["SiO2", "Al2O3"], include_similar=True)

Batch Processing
----------------

.. automodule:: xraylabtool.data_handling.batch_processing
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   from xraylabtool.data_handling.batch_processing import (
       BatchConfig,
       calculate_batch_properties,
       load_batch_input,
       save_batch_results,
   )

   formulas, densities, _ = load_batch_input("materials.csv")  # formula,density columns
   config = BatchConfig(max_workers=4, chunk_size=100, enable_progress=True)

   results = calculate_batch_properties(formulas, [5.0, 8.0, 10.0], densities, config=config)
   # dict keyed "formula@density", value XRayResult (or None on failure)

   save_batch_results(results, "results.csv")

Chunks of 8 or more materials run in a ``ProcessPoolExecutor``; smaller chunks use a
``ThreadPoolExecutor``. ``BatchConfig.memory_limit_gb`` triggers garbage collection between
chunks.

Data Sources
------------

Atomic scattering factor data is sourced from:

1. **CXRO Database**: Center for X-ray Optics, Lawrence Berkeley National Laboratory
2. **NIST Database**: National Institute of Standards and Technology
3. **Henke Tables**: Widely used X-ray optical constants

The data files are in Henke format (.nff files); XRayLabTool exposes the 0.03–30 keV range with PCHIP interpolation between tabulated values.
