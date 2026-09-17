Performance and Optimization
============================

Where the time goes
-------------------

- **Atomic data**: all 92 elements are preloaded into an in-process cache at import;
  scattering-factor interpolators are built lazily per element and memoized
  (:mod:`xraylabtool.calculators.cache`).
- **Per-call work**: formula parsing, interpolation of f1/f2 at the requested energies, and
  the vectorised δ/β/derived-quantity kernels (:mod:`xraylabtool.calculators.kernels`).
  Cost scales with the number of energy points, not the number of calls, so pass an energy
  array instead of looping.
- **Backend**: :mod:`xraylabtool.backend` selects JAX automatically when JAX and an NVIDIA
  GPU are present, otherwise NumPy. On CPU-only machines NumPy is faster for typical
  workloads because it avoids XLA dispatch overhead.

Measure before tuning:

.. code-block:: python

   import time
   import numpy as np
   import xraylabtool as xrt

   energies = np.logspace(0, np.log10(30), 500)
   xrt.calculate_single_material_properties("SiO2", energies, 2.2)  # warm-up

   start = time.perf_counter()
   xrt.calculate_single_material_properties("SiO2", energies, 2.2)
   print(f"{(time.perf_counter() - start) * 1e3:.2f} ms for {energies.size} energies")

Optimization Strategies
-----------------------

**Energy arrays, not loops**

.. code-block:: python

   # Good: one call, vectorised over 100 energies
   result = xrt.calculate_single_material_properties("Si", np.linspace(5, 15, 100), 2.33)

   # Slow: 100 calls, each re-parsing the formula and re-interpolating
   for e in np.linspace(5, 15, 100):
       xrt.calculate_single_material_properties("Si", e, 2.33)

**Many materials**

:func:`xraylabtool.calculate_xray_properties` processes materials sequentially and returns
``dict[formula, XRayResult]``:

.. code-block:: python

   results = xrt.calculate_xray_properties(["Si", "SiO2", "Al2O3"], 8.0, [2.33, 2.2, 3.95])

For large CSV-driven jobs use
:func:`xraylabtool.data_handling.batch_processing.calculate_batch_properties`, which chunks
the input (``BatchConfig.chunk_size``, default 100), runs chunks of 8 or more materials in a
``ProcessPoolExecutor`` and smaller chunks in a ``ThreadPoolExecutor``, and watches memory
against ``BatchConfig.memory_limit_gb``:

.. code-block:: python

   from xraylabtool.data_handling.batch_processing import (
       BatchConfig,
       calculate_batch_properties,
       load_batch_input,
       save_batch_results,
   )

   formulas, densities, _ = load_batch_input("materials.csv")  # formula,density columns
   config = BatchConfig(max_workers=8, chunk_size=200, enable_progress=True)
   results = calculate_batch_properties(formulas, [5.0, 8.0, 10.0], densities, config=config)
   save_batch_results(results, "results.csv")

The same path is available from the shell: ``xraylabtool batch materials.csv -o results.csv --workers 8``.

**Cache warming**

The element cache is already full after import. What is built lazily is the per-element
interpolator; warm it for the elements you are about to use:

.. code-block:: python

   from xraylabtool.data_handling.atomic_cache import (
       get_cache_stats,
       warm_cache_for_compounds,
       warm_up_cache,
   )

   warm_up_cache(["Si", "O", "Al", "Fe"])
   warm_cache_for_compounds(["SiO2", "Al2O3"])  # adds related elements by compound family
   print(get_cache_stats())  # {'preloaded_elements': 92, ...}

**JAX backend**

.. code-block:: python

   from xraylabtool.backend import set_backend

   set_backend("jax")    # JIT-compiled kernels; first call pays compilation
   set_backend("numpy")  # default on CPU-only machines

See :doc:`../architecture/jax_architecture` for when JAX wins and when it does not.

Energy grids
------------

Fewer, well-placed points beat dense uniform grids:

.. code-block:: python

   # Logarithmic spacing over the tabulated range (keV)
   energies = np.logspace(np.log10(0.03), np.log10(30), 100)

   # Dense only around an absorption edge (Si K edge at 1.839 keV)
   low = np.logspace(np.log10(0.03), np.log10(1.8), 30)
   edge = np.linspace(1.80, 1.88, 50)
   high = np.logspace(np.log10(1.9), np.log10(30), 30)
   energies = np.concatenate([low, edge, high])

Troubleshooting
---------------

- **First call slow**: interpolator construction (and JIT compilation on the JAX backend).
  Warm the cache or run one throw-away call before timing.
- **Slow per-call on CPU with JAX**: switch to ``set_backend("numpy")``; JAX only pays off on
  GPU or for very large energy arrays.
- **Memory growth in batch jobs**: lower ``BatchConfig.chunk_size`` or ``memory_limit_gb``.
- **Reset interpolators** (rarely needed):
  :func:`xraylabtool.calculators.cache.clear_scattering_factor_cache`.
