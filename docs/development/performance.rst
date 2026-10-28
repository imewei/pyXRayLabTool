Performance and Optimization
============================

Where the time goes
-------------------

- **Atomic data**: the f1, f2 and PCHIP slopes of all 92 elements are one stacked table
  (:mod:`xraylabtool.calculators.tables`), read from ``scattering_tables.npz`` and uploaded to
  the JAX device once per process. It is the only thing the first call has to load.
- **Per-call work**: formula parsing and validation on the host, then one compiled device
  program (:mod:`xraylabtool.calculators.optics`) that interpolates f1/f2 and derives δ, β and
  the derived quantities, with one upload and one download. Cost scales with the number of
  energy points, not the number of calls, so pass an energy array instead of looping.
- **Backend**: everything runs in JAX float64 on JAX's default device (the CPU, or the GPU of a
  CUDA install); there is no NumPy path (ADR-006, :doc:`../architecture/006-adr-core-jax-only`).
  A small single call costs a few tenths of a millisecond (launch latency), more than the
  removed NumPy path did, and on CPU NumPy was also faster at 20 000 energies; batches (and, on a GPU, large energy grids) are where the device program pays off.
- **Compilation**: shapes are padded to buckets (the core energy axis to powers of two from 64
  to 4096, then a 1.25 ladder), so a program compiled for one energy count
  serves every count in the same bucket and a repeat call compiles nothing. The first call
  compiles; ``JAX_PLATFORMS=cpu`` selects the CPU device.

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

:func:`xraylabtool.calculate_xray_properties` runs the materials as blocks through one device
program (grouped by element-count bucket, results in input order) and returns
``dict[formula, XRayResult]``:

.. code-block:: python

   results = xrt.calculate_xray_properties(["Si", "SiO2", "Al2O3"], 8.0, [2.33, 2.2, 3.95])

For large CSV-driven jobs use
:func:`xraylabtool.data_handling.batch_processing.calculate_batch_properties`, which chunks
the input (``BatchConfig.chunk_size``, default 100, also the most materials per device block)
and watches memory against ``BatchConfig.memory_limit_gb``. There are no worker pools:
``BatchConfig.max_workers`` and ``--workers`` are accepted and ignored.

.. code-block:: python

   from xraylabtool.data_handling.batch_processing import (
       BatchConfig,
       calculate_batch_properties,
       load_batch_input,
       save_batch_results,
   )

   formulas, densities, _ = load_batch_input("materials.csv")  # formula,density columns
   config = BatchConfig(chunk_size=200, enable_progress=True)
   results = calculate_batch_properties(formulas, [5.0, 8.0, 10.0], densities, config=config)
   save_batch_results(results, "results.csv")

The same path is available from the shell: ``xraylabtool batch materials.csv -o results.csv``.

**Warming up**

The first compute call loads the resident tables and compiles the program of its own shape
buckets. To pay that cost up front, call ``xraylabtool.warm_up`` (explicit; nothing warms
up behind your back):

.. code-block:: python

   import xraylabtool as xrt

   xrt.warm_up(("SiO2", "Al2O3"))  # compiles the default programs for these element counts
   xrt.warm_up(("SiO2",), energy_counts=(1, 500, 2000))  # also the energy buckets 64, 512 and 2048

A new energy bucket costs a 100-300 ms compile, so pass the grid sizes you will use as
``energy_counts``.

The CLI and GUI also keep JAX's persistent compile cache, so later processes skip the
compile (``XRAYLABTOOL_NO_JAX_CACHE=1`` opts out). The library does not touch JAX's global
configuration unasked; set ``XRAYLABTOOL_JAX_CACHE=1`` to enable the cache for your own
scripts (a warm cache saves about 170 ms on a fresh process's first call). :func:`xraylabtool.calculators.cache.clear_scattering_factor_cache`
drops the resident tables and the encoding memo (rarely needed).

**Staying on the device**

:func:`xraylabtool.calculators.optics.optical_properties` is the pure device function behind
the calculators. It works under ``jax.jit``, ``jax.vmap`` and ``jax.grad`` and avoids the
host round trip entirely when you already hold device arrays; see
:doc:`../architecture/jax_architecture`.

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

- **First call slow**: it loads the tables and compiles a program. Call
  :func:`xraylabtool.warm_up` first, or rely on the persistent compile cache in the CLI and GUI.
- **Small single calls are slower than before**: a few tenths of a millisecond per call is the
  launch latency of a device program (ADR-006). Pass energy arrays or batches instead of looping.
- **Memory growth in batch jobs**: lower ``BatchConfig.chunk_size`` or ``memory_limit_gb``.
