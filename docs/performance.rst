Performance and Optimization
============================

**Key Features:** Atomic data cache (10-50x speedup), vectorized calculations, batch processing

**Typical Performance:**
- Single calculation: < 0.1 ms
- Batch 1000 materials: < 10 ms
- Energy array (100 points): < 1 ms

Performance Benchmarks
----------------------

**Single Material Performance:**
- Simple element (Si): 0.5 ms → 0.05 ms (warm cache, 10x speedup)
- Complex formula: 2.1 ms → 0.15 ms (warm cache, 14x speedup)

**Batch Processing Scaling:**
- 1,000 materials: 1.5s sequential → 0.05s batch (30x speedup)
- 100,000 materials: 150s sequential → 2.5s batch (60x speedup)

**Memory Usage:**
- Atomic data cache: 10-50 MB
- Batch 1000 materials: 2-5 MB
- Energy array (1000 points): 8-15 MB

Optimization Strategies
-----------------------

**Caching:**

.. code-block:: python

   from xraylabtool.data_handling.atomic_cache import preload_elements
   import xraylabtool as xrt

   # Preload common elements
   preload_elements(["Si", "O", "Al", "Fe", "C", "N"])

   # Configure caching
   xrt.configure_cache(disk_cache=True, max_memory_mb=100)

**Batch Processing:**

.. code-block:: python

   # Efficient batch processing
   results = xrt.calculate_xray_properties(materials, energies)

   # For large datasets, use chunks
   results = xrt.calculate_xray_properties(
       materials, energies, chunk_size=1000
   )

**Energy Arrays:**

.. code-block:: python

   import numpy as np

   # Use logarithmic spacing
   energies = np.logspace(3, 5, 100)  # 1-100 keV

   # Adaptive spacing near edges
   edge_region = np.linspace(7900, 8100, 200)
   far_region = np.logspace(3, 5, 50)
   energies = np.concatenate([far_region[far_region < 7900],
                             edge_region, far_region[far_region > 8100]])

Performance Monitoring
----------------------

.. code-block:: python

   import xraylabtool as xrt
   import time

   # Built-in profiling
   xrt.enable_profiling()
   results = xrt.calculate_xray_properties(materials, energies)
   stats = xrt.get_performance_stats()
   print(f"Time: {stats['total_time']:.3f}s, Cache: {stats['cache_hit_rate']:.1%}")

   # Custom benchmarking
   start = time.time()
   result = xrt.calculate_xray_properties(materials, energies)
   print(f"Calculation time: {time.time() - start:.3f}s")

Platform Optimizations
----------------------

.. code-block:: bash

   # Check NumPy configuration
   python -c "import numpy; numpy.show_config()"
   conda install numpy  # Intel MKL optimized

.. code-block:: python

   import os
   # Control threading
   os.environ['OMP_NUM_THREADS'] = '4'
   os.environ['MKL_NUM_THREADS'] = '4'

Best Practices
--------------

**Do:**
- Use batch processing for multiple materials
- Preload common elements at startup
- Use NumPy arrays for energy ranges
- Profile code to identify bottlenecks

**Don't:**
- Process materials individually in loops
- Use Python lists for large energy arrays
- Clear caches unnecessarily
- Use excessive energy points

Tuning Examples
---------------

**Energy Scan Optimization:**

.. code-block:: python

   # Bad: too many points
   energies_bad = np.linspace(1000, 30000, 10000)

   # Good: logarithmic spacing
   energies_good = np.logspace(3, 4.5, 100)

   # Best: adaptive spacing
   low_e = np.logspace(3, 3.85, 30)
   si_edge = np.linspace(1830, 1860, 50)
   high_e = np.logspace(3.9, 4.5, 30)
   energies_adaptive = np.concatenate([low_e, si_edge, high_e])

**Large Dataset Processing:**

.. code-block:: python

   def process_huge_dataset(filename, output_filename):
       import csv
       with open(filename, 'r') as infile, open(output_filename, 'w') as outfile:
           reader, writer = csv.DictReader(infile), csv.writer(outfile)
           batch, batch_size = [], 1000

           for row in reader:
               batch.append({'formula': row['formula'], 'density': float(row['density'])})
               if len(batch) >= batch_size:
                   results = xrt.calculate_xray_properties(batch, [8000])
                   for result in results:
                       writer.writerow([result.formula, result.density_g_cm3, ...])
                   batch = []

Troubleshooting
---------------

**Slow Calculations:**
- Check cache hit rate (should be >90%)
- Verify optimized NumPy/BLAS installation
- Use chunked processing for large datasets

**High Memory Usage:**
- Process data in chunks
- Clear caches: ``xrt.clear_cache()``
- Use generators for large datasets

**Cache Misses:**
- Preload frequently used elements
- Use consistent energy grids
- Warm up cache before timing

Enhanced Performance Mode
-------------------------

**New Optimizations:** 20-40x speedup for single calculations, 2-3x faster data loading

.. code-block:: python

   # Enable optimizations
   import os
   os.environ['XRAYLABTOOL_ENABLE_OPTIMIZATIONS'] = '1'

   # Or programmatically
   from xraylabtool.optimization import optimized_core
   optimized_core.enable_optimizations()

**Performance Improvements:**
- Single calculation: 2.1ms → 0.05ms (42x speedup)
- Data loading: 18-21ms → 6-7ms (2.9x speedup)
- Arrays: 1.4x speedup for 50-500 point arrays

**Future Plans:** GPU acceleration, JIT compilation, distributed processing
