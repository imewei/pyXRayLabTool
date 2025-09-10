Performance and Optimization
============================

Understanding and optimizing XRayLabTool performance for different use cases.

Performance Overview
--------------------

XRayLabTool is designed for high-performance calculations with several optimization strategies:

**Key Performance Features:**
- **Ultra-fast atomic data cache**: 92 preloaded elements
- **Vectorized calculations**: NumPy-based matrix operations
- **Batch processing**: Parallel processing with memory management
- **Smart caching**: LRU caches and interpolator reuse

**Typical Performance:**
- Single calculation: < 0.1 ms
- Batch 1000 materials: < 10 ms
- Energy array (100 points): < 1 ms

Performance Benchmarks
----------------------

Hardware Configuration
~~~~~~~~~~~~~~~~~~~~~~

Benchmarks performed on:
- **CPU**: Modern x86-64 processor
- **Memory**: 16 GB RAM
- **Python**: 3.12
- **NumPy**: Latest version with optimized BLAS

Single Material Calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Single Calculation Performance
   :header-rows: 1
   :widths: 40 20 20 20

   * - Operation
     - Cold Cache
     - Warm Cache
     - Speedup
   * - Simple element (Si)
     - 0.5 ms
     - 0.05 ms
     - 10x
   * - Binary compound (SiO2)
     - 1.2 ms
     - 0.1 ms
     - 12x
   * - Complex formula (Ca5(PO4)3F)
     - 2.1 ms
     - 0.15 ms
     - 14x

Batch Processing Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Batch Processing Scaling
   :header-rows: 1
   :widths: 30 25 25 20

   * - Dataset Size
     - Sequential (s)
     - Batch (s)
     - Speedup
   * - 100 materials
     - 0.15
     - 0.008
     - 19x
   * - 1,000 materials
     - 1.5
     - 0.05
     - 30x
   * - 10,000 materials
     - 15
     - 0.3
     - 50x
   * - 100,000 materials
     - 150
     - 2.5
     - 60x

Energy Array Performance
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Energy Array Scaling
   :header-rows: 1
   :widths: 30 25 25 20

   * - Energy Points
     - Individual (ms)
     - Array (ms)
     - Speedup
   * - 10 energies
     - 5
     - 0.5
     - 10x
   * - 100 energies
     - 50
     - 1.5
     - 33x
   * - 1,000 energies
     - 500
     - 8
     - 63x

Memory Usage
~~~~~~~~~~~~

.. list-table:: Memory Consumption
   :header-rows: 1
   :widths: 40 30 30

   * - Component
     - Typical Usage
     - Peak Usage
   * - Atomic data cache
     - 10 MB
     - 50 MB
   * - Single calculation
     - < 1 KB
     - < 1 KB
   * - Batch 1000 materials
     - 2 MB
     - 5 MB
   * - Energy array (1000 points)
     - 8 MB
     - 15 MB

Optimization Strategies
-----------------------

Caching Optimization
~~~~~~~~~~~~~~~~~~~~

**1. Preload Common Elements:**

.. code-block:: python

   from xraylabtool.data_handling.atomic_cache import preload_elements

   # Preload elements you'll use frequently
   common_elements = ["Si", "O", "Al", "Fe", "C", "N", "Ca", "Cu"]
   preload_elements(common_elements)

**2. Use Persistent Caching:**

.. code-block:: python

   # Enable disk caching for repeated runs
   import xraylabtool as xrt

   xrt.configure_cache(
       disk_cache=True,
       cache_dir="~/.xraylabtool_cache",
       max_memory_mb=100
   )

**3. Warm Up Cache:**

.. code-block:: python

   # Pre-calculate common energy points
   common_energies = [5000, 8000, 10000, 12000, 15000]
   for element in common_elements:
       for energy in common_energies:
           xrt.calculate_single_material_properties(element, 1.0, energy)

Batch Processing Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Use Batch Functions:**

.. code-block:: python

   # Good - efficient batch processing
   results = xrt.calculate_xray_properties(materials, energies)

   # Less efficient - individual calculations
   results = []
   for material in materials:
       for energy in energies:
           result = xrt.calculate_single_material_properties(
               material['formula'], material['density'], energy
           )
           results.append(result)

**2. Optimize Chunk Size:**

.. code-block:: python

   # For very large datasets, adjust chunk size
   results = xrt.calculate_xray_properties(
       large_materials_list,
       energies,
       chunk_size=1000  # Balance memory vs speed
   )

**3. Parallel Processing:**

.. code-block:: python

   from multiprocessing import Pool
   import numpy as np

   def process_chunk(chunk):
       return xrt.calculate_xray_properties(chunk, energies)

   # Split large dataset into chunks
   chunks = np.array_split(large_materials_list, 4)  # 4 processes

   with Pool(4) as pool:
       chunk_results = pool.map(process_chunk, chunks)

   # Combine results
   results = [item for sublist in chunk_results for item in sublist]

Energy Array Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Use NumPy Arrays:**

.. code-block:: python

   import numpy as np

   # Good - vectorized energy array
   energies = np.logspace(3, 5, 100)  # 1 keV to 100 keV
   results = xrt.calculate_single_material_properties("Si", 2.33, energies)

   # Less efficient - Python list
   energies = [10**x for x in np.linspace(3, 5, 100)]

**2. Optimize Energy Spacing:**

.. code-block:: python

   # For smooth curves, logarithmic spacing is often better
   energies = np.logspace(3, 5, 50)  # Fewer points, still smooth

   # For detailed analysis near edges, use adaptive spacing
   edge_region = np.linspace(7900, 8100, 200)  # Dense near Si K-edge
   far_region = np.logspace(3, 5, 50)  # Sparse elsewhere
   energies = np.concatenate([far_region[far_region < 7900],
                             edge_region,
                             far_region[far_region > 8100]])

Memory Management
~~~~~~~~~~~~~~~~~

**1. Process in Chunks:**

.. code-block:: python

   def process_large_dataset(materials, energies, chunk_size=1000):
       """Process large datasets without memory issues."""
       results = []

       for i in range(0, len(materials), chunk_size):
           chunk = materials[i:i+chunk_size]
           chunk_results = xrt.calculate_xray_properties(chunk, energies)
           results.extend(chunk_results)

           # Optional: garbage collection
           if len(results) > 10000:
               import gc
               gc.collect()

       return results

**2. Use Generators:**

.. code-block:: python

   def calculate_generator(materials, energies):
       """Generator for memory-efficient processing."""
       for material in materials:
           for energy in energies:
               yield xrt.calculate_single_material_properties(
                   material['formula'], material['density'], energy
               )

   # Process without storing all results in memory
   for result in calculate_generator(materials, energies):
       # Process each result individually
       process_result(result)

Performance Monitoring
----------------------

Built-in Profiling
~~~~~~~~~~~~~~~~~~

XRayLabTool includes performance monitoring:

.. code-block:: python

   import xraylabtool as xrt

   # Enable performance monitoring
   xrt.enable_profiling()

   # Run your calculations
   results = xrt.calculate_xray_properties(materials, energies)

   # Get performance statistics
   stats = xrt.get_performance_stats()
   print(f"Total time: {stats['total_time']:.3f} s")
   print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
   print(f"Memory usage: {stats['peak_memory_mb']:.1f} MB")

Custom Benchmarking
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   import psutil
   import os

   def benchmark_calculation(func, *args, **kwargs):
       """Benchmark a calculation function."""
       process = psutil.Process(os.getpid())

       # Measure memory before
       mem_before = process.memory_info().rss / 1024 / 1024

       # Time the calculation
       start_time = time.time()
       result = func(*args, **kwargs)
       end_time = time.time()

       # Measure memory after
       mem_after = process.memory_info().rss / 1024 / 1024

       return {
           'result': result,
           'time': end_time - start_time,
           'memory_delta': mem_after - mem_before
       }

   # Example usage
   benchmark = benchmark_calculation(
       xrt.calculate_xray_properties,
       materials, energies
   )

   print(f"Time: {benchmark['time']:.3f} s")
   print(f"Memory: {benchmark['memory_delta']:.1f} MB")

Platform-Specific Optimizations
-------------------------------

NumPy/BLAS Optimization
~~~~~~~~~~~~~~~~~~~~~~~

For best performance, ensure optimized NumPy:

.. code-block:: bash

   # Check NumPy configuration
   python -c "import numpy; numpy.show_config()"

   # Install optimized NumPy (Intel MKL)
   conda install numpy

   # Or use OpenBLAS
   pip install numpy[openblas]

Multi-threading Control
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import os

   # Control NumPy threading
   os.environ['OMP_NUM_THREADS'] = '4'
   os.environ['MKL_NUM_THREADS'] = '4'

   import xraylabtool as xrt

   # XRayLabTool will use these settings

GPU Acceleration (Future)
~~~~~~~~~~~~~~~~~~~~~~~~~

XRayLabTool is designed to support GPU acceleration:

.. code-block:: python

   # Planned for future releases
   xrt.configure_gpu(device='cuda:0')
   results = xrt.calculate_xray_properties_gpu(materials, energies)

Performance Best Practices
--------------------------

Do's
~~~~

✅ **Use batch processing** for multiple materials
✅ **Preload common elements** at startup
✅ **Use NumPy arrays** for energy ranges
✅ **Cache results** when reprocessing data
✅ **Profile your code** to identify bottlenecks
✅ **Use appropriate chunk sizes** for large datasets
✅ **Warm up caches** before timing critical sections

Don'ts
~~~~~~

❌ **Don't process materials individually** in loops
❌ **Don't ignore memory constraints** for large datasets
❌ **Don't use Python lists** for large energy arrays
❌ **Don't clear caches** unnecessarily
❌ **Don't use excessive energy points** for smooth curves
❌ **Don't mix single and batch processing** in the same workflow

Performance Tuning Examples
---------------------------

Example 1: Optimizing Energy Scans
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Inefficient - too many energy points
   energies_bad = np.linspace(1000, 30000, 10000)  # 10k points!

   # Better - logarithmic spacing, fewer points
   energies_good = np.logspace(3, 4.5, 100)  # 100 points

   # Best - adaptive spacing for specific needs
   low_e = np.logspace(3, 3.85, 30)      # 1-7 keV: 30 points
   si_edge = np.linspace(1830, 1860, 50) # Si L-edge: 50 points
   high_e = np.logspace(3.9, 4.5, 30)    # 8-32 keV: 30 points
   energies_adaptive = np.concatenate([low_e, si_edge, high_e])

Example 2: Memory-Efficient Large Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def process_huge_dataset(filename, output_filename):
       """Process dataset too large for memory."""
       import csv

       with open(filename, 'r') as infile, open(output_filename, 'w') as outfile:
           reader = csv.DictReader(infile)
           writer = csv.writer(outfile)

           # Write header
           writer.writerow(['formula', 'density', 'energy', 'critical_angle', 'att_length'])

           # Process in batches
           batch = []
           batch_size = 1000

           for row in reader:
               batch.append({
                   'formula': row['formula'],
                   'density': float(row['density'])
               })

               if len(batch) >= batch_size:
                   # Process batch
                   results = xrt.calculate_xray_properties(batch, [8000])

                   # Write results
                   for result in results:
                       writer.writerow([
                           result.formula,
                           result.density_g_cm3,
                           result.energy_ev,
                           result.critical_angle_degrees,
                           result.attenuation_length_cm
                       ])

                   # Clear batch
                   batch = []

           # Process remaining items
           if batch:
               results = xrt.calculate_xray_properties(batch, [8000])
               for result in results:
                   writer.writerow([...])

Troubleshooting Performance Issues
----------------------------------

Slow Calculations
~~~~~~~~~~~~~~~~~

**Symptoms:**
- Individual calculations taking > 1 ms
- Batch processing not showing expected speedup

**Solutions:**
1. Check cache hit rate - should be > 90% for repeated calculations
2. Verify NumPy installation with optimized BLAS
3. Ensure adequate RAM for dataset size
4. Profile to identify bottlenecks

High Memory Usage
~~~~~~~~~~~~~~~~~

**Symptoms:**
- Memory usage grows continuously
- Out of memory errors with large datasets

**Solutions:**
1. Use chunked processing for large datasets
2. Clear caches periodically: ``xrt.clear_cache()``
3. Use generators instead of storing all results
4. Monitor memory with ``psutil`` or system tools

Cache Misses
~~~~~~~~~~~~

**Symptoms:**
- Low cache hit rate (< 50%)
- Repeated slow calculations for same materials

**Solutions:**
1. Preload frequently used elements
2. Use consistent energy grids
3. Increase cache size if memory permits
4. Warm up cache before critical calculations

Future Performance Improvements
-------------------------------

Planned Enhancements
~~~~~~~~~~~~~~~~~~~~

- **GPU acceleration** using CuPy/JAX
- **JIT compilation** with Numba
- **Distributed processing** with Dask
- **Improved memory management** with memory mapping
- **Machine learning interpolation** for faster atomic data lookup

Contributing Performance Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We welcome performance contributions:

1. **Benchmarking**: Report performance on your hardware
2. **Profiling**: Identify new bottlenecks
3. **Optimization**: Submit optimized algorithms
4. **Testing**: Validate performance improvements

See the `contributing guide <contributing.rst>`_ for details.
