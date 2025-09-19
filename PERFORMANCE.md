# Performance Guide

XRayLabTool v0.2.5 includes major performance optimizations that address regressions from v0.2.4 and exceed v0.2.3 baseline performance.

## Version Performance Summary

| Version | Cold Start | Cache Efficiency | Batch Processing | Memory Usage |
|---------|------------|------------------|------------------|--------------|
| v0.2.3  | ~60ms      | 13x speedup     | ~7ms            | ~0.006MB     |
| v0.2.4  | ~912ms     | 8.5x speedup    | ~20ms           | ~2.31MB      |
| v0.2.5  | ~130ms     | 13.4x speedup   | ~1.7ms          | ~0MB         |

## Smart Cache Warming

Smart cache warming loads only the atomic data needed for a specific calculation instead of all priority elements.

### How It Works

```python
import xraylabtool as xlt

# First calculation for SiO2 - loads only Si and O data
result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)

# Subsequent calculations reuse cached Si and O data
result2 = xlt.calculate_single_material_properties("SiO2", 12.0, 2.2)  # Fast
```

### Features

- **Formula-Specific Loading**: Parses chemical formula to determine required elements
- **Background Priority Warming**: Falls back to full priority warming if formula parsing fails
- **One-Time Warming**: Cache warming only occurs on first calculation
- **Error Handling**: Graceful fallback to traditional warming on any errors

### Performance Impact

- 90% faster cold start compared to v0.2.4
- Reduces initial calculation time from 912ms to 130ms
- No impact on warm cache performance

## Adaptive Batch Processing

Batch processing automatically switches between sequential and parallel modes based on workload size.

### Threshold Behavior

```python
# Small batches (<20 items) - sequential processing
small_batch = ["Si", "SiO2", "Al2O3"]  # 3 items
energies = [10.0] * 3
densities = [2.33, 2.2, 3.95]
results = xlt.calculate_xray_properties(small_batch, energies, densities)

# Large batches (≥20 items) - parallel processing with ThreadPoolExecutor
large_batch = ["Si"] * 25  # 25 items
energies = [10.0] * 25
densities = [2.33] * 25
results = xlt.calculate_xray_properties(large_batch, energies, densities)
```

### Optimization Logic

- **20-item threshold**: Automatically determined optimal switch point
- **Sequential for small batches**: Eliminates threading overhead for small workloads
- **Parallel for large batches**: Maximizes CPU utilization for large workloads
- **CPU-aware**: Uses optimal number of worker threads based on system cores

### Performance Impact

- 75% faster than v0.2.3 baseline for small batches
- Reduces batch processing overhead from threading for small workloads
- Maintains high throughput for large datasets

## Environment-Controlled Features

Performance monitoring features are disabled by default and can be enabled via environment variables.

### Environment Variables

```bash
# Enable cache metrics tracking (disabled by default)
export XRAYLABTOOL_CACHE_METRICS=true

# Enable memory profiling (disabled by default)
export XRAYLABTOOL_MEMORY_PROFILING=true
```

### Cache Metrics

```python
# When XRAYLABTOOL_CACHE_METRICS=true
from xraylabtool.data_handling.cache_metrics import get_cache_stats

stats = get_cache_stats()
print(stats)
# {'hits': 45, 'misses': 5, 'total': 50, 'hit_rate': 0.9}

# When disabled (default) - returns empty dict
print(get_cache_stats())  # {}
```

### Memory Profiling

```python
# Memory profiling structures are None until activated
from xraylabtool.optimization.memory_profiler import _memory_snapshots

# Disabled by default - None (no memory overhead)
print(_memory_snapshots)  # None

# Only initialized when XRAYLABTOOL_MEMORY_PROFILING=true
```

### Benefits

- **Zero overhead when disabled**: No performance impact in production
- **Optional debugging**: Enable metrics only when needed
- **Lazy initialization**: Structures only created when required
- **Memory efficient**: No background tracking by default

## Memory Optimizations

v0.2.5 includes several memory optimizations that reduce overhead to near zero.

### Lazy Module Loading

Heavy dependencies are only imported when needed:

```python
# scipy only imported when interpolators are needed
# pandas only imported for file I/O operations
# matplotlib only imported for plotting functions
```

### Cache Metrics Simplification

The cache metrics module was simplified by 77%:

- **Before**: 465 lines with complex threading and tracking
- **After**: 108 lines with lightweight counters
- **Memory reduction**: Eliminated background data structures
- **Performance gain**: Removed overhead from tracking operations

### Memory Profiling Improvements

- **Lazy initialization**: Profiling structures only created when enabled
- **Environment controlled**: No memory allocation unless explicitly enabled
- **Zero overhead**: Disabled by default with no background processes

## Performance Best Practices

### Maximum Speed Configuration

```python
import xraylabtool as xlt

# 1. Smart cache warming is automatic - no configuration needed
result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)

# 2. Metrics disabled by default - no environment variables needed
# Production code runs at maximum speed by default

# 3. Adaptive batch processing is automatic
# Small batches: sequential, large batches: parallel

# 4. Reuse calculations when possible
result1 = xlt.calculate_single_material_properties("Si", 10.0, 2.33)  # Warms cache
result2 = xlt.calculate_single_material_properties("Si", 12.0, 2.33)  # Fast
```

### Debugging Performance Issues

```bash
# Enable metrics only when debugging
export XRAYLABTOOL_CACHE_METRICS=true
export XRAYLABTOOL_MEMORY_PROFILING=true
```

```python
# Monitor cache performance
from xraylabtool.data_handling.cache_metrics import get_cache_stats

stats = get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.1%}")
print(f"Total operations: {stats['total']}")
```

### Batch Processing Guidelines

```python
# For small datasets (<20 items)
# Use standard API - sequential processing is automatic
results = xlt.calculate_xray_properties(formulas, energies, densities)

# For large datasets (≥20 items)
# Use standard API - parallel processing is automatic
# No special configuration needed
```

## Performance Monitoring

### Cache Statistics

```python
# Enable cache metrics
import os
os.environ['XRAYLABTOOL_CACHE_METRICS'] = 'true'

# Import after setting environment variable
from xraylabtool.data_handling.cache_metrics import get_cache_stats, reset_cache_stats

# Reset counters
reset_cache_stats()

# Perform calculations
result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)

# Check statistics
stats = get_cache_stats()
print(f"Operations: {stats['total']}")
print(f"Hit rate: {stats['hit_rate']:.1%}")
```

### Memory Usage

```python
# Enable memory profiling
import os
os.environ['XRAYLABTOOL_MEMORY_PROFILING'] = 'true'

# Import after setting environment variable
from xraylabtool.optimization.memory_profiler import get_memory_usage

# Check memory usage
memory_mb = get_memory_usage()
print(f"Memory usage: {memory_mb:.1f} MB")
```

### Performance Benchmarking

```python
import time
import xraylabtool as xlt

# Clear cache for cold start test
from xraylabtool.calculators.core import clear_scattering_factor_cache
clear_scattering_factor_cache()

# Measure cold start time
start = time.perf_counter()
result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)
cold_start_time = time.perf_counter() - start

print(f"Cold start time: {cold_start_time*1000:.1f}ms")

# Measure warm cache time
start = time.perf_counter()
result = xlt.calculate_single_material_properties("SiO2", 12.0, 2.2)
warm_time = time.perf_counter() - start

print(f"Warm cache time: {warm_time*1000:.1f}ms")
print(f"Cache speedup: {cold_start_time/warm_time:.1f}x")
```

## Troubleshooting Performance Issues

### Slow Cold Start

If cold start is slower than expected:

1. Check that smart cache warming is working:
```python
# Verify cache warming occurs
from xraylabtool.calculators.core import clear_scattering_factor_cache
clear_scattering_factor_cache()

# This should trigger smart warming
result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)
```

2. Check for formula parsing issues:
```python
# Test formula parsing
from xraylabtool.utils import parse_formula
try:
    elements, counts = parse_formula("SiO2")
    print(f"Elements: {elements}")  # Should be ['Si', 'O']
except Exception as e:
    print(f"Parsing error: {e}")  # Falls back to priority warming
```

### Poor Cache Performance

If cache performance is poor:

1. Enable cache metrics:
```bash
export XRAYLABTOOL_CACHE_METRICS=true
```

2. Monitor hit rates:
```python
from xraylabtool.data_handling.cache_metrics import get_cache_stats
stats = get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
```

3. Check if different formulas are being used:
```python
# Same formula - good cache reuse
result1 = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)
result2 = xlt.calculate_single_material_properties("SiO2", 12.0, 2.2)

# Different formulas - different cache entries
result3 = xlt.calculate_single_material_properties("Al2O3", 10.0, 3.95)
```

### High Memory Usage

If memory usage is high:

1. Check if profiling is enabled:
```bash
echo $XRAYLABTOOL_MEMORY_PROFILING  # Should be empty or 'false'
```

2. Disable metrics if enabled in production:
```bash
unset XRAYLABTOOL_CACHE_METRICS
unset XRAYLABTOOL_MEMORY_PROFILING
```

3. Use batch processing for large datasets:
```python
# For very large datasets, consider chunking
def process_in_chunks(formulas, energies, densities, chunk_size=100):
    results = {}
    for i in range(0, len(formulas), chunk_size):
        chunk_formulas = formulas[i:i+chunk_size]
        chunk_energies = energies[i:i+chunk_size] if isinstance(energies, list) else energies
        chunk_densities = densities[i:i+chunk_size]

        chunk_results = xlt.calculate_xray_properties(chunk_formulas, chunk_energies, chunk_densities)
        results.update(chunk_results)

    return results
```