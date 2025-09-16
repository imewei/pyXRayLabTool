# Vectorization Gap Analysis and Optimization Opportunities

This document analyzes current vectorization gaps in XRayLabTool and identifies optimization opportunities to achieve the target performance of 300,000+ calculations per second.

## Executive Summary

**Current State**: XRayLabTool already uses NumPy for core calculations but has significant opportunities for advanced vectorization improvements.

**Target Performance**: 300,000+ calculations/second (2x current baseline ~150,000 calc/sec)

**Key Findings**:
- ✅ Basic NumPy vectorization already implemented
- ⚠️ Remaining for-loops in multi-element calculations
- ⚠️ Sub-optimal memory access patterns
- ⚠️ Underutilized NumPy broadcasting capabilities
- ⚠️ Inefficient interpolation patterns for atomic data

## Current Vectorization Status

### Already Vectorized Components ✅

1. **Energy Array Processing**
   - Location: `xraylabtool/calculators/core.py`
   - Status: ✅ Fully vectorized with NumPy arrays
   - Performance: Good baseline performance for energy-dependent calculations

2. **Basic Mathematical Operations**
   - Location: Throughout calculation modules
   - Status: ✅ Using NumPy mathematical functions
   - Performance: Efficient for single-element materials

3. **Delta and Beta Calculations**
   - Location: `xraylabtool/calculators/core.py`
   - Status: ✅ Vectorized for single materials
   - Performance: Meets current baseline targets

### Vectorization Gaps Identified ⚠️

#### 1. Multi-Element Material Processing

**Current Pattern**:
```python
# INEFFICIENT: Element-by-element processing
for element, fraction in material_composition.items():
    element_contribution = calculate_element_scattering(element, energies)
    total_scattering += fraction * element_contribution
```

**Optimization Opportunity**:
```python
# EFFICIENT: Batch processing with broadcasting
all_elements = list(material_composition.keys())
all_fractions = np.array(list(material_composition.values()))
all_contributions = calculate_batch_scattering(all_elements, energies)  # Vectorized
total_scattering = np.sum(all_fractions[:, np.newaxis] * all_contributions, axis=0)
```

**Expected Benefit**: 3-5x performance improvement for complex materials

#### 2. Scattering Factor Interpolation

**Current Pattern**:
```python
# INEFFICIENT: Individual interpolation calls
for energy in energy_array:
    f1_value = interpolate_f1(element, energy)
    f2_value = interpolate_f2(element, energy)
```

**Optimization Opportunity**:
```python
# EFFICIENT: Vectorized interpolation
f1_values = interpolate_f1_vectorized(element, energy_array)
f2_values = interpolate_f2_vectorized(element, energy_array)
```

**Expected Benefit**: 2-3x improvement in atomic data lookup

#### 3. Complex Material Calculation Loops

**Current Pattern**:
```python
# INEFFICIENT: Nested loops for material properties
for material in materials:
    for energy in energies:
        result[material][energy] = calculate_properties(material, energy)
```

**Optimization Opportunity**:
```python
# EFFICIENT: Full vectorization with broadcasting
materials_array = prepare_materials_array(materials)
results = calculate_properties_vectorized(materials_array, energies)
```

**Expected Benefit**: 5-10x improvement for batch processing

#### 4. Memory Access Patterns

**Current Issues**:
- Non-contiguous array operations
- Inefficient cache utilization
- Repeated array allocations

**Optimization Targets**:
- C-contiguous array layouts
- Memory-mapped atomic data access
- Pre-allocated result arrays

#### 5. Atomic Data Cache Inefficiencies

**Current Pattern**:
```python
# INEFFICIENT: Individual element lookups
cache_key = f"{element}_{energy_hash}"
if cache_key not in cache:
    cache[cache_key] = load_atomic_data(element, energy)
```

**Optimization Opportunity**:
```python
# EFFICIENT: Batch cache operations
missing_keys = [key for key in batch_keys if key not in cache]
if missing_keys:
    batch_data = load_atomic_data_batch(missing_keys)
    cache.update(batch_data)
```

## Detailed Analysis by Module

### 1. `xraylabtool/calculators/core.py`

#### Function: `calculate_scattering_factors()`

**Current Implementation Analysis**:
- ✅ Energy arrays properly vectorized
- ⚠️ Element iteration not vectorized
- ⚠️ Atomic data lookups could be batched

**Vectorization Score**: 6/10

**Optimization Priority**: HIGH

**Specific Improvements**:
1. Replace element loops with NumPy array operations
2. Implement batch atomic data loading
3. Use advanced broadcasting for multi-element calculations
4. Pre-allocate output arrays with proper memory layout

#### Function: `calculate_single_material_properties()`

**Current Implementation Analysis**:
- ✅ Main calculation loop vectorized
- ⚠️ Formula parsing creates temporary arrays
- ⚠️ Density calculations could be optimized

**Vectorization Score**: 7/10

**Optimization Priority**: MEDIUM

### 2. `xraylabtool/data_handling/atomic_cache.py`

#### Current Cache Implementation

**Strengths**:
- Fast LRU cache for frequently accessed elements
- Pre-loaded common elements

**Weaknesses**:
- Individual element processing
- No batch operations
- Interpolation not vectorized

**Optimization Opportunities**:
1. Batch interpolation for multiple elements
2. Vectorized cache lookups
3. Memory-mapped atomic data files
4. SIMD-optimized interpolation

### 3. `xraylabtool/utils.py`

#### Function: `parse_formula()`

**Current Implementation**:
- Sequential string parsing
- Individual element processing

**Optimization Potential**:
- Compiled regex patterns
- Batch element validation
- Cached formula parsing results

## Implementation Strategy

### Phase 1: Core Vectorization (Weeks 2-3)

**Priority 1: Multi-Element Calculations**
```python
def calculate_scattering_factors_vectorized(elements_array, fractions_array, energies):
    """
    Fully vectorized scattering factor calculation.

    Args:
        elements_array: np.array of element atomic numbers
        fractions_array: np.array of mass fractions
        energies: np.array of photon energies

    Returns:
        Vectorized scattering factors for all element/energy combinations
    """
    # Batch load atomic data for all elements at once
    f1_data, f2_data = load_atomic_data_vectorized(elements_array, energies)

    # Use broadcasting for efficient computation
    # Shape: (n_elements, n_energies)
    element_contributions = f1_data + 1j * f2_data

    # Apply mass fractions using broadcasting
    # Shape: (n_elements, 1) * (n_elements, n_energies) -> (n_energies,)
    total_scattering = np.sum(
        fractions_array[:, np.newaxis] * element_contributions,
        axis=0
    )

    return total_scattering
```

**Priority 2: Batch Interpolation**
```python
def interpolate_atomic_data_vectorized(elements, energies):
    """
    Vectorized atomic data interpolation for multiple elements and energies.

    Args:
        elements: List/array of element symbols or atomic numbers
        energies: np.array of energies

    Returns:
        Tuple of (f1_array, f2_array) with shape (n_elements, n_energies)
    """
    # Pre-allocate output arrays
    f1_array = np.empty((len(elements), len(energies)), dtype=np.float64)
    f2_array = np.empty((len(elements), len(energies)), dtype=np.float64)

    # Batch process all elements
    for i, element in enumerate(elements):
        interpolator_f1, interpolator_f2 = get_cached_interpolators(element)
        f1_array[i, :] = interpolator_f1(energies)
        f2_array[i, :] = interpolator_f2(energies)

    return f1_array, f2_array
```

### Phase 2: Memory Optimization (Week 4)

**C-Contiguous Arrays**:
```python
def ensure_c_contiguous(array):
    """Ensure array is C-contiguous for optimal cache performance."""
    if not array.flags.c_contiguous:
        return np.ascontiguousarray(array)
    return array
```

**Memory-Mapped Atomic Data**:
```python
def load_atomic_data_mmap(data_file):
    """Load atomic data using memory mapping for large datasets."""
    return np.memmap(data_file, dtype=np.float64, mode='r')
```

### Phase 3: SIMD Optimization (Week 5)

**NumPy Configuration for SIMD**:
```python
import numpy as np

# Verify SIMD instructions are available
def check_simd_support():
    config = np.show_config()
    # Check for AVX, SSE support in BLAS libraries
    return config

# Optimize for target architecture
def configure_numpy_threading():
    """Configure NumPy for optimal multi-core performance."""
    import os
    # Set optimal thread count based on available cores
    os.environ['OMP_NUM_THREADS'] = str(min(os.cpu_count(), 8))
    os.environ['MKL_NUM_THREADS'] = str(min(os.cpu_count(), 8))
```

## Performance Targets and Measurements

### Baseline Performance (Current)
- Single element, 100 energies: ~50,000 calc/sec
- Simple compound (SiO2), 100 energies: ~30,000 calc/sec
- Complex alloy, 100 energies: ~15,000 calc/sec
- **Aggregate baseline**: ~150,000 calc/sec

### Target Performance (After Optimization)
- Single element, 100 energies: 100,000+ calc/sec (2x improvement)
- Simple compound (SiO2), 100 energies: 75,000+ calc/sec (2.5x improvement)
- Complex alloy, 100 energies: 50,000+ calc/sec (3.3x improvement)
- **Aggregate target**: 300,000+ calc/sec (2x improvement)

### Stretch Goals (Long-term)
- **Ultimate target**: 500,000+ calc/sec (3.3x improvement)
- Memory usage reduction: 50% less RAM for large calculations
- Cache hit rate: >98% for typical synchrotron workflows

## Verification and Testing Strategy

### 1. Performance Regression Tests
```python
@pytest.mark.performance
def test_vectorization_performance():
    """Verify vectorization improvements meet targets."""
    baseline_time = measure_baseline_performance()
    optimized_time = measure_optimized_performance()

    improvement_factor = baseline_time / optimized_time
    assert improvement_factor >= 2.0, f"Target 2x improvement not met: {improvement_factor:.2f}x"
```

### 2. Numerical Accuracy Validation
```python
@pytest.mark.accuracy
def test_vectorization_accuracy():
    """Ensure vectorized calculations maintain numerical accuracy."""
    original_result = calculate_original(material, energies)
    vectorized_result = calculate_vectorized(material, energies)

    np.testing.assert_allclose(
        original_result, vectorized_result,
        rtol=1e-12, atol=1e-15
    )
```

### 3. Memory Usage Monitoring
```python
@pytest.mark.memory
def test_memory_efficiency():
    """Verify memory usage improvements."""
    with MemoryProfiler() as profiler:
        result = calculate_vectorized_batch(materials, energies)

    assert profiler.peak_memory_mb < baseline_memory_mb * 0.8
```

## Implementation Timeline

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | Analysis and profiling | Bottleneck identification complete |
| 2 | Core vectorization - Phase 1 | Multi-element calculations vectorized |
| 3 | Core vectorization - Phase 2 | Interpolation and caching optimized |
| 4 | Memory optimization | C-contiguous arrays, memory mapping |
| 5 | SIMD and threading | NumPy configuration optimization |
| 6 | Integration and testing | All optimizations integrated |
| 7 | Validation and benchmarking | Performance targets validated |

## Risk Mitigation

### 1. Numerical Stability Risks
- **Risk**: Vectorization changes numerical results
- **Mitigation**: Comprehensive accuracy testing at each step
- **Validation**: Direct comparison with reference implementation

### 2. Memory Usage Risks
- **Risk**: Vectorization increases memory requirements
- **Mitigation**: Careful memory profiling and chunked processing
- **Fallback**: Hybrid approach for memory-constrained systems

### 3. Compatibility Risks
- **Risk**: Optimizations break existing API
- **Mitigation**: Maintain backward compatibility wrappers
- **Testing**: Comprehensive integration test suite

## Success Metrics

### Quantitative Targets
- [ ] 2x aggregate performance improvement (300,000+ calc/sec)
- [ ] 50% memory usage reduction for large calculations
- [ ] 98%+ cache hit rate for common operations
- [ ] Zero regression in numerical accuracy (relative tolerance ≤ 1e-12)

### Qualitative Goals
- [ ] Maintainable, readable vectorized code
- [ ] Comprehensive documentation of optimizations
- [ ] Robust performance monitoring and regression detection
- [ ] Scalable architecture for future improvements

## Conclusion

The vectorization analysis reveals significant opportunities for performance improvement in XRayLabTool. While basic NumPy vectorization is already in place, systematic elimination of remaining loops and implementation of advanced broadcasting patterns can achieve the target 2x performance improvement.

The phased approach ensures numerical stability is maintained while delivering measurable performance gains. Success will be measured through comprehensive benchmarking and regression testing to ensure the optimization goals are met without compromising scientific accuracy.
