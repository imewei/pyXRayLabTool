# xraylabtool Performance Baseline Profiling Report

**Date:** 2026-04-06  
**Profiler:** systems-profiler agent  
**Tool:** cProfile + tracemalloc + manual microbenchmarks  
**Python:** 3.13.9 | Platform: darwin (Apple Silicon)

---

## Executive Summary

The library has two radically different performance regimes:

| Scenario | Time |
|---|---|
| **Cold start** (first call, scipy not imported) | ~270ms |
| **First call per formula** (scipy imported, no interpolators) | ~0.7–1.5ms |
| **Warm cache** (interpolators cached, 500 pts) | ~0.1ms |
| **Warm cache** (interpolators cached, 5000 pts) | ~0.35ms |

The library is **fast once warm**. All optimization effort should focus on cold-start and first-call-per-formula latency.

---

## 1. Hot Function Analysis (Warm Cache, 80 calls × 5000-point arrays)

Total profiled runtime: 27ms across 80 calls (0.34ms/call avg).

| Rank | Function | Total Time | % of Runtime | Notes |
|---|---|---|---|---|
| 1 | `scipy.interpolate._interpolate._evaluate` | 10ms | 37% | Called per element per call; 2–3 per material |
| 2 | `calculate_derived_quantities` | 2ms | 7% | NaN/Inf validation overhead dominant |
| 3 | `load_scattering_factor_data` (CSV parse) | 2ms | 7% | Only on first call per element |
| 4 | `calculate_scattering_factors` | 1ms | 4% | np.stack + einsum overhead |
| 5 | `_validate_single_material_inputs` | 1ms | 4% | `np.any(ops.asarray(...))` repeated |
| 6 | `numpy.stack` | 1ms | 4% | Allocates intermediate matrix every call |
| 7 | NaN/Inf validation (`ops.any(ops.isnan/isinf)`) | 0.8ms | 3% | 4 separate array scans per call |
| 8 | `scipy.PchipInterpolator.__call__` | — | ~37% | Wraps `_evaluate` above |
| 9 | `create_scattering_factor_interpolators` (cache miss) | 0.2ms each | — | CSV parse + 2× PCHIP construction |
| 10 | deprecated property aliases (9 accesses) | 5.7μs/batch | negligible | `calculate_multiple_xray_properties` only |

---

## 2. Cold-Start Analysis

**Total cold-start time: ~270ms**

Breakdown:
- `scipy.interpolate` module import: **~265ms** (one-time, ~98% of cold start)
- `.nff` file CSV parsing (per element): 0.13–0.14ms each
- `PchipInterpolator` construction: ~0.1ms per element pair

The cold start is entirely dominated by lazy `scipy.interpolate` import inside `InterpolationFactory.create_pchip()` (called during first `create_scattering_factor_interpolators()` invocation). After scipy is imported, first-call-per-formula is ~0.7–1.5ms.

---

## 3. Memory Allocation Hotspots

From tracemalloc over 50 calls × 5000-point arrays:

| Location | Size per 50 calls | Source |
|---|---|---|
| `numpy.shape_base.py:465` (np.stack) | 137 KiB | `calculate_scattering_factors` multi-element matrix build |
| `core.py:591` (ScatteringData constructor) | 51.6 KiB | CSV data buffered per element |
| `core.py:986–989` (f1/f2 computed arrays) | 39.2 KiB × 6 | Intermediate arrays in `calculate_scattering_factors` |
| `core.py:1068–1082` (SLD + critical angle) | 39.2 KiB × 4 | Derived quantities arrays |
| `scipy array_api_compat` | 35 KiB | Called per interpolation `__call__` |

The dominant allocation is the **intermediate f1/f2 matrix** created by `np.stack(f1_rows)` in `calculate_scattering_factors` (core.py:951–952). This is allocated and discarded every call even though the formula doesn't change.

---

## 4. I/O Bottlenecks

- **`.nff` file reads:** 0.13–0.14ms per element (csv.reader). The `_scattering_factor_cache` dict eliminates re-reads after first access — this is well-handled.
- **No network or database I/O** in the hot path. `Mendeleev` queries only for unknown elements (extremely rare if preloaded cache covers them, which it does for all 92 elements).
- **No I/O bottleneck in warm path.**

---

## 5. Algorithmic Complexity Concerns

### 5.1 Interpolation: O(n log n) per element per call
`scipy.PchipInterpolator.__call__` does binary search (O(n log n) in energy array size). At 5000 points, this costs ~36μs per call (2 calls per element). For a 3-element formula, that is ~210μs just for interpolation. This is the **dominant warm-path cost**.

### 5.2 calculate_derived_quantities: 4 redundant array scans
At core.py:1030–1039, four separate validation checks each do a full array scan:
```
ops.any(ops.isnan(dispersion))   # scan 1
ops.any(ops.isnan(absorption))   # scan 2
ops.any(ops.isinf(dispersion))   # scan 3
ops.any(ops.isinf(absorption))   # scan 4
```
A single `np.all(np.isfinite(...))` call cuts this to 2 scans. Measured savings: 8.0μs → 3.8μs per call.

### 5.3 `_validate_single_material_inputs`: repeated scalar conversion
At core.py:1030–1039 and 1175–1186, `ops.asarray(mass_density)` is called multiple times across the call chain for a scalar that never changes.

### 5.4 Multi-element path: matrix allocated every call
`calculate_scattering_factors` (core.py:940–962) builds `f1_matrix` and `f2_matrix` via `np.stack` on every invocation, even for repeated formulas. For a warm-cache call with a cached formula, these intermediate 2D arrays (n_elements × n_energies) are the primary allocations.

### 5.5 `calculate_multiple_xray_properties` accesses 9 deprecated aliases
Each deprecated property access involves `warnings.warn()` + frame inspection. For batch multi-material workflows this is unnecessary overhead (5.7μs per 9-property batch).

---

## 6. Vectorization Assessment

- `calculate_scattering_factors` is already vectorized over energy using einsum for multi-element case (good).
- `vectorized_core.py` and `optimized_core.py` are deprecated shims that redirect to JAX backend — they add a deprecation-warning call overhead but are not in the hot path.
- `InterpolationFactory` supports switching to JAX/interpax backend via `set_backend('jax')` but the default numpy/scipy path is used.
- No O(n²) algorithms found in the hot path.
- `calculate_multiple_xray_properties` processes materials sequentially in a Python loop (not batched). For N>10 materials, this is a multi-threading opportunity.

---

## 7. Top 10 Optimization Priority Order

| Priority | Bottleneck | Expected Gain | Effort |
|---|---|---|---|
| 1 | **Eagerly import `scipy.interpolate` at module load** (not lazily) | Eliminates 270ms cold-start penalty | Trivial — 1 line |
| 2 | **Replace 4× NaN/Inf checks with single `np.isfinite` call** | 52% reduction in validation overhead (~4μs/call) | Trivial — 2 lines |
| 3 | **Cache interpolator results per (formula, energy_hash)** | Eliminates `np.stack` allocation + einsum for repeated formula+energy combos | Low — add result cache |
| 4 | **Switch to JAX backend + `interpax.PchipInterpolator`** | JIT-compiled interpolation, eliminates scipy overhead | Medium — test harness |
| 5 | **Batch `scipy.PchipInterpolator` evaluation** | Reduce per-call overhead for multi-element compounds | Low — restructure loop |
| 6 | **Fix `calculate_multiple_xray_properties`** to use new field names | Remove `warnings.warn` cost from multi-material hot path | Trivial |
| 7 | **Pre-build f1/f2 matrix once per (formula)** and store in interpolator cache | Eliminates np.stack alloc on every call | Low — extend cache value |
| 8 | **Use `np.frompyfunc` or `np.genfromtxt` for .nff parsing** | Minor improvement on first-call-per-element (~0.02ms) | Low |
| 9 | **Parallelize `calculate_multiple_xray_properties`** over materials | Linear speedup for N>cpu_count materials | Low — already have batch_processing |
| 10 | **Reduce `_validate_single_material_inputs` scalar conversions** | Remove redundant `ops.asarray` wrapping | Trivial |

---

## 8. Recommended Immediate Actions

### High Impact, Low Effort
1. **Add `from scipy.interpolate import PchipInterpolator` to module-level imports** in `xraylabtool/backend/interpolation.py`. This moves the 265ms scipy import cost to module load time (done once, not on first calculation).

2. **Merge the 4 validation array-scans in `calculate_derived_quantities`** (core.py:1030–1039) into:
   ```python
   if not (np.all(np.isfinite(dispersion)) and np.all(np.isfinite(absorption))):
       raise ValueError(...)
   ```

3. **Fix `calculate_multiple_xray_properties`** to access `result.formula`, `result.molecular_weight_g_mol` etc. instead of deprecated aliases, removing the `warnings.warn` call chain from the multi-material path.

### Medium Impact, Medium Effort
4. **Enable JAX backend** as the recommended path (it is already implemented) — `set_backend('jax')` enables XLA-JIT-compiled interpolation via `interpax`, which will eliminate the scipy overhead entirely for large energy arrays.

---

## 9. Baseline Numbers (Summary)

| Metric | Value |
|---|---|
| Cold start latency | 270ms (scipy import dominated) |
| First call per new formula (scipy warm) | 0.7–1.5ms |
| Warm cache latency, 500 pts | ~0.1ms |
| Warm cache latency, 5000 pts | ~0.35ms |
| Interpolation time per element, 5000 pts | ~36μs |
| `calculate_derived_quantities` overhead, 5000 pts | ~35μs |
| NaN/Inf validation overhead | 8μs (reducible to 4μs) |
| Batch (5 unique materials, 500 pts, parallel=4) | 6.1ms |
| Memory per call (5000 pts, 3-element formula) | ~500 KiB peak |
