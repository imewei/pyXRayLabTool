# ADR-001: JAX vs NumPy Computation Backend

**Status:** ACCEPTED
**Date:** 2026-04-06
**Deciders:** Architecture Team
**Supersedes:** None

---

## Context

pyXRayLabTool v0.3.0 performs all X-ray optical property calculations using NumPy arrays with SciPy for interpolation. The computation pipeline is:

```
parse_formula -> load_scattering_data -> PchipInterpolator(energy)
  -> calculate_scattering_factors (einsum/broadcast)
  -> calculate_derived_quantities (sqrt, exp, division)
  -> XRayResult
```

**Current performance characteristics observed in the codebase:**
- `calculate_scattering_factors()` dominates runtime for energy sweeps (>100 points)
- The inner loop iterates over elements, calling interpolators and accumulating weighted sums
- `optimization/vectorized_core.py` already restructures this as matrix ops (`np.einsum`)
- `optimization/optimized_core.py` provides monkey-patching for data loading (2-3x faster I/O)
- Manual SIMD heuristics in `vectorized_core.py` (AVX detection, threshold tuning) indicate the team has hit the ceiling of what NumPy can do

**What JAX offers for this workload:**
1. **JIT compilation:** Fuses the entire `scattering_factors + derived_quantities` pipeline into a single XLA program. Eliminates Python dispatch overhead and intermediate array allocations.
2. **vmap:** Replaces the explicit for-loop over materials in `compute_multiple()` and `batch_processing.py` with automatic vectorization.
3. **Automatic differentiation:** Enables future gradient-based optimization (NLSQ warm-start for Bayesian fitting per CLAUDE.md guidelines).
4. **Platform portability:** Same code runs on CPU (via XLA), GPU (CUDA/ROCm), and Apple Silicon (Metal plugin).

**Which computations benefit most:**

| Function | Pattern | JIT Benefit | vmap Benefit |
|----------|---------|-------------|--------------|
| `calculate_scattering_factors` | Matrix multiply (einsum) + broadcast | HIGH -- fuses multiply+sum+scale into single kernel | HIGH -- vectorize over materials |
| `calculate_derived_quantities` | Element-wise sqrt, division, multiply | MODERATE -- eliminates 5 intermediate arrays | LOW -- already vectorized |
| `PchipInterpolator.__call__` | Piecewise polynomial evaluation | HIGH -- fuses search+evaluate into single kernel | HIGH -- vectorize over elements |
| `energy_to_wavelength` | Scalar division | LOW -- too simple to benefit | N/A |
| `parse_formula` | Regex string parsing | NONE -- not numerical | N/A |
| Multi-material batch | Loop over formulas | MODERATE | HIGH -- `vmap` eliminates loop |

## Decision

**Adopt JAX as the primary computation backend, with NumPy retained at I/O boundaries.**

Specifically:
1. All numerical computation in `calculators/`, `optimization/`, and `data_handling/` will use `jax.numpy` via a backend abstraction layer.
2. The `backend/` module will provide a `Protocol`-based `ArrayBackend` interface with `NumpyBackend` and `JaxBackend` implementations.
3. `NumpyBackend` will be the default during migration (zero behavior change), switchable to `JaxBackend` via `set_backend("jax")` or environment variable.
4. After Phase 2, `JaxBackend` becomes the default with `NumpyBackend` as fallback.
5. I/O operations (file loading, CSV export, pandas DataFrames) remain NumPy/pandas.
6. The `PchipInterpolator` dependency is abstracted behind `InterpolationFactory`, allowing scipy and interpax implementations to coexist.

## Consequences

### Positive
- **Performance:** JIT-compiled scattering factor calculation should achieve 2-10x speedup for energy sweeps >100 points by eliminating Python dispatch and fusing operations.
- **Simplification:** The entire `optimization/vectorized_core.py` module (700 lines of manual SIMD heuristics, AVX detection, contiguity checks) becomes unnecessary -- JAX's XLA compiler handles all of this automatically.
- **Future capability:** `jax.grad` enables automatic differentiation through the full calculation pipeline, unlocking GPU-accelerated NLSQ fitting and Bayesian inference via NumPyro.
- **Batch processing:** `jax.vmap` replaces `ThreadPoolExecutor`-based parallel processing in `batch_processing.py` with hardware-vectorized batch computation.

### Negative
- **Cold start:** JIT compilation adds 100ms-2s to the first calculation call. Mitigated by AOT compilation at import time for hot-path functions.
- **Debugging complexity:** JIT-compiled functions produce opaque XLA error messages. Mitigated by `jax.disable_jit()` context manager for development.
- **Dependency weight:** JAX + jaxlib adds ~200MB to the install footprint. Mitigated by making JAX optional (`pip install xraylabtool[jax]`).
- **Learning curve:** Team members need to understand JAX's functional purity constraints (no in-place mutation, no Python control flow in JIT).

### Risks
- **interpax maturity:** The PCHIP interpolation adapter depends on interpax for JIT-compatible interpolation. If interpax proves inadequate, fallback to `jax.pure_callback` wrapping scipy (with JIT boundary penalty).
- **Numerical equivalence:** JAX's XLA compiler may reorder floating-point operations, producing results that differ at the ~1e-15 level. The golden test suite with 1e-12 tolerance provides a safety net.

---

## Appendix: Functions to JIT-Compile (Priority Order)

1. `calculate_scattering_factors()` -- Hot path, called once per material per calculation
2. `calculate_derived_quantities()` -- Called immediately after scattering factors
3. `vectorized_interpolation_batch()` -- Inner loop of multi-element processing
4. `vectorized_multi_material_batch()` -- Outer batch loop (candidate for vmap)
5. `EnergyConfig.to_array()` -- Trivial but called frequently

## Appendix: Functions that MUST NOT be JIT-compiled

1. `load_scattering_factor_data()` -- File I/O, side effects (caching)
2. `create_scattering_factor_interpolators()` -- Object creation, side effects (LRU cache)
3. `parse_formula()` -- Regex string processing
4. `_warm_priority_cache()` -- Threading, side effects
5. Any function that raises `ValueError`/`FileNotFoundError` -- JAX JIT traces through exceptions
