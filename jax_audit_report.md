# JAX/Vectorization Audit Report
## pyXRayLabTool — 2026-04-06

---

## Executive Summary

The codebase has a well-designed backend abstraction (`NumpyBackend` / `JaxBackend` via `ops` proxy in `backend/array_ops.py`) but **the JAX backend is never activated by default** — `_backend = NumpyBackend()` at line 166. All production paths use NumPy. The interpolation layer (`backend/interpolation.py`) correctly switches to `interpax.PchipInterpolator` when `JaxBackend` is active, but the per-call interpolator invocations inside the hot loop are not JIT-compiled. The result: the infrastructure for JAX exists but delivers zero GPU benefit in practice.

---

## File-by-File Findings

### 1. `backend/array_ops.py`

| # | Function / Location | Issue | Recommendation | Speedup | Difficulty | Priority |
|---|---|---|---|---|---|---|
| A1 | `_backend = NumpyBackend()` (line 166) | JAX backend never activated by default | Call `set_backend("jax")` at startup (or via env var `XRAYLABTOOL_BACKEND=jax`) | Unlocks all downstream gains | Easy | **P0** |
| A2 | `JaxBackend.any()` (line 150) | `bool(jnp.any(x))` is a host-device transfer on every validation call | Acceptable at I/O boundary; annotate clearly, not a hot path | N/A | N/A | Low |
| A3 | `JaxBackend` — no JIT-compiled operation kernels | Backend methods are bare `jnp.*` calls with no `@jax.jit` | Create a `@jax.jit`-compiled kernel module imported once; delegate backend methods to it | 2–5× (eliminates per-call trace overhead) | Medium | P1 |

---

### 2. `calculators/core.py` — Primary Optimization Target

#### 2a. `calculate_scattering_factors` (lines 870–991) — **Hottest function**

| # | Location | Issue | Recommendation | Speedup | Difficulty | Priority |
|---|---|---|---|---|---|---|
| B1 | Lines 940–962 (multi-element path) | Python `for` loop over elements to build `f1_rows` / `f2_rows`, then `np.stack` | Replace with `jax.vmap` over elements once interpolators return JAX arrays; or pre-stack all element tables into a `(n_elements, n_energy_grid)` array and do a single batched lookup | 3–10× for multi-element materials | Medium | **P0** |
| B2 | Lines 950–951: `import numpy as _np; np.stack(f1_rows)` | Explicit NumPy import inside hot function; host-device roundtrip if JAX backend active | Use `ops.asarray(jnp.stack(...))` or restructure to avoid host numpy entirely | 1.5–2× | Easy | P1 |
| B3 | Lines 969–989 (single-element path) | `float(count)` and scalar multiplication outside JAX — minor but unnecessary host-side scalar | Minor; keep as is unless profiling shows it matters | <1.1× | Easy | P3 |
| B4 | `calculate_scattering_factors` as a whole | Not decorated with `@jax.jit` | Wrap in `jax.jit` after ensuring all inputs are static or traceable; use `functools.partial` to freeze static args | 2–4× (eliminates re-tracing) | Medium | **P0** |

#### 2b. `calculate_derived_quantities` (lines 994–1084)

| # | Location | Issue | Recommendation | Speedup | Difficulty | Priority |
|---|---|---|---|---|---|---|
| B5 | Lines 1045–1059 | Three `np.asarray(x).item()` calls — explicit host-device transfers | Replace with `float(x)` directly on Python scalars, or pass scalars as Python floats from caller; never call `.item()` on JAX arrays | 1.1–1.3× | Easy | P1 |
| B6 | Line 1030–1035 | `ops.any(ops.isnan(...))` / `ops.any(ops.isinf(...))` validation — 4 separate device calls | Combine into a single `jax.jit`-compiled validation kernel that checks all conditions in one pass | 1.2–1.5× | Easy | P2 |
| B7 | Line 1078 | `wavelength**2` — Python `**` operator on JAX array forces Python dispatch | Use `jnp.square(wavelength)` via `ops.square()` (already available) | <1.1× | Easy | P3 |
| B8 | `calculate_derived_quantities` as a whole | Pure arithmetic on arrays — ideal `@jax.jit` candidate | Decorate with `@jax.jit`; scalar outputs (`electron_density`) can be returned as 0-d JAX arrays, converted at the output boundary | 2–4× | Easy | **P0** |

#### 2c. `_calculate_molecular_properties` (lines 1208–1225)

| # | Location | Issue | Recommendation | Speedup | Difficulty | Priority |
|---|---|---|---|---|---|---|
| B9 | Lines 1217–1223 | Python `for` loop over elements accumulating scalars | Pre-vectorize: load `atomic_number` and `atomic_mass` into arrays once, then use `jnp.dot(counts, masses)` and `jnp.dot(counts, atomic_numbers)` | 1.5–3× for formulas with many elements | Easy | P2 |

#### 2d. `create_scattering_factor_interpolators` (lines 1087–1163)

| # | Location | Issue | Recommendation | Speedup | Difficulty | Priority |
|---|---|---|---|---|---|---|
| B10 | Lines 1147–1152 | Energy sort check: `ops.any(ops.asarray(energy_values[:-1]) > ...)` — device transfer just to validate pre-sorted data | Move to a one-time assert at data load time; atomic data is always pre-sorted | 1.1× per cold start | Easy | P3 |
| B11 | `interpax.PchipInterpolator` (via `InterpolationFactory`) | `interpax` interpolators are JAX-native and JIT-compatible, but calling them in a Python loop prevents XLA from fusing the interpolation with downstream ops | Compose interpolation + `calculate_scattering_factors` into a single `@jax.jit` region; or batch all element interpolations into one `jax.vmap` call | 2–5× | Hard | P1 |

#### 2e. `XRayResult.__post_init__` (lines 178–193) and output boundary (lines 1580–1606)

| # | Location | Issue | Recommendation | Speedup | Difficulty | Priority |
|---|---|---|---|---|---|---|
| B12 | Lines 184–193 | 10× `np.asarray(field)` calls on every result — forces host copy of JAX arrays | Convert once at the very end (after all calculations) using a single `jax.device_get` call; or return JAX arrays directly and convert lazily | 1.2–1.5× | Medium | P2 |
| B13 | Lines 1580–1605 | 10× `np.ascontiguousarray(..., dtype=np.float64)` — redundant if arrays are already contiguous float64 JAX arrays | Check dtype/layout once and skip conversion if already correct | 1.1× | Easy | P3 |

#### 2f. `calculate_multiple_xray_properties` / `calculate_xray_properties` (lines 1339–1886)

| # | Location | Issue | Recommendation | Speedup | Difficulty | Priority |
|---|---|---|---|---|---|---|
| B14 | Lines 1379–1411 | Python `for` loop calling `calculate_single_material_properties` per material | Use `jax.vmap` over materials (requires homogeneous energy grids); for heterogeneous cases use `jax.lax.map` | 5–20× for large batches | Hard | P1 |
| B15 | Lines 1740–1756 | `ThreadPoolExecutor` with NumPy — GIL-limited; not GPU-parallelized | Replace with JAX vmap batch path; retire threading for the JAX backend | 2–10× | Hard | P1 |

---

### 3. `optimization/vectorized_core.py`

This module is already deprecated (it advertises `set_backend('jax')` as the replacement). The code within it is NumPy-based manual SIMD heuristics. No new JAX work is needed here — the priority is to wire up the actual JAX backend.

| # | Location | Issue | Recommendation | Speedup | Difficulty | Priority |
|---|---|---|---|---|---|---|
| C1 | `vectorized_interpolation_batch` (lines 59–129) | Builds Python list of rows then stacks — same pattern as `calculate_scattering_factors` | Already deprecated; ensure callers are routed to the JAX backend path | — | — | Remove |

---

### 4. `optimization/optimized_core.py`

Manual data-loading optimization using `np.loadtxt`. Superseded by JAX path but still the active data loader. One actionable item:

| # | Location | Issue | Recommendation | Speedup | Difficulty | Priority |
|---|---|---|---|---|---|---|
| D1 | `load_scattering_factor_data_optimized` (lines 74–150) | Loads .nff file to NumPy, then downstream code wraps in `jnp.asarray` — double allocation | Load directly to JAX arrays when `JaxBackend` is active; check `isinstance(_backend, JaxBackend)` at load time | 1.2× | Easy | P2 |

---

### 5. `data_handling/batch_processing.py`

| # | Location | Issue | Recommendation | Speedup | Difficulty | Priority |
|---|---|---|---|---|---|---|
| E1 | `process_batch_chunk` (lines 154–206) | `ThreadPoolExecutor` with per-material Python calls — no GPU utilization | Accumulate all materials and dispatch as a single JAX vmap batch | 10–50× for large batches | Hard | P1 |
| E2 | `_prepare_result_data` (lines 339–357) | Python `for` loop indexing numpy arrays element-by-element (`result.energy_kev[index]`) | Use vectorized pandas/numpy operations; or convert after computation, not per-row | 2–5× on export | Easy | P2 |
| E3 | `_prepare_energies_array` (lines 217–235) | `np.any(energies_array <= 0)` — redundant if JAX backend validates | Keep at input boundary; acceptable | — | — | Low |

---

## scipy Usage

| Location | scipy call | JAX replacement |
|---|---|---|
| `backend/interpolation.py` line 15 | `scipy.interpolate.PchipInterpolator` | `interpax.PchipInterpolator` (already wired for JAX backend) |
| No other scipy calls found in hot paths | — | — |

The `interpax` replacement is **already in place** — it just needs the JAX backend to be activated.

---

## Dynamic Shapes / Recompilation Risks

| Location | Risk | Mitigation |
|---|---|---|
| `calculate_scattering_factors` — `n_elements` Python loop | Each unique `n_elements` triggers a new JAX trace | Pad to fixed max elements (e.g., 10) with a mask array |
| Energy array length varies per call | Different `len(energy_ev)` = different trace | Use static shapes or `jax.jit(... static_argnums=...)` for shape |
| `calculate_derived_quantities` — `if ops.any(...)` validation | Python `if` on JAX value = concrete evaluation + recompile | Move guards outside JIT region |

---

## Top 3 Highest-Impact Opportunities (Priority Ranking)

### #1 — Activate JAX Backend by Default + JIT `calculate_scattering_factors` + `calculate_derived_quantities`
- **Files:** `backend/array_ops.py:166`, `calculators/core.py:870`, `calculators/core.py:994`
- **What:** Set `_backend = JaxBackend()` (or env-var toggle), then decorate `calculate_scattering_factors` and `calculate_derived_quantities` with `@jax.jit`. Fix the embedded `import numpy as _np; np.stack(...)` at line 950–951 to use `jnp.stack` so the entire computation stays on-device.
- **Estimated speedup:** 3–8× for a single material at 1000 energy points on CPU (XLA kernel fusion); 10–50× on GPU.
- **Difficulty:** Medium (main risk: `.item()` and `np.asarray()` calls must be moved to output boundary first — see B5, B12).

### #2 — Replace Per-Material Python Loop with `jax.vmap` in `calculate_xray_properties`
- **Files:** `calculators/core.py:1379`, `calculators/core.py:1711`
- **What:** When all materials share the same energy grid (the common case), batch the entire `(material → dispersion/absorption/f1/f2)` computation into a single `jax.vmap` call over the material axis. This vectorizes across materials on the GPU in one kernel launch, replacing both the Python `for` loop and the `ThreadPoolExecutor`.
- **Estimated speedup:** 5–20× for batches of 10+ materials.
- **Difficulty:** Hard (requires homogeneous element count across materials OR a padded/masked representation).

### #3 — Batch Interpolation with `jax.vmap` over Elements in `calculate_scattering_factors`
- **Files:** `calculators/core.py:940–962`
- **What:** Replace the Python `for count, f1_interp, f2_interp in element_data` loop with `jax.vmap(lambda interp, e: interp(e))(stacked_params, energy_ev)`. Pre-build a stacked parameter array for all elements once (at interpolator construction time) so the per-energy interpolation is a single vectorized kernel.
- **Estimated speedup:** 3–10× for multi-element compounds (SiO2, Al2O3, etc.).
- **Difficulty:** Medium (requires restructuring `interpax` interpolators to accept batched parameters).

---

## Implementation Roadmap

```
Phase 1 (unblocked, Easy):
  - Fix B5: remove .item() calls in calculate_derived_quantities
  - Fix B12/B13: consolidate np.ascontiguousarray at output boundary
  - Fix A1: add XRAYLABTOOL_BACKEND env-var to activate JaxBackend

Phase 2 (Medium, depends on Phase 1):
  - Fix B2: remove embedded numpy import in hot path
  - Fix B4/B8: @jax.jit on calculate_scattering_factors and calculate_derived_quantities
  - Fix D1: load .nff data directly to JAX arrays

Phase 3 (Hard, depends on Phase 2):
  - Fix B11/B14: jax.vmap over elements and over materials
  - Fix E1: replace ThreadPoolExecutor with JAX vmap in batch_processing
```
