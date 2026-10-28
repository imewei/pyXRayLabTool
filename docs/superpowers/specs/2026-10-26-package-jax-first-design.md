# Package JAX-first — design

Date: 2026-10-26. Status: draft for execution (no questions round: Wei's
directive was "JAX-first is mandatory; go straight to a design spec and an
implementation plan"). Every choice below that Wei did not state is listed under
"Calls made by me". Builds on `2026-10-25-kk-jax-first-design.md` (kk is already
JAX-only; this spec extends the same architecture to the rest of the package).

## Intent

`xraylabtool.kk` is one device-resident `jax.numpy` implementation. The rest of
the package is NumPy-first: the core calculators run on NumPy unless an NVIDIA GPU
is present, the JAX backend is an optional twin (`NumpyBackend`/`JaxBackend`/`ops`),
interpolation is one SciPy or JAX object per element, and batch runs fork process
pools. Wei's directive: the **entire package is JAX-first, mandatory**. The target
is the kk model everywhere: one implementation in `jax.numpy` float64, one device
program per public call, one upload and one download, NumPy only at the I/O, GUI and
reporting edges.

## Evidence (measured 2026-10-26, this machine: RTX 4090 Laptop, CUDA 13, 20 cores; CPU runs use `JAX_PLATFORMS=cpu`)

Method: each row is the median of 5 fresh-process runs of a scratch script (`scripts/core_bench.py`
in Task 0 replaces it); "warm" means after one warm-up call; the machine was idle. An earlier
pass of this section was taken while a 41-minute test run was loading the machine and mixed
first-use compilation with steady state; it is replaced by the table below.

| Path | first call | SiO2 ×200 E, warm | SiO2 ×20 000 E, warm | 200 formulas ×200 E, warm | 2000 formulas ×200 E, warm |
|---|---|---|---|---|---|
| NumPy backend (CPU-host default) | 2 ms | 0.09 ms | 1.84 ms | 17.9 ms | 202 ms |
| JAX backend today, CPU | 1398 ms | 0.86 ms | 4.86 ms | 176 ms | 1828 ms |
| JAX backend today, GPU | 2025 ms | 3.89 ms | 4.52 ms | 759 ms | 8088 ms |
| Spike (design), CPU | 174 ms | 0.263 ms | 3.81 ms | 12.9 ms | 165 ms |
| Spike (design), GPU | 238 ms | 0.622 ms | 0.833 ms | 1.63 ms | 10.2 ms |

Today's JAX backend is therefore 9× (single call) to 10× (batches) slower than NumPy on CPU
and 40× slower on GPU, and its first batch in a fresh process costs 1.2 to 4.0 s. Root causes
(measured by `cProfile`, stage timers and `jax.log_compiles`, CPU):

1. **Shape-keyed eager compilation.** The first call compiles 68 programs (1.5 s in XLA); every
   new energy-array length compiles 16 more (0.45 to 0.54 s) and every new table length about
   18 (0.4 s), because the glue (`ops.*`, slope construction) runs op by op, not in one jit.
2. **Host/device ping-pong per call.** The XLA kernel is 24 µs of a 896 µs call (2.6%). The rest
   is about 60 tiny dispatches and 6 blocking syncs: validation with JAX scalar ops and
   `np.any` on JAX arrays (259 µs, 29%), `ops.asarray(list_of_device_arrays)` row stacking
   (247 µs, 28%), four interpolator calls each uploading the energies (134 µs, 15%), the host
   `isfinite` guard plus eager tail (145 µs, 16%), ten separate downloads in `XRayResult` (38 µs).
3. **Batch is the single path in a Python loop** (no `vmap`), so every formula repeats cause 2.
4. **Structure:** JAX hides behind an `ops` layer built for NumPy semantics, and a host sync
   splits the pipeline into two kernels, so XLA never sees the whole computation.

Throwaway spike (not product code; stacked tables, vmap'd PCHIP with one shared interval search
for f1 and f2, one jit program, one `device_put`, one fetch). Parity with today's
`scattering_factor_f1/f2`: max relative error 2.5e-16 (f1), 3.2e-16 (f2) on CPU, 0.0 on GPU. The
spike times include `parse_formula`, `device_put` of the inputs and the fetch; K=2 elements for
the single material, K=4 for the batches.

CLI cold start (median of 5, `uv run xraylabtool …`): `calc SiO2 -e 10 -d 2.2` 0.86 s on a CPU
host and 3.30 s on this GPU host; `--version` 0.87 s and 1.27 s. On a GPU host today's backend
auto-select imports JAX and initializes CUDA even for `--version`.

Other facts the design rests on: 92 element tables (93 files: `C.nff` and `c.nff` both exist) of 433 to 449 points each, 29.3 to 30 000 eV,
so the five padded arrays (x, f1, f2 and two slope arrays) are 1.7 MB in float64; `import jax`
costs 0.73 s and the first tiny jit 0.22 s; `ProcessPoolExecutor` is used only in
`data_handling/batch_processing.py`; 21 `set_backend(` call sites in 4 test files.

## Task 0 baselines (`scripts/core_bench.py`, measured 2026-10-26, idle machine)

Median of 5 fresh processes per cell; commit `4c3e2e9` plus the Task 0 files (no change under
`xraylabtool/`). Inputs: SiO2 on `linspace(1, 29, 200)` and `linspace(0.05, 29.95, 20000)` keV;
batches call `calculate_xray_properties` with a fixed pool of ten 2- to 4-element formulas (mean
2.4 elements) cycled over 200 and 2000 entries, 200 energies. "first" = first call in the process
(tables, compilation); "warm" = median per call afterwards (ms).

| Path | first SiO2 ×200 E | SiO2 ×200 E, warm | SiO2 ×20 000 E, warm | 200 formulas, warm | 2000 formulas, warm |
|---|---|---|---|---|---|
| NumPy backend, CPU | 1.95 | 0.083 | 1.63 | 19.8 | 198 |
| JAX backend today, CPU | 1420 | 0.816 | 5.71 | 184 | 1898 |
| JAX backend today, GPU | 1823 | 3.27 | 3.98 | 728 | 7289 |

First call of the other cases (ms): NumPy 3.85 / 29.6 / 219; JAX CPU 1277 / 2213 / 3952; JAX GPU
1888 / 3309 / 9813. Compiles in the first call today: 68 (single), 92 (batches). P12 (repeat, same
shape): 0. P13 (new energy count, 200 to 201 and 201 to 250): 16 each. A new material with the
same element count compiles 2 programs. Table lengths are not varied by the bench.

CLI and import cold start (median of 5, `.venv/bin/xraylabtool`, so without the `uv run` wrapper
of the earlier measurement; `cpu-host` hides `nvidia-smi` from `PATH`, which made the NumPy-era backend auto-select pick NumPy on the CPU; it did not pin JAX to the CPU, so these rows are not like-for-like with later rows that set `JAX_PLATFORMS=cpu`). Every worker and cold CLI
run is explicitly uncached (JAX cache variables removed, `XRAYLABTOOL_NO_JAX_CACHE=1`, fresh
`XDG_CACHE_HOME`); a nonzero exit aborts the bench:

| Host | `import xraylabtool` | calc | convert | bragg | --version | --help | formula | atomic | list | completion |
|---|---|---|---|---|---|---|---|---|---|---|
| cpu-host (s) | 0.283, no JAX | 0.357 | 0.347 | 0.338 | 0.361 | 0.350 | 0.344 | 1.141 | 0.349 | 0.338 |
| gpu-host (s) | 0.759, JAX imported | 2.829 | 1.004 | 1.007 | 0.996 | 1.009 | 1.008 | 1.859 | 0.988 | 0.995 |

P9 warm persistent cache (a dedicated cache directory populated by one run, then reused): cpu-host
calc/convert/bragg 0.344/0.337/0.346 s, gpu-host 2.838/1.006/0.982 s. They equal the cold rows
today because `calc`, `convert` and `bragg` do not enable the cache yet.

Gate revisions (the rest of the table is confirmed unchanged by these baselines):

- **P3** and **P4** are relative: "at most the NumPy row measured by `scripts/core_bench.py` on the
  same host" (200 formulas: 19.8 ms measured; 2000 formulas: 198 ms measured). The 17.9 ms in the
  Evidence table came from an ad-hoc scratch pool, not the committed one; the Anti-goal uses the
  same relative form.
- **P10** `atomic` ≤ 1.3 s, the other commands stay ≤ 0.9 s: `atomic` imports `mendeleev`
  (1.14 s on the cpu-host today), which is not JAX and is out of scope here.
- P9 and P10 numbers above are for the console script; the earlier `uv run` figures (0.86 s, 3.30 s)
  include the wrapper, so compare future runs with this table, not with those.

## Goals

1. **One implementation.** All physics in `jax.numpy` float64. No NumPy twin of any
   algorithm, no `ops` dispatch, no SciPy PCHIP at runtime, no `pure_callback`.
2. **One device program per public call** (spec §1, §4) with a single upload group
   and a single fetch group, a device-side status flag instead of mid-pipeline syncs,
   and bucketed shapes so compilation is bounded (§3).
3. **Batch is `vmap` over materials** inside the same program; the process and
   thread pools are gone (§7).
4. **A public pure device function** (`optical_properties`, §5) that works under
   `jit`, `vmap` and `grad`, the analogue of `kk_f1`.
5. **Parity.** Results equal today's NumPy-path results to `rtol=1e-12` (goldens
   frozen from the NumPy path before it is deleted, §Verification); exception types
   and messages are unchanged.
6. **Public API continuity.** Public function and class names, signatures and result
   types are unchanged except the removals listed in §8.
7. **Import discipline.** `import xraylabtool` stays JAX-free; JAX is imported and
   x64 enabled at the first compute entry; CLI and GUI enable the persistent compile
   cache for compute paths (§10); the library does so only with
   `XRAYLABTOOL_JAX_CACHE=1`.
8. **Measured gates** (§Acceptance gates) with documented deviations, as in kk.
9. **A written exemption list** (§9): everything that stays on the host, with the
   reason.

## Non-goals

- No change to the physics, constants, data files, units, the 0.03 to 30 keV
  validation range or `XRayResult`'s fields and legacy accessors.
- No float32 or mixed precision; no `pmap`/sharding; no multi-GPU.
- No rewrite of the GUI, plotting, CLI argument handling, pandas export or file I/O
  into JAX (exemptions, §9).
- No version bump or release in this work (a bump touches about 10 places;
  `scripts/check_version.py`); the changelog states the break for the release.
- Not forcing a GPU: the program runs on JAX's default device, CPU or GPU.

## Anti-goals (a failure even if every test passes)

- A silent NumPy fallback or any second implementation of an algorithm.
- A warm call path slower than today's JAX backend, or a warm batch of 200 formulas
  slower than the NumPy row that `scripts/core_bench.py` measures on the same host (19.8 ms
  measured), or a recompile per material,
  element count or energy count (within a bucket).
- `import xraylabtool` (or `--help`, `--version`, `completion`, `formula`, `atomic`,
  `list`) importing JAX.
- `fork` after JAX has started (process pools, `multiprocessing` with fork).
- Loosening a golden tolerance, or deleting the NumPy path before its goldens exist.
- A host sync inside the device program, or float32 anywhere.

## Constraints (from the project)

Python ≥ 3.12; JAX floor `>=0.12.0.1`; `uv`; `mypy --strict`, `ruff` clean;
NumPy-style docstrings; explicit imports; float64 always (ADR-003); transfers per
ADR-004; `filterwarnings = error`; GUI tests run serially and need `libEGL`
(unit tests `importorskip("PySide6.QtWidgets")`); CI has no GPU, so every test must
pass on JAX-CPU and on a GPU host; goldens never loosened without a recorded
rationale; test names must not contain `performance`, `benchmark` or `timing` unless
intended (auto-marked by `tests/conftest.py`).

## Current state → target

| Area | Today | Target |
|---|---|---|
| `backend/array_ops.py` | `NumpyBackend`, `JaxBackend`, `ops`, `set_backend`, auto-select (GPU only) | removed; `set_backend`/`get_backend` become explicit stubs (§8) |
| `backend/interpolation.py` | `InterpolationFactory` (SciPy or JAX) | `JaxPchipInterpolator` + `_pchip_slopes` only (already on this branch) |
| `kk/_device.py`, `kk/_shapes.py` | generic runtime helpers inside kk | generic part promoted to `backend/runtime.py`, `backend/shapes.py`; kk re-imports (§3) |
| `calculators/cache.py` | per-element interpolators, Python dict caches | resident `ScatteringTables` + thin compatibility shims |
| `data_handling/atomic_cache.py`, `calculators/scattering_data.py` | complex128 `get_scattering_factors`, placeholder `AtomicScatteringFactor`, cmath `CrystalStructure` | read the resident tables; `CrystalStructure` and `AtomicScatteringFactor` moved to `jnp` (§8) |
| `constants.py`, `utils.py` conversions | NumPy/Python floats; `critical_angle_degrees` raises for ≤ 0 | pure `jnp` primitives in `_physics.py`; the raising wrappers keep their messages (§10) |
| `calculators/kernels.py` | two kernels via `ops`, lazy jit per backend | removed; replaced by `calculators/optics.py` pure functions |
| `calculators/core.py` | Python loop over elements, NumPy glue, host `isfinite` sync | host validation/encode, one program call, `XRayResult` build |
| `data_handling/batch_processing.py` | chunks, Process/Thread pools | chunks of materials, `vmap` blocks on device |
| `utils.py` physics and signal helpers | NumPy | pure `jnp` functions + host wrappers; `find_peaks` exempt |
| `validation`, `io`, CLI, GUI plotting, `analysis` reporting | NumPy/pandas | unchanged except they stop importing `ops` (exempt, §9) |

## Design

### 1. Boundary and transfer policy (ADR-004 applied)

Host: parse the formula (`parse_formula`, now `lru_cache`d), validate inputs (the
trust boundary: types, finiteness, 0.03 to 30 keV, positive density), compute the
per-material scalars `molecular_weight` and `number_of_electrons` as Python floats
exactly as today (same summation order, so they stay bit-identical); `electron_density`
is computed on the device from `rho`, `mw`, `n_electrons` with today's expression
(`1e6·ρ/mw·N_A·n_e/1e30`) and operation order (it must be a function of `rho` for `grad`, §5), so it is covered by the
`rtol=1e-12` goldens rather than being bit-identical,
and encode `z_idx` (int32) and `counts` (float64). One `runtime.upload` of the tuple
`(z_idx[F,K], counts[F,K], mw[F], rho[F], n_electrons[F], energies_keV[M])`. One compiled program.
One `runtime.fetch` of `{"out": (F, 9, M), "electron_density": (F,), "ok": (F,)}`. Host slices off padding and
builds `XRayResult` (rows of the one downloaded buffer; the energy array is a fresh
copy, never an alias of the caller's data). The device-only region runs under
`jax.transfer_guard("disallow_explicit")` in tests; budgets are asserted from the
`runtime` counters: ≤ 1 upload call and ≤ 1 fetch call per **device program call**, which
is one per public call for single-material functions and one per `F`-block for batches
(§3, §7).

### 2. Resident scattering tables and the shipped data

`xraylabtool/data/scattering_tables.npz` holds `x, f1, f2, d1, d2` as `[Z, N]`
float64 (`Z` = 92 elements, `N` = 449, the longest `.nff` table, derived by the
generator), `n[Z]` int32 and `symbols[Z]` in atomic-number order. The data directory
holds 93 files because `C.nff` and `c.nff` both exist; the generator applies the same
rule as `scattering_data._AVAILABLE_ELEMENTS` (capitalise, dedupe) and fails if the two
files differ in content. `scripts/gen_scattering_tables.py` builds the npz from the
`.nff` files (which stay the source of truth) using the package's own `_pchip_slopes`,
so there is one slope implementation. Padding past `n[z]` uses
strictly increasing dummy knots `x_n·(1+k)`, values equal to the last value, slopes
0, so a padded row evaluates exactly like the unpadded one on `[x[0], x[n-1]]`.
`ScatteringTables` is a registered pytree (`x, f1, f2, d1, d2, n`). `load_tables()`
reads the npz once, uploads once, caches under a lock (GUI worker threads), and the
tables are a program **argument**, never a closed-over constant (no 1.7 MB in the
HLO or the compile-cache key). `clear_scattering_factor_cache()` drops them. A test
regenerates the npz and requires equality with the shipped file; another pins every
row's slopes to SciPy `PchipInterpolator` at 1e-12.

### 3. Shape buckets and shared runtime

Importing `backend.runtime` or `backend.shapes` has **no side effect**: no `ensure_x64()`
call, no cache enabling, no device query; `backend/__init__.py` imports neither of them
(JAX stays out of `import xraylabtool`). The `kk` modules keep their own import-time
`ensure_x64()` (kk is lazy and JAX-only, ADR-005); core entries call it at first use.

Promote the generic parts of `kk/_device.py` (`ensure_x64`, `enable_persistent_cache`,
`_user_cache_dir`, `worker_guard`, counters, `upload`, `fetch`, `fetch_status`) to
`backend/runtime.py` and `bucket`, `bucket_pow2`, `block_for` to `backend/shapes.py`.
`kk` modules import them from there; `kk._device` keeps the names it still defines
(`Spectrum`, `validated_arrays`, `check_capacity`) and re-exports the rest so the
`kk` import paths keep working. Counters live in one module only. Buckets:
energies `M` use `bucket(m, minimum=64)` (geometric ladder, padded with the last real
energy so no NaN enters), elements `K` use `bucket_pow2(k, minimum=2)` (2 to 128;
92 elements max), materials `F` use `bucket_pow2(f, minimum=1)`. Padded material
rows are `z=0, counts=0, mw=1, rho=1` (finite, contribute nothing) and are sliced
away. Large batches run as blocks of `F_block × K × M ≤ budget` (platform-aware, as
`energy_blocks`); every block reuses the same compiled program.

### 4. The device program (`calculators/optics.py`)

Pure functions, x64 as a precondition:

- `interpolate_f(tables, z, counts, e_ev)`: gather the rows `tables.x[z]` etc. into
  `(K, N)`, `vmap` over elements a PCHIP that does **one** `searchsorted` per
  `(row, query)` and evaluates f1 and f2 together (Hermite basis shared), NaN outside
  `[x[0], x[n-1]]`, then a left fold over the K elements with `counts` (`lax.scan`), not an
  `einsum`, so padding stays bit-exact.
- `derive(energy_kev, f1_sum, f2_sum, rho, mw, n_electrons)`: wavelength, δ, β from the
  count-weighted f1/f2 totals (the `rho/mw` scattering prefactor needs `rho` and `mw`), then critical angle, attenuation
  length, real and imaginary SLD; the formulas are copied verbatim from
  `_derived_quantities_kernel` and `_properties_from_totals` (including the `1e-30`
  β clamp, the `max(2δ, 0)` critical-angle rule, and `ENERGY_TO_WAVELENGTH_FACTOR`).
- `optical_properties(tables, z, counts, rho, mw, n_electrons, energy_kev)`: the two
  above for one material, with `electron_density = 1e6·ρ/mw·N_A·n_electrons/1e30` (electrons/Å³, the
  exact expression and operation order of `kernels.calculate_derived_quantities`,
  including the `/1e30`) computed inside; `jax.vmap` over `(z, counts, rho, mw, n_electrons)` is the batch. Returns the
  tuple `(OpticalProps, ok)`: `OpticalProps` is the pytree of the nine per-energy
  arrays (the `XRayResult` per-energy fields plus `electron_density` as a scalar leaf)
  and `ok` a scalar bool.
- Status: `ok = all(isfinite(δ)) & all(isfinite(β)) & isfinite(SCATTERING_FACTOR * rho / mw)`
  per material, returned with the result in the same fetch. Amended in Task 3 (R17,
  superseded by R25): `derive` and the programs form δ and β as
  `((λ² · (SCATTERING_FACTOR · ρ)) · f_sum) / mw`
  (the molecular weight divides last, so no subnormal intermediate appears), and the
  extra `isfinite` term keeps the legacy failure for a tiny molecular weight, where
  `SCATTERING_FACTOR · ρ / mw` overflowed.

**Standalone primitives are separate.** The public helpers `calculate_critical_angle`,
`calculate_attenuation_length` and `calculate_transmission` keep their own semantics,
which differ from the in-program `derive` rules: today `calculate_critical_angle(-1e-6)` returns `[nan]`
(δ < 0) and `calculate_attenuation_length(β=0)` returns `[inf]`, whereas `derive` clamps
β at `1e-30` and applies `max(2δ, 0)`. They are therefore separate small `jnp`
functions in `calculators/optics.py` (`critical_angle_from_delta`,
`attenuation_length_from_beta`, `transmission`), not calls into `derive`; goldens pin
both families, including the NaN/inf edges. `calculate_derived_quantities` is **not** one of
these: today it checks finiteness (same `ValueError`) and calls the clamped kernel
(negative δ gives critical angle 0, β=0 gives a finite attenuation length), so it runs
the `derive` rules; its own golden test pins that contract independently of the two
unclamped helpers. `constants.critical_angle_degrees` keeps its
`ValueError` for δ ≤ 0 in its host wrapper (§10).

`calculate_properties_from_factors` (used by `kk.KKResult.to_xray_result`) runs the
same `derive` through a second small program that takes supplied f1/f2; one tail.

### 5. Public pure device API

`xraylabtool.calculators.optics` exports `ScatteringTables`, `load_tables`,
`encode_formula(formula) -> (z, counts, mw, n_electrons)` (host, NumPy/Python floats),
`OpticalProps` and `optical_properties`. `optical_properties` is `jit`/`vmap`/`grad`
-compatible when its array arguments are device arrays and `tables` comes from
`load_tables()`; it requires x64 (precondition, as in kk). Contract for `grad`: with
the other arguments fixed, `jax.grad(lambda rho: optical_properties(tables, z, counts,
rho, mw, ne, e)[0].dispersion.sum())(rho)` equals `dispersion.sum()/rho` to `1e-12`
(δ is linear in ρ; the NaN-outside-range branch is not differentiated). Gradients
w.r.t. energy are not part of the contract (PCHIP `searchsorted` is piecewise). The `XRayResult` functions are host wrappers over them.
`xraylabtool.calculators` re-exports them lazily (no JAX import on `import
xraylabtool`).

### 6. Status and error behaviour

Single material: input validation raises exactly today's exceptions and messages
(`FormulaError`, `EnergyError`, `ValueError("Mass density must be positive")`); a
failed status raises `ValueError("Non-finite values (NaN or Inf) detected in
dispersion or absorption coefficients")` after the fetch. Batch and
`calculate_xray_properties`: formulas are screened on the host (parse, density), the
valid ones are batched, a per-material failed status is logged with
`_logger.warning("Failed to process formula", ...)` and skipped, and
`RuntimeError("Failed to process any formulas successfully")` keeps its condition.
`calculate_multiple_xray_properties` keeps returning `{}` (not raising) when every
formula fails, and its result keys follow `unique_result_key` unchanged.
`calculate_xray_properties` no longer sorts and restores energy order: the program
is order-independent, so the output follows the input order and equals today's.

### 7. Batch (`data_handling/batch_processing.py`)

`calculate_batch_properties` groups materials by `K` bucket, runs `F`-blocks through
the program, and keeps the progress callback per block. Contracts preserved from
today: `process_single_calculation` still warns and returns `None` for a failing item;
`_process_chunks` result keys stay `f"{formula}@{density:.3f}"` with the `#N` suffix for
repeats, in input order; `chunk_iterator`, `save_batch_results` and `load_batch_input`
are untouched. Each `F`-block costs one upload and one fetch (P11 is per block). `BatchConfig.chunk_size` is
the upper bound of materials per block; `max_workers` and the parallel switches are
accepted and ignored (logged once at DEBUG) so existing configs keep working.
Batch inputs follow one rule: each item is screened on the host before its row is
appended, so a bad item never shifts the others. A per-item density that is a
non-scalar sequence or array (`ndim > 0`) is rejected like any non-number:
`calculate_xray_properties` raises the up-front `ValueError`,
`calculate_multiple_xray_properties` skips and logs the item, and
`calculate_batch_properties` warns and returns `None` for it. A 0-d array density is
converted with `float()` and accepted by the last two, but `calculate_xray_properties`'
up-front check rejects it too. The three entry points take one shared energies argument
(a 2-D list raises `EnergyError`); per-item energies exist only in the chunks of
`process_batch_chunk`, which groups them by raw `(shape, dtype, bytes)` so items with
different energy inputs run in their own group and follow their own energies.
`calculate_derived_quantities` accepts plain lists (the NumPy-era code raised `TypeError`).
`MemoryMonitor` keeps checking host result memory; CSV/Parquet loading and saving stay
on the host. `ProcessPoolExecutor` and `ThreadPoolExecutor` are removed (fork after
JAX threads exist can deadlock, and `filterwarnings = error` turns the JAX
`RuntimeWarning` into failures; CI runs Python 3.12 and 3.13 where Linux still forks).

### 8. Removed and kept public names

Removed: `ops`, `NumpyBackend`, `JaxBackend`, `ArrayBackend`, `InterpolationFactory`,
`calculators.kernels` (`_scattering_math_kernel*`, `_get_*_kernel`, `_jit_cache`).
`set_backend(name)` accepts `"jax"` silently and raises
`ValueError("the NumPy backend was removed; see ADR-006")` for anything else;
`get_backend()` returns the string `"jax"`. Kept with new bodies: the
`calculators.cache` names (`create_scattering_factor_interpolators` returns
`TableInterpolator` callables that read a row of the resident tables;
`get_cached_elements`, `is_element_cached`, `clear_scattering_factor_cache`,
`get_bulk_atomic_data`), `calculate_scattering_factors` and
`calculate_derived_quantities` (host arrays in, NumPy out, one program call). Both are
defined today in `calculators/kernels.py`; they move to `calculators/derived_quantities.py`
(re-exported unchanged from `calculators/__init__.py` and `core.py`) when `kernels.py` is deleted.
`calculate_scattering_factors(..., element_data)` accepts `TableInterpolator`
entries only; an arbitrary callable raises `TypeError` (documented). `KKResult.backend`
stays `"jax"`. Also in scope, because they use `ops` or NumPy physics today:
`data_handling.atomic_cache.get_scattering_factors` (still returns `complex128` NumPy,
filled from the resident tables), `scattering_data.AtomicScatteringFactor` (the
placeholder lookup) and `CrystalStructure.calculate_structure_factor` (`cmath` →
`jnp`, same results), and the `calculators/core.py` re-exports of the removed kernel
names, which disappear with a grep of tests and docs. ADR-006 supersedes ADR-001; ADR-003 and ADR-004 apply unchanged.

### 9. Exemptions (host code by design)

| Code | Why it stays on the host |
|---|---|
| `.nff` reading, npz loading, CSV/Parquet/JSON/pandas export, `io/*` | file I/O and containers |
| `parse_formula` and the `mendeleev` lookup | string parsing, third-party tables |
| `validation/*` and input checks in `core.py` | the trust boundary, run before upload |
| `utils.find_peaks`, `progress_bar`, `save_processed_data` | data-dependent output length (cannot be jitted), tty/progress, file I/O |
| `analysis/comparator.py` statistics and report text, `interfaces/cli/formatting.py` | reporting on already-downloaded results; its numbers come from one batched call |
| GUI widgets, pyqtgraph/Matplotlib plotting, `gui/services.py` glue | view layer, takes NumPy |
| `XRayResult` and `KKResult` containers | frozen host containers, NumPy fields |

Everything else that produces physics output runs in `jax.numpy`.

### 10. Utilities, import cost, CLI, GUI

`utils` physics (`energy_to_wavelength`, `wavelength_to_energy`, `bragg_angle`,
`d_spacing_*`, `q_from_angle`, `angle_from_q`, `critical_angle_degrees`) and signal
helpers (`smooth_data`, `background_subtraction`, `normalize_intensity`) become pure
`jnp` functions in `xraylabtool/_physics.py`; the public wrappers validate and convert
units on the host, call a jitted function (unit strings static) and return a Python
float or a NumPy array as today; wrappers that raise today (`critical_angle_degrees`
for δ ≤ 0, `_check_lattice`) check on the host before the call and keep their messages. `import xraylabtool` stays JAX-free: `utils.py` imports `_physics` (which imports
`jax` at module top) **inside** each wrapper, never at module import; `calculators/`
modules import `optics`/`tables` inside the compute functions; the CLI command modules
(imported at the top of `interfaces/cli/__init__.py`) import `core`, `runtime` and
`utils` physics only inside their handlers, so `--help`, `--version`, `completion`,
`formula`, `atomic` and `list` never load JAX. `ensure_x64` and `import jax` happen at
the first compute entry, and the package `__init__` no longer
calls `_configure_jax_float64`. CLI `calc`, `batch`, `compare`, `convert`, `bragg`,
`kk` call `runtime.enable_persistent_cache()` before computing; `--help`,
`--version`, `completion`, `formula`, `atomic`, `list` never import JAX. The GUI
warms the core programs for its Compute path on an idle worker (as
`gui/services.warm_up_kk`) and its worker threads use `worker_guard`.
`xraylabtool.warm_up(formulas=("SiO2",))` is a public precompile helper.

## Verification (what "correct" means)

- **Frozen goldens first (Task 0).** Before any deletion, generate from the NumPy
  path: single, multiple, `calculate_xray_properties`, `calculate_properties_from_factors`,
  `calculate_derived_quantities`, `calculate_scattering_factors`, every `utils` physics
  and signal helper, over about 30 formulas (H2O to Au, 1 to 6 elements, repeated
  elements, parentheses, hydrates) and energy grids (scalar, 3, 200, unsorted, edges
  of 0.03 and 30 keV, at absorption edges). Stored under
  `tests/characterization/core/` with the generator and recorded provenance. New code
  must match at `rtol=1e-12` (`atol` only where the golden is 0); no tolerance is
  loosened later without a rationale in the commit message.
- **Table integrity:** regenerated npz equals shipped; row slopes equal SciPy at 1e-12;
  every row evaluated at its own nodes returns its values exactly.
- **Padding invariance** (`K`, `M`, `F` buckets): results are bit-exact between a
  tight and a padded call; block size does not change results. Where XLA fusion may differ by one ulp
  (batch vs loop, block sizes) the tests use a new generic helper
  `tests/fixtures/core_rounding.py::assert_ulp_close(actual, expected, ulps=4)`;
  `assert_kk_close` is KK-specific (its scale comes from the KK pipeline) and is not
  reused.
- **Batch equals loop:** `calculate_xray_properties` equals a loop of single calls.
- **Differentiation:** the §5 `grad` contract (sum of dispersion w.r.t. density equals
  `dispersion.sum()/rho`); `vmap` over densities equals the batch.
- **Behaviour parity:** exception types and messages, skip-and-log batch semantics,
  negative f1 near edges kept (δ < 0, critical angle 0), unsorted energies.
- **Budgets:** transfer counters (≤ 1 upload, ≤ 1 fetch), compile counts (second call
  with the same buckets compiles nothing; a default run compiles ≤ 2 programs),
  `import xraylabtool` and `--help` import no JAX (subprocess `sys.modules` check),
  no `fork`-based pool remains (an AST test over `xraylabtool/` finds no import of
  `multiprocessing` and no use of `ProcessPoolExecutor`/`ThreadPoolExecutor`).
- Existing suites (golden interpolation, characterization, integration, CLI, GUI)
  pass unchanged except the 21 `set_backend` call sites and the backend-dispatch
  tests, which are rewritten (not weakened) for the new design.

## Acceptance gates (CPU = this machine with `JAX_PLATFORMS=cpu`)

CPU rows run with `JAX_PLATFORMS=cpu`, so on a CUDA machine the cpu-host row is the CPU
device; the CLI cpu-host rows also hide `nvidia-smi` from `PATH`. GPU rows leave the
platform unset.

| # | Gate (warm unless stated) | Binding value | Spike | Today |
|---|---|---|---|---|
| P1 | SiO2 ×200 E, CPU | ≤ 0.5 ms | 0.263 ms | NumPy 0.09, JAX 0.86 |
| P2 | SiO2 ×20 000 E, CPU | ≤ 6 ms | 3.81 ms | NumPy 1.84, JAX 4.86 |
| P3 | 200 formulas ×200 E, CPU | ≤ the NumPy row of `core_bench.py` on the same host (19.8 ms measured) | 12.9 ms | NumPy 17.9 (scratch pool) / 19.8 (`core_bench.py`), JAX 176 / 184 |
| P4 | 2000 formulas ×200 E, CPU | ≤ the NumPy row of `core_bench.py` on the same host (198 ms measured) | 165 ms | NumPy 198, JAX 1898 (`core_bench.py`); 202 / 1828 with the spike's scratch pool |
| P5 | SiO2 ×200 E, GPU | ≤ 1.5 ms | 0.622 ms | JAX 3.89 |
| P6 | 200 formulas ×200 E, GPU | ≤ 5 ms | 1.63 ms | JAX 759 |
| P7 | SiO2 ×20 000 E, GPU | ≤ 2 ms | 0.833 ms | JAX 4.52 |
| P8 | first call incl. compile, no cache, CPU | ≤ 1.0 s | 174 ms | JAX 1398 ms |
| P9 | CLI `calc`, `convert`, `bragg` cold, warm persistent cache | CPU host ≤ 2.0 s; GPU host ≤ 3.3 s | | 0.86 s; 3.30 s (`calc`) |
| P10 | CLI `--version`, `--help`, `formula`, `atomic`, `list`, `completion` cold | ≤ 0.9 s on both hosts (no JAX import); `atomic` ≤ 1.3 s (mendeleev import, not JAX, revised; it read 1.86 s on the GPU host while JAX leaked into the import, 1.14 s on the CPU host) | | 0.87 s; 1.27 s GPU host (`uv run`); console script 0.34 to 0.36 s (`atomic` 1.14 s) / 0.99 to 1.01 s (`atomic` 1.86 s) |
| P11 | transfers per device program call (single material: per public call; batch: per `F`-block) | ≤ 1 upload, ≤ 1 fetch | | |
| P12 | compiles after warm-up, same buckets | 0 | | |
| P13 | compiles for a new energy count inside a bucket | 0 | | 16 |

Gate ownership by PR (spec C15): PR 1 measures P1–P9 for `calc`/`batch`/`compare`, P10
and P11–P13; PR 2 measures P9 for `convert`/`bragg` after the utilities move. ADR-006 in
PR 1 states that core calculators, resident tables and batch are JAX-first and that
utilities and the GUI warm-up follow in PR 2.

Known deviation, accepted with the mandate and to be recorded in ADR-006: a single
small call on CPU costs about 0.26 ms against NumPy's 0.09 ms (about 3×), and on GPU about
0.62 ms (launch latency); `JAX_PLATFORMS=cpu` selects the CPU device. `scripts/core_bench.py`
(CPU) and `scripts/core_gpu_check.py` (manual, CUDA) are standalone gates, not
collected by pytest. Gate values are confirmed or revised from the Task 0 baselines;
a miss is recorded as a deviation, never silently relaxed.

## Risks

1. **Small-call latency** above NumPy's: bounded by P1, documented. Mitigation if P1
   fails: cache the encoded `(z, counts, mw)` per formula and keep energies as the only
   per-call upload.
2. **Compile explosion** from `(F, K, M)` combinations: three geometric ladders,
   blocks of fixed shape, P12 and a compile-count test.
3. **Cold start** (+0.73 s `import jax`): lazy import, persistent cache, P9/P10.
4. **GUI threads:** concurrent first use of the tables and of `jit`; table cache under
   a lock, `worker_guard`, a `QThreadPool` agreement test.
5. **Downstream code using `set_backend("numpy")`/`ops`:** loud `ValueError` with the
   ADR pointer; the changelog marks the break.
6. **x64 disabled by another library:** programs and the PCHIP run inside
   `enable_x64(True)` scopes (already done in `JaxPchipInterpolator`); a test disables
   x64 around a call.
7. **Table drift** between `.nff` and npz: integrity test in the suite.
8. **`calculate_scattering_factors` with arbitrary callables** stops working
   (TypeError): internal-facing function, documented in the changelog.
9. **Scope creep in utilities:** they are the last task and independent; they can
   ship after the core without blocking it.
10. **JAX-CPU vs GPU numeric differences** at the ulp level: comparisons use the
    rounding-scale helper, not flat `rtol`, where fusion differs.
11. **Removal order:** `ops` is imported by `validation/validators.py`,
    `typing_extensions.py`, `xraylabtool/__init__.py` and `calculators/*`; deleting
    `array_ops` before those are migrated breaks `import xraylabtool`. The plan migrates
    them first (Task 2.0).
12. **Standalone helper semantics** (`calculate_critical_angle`,
    `calculate_attenuation_length`) differ from the in-program rules; they are kept as
    separate primitives with their own goldens (§4).

## Delivery order (each stage ships green)

0. Baselines, frozen goldens, bench scripts, branch (no behaviour change).
1. Foundations: shared `runtime`/`shapes`, tables npz + generator + `ScatteringTables`,
   pure `optics` functions, new tests (old path untouched).
2. Core switchover: `core.py`, `cache.py`, `derived_quantities.py`, remove kernels,
   `array_ops`, `InterpolationFactory`, rewrite the backend tests.
3. Batch and multiple: `calculate_xray_properties`, `calculate_batch_properties`,
   comparator, remove pools.
4. Utilities, validators and `typing_extensions` cleanup, CLI/GUI wiring, import and
   cold-start discipline, `warm_up`.
5. Docs (ADR-006, `CLAUDE.md` patterns 4 and 5, guides, changelogs), CI, GPU gate,
   graph update.

## Decisions confirmed by Wei (2026-10-26)

1. The entire package is JAX-first; this is mandatory.
2. No further questions or arguments: produce the design spec and the plan directly.
3. Standing: JAX floor `>=0.12.0.1`; kk stays JAX-only; goldens are never loosened
   without a recorded rationale; do not merge without being asked.

## Calls made by me (please check)

C1. **JAX-only core, no NumPy twin** (the kk precedent), instead of "JAX default with
NumPy fallback". Reason: Wei's "mandatory"; a twin doubles tests and invites drift.
C2. **Stacked resident tables with precomputed slopes in a shipped npz** instead of
per-element interpolators; cold start reads one 1.7 MB file instead of 93 `.nff` files.
C3. **Per-material scalars stay Python floats on the host** (bit-exact with today).
C4. **No energy sort/restore** in `calculate_xray_properties` (order-independent).
C5. **`set_backend("numpy")` raises** rather than being silently ignored; `get_backend()`
returns `"jax"`.
C6. **`calculate_scattering_factors` accepts table interpolators only.**
C7. **`max_workers`/pool options accepted and ignored** in `BatchConfig`.
C8. **`utils` unit conversions go through `jnp`** even for scalars, so the CLI
`convert`/`bragg` commands import JAX (about 0.7 s). Revisit only if
`convert` or `bragg` miss the P9 cold-start threshold (≤ 2.0 s CPU host, ≤ 3.3 s GPU
host); the exemption list does not include them.
C9. **Shared helpers live in `backend/runtime.py` and `backend/shapes.py`**; kk
re-exports so its import paths survive.
C10. **Gate numbers** come from a throwaway spike; Task 0 re-measures them with the
real code before they bind.
C11. **No version bump** in this work; a minor bump (0.5.0) is suggested at release.
C12. **Energy bucket ladder** is `bucket(m, minimum=64)`; `K` and `F` ladders are
powers of two. The builder may tune the ladders if P1 to P4 and P12 hold.
C14. **Review fixes (three-brain, 2026-10-26):** `electron_density` moved on-device so
`grad` works; standalone derived helpers kept separate; tables are 92 × 449; P11 is
per program call; a generic `assert_ulp_close` replaces `assert_kk_close`; the
table-integrity check runs in the test job, not the lint job.
C15. **Confirmed by Wei (2026-10-26, grill round):** C5, C6 and C8 are accepted as
written; the single-call latency relative to NumPy (CPU ≈ 0.26 ms vs 0.09 ms, GPU ≈
0.62 ms) is accepted and recorded in ADR-006 — P1 (≤ 0.5 ms) and P5 (≤ 1.5 ms) stay
binding, and the cache-the-encoding mitigation triggers only if P1 misses. Delivery: Tasks 0 to 3 and 5 ship in one PR stacked
on `feat/jax-native-pchip`, including the CLI wiring for `calc`/`batch`/`compare`/`kk`
(persistent cache, lazy handler imports, per-command import-discipline test), the public
`xraylabtool.warm_up` and the `backend` lazy-submodule/stub exports; Task 4 (`utils`
physics on `jnp`, `convert`/`bragg` wiring, GUI idle warm-up and thread test, P9 for
`convert`/`bragg`) is a follow-up PR. No version bump; 0.5.0 suggested at release.
Execution: `superpowers:subagent-driven-development`, started only when Wei says go.
C13. **Builder check skipped:** Wei opted out of the question round, so no
fresh-reader pass was run on this document before it was handed over.
