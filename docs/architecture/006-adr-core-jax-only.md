# ADR-006: The Core Calculators Are JAX-Only

**Status:** ACCEPTED (core calculators, resident tables, batch, `utils` helpers, CLI and GUI wiring; see "Scope and what follows")
**Date:** 2026-10-27
**Deciders:** Wei Chen
**Supersedes:** ADR-001 (JAX vs NumPy computation backend). ADR-003 (dtype policy) applies unchanged; the transfer policy of ADR-004 is unchanged (its flow diagram was updated to the shipped design).

---

## Context

ADR-005 made `xraylabtool.kk` one `jax.numpy` implementation. The rest of the
package was still NumPy-first: the core calculators ran on NumPy unless an
NVIDIA GPU was present, the JAX backend was an optional twin
(`NumpyBackend`, `JaxBackend`, `ops`), interpolation was one SciPy or JAX
object per element, and batch runs forked process pools.

Measured on 2026-10-26 (RTX 4090 laptop, CUDA 13, 20 cores, median of 5 fresh
processes, `scripts/core_bench.py`, commit `4c3e2e9`; times in ms):

| Path | SiO2 x200 E | SiO2 x20 000 E | 200 formulas x200 E | 2000 formulas x200 E |
|---|---:|---:|---:|---:|
| NumPy backend, CPU | 0.083 | 1.63 | 19.8 | 198 |
| JAX backend (then), CPU | 0.816 | 5.71 | 184 | 1898 |
| JAX backend (then), GPU | 3.27 | 3.98 | 728 | 7289 |

The old JAX backend was 10x slower than NumPy on CPU for single calls and
batches and 40x slower on GPU, and its first batch in a fresh process cost 1.2
to 4.0 s. The causes were structural, not the physics: shape-keyed eager
compilation (68 programs in the first call, 16 more per new energy count),
about 60 tiny dispatches and 6 blocking syncs per call, ten separate
downloads, and a batch that was the single-material path in a Python loop.
ADR-001's premise (NumPy is faster on CPU for small arrays) was therefore true
of the old JAX glue, not of JAX.

## Decision

**The core calculators are one implementation in `jax.numpy` float64.** No
NumPy twin of any algorithm, no `ops` dispatch, no SciPy PCHIP at runtime, no
`pure_callback`, no float32. Wei's directive of 2026-10-26: the whole package
is JAX-first and this is mandatory.

1. **One device program per public call.** Host: parse and validate the
   formula and inputs (the trust boundary), compute the per-material scalars
   (molecular weight, electron count) as Python floats in today's summation
   order, and encode `z_idx` (int32) and `counts` (float64). One
   `runtime.upload`, one compiled program, one `runtime.fetch` of
   `{"out": (F, 9, M), "electron_density": (F,), "ok": (F,)}`. The device-only
   region runs under `jax.transfer_guard("disallow_explicit")` in tests and
   the budgets (at most 1 upload and 1 fetch per device program call: per
   public call for single materials, per `F`-block for batches) are asserted
   from the `runtime` counters.
2. **Resident scattering tables.**
   `xraylabtool/data/scattering_tables.npz` holds `x, f1, f2` and the PCHIP
   slopes `d1, d2` for the 92 elements as `[92, 449]` float64 (1.7 MB), built by
   `scripts/gen_scattering_tables.py` from the `.nff` files (which stay the
   source of truth) with the package's own `_pchip_slopes`. `load_tables()`
   reads and uploads them once under a lock. They are a program *argument*,
   never a closed-over constant, so they are not in the HLO or the compile-cache
   key. `scripts/gen_scattering_tables.py --check` runs in CI; a test pins every
   row's slopes to SciPy `PchipInterpolator` at 1e-12.
3. **Pure public device API.** `xraylabtool.calculators.optics` exports
   `optical_properties`, `OpticalProps`, `interpolate_f`, `derive`,
   `encode_formula` and the standalone primitives
   `critical_angle_from_delta`, `attenuation_length_from_beta`, `transmission`.
   `optical_properties` is `jit`/`vmap`/`grad`-compatible (x64 precondition);
   `d(sum dispersion)/d(rho)` equals `sum dispersion / rho` to 1e-12.
   `xraylabtool.calculators.tables` exports `ScatteringTables`, `load_tables`.
4. **Shape buckets and shared runtime.** `backend/runtime.py`
   (`ensure_x64`, `enable_persistent_cache`, `worker_guard`, `upload`,
   `fetch`, `fetch_status`, counters) and `backend/shapes.py` (`bucket`,
   `bucket_pow2`, `block_for`) are promoted from `kk` and shared; `kk`
   re-exports them. Energies `M` use `bucket(m, minimum=64)`, elements `K` and
   materials `F` use powers of two. Padded rows are `z=0, counts=0, mw=1,
   rho=1`; padded energies repeat the last real energy. Padding invariance is
   tested bit-exact. Importing `runtime` or `shapes` has no side effect and
   `import xraylabtool` does not import JAX.
5. **Batch is `vmap` over materials.**
   `calculate_xray_properties`, `calculate_multiple_xray_properties`,
   `data_handling.batch_processing.calculate_batch_properties` and the
   comparator group materials by `K` bucket and run `F`-blocks through the
   same program. Output order equals input order (the energy sort and restore
   are gone: the program is order-independent). `ProcessPoolExecutor`,
   `ThreadPoolExecutor` and `multiprocessing` are removed from `xraylabtool/`
   (fork after JAX has started threads can deadlock, and `filterwarnings =
   error` turns the JAX fork `RuntimeWarning` into failures); an AST test
   keeps them out. `BatchConfig.max_workers` and the parallel switches are
   accepted and ignored.
6. **Removed names.** `ops`, `NumpyBackend`, `JaxBackend`, `ArrayBackend`,
   `InterpolationFactory`, `calculators.kernels` (`backend/array_ops.py` is
   deleted). `set_backend("jax")` returns `None`; any other name raises
   `ValueError("the NumPy backend was removed; see ADR-006")`;
   `get_backend()` returns `"jax"`. `calculate_scattering_factors(...,
   element_data)` accepts `TableInterpolator` entries only (an arbitrary
   callable raises `TypeError`).
7. **Explicit warm-up.** There is no hidden first-call warm-up.
   `xraylabtool.warm_up(formulas=("SiO2",))` is public and explicit; the first
   public call after `clear_scattering_factor_cache()` only loads the resident
   tables and compiles the program of its own buckets.
8. **CLI.** `calc`, `batch`, `compare` and `kk` enable the persistent compile
   cache (`runtime.enable_persistent_cache()`; `XRAYLABTOOL_NO_JAX_CACHE=1`
   opts out); `--help`, `--version`, `completion`, `formula`, `atomic` and
   `list` never import JAX. Per-command subprocess tests enforce both.

## Exemptions (host code by design)

| Code | Why it stays on the host |
|---|---|
| `.nff` reading, npz loading, CSV/Parquet/JSON/pandas export, `io/*` | file I/O and containers |
| `parse_formula` and the `mendeleev` lookup | string parsing, third-party tables |
| `validation/*` and input checks in `core.py` | the trust boundary, run before upload |
| `utils.find_peaks`, `progress_bar`, `save_processed_data` | data-dependent output length, tty, file I/O |
| `analysis/comparator.py` statistics and report text, `interfaces/cli/formatting.py` | reporting on already-downloaded results |
| GUI widgets, pyqtgraph/Matplotlib plotting, `gui/services.py` glue | view layer, takes NumPy |
| `XRayResult` and `KKResult` containers | frozen host containers, NumPy fields |

Everything else that produces physics output runs in `jax.numpy`.

## Scope and what follows

The **core calculators, the resident tables and batch** (PR 1) and the **`utils`
physics and signal helpers** (PR 2, `xraylabtool/_physics.py`) are JAX-first, so
the statement "the entire package is JAX-first" holds for every function that
produces physics output. `convert` and `bragg` enable the persistent compile
cache like `calc`, and the GUI warms the core programs (and the two conversion
programs of the Single tab) on an idle worker (`gui.services.warm_up_core`), with
the compute services under `runtime.worker_guard()` and a `QThreadPool`
agreement test. The host exemptions of the spec (section 9) stay host:
`find_peaks`, `progress_bar`, `save_processed_data`, `parse_formula`.

One deliberate host step inside a utility: the three-point quadratic fit of
`background_subtraction(method="polynomial")` is `numpy.polyfit` on the host.
The frozen golden contains LAPACK's rounding noise in that fit (the residual at
the end points is a few ulp, not zero), and no `jax.numpy` solver reproduces it
(exact rational coefficients differ from NumPy's in the last bits too); the
Horner evaluation and the subtraction run on the device, with an optimisation
barrier so XLA does not fuse the multiply and add into an FMA.

## Accepted: single-call latency relative to NumPy

The mandate trades the NumPy row's small-call latency for one implementation.
Accepted by Wei on 2026-10-26 (spec C15). A single small call on CPU is slower
than the removed NumPy path (frozen Task 0 row: 0.083 ms for SiO2 x200 E):
0.42 to 0.45 ms measured with the device program (about 5x, wider than the 3x
the design spike suggested), and on GPU 0.60 to 0.71 ms (launch latency). The
gates P1 (at most 0.5 ms) and P5 (at most 1.5 ms) are met, not waived; they
bind. The memoised per-formula encoding (commit `3a8abcc`) landed during
Task 3B as an optimisation for P3 and P6, not as the spec's risk-1 mitigation
(P1 did not miss). At 20 000 energies NumPy remains faster on CPU
(1.63 ms against 2.0 to 5.4 ms measured, gate P2 at most 6 ms). In exchange,
batches are 1.5x to 2.3x faster than the frozen NumPy row on CPU (provisional, see
below) and the same code runs on GPU.

## Gate evidence (to be filled by Task 5B on a quiet host)

**PROVISIONAL.** Every number below was measured on a loaded machine (load
average 4 to 20 from unrelated Julia jobs) by the task reports
`task-2c-report.md` and `task-3b-report.md`, CPU runs with `JAX_PLATFORMS=cpu`,
GPU runs on the RTX 4090 laptop. Task 5B replaces this table with the output of
`scripts/core_gpu_check.py` on a quiet host. P3 and P4 are relative to the
frozen NumPy row recorded in the spec's "Task 0 baselines" (19.8 ms, 198 ms),
because the NumPy path was deleted and the row can no longer be re-measured.

| # | Gate | Spec value | Provisional measurement | Status |
|---|---|---|---|---|
| P1 | SiO2 x200 E, CPU | at most 0.5 ms | 0.42 ms (2C), 0.45 ms (3B) | met |
| P2 | SiO2 x20 000 E, CPU | at most 6 ms | 5.3 ms (2C, before `dc70f2a`), 2.0 ms (3B, load 2) | met; thin headroom in the 2C reading, re-measure |
| P3 | 200 formulas x200 E, CPU | at most 19.8 ms | 8.79 ms best (10.2, 12.7 other runs) | met |
| P4 | 2000 formulas x200 E, CPU | at most 198 ms | 99.3 ms best (110, 133 other runs) | met |
| P5 | SiO2 x200 E, GPU | at most 1.5 ms | 0.60 to 0.71 ms | met |
| P6 | 200 formulas x200 E, GPU | at most 5 ms | **5.50 ms best**, 7.6 to 8.9 ms at load 7 to 9 | **open: miss by 0.5 ms under load** |
| P7 | SiO2 x20 000 E, GPU | at most 2 ms | 1.1 to 1.5 ms | met |
| P8 | first call incl. compile, CPU | at most 1.0 s | 0.27 s | met |
| P9 | CLI `calc`, `convert`, `bragg`, cold / warm persistent cache | CPU host at most 2.0 s, GPU host at most 3.3 s | CPU 1.03 s / 0.90 s, GPU 1.73 s / 1.36 s | met for `calc`; `convert` and `bragg` (warm cache, median of 9 fresh processes): CPU 0.59 s / 0.59 s, GPU 1.06 s / 1.06 s; cold cache CPU 0.86 s / 1.05 s, GPU 1.70 s / 1.47 s (machine loaded by other jobs, provisional) | met |
| P10 | `--version`, `--help`, `formula`, `list`, `completion` | at most 0.9 s, no JAX | 0.18 to 0.21 s on both hosts | met |
| P10 | `atomic` (mendeleev import, not JAX) | at most 1.3 s | 0.97 s when quiet, 1.32 s at load 4 | **open: over the gate under load** |
| P11 | transfers per device program call | at most 1 upload, 1 fetch | asserted from the counters in tests | met |
| P12 | compiles after warm-up, same buckets | 0 | 0 (CPU, GPU) | met |
| P13 | compiles for a new energy count in a bucket | 0 | 0 (200 to 201, 201 to 216; CPU, GPU) | met |

Also measured: `import xraylabtool` 18 ms with no JAX import; first-call
compiles 1 (single material) and 2 (batch). Open items to close in 5B, stated
plainly: P6 reads above its gate under load (best 5.50 ms); P10 `atomic` is
over its gate under load; P2 had thin headroom in the earlier reading; P9 is met for
`convert` and `bragg` too. If a quiet re-measurement still misses
P6, the options are views of one per-item copy for the nine result rows (changes
the "fresh arrays" semantics of public results) or merging small `K` groups into
one block (changes the grouping design); neither is applied.

## Consequences

- **Cold start grows for compute commands.** `import jax` costs about 0.7 s and
  the first program compiles; the persistent cache (CLI and GUI only, never
  the library) brings `calc` back to the P9 figures above.
- **One implementation, one set of tests.** The goldens (frozen from the NumPy
  path in Task 0, `rtol=1e-12`, never loosened; worst measured relative error
  4.4e-15) replace the NumPy/JAX parity tests.
  `scripts/gen_core_golden.py` and `scripts/gen_kk_prejax_baseline.py` need the
  deleted NumPy path; each carries a note naming the commit to check out.
- **Break for downstream code.** `set_backend("numpy")`, `ops`,
  `NumpyBackend`, `JaxBackend`, `InterpolationFactory` and `calculators.kernels`
  are gone and raise or fail to import. The changelog marks it; suggested
  release 0.5.0 (no version bump in this work).
- **Benchmarks.** `scripts/core_bench.py` (CPU) and `scripts/core_gpu_check.py`
  (manual, CUDA) are standalone gates, not collected by pytest. The Task 0
  "cpu-host" rows ran on the CPU through the NumPy backend and are not
  like-for-like with the later row that pins JAX to the CPU
  (`JAX_PLATFORMS=cpu`).

### Amendments to the design spec, recorded here

- `_status` (the device-side `ok` flag) also requires
  `isfinite(SCATTERING_FACTOR * rho / mw)`, which keeps the legacy failure for
  tiny molecular weights. `derive` and the programs compute
  `(wavelength^2 * SCATTERING_FACTOR * rho) * (f_sum / mw)`, so no subnormal
  intermediate is formed.
- `calculate_derived_quantities` now accepts plain lists (the NumPy-era base
  raised `TypeError`).
- Standalone `calculate_scattering_factors` with a tiny molecular weight
  returns the finite value (the NumPy-era base returned `inf`).
- P3 and P4 bind against the frozen NumPy numbers of the spec's Task 0
  baselines (19.8 ms, 198 ms).

### Known limitations

- A molecular weight above about 4.5e307 g/mol (for example `"H1" + "0" * 308`)
  gives `delta = beta = 0` with an accepted status: XLA rewrites `x / m` as
  `x * (1 / m)` and `1 / m` is flushed as a subnormal. It is physically
  unreachable and parked; a status guard (fail when `1 / mw` is subnormal) is
  the fix if wanted.
- The molecular weight is divided out last (`((λ²·S·ρ)·f)/M`), so the product
  `(λ²·S·ρ)·f` overflows for a factor sum of about 1e300 or more with a large
  density (for example `"H1" + "0" * 305` at `rho = 1e4`); the status check then
  raises the non-finite `ValueError` rather than returning a zero. Non-physical
  and accepted; the `f / M`-first order underflowed at the opposite extreme.
- A single material whose own padded `K x M` exceeds the block budget (for
  example 9 distinct elements at roughly 300k energies, or 65 at 40k)
  runs as one `F = 1` block, up to about 1.3x over the planning bound. The budget
  is a conservative bound on `F`; splitting one material would need a second
  program call per public call, which the transfer contract (one upload and one
  fetch per call) forbids.
- A device exception (for example out-of-memory) aborts the whole
  `calculate_xray_properties` or batch call. Skip-and-log is per formula only
  (spec section 6); the `F`-block budget makes device failures unlikely.
- `calculate_scattering_factors(..., element_data)` no longer accepts arbitrary
  callables.
