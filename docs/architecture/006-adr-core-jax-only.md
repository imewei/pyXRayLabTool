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
physics and signal helpers** (PR 2, `xraylabtool/_physics.py`) are JAX-first.
`convert` and `bragg` enable the persistent compile cache like `calc`, and the
GUI warms the core programs (and the two conversion programs of the Single tab)
on an idle worker (`gui.services.warm_up_core`), with the compute services under
`runtime.worker_guard()` and a `QThreadPool` agreement test.

Host by design, stated plainly so that "JAX-first" is not read as "every number
is computed on the device":

- spec section 9: `find_peaks`, `progress_bar`, `save_processed_data`,
  `parse_formula`;
- the scalar helpers of `constants.py` (`critical_angle_degrees`,
  `attenuation_length_cm`, `energy_to_wavelength_angstrom`,
  `wavelength_angstrom_to_energy`): plan step 2.2 ruled that they stay host
  functions (`math` on Python floats). Spec section 10 lists
  `critical_angle_degrees` among the `jnp` utilities and section 9 does not list
  these helpers; the discrepancy is recorded here, the plan ruling stands. The
  device equivalents are `calculators.optics.critical_angle_from_delta` and
  `attenuation_length_from_beta`;
- the three-point fit of the polynomial background (next paragraph).

A first click on the Single tab's Convert fields before the idle warm-up has
imported JAX runs the conversion on the UI thread and pays the `import jax` and
the compile (about 1 s) once; afterwards a conversion costs about 125 microseconds.

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
0.446 ms measured with the device program (median of 21 fresh processes; about
5.4x, wider than the 3x the design spike suggested), and on GPU 0.59 to 0.61 ms
(launch latency). The gates P1 (at most 0.5 ms) and P5 (at most 1.5 ms) are met,
not waived; they bind, and P1 has 11 % headroom. The memoised per-formula
encoding (commit `3a8abcc`) landed during Task 3B as an optimisation for P3 and
P6, not as the spec's risk-1 mitigation (P1 did not miss). At 20 000 energies
NumPy remains faster on CPU (1.63 ms against 3.56 ms measured, 2.3 to 4.5 ms
across processes; gate P2 at most 6 ms). In exchange, batches are about 1.8x
faster than the frozen NumPy row on CPU (P3 11.3 ms against 19.8 ms, P4 111 ms
against 198 ms) and the same code runs on GPU.

## Gate evidence

Authoritative measurement of Task 5B on 2026-10-27, HEAD `bacf45c` plus the
`core_gpu_check` commits, machine: Linux laptop, RTX 4090 Laptop GPU (16 GB),
JAX 0.12.0.1, x64. The unrelated Julia jobs that loaded the host earlier in the
work had finished; every run started after the 1-minute load average had fallen
below 2.0 (CPU core rows 0.78, 1.99, 1.79; CLI rows 1.04, 1.05; GPU rows 0.86 to
0.97) with the GPU idle (0 % utilisation, 15 MiB). The load average printed by
each script run (start and end) is in `task-5b-report.md`; the end values are
higher because the run itself loads the host. CPU rows run with
`JAX_PLATFORMS=cpu` (so the "cpu-host" rows use the CPU device even on this CUDA
machine); GPU rows use the default device. Every number is the median of fresh
interpreter processes (no persistent compile cache), 21 processes for the CPU
core rows and for the GPU timings of `core_gpu_check.py`, 14 runs per CLI
command. P3 and P4 are relative to the frozen NumPy row recorded in the spec's
"Task 0 baselines" (19.8 ms, 198 ms), because the NumPy path was deleted and the
row can no longer be re-measured.

| # | Gate | Spec value | Measured (median, range over processes) | Status |
|---|---|---|---|---|
| P1 | SiO2 x200 E, CPU | at most 0.5 ms | 0.446 ms (0.416 to 1.11) | met, 11 % headroom |
| P2 | SiO2 x20 000 E, CPU | at most 6 ms | 3.56 ms (2.31 to 4.46) | met |
| P3 | 200 formulas x200 E, CPU | at most 19.8 ms | 11.3 ms (9.4 to 17.7) | met |
| P4 | 2000 formulas x200 E, CPU | at most 198 ms | 111 ms (93.7 to 128) | met |
| P5 | SiO2 x200 E, GPU | at most 1.5 ms | 0.607 ms (0.594 to 0.618) | met |
| P6 | 200 formulas x200 E, GPU | at most 5 ms | **7.29 ms (6.03 to 9.90)** | **MISS by 2.3 ms (46 %)**, see below |
| P7 | SiO2 x20 000 E, GPU | at most 2 ms | 1.70 ms (1.30 to 3.24) | met; thin, single processes reach 3.2 ms |
| P8 | first call incl. compile, CPU | at most 1.0 s | 0.289 s (0.279 to 0.307) | met |
| P9 | CLI `calc`, cold / warm persistent cache, CPU host | at most 2.0 s | 0.75 s / 0.61 s | met |
| P9 | CLI `convert`, cold / warm, CPU host | at most 2.0 s | 0.51 to 0.52 s / 0.47 to 0.48 s | met |
| P9 | CLI `bragg`, cold / warm, CPU host | at most 2.0 s | 0.50 to 0.52 s / 0.48 s | met |
| P9 | CLI `calc`, cold / warm, GPU host | at most 3.3 s | 1.29 s / 1.01 s | met |
| P9 | CLI `convert`, cold / warm, GPU host | at most 3.3 s | 1.02 to 1.05 s / 0.96 to 0.98 s | met |
| P9 | CLI `bragg`, cold / warm, GPU host | at most 3.3 s | 1.03 to 1.08 s / 0.97 to 1.00 s | met |
| P10 | `--version`, `--help`, `formula`, `list`, `completion` | at most 0.9 s, no JAX | 0.10 to 0.15 s on both hosts; `import xraylabtool` 15 to 16 ms, no JAX import | met |
| P10 | `atomic` (mendeleev import, not JAX) | at most 1.3 s | 0.91 s on both hosts (0.90 to 0.95) | met |
| P11 | transfers per device program call | at most 1 upload, 1 fetch | GPU, guard live: single 1/1, from factors 1/1, batch of 200 (one block) 1/1, mixed-K batch (three blocks) 3/3; CPU asserted by the counters in tests | met |
| P12 | compiles after warm-up, same buckets | 0 | 0 (CPU 21 processes, GPU 42) | met |
| P13 | compiles for a new energy count in a bucket | 0 | 0 (200 to 201, 201 to 216; CPU and GPU) | met |

The goldens (`rtol=1e-12`) pass on the GPU for all 24 families (worst relative
error 2.0e-14, `utils_signal`; the core families stay below 3.6e-15), and the
four padding-invariance tests are bit-exact on the GPU.

Also measured: first-call compiles 1 (single material) and 2 (batch); GPU first
call 0.8 s (single) and 1.4 s (batch of 200) in a fresh process.

**P6 is a miss and is recorded as a deviation, not relaxed.** `core_gpu_check.py`
(three runs of 7 fresh processes, medians 6.92, 7.45 and 7.33 ms) and
`core_bench.py --device gpu` (two runs of 7, medians 6.89 and 6.65 ms) agree on
6.6 to 7.5 ms. The earlier "best of 5.50 ms" was a best single reading; the
quiet host does not bring the median under 5 ms. A first version of the script
that kept a CUDA context resident in the parent process while the children were
timed read 5.3 to 5.5 ms (21 processes, minimum 4.4 ms) and a first-call time
200 ms shorter; with the parent holding the GPU the driver and clocks stay up
(the laptop GPU idles at 210 MHz, performance state P8), so that figure does not model a fresh user
process and the script now times first. The spread across processes is wide
(6.0 to 9.9 ms), and one warm call is mostly host work around a short device
program. No optimisation was applied in this task. The two options on record are
views of one per-item copy for the nine result rows (changes the "fresh arrays"
semantics of public results) and merging small `K` groups into one block (changes
the grouping design); a decision on either, or on relaxing the gate with this
evidence, belongs to Wei.

## Consequences

- **Cold start grows for compute commands.** `import jax` costs about 0.7 s and
  the first program compiles; the persistent cache (CLI and GUI only, never
  the library) brings `calc` back to the P9 figures above.
- **One implementation, one set of tests.** The goldens (frozen from the NumPy
  path in Task 0, `rtol=1e-12`, never loosened; worst measured relative error
  4.4e-15 for the core families on CPU, 3.6e-15 on GPU, 2.0e-14 for `utils_signal`
  on GPU) replace the NumPy/JAX parity tests.
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
  `((wavelength^2 * (SCATTERING_FACTOR * rho)) * f_sum) / mw` (the molecular
  weight divides last, R25), so no subnormal intermediate is formed.
- `calculate_derived_quantities` now accepts plain lists (the NumPy-era base
  raised `TypeError`).
- Standalone `calculate_scattering_factors` with a tiny molecular weight
  returns the finite value (the NumPy-era base returned `inf`).
- P3 and P4 bind against the frozen NumPy numbers of the spec's Task 0
  baselines (19.8 ms, 198 ms).
- The spec's "cpu-host" definition is the CPU device: CPU rows run with
  `JAX_PLATFORMS=cpu`, so a CUDA machine's cpu-host row is the CPU device.

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
