# ADR-005: The Kramers-Kronig Package Is JAX-Only

**Status:** ACCEPTED
**Date:** 2026-10-26
**Deciders:** Wei Chen
**Supersedes:** The "NumPy is the default backend" part of ADR-001, for `xraylabtool.kk` only

---

## Context

`xraylabtool.kk` (Kramers-Kronig f1 from tabulated f2) first shipped as a NumPy
implementation with a JAX twin selected through `set_backend()`, following
ADR-001. ADR-001 rests on the premise that NumPy is the faster choice for the
small arrays of a CPU-only user. For this kernel that premise does not hold:
the closed-form sum over 1201 segments (SiO2) at m energies is a dense
broadcast that XLA fuses and threads, while NumPy materialises every
temporary.

Two implementations also meant two sets of rounding, two sets of tests and a
device boundary crossed on every chunk.

## Decision

`xraylabtool.kk` is one implementation, written with `jax.numpy`. There is no
NumPy twin of any algorithm and no `ops`-proxy dispatch inside `kk`.

1. **Device-resident pipeline.** One upload of all inputs, one status fetch
   after assembly and near-edge merge, one status fetch per refinement pass and
   one final download. The device-only region runs under
   `jax.transfer_guard("disallow_explicit")` in tests; the three boundary
   helpers (`upload`, `fetch_status`, `fetch`) in `xraylabtool.kk._device` are the only crossings and count
   themselves (see ADR-004 for the general policy).
2. **Padded `Spectrum` pytree.** Segment tables are padded to a capacity
   bucket with `x_n * (1 + k)` knots and all-zero coefficients, which add
   exactly nothing to the transform. The true count `n_segments` is a traced
   int32 leaf, so spectra of different sizes but equal capacity can be
   stacked and `vmap`-ed. `Spectrum` and `kk_f1` are public and
   `jit`/`vmap`/`grad`-compatible.
3. **Shape buckets.** Every compiled entry sees only bucketed shapes
   (a geometric ladder with ratio at most 1.25 for segment capacities; for
   energy counts a power of two on CPU, the fine bucket ladder in equal blocks
   on accelerators), so the set of compiled programs is small. A
   default refined `SiO2` run compiles five top-level programs
   (`_assemble`, `_evaluate`, `_first_pass`, `_refine_pass`, `_finalize`); a
   second call with other values and the same buckets compiles none.
4. **Status pytrees, not exceptions, on the device.** Capacity overflow,
   near-edge merge problems and refinement bookkeeping are returned as small
   status pytrees, fetched once and turned into the established exceptions and
   log messages on the host.
5. **float64 always.** x64 is established at every concrete public entry and
   every worker entry, not only at import.
6. **Persistent compile cache for the CLI and GUI only.** They call
   `xraylabtool.kk._device.enable_persistent_cache()`; the library does so only when `XRAYLABTOOL_JAX_CACHE=1` is set.
   `XRAYLABTOOL_NO_JAX_CACHE=1` opts out. The GUI also warms, on an idle
   worker, the programs of its own default Compute click (the public
   `kk_calculate_real` with the tab's explicit grid; `kk.warm_up()` does the
   same for the default grid); the CLI never warms up.

## Evidence

The ADR-001 assumption, refuted for this kernel. CPU only (20-core machine, no
GPU in the measuring session), `SiO2`, host wrapper (upload, kernel,
download), median of 15 warm calls, milliseconds. The NumPy column is the
historical measurement of the removed implementation (commit `1bf56d9`,
recorded in `scripts/kk_bench.py`); the JAX column was measured with
`scripts/kk_bench.py --platform cpu` on 2026-10-26 on a shared machine whose
load average was between 1 and 7, so read the figures to within roughly 30 %.

| m (energies) | NumPy (removed) | JAX on CPU | JAX faster by |
|---:|---:|---:|---:|
| 100 | 17.7 | 3.5 | 5x |
| 1 000 | 157 | 16.0 | 9.8x |
| 10 000 | 1 892 | 139 | 13.6x |
| 100 000 | 14 746 | 1 381 | 10.7x |

`kk_calculate_real("SiO2", refine_tolerance=0.01)` on CPU (1383 points, 267
refined):

| | NumPy (removed) | JAX |
|---|---:|---:|
| cold, fresh interpreter | 0.29 s | 1.31-1.46 s (three runs) |
| cold, persistent cache filled | - | 0.58-0.65 s (two runs) |
| warm, median | 0.273 s | 0.154-0.158 s (min 0.096-0.119 s) |
| first `kk_calculate_real` after import, 1 000 energies | - | 0.50 s |

On the RTX 4090 laptop GPU the design measurement (2026-10-25) gave the JAX
kernel at 2.2 / 3.2 / 16 / 107 ms for m = 100 / 1 000 / 10 000 / 100 000
against 11 / 145 / 1 555 / 16 649 ms for NumPy. Those GPU figures are the
spec's, not repeated here; `scripts/kk_gpu_check.py` reruns them on the
target machine.

Accuracy is unchanged: the upstream goldens and an independent longdouble
quadrature oracle (`tests/unit/kk/test_oracle.py`) pass at tolerances that
were measured, not loosened.

## Consequences

- **Cold start grows.** A cold refined `SiO2` run took 0.29 s with NumPy and now
  takes 1.31-1.46 s when it must compile, and 0.58-0.65 s with the persistent
  cache filled. The persistent cache is therefore on by default in the CLI and
  GUI, with the compile-time threshold set to 0 because JAX's default of 1 s
  would skip nearly every kk program.
- **`set_backend()` has no effect on `kk`.** `set_backend("numpy")` is accepted
  and ignored by `kk` (no warning, no error); the rest of the package keeps its
  backend selection. `KKResult.backend` is always `"jax"` from `kk`; the legacy
  value `"numpy"` is still accepted when constructing a `KKResult`.
- `ensure_float64` is removed from `xraylabtool.backend.array_ops`; `kk`
  establishes x64 itself.
- Tests that were parametrised over NumPy and JAX collapse to JAX. The
  goldens and the oracle replace the parity comparison.
- **JAX floor is `>=0.12.0.1`.** The golden tolerances were measured on this
  release; releases 0.8.0 to 0.12.0 differ in the last digits of fused XLA
  expressions and miss some of them by a few units in the twelfth digit. The
  tolerances were not loosened; the dependency floor was raised instead.

### Known deviations from the acceptance gates

Measured on 2026-10-26 on a shared 20-core CPU and an RTX 4090 laptop GPU
(times in ms; min / median where both are given). These gates were not met and
are accepted:

- **CPU refined run (`SiO2`, `refine_tolerance=0.01`) is 10-20 % slower in
  steady state than the forced-JAX pipeline at `1bf56d9`.** Median at HEAD
  128.5 / 151.3 / 132.5 / 143.9 against 119.4 / 117.6 / 124.5 / 122.5 (four
  interleaved rounds). The gate "refined warm <= 0.10 s" is **not met**: the
  median is about 0.125-0.155 s on this machine (minimum 0.096-0.119 s).
- **GPU fixed overhead at small m** (device-side assembly plus the status
  fetch), minimum ms, `1bf56d9` -> HEAD: m = 10^3 2.5 -> 4.1; m = 3100
  5.2 -> 7.0; m = 10^4 13.0 -> 15.8 (the kernel gate of 20 ms is met);
  m = 10^5 108.9 -> 115.1.
- **Bare-library GPU refined cold run is 3.7-4.4 s against the 2.5 s gate**
  (`1bf56d9` itself took 3.5 s). With the persistent cache primed it takes
  0.54 s, which is what the CLI and GUI get.

The cause is the design itself: the spectrum is assembled and merged on the
device, so the inputs go up once and the result comes down once, at the price
of a status fetch and a few more small programs than the host-assembled
pipeline; cold compiles on a GPU are slower than on a CPU.

### x64 requirement

All kk entries need 64-bit mode and set `jax_enable_x64` globally when they are
called. A surrounding thread-local `jax.enable_x64(False)` context is not
overridden: inside it the call raises `TypeError` naming `jax_enable_x64`.
This is a documented limitation.

## Alternatives considered

- **Extend `ArrayBackend` so one `ops`-written path also runs on NumPy.**
  Rejected: loops (`lax.map`, compaction), `searchsorted` and
  `unique(size=)` have no common-denominator form, and the measurements show
  NumPy has no remaining advantage.
- **Keep the NumPy twin as the CPU default.** Rejected: slower at every size
  measured, and a second implementation to keep in step with the goldens.
- **A single `while_loop` for refinement.** Rejected in favour of
  host-orchestrated, device-resident passes with one status fetch each; the
  loop's data-dependent output size does not fit a static shape.

## References

- Design: `docs/superpowers/specs/2026-10-25-kk-jax-first-design.md`
- ADR-001 (backend), ADR-003 (dtype policy), ADR-004 (transfers)
- `docs/guides/kk_guide.rst`, `docs/architecture/jax_architecture.rst`
