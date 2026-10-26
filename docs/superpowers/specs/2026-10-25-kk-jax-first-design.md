# kk JAX-first — design

**Date:** 2026-10-25 · **Status:** revised after cross-model review (Codex + Agy, see [Review log](#review-log)); no code written · **Supersedes:** the "host-side pieces (not JIT, by design)" section of `2026-10-24-kkcalc-integration-design.md`
**Owner's directive (Wei, verbatim):** "The whole package is JAX-first. This is the principle that you MUST obey and follow."
Wei declined the clarifying questions, so every how-it-gets-made choice below is **my call**, listed in [Calls made by me](#calls-made-by-me-please-check).

## Intent

`xraylabtool.kk` (Kramers-Kronig f2 → f1) was built "JAX-first" only at the innermost kernel. Everything around it
(compound-spectrum assembly, `evaluate_f2`, near-edge merge, adaptive refinement, orchestration) is host-side NumPy, and the
kernel is entered and left through NumPy on every call. The result is a JAX kernel inside a NumPy pipeline: transfers
on every call and on every refinement pass, a second NumPy implementation of the kernel kept alive as a "backend", and
shape-dependent recompiles that make a refined run slower than NumPy when cold.

Make kk JAX-first end to end: **one implementation, written in `jax.numpy`, device-resident from the single
upload of user inputs to the single download that builds `KKResult`** (ADR-004, applied to kk), jit/vmap/grad-able at its
functional core. NumPy stays only where the work is inherently host-side: reading text files, parsing formulas, loading the
`.npz`, validating concrete user inputs, logging, and the frozen result containers.

## Evidence (measured 2026-10-25, this machine: RTX 4090 Laptop, CUDA 13, 20-core CPU, jax 0.12.0.1, SiO2 = 1201 segments)

| Measurement | NumPy path | JAX path |
|---|---|---|
| Kernel, steady state, GPU, m = 100 / 1 000 / 10 000 / 100 000 energies | 11 / 145 / 1 555 / 16 649 ms | 2.2 / 3.2 / 16 / 107 ms (×5 / ×45 / ×96 / ×156) |
| Kernel, steady state, **JAX on CPU**, same m | 27 / 258 / 1 645 / 16 963 ms | 5.7 / 27 / 207 / 1 355 ms (**×5 – ×12.5 faster than NumPy at every size**) |
| `import jax` + x64 (CPU) / first kernel call incl. compile (m = 1000) | – | 340 ms / 217 ms (80 ms with persistent compile cache) |
| `kk_calculate_real("SiO2", refine_tolerance=0.01)`, CPU: cold / warm | 0.41 s / 0.36 s | 1.69 s / 0.07 s (cold 0.64 s with persistent cache) |
| Same, GPU: cold / warm; distinct shapes compiled | – | 3.43 s / 0.02 s; **13** |

Consequences for the design:
1. ADR-001's premise "NumPy is faster on CPU below ~5 000 points" **does not hold for this kernel**. JAX-on-CPU beats NumPy
   at all sizes measured, so "JAX always, for kk" costs CPU-only users nothing in steady state.
2. The one real cost is **cold start** (import 0.34 s + compiles), and it is dominated by *recompiling for every new array shape*.
   Static-shape discipline plus an optional persistent compile cache are therefore central, not optional.
3. Independent validation done the same day (longdouble principal-value quadrature of Henke's f1, 8 compounds): the
   kernel agrees to a median 1e-14 … 3e-9, worst ≈ 1e-4 only at 10–14 eV for Au/Pb/U (the documented rounding limit), on both
   NumPy and GPU-JAX. That quadrature becomes a permanent oracle (see Verification).

## Goals

- **G1 One implementation.** All kk numerical code is `jax.numpy`; no NumPy twin of any algorithm, no `ops`-proxy dispatch in kk.
- **G2 Device-resident pipeline.** Between the input upload and the result download there are no implicit host↔device
  transfers. Enforced by (a) running the device-only region under `jax.transfer_guard("disallow_explicit")` with the
  boundary operations (§1) as the only allowed crossings, and (b) counting those boundary operations (a CPU-only guard cannot
  prove absence of host materialisation, because CPU fetches are never guarded).
- **G3 Static-shape discipline.** Every *internal compiled entry* sees only bucketed shapes, so the set of compiled programs is
  small and bounded: a default refined run compiles ≤ 6 distinct top-level numerical programs (today 13 shapes of one kernel;
  the exact budget is fixed by the Task 4 prototype, see Acceptance), and a second call with different *values* but the same
  buckets compiles nothing.
- **G4 Pure functional core, publicly usable.** `Spectrum` (a padded pytree) and `kk_f1(...)` are jit/vmap/grad-compatible and
  documented as such (gradient w.r.t. coefficients and energies is finite and matches finite differences).
- **G5 Behaviour parity.** Public signatures, exception types and messages, output values (to the documented rounding
  level), ordering/dedup rules, logging, CLI and GUI behaviour are unchanged. The existing characterization (golden) tests pass unmodified in meaning.
- **G6 Performance gates** (from the evidence above, see [Acceptance](#acceptance-gates)): never slower than today's JAX
  path at steady state; CPU never slower than the NumPy path; cold-start budgets for first call and refined run.
- **G7 Reproducibility.** float64 always — x64 is (re)established at every concrete public entry and every worker entry,
  not only at import (an existing test disables x64 after import and calls the wrapper; that behaviour must survive);
  device, platform and jax version logged at INFO once per process; results are deterministic for a given device + shapes.
- **G8 Docs.** New ADR-005 records "kk is JAX-only" (and the refuted ADR-001 assumption); `jax_architecture.rst` and the kk guide describe the device pipeline.

## Non-goals

- Changing the package-wide default backend or any module outside `xraylabtool/kk/` (calculators, `ops`, `set_backend` semantics for them).
  *Consequence, stated plainly:* `set_backend("numpy")` no longer affects kk; the device is chosen by JAX (`JAX_PLATFORMS`, default GPU if present else CPU).
- A multi-formula `vmap` API (the functional core makes it possible; not built here).
- Differentiable refinement or differentiating through the merge; gradients are specified for `kk_f1` only.
- float32/mixed precision, Pallas/custom kernels, sharding or multi-GPU.
- Making `KKResult`/`NearEdgeSpectrum`/`RefineOutcome` hold device arrays: they stay frozen NumPy containers (the consumption boundary, ADR-004).
- Parsing files or formulas on device.

## Anti-goals (a failure even if every test passes)

- A NumPy fallback that re-implements any algorithm (two code paths drift; this is precisely what JAX-first rejects).
- Any `jax.pure_callback`/`io_callback` into NumPy/SciPy to "make it jit".
- Python control flow on traced values; `ops.any/all` or `float()`/`bool()`/`.item()` on a device value inside the pipeline other than the listed O(1) readbacks.
- Accuracy loss: no float32 anywhere, no loosened golden tolerances to make a refactor pass.
- Recompiling per call, per formula or per refinement pass in steady use.
- A silent global JAX configuration change beyond `jax_enable_x64` (already package precedent, ADR-003).

## Constraints (from the project)

- `mypy --strict` and `ruff` clean (ruff complexity limits apply: split helpers rather than `noqa`); NumPy-style docstrings with `References`; explicit imports; type hints on public APIs.
- ADR-001 (JAX primary, NumPy at I/O boundaries), ADR-003 (float64 non-negotiable, x64 on), ADR-004 (transfer policy).
- `import xraylabtool` must not import `xraylabtool.kk`; kk may import JAX at module import (kk users pay it once).
- `filterwarnings = error`; division/invalid warnings are ignored by config, so numerical tests assert finiteness explicitly.
- GUI computations run on `QThreadPool` workers; GUI tests run serially. CLI exit-code policy unchanged.
- CI has no GPU. GPU verification is a manual, scripted gate (below); tests must pass on JAX-CPU.
- Upstream conventions preserved from the original spec: eV inside / keV at the API, 10 eV–500 keV core range, D5 knot/endpoint conventions, D6 negative f2 kept, D7 sorted unique refinement grid.

## Current state → target

| Piece | Today | Target |
|---|---|---|
| `transform.py` kernel | `ops`-based; JAX via lazy jit; NumPy chunk loop twin; entered/left via NumPy each call | `jax.numpy` only; one jitted block driver; takes/returns device arrays |
| `database.py` | NumPy tables (read-only) | host tables kept (I/O) **plus** device-resident packed tables, uploaded once |
| `spectrum.py` (`compound_spectrum`, `evaluate_f2`) | host NumPy, exact-size arrays | device functions on padded `Spectrum`; host wrappers keep the public contract |
| `nearedge.py` `_merge` | host NumPy, data-dependent sizes | device merge into a padded `Spectrum`; loading/validation stay host |
| `refine.py` | host NumPy loop, kernel call per pass (transfers + a recompile per new size) | device-resident passes on bucketed buffers |
| `api.py` | host orchestration, NumPy in/out at each step | validate on host → one upload → device pipeline → one download |
| `result.py` | frozen NumPy containers | unchanged (consumption boundary) |

## Design

### 1. Boundary and transfer policy (ADR-004 applied)

| Boundary | Policy |
|---|---|
| Text/`.npz` read, formula parse | Host. The ASF tables are additionally uploaded **once** (lazy, cached with the host tables) as packed device arrays. |
| User arrays (energies, near-edge data, merge points) | Validated on the host *before* upload (no transfer needed to validate), then a single `jax.device_put` each. |
| Pipeline | Device → device only. No `np.` in device modules except dtype constants. |
| Data-dependent errors and log data (flat near-edge end points, non-finite normalised values, empty default grid, negative-f2 WARNING data) | Computed on device into a small **status pytree** and raised/logged on the host after a status fetch. The complete allow-list of boundary operations, each a single `_device` helper call (counted by the tests; a pytree is one call regardless of leaf count): **upload** (1 per call of the pipeline, inputs as one pytree; the cold ASF-table upload is counted separately and happens once per process), **status fetch after assembly/merge** (1; carries error flags, endpoint steps, negative-f2 count/min/inside-window count, true segment count and the default-grid size), **status fetch per refinement pass** (1; see §7), **final download** (1). Anything else is a bug. |
| Result | One `_device.fetch` (explicit `jax.device_get`) of the whole output pytree, then host-side keV conversion fix-up and `KKResult`. |

The device-only region runs under `jax.transfer_guard("disallow_explicit")`; the three helpers `_device.upload`, `_device.fetch_status`, `_device.fetch` open a narrow `transfer_guard("allow")` scope and bump a counter, which is what the budget tests assert. (Plain `transfer_guard("disallow")` forbids only *implicit* transfers and would let a stray explicit `device_get` through; on CPU no fetch is ever guarded, so counters, not the guard, enforce the budget there; the GPU gate script additionally runs with the guard live on the real device.) The guard is thread-local: worker entries open their own scope.

Requested energies come back **bit-exact** (the keV values the caller passed are restored host-side after download); the eV array is the canonical device value and keV is never recomputed from it for requested points. (This closes a defect found in review: recomputing eV from keV moved default-grid edge knots by 1 ulp and changed f1 by up to 3.8 e-8.)

### 2. Padded `Spectrum` pytree and the exactness of padding

`Spectrum(seg_ev: (N+1,), coeffs: (N,5), n_segments: int32 scalar)` — `N` is a **bucket capacity** ≥ the true segment count.
Padding convention: knots beyond the last real knot `x_n` are `x_n·(1+k)`, k = 1…, strictly increasing and finite, with **all-zero coefficients**.

Why padding is **algebraically** neutral (checked term by term against `transform.py:185–226` in review): every term of a segment's contribution (`symb_1`, `symb_2`, `symb_3`, and the interior-knot `singular` term) is a coefficient multiple of a finite factor, so a zero-coefficient segment contributes 0. `symb_3` sets numerator and denominator to 1 on an adjacent-knot hit before taking the logarithm, so there is no unprotected `log(0)`. For `E = x_n` the first pad segment is masked by the existing `adjacent` `where`, and the newly interior knot `x_n` uses the right-hand pad polynomial `p_plus = 0` times a finite log, so no compensation is introduced and upstream's endpoint convention (drop the boundary log) is reproduced. For `E` equal to a pad knot all newly adjacent contributions are zero rows. Preconditions the implementation must keep: pad knots finite and strictly increasing (an infinite sentinel would give `0·inf = NaN`), and the sentinel never reaches a polynomial evaluation.

What is **not** claimed: bit-identical floating-point results across capacities. XLA may fuse/reduce differently for different shapes, and the kernel's documented cancellation limit applies (measured: Au at 10 eV differs ≈1e-6 absolute between capacities 509 and 800 on CPU, 0 at the other probes). Hence the padding test (Verification 3) uses the repository's existing cancellation-scaled comparison (`tests/fixtures/kk_rounding.py::assert_kk_close`) on real spectra, plus a well-conditioned synthetic all-five-basis spectrum at a tight tolerance to isolate algebraic padding errors. Gradients: the core masks coefficients by `n_segments`, so differentiating the full padded coefficient buffer cannot create a physical tail; gradients are specified w.r.t. the real coefficients and the energies.

### 3. Shape buckets

One helper (`kk/_shapes.py`) maps a true size to a bucket. Ladder (deterministic recurrence): `b₀ = 512`, `b_{k+1} = 8·floor(1.25·b_k / 8)` (512, 640, 800, 1000, 1248, 1560, 1944, …; the numbers are generated, not constants), so every bucket is a multiple of 8, `bucket(n) = 512` for `n ≤ 512` (the minimum is a fixed parameter validated to be a multiple of 8 and ≥ 64: the recurrence stalls for tiny minima; use `bucket_pow2` for small capacities), and for `n > 512` the padding waste is ≤ 25 % (`bucket(n) ≤ 1.25·n`). About 10 buckets cover all realistic formulas (element grids 505 (H) – 913 (Mo) knots, U 518; measured compounds 509–1228 segments). Powers of two (`bucket_pow2`; minimum 8 by default, minimum 1 for the energy count `m_b`) are used for energy blocks and refinement buffers.

**Capacity is always chosen on the host from host-known counts, and is a conservative upper bound, so overflow cannot occur** (`jnp.unique(size=…)` silently truncates when undersized; it is a static allocator, not a dynamic one): compound capacity = `bucket(Σᵢ (n_knotsᵢ − 1))` (the union of knots cannot exceed the sum); merge capacity = `bucket(n_seg_in + n_nearedge + 2)`; the refinement accumulator starts at a host-known size from the initial grid and grows on demand (§7), never from `max_points` alone. The true count is still carried on device (`n_segments`) and asserted ≤ capacity in tests.

**Entry shapes.** A jitted function specialises on its raw input shapes *before* its body pads. Therefore: the **internal compiled entries** (block driver, assembly, merge, refine pass) take arrays already padded to bucket sizes plus an `n_valid`/`n_segments` scalar; **host wrappers** pad before upload (energies to `block × n_blocks` using an in-range edge energy, where `m_b = bucket_pow2(m, minimum=1)`, `block = block_for(chunk_size, capacity, m_b)` and `n_blocks = ⌈m_b / block⌉` (`_shapes.energy_blocks`; padded length below `2m` plus at most one block, and the distinct `(block, n_blocks)` pairs grow only with `log₂ m`; bucketing the block count with a minimum of 8 padded small and mid-size inputs 5-8x and is not used), element lists to a bucketed `k`, near-edge arrays to `bucket_pow2`) and slice after download; the public convenience `kk_f1(eval_ev, spectrum)` is documented as specialising on `eval_ev.shape` when wrapped in a user `jit` (static `m`, static slice, empty input handled by a static branch). `block`, `capacity` and `size=` are static arguments. Every padded array travels with an explicit **valid count** (`n_elements` for the element list, `n_ne` for near-edge points, `n_valid` for energies, `n_segments` for spectra) and padded rows are excluded by masks, not by a safe-dummy convention: padded elements get count 0 and their knot rows are masked to `+inf` before the union; padded near-edge entries are forced to `+inf` before the sort and carry zero weight in interpolation and in every status statistic. Results never expose padded entries; bucket choice must not change results beyond the rounding scale (tested).

### 4. Kernel (`transform.py`)

`kk_piecewise_poly` keeps its maths and docstring (Eq. 10 with the a₋₃ sign correction, knot conventions D5) but uses `jnp` directly.
New public pure function `kk_f1(eval_ev, spectrum, rel_corr, *, block=None)` → `jax.Array` of shape `eval_ev.shape`: chunked via `lax.map` over blocks; documented as `jit`/`vmap`/`grad` compatible (`Spectrum.n_segments` is an int32 **leaf**, not static metadata, so vmap over spectra with different counts works; pytree unflatten does no host validation, so it is safe on tracers). `kk_real_spectrum(...)` stays as the validated host-facing wrapper (NumPy in/out, same errors, empty input returns empty). The block size is an **upper bound** derived from `min(chunk_size, 32 MiB budget / capacity, m)`, rounded *down* to a ladder value; the number of blocks is bucketed. The `_jit_cache`, the NumPy chunk loop and `ensure_float64`-based casting disappear. x64 policy: `_device.ensure_x64()` runs at import **and** at every concrete public entry and worker entry before any upload or trace (an existing test disables x64 after import and expects float64 output; a stored float64 buffer would otherwise be canonicalised to float32 inside a later trace); the pure core states x64 as a precondition and asserts dtype.

### 5. Spectrum assembly and f2 on device (`spectrum.py`, `database.py`)

- Packed device tables: knots `(92, Kmax)` and coefficients `(92, Kmax−1, 5)` with the padding convention, plus per-element true lengths.
- `compound_spectrum_device(atomic_numbers, counts)`: gathers the k element rows (k is static), first **masks each gathered row's padding by its true knot count** (dummy knots set to `+inf` so they sort to the tail and can never be mistaken for real knots; genuine and padded entries are otherwise indistinguishable to `unique`), forms the sorted union with a **static-capacity** `jnp.unique(size=cap, fill_value=+inf)`, computes the true count from the mask (independent of any truncated output), replaces fills by the pad rule, and accumulates `count_i · coeffs_i[idx_i]` **in the order given** with `idx_i = clip(searchsorted(knots_i, seg_lower, 'right') − 1, 0, n_knotsᵢ − 2)` (the search runs over the whole finite, increasing padded row — no traced-length slice, which would not trace — and the clip to the true length does the rest) selected by each output segment's **lower boundary**; coefficients of segments at or beyond `n_segments` are forced to exactly 0 (so sums match today's to rounding and are independent of bucket capacity).
- `evaluate_f2_device(E, spectrum)`: `searchsorted` + gather + Laurent polynomial; same right-continuous edge convention; the index is clamped to the last **real** segment `n_segments − 1` (not the capacity), so the final real knot uses the last real polynomial as today.
- `relativistic_correction`: one `jnp` expression over Z and counts.
- Host wrappers `compound_spectrum(symbols, counts)` and `evaluate_f2(energy, seg, coeffs)` keep their public contract (validated NumPy in, exact-size NumPy out).

### 6. Near-edge merge on device (`nearedge.py`)

Loading, unit handling, `NearEdgeSpectrum` validation on the host arrays, merge-point resolution and snapping stay on the host. Two device functions replace the numerical part, with separate contracts (the physical conversion `to_asf_scale` is independent of any tabulated spectrum and keeps its standalone host API and tests; normalisation needs the tabulated end-point values):
- `to_asf_scale_device(...)` — the physical conversion (host `to_asf_scale` wraps it).
- `merge_device(spec, ne_energy_ev, ne_f2_converted, n_ne, low_ev, high_ev, *, capacity) -> MergeResult(spectrum, scaled_f2, used_points_ev, status)`: `inside = (E_ne > low) & (E_ne < high)`; candidate knots = tabulated knots outside `[low, high]`, `low`, the inside near-edge energies, `high`, with excluded entries sent to the sort tail and re-padded by rule; the source segment of each output segment is selected by its **lower boundary** with the right-continuous convention, clamped to the last real segment (a midpoint is wrong for adjacent representable floats, where `(lo+hi)/2` can equal `hi` and pick the next polynomial); outside the window the tabulated coefficients are copied unchanged, inside it the piecewise-linear near-edge ones are used. Normalisation `(v − v_low)·(tab_step/step) + a_low` happens here and `scaled_f2` returns the **full** normalised near-edge curve (including points outside the replaced window), which `KKResult.nearedge_scaled_kev_f2` needs.
- `status` carries everything the host must raise or log, evaluated without trapping on invalid arithmetic: the four end-point values (`v_low`, `v_high` measured; `a_low`, `a_high` tabulated) and both steps — today's flatness checks compare each step against the end-point *magnitudes* and the messages print the values (`nearedge.py:460–473`) —, non-finite flag over all normalised values (raised as today's `NearEdgeDataError`, not as a later generic error), negative count, minimum and inside-window count (the D6 WARNING), and the true segment count. It is read in the single status fetch of §1.
"Segment that starts at a tabulated edge coinciding with a merge point" behaviour is preserved and tested.

### 7. Adaptive refinement on device (`refine.py`)

Host-**orchestrated**, device-**resident** passes (my call; alternatives below):
state lives in bucketed device buffers (active intervals: `lo, hi, f1_lo, f1_hi, f2_lo, f2_hi, pole_lo, pole_hi`; accumulated points). One jitted `refine_pass` per **active-capacity bucket** does: pole-sliver split, midpoint, midpoint f1 and `evaluate_f2` at midpoints (f1 through `_midpoint_f1`, a block driver specialised to the live intervals: its loop runs only the blocks that hold live intervals, so a pass costs in proportion to them, and it calls the same `kk_piecewise_poly` per block as `kk_f1`, so the algorithm is shared and only the batching loop differs; not reverse-differentiable, which refinement does not need), the `bad` test (single absolute `tolerance`; no `rtol`), the `max_points` truncation, stream-compaction of the new active set and appended points. Semantics to reproduce exactly (they are the current `refine.py` behaviour; the builder reads it and pins each with a test): the next active order is **all retained left halves followed by all retained right halves** (not interleaved, not energy order), so the truncation "first `max_points − added` failing intervals in internal order" is a cumulative-sum mask over that order; `max_points` fires when `added + raw_bad_count > max_points` (strict), and a later `max_points` reason overrides an earlier `float spacing` one; float-spacing drops only the unsplittable intervals, sets a sticky reason and continues with the others; slivers are removed before the spacing/midpoint test, a sliver-only pass still counts as a pass without the normal midpoint DEBUG line, and the wide-sliver *classification* uses the last **real** segment width, never the pad width (the "widest sliver" *statistic* is the widest dropped interval and is a separate quantity).

**Control protocol.** The host loop performs **one structured status fetch per pass** (a handful of int scalars: next active count, accepted count, checked/splittable count, stop/reason flags — only what the per-pass DEBUG line and the continue/stop/rebucket decision need) — enough to emit the same per-pass DEBUG line, decide continue/stop, and choose the next active bucket. Buffer allocation is fixed by the host before each pass: an active buffer of capacity `B` can produce `2B` children and an accumulator holds appended points. Its capacity is **grown on demand from host-known counts**, never sized from `max_points` (a public, unbounded argument: `10**9` would otherwise allocate tens of GB): it starts at `bucket_pow2(max(min(max_points, 4·(m−1)), first-pass need))` and, before any pass, if `added + min(n_next, max_points − added)` exceeds it, the host pads it to the next power of two (a pass adds at most one point per interval and at most what is left of `max_points`, so overflow is excluded by construction; no transfer is needed because `added` and `n_next` are host ints from the status fetch). Pass buffers are donated, so a pass copies nothing; each accumulator bucket adds at most one `_refine_pass` compile; when the new active count crosses a power-of-two boundary the host rebuckets between passes (compiling at most one program per distinct active bucket, which is why the budget is "top-level programs", bounded by `log₂` of the largest active set). `exhausted` reasons (`float spacing`, `max_points`, `max_passes`), `n_pole_skipped`, the widest-sliver log and every warning keep their current semantics; the widest-sliver statistic and sliver count accumulate in device state and arrive with the final download (they are not part of the per-pass status). The final sort-and-union is a device `argsort`, then the single download.

Alternatives considered and rejected:
- *Single `lax.while_loop` over a fixed `max_points` buffer:* zero syncs, but evaluates the kernel over the whole masked buffer every pass (≈ 10⁵ midpoints × segments × 50 passes) and cannot emit per-pass DEBUG logs without reconstructing them; exact-ordering parity is harder to prove.
- *Leave refine on the host:* violates G1/G2 and keeps a transfer per pass.

### 8. Orchestration (`api.py`)

`kk_calculate_real` validates inputs on the host exactly as today (same exception types/messages — including the new non-string formula `FormulaError`, complex-input rejection, and merge-point reporting that returns the *used* values), uploads once, runs the device pipeline, downloads once. `chunk_size` keeps its meaning: an **upper bound** on energies evaluated per block (rounded *down* to the block ladder, never up; the number of blocks is bucketed instead). Energy bookkeeping: for **explicit** energies the host canonicalises to eV and, under refinement, applies `np.unique(..., return_index=True)` (first keV representative of eV-colliding points kept) before upload; requested order and duplicates are preserved without refinement. For the **default grid** (assembled on device) the knots are already sorted and unique, the working grid is that clipped knot array on device, no representative mapping exists, and its keV values are `eV/1000` computed host-side after the final download (no download/re-upload of the grid).

### 9. Public API compatibility

| Item | Change |
|---|---|
| All current public functions/classes and signatures | Unchanged; accept NumPy/array-likes (and JAX arrays); return NumPy at the boundary |
| `KKResult.backend` | kk now always reports `"jax"`; the field still accepts `"numpy"` when constructed directly (legacy) |
| `set_backend("numpy")` | No effect on kk (documented; non-goal to change). Everything else unaffected |
| New, additive | `Spectrum`, `kk_f1` (device-level, jit/vmap/grad-able), documented in `docs/api/kk.rst` |
| Removed | kk's use of `ops`, kk's `_jit_cache`, kk's `ensure_float64` use. `array_ops.ensure_float64` has no other caller (verified by grep over `xraylabtool/` and `tests/`), so it is deleted together with its test file `tests/unit/test_backend_kk_ops.py` — **after** the new x64-policy tests (kept from `test_transform.py::test_float32_inputs_give_float64_output`, adapted) exist |
| Tests that must change (listed so no assertion is silently weakened) | backend parametrisation in `tests/unit/kk/test_api.py`, `tests/characterization/kk/test_{transform_parity,nearedge_golden,end_to_end_golden}.py`, `tests/unit/kk/test_transform.py` (drop `_jit_cache`/`set_backend` cache test, keep float32-input test without backend fixture); `test_api.py::test_backend_is_recorded…` → asserts `backend == "jax"` and chunk-size invariance; `tests/unit/kk/test_refine.py::test_refine_forwards_chunk_size` spies on `refine.kk_real_spectrum` → asserts the block size actually seen by the device block driver (≤ requested); `tests/unit/kk/test_docs.py` requires every `kk.__all__` name in `docs/api/kk.rst`, so API docs for new exports land in the same task as the export. Numeric assertions and golden tolerances are untouched. |

**Recorded contract change (Task 1):** `test_jump_and_endpoint_follow_upstream_convention[ne]` was `rtol=1e-10` (atol 0) on the NumPy default backend; on JAX-CPU Ne at its 10 eV end point drifts by 7.6e-9 (XLA reassociation of cancelling terms), so the replacement bound is `rtol=1e-10, atol=5e-8` for Ne only (Ar and synthetic keep atol 0).

### 10. Compilation, threads, logging

- **Persistent compile cache:** enabled by the CLI `kk` command and the GUI launcher (`xraylabtool/gui/__main__.py`) only, not by the library, and only if the user has not configured `jax_compilation_cache_dir`; opt-out `XRAYLABTOOL_NO_JAX_CACHE=1`. Directory: OS-appropriate user cache (`$XDG_CACHE_HOME` or `~/.cache` on Linux, `~/Library/Caches` on macOS, `%LOCALAPPDATA%` on Windows) under `xraylabtool/jax`. JAX only persists programs whose compile time exceeds `jax_persistent_cache_min_compile_time_secs` (default 1.0 s, so most small kernels are *not* cached): when the helper enables the cache itself it also sets that threshold to 0 (documented side effect, skipped if the user set it); the cached-cold budget below is stated for that configuration. Measured benefit: refined cold run 1.69 → 0.64 s (CPU).
- **Warm-up:** `xraylabtool.kk.warm_up(formulas=…, refine_tolerance=…)` precompiles the buckets those formulas and a default refined run touch (the refinement tolerance changes which active buckets are visited); idempotent, blocks until compiled (`block_until_ready`). Only the **GUI** calls it, on an idle worker after the window is shown and not through the normal KK-result signal handlers; the **CLI never calls it** (a one-shot call must not pay for formulas it was not asked for) — it only enables the cache.
- **Threads:** JAX dispatch is thread-safe, but lazy initialisation is not automatically single-flight: the device ASF-table upload and the persistent-cache setup are guarded by a lock (single upload per process and default device; the cache key includes the device). The x64 setting and transfer-guard contexts are thread-local/worker-local, so each worker entry establishes its own. The first-call compile happens on the worker thread, never the UI thread.
- **Logging:** one INFO line per process with platform, device, jax version and x64; refinement logging unchanged.

## Verification (what "correct" means)

1. **Goldens unchanged:** `tests/characterization/kk/*` (upstream kkcalc goldens, CXRO, near-edge, refine) pass with their existing tolerances. NumPy-vs-JAX parity parametrization is dropped (one implementation); the goldens and the oracle below take its place.
2. **Independent quadrature oracle (new, permanent):** longdouble principal-value integration of Henke's f1 = Z* + (2/π) PV∫ x f2/(E²−x²) dx (removable-singularity subtraction + Gauss–Legendre **split at every polynomial boundary and at the singular point**, convergence confirmed by doubling the node count), fixed seed, ≥ 6 compounds incl. H, Si, SiO2, Au, U, Mo. Energies: log-uniform over 10 eV–500 keV (so low energies are sampled) plus deterministic low-energy, near-edge and knot cases. Tolerances are **measured, not guessed**: Task 0 records the per-compound maximum error of the current code and sets `tol = 3 × measured` (documented in the test), scaled per energy by the repository's cancellation allowance (`assert_kk_close`) where that is larger; the heavy-element/low-energy ≈1e-4 limit is the known worst case. Knot cases are restricted to **continuous interior knots** (where the principal value is an ordinary limit); jump knots and domain end points are upstream *conventions* and are checked separately against the locked goldens, not by this oracle.
3. **Padding/bucket invariance:** the same inputs at capacity `N` and at the next two buckets, at E exactly on the first knot, the last real knot, an interior knot and a pad knot, agree under the cancellation-scaled comparison (`assert_kk_close`) on real spectra; a synthetic well-conditioned spectrum exercising all five basis terms agrees to `rtol=1e-12`. All values finite. (Algebraic neutrality, §2; not a bit-identity claim.)
4. **Additivity:** `f1(compound) = Σ nᵢ f1(elementᵢ)` and the same for f2 on the same energies, within the **sum of the compound's and the elements' rounding allowances** (the segment partitions differ, so a fixed `rtol=1e-11` is not achievable: the current NumPy code misses it by up to 1.4e-7 absolute on Fe2O3 at 10 eV-scale energies). Guards the union-grid assembly.
5. **Transfer budget:** device-only region under `transfer_guard("disallow_explicit")`; boundary-helper counters assert: 1 upload, 1 assembly/merge status fetch, 1 status fetch per refinement pass, 1 final download on a warm call (the cold ASF-table upload is asserted separately, once per process, including under a barrier-released concurrent first call in a subprocess). The GPU script reruns this with the guard live on the CUDA device.
6. **Compile budget:** a version-tested recorder captures actual compile events (on the installed jax they are emitted by `jax._src.interpreters.pxla`, not `dispatch`; the recorder is self-tested with a deliberately fresh function so a silent zero cannot pass). Counted unit: top-level numerical programs of `xraylabtool.kk` (block driver, assembly, merge, refine pass per active bucket, final sort/union), excluding JAX's trivial primitive compiles; default refined run ≤ 6 (budget confirmed or revised by the Task 4 prototype); a second call with different values and the same buckets compiles 0. Persistent-cache hits are reported separately.
7. **Autodiff/vmap:** `jax.grad` of a scalar function of `kk_f1` w.r.t. the real coefficients and the energies is finite and agrees with central differences (per-basis-term step sizes, absolute allowance near zero derivatives, energies away from knots); `vmap` over stacked equal-capacity spectra with **different** `n_segments` equals a Python loop; empty, single-point, odd-length, exact-block and partial-block energy inputs behave.
8. **Fuzz/invariants:** the randomized valid-call fuzz (near-edge, all input types, chunking, refinement) asserts finite outputs, order preservation without refinement, strictly increasing grid with refinement.
9. **CLI/GUI end-to-end:** the scripted unmocked CLI (31 checks) and offscreen GUI (worker thread) checks written during the 2026-10-25 double-check are promoted to committed tests; error paths still exit 1 / show the dialog.
10. **GPU gate (manual, scripted, not CI):** `scripts/kk_gpu_check.py` (run unsandboxed on the CUDA machine) reproduces the oracle, additivity and performance tables on the real device and prints a pass/fail summary.

## Acceptance gates

| Gate | Threshold |
|---|---|
| Steady-state kernel, GPU, SiO2 m = 10⁴ | ≤ 20 ms (today 16 ms) |
| Steady-state kernel, CPU, SiO2 m = 10³ / 10⁴ | ≤ 35 / ≤ 250 ms (today 27 / 207); never slower than the removed NumPy path (258 / 1 645) |
| First `kk_calculate_real` (SiO2, 1 000 explicit energies), after import, CPU | ≤ 1.0 s |
| Refined run `refine_tolerance=0.01`, SiO2, CPU: cold / warm | ≤ 1.7 s / ≤ 0.10 s (cold ≤ 0.7 s with the persistent cache as configured in §10); GPU cold ≤ 2.5 s. Measured in fresh subprocesses with the platform selected before importing jax, warm timings after `block_until_ready` / completed host output |
| Top-level numerical programs compiled in that refined run | ≤ 6, confirmed by the Task 4 prototype: the run compiles exactly 6 today (`compound_spectrum_device`, `_kk_f1_blocks`, `evaluate_f2_device`, and the refinement's `_first_pass`, `_refine_pass`, `_finalize`). The active buffer has a 2048-interval floor, so the pass program is shared by every tabulated formula whose grid fits it and whose accumulator stays in one bucket; `_first_pass` is keyed on `bucket_pow2(m)` and `_finalize` on `out_len`, so formulas of different grid size still compile those. Task 5 must not add a program on top |
| Accuracy | oracle and goldens as in Verification; no tolerance loosened |
| Quality | `ruff`, `mypy --strict`, whole test tree green on CPU; GPU gate green on the CUDA machine |

## Risks

- **Padding at knots** is the subtle one (§2): algebraic neutrality was verified term by term in review; Task 1 proves it numerically (Verification 3) before anything else is built on it. Capacity overflow is excluded by host-side conservative bounds (§3), not by detection.
- **`jnp.unique(size=…)`/compaction cost on GPU** for the compound union: measure in the first spike of stage 2; the fallback is a sort + mask formulation.
- **Refinement parity** (ordering under `max_points`, sliver logic) — covered by the existing refine goldens plus a bucket-invariance test.
- **Cold start for one-shot CLI calls** rises from ~0.3 s (NumPy) to import 0.34 s + compile; bounded by the gates and the persistent cache.
- **XLA fusion changes rounding** at the documented ~1e-4 (heavy elements, low E) level — already the accepted budget; oracle tolerances use it.
- **x64 is global**: unchanged package precedent, but the new tests must assert float64 outputs explicitly.
- **jax API drift** (`transfer_guard`, `unique(size=)`, pytree registration, compile-log logger names): `pyproject.toml` declares `jax>=0.8.0` but CI installs `jax[cpu]>=0.9.0` and tests only the latest. Task 0 runs the kk suite on jax 0.8.0; the plan adds a minimum-version CPU CI job, or, if 0.8.0 fails, raises the floor in `pyproject.toml` and `uv.lock`. Instrumentation that depends on private logger paths is self-tested.

## Delivery order (each stage ships green)

0. Baselines and oracles: freeze goldens, add the quadrature oracle, additivity, fuzz, perf baseline script (on the *current* code, so the new tests prove parity).
1. Foundations: `_shapes`, `Spectrum`, jnp-only kernel + `kk_f1`, padding-exactness test, x64 handling.
2. Device spectrum: packed tables, `compound_spectrum_device`, `evaluate_f2_device`, relativistic correction.
3. Device merge.
4. Device refinement.
5. Orchestration: single upload/download, transfer-guard and compile-budget tests, removal of NumPy-era code and kk's `ops` use.
6. Docs (ADR-005, `jax_architecture.rst`, kk guide, API docs, CHANGELOG), compile-cache/warm-up wiring, CLI/GUI promotion of scripted checks, GPU gate script.

## Decisions confirmed by Wei (2026-10-25, grilling round 1)

Wei accepted all eight recommended answers and said **not to implement the plan yet**; this document and the plan stay uncommitted drafts until execution is requested.

1. `set_backend("numpy")` is silently ignored by kk (documented in the changelog and ADR-005); no warning, no error.
2. `Spectrum` and `kk_f1` are public in this change, with a deliberately minimal `kk_f1` signature.
3. The slower cold start for one-shot CLI calls (≈0.4 s → ≈1.7 s cold, ≈0.64 s with a filled cache) is accepted.
4. The CLI and GUI enable the persistent compile cache by default (OS user-cache directory, threshold lowered to 0), skipped when the user configured JAX's cache; the library never enables it; `XRAYLABTOOL_NO_JAX_CACHE=1` opts out.
5. JAX floor: keep `>=0.8.0`; Task 0 tests 0.8.0 and either adds a minimum-version CI job or raises the floor.
6. GPU verification stays a manual script (`scripts/kk_gpu_check.py`).
7. At execution start, the pending uncommitted kk hardening changes (code + `tests/unit/kk/test_input_hardening.py`, separate from the graphify files) are committed on `docs/cli-api-and-track-claude-md` first, then `feat/kk-jax-first` is cut from there and the spec and plan are committed on it. Not done yet.
8. Execution mode: subagent-driven for Tasks 1–4, native for Tasks 0, 5 and 6.

## Calls made by me (please check)

1. **kk is JAX-only** — no `ops`/NumPy twin. Alternative: extend the `ArrayBackend` protocol so one `ops`-written code path also runs on NumPy; rejected because loops (`lax.map`, compaction) and `searchsorted`/`unique(size=)` have no common-denominator form, and the measurements show NumPy has no remaining advantage.
2. `set_backend("numpy")` no longer affects kk (documented), rather than raising or warning.
3. Padded-pytree `Spectrum` with a zero-coefficient tail; geometric bucket ladder, ratio ≤ 1.25.
4. Host-orchestrated, device-resident refinement passes (one structured status fetch per pass), not a single `while_loop`.
5. Public wrappers keep NumPy returns; `Spectrum` and `kk_f1` are the new additive device-level API.
6. `KKResult.backend` always `"jax"` from kk; legacy value `"numpy"` still accepted on construction.
7. Persistent compile cache on in CLI/GUI only, off in the library; env opt-out.
8. Validation stays on the host before upload; the boundary operations of §1 are the complete allow-list.
9. Parametrized NumPy/JAX kk tests collapse to JAX-only; `test_backend_kk_ops.py` and `ensure_float64` are deleted (no other user), after the x64-policy tests exist.
10. GPU verification is a manual scripted gate because CI has no GPU.
11. This document and the plan are **not committed** (see Decisions 7: committed on the feature branch at execution start).

## Left to the builder

Helper names (the bucket recurrence itself is fixed in §3); the compaction primitive (`nonzero(size=)`, cumsum+scatter, or sort) per measurement; log wording; whether `kk_f1` takes a `Spectrum` or unpacked arrays internally; file split inside `kk/` (e.g. a private `_device.py`); how warm-up chooses buckets; thresholds for the GPU script's pass/fail beyond the table.

## Review log

Cross-model review on 2026-10-25 (Codex: correctness/feasibility, run against the real code with executed probes; Agy: architecture/consistency). Reports: `three-brain-out/2026-10-25-kk-jax-spec-review/`.

Accepted and applied (spec section → fix): capacity/overflow/dummy-knot masking (§3, §5); refinement control protocol and exact ordering/precedence semantics (§7); merge result/status contract, separate `to_asf_scale_device` (§6, §1); x64 policy at every entry (§4, G7); lower-boundary segment selection instead of midpoint (§6); padding claim restated as algebraic, numeric gate cancellation-scaled (§2, Verification 3); oracle/additivity bounds measured, energy distribution and quadrature splitting defined (Verification 2, 4); bucket recurrence made consistent (§3); transfer-guard semantics and counting (§1, Verification 5); compile recorder logger and counting unit (Verification 6, Acceptance); entry-shape contract, empty input, static args (§3, §4); test adaptations enumerated (§9); `chunk_size` as an upper bound (§8); concurrency and single-flight init (§10); persistent-cache thresholds, OS-aware path, CLI never warms up (§10); jax floor (Risks); `ensure_float64` deletion made explicit (§9).

Checked and rejected: Agy findings that quote text absent from both documents (a `Spectrum` "static `n_segments` with aux_data", `005-adr-kramers-kronig-jax-first.md`, `tests/unit/gui/…`, "plan line 23: CLAUDE.md untracked", a `rel_corr` field on `Spectrum`, `test_quadrature_oracle.py`, a Task 2 instruction to regenerate goldens) — grep of both documents finds none of them; Codex confirms `n_segments` as an int32 leaf is the right vmap design. Agy's remaining findings that do hold (CI/Makefile GUI exclusion lists, `docs/physics/kramers_kronig.rst` NumPy-backend sentence, `docs/development/changelog.rst` lockstep with `CHANGELOG.md`, `warm_up` lazy-export registration, `performance`-marker auto-assignment by test name, OS-aware cache path) are applied in the plan.

Second round (Codex re-check of the revision): 12 of the 17 findings resolved, 5 partial, 8 new defects found and fixed in the same pass — traced-length slice in assembly replaced by a whole-row search plus clip and explicit valid counts/masks for every padded array (§3, §5); default-grid bookkeeping split from explicit-energy bookkeeping (§8); `MergeStatus` carries the end-point values its messages print (§6); baseline now includes Au and a shared probe generator (plan Task 0); adjacent-float merge fixture corrected so the merge is not rejected as flat; CLI test calls `main()` with patched `sys.argv`; `_refine_pass` takes a static `block` and both old chunk-spy tests are adapted; `bucket` validates its minimum. Also aligned: oracle restricted to continuous interior knots, per-device table cache key, sliver statistic kept out of the per-pass status, CPU-pinned floor check.
