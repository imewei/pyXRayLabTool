# kk JAX-first Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the NumPy/`ops`-based and host-side parts of `xraylabtool.kk` with one `jax.numpy` implementation that stays on device from a single upload to a single download, with bounded recompilation.

**Architecture:** A padded-pytree `Spectrum` (zero-coefficient tail, strictly increasing dummy knots) with a geometric shape-bucket ladder makes every internal compiled entry see few static shapes; capacities are conservative host-side upper bounds. Kernel, compound assembly, `evaluate_f2`, near-edge splice and refinement passes become pure device functions that report errors through small status pytrees; validation, file I/O, logging and the frozen `KKResult` containers stay on the host. Public signatures are unchanged; `Spectrum` and `kk_f1` are added.

**Tech Stack:** Python ≥3.12, JAX (x64) on CPU/GPU, NumPy only at I/O boundaries, pytest, mypy `--strict`, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-10-25-kk-jax-first-design.md` (read it first, including its Review log; this plan uses its section numbers as §n). Prior plan for context: `docs/superpowers/plans/2026-10-24-kkcalc-integration.md`.

## Global Constraints

- Owner's directive: the whole package is JAX-first; kk is JAX-only. No NumPy twin of any algorithm, no `ops` dispatch in `xraylabtool/kk/`, no `pure_callback`/`io_callback`, no float32.
- Python ≥ 3.12; `mypy xraylabtool/ --strict`, `ruff check` and `ruff format` clean (split helpers instead of `noqa` for PLR0912/PLR0915); NumPy-style docstrings with `References`; explicit imports.
- float64 always (ADR-003). `_device.ensure_x64()` runs at import **and** at every concrete public entry and worker entry (spec §4); the pure core states x64 as a precondition.
- Transfers per ADR-004 and spec §1: the only crossings are the `_device` helpers `upload`, `fetch_status`, `fetch`; the device-only region runs under `jax.transfer_guard("disallow_explicit")`; budgets are asserted from helper counters (a CPU guard cannot prove absence of host fetches).
- Capacities are conservative host-side bounds (spec §3); overflow is excluded by construction, `n_segments ≤ capacity` is asserted in tests.
- `filterwarnings = error`; divide/invalid warnings are ignored by config, so numerical tests assert `np.isfinite(...).all()` explicitly.
- `import xraylabtool` must not import `xraylabtool.kk`. GUI tests run serially. CI has no GPU: every test must pass on JAX-CPU.
- Goldens (`tests/characterization/kk/*`) keep their tolerances; numeric assertions are never loosened. Test changes listed in spec §9 are test maintenance, not weakening. Documented kernel rounding limit: abs error ≈1e-4 only at 10–14 eV for Au/Pb/U, otherwise small; new comparisons use `tests/fixtures/kk_rounding.py::assert_kk_close` scaling unless a tolerance is stated.
- Behaviour parity (§9): exception types and messages, upstream knot/endpoint conventions (D5), negative f2 kept with WARNING (D6), sorted unique refinement grid (D7), requested keV energies bit-exact, `merge_points_kev` reports the used (snapped) values, `chunk_size` is an upper bound on energies per block.
- Padding convention (§2): knots past the last real knot are `x_n·(1+k)`, k=1,2,…; coefficients of padded segments are exactly 0 and forced to 0 by `n_segments` masking in the core.
- Test names must not contain `performance`, `benchmark` or `timing` unless intended: `tests/conftest.py` auto-marks such names `performance` (skipped without `--run-performance`). `scripts/kk_bench.py` and `scripts/kk_gpu_check.py` are standalone scripts, not collected by pytest.
- **Do not start implementing until Wei asks.** At execution start (spec Decisions 7): commit the pending kk hardening changes (code + `tests/unit/kk/test_input_hardening.py`, not the graphify files) on `docs/cli-api-and-track-claude-md`, cut `feat/kk-jax-first` from there, and commit the spec and this plan on it. Execution mode (spec Decisions 8): subagent-driven for Tasks 1–4, native for Tasks 0, 5 and 6.
- Run GPU checks outside the sandbox (`dangerouslyDisableSandbox`), CPU checks inside.

## Review Focus

Failure modes the spec implies but no single task's happy-path tests would catch, most likely first:

1. **Energy exactly on a knot, or a requested point repeated / unsorted / on a segment edge** (first, last, interior knot, pad knot, duplicates, eV-colliding keV values): finite, and equal to the pre-migration baseline within the cancellation-scaled tolerance. Pinned in Task 1 (`test_padding_neutral_at_knots`) and Task 5 (`test_unsorted_duplicate_energies_order_preserved`, `test_ev_collision_first_representative_kept`).
2. **A single-element compound with the fewest knots (H, 505), the most (Mo, 913), a heavy one (U), multi-element with identical elements repeated, and a 1-energy / empty energy input**: no shape error, no NaN from padding, no dummy knot or fill leaking into the union, no truncation. Pinned in Task 2.
3. **Near-edge window covering zero, one, or all tabulated knots, touching the range ends (snap), adjacent representable floats at a window edge, and non-finite normalised values**: result independent of bucket choice, same `NearEdgeDataError`s. Pinned in Task 3.
4. **Refinement that exhausts each limit (`max_points`, `max_passes`, float spacing) at pass ≥ 2, at exact cap equality, with mixed spacing + splittable intervals, a sliver-only final pass, both endpoint slivers, and one that adds nothing**: same grid, reasons, counters and per-pass log fields as today; no padded point escapes. Pinned in Task 4.
5. **x64 disabled after import, calls from `QThreadPool` workers, the very first concurrent calls, and identical shapes with different values**: float64 output, one table upload, no recompilation, identical results. Pinned in Task 5.

---

### Task 0: Baselines, oracles, jax-floor check (on the current code)

**Files:**
- Create: `scripts/gen_kk_prejax_baseline.py` (kept; deterministic generator), `tests/unit/kk/data/prejax_baseline.npz` (its output, committed), `tests/fixtures/kk_oracle.py`
- Create: `tests/unit/kk/test_oracle.py`, `tests/unit/kk/test_additivity.py`, `tests/unit/kk/test_fuzz.py`, `scripts/kk_bench.py`
- Existing and kept as the permanent golden generator: `scripts/gen_kk_golden.py` (goldens are *locked* now: no regeneration after Task 1)

**Interfaces:**
- Produces: `kk_oracle.f1_reference(energy_ev: float, seg_ev: NDArray, coeffs: NDArray, rel_corr: float) -> float` (Henke sign `rel − (minus+plus)/π`; removable-singularity subtraction; Gauss–Legendre in ln x **split at every polynomial boundary and at the singular point**; `np.longdouble`; node count doubled until the result changes < 1e-15 relative, raise otherwise). Valid only for continuous interior knots; jump knots and end points are upstream conventions checked against the locked goldens instead.
- Produces: `prejax_baseline.npz` with keys per case `<case>_seg`, `<case>_coeffs`, `<case>_energy_ev`, `<case>_f1` for cases `H, Mo, U, Au, SiO2, Fe2O3, Ar` with probe energies generated by one shared function `tests/fixtures/kk_probes.py::probe_energies(seg_ev) -> NDArray` (used by the generator **and** by every later test, so the sets match by construction): 200 log-uniform in 10 eV–500 keV from `default_rng(2026)`, the first knot, the first three segment midpoints, the last real knot, three interior continuous knots, a quarter-way point of two segments, and three above-range energies `x_n·(1+k)` (the pad-knot positions of the padding rule). Values at above-range energies are computed by the pre-migration kernel on the unpadded spectrum, which is the reference the padded evaluation must reproduce, `merge_*` (polystyrene + `tests/characterization/kk/nearedge_example.txt`) and `refine_*` (Si, tolerance 0.05). Tasks 1–4 compare against this file; it is generated from the pre-migration code only (provenance recorded in the generator's docstring: git commit and jax version).
- Tests only use public functions (`compound_spectrum`, `kk_real_spectrum`, `kk_calculate_real`), so they run unchanged before and after.

- [ ] **Step 1: Check the jax floor.** Run `JAX_PLATFORMS=cpu uv run --with jax==0.8.0 --with jaxlib==0.8.0 pytest tests/unit/kk tests/characterization/kk --no-cov -q` (CPU). If it passes, the floor stands and Task 6 adds a minimum-version CI job; if it fails for reasons unrelated to the migration, record the lowest passing version and make Task 6 raise the floor in `pyproject.toml` + `uv.lock` instead.
- [ ] **Step 2: Write `scripts/gen_kk_prejax_baseline.py` and run it** (`uv run python scripts/gen_kk_prejax_baseline.py`), commit the `.npz`.
- [ ] **Step 3: Write `test_oracle.py::test_oracle_agreement`** parametrized over `H, O, Si, SiO2, Fe2O3, Au, Pb, U, Mo`, 120 interior points per compound (same log-uniform distribution as above plus deterministic low-energy/near-edge/knot cases). Run it once with tolerances unset to print the per-compound maximum and median error of the current code, then set the per-energy tolerance to `max(3 × measured_max_for_compound, cancellation allowance from assert_kk_close)` (the measured value is written into the test in a comment); also assert median < 1e-8. Add `test_oracle_converges` (node-count doubling).
- [ ] **Step 4: Write `test_additivity.py::test_additivity`** for `SiO2, Fe2O3, CaCO3` at 200 log-uniform energies plus knots: compound f1/f2 equal `Σ nᵢ·f(elementᵢ)` within `assert_kk_close`-style allowance summed over the compound and the elements (measured on the current code first, then `3×`; the current code differs by up to 1.4e-7 absolute, so a fixed `rtol=1e-11` is not used).
- [ ] **Step 5: Write `test_fuzz.py::test_fuzz_valid_calls`**: 40 seeded random valid `kk_calculate_real` calls (formulas, explicit/default energies, optional `nearedge_example.txt`, `chunk_size` in {1, 7, 4096}, `refine_tolerance` in {None, 0.05}); assert finite outputs, input order preserved when unrefined, strictly increasing energies when refined.
- [ ] **Step 6: Run** `uv run pytest tests/unit/kk/test_oracle.py tests/unit/kk/test_additivity.py tests/unit/kk/test_fuzz.py --no-cov -q`. Expected: PASS on the current code (any failure beyond the measured tolerances is a pre-existing defect: report it, do not hide it).
- [ ] **Step 7: Write `scripts/kk_bench.py`** (standalone, not pytest-collected): fresh subprocess per cold measurement with `JAX_PLATFORMS` chosen before import; kernel m = 100/1k/10k/100k with `block_until_ready`; `kk_calculate_real("SiO2", refine_tolerance=0.01)` cold/warm; counts top-level kk programs through the compile recorder from Task 5 once it exists (until then it records `jax.log_compiles` lines from `jax._src.interpreters.pxla`). Run on CPU and paste the pre-migration table into its docstring.
- [ ] **Step 8: Commit** `test(kk): add pre-migration baseline, quadrature oracle, additivity and fuzz tests`.

### Task 1: Foundations — `_shapes`, `_device`, `Spectrum`, jnp-only kernel, `kk_f1` (§2–§4)

**Files:**
- Create: `xraylabtool/kk/_shapes.py`, `xraylabtool/kk/_device.py`
- Modify: `xraylabtool/kk/transform.py`, `xraylabtool/kk/__init__.py` (`_EXPORTS` + `__all__` += `Spectrum`, `kk_f1`), `docs/api/kk.rst` (document both, required by `tests/unit/kk/test_docs.py`)
- Test: `tests/unit/kk/test_shapes.py`, `tests/unit/kk/test_device_kernel.py`; adapt `tests/unit/kk/test_transform.py`, `tests/characterization/kk/test_transform_parity.py` per spec §9

**Interfaces:**
- Produces:
  - `_shapes.bucket(n: int, minimum: int = 512) -> int` — recurrence `b₀ = minimum`, `b_{k+1} = 8·floor(1.25·b_k/8)`; smallest `b ≥ n`; raises `ValueError` unless `minimum` is a multiple of 8 and ≥ 64 (the recurrence stalls below that); `_shapes.bucket_pow2(n: int, minimum: int = 8) -> int`; `_shapes.block_for(chunk_size: int | None, capacity: int, m: int) -> int`, `_shapes.energy_blocks(chunk_size, capacity, m) -> tuple[int, int]` (`m_b = bucket_pow2(m, minimum=1)`, `block = block_for(chunk_size, capacity, m_b)`, `n_blocks = ceil(m_b/block)`) (largest ladder/power-of-two value ≤ `min(chunk_size, 2**22 // capacity, m)`, at least 1).
  - `_device.ensure_x64() -> None`; `_device.upload(tree) -> tree`, `_device.fetch_status(tree) -> tree`, `_device.fetch(tree) -> tree` (each opens `jax.transfer_guard("allow")`, bumps a module-level counter exposed via `_device.transfer_counts() -> dict[str, int]` and `_device.reset_transfer_counts()`).
  - `_device.Spectrum` — frozen dataclass pytree: `seg_ev: jax.Array (N+1,)`, `coeffs: jax.Array (N, 5)`, `n_segments: jax.Array` int32 scalar (a leaf; unflatten performs no host validation); `Spectrum.from_numpy(seg_ev, coeffs, capacity: int | None = None) -> Spectrum` (validates through `_real_float64`, applies the padding rule); `Spectrum.to_numpy() -> tuple[NDArray, NDArray]` (slices to the true size).
  - `transform.kk_f1(eval_ev: jax.Array, spectrum: Spectrum, rel_corr: float | jax.Array, *, block: int | None = None) -> jax.Array` — shape `eval_ev.shape`, float64, masks coefficients by `n_segments`, `lax.map` over blocks, static `block`; empty input via a static branch; jit/vmap/grad-compatible.
  - Internal compiled entry `transform._kk_f1_blocks(padded_energies, spectrum, rel_corr, n_valid, *, block)` taking bucket-padded energies.
  - `transform.kk_piecewise_poly` keeps maths (Eq. 10 corrected a₋₃ sign, D5 conventions) on `jax.Array`; `kk_real_spectrum(...)` keeps its host signature/errors (pads energies to a bucket before upload, slices after download; empty in → empty out).
- Removes: `_jit_cache`, `_get_chunked_kernel`, NumPy chunk loop, `ops`/`ensure_float64` use in `kk/transform.py`.

- [ ] **Step 1: Write `test_shapes.py`**: `bucket(1)==512`, `bucket(512)==512`, `bucket(513)==640`; every bucket is a multiple of 8; `bucket(n) >= n` and monotone for n in 1..20000; for `n > 512`: `bucket(n) <= 1.25*n`; fewer than 14 distinct buckets for n in 512..5000; `bucket_pow2(9)==16`; `block_for(7, 1228, 1000) == 4`, `block_for(None, 1228, 1)==1`, never exceeds the request.
- [ ] **Step 2: Write `test_device_kernel.py`**:
  (a) `test_padding_neutral_at_knots` — Au and SiO2 from `prejax_baseline.npz`, padded to capacity `N`, `bucket(N+1)`, next bucket; energies: first knot, last real knot, three interior knots, a pad knot, midpoints, 10 random; all finite, equal to the baseline `f1` within `assert_kk_close`; plus `test_padding_neutral_synthetic` — a well-conditioned synthetic spectrum with all five basis terms nonzero, capacities N vs 2 buckets up, `rtol=1e-12`.
  (b) `test_kk_f1_jit_vmap` — `jax.jit(kk_f1, static_argnames="block")` equals eager; `jax.vmap` over two equal-capacity spectra with **different** `n_segments` equals a Python loop.
  (c) `test_kk_f1_grad` — gradient of `kk_f1(E, Spectrum(seg, c, jnp.int32(n)), rel).sum()` w.r.t. `c[:n]` and `E` (E away from knots) is finite and equals central finite differences (`rtol=1e-6`, per-basis-term step sizes, absolute allowance where the derivative is near zero); gradient w.r.t. padded rows is 0 (masking).
  (d) `test_output_float64_and_shapes` — dtype float64; empty, single-point, odd-length, exact-block and partial-block inputs return shape `(m,)` and equal the baseline.
  (e) `test_x64_restored_by_entry` — `jax.config.update("jax_enable_x64", False)` then `kk_real_spectrum` with float32 inputs still returns float64 (replaces `test_float32_inputs_give_float64_output`); restore the config in a fixture.
- [ ] **Step 3: Run.** Expected: FAIL (`kk_f1`/`Spectrum` undefined).
- [ ] **Step 4: Implement `_shapes.py` and `_device.py`** per the interfaces; register `Spectrum` with `jax.tree_util.register_pytree_node` (children = `(seg_ev, coeffs, n_segments)`, no aux data).
- [ ] **Step 5: Implement `kk_f1` and port `kk_piecewise_poly` to `jnp`.** Block size from `_shapes.block_for`; `_shapes.energy_blocks(chunk_size, capacity, m)` returns `(block, n_blocks)` with `m_b = bucket_pow2(m, minimum=1)`, `block = block_for(chunk_size, capacity, m_b)`, `n_blocks = ceil(m_b/block)` (padded length < 2m plus one block); pad energies with an in-range edge energy (only when `m ≥ 1`). If a non-finite appears at `E == x_n` or at a pad knot, fix it inside the kernel, not in the tests.
- [ ] **Step 6: Adapt existing tests per spec §9** (remove backend parametrisation and the `_jit_cache`/`set_backend` cache test; keep every golden/value/error assertion; keep chunk-size invariance checks). Add `Spectrum`/`kk_f1` entries to `docs/api/kk.rst`.
- [ ] **Step 7: Run** `uv run pytest tests/unit/kk tests/characterization/kk --no-cov -q`, `uv run mypy xraylabtool/kk --strict`, `uv run ruff check xraylabtool/kk tests/unit/kk`. Expected: PASS / clean.
- [ ] **Step 8: Commit** `feat(kk): jnp-only kernel, kk_f1 and padded Spectrum pytree`.

### Task 2: Device spectrum assembly and f2 (§5)

**Files:**
- Modify: `xraylabtool/kk/database.py` (device table cache), `xraylabtool/kk/spectrum.py`, `xraylabtool/kk/transform.py` (`relativistic_correction`)
- Test: `tests/unit/kk/test_spectrum.py` (existing assertions unchanged), `tests/unit/kk/test_device_spectrum.py`

**Interfaces:**
- Consumes: `Spectrum`, `_shapes.bucket`, `_device.upload`.
- Produces:
  - `AsfDatabase.device_tables() -> DeviceTables` — lazily built **under a lock (single-flight)**, cached per process and default device (the cache key includes the device): `knots: jax.Array (92, Kmax)` with padding set to the pad rule, `coeffs: (92, Kmax−1, 5)` with zero padding, `n_knots: (92,)` int32. Also `AsfDatabase.compound_capacity(z: Sequence[int]) -> int` = `bucket(Σ (n_knots − 1))` computed from the host tables.
  - `spectrum.compound_spectrum_device(z: jax.Array (k,), counts: jax.Array (k,), n_elements: jax.Array, tables: DeviceTables, *, capacity: int) -> Spectrum` — the element list is padded to a bucketed `k` (padded elements: count 0, masked out by `n_elements`; their knot rows are set to `+inf` before the union); **masks each gathered row's padding to `+inf` by `n_knots` before** `jnp.unique(size=capacity+1, fill_value=jnp.inf)`; true count from the mask; per-element coefficient index `clip(searchsorted(knots_i, seg_lower, 'right') − 1, 0, n_knotsᵢ − 2)` over the whole padded row (no traced-length slice); segments `≥ n_segments` get exact-zero coefficients; accumulation in the given element order.
  - `spectrum.evaluate_f2_device(energy_ev: jax.Array, spectrum: Spectrum) -> jax.Array` — right-continuous at knots; index clamped to `n_segments − 1`.
  - `transform.relativistic_correction` accepts device arrays; host call with Python lists keeps working.
  - Host wrappers `compound_spectrum(symbols, counts)` and `evaluate_f2(energy, seg, coeffs)` keep their contracts and errors.

- [ ] **Step 1: Write `test_device_spectrum.py`**: `test_device_matches_baseline` for `H, Mo, U, SiO2, Fe2O3` against `prejax_baseline.npz` spectra (`rtol=1e-13` on true-size arrays); `test_capacity_independent` (bucket and 2 buckets up give identical true-size arrays); `test_union_no_dummy_or_fill_leak` (true knots are exactly the sorted unique union of the *real* element knots — assert against a host `np.unique` of the real knots — all `seg_ev` finite and strictly increasing including padding, for H, Mo, U and a repeated-element formula `H2HO`-style list `["H","O","H"]`); `test_count_within_capacity` (`n_segments <= capacity` for every element pair among a sample incl. Mo+U); `test_evaluate_f2_device_edges` (first knot, last real knot uses the last real polynomial, a jump knot of Ar from the Ar baseline, one-element and empty input, pad region never selected); `test_relativistic_correction_device`.
- [ ] **Step 2: Run.** Expected: FAIL (functions undefined).
- [ ] **Step 3: Implement** `DeviceTables` packing (host packing is I/O preparation, uploaded once through `_device.upload`) and the device functions. Spike first: time `jnp.unique(size=…)` on CPU and GPU at the Mo+U size; if it is slow on GPU, use sort + mask-compaction (record the measurement in the commit message).
- [ ] **Step 4: Re-point the host wrappers** to the device functions (`Spectrum.to_numpy()` at the end); keep `_real_float64` validation and all messages; run existing `test_spectrum.py`, `test_input_hardening.py`.
- [ ] **Step 5: Run** `uv run pytest tests/unit/kk tests/characterization/kk --no-cov -q`; mypy/ruff on touched files. Expected: PASS.
- [ ] **Step 6: Commit** `feat(kk): device-resident compound spectrum, f2 evaluation and Z* correction`.

### Task 3: Device near-edge merge (§6)

**Files:**
- Modify: `xraylabtool/kk/nearedge.py`
- Test: `tests/unit/kk/test_nearedge.py` (existing assertions unchanged), `tests/unit/kk/test_device_merge.py`

**Interfaces:**
- Consumes: `Spectrum`, `evaluate_f2_device`, `_shapes.bucket`/`bucket_pow2`.
- Produces:
  - `nearedge.to_asf_scale_device(...)` with the same argument semantics as host `to_asf_scale` (host function wraps it; standalone tests unchanged).
  - `nearedge.MergeResult` — NamedTuple `(spectrum: Spectrum, scaled_f2: jax.Array, used_points_ev: jax.Array, status: MergeStatus)`; `MergeStatus` — NamedTuple of device scalars `(n_segments, v_low, v_high, a_low, a_high, measured_step, tabulated_step, nonfinite, n_negative, f2_min, n_inside)` (flatness checks compare each step with the end-point magnitudes and the messages print the values, as `nearedge.py:460–473` does today); padded near-edge entries (beyond `n_ne`) are masked out of every field.
  - `nearedge.merge_device(spec: Spectrum, ne_energy_ev: jax.Array, ne_f2: jax.Array, n_ne: jax.Array, low_ev: float | jax.Array, high_ev: float | jax.Array, *, capacity: int) -> MergeResult` where `capacity = bucket(spec_n_segments_upper_bound + n_nearedge + 2)`; source segment by **lower boundary**, right-continuous, clamped to the last real segment; no midpoints.
  - `nearedge.raise_or_log(status: MergeStatus-as-NumPy, ...) -> None` (host): raises today's `NearEdgeDataError` messages for flat end points / non-finite values and emits the D6 WARNING (count, minimum, inside count) exactly as today.
  - `merge_spectra`, `to_asf_scale`, `load_nearedge`, `NearEdgeSpectrum` keep host signatures/errors; `_merge` calls `merge_device`, `fetch_status`, `raise_or_log`.

- [ ] **Step 1: Write `test_device_merge.py`**: `test_merge_reproduces_goldens` (the `nearedge_example.txt` merge equals `prejax_baseline.npz` `merge_*` and `tests/characterization/kk/test_nearedge_golden.py`); `test_window_covers_no_knots`, `test_window_covers_all_knots`, `test_window_touches_range_ends_snaps`; `test_adjacent_float_knots_use_lower_boundary` (the case: `a=nextafter(1,inf)`, `b=nextafter(a,inf)`, `seg=[a,b,2,3]`, constant polynomials `c[:,1]=[10,20,30]` with `c[2,0]=1` so the tabulated f2 differs at the merge points 2 and 3 eV (otherwise the merge rejects a flat tabulated end), merge window `(2,3)` → outside-window coefficients unchanged: `out[0,1]==10.0`); `test_merge_point_on_source_knot`; `test_merge_bucket_invariant` (capacity at bucket and 2 buckets up identical on true-size arrays); `test_scaled_f2_covers_full_nearedge_curve` (values outside the replaced window present, equal to today's `nearedge_scaled_kev_f2`); `test_status_errors_unchanged` (flat measured end, flat tabulated end, NaN after normalisation → same `NearEdgeDataError` types and messages as today's tests); `test_negative_f2_warning_unchanged` (message, count, minimum, inside count).
- [ ] **Step 2: Run.** Expected: new tests FAIL.
- [ ] **Step 3: Implement** `to_asf_scale_device`, `merge_device` (static-capacity candidate construction: mask excluded entries to the sort tail, sort, `unique(size=…)`, pad rule) and the host `raise_or_log`; compute status without trapping on invalid arithmetic.
- [ ] **Step 4: Run** `uv run pytest tests/unit/kk tests/characterization/kk --no-cov -q`; mypy/ruff. Expected: PASS.
- [ ] **Step 5: Commit** `feat(kk): device-resident near-edge splice with status pytree`.

### Task 4: Device refinement passes (§7)

**Files:**
- Modify: `xraylabtool/kk/refine.py`
- Test: `tests/unit/kk/test_refine.py` (existing assertions unchanged except the chunk-size spy, see below), `tests/characterization/kk/test_refine_golden.py` (unchanged), `tests/unit/kk/test_device_refine.py`

**Interfaces:**
- Consumes: `kk_f1`, `evaluate_f2_device`, `Spectrum`, `_shapes.bucket_pow2`, `_device.fetch_status`.
- Produces:
  - `refine._PassState` — pytree of bucketed buffers: active intervals `lo, hi, f1_lo, f1_hi, f2_lo, f2_hi, pole_lo, pole_hi` (capacity `B`, power of two), accumulated points buffer (grown on demand from host-known counts: starts at `bucket_pow2(min(max_points, 4·(m−1)))` and is padded before a pass that could overflow it; never sized from `max_points` alone), counters.
  - `refine._refine_pass(state: _PassState, spec: Spectrum, rel_corr, tolerance, max_points_left, *, block: int) -> tuple[_PassState, _PassStatus]` jitted per active bucket `B` with static `block` (derived from the caller's `chunk_size` via `_shapes.block_for`, wired from both `refine_spectrum` and `kk_calculate_real`); `_PassStatus` = device int scalars `(n_next_active, n_accepted, n_checked, n_float_spacing, hit_max_points)` — only what the per-pass DEBUG line, reasons and the continue/stop/rebucket decision need (names indicative). The sliver count and widest-sliver statistic accumulate in `_PassState` and come with the final `fetch`.
  - The host loop: one `fetch_status` per pass, rebucket between passes when the new active count crosses a power-of-two boundary, same logging.
  - `refine_spectrum(...)` / `RefineOutcome` keep signatures and semantics; criterion stays the single absolute `tolerance`; the final sorted unique union is a device `argsort` followed by the single `fetch`.

- [ ] **Step 1: Read `xraylabtool/kk/refine.py` completely and write `test_device_refine.py`** pinning the semantics listed in spec §7: `test_refine_matches_golden_on_device` (existing refine golden and `prejax_baseline.npz` `refine_*`); `test_next_active_order_left_halves_then_right_halves` (compare internal order on a small constructed case against the pre-migration implementation's order, captured as a fixture from the baseline script); `test_max_points_strict_inequality_and_exact_equality`; `test_max_points_fires_on_pass_two_or_later`; `test_float_spacing_sticky_reason_and_continues`; `test_max_points_overrides_spacing_reason`; `test_max_passes_exhaustion`; `test_nothing_to_refine`; `test_sliver_only_final_pass_no_midpoint_log`; `test_both_endpoint_slivers`; `test_widest_sliver_uses_real_width`; `test_per_pass_debug_log_fields_unchanged` (caplog on the `xraylabtool` logger via the `kk_log` fixture); `test_refine_buffer_invariance` (forcing one bucket larger gives identical grid/values); `test_no_padded_point_in_result`; `test_one_status_fetch_per_pass` (helper counters: `fetch_status` calls == passes performed + assembly status; `fetch` == 1).
- [ ] **Step 2: Adapt both chunk-spy tests in `tests/unit/kk/test_refine.py`** (find them with `grep -n "_spy_chunk_sizes" tests/unit/kk/test_refine.py`; they are `test_refine_forwards_chunk_size` and its sibling near line 443, which today requires exact forwarding of 3): replace the spy on `refine.kk_real_spectrum` by an assertion of the block size seen by the device block driver (≤ the requested `chunk_size` for 1, 3, 7 and `None`).
- [ ] **Step 3: Run.** Expected: new tests FAIL.
- [ ] **Step 4: Implement** the pass kernel (cumulative-sum mask for truncation in the preserved order; compaction primitive per Task 2's measurement) and the host loop; reuse `_checked_inputs`. Prototype: count top-level programs compiled by a default SiO2 refined run (compile recorder from Task 5 Step 1 may be developed first as `tests/fixtures/kk_compiles.py`); if the count exceeds 6, adjust the ladder or merge programs, and record the final budget in the spec's Acceptance table.
- [ ] **Step 5: Run** `uv run pytest tests/unit/kk tests/characterization/kk --no-cov -q`; mypy/ruff (split pass helpers for PLR0915). Expected: PASS.
- [ ] **Step 6: Commit** `feat(kk): device-resident adaptive refinement`.

### Task 5: Orchestration, transfer and compile budgets (§1, §8, §9)

**Files:**
- Create: `tests/fixtures/kk_compiles.py` (version-tested compile recorder), `tests/unit/kk/test_pipeline_device.py`
- Modify: `xraylabtool/kk/api.py`, `xraylabtool/kk/result.py` (backend), `xraylabtool/gui/workers.py` / `xraylabtool/gui/services.py` (worker entry calls `_device.ensure_x64()` and opens its own guard scope when running under tests), kk-related `set_backend`/`ops` references
- Delete: `xraylabtool/backend/array_ops.py::ensure_float64` and `tests/unit/test_backend_kk_ops.py` (no other user: verified by `grep -rn ensure_float64 xraylabtool tests`), only after the x64 tests from Task 1 exist
- Adapt: `tests/unit/kk/test_api.py` (backend test → `backend == "jax"` + chunk invariance), `tests/characterization/kk/{test_end_to_end_golden,test_nearedge_golden}.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: `kk_calculate_real(...)` unchanged signature; internally validate on host → `_device.upload` (one call) → device pipeline → `_device.fetch_status` (once after assembly/merge) → refinement → `_device.fetch` (once) → bit-exact restore of requested keV values → `KKResult(backend="jax")`. `kk_compiles.record() -> ContextManager[list[CompileEvent]]` capturing JAX compile events from the logger that emits them on the installed jax (`jax._src.interpreters.pxla` today), with `test_recorder_sees_fresh_function` as its self-test; `kk_compiles.kk_programs(events) -> list[str]` filters to top-level kk programs by name.

- [ ] **Step 1: Write `kk_compiles.py` and its self-test**, then `test_pipeline_device.py`:
  `test_pipeline_under_disallow_explicit_guard` (device-only region run under `jax.transfer_guard("disallow_explicit")`; crossings only through `_device` helpers); `test_transfer_budget_warm_call` (counters: uploads 1, status fetch 1 + 1 per refinement pass, final fetch 1; the cold table upload counted separately, once); `test_second_call_no_recompile` (different values, same buckets → 0 kk programs compiled); `test_refined_run_program_budget` (≤ the budget recorded in the spec); `test_unsorted_duplicate_energies_order_preserved`; `test_ev_collision_first_representative_kept`; `test_requested_kev_bit_exact`; `test_backend_field_is_jax`; `test_set_backend_numpy_does_not_affect_kk` (fixture restores the backend); `test_x64_disabled_after_import_still_float64`; `test_threaded_calls_agree` (4 threads, **cold** subprocess variant with a `threading.Barrier` releasing the first callers simultaneously: exactly one table upload; then warm variant: results bit-equal to the serial call; each worker thread opens its own guard scope); `test_chunk_size_is_upper_bound` (block shape seen by the driver ≤ requested for 1, 3, 7).
- [ ] **Step 2: Run.** Expected: FAIL (guard trips on remaining host hops, extra fetches, compile count).
- [ ] **Step 3: Rewire `api.py`** to the device pipeline; chase every guard violation to its source and remove it (do not widen the allow-list beyond spec §1). Log platform/device/jax version once per process at INFO (module flag guarded by a lock).
- [ ] **Step 4: Remove NumPy-era code in `kk/`**; check with `grep -rnE "xraylabtool\.backend|\bops\." xraylabtool/kk` → no matches (the `KKResult.backend` field and its legacy validation remain by design). Delete `ensure_float64` and `test_backend_kk_ops.py` after confirming the grep over `xraylabtool/` and `tests/` shows no other user.
- [ ] **Step 5: Adapt the remaining backend-parametrised tests** per spec §9 (assertions of values unchanged).
- [ ] **Step 6: Run the whole tree exactly as the Makefile does**: `make test-fast` (it excludes GUI tests from the parallel pass and runs them serially), then `uv run ruff check . && uv run ruff format --check . && uv run mypy xraylabtool --strict`. Expected: all green.
- [ ] **Step 7: Run `scripts/kk_bench.py`** (CPU, fresh subprocesses) and check the spec's acceptance table CPU rows. Record the numbers in the commit message; any missed gate is reported and fixed before proceeding.
- [ ] **Step 8: Commit** `feat(kk): single-upload single-download JAX pipeline; drop NumPy twin`.

### Task 6: Docs, warm-up, compile cache, CLI/GUI end-to-end tests, CI, GPU gate (§10, G8)

**Files:**
- Create: `docs/architecture/005-adr-kk-jax-only.md`, `scripts/kk_gpu_check.py`, `tests/integration/test_kk_cli_end_to_end.py`, `tests/test_gui_kk_worker_end_to_end.py`, `tests/unit/kk/test_warm_up.py`
- Modify: `docs/architecture/index.rst` (add `005-adr-kk-jax-only`), `docs/architecture/jax_architecture.rst`, `docs/guides/kk_guide.rst`, `docs/api/kk.rst`, `docs/physics/kramers_kronig.rst` (the “NumPy backend agrees to 1.2e-12” sentence near line 292 and the accuracy discussion: restate on the JAX kernel against the locked goldens), `CHANGELOG.md` **and** `docs/development/changelog.rst` (kept in lockstep; `scripts/check_version.py` enforces it), `CLAUDE.md` (tracked; kk bullet and Architecture note: “kk is JAX-only”), `xraylabtool/kk/__init__.py` (`_EXPORTS["warm_up"]` and `__all__`), `xraylabtool/interfaces/cli/commands/` kk command, `xraylabtool/gui/__main__.py` (`main`, line ~32, where logging is configured), `xraylabtool/gui/main_window/kk_tab.py` (idle warm-up), `Makefile` (`GUI_TESTS` += `tests/test_gui_kk_worker_end_to_end.py`), `.github/workflows/ci.yml` (`IGNORE_GUI` at line ~326 and the serial list at line ~416 += the new GUI test; new minimum-jax CPU job per Task 0 Step 1, or floor bump in `pyproject.toml` + `uv.lock`)

**Interfaces:**
- Produces: `xraylabtool.kk.warm_up(formulas: Sequence[str] = ("SiO2",), refine_tolerance: float | None = 0.01) -> None` (compiles the buckets those formulas and that refinement touch; idempotent; `block_until_ready`; **called only by the GUI**); `xraylabtool.kk._device.enable_persistent_cache() -> bool` (called by the CLI `kk` command and the GUI launcher; sets `jax_compilation_cache_dir` to the OS user-cache directory + `xraylabtool/jax` only if unset and `XRAYLABTOOL_NO_JAX_CACHE != "1"`; also sets `jax_persistent_cache_min_compile_time_secs = 0` unless the user configured it; returns whether it enabled the cache; guarded by a lock).

- [ ] **Step 1: Write tests**: `test_warm_up_then_no_compile` (after `warm_up(("SiO2",))`, a default `kk_calculate_real("SiO2", ...)` and the default refined run compile 0 kk programs); `test_warm_up_exported_lazily` (`"warm_up" in kk.__all__`, `import xraylabtool` does not import `xraylabtool.kk`); `test_cache_helper_respects_opt_out_and_existing_config`, `test_cache_dir_is_os_appropriate` (monkeypatch platform/env), `test_library_import_does_not_enable_cache`, `test_cli_does_not_call_warm_up`.
  `test_kk_cli_end_to_end.py`: run `xraylabtool kk` in-process (monkeypatch `sys.argv`, call `main() -> int` from `xraylabtool/interfaces/cli/__init__.py`) and via subprocess for: valid compound → exit 0 and finite table; `--energies` list; `--refine 0.05`; `--nearedge tests/characterization/kk/nearedge_example.txt --merge …`; `-o file --format csv/json` round-trip; `--version` unaffected; error cases each exiting 1 with a stderr message: bad formula, non-finite energy, energy out of 10 eV–500 keV, missing near-edge file, flat near-edge end, `--merge` without near-edge, bad `--format`.
  `test_gui_kk_worker_end_to_end.py` (serial, `QT_QPA_PLATFORM=offscreen`, `QMessageBox.critical` stubbed): run a KK calculation on the worker thread → plot/table populated and matches `kk_calculate_real`; refined run; near-edge run; error path (bad formula) shows the stubbed dialog and leaves the UI responsive; window close while idle warm-up is running does not hang.
- [ ] **Step 2: Run.** Expected: FAIL (`warm_up`, `enable_persistent_cache` missing; CLI/GUI hooks absent).
- [ ] **Step 3: Implement** `warm_up`, `enable_persistent_cache`, the CLI/GUI hooks (GUI warm-up on an idle `QThreadPool` worker after the window is shown, not through the KK-result handlers), registrations, Makefile and CI edits.
- [ ] **Step 4: Write `scripts/kk_gpu_check.py`** (standalone, not collected): runs the oracle, additivity, transfer-budget (guard live on the device), padding invariance and benchmark checks on the default JAX device, prints a table, exits 1 on any miss against the spec's thresholds. Run it outside the sandbox on the RTX 4090; keep the output for the PR description, not the repo.
- [ ] **Step 5: Docs**: ADR-005 (kk is JAX-only; records the refuted ADR-001 “NumPy faster on CPU” assumption with the measured table; the `set_backend` consequence; padding/bucket/status design in brief); update `jax_architecture.rst` sections “NumPy vs JAX”, “Compilation in XRayLabTool”, “Data Flow”, “Memory Management”, “GPU Acceleration”, “Configuration & Control”; kk guide (device pipeline, warm-up, cache env var and thresholds, gradient example as a doctest); `docs/api/kk.rst`; the physics page sentence; both changelogs.
- [ ] **Step 6: Verify**: `make test-fast`; `uv run python scripts/check_version.py`; `uv run sphinx-build -b html docs docs/_build/html` and compare the warning list with a pre-change run of the same command (no new warnings; `tests/integration/test_kk_docs_build.py` filters to kk-related warnings and is not a whole-tree gate); `uv run ruff check . && uv run ruff format --check . && uv run mypy xraylabtool --strict`. Expected: all green.
- [ ] **Step 7: Run `graphify update .`** and commit the refreshed tracked graph files separately: `chore: refresh graphify graph`.
- [ ] **Step 8: Commit** `docs(kk): ADR-005 and JAX-first pipeline documentation; warm-up, compile cache, CI`.

---

## Self-review notes

- **Spec coverage:** G1 (Task 5 Step 4), G2 (Task 5 tests), G3 (Tasks 1, 4, 5), G4 (Task 1), G5 (Tasks 0–5 keep existing assertions; spec §9 lists each necessary test change), G6 (Task 5 Step 7, Task 6 Step 4), G7 (Tasks 1, 5), G8 (Task 6). Verification items 1–10 map to Tasks 0, 1, 2 (+0 for additivity), 3, 4, 5, 6 as before; the review's added contracts (capacity bounds → Task 2/3, status pytrees → Task 3/4, transfer counters → Tasks 1/5, compile recorder → Task 5) each have a named test.
- **Types:** `Spectrum`, `kk_f1`, `bucket`, `bucket_pow2`, `block_for`, `compound_spectrum_device`, `evaluate_f2_device`, `MergeResult`/`MergeStatus`, `merge_device`, `_refine_pass`/`_PassStatus`, `_device.upload/fetch_status/fetch`, `warm_up`, `enable_persistent_cache` keep identical names wherever consumed.
- **Builder decisions left open (spec “Left to the builder”):** compaction primitive (measured in Task 2), exact `_PassStatus` field names, private module layout. First risks to retire: padding at knots (Task 1), `jnp.unique(size=)` GPU cost (Task 2), refined-run program count (Task 4).
