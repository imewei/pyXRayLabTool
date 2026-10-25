# kkcalc Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a JAX-first Kramers-Kronig module (`xraylabtool.kk`) that turns a formula plus an optional measured near-edge spectrum into f1/f2, exposed through a Python API, an `xraylabtool kk` CLI subcommand and a "Kramers-Kronig" GUI tab.

**Architecture:** Port upstream kkcalc's `KK_PP` piecewise-Laurent-polynomial transform as a pure `ops`-based kernel (jit-lazy like `calculators/kernels.py`), driven by vendored upstream `ASF.json` repacked to `.npz`. Host-side code handles near-edge loading/merging and adaptive refinement. CLI and GUI are thin layers over the core. Stages ship in order: core → CLI → GUI → docs.

**Tech Stack:** Python ≥3.12, NumPy/JAX via `xraylabtool.backend.ops`, PySide6 + PyQtGraph, pytest, mypy `--strict`, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-10-24-kkcalc-integration-design.md` (read it first, including "Review round 1" and decisions D1–D4).

**Status (as built):** all 12 tasks executed on `feat/kk-integration` (PR #31). Line numbers quoted below are insertion points in the base commit `19d86f8` and have shifted in the built tree. Deviations from the text, each decided during execution: `kk_calculate_real` lives in `kk/api.py` (the lazy `__getattr__` in `kk/__init__.py` needs a submodule); `KKResult`, `NearEdgeSpectrum` and `RefineOutcome` are `@dataclass(frozen=True, eq=False)` (array fields make generated `__eq__` ambiguous); `RefineOutcome` gains `n_pole_skipped: int = 0`; `refine_spectrum` and `kk_calculate_real` forward `chunk_size`; `CLAUDE.md` is git-ignored (`.gitignore:215`), so its kk note is local only; added files not listed here: `tests/fixtures/kk_rounding.py` (JAX/golden rounding-budget helper), `tests/unit/kk/conftest.py` (`kk_log` fixture, because the `xraylabtool` logger has `propagate=False`), `tests/unit/kk/test_container_validation.py` (frozen-container invariants of `KKResult` and `RefineOutcome`), `docs/_static/kk/{make_truncation_figure.py,truncation.png}` (Task 11 figure and its generator), `tests/characterization/kk/test_refine_golden.py`, `scripts/gui_headless_smoke.py`, `.gitignore` negations for the vendored data, `README.md`, `CHANGELOG.md`. Post-review hardening (commits 5a3bfaf, 2c854ad, 09653eb, 21a77b0): cached ASF arrays are read-only; `calculate_properties_from_factors` rejects complex inputs; a merge that leaves no knot in 0.03–30 keV raises `EnergyError` when no explicit energies are given; the GUI start/end range is 0.01–500 keV (default 1–30).

## Global Constraints

- Work was done on branch `feat/kk-integration` and merged to `main` as c5dbe45 (PR #31); the commit steps below are the commits that were made.
- Python ≥ 3.12; `mypy xraylabtool/ --strict` and `ruff check/format` clean; NumPy-style docstrings; explicit imports; type hints on public APIs.
- `filterwarnings = error` — but `pyproject.toml:255-256` *ignores* divide-by-zero/invalid-value warnings, so every numerical test asserts finiteness explicitly (`np.isfinite(...).all()`).
- Public API energy in **keV**; the transform runs in **eV** internally (ASF coefficients are in eV); convert only at API boundaries. File I/O energy unit defaults to eV (`--energy-unit eV|keV`); `--merge` follows `--energy-unit` (D4).
- Core accepts **10 eV–500 keV** (ASF data range); out-of-range → `EnergyError`, never extrapolate. Default output grid (no explicit energies) = the data's own grid clipped to 0.03–30 keV with an INFO log of the clip. Only `to_xray_result` enforces 0.03–30 keV (D3).
- Port upstream `KK_PP` (fixed orders `E¹,E⁰,E⁻¹,E⁻²,E⁻³`, 5 coefficient columns, **no `orders` argument**) — D1. Do not port `KK_General_PP`, "add background" or "fix distortions".
- Kernel is float64 always: enable x64 before tracing, cast inputs explicitly. Kernel never calls `ops.any/all` (they return Python `bool`, raise `TracerBoolConversionError` under jit) and has no data-dependent Python branching.
- Vendor `ASF.json` as-is (D2), repacked to `.npz`; pin **one** upstream commit SHA (upstream's `main` and `master` files differ in size). Zlib: keep the complete upstream notice, mark altered files "altered from the original", ship licence text in sdist and wheel, conda recipe `license: MIT AND Zlib`.
- Tolerances: transform-only parity on identical arrays `rtol≈1e-10` (JAX `1e-8`); end-to-end with near-edge data uses a tolerance set from the measured error (repo `hc`/Avogadro/`r_e` differ from upstream by ~1.3e-7/1.2e-7/8e-9). Chunk invariance is within tolerance, not bit-for-bit.
- Lazy imports: `import xraylabtool` must not import `xraylabtool.kk`. GUI tests run serially (not under xdist).
- Existing `parse_energy_string("nan")` and `parse_formula("Xx0")` do **not** reject bad input; `kk` validates its own.
- Review round 2 calls (mine, flagged for Wei): **D5** poles — at an interior knot where f2 is continuous the result is the continuous principal value; at a jump in f2 or at a domain endpoint with nonzero f2 (measured in Task 2: real f2 jumps exist only for He, Ne, Ar, Kr and negligibly Xe — tabulated absorption edges are narrow continuous segments, so the default grid rarely hits a jump; jump goldens come from `ar.npz`, `ne.npz`, `synthetic.npz`, key `eval_kind`==2) the kernel returns upstream's finite value, pinned by a golden point, documented as a convention (the ordinary integral diverges there). **D6** negative f2 after near-edge normalisation is kept, logged at WARNING (count and minimum), never clipped and never rejected; non-finite values are rejected. **D7** refinement works on a sorted, unique working grid: with `refine_tolerance` set, the output is the sorted unique union of the requested energies and the added midpoints.
- Reference paper: B. Watts, *Opt. Express* **22**(19), 23628–23639 (2014), doi:10.1364/OE.22.023628 (PDF read in review). Its typeset Eq. (10) has a **sign error in the a₋₃ term** (printed `+2a₋₃/E·(x⁻¹ⱼ₊₁−x⁻¹ⱼ)`; the correct term is `−2a₋₃/E·(x⁻¹ⱼ₊₁−x⁻¹ⱼ)` = `+2a₋₃/E·(x⁻¹ⱼ−x⁻¹ⱼ₊₁)`, as in Table 1, upstream code and Cauchy quadrature; subscripts are correct); implement and document the corrected form. Every public `kk` function carries a NumPy-style docstring with units, shapes, raised exceptions, a doctest-able `Examples` block, and a `References` section citing the paper equation it implements (or saying it is upstream-only). Never copy paper text or figures into docs (© Optica).
- CLI error policy: `main()` converts argparse failures to return code 1 (`interfaces/cli/__init__.py:114-120`); every `kk` usage and data error therefore exits **1** with a stderr message. Do not change that wrapper.
- `--energies` is always keV (repo convention); `--energy-unit` governs only the near-edge file and `--merge`.
- GUI tests run serially: any new GUI test file is added to `Makefile` `GUI_TESTS` (line 60), the `IGNORE_GUI` list in `.github/workflows/ci.yml` (line 325) and the serial GUI `pytest` call (line 410).

## Review Focus

Failure modes the spec implies but no happy-path test exercises, most likely first:
1. Eval energy exactly on a segment edge, duplicated, or unsorted → finite; continuous-f2 knots equal their neighbours within tolerance, jump knots and endpoints follow D5 (Task 3).
2. Near-edge merge with flat endpoints, reversed merge points, or merge points outside the data → clear `NearEdgeDataError`, never `inf`/silent garbage (Task 5).
3. Near-edge file declared eV but really keV (energies ≈ 0.28) → "no overlap with tabulated range" error whose message states the file's parsed energy range and the required range (Task 5; CLI Task 8).
4. Negative f1 near an edge → preserved through to `XRayResult`, critical angle 0, no error (Task 7).
5. JAX with float32 inputs or x64 disabled before first call → still float64 results (Task 3).

---

## File Structure

```
xraylabtool/kk/__init__.py        lazy public API (`__all__` == `_EXPORTS`): kk_calculate_real, KKResult, NearEdgeSpectrum, load_nearedge, ...
xraylabtool/kk/api.py             kk_calculate_real() (validation, merge, kernel, refine)
xraylabtool/kk/database.py        load_asf() -> AsfDatabase (lazy, cached)
xraylabtool/kk/spectrum.py        compound_spectrum(), evaluate_f2()
xraylabtool/kk/transform.py       relativistic_correction(), kk_piecewise_poly(), kk_real_spectrum(), _jit_cache
xraylabtool/kk/nearedge.py        NearEdgeSpectrum, load_nearedge(), to_asf_scale(), merge_spectra()
xraylabtool/kk/refine.py          refine_spectrum(), RefineOutcome
xraylabtool/kk/result.py          KKResult
xraylabtool/data/kkcalc/{asf.npz, LICENSE-kkcalc.txt}
docs/physics/kramers_kronig.rst   physics note grounded in Watts 2014 (own wording/figures)
docs/api/kk.rst                   automodule pages for xraylabtool.kk and submodules
docs/guides/kk_guide.rst          API / CLI / GUI usage, D1-D7, limits, attribution
scripts/build_kk_data.py          upstream ASF.json -> asf.npz (pinned SHA)
scripts/gen_kk_golden.py          dev-only oracle: runs pinned upstream, writes golden files
xraylabtool/interfaces/cli/commands/kk.py
xraylabtool/gui/main_window/kk_tab.py     KKTabMixin
xraylabtool/gui/widgets/kk_plot.py        KKPlot
tests/{unit/kk,characterization/kk,integration}/...
```

Modified (line numbers refer to base `19d86f8`): `backend/array_ops.py`, `calculators/core.py`, `xraylabtool/__init__.py:72`, `interfaces/cli/{parser.py,__init__.py:166}`, `interfaces/completion_v2/shells.py:711`, `gui/main_window/{__init__.py,_protocol.py,shared.py:88}`, `gui/services.py`, `exceptions.py`, `pyproject.toml:16,198`, `conda-recipe/meta.yaml:54-55`, `Makefile:60`, `.github/workflows/ci.yml:325,410`, `tests/integration/test_cli.py:511`, `scripts/gui_headless_smoke.py`, `.gitignore`, `README.md`, `CHANGELOG.md`, docs; `CLAUDE.md` is git-ignored (local note only).

---

### Task 1: Backend primitives and float64 guard

**Files:**
- Modify: `xraylabtool/backend/array_ops.py` (Protocol, `NumpyBackend`, `JaxBackend`)
- Test: `tests/unit/test_backend_kk_ops.py`

**Interfaces:**
- Produces: `ops.log(x)`, `ops.abs(x)` (plus any other primitive Task 3 proves it needs, added to all three classes), and `ensure_float64(x: Any) -> Array` (module-level in `array_ops.py`; casts to float64 on the active backend and, for JAX, calls `jax.config.update("jax_enable_x64", True)` first).

- [ ] **Step 1: Write failing tests** — `test_log_abs_parity_numpy_jax` (importorskip jax; `ops.abs` on a mixed-sign array and `ops.log` on strictly positive values, both equal to `np.abs/np.log` at `rtol=1e-15` with `np.isfinite(...).all()`; no assertion on `log` of non-positive input — pole safety is the kernel's job in Task 3), `test_ensure_float64_casts_float32` (numpy and jax: `ensure_float64(np.float32([1,2])).dtype == float64`), `test_ensure_float64_enables_x64_under_jax`.
- [ ] **Step 2:** Run `uv run pytest tests/unit/test_backend_kk_ops.py -v --no-cov` → FAIL (`AttributeError`).
- [ ] **Step 3:** Add the methods to `ArrayBackend`, `NumpyBackend`, `JaxBackend` (same delegation style as `sqrt`) and `ensure_float64`.
- [ ] **Step 4:** Re-run → PASS. Run `uv run mypy xraylabtool/backend --strict`.
- [ ] **Step 5:** Commit `feat(backend): add log/abs ops and ensure_float64`.

### Task 2: Vendored data and upstream oracle

**Files:**
- Create: `scripts/build_kk_data.py`, `scripts/gen_kk_golden.py`, `xraylabtool/data/kkcalc/{asf.npz,LICENSE-kkcalc.txt}`, `xraylabtool/kk/__init__.py` (empty docstring for now), `xraylabtool/kk/database.py`, `tests/characterization/kk/golden/*.npz`, `tests/characterization/kk/nearedge_example.txt` (small upstream example, with attribution), `NOTICE`
- Modify: `pyproject.toml:198-202` (package-data), `conda-recipe/meta.yaml:54-55`
- Test: `tests/unit/kk/test_database.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) class AsfDatabase` with `symbols: tuple[str, ...]`, `mass: NDArray[float64]` (92,), and per-Z (1-based) `energy_ev(z: int) -> NDArray` shape `(n+1,)`, `im_coeffs(z: int) -> NDArray` shape `(n, 5)`; `load_asf() -> AsfDatabase` (`functools.cache`d, lazy). Golden files hold, per case, the arrays `eval_ev, seg_ev, coeffs, rel_corr, f1_kk_pp` (transform-only; `eval_ev` includes points exactly on knots, one at an f2 jump, and one at the first/last edge, for D5) and, for formulas, `energy_ev, f1, f2` from upstream `kk_calculate_real` on its **full** integration grid (unclipped). Separate `refine_*.npz` files hold, per refined case, the input grid, `seg_ev, coeffs, rel_corr`, the absolute tolerance, upstream's recursion/arguments, and the refined `energy_ev, f1, f2`.

- [ ] **Step 1: Write failing tests** — `test_asf_has_92_elements`, `test_asf_grid_spans_10ev_to_500kev` (all Z: `energy_ev[0]==10.0`, `[-1]==500000.0`), `test_im_coeffs_shape` (`(len(E)-1, 5)`), `test_pyproject_package_data_lists_kk_files` (parse `pyproject.toml`; both `data/kkcalc/asf.npz` and `data/kkcalc/LICENSE-kkcalc.txt` present), `test_nff_divergent_elements` (evaluate the ASF piecewise f2 at each `.nff` file's own tabulated energies, 30 eV–30 keV, and compare to the `.nff` f2 column with **no** interpolation of the `.nff` data — repo PCHIP or linear interpolation alone differs by >1e-6 for every element, so interpolation must not enter this test; the set of elements with any relative deviation > 1e-6 equals `{Sc, Mo, Ru, Be, Ti, Si, La, Gd, W}`. If this procedure does not reproduce exactly that set, stop and report; do not change the set or the threshold without Wei's sign-off. Record the procedure in the test docstring).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Pin upstream `main` at `f331ca31a654af946f7d550a22cf4e9c11fa8afc` (2024-07-19); its `kkcalc/ASF.json` has sha256 `9edf4fac0d8dbc0552938abb106039a71c34e9c7ea71969964a41b56eaf79f66` (verify in the build script). Checked during plan review: unmodified upstream `KK_PP`, `kk_calculate_real` (also with `curve_tolerance`) and `data.calculate_asf` run on NumPy 2.5.4 with all-finite output for Si, Au, SiO2, CH and the polystyrene example, so **no compat patch is needed**; the oracle runs under `python -W ignore` because upstream emits an import-time `SyntaxWarning` (`data.py:217`) and unclosed-file `ResourceWarning`s, which `filterwarnings = error` would turn into failures. Do not use upstream's `test_kk.py` as an oracle: it calls signatures that no longer exist. `scripts/build_kk_data.py`: download upstream `ASF.json` at the pinned SHA constant, write `asf.npz` (keep only `E`, `Im`, `mass`, `symbol`; record SHA in the npz and in `NOTICE`). `scripts/gen_kk_golden.py`: clone upstream at the same SHA into a temp dir, run `KK_PP` on fixed arrays and `kk_calculate_real` for `Si`, `Au`, `SiO2`, `C8H8` (polystyrene, with the example near-edge file, `merge_points` as in upstream's example), plus upstream `improve_accuracy` on the `Si` and `SiO2` spectra at a recorded absolute tolerance, and write the golden `.npz` files; apply and list any NumPy-compat patch. Never import upstream from the package. Implement `load_asf()`.
- [ ] **Step 4:** Edit package-data (add `data/kkcalc/asf.npz` and `data/kkcalc/LICENSE-kkcalc.txt` explicitly, as built; setuptools' default `license-files` patterns already ship a root `NOTICE`), `pyproject.toml:16` (`license = "MIT AND Zlib"`, SPDX), conda recipe (`license: MIT AND Zlib`; `license_file` at `meta.yaml:54` becomes a list: `LICENSE`, `NOTICE`, `xraylabtool/data/kkcalc/LICENSE-kkcalc.txt`), write `NOTICE`. Task 12's build test is what proves sdist and wheel contain them. Run the scripts, commit generated data. Re-run tests → PASS.
- [ ] **Step 5:** Commit `feat(kk): vendor kkcalc ASF data and golden oracle`.

### Task 3: Transform kernel

**Files:**
- Create: `xraylabtool/kk/transform.py`
- Modify: `xraylabtool/backend/array_ops.py` (`set_backend` clears `kk.transform._jit_cache` in the same try/except style as the existing `_jit_cache` clear)
- Test: `tests/unit/kk/test_transform.py`, `tests/characterization/kk/test_transform_parity.py`

**Interfaces:**
- Consumes: Task 1 `ops.log/abs`, `ensure_float64`; Task 2 golden files.
- Produces: `relativistic_correction(atomic_numbers: Sequence[int], counts: Sequence[float]) -> float` (= `Σ nᵢ(Zᵢ − (Zᵢ/82.5)^2.37)`); `kk_piecewise_poly(eval_ev: Array, seg_ev: Array, coeffs: Array, rel_corr: float) -> Array` (pure, jit-safe, `seg_ev` has n+1 edges, `coeffs` n×5 columns `A₁,A₀,A₋₁,A₋₂,A₋₃`); `kk_real_spectrum(eval_ev, seg_ev, coeffs, rel_corr, chunk_size: int | None = None) -> NDArray[float64]` (chunks over `eval_ev`, `lax.map` only on JAX, a Python loop on NumPy; `_jit_cache: dict[str, Any]` as in `kernels.py`).

- [ ] **Step 1: Write failing tests**
  - `test_matches_upstream_kk_pp_golden` — every golden case: `kk_real_spectrum(...)` vs `f1_kk_pp`, `rtol=1e-10` (JAX `1e-8`), `np.isfinite(out).all()`.
  - `test_constant_band_closed_form` — one segment `[a,b]`, `coeffs=[[0,c,0,0,0]]`, `rel_corr=0`, E well outside: `f1 == (c/π)·ln|(E²−a²)/(E²−b²)|`, `rtol=1e-12`. If the sign or factor disagrees with the golden data, stop and report; do not flip the test to match.
  - `test_matches_independent_principal_value_quadrature` — a multi-segment spectrum in which **all five coefficient columns are nonzero** (a spectrum without a₋₂/a₋₃ cannot see the paper's Eq. 10 a₋₃ sign slip) evaluated by `scipy.integrate.quad(..., weight="cauchy")` (test-only use of SciPy, not a runtime dependency); agrees to `rtol=1e-8` away from edges, so upstream is not the only definition of correct (D1). Parametrise a companion `test_each_basis_term_matches_quadrature` over the five columns, one nonzero at a time, with eval energies below, inside and above the segment (checked in plan review for `[a₁,a₀,a₋₁,a₋₂,a₋₃] = [0.3,−0.7,1.1,2.3,−1.9]` on `[2,5]` at E=1, 3.3, 7: quadrature, upstream and the corrected Eq. 10 agree to ~1e-15).
  - `test_matches_paper_eq10_as_corrected` — an independent plain-NumPy transcription of Eq. (10) in the test file (first bracket `E²a₁−Ea₀+a₋₁−a₋₂/E+a₋₃/E²` times `ln|(xⱼ₊₁+E)/(xⱼ+E)|`, minus second bracket `E²a₁+Ea₀+a₋₁+a₋₂/E+a₋₃/E²` times `ln|(xⱼ₊₁−E)/(xⱼ−E)|`, `−2Ea₁(xⱼ₊₁−xⱼ)`, `+(2a₋₂/E)ln|xⱼ₊₁/xⱼ|`, `+(2a₋₃/E)(`**`xⱼ⁻¹−xⱼ₊₁⁻¹`**`)` (the typeset paper has the opposite sign, `+(2a₋₃/E)(xⱼ₊₁⁻¹−xⱼ⁻¹)`), all times `1/(πE)`, plus Z*) at off-knot energies; against the kernel; in review the corrected transcription differed from upstream by 6e-8 (Si), 1e-9 (CH) and 8e-5 absolute for Au (values up to 79, cancellation in the logs), so set the tolerance from the measured agreement per element rather than a blanket `rtol=1e-10`. `test_printed_eq10_a3_sign_is_wrong` — on the single-segment case above the printed-sign variant `+(2a₋₃/E)(xⱼ₊₁⁻¹−xⱼ⁻¹)` misses quadrature by > 1e-3 while the corrected term agrees; this pins the documented erratum.
  - `test_single_lorentz_oscillator_closed_form` — spec Verification 2. `f2 = Aγω/((ω0²−ω²)²+γ²ω²)` sampled piecewise-linearly on a dense log grid over `[1e-3·ω0, 1e3·ω0]`, `rel_corr=0`; the x-ray-convention result is `f1 = A(ω²−ω0²)/((ω0²−ω²)²+γ²ω²)` — note the sign is **opposite** to the optical `Re χ`, because `f1−Z*` is positive above resonance (as the constant-band case). The sign was checked against `scipy.integrate.quad(weight="cauchy")` during plan review (A=1, γ=0.3, ω0=1: −1.28205 at ω=0.5, +1.36139 at ω=1.2); do not flip it to match the kernel. Assert agreement across the pole at two grid densities: error is below a documented bound and shrinks as the grid is refined (truncation tails bounded analytically), so the test shows convergence, not tuning.
  - `test_interior_knot_with_continuous_f2_is_finite_and_continuous` — E at a knot where f2 is continuous, and `nextafter(edge, ±inf)`: all finite, agree within a documented tolerance, and `jax.grad` w.r.t. `eval_ev` is finite. `test_jump_and_endpoint_follow_upstream_convention` (D5) — E exactly at a knot where f2 jumps and at the first/last edge: finite and equal to the golden upstream value; docstring states this is a convention, not the divergent ordinary integral.
  - `test_unsorted_duplicate_energies` — shuffled input with repeats returns values equal per energy to the sorted run.
  - `test_chunk_invariance` — chunk sizes 1, 7 (non-dividing), larger than `n_eval`; all within `rtol=1e-12` of unchunked.
  - `test_jax_numpy_parity`, `test_kernel_jits_without_tracer_error` (importorskip jax), `test_float32_inputs_give_float64_output` (JAX, including x64 initially disabled), `test_set_backend_clears_kk_cache`.
- [ ] **Step 2:** Run `uv run pytest tests/unit/kk/test_transform.py tests/characterization/kk -v --no-cov` → FAIL.
- [ ] **Step 3:** Port upstream `KK_PP` and `calc_relativistic_correction` as broadcasting (no `tile`), with pole masks and the double-`where` pattern around `log` so neither value nor gradient is NaN at edges. Neighbouring-segment singularities and endpoints follow upstream's treatment; document each case in the docstring. Default chunk size constant chosen so the largest tensor stays under ~256 MB at float64. Docstrings for `relativistic_correction` (Eq. 9), `kk_piecewise_poly` (Eqs. 7 and 10, corrected a₋₃ sign, D5 conventions at knots/ends) and `kk_real_spectrum` follow the Global Constraints docstring rule.
- [ ] **Step 4:** Re-run → PASS; `uv run mypy xraylabtool/kk --strict`.
- [ ] **Step 5:** Commit `feat(kk): JAX-first KK_PP transform kernel`.

### Task 4: Compound spectrum, `KKResult`, tabulated-only API

**Files:**
- Create: `xraylabtool/kk/spectrum.py`, `xraylabtool/kk/result.py`, `xraylabtool/kk/api.py` (`kk_calculate_real`); flesh out `xraylabtool/kk/__init__.py`
- Modify: `xraylabtool/__init__.py:72` (add `"kk"` to `_LAZY_SUBMODULES`)
- Test: `tests/unit/kk/test_spectrum.py`, `tests/unit/kk/test_api.py`, `tests/characterization/kk/test_end_to_end_golden.py`

**Interfaces:**
- Consumes: `load_asf`, `kk_real_spectrum`, `relativistic_correction`, `xraylabtool.utils.parse_formula`, `ops`.
- Produces: `compound_spectrum(symbols: Sequence[str], counts: Sequence[float]) -> tuple[NDArray, NDArray]` (`seg_ev` on the union of elemental boundaries, `coeffs` summed with counts); `evaluate_f2(energy_ev, seg_ev, coeffs) -> NDArray` (at the final edge use the last polynomial, as upstream does); `@dataclass(frozen=True, eq=False) class KKResult(formula: str, energy_kev: NDArray, f1: NDArray, f2: NDArray, relativistic_correction: float, merge_points_kev: tuple[float, float] | None, n_refined_points: int, backend: Literal["numpy", "jax"], nearedge_scaled_kev_f2: tuple[NDArray, NDArray] | None = None)` (the last field is the near-edge data on the ASF scale on its own energy grid, filled in Task 5, so a plot can draw it without a second computation); `kk_calculate_real(formula: str, *, energies_kev: ArrayLike | None = None, chunk_size: int | None = None) -> KKResult` (near-edge and refine parameters are added in Tasks 5–6).

- [ ] **Step 1: Write failing tests** — `test_end_to_end_matches_upstream_golden` (Si, Au, SiO2: call the API with `energies_kev = golden energy_ev/1000` — explicit energies may lie anywhere in 10 eV–500 keV — and compare to the full-grid golden `f1, f2` at `rtol=1e-10`), `test_compound_grid_is_union_of_boundaries`, `test_default_grid_clipped_to_0p03_30_kev_and_logged` (caplog INFO; asserts only the clip and its log, plus that the clipped values equal the golden values masked to 30–30000 eV), `test_explicit_energy_between_10ev_and_500kev_ok`, `test_energy_out_of_data_range_raises_energy_error` (9 eV, 501 keV), `test_invalid_energies_raise_energy_error` (NaN/inf/empty energies → `EnergyError`) and `test_invalid_formulas_raise_formula_error` (`"Xx0"`, `""`, zero total composition, unknown element → `FormulaError`), `test_keV_input_equals_internal_eV` (same physics via two unit paths), `test_import_xraylabtool_does_not_import_kk` (fresh subprocess; `"xraylabtool.kk" not in sys.modules` until `xraylabtool.kk` is accessed), `test_kk_result_is_frozen`, `test_tabulated_kk_f1_close_to_cxro_f1` (30 eV–30 keV, for elements outside the nine divergent ones; tolerance documented in the test from the measured agreement; the paper notes small deviations are expected because the published CXRO f2 values are a discrete sampling of the piecewise function), `test_30kev_truncation_matches_paper_trend` (the paper's Figs. 3–5 behaviour: clip `seg_ev, coeffs` at 30 keV and compare with the full 500 keV integration at E = 0.1, 1, 10, 29 keV; measured with upstream during review as `f1_full − f1_trunc`: C ≈ −1e-4 (assert `<1e-3`), Fe ≈ −0.10 to −0.28 (assert between 0.05 and 0.5 in magnitude), Pb ≈ −2.5 to −5.2 (assert `>1`)).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. `kk_calculate_real` validates (finite, supported elements, positive composition, in-range energies), converts keV→eV once, runs the kernel, evaluates f2, records `backend` from `get_backend()`. `kk_calculate_real` is defined in `kk/api.py`; public names are exported lazily via module `__getattr__` in `kk/__init__.py`.
- [ ] **Step 4:** Re-run → PASS; mypy strict.
- [ ] **Step 5:** Commit `feat(kk): compound spectrum, KKResult and kk_calculate_real`.

### Task 5: Near-edge loading, conversion and merging

**Files:**
- Create: `xraylabtool/kk/nearedge.py`
- Modify: `xraylabtool/exceptions.py` (add `NearEdgeDataError(ValidationError)`), `xraylabtool/kk/api.py` (`kk_calculate_real`, which also fills `KKResult.nearedge_scaled_kev_f2`)
- Test: `tests/unit/kk/test_nearedge.py`, `tests/characterization/kk/test_nearedge_golden.py`

**Interfaces:**
- Consumes: `compound_spectrum`, `kk_real_spectrum`, Task 2 example file and golden polystyrene case.
- Produces: `@dataclass(frozen=True, eq=False) class NearEdgeSpectrum(energy_ev: NDArray, values: NDArray)` whose `__post_init__` runs the same validation as the loader (so a directly constructed spectrum cannot bypass it); `load_nearedge(path: str | Path, *, energy_unit: Literal["eV","keV"] = "eV") -> NearEdgeSpectrum` (first column energy, last column absorption; validates shape, dtype, NaN/inf, strictly increasing energy, ≥2 points); `to_asf_scale(spec: NearEdgeSpectrum, input_type: Literal["nexafs","photoabsorption","xanes","beta","asf"], *, density: float | None = None, formula_mass: float | None = None) -> NearEdgeSpectrum`; `merge_spectra(spec, seg_ev, coeffs, *, merge_points_ev: tuple[float, float] | None) -> tuple[NDArray, NDArray]`. `kk_calculate_real` gains keyword-only `nearedge: NearEdgeSpectrum | None = None`, `input_type: ... | None = None` (required when `nearedge` is given → `TypeError`), `merge_points_kev: tuple[float, float] | None = None`.

- [ ] **Step 1: Write failing tests** — `test_load_rejects_non_increasing_nan_and_short`, `test_keV_unit_equals_eV_unit` (same file scaled, same result), `test_merge_requires_ordered_points_inside_both_datasets`, `test_flat_endpoints_raise_not_inf` (Review Focus 2), `test_file_in_kev_declared_ev_raises_no_overlap` (Review Focus 3; message contains both the file's parsed range and the required range), `test_directly_constructed_spectrum_is_validated`, `test_negative_normalised_f2_kept_and_warned` (D6: affine matching that makes an interior value negative keeps it, logs WARNING with count and minimum; a NaN/inf after scaling raises `NearEdgeDataError`), `test_input_type_required_with_nearedge`, `test_beta_round_trip_and_density_cancels_at_merge`, `test_nearedge_result_matches_upstream_golden` (polystyrene; tolerance = measured error, recorded in the test), `test_merge_points_outside_data_raise`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Port `convert_data`/`convert_NEXAFS_to_ASF`/`convert_Beta_to_ASF`/`merge_spectra` from upstream `data.py`, keeping the algorithm (affine endpoint matching), using repo `constants`. Changes from upstream: the validations above, a conditioning guard on the endpoint difference (raise `NearEdgeDataError`, never emit inf), eV/keV handled at load, no `add_background`/`fix_distortions`. Merged `seg_ev`/`coeffs` feed `kk_real_spectrum` unchanged.
- [ ] **Step 4:** Re-run → PASS; mypy strict.
- [ ] **Step 5:** Commit `feat(kk): near-edge loading, conversion and merging`.

### Task 6: Adaptive refinement

**Files:**
- Create: `xraylabtool/kk/refine.py`
- Modify: `xraylabtool/kk/api.py::kk_calculate_real` (and add new public names to `_EXPORTS`/`__all__` in `kk/__init__.py`) (adds `refine_tolerance: float | None = None`, `refine_max_passes: int = 50`, `refine_max_points: int = 100_000`)
- Test: `tests/unit/kk/test_refine.py`, golden check in `tests/characterization/kk/`

**Interfaces:**
- Produces: `@dataclass(frozen=True, eq=False) class RefineOutcome(energy_ev: NDArray, f1: NDArray, f2: NDArray, n_added: int, exhausted: bool, n_pole_skipped: int = 0)` (`n_pole_skipped` counts intervals whose midpoint fell on a pole and could not be refined; `KKResult` does not expose it); `refine_spectrum(energy_ev, f1, f2, seg_ev, coeffs, rel_corr, *, tolerance: float, max_passes: int = 50, max_points: int = 100_000, chunk_size: int | None = None) -> RefineOutcome` (`chunk_size` is forwarded to every `kk_real_spectrum` call, and `kk_calculate_real` passes its own) (host-side loop; calls `kk_real_spectrum` on batches of midpoints; tolerance is **absolute** in f1/f2 units, midpoints only). Precondition (D7): `energy_ev` is strictly increasing and unique; `kk_calculate_real` sorts and de-duplicates the requested energies before calling it, so the refined result is the sorted unique union of requested energies and added midpoints.

- [ ] **Step 1: Write failing tests** — `test_matches_upstream_improve_accuracy_golden` (uses the `refine_*.npz` files from Task 2), `test_refine_with_shuffled_duplicate_energies_returns_sorted_unique_union` (through `kk_calculate_real`, also a length-one grid), `test_refine_rejects_non_increasing_input`, `test_original_points_preserved`, `test_loose_tolerance_adds_nothing`, `test_exhaustion_flagged_and_logged` (`max_passes=2`, tiny tolerance → `exhausted is True`, WARNING in caplog), `test_point_cap_stops_and_flags`, `test_midpoint_stagnation_terminates`, `test_n_refined_points_reported_in_result`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Port `improve_accuracy`; log points added per pass at DEBUG and the total at INFO. Never drop input points.
- [ ] **Step 4:** Re-run → PASS; mypy strict.
- [ ] **Step 5:** Commit `feat(kk): adaptive accuracy refinement`.

### Task 7: Adapter to `XRayResult`

**Files:**
- Modify: `xraylabtool/calculators/core.py` (extract the shared tail of `calculate_single_material_properties` into the new helper), `xraylabtool/kk/result.py`
- Test: `tests/unit/test_properties_from_factors.py`, `tests/unit/kk/test_result_adapter.py`

**Interfaces:**
- Produces: `calculate_properties_from_factors(formula: str, energy_kev: EnergyArray, density: float, f1_total: OpticalConstantArray, f2_total: OpticalConstantArray) -> XRayResult` (δ/β/derived quantities from supplied per-formula f1/f2; no `.nff` lookup; negative f1 preserved; validates before computing: `energy_kev`, `f1_total`, `f2_total` are equal-length 1-D, all finite, `density` finite and positive, else `ValidationError`; negative f2 is kept per D6 since the existing attenuation kernel already clamps absorption at `1e-30`); `KKResult.to_xray_result(density: float) -> XRayResult` (raises `EnergyError` outside 0.03–30 keV).

- [ ] **Step 1: Write failing tests** — `test_helper_reproduces_lookup_path` (feed the CXRO f1/f2 totals for `SiO2` from the lookup; equals `calculate_single_material_properties` on every `XRayResult` field, `rtol=1e-12`; guards the refactor), `test_negative_f1_preserved_critical_angle_zero` (Review Focus 4), `test_to_xray_result_rejects_out_of_range_energy`, `test_to_xray_result_uses_kk_f1`, `test_helper_rejects_mismatched_nonfinite_or_bad_density` (length mismatch, a broadcastable wrong shape, NaN/inf, density ≤ 0).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Refactor so both paths share one tail; implement the adapter. **Step 4:** Re-run those tests plus `uv run pytest tests/unit tests/characterization -q --no-cov` → PASS (the refactor must not change any golden output).
- [ ] **Step 5:** Commit `feat(calculators): build XRayResult from supplied f1/f2`.

### Task 8: CLI subcommand

**Files:**
- Create: `xraylabtool/interfaces/cli/commands/kk.py`
- Modify: `interfaces/cli/parser.py` (`add_kk_command`, call next to `add_convert_command` at line 93), `interfaces/cli/__init__.py:166` (`"kk": cmd_kk`), `interfaces/completion_v2/shells.py:711` (static table), `docs/guides/cli_reference.rst`, `tests/integration/test_cli.py:511` (add `"kk"` to `expected_commands` in `test_all_expected_commands_available`)
- Test: `tests/integration/test_cli_kk.py`; existing `tests/integration/test_cli.py` completion-parity tests must pass

**Interfaces:**
- Consumes: `kk_calculate_real`, `load_nearedge`, `parse_energy_string`.
- Produces: `cmd_kk(args: Any) -> int`; flags `FORMULA [--nearedge FILE] [--input-type nexafs|beta|asf] [--energy-unit eV|keV] [--merge LOW,HIGH] [-d/--density G_CM3] [--refine TOL] [--energies SPEC] [-o OUT] [--format csv|json]`. Output columns `energy_keV,f1,f2`. The CLI offers a subset of the API's five `input_type` values (`photoabsorption` and `xanes` are API-only aliases of `nexafs`). `--density` is validated as finite and positive, is meaningful for `beta` input only, and cancels in the near-edge normalisation, so it never changes the result. `--input-type` is required with `--nearedge`: enforced at the start of `cmd_kk` (a conditional requirement is not expressible with `required=True`), printed to stderr, return **1** per the CLI error policy. `--merge` is in `--energy-unit`; `--energies` is always keV.

- [ ] **Step 1: Write failing tests** — `test_kk_tabulated_only_matches_api`, `test_nearedge_requires_input_type` (`main()` returns 1, stderr names `--input-type`), `test_energies_are_kev_regardless_of_energy_unit`, `test_merge_units_follow_energy_unit`, `test_kev_file_declared_ev_errors_cleanly` (exit 1, message on stderr, Review Focus 3), `test_json_format`, `test_output_file`, `test_energies_nan_rejected`, `test_refine_reports_added_points`, `test_xlt_alias_runs_kk`, `test_completion_lists_kk`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement following `commands/calc.py` conventions (error print to stderr, return int). **Step 4:** Run the new tests and `uv run pytest tests/integration -q --no-cov` → PASS.
- [ ] **Step 5:** Commit `feat(cli): add kk subcommand`.

### Task 9: GUI service and plot widget

**Files:**
- Create: `xraylabtool/gui/widgets/kk_plot.py`
- Modify: `xraylabtool/gui/services.py` (`KKComputation`, `compute_kk`)
- Test: `tests/unit/test_kk_service.py`, `tests/unit/test_kk_plot.py` (serial GUI file, offscreen, like `tests/unit/test_gui_widgets.py`)

**Interfaces:**
- Consumes: `kk_calculate_real`, `load_nearedge`, `KKResult.nearedge_scaled_kev_f2`, the existing CXRO lookup (`calculators`), palette helpers (`apply_palette_to_item`).
- Produces: `@dataclass(frozen=True) class KKComputation(result: KKResult, cxro_energy_kev: NDArray, cxro_f1: NDArray)` — the CXRO f1 totals for the formula on the energies of `result` that lie inside 0.03–30 keV (empty arrays if none), computed in the service so no numerical work runs on the UI thread; `compute_kk(formula: str, nearedge_path: str | None, input_type: str | None, energy_unit: str, merge_points: tuple[float, float] | None, refine_tolerance: float | None, energies_kev: ArrayLike | None = None) -> KKComputation` (no Qt; merge points in `energy_unit`; `energies_kev` carries the tab's range controls); `KKPlot.set_data(computation: KKComputation) -> None` draws four curves, each on its own grid: measured f2 (`result.nearedge_scaled_kev_f2`), spliced f2 (`result.energy_kev`, `result.f2`), KK f1 (`result.energy_kev`, `result.f1`) and CXRO f1 (`cxro_energy_kev`, `cxro_f1`); curves with no data are hidden. `F1F2Plot` draws only two curves on one grid, hence the new widget.

- [ ] **Step 1: Write failing tests** — `test_compute_kk_service_matches_api_without_nearedge` and `test_compute_kk_service_matches_api_with_nearedge`, `test_compute_kk_forwards_energies_kev`, `test_cxro_comparison_empty_outside_tabulated_range`, `test_plot_draws_curves_on_unequal_grids` (measured, spliced and CXRO arrays of different lengths; no resampling error), `test_plot_hides_missing_curves` (no file → measured curve hidden), `test_plot_follows_theme_toggle`.
- [ ] **Step 2:** Run `uv run pytest tests/unit/test_kk_service.py tests/unit/test_kk_plot.py -v --no-cov -p no:xdist` → FAIL. **Step 3:** Implement `KKComputation`, `compute_kk` and `KKPlot`, mirroring `F1F2Plot` for palette handling. **Step 4:** Re-run → PASS; `uv run mypy xraylabtool --strict`.
- [ ] **Step 5:** Register `tests/unit/test_kk_plot.py` in the serial GUI lists (see Global Constraints). Commit `feat(gui): add KK compute service and plot widget`.

### Task 10: GUI tab

**Files:**
- Create: `xraylabtool/gui/main_window/kk_tab.py`
- Modify: `gui/main_window/__init__.py` (class bases at line 28 gain `KKTabMixin`; `addTab(self._build_kk_tab(), "Kramers-Kronig")` after line 106), `gui/main_window/_protocol.py` (declare the tab's members for `mypy --strict`), `gui/main_window/shared.py:88` (`_set_tab_order` for the new tab), `scripts/gui_headless_smoke.py`, `Makefile:60` (`GUI_TESTS`), `.github/workflows/ci.yml:325` (`IGNORE_GUI`) and `:410` (serial GUI call) for `tests/test_gui_kk_tab.py` (`tests/unit/test_kk_plot.py` was registered in Task 9)
- Test: `tests/test_gui_kk_tab.py` (serial, offscreen, like `test_gui_smoke.py`)

**Interfaces:**
- Consumes: Task 9 `compute_kk`, `KKComputation`, `KKPlot`; `CalculationWorker`.
- Produces: `KKTabMixin._build_kk_tab(self) -> QWidget`, `_run_kk(self) -> None`, `_export_kk_csv(self) -> str | None`. The tab has its own small form (formula, density for the optional XRayResult export, near-edge file, input type, unit, merge points, refine tolerance, start/end energy in keV, N points) — `MaterialInputForm` hardcodes 0.03–30 keV (`material_form.py:34-46`) so it cannot be reused. Cancellation of a running refine is not provided (`CalculationWorker` has none); `refine_max_points` bounds the work. Skipped deliberately.

- [ ] **Step 1: Write failing tests** — `test_tab_present_and_named` (`main_tabs.tabText(2) == "Kramers-Kronig"`), `test_run_without_file_populates_plot_and_table`, `test_energy_range_controls_reach_the_service`, `test_worker_error_shows_toast_not_crash` (invalid formula), `test_tab_order_includes_kk_widgets`, `test_export_csv_columns`, `test_theme_toggle_updates_kk_plot`.
- [ ] **Step 2:** Run (serially) `uv run pytest tests/test_gui_kk_tab.py -v --no-cov -p no:xdist` → FAIL. **Step 3:** Implement mirroring `single_tab.py` (worker off the UI thread, result handler, CSV export helper). **Step 4:** Re-run → PASS; `uv run python scripts/gui_headless_smoke.py`; `uv run mypy xraylabtool --strict`; `make test-fast` confirms the new GUI files run in the serial pass, not under xdist.
- [ ] **Step 5:** Commit `feat(gui): add Kramers-Kronig tab`.

### Task 11: Sphinx documentation grounded in Watts 2014

**Files:**
- Create: `docs/physics/kramers_kronig.rst`, `docs/api/kk.rst`, `docs/guides/kk_guide.rst`
- Modify: `docs/index.rst` (add `physics/kramers_kronig` after `physics/calculations` at line 126 and `guides/kk_guide` after line 117), `docs/api/index.rst` (add `kk` to "Extended API" and the sub-package list), `docs/guides/cli_reference.rst` (if not done in Task 8), `CLAUDE.md` (source layout + KK note; git-ignored, so the edit stays local), the docstrings written in Tasks 3-6 (finish any missing `References`/`Examples`)
- Test: `tests/unit/kk/test_docs.py`, `tests/integration/test_kk_docs_build.py` (marked slow)

**Interfaces:**
- Consumes: every public name of `xraylabtool.kk` (`kk_calculate_real`, `KKResult`, `NearEdgeSpectrum`, `load_nearedge`, `to_asf_scale`, `merge_spectra`, `compound_spectrum`, `evaluate_f2`, `relativistic_correction`, `kk_piecewise_poly`, `kk_real_spectrum`, `refine_spectrum`, `RefineOutcome`, `load_asf`, `AsfDatabase`) and `calculate_properties_from_factors`.
- Produces: the three pages above, and a module-level `__all__` in `xraylabtool/kk/__init__.py` that the completeness test reads (add it if Task 4 did not).

**Content (what each page must contain — spec section "Documentation (Sphinx), grounded in Watts 2014").** Own wording and own figures only; equations as `.. math::`; cite the paper as `Watts 2014, Eq. (n)`; one `References` section at the bottom of each page.
- `kramers_kronig.rst`, in this order:
  1. Henke convention `n = 1 − δ − iβ` with `δ, β ∝ Σ N_q f1_q, f2_q`, and the KK relation `f1(E) = Z* + (2/π) P∫ x f2(x)/(E²−x²) dx` (paper Eq. 1, sign written as used here).
  2. Piecewise Laurent representation (Eq. 8) with the five coefficient columns `A_1, A_0, A_-1, A_-2, A_-3` and coefficients in **eV**.
  3. Closed form per interval = Eq. (10) with the corrected a₋₃ sign, plus Table 1 as an RST list-table, and a boxed **erratum** note: the typeset Eq. (10) ends in `+(2a₋₃/E)(xⱼ₊₁⁻¹−xⱼ⁻¹)`, with the wrong sign (subscripts are correct); the correct term is `+(2a₋₃/E)(xⱼ⁻¹−xⱼ₊₁⁻¹)`, matching Table 1, upstream code and quadrature (the printed sign is rejected by `test_printed_eq10_a3_sign_is_wrong`).
  4. Singularities: why the paper's two-interval treatment (Eqs. 5–6) is valid only where f2 is continuous and not at the first/last edge; our conventions (D5) stated as conventions.
  5. Integration limits: the 0→∞ conditions (`aₙ=0` for `n<0` in the first interval, `n≥−1` in the last), why the Biggs-Lighthill tail beyond 500 keV cannot be used, hence 10 eV–500 keV with f2 = 0 outside.
  6. Relativistic correction (Eq. 9) and the paper's six-step workflow as a table mapping each step to the function that implements it.
  7. Accuracy: what the paper shows (Ohta-Ishida model, rms 2.4e-7 with 4th-to-0th-order polynomials, which the fixed five-term kernel does not reproduce) and what this implementation shows (the Task 3-5 checks, measured numbers filled in from the actual test results).
  8. The truncation example (C, Fe, Pb at 30 keV vs 500 keV) with a matplotlib figure generated by a small script under `docs/_static/` or an inline `.. plot::`-free static PNG committed with its generating script; values from `test_30kev_truncation_matches_paper_trend`.
  9. The nine-element note (Sc, Mo, Ru, Be, Ti, Si, La, Gd, W) and D2.
- `api/kk.rst`: `automodule` blocks (`:members:`, `:undoc-members:`, `:show-inheritance:`, like `docs/api/backend.rst`) for `xraylabtool.kk`, `.transform`, `.spectrum`, `.nearedge`, `.refine`, `.result`, `.database`, plus `calculate_properties_from_factors`; a short "which function implements which paper equation" table at the top.
- `kk_guide.rst`: API example (tabulated only; with a near-edge file and merge points), CLI examples for every flag, GUI tab walkthrough (text, no screenshots required), input-file format and units (`--energy-unit`, `--energies` always keV), D1–D7 behaviour in plain language, limits, Zlib attribution.

- [ ] **Step 1: Write failing tests** — `test_every_public_kk_name_is_documented` (each name in `xraylabtool.kk.__all__` appears in `docs/api/kk.rst` via a directive or the equation table), `test_transform_docstrings_cite_paper_equations` (`relativistic_correction` mentions "Eq. (9)", `kk_piecewise_poly` "Eq. (10)", `evaluate_f2` "Eq. (8)"; `refine_spectrum` says it is not in the paper), `test_public_docstrings_have_examples_and_references` (every public function docstring has `Examples` and `References` sections), `test_docs_pages_toctree_linked` (the three pages are referenced from `docs/index.rst`/`docs/api/index.rst`), `test_docs_state_the_eq10_erratum` (the physics page contains "a₋₃" or `a_{-3}` and "erratum"), `test_kk_docs_build_adds_no_warnings` (slow: `sphinx-build -b html docs <tmp>`; no warning line mentions `kk`, `kramers_kronig` or `kk_guide`; pre-existing warnings elsewhere are not this test's concern).
- [ ] **Step 2:** Run `uv run pytest tests/unit/kk/test_docs.py -v --no-cov` → FAIL.
- [ ] **Step 3:** Write the three pages and finish the docstrings. Reread the paper's Eqs. 1, 8, 9, 10 and Table 1 against the printed pages (not extracted text) when transcribing; fill measured numbers from the real test output.
- [ ] **Step 4:** Re-run → PASS; `uv run sphinx-build -b html docs docs/_build/html` builds; `make docs-doctest` passes for the new examples.
- [ ] **Step 5:** Commit `docs(kk): physics note, API reference and guide`.

### Task 12: Packaging verification and full gate

**Files:**
- Test: `tests/integration/test_kk_packaging.py` (marked slow)

- [ ] **Step 1: Write failing test** — `test_built_wheel_and_sdist_contain_kk_data_and_licence` (`uv build` into a tmp dir; assert both archives contain `xraylabtool/data/kkcalc/asf.npz` and `LICENSE-kkcalc.txt`, and the root `NOTICE`; wheel imports `xraylabtool.kk` in a clean venv and runs `kk_calculate_real("Si")`).
- [ ] **Step 2:** Run → FAIL or PASS depending on Task 2; fix packaging until PASS.
- [ ] **Step 3:** Run `make claude` and `make docs-doctest` → green; `graphify update .`. Confirm the citation text in the docs reads B. Watts, *Opt. Express* **22**(19), 23628–23639 (2014), doi:10.1364/OE.22.023628.
- [ ] **Step 4:** Commit `test(kk): packaging verification`; then use superpowers:finishing-a-development-branch.
