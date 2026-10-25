# kkcalc integration — design

Date: 2026-10-24 · Status: reviewed (round 1), decisions D1–D4 made, awaiting Wei's final read · Owner: Wei Chen

## Intent

Adopt [benajamin/kkcalc](https://github.com/benajamin/kkcalc) (Watts &
Lauk, Zlib licence) into XRayLabTool. kkcalc turns a measured near-edge
absorption spectrum (NEXAFS/XANES) into the **real part of the atomic
scattering factor, f1**, via a Kramers-Kronig (KK) transform, splicing the
measurement into tabulated scattering factors so the integral has the
full-range input it needs.

XRayLabTool today only *looks up* tabulated f1/f2 (CXRO). This adds the
missing direction: *measured f2 → f1*. The result feeds the existing
δ/β/critical-angle/attenuation machinery, so users can compute optical
constants of a real material near an absorption edge from their own data.

Why a port and not a dependency: upstream's GUI is wxPython (conflicts with
the PySide6 stack), its transform is NumPy-only (no JAX), it ships its own
copy of the CXRO data, and it is unmaintained-style research code. We take
the algorithm, not the packaging.

## Goals

1. `xraylabtool.kk` — a JAX-first core: given a formula, optionally a
   near-edge spectrum, return f1 and f2 on an energy grid.
2. `xraylabtool kk` (alias `xlt kk`) — scriptable CLI over the core.
3. A **"Kramers-Kronig" tab** in the existing PySide6 main window.
4. Numerical agreement with upstream kkcalc on its own test cases (see
   Verification).
5. Zlib compliance: upstream copyright/notice preserved, modified files
   marked as altered, Watts 2014 cited in docs.

## Non-goals (this spec)

- kkcalc's **"Add background"** option — upstream marks it disabled/not
  reimplemented. Not ported.
- kkcalc's **"Fix distortions"** option (SciPy-based skewing of near-edge
  data). Deferred. *(Builder-visible call: see Calls Made.)*
- Fitting/differentiating through the transform (free with JAX, but no
  API or UI for it now).
- Replacing the existing CXRO lookup path or its data files.
- A standalone wx/Qt app or a popup window.

## Anti-goals

- Wrong numbers that look plausible. Correctness over speed (project
  principle 1). A fast transform that drifts from upstream is a failure.
- A second GUI toolkit, a second copy of CXRO data, or a hard `scipy`
  requirement for the hot path.
- Silent behaviour: no silent resampling, trimming, or extrapolation of the
  user's data (project principle 2 and §3 "no silent subsampling").

## Constraints (from the project)

- Python ≥ 3.12, `uv`, JAX-first, `mypy --strict` clean, ruff clean,
  NumPy-style docstrings, explicit imports, `filterwarnings = error`.
- Backend discipline: array math goes through `xraylabtool.backend.ops`;
  NumPy fallback stays correct; JAX is auto-selected only on NVIDIA GPU
  (existing rule — unchanged). Kernels are pure and JIT-lazy like
  `calculators/kernels.py`.
- Public API energy in **keV** (project rule). File I/O boundary may be eV
  (see Calls Made).
- `XRayResult`-style result: snake_case fields, per-energy arrays.
- GUI is a view layer; numerics stay in the core; heavy work off the UI
  thread via the existing `CalculationWorker`.
- Lazy imports: `import xraylabtool` must not pull `kk` until used.

## Key technical finding: the data problem

kkcalc's transform is **not** run on the `.nff` point tables. It runs on a
*piecewise Laurent-polynomial* representation of f2 per energy segment —
coefficients of `E¹, E⁰, E⁻¹, E⁻², E⁻³` — stored in `ASF.json`, built by
upstream's `PackRawData.py` from CXRO/Henke **plus Biggs & Lighthill
high-energy data**. Verified (review round 1): the vendored grids span
**10 eV–500 keV** for all 92 elements (an earlier draft said ~100 MeV;
that was wrong). The transform integrates only to the last segment boundary
and adds the relativistic-correction baseline; there is no further tail
term. Our `.nff` files stop at 30 keV, so they cannot supply the
30–500 keV contribution.

Decision: **vendor upstream's `ASF.json`** as package data
(`xraylabtool/data/kkcalc/ASF.json`) with its licence text, rather than
regenerating it from `.nff`. Regenerating would need Biggs-Lighthill source
data we do not have and would invalidate comparison with upstream. The
existing `.nff` path is untouched and remains the source for the ordinary
`calc` commands. *(Open for the builder: JSON size on disk and load time;
if large, convert once to `.npz` at build time — still no runtime cost
beyond a lazy load.)*

## Architecture

```
xraylabtool/kk/
├── __init__.py       # lazy exports: kk_calculate_real, KKResult, ...
├── database.py       # lazy-load ASF.json -> backend arrays; stoichiometry sum
├── transform.py      # PURE kernels: piecewise-poly KK, relativistic correction
├── nearedge.py       # load / unit-convert / merge near-edge data (host-side)
├── refine.py         # improve_accuracy (adaptive midpoint insertion, host loop)
└── result.py         # KKResult dataclass (snake_case)
xraylabtool/data/kkcalc/{ASF.json, LICENSE-kkcalc.txt}
xraylabtool/interfaces/cli/commands/kk.py
xraylabtool/gui/main_window/kk_tab.py     # KKTabMixin
tests/{unit,integration,characterization}/…  # see Verification
docs/{guides,api,physics}/…                  # guide, API page, Watts 2014 note
NOTICE (or docs/third_party.rst)             # Zlib attribution
```

Formula parsing reuses `utils.parse_formula()`; upstream's own parser is not
ported. Constants come from `constants.py`, not upstream's inline values
(check they agree to float64; if not, that is a finding to report).

### Core kernel (`transform.py`)

Pure function of arrays → arrays, no I/O, no data-dependent Python
branching, so it is `jax.jit`-safe:

`kk_piecewise_poly(eval_energy, seg_energy, coeffs, rel_corr) -> f1`
(ports upstream `KK_PP`: fixed orders `E¹…E⁻³`, 5 coefficient columns —
decision D1; there is no `orders` argument)

- Upstream builds `(n_seg × n_eval × n_orders)` tiled tensors and sums
  `ln|…|` terms; poles (eval energy equal to a segment edge) are handled with
  boolean masks. Port as **broadcasting, not `tile`**, and keep the pole
  masks. Use the double-`where` pattern around `log` so neither value nor
  gradient is NaN at poles.
- **Float64 is mandatory** (cancellation in the log terms); the kernel must
  assert/enable x64 under JAX rather than silently run in float32.
- **Memory:** the full tensor is `n_seg·n_eval·n_orders` floats. Evaluate in
  eval-energy chunks (`jax.lax.map`/scan) with a documented default chunk;
  chunking must be mathematically identical to a single pass (project §3
  "streaming only if equivalent") and tested for it.
- `calc_relativistic_correction`: `Σ nᵢ (Zᵢ − (Zᵢ/82.5)^2.37)` — scalar,
  trivial.
- Backend extension: `ops` currently lacks `log`, `abs`, `cumsum`-class
  helpers the kernel needs; add them to `ArrayBackend`, `NumpyBackend`,
  `JaxBackend` (small, additive).

### Host-side pieces (not JIT, by design)

- `nearedge.py`: load columnar file (first column energy, last column
  absorption), convert type (`photoabsorption`/`nexafs`/`xanes`, `beta`,
  `asf`) to scattering-factor scale, merge into the tabulated spectrum at
  `merge_points` (normalise so the spectra are continuous). Validation at
  this boundary per §3: shape, dtype, NaN/inf, **strictly increasing
  energy**, non-empty overlap with the tabulated range — each a raised,
  specific `XRayLabToolError` subclass, never a silent fix.
- `refine.py`: `improve_accuracy` inserts midpoints recursively where linear
  interpolation of f1/f2 exceeds `tolerance` (≤ `recursion` passes). Dynamic
  shapes ⇒ stays a host loop that calls the jitted kernel on batches of
  midpoints. Every inserted point count is logged (this *adds* points; it
  never drops any).

### Result

`KKResult` (frozen dataclass): `energy_keV`, `f1`, `f2`, `formula`,
`relativistic_correction`, `merge_points`, `n_refined_points`,
`backend`. Convenience `.to_xray_result(density)` to reuse existing
δ/β/derived-quantity code (builder to confirm the cleanest hook into
`calculators/kernels.py`).

### CLI (`xraylabtool kk`)

One file in `commands/`, registered in `parser.py`, covered by the shell
completion tests like the other subcommands. Interface:

```
xraylabtool kk FORMULA [--nearedge FILE] [--input-type nexafs|beta|asf]
               [--energy-unit eV|keV] [--merge LOW,HIGH] [--density G_CM3]
               [--refine TOL] [--energies SPEC] [-o OUT.csv] [--format csv|json]
```

`--energies` reuses `parse_energy_string`. Without `--nearedge` it simply
returns the KK-consistent f1 of the tabulated spectrum (also the sanity
check against CXRO f1). Exit codes and error printing follow the other
commands.

### GUI tab (`KKTabMixin`)

Third tab beside "Single Material" and "Multiple Materials", built the same
way (mixin + `_build_kk_tab()` in `MainWindow.__init__`):

- Inputs: formula, load-file button, input-type and energy-unit selectors,
  start/end range, merge points, refine tolerance. A new small form:
  `MaterialInputForm` hardcodes 0.03–30 keV ranges and its own compute
  button (`gui/widgets/material_form.py:34-46`), so it is not reusable.
- Plot: a new `KKPlot` in the PyQtGraph widget family (palette helpers
  reused) showing measured f2, spliced f2, resulting f1 and CXRO f1.
  `F1F2Plot` takes `XRayResult` field names and draws two curves
  (`gui/widgets/sweep_plots.py:45`), so it cannot be reused as-is.
- Compute runs in `CalculationWorker` (off the UI thread); progress/errors
  through the existing toast/log path. Save: CSV via the same export helper
  the other tabs use.
- Theming follows `ThemeManager`; no new styling system.

## Verification (what "correct" means)

1. **Upstream oracle.** Pin an upstream commit; a one-off script (dev-only,
   not a dependency, not shipped) runs real kkcalc on a fixed set of cases
   and writes golden CSVs into `tests/characterization/`. Cases: a light
   element, a heavy element, a compound, and a compound **with** a bundled
   near-edge example from upstream's `examples/`. Tolerance: NumPy backend
   `rtol≈1e-10`, JAX backend `rtol≈1e-8` (builder to tighten from actual
   results; report the achieved error).
2. **Analytic check.** KK of a single Lorentz oscillator f2 has a closed-form
   f1; the kernel must match it. Catches sign/pole errors that a
   self-comparison with upstream would inherit. In the x-ray convention
   (Watts Eq. 1) the pair is `f2 = Aγω/D`, `f1 − Z* = A(ω²−ω0²)/D` with
   `D = (ω0²−ω²)²+γ²ω²` — the opposite sign to the optical `Re χ`. Also check
   each of the five basis terms (`E¹…E⁻³`) separately against Cauchy
   quadrature, and the kernel against an independent NumPy transcription of
   the paper's Eq. (10) **with its a₋₃ term's sign corrected**: the
   typeset Eq. (10) ends in `+(2a₋₃/E)(x⁻¹ⱼ₊₁−x⁻¹ⱼ)` (verified on a 300 dpi
   render of the published page); the sign is wrong and the subscripts are
   correct. The correct term is `−(2a₋₃/E)(x⁻¹ⱼ₊₁−x⁻¹ⱼ)` = `+(2a₋₃/E)(x⁻¹ⱼ−x⁻¹ⱼ₊₁)`,
   which agrees with the paper's Table 1, upstream's code and quadrature.
   Evaluating the printed term as typeset errs by up to 1e8 on Au.
3. **Sum-rule/consistency.** KK of the tabulated spectrum alone ≈ tabulated
   CXRO f1 within a documented tolerance.
4. **Backend parity** (NumPy vs JAX, skipped without JAX) and **chunk
   invariance** (any chunk size ⇒ same result).
5. Validation errors tested (non-monotonic energy, NaN, no overlap).
6. CLI integration test; GUI smoke test through the existing offscreen
   harness (`scripts/gui_headless_smoke.py`, serial GUI tests).
7. `make claude` green: ruff, `mypy --strict`, coverage floor; docs build.

## Documentation (Sphinx), grounded in Watts 2014

Source: B. Watts, "Calculation of the Kramers-Kronig transform of X-ray
spectra by a piecewise Laurent polynomial method", *Opt. Express* **22**(19),
23628–23639 (2014), doi:10.1364/OE.22.023628 (PDF read in full during plan
review; published 19 Sep 2014). Three pages plus docstrings; the docs are the
user-facing record of what each function implements and where it departs from
the paper.

- **`docs/physics/kramers_kronig.rst`** — in our own words and with our own
  figures (the paper is © Optica; no copied text or figures):
  - Henke convention `n = 1 − δ − iβ`, `f1`/`f2` per element; the KK
    relation `f1(E) = Z* + (2/π) P∫ x f2(x)/(E²−x²) dx` (Eq. 1, written with
    the sign convention used in the code and in the constant-band and Lorentz
    checks).
  - Piecewise Laurent representation `f2 = a₁x + a₀ + a₋₁x⁻¹ + a₋₂x⁻² +
    a₋₃x⁻³` per interval (Eq. 8), and the closed form per interval (Eq. 10,
    **with the corrected a₋₃ sign**, plus Table 1 as an RST table). State the
    printed-equation discrepancy and how it was established.
  - Singularity handling: the paper's double-interval form (Eqs. 5–6) is
    valid at an interior knot because F is continuous there, and not at the
    first/last edge. Our conventions at a jump in f2 and at the end points
    are upstream's finite values (decision D5), labelled as conventions.
  - Integration limits: the paper shows 0→∞ integration needs `aₙ=0` for
    `n<0` in the first interval and `n≥−1` in the last; the Biggs-Lighthill
    tail beyond 500 keV violates the latter, so the data (and our core) stop at
    10 eV–500 keV and f2 is taken as zero outside.
  - Relativistic correction (Eq. 9); the six-step workflow of the paper
    mapped to our functions (database lookup + stoichiometry sum →
    `compound_spectrum`; near-edge convert + merge → `to_asf_scale`,
    `merge_spectra`; Eq. 9 → `relativistic_correction`; Eq. 10 →
    `kk_piecewise_poly`; Eq. 8 → `evaluate_f2`).
  - Accuracy: what the paper demonstrates (Ohta-Ishida model, rms 2.4e-7
    over 1000–3000 cm⁻¹ using 4th-to-0th-order polynomials, which our fixed
    five-term kernel cannot reproduce — say so) versus what *this*
    implementation demonstrates (upstream parity, independent quadrature,
    Lorentz closed form, CXRO f1 agreement, the nine-element data note).
  - Truncation example matching the paper's Figs. 3–5 qualitatively:
    integrating only to 30 keV changes `f1` by about 1e-4 for C, 0.1 for Fe
    and 2.5 for Pb at 0.1–10 keV (values measured with upstream `KK_PP` during
    plan review; ours are pinned by a test).
- **`docs/api/kk.rst`** — `automodule` for `xraylabtool.kk` and each
  submodule (the existing `docs/api/*.rst` style), every public function and
  class documented. Each public docstring is NumPy-style with units (eV
  internal, keV public), shapes, the exceptions it raises, an `Examples`
  block that runs under `make docs-doctest`, and a `References` section that
  cites the paper equation it implements (e.g. `kk_piecewise_poly` → Eqs. 7
  and 10; `relativistic_correction` → Eq. 9; `evaluate_f2` → Eq. 8) or says
  it is not in the paper (`refine_spectrum` → upstream `improve_accuracy`).
- **`docs/guides/kk_guide.rst`** — Python API, CLI (`xraylabtool kk`) and GUI
  tab usage, input-file format, D1–D7 behaviour, limits (500 keV, no tail, no
  add-background / fix-distortions), and the Zlib attribution.
- Completeness is tested, not just written: every name exported by
  `xraylabtool.kk` appears in `docs/api/kk.rst`; transform docstrings contain
  the equation citations; the doctests run; the Sphinx build adds no warnings
  from the new pages.

## Delivery order

Core + kernel tests → CLI → GUI tab → docs. Each stage is shippable alone;
nothing in stage N+1 may require reworking stage N's public API.

## Calls made by me (not discussed with Wei — please check)

- Vendor `ASF.json` rather than regenerate from `.nff` — confirmed by Wei
  as decision D2 after review found it disagrees with the repo's `.nff` for
  9 elements.
- Defer "fix distortions"; drop "add background" (upstream-disabled).
- Near-edge **files** default to `--energy-unit eV` (the convention of
  beamline NEXAFS files); the public Python API and all outputs stay keV.
- Result type is a new `KKResult`, with an adapter to `XRayResult`, not an
  extension of `XRayResult`.
- Chunked evaluation is on by default; JAX stays opt-in via the existing
  auto-selection (no new backend flag).
- Cite: B. Watts, *Opt. Express* **22**(19), 23628–23639 (2014),
  doi:10.1364/OE.22.023628 — verified against Crossref and the PDF.
- Round 2 plan review added D5 (pole/jump/end-point convention = upstream's
  finite values), D6 (negative f2 after normalisation kept with a WARNING)
  and D7 (refinement on a sorted unique grid); see the plan's Global
  Constraints.

## Left to the builder

Chunk size default; exact `ops` additions; JSON-vs-npz packaging; module
file splits inside `kk/`; naming of CLI flags beyond the sketch; GUI widget
layout details; whether the adapter lives on `KKResult` or in `calculators/`.

## Review round 1 (2026-10-24: builder check + Codex + Agy)

Sources: a cold builder check, a Codex numerics review (ran probes against
upstream and this repo) and an Agy architecture review. Facts below marked
✔ were re-verified by me against the repo or upstream.

### Decisions (Wei, 2026-10-24) — all four took the recommendation

**D1 → port `KK_PP` + independent reference. D2 → vendor ASF.json as-is,
document the 9 differing elements with a test. D3 → core accepts the full
10 eV–500 keV; only the `XRayResult` adapter enforces 0.03–30 keV.
D4 → `--merge` follows `--energy-unit`.** Rationale as listed:

- **D1. Reference kernel.** The golden-file path (`kk_calculate_real`) calls
  upstream `KK_PP` (fixed 5 orders, own boundary handling), not
  `KK_General_PP` (the `orders` form this spec described). On identical
  ASF.json data they differ by up to 2.8e-3 (Au), 1.5e-5 (Cu), 2e-7 (Si);
  `KK_General_PP` also returns a NaN per element and needs patches for
  current NumPy. Recommendation: port `KK_PP` with static orders (also what
  JIT needs), and add an independent high-precision reference test so
  parity with upstream is not the only definition of "correct".
- **D2. Data source.** Upstream's f2 differs from this repo's `.nff` for 9
  elements (e.g. Sc median 12%, 90th percentile 91%; Mo, Ru, Be, Ti, Si,
  La, Gd, W smaller). Vendoring as-is means `kk` and `calc` disagree for
  those elements and "KK of tabulated ≈ CXRO f1" fails there.
  Options: (a) vendor as-is, parity with upstream, documented
  discrepancy; (b) hybrid — `.nff` for 30 eV–30 keV plus upstream's
  high-energy segments (consistent with `calc`, loses exact parity for
  those 9). Recommendation: (a) for v1 with a documented, tested list of
  the 9 elements.
- **D3. Energy range.** ASF.json spans 10 eV–500 keV; the package's rule and
  `calculators/core.py` enforce 0.03–30 keV. Decide the default output grid
  of `xlt kk`, whether near-edge files and `--energies` may leave
  0.03–30 keV (VUV <30 eV, K-edges 25–60 keV), and whether
  `to_xray_result` rejects out-of-range energies.
  Recommendation: the KK core accepts the full 10 eV–500 keV, output
  default is the data's own grid clipped to 0.03–30 keV and logged; the
  adapter enforces 0.03–30 keV.
- **D4. `--merge` units.** Follow `--energy-unit` (same as the file).
  Recommendation: yes; builder-check's "silent ×1000" risk goes away.

### Corrections applied to the design (no decision needed)

- **Units contract.** ASF coefficients are in eV: `E¹…E⁻³` terms rescale by
  `1000ⁿ` if the variable changes to keV. Keep the transform in eV
  internally; convert at the API boundary; integration grid is separate
  from the requested output energies.
- **Poles/knots.** Specify the principal-value limit at exact knots,
  neighbouring-segment singularities (upstream patches these), endpoints
  and discontinuities. Masks plus double-`where` only stop NaNs; they do
  not supply the limit.
- **Compound spectra** sum coefficients on the union of elemental segment
  boundaries. Relativistic correction is applied exactly once. The
  formula `Σ nᵢ(Zᵢ − (Zᵢ/82.5)^2.37)` matches upstream (✔); Agy's
  "conflation" objection was rejected as unsupported.
- **Merge normalisation** is affine endpoint matching: require ordered merge
  points inside both datasets, enough samples, a conditioning check on the
  endpoint difference (flat endpoints → inf), and a negative-result
  policy. `--input-type` is required whenever `--nearedge` is given
  (upstream crashes without it). For `beta` input density cancels at the
  merge rescale (builder check) — keep `--density` optional but test it.
- **JIT/backend.** `ops.any/all` return Python `bool` and raise
  `TracerBoolConversionError` under JIT (✔ Codex probe) — the kernel must
  not use them. Static integer orders, no data-dependent branching.
  Chunking is orchestrated separately for NumPy; `lax.map` only on the JAX
  path. Register KK caches so `set_backend()` invalidates them. x64 must
  be enabled *before* tracing and inputs explicitly cast to float64
  (`asarray(float32)` preserves float32, ✔ Codex probe).
- **Tolerances.** This repo's `hc`, Avogadro and classical electron radius
  differ from upstream's by ~1.3e-7, 1.2e-7, 8e-9 (✔ Codex probe).
  Split verification: *transform-only parity* on identical input arrays
  (rtol≈1e-10) vs *end-to-end* (rtol set from measured error). Chunk
  invariance is within tolerance, not bit-for-bit (summation order).
- **`improve_accuracy`.** Tolerance is absolute (f1/f2 units), measured at
  midpoints only; specify the point cap, exhaustion reporting and
  stagnation handling. "Differentiable for free" holds for the kernel, not
  across adaptive decisions — removed from claims.
- **Validation.** Existing helpers are insufficient: `parse_energy_string`
  accepts `"nan"` and returns empty for `"1-2:0"`; `parse_formula("Xx0")`
  returns an unknown symbol with zero count (✔ per Codex). `kk` validates
  its own inputs: finite, supported elements, positive composition,
  density, refine controls, chunk size.
- **Integration points the first draft missed (all ✔):**
  `_LAZY_SUBMODULES` frozenset (`xraylabtool/__init__.py:72`);
  `command_handlers` dict (`interfaces/cli/__init__.py:166`) in addition to
  parser registration; static completion table
  (`completion_v2/shells.py:711`, parity tests in
  `tests/integration/test_cli.py`); `MainWindowProtocol`
  (`gui/main_window/_protocol.py`) must declare the new tab's members under
  `mypy --strict`; `_set_tab_order` (`shared.py:88`) is hardcoded to the
  single/multi tabs; `[tool.setuptools.package-data]` (`pyproject.toml:198`)
  lists only `.nff`; conda recipe `license: MIT` /
  `license_file: LICENSE` (`conda-recipe/meta.yaml:54-55`) needs
  `MIT AND Zlib` plus the upstream licence text.
- **Adapter.** `KKResult.to_xray_result` needs a new calculators helper that
  builds δ/β/derived quantities from supplied f1/f2 without re-running the
  `.nff` lookup, and must preserve negative f1. The helper is part of the
  core stage.
- **Packaging.** Real size is 3.8–5.8 MB (two fetches of upstream's file
  disagreed: master 3.76 MB, main 5.76 MB ✔ — pin one commit). Convert to
  a binary `.npz` at build time (now required, not optional); ship the
  Zlib text, mark altered files, and verify installed wheel/sdist contain
  both.
- **Tests added:** independent principal-value quadrature and constant-band
  analytic case; exact knots and `nextafter` neighbours; discontinuities;
  chunk sizes 1/odd/oversized; shuffled/duplicate eval energies; NumPy↔JAX
  switches; float32 and x64-disabled entry; merge degeneracy; beta round
  trip; eV↔keV equivalence; refinement exhaustion; explicit finiteness
  asserts; installed-artifact data/licence checks. The Lorentz oscillator
  test needs controlled truncation convergence against its infinite-domain
  closed form.
