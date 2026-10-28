---
orphan: true
---

# Codebase review — 2026-10-28

Scope: the production source tree at `c2af42a`, followed by a review of the
combined fixes. Parallel reviews covered calculators/backend/utilities, KK,
and CLI/completion/GUI. The coordinating review covered validation, I/O,
batch data handling, and analysis. Existing specs, ADRs, API documentation,
and public docstrings supplied the behavior contracts. This was a full-tree
audit, rather than a comparison with an earlier release.

The review used the diagnosing-bugs, code-review, and systematic-debugging
skills. Each behavior fix has a regression observed failing before the fix.
The findings below describe confirmed defects; they do not establish that
every possible input or platform is free of defects.

## Standards

One documented-standard finding, fixed: the newly exported
`calculators.host.run_derived_fields` lacked the NumPy-style Parameters and
Returns sections and physical units required by AGENTS.md and
`docs/development/contributing.rst`. Its documentation now describes metres,
g/cm³, g/mol, output units, and independent scalar/array/empty shapes.

One heuristic smell, addressed: **possible Duplicated Code** in the manually
copied completion schema. The duplicated parser definitions had demonstrably
drifted; completion metadata now comes from argparse. No other actionable
smells were confirmed. Tooling-enforced findings are covered by Ruff and mypy.

## Spec

All 20 confirmed behavior findings below are fixed.

| Area | Reproduction / previous symptom | Root cause and correction |
| --- | --- | --- |
| Derived quantities | Independent dispersion/absorption shapes or empty operands raised broadcast/reshape errors. | A shared broadcast grid incorrectly coupled independent formulas. One device call now uses each field's own grid, preserving the §8 legacy contract. |
| Result accessor | Zero attenuation length emitted division-by-zero warnings. | `np.where` evaluated the quotient eagerly. Masked `np.divide` preserves the intended zero result. |
| Density validation | NumPy complex density silently lost its imaginary component. | Complex values passed the numeric gate. Reject before real conversion. |
| Energy validation | Complex/empty energies passed; nonnumeric energies leaked NumPy errors. | Dtype and cardinality were unchecked. Raise the documented `EnergyError`. |
| Atomic-data provider | Complex energy arrays lost imaginary components. | Float conversion preceded dtype validation. Raise `EnergyError` before conversion. |
| KK scalar inputs | Complex merge points, tolerances, density/mass, and counts reached real casts. | Broad numeric gates admitted complex scalars. Enforce the KK spec's “complex-input rejection” before conversion. |
| KK refinement | Empty/scalar/2-D segment grids leaked indexing or ambiguous-truth errors. | Endpoint access preceded shape checks. Validate the documented segment shape first. |
| File loading | Headerless CSV/NFF lost its first numeric row; whitespace headers failed. | Unconditional header skipping and inconsistent delimiter fallbacks. Detect the first content row's delimiter and optional header. |
| Result formatting | Formatting evaluated deprecated aliases and could fail with warnings treated as errors. | Reflection inspected every property. Use dataclass fields for result data. |
| CSV summary | Multi-energy arrays disappeared; methods entered the export. | Summary retained only length-one arrays and unfiltered attributes. Reuse data-only, one-row-per-energy formatting. |
| Selected CSV fields | First requested field was replaced by `formula`. | Wide export assumed formula occupied the first field. Honor the entire requested list and order. |
| Empty batch input | Header-only input with an energy column raised `IndexError`. | Loader inspected the first cell unconditionally. Parse cells individually, allowing an empty list. |
| Batch energy column | An explicitly requested absent energy column was silently ignored. | Required-column checks omitted it. Include requested energy columns in validation. |
| Material comparison | Zero critical angles crashed; negative means suppressed variation warnings. | Variation divided by the signed mean. Compare standard deviation against half the absolute mean instead. |
| Batch CLI rows | A truncated CSV row aborted processing of subsequent valid rows. | Missing cells produced uncaught `TypeError`. Apply the existing skip-and-warn behavior. |
| Verbose batch export | `--fields energy_kev,dispersion_delta --verbose` failed after saving. | Summary read formula after projection. Count materials before selecting fields. |
| GUI material entry | Invalid formula/density escaped the slot instead of showing a toast. | The handler caught `ValueError` but validators raise domain exceptions. Catch those at the UI boundary. |
| Completion schema | Supported flags/commands were missing and removed flags were suggested. | Static metadata drifted from argparse. Derive it from the parser. |
| Bash format choices | Completion offered formats the selected command rejects. | One hardcoded wordlist served every command. Use each option's actual choices. |
| Bash global flags | Leading `--verbose` broke command, option, and subcommand completion. | Fixed word offsets assumed the command immediately followed the executable. Locate its actual position. |

Regression coverage is in the changed existing tests, the new
`tests/unit/test_review_*_regressions.py` files, and
`tests/integration/test_completion_behavior.py`. The Bash tests execute the
generated script, and numerical regression tests exercise real calculations.

## Follow-up review

A second, independent read-only review of the fixes above found and corrected:

- KK element counts again accept JAX arrays, 0-d arrays, `Fraction` and bool;
  only complex and non-numeric counts are rejected.
- The derived-quantity grids are padded to one common width, so mixed scalar and
  array calls share one compiled program per bucket (previously up to three).
- `linear_absorption_coefficient` computes in float64 for any assigned dtype.
- The atomic-data provider rejects non-finite and non-numeric energies; batch
  input rejects blank or non-finite energy cells.
- Batch rows with unquoted extra energy cells are skipped with a warning rather
  than truncated; batch input accepts a UTF-8 BOM.
- CSV export raises `ValueError` for unknown fields instead of writing a ragged
  file; the default CSV/JSON again include `energy_ev`, `critical_angle_mrad`
  and `linear_absorption_coefficient`.
- Zsh and fish completion descriptions are escaped.

Not changed: `refine_spectrum` keeps its documented `ValueError` for `seg_ev`;
a data file whose header row is literally `nan` or `inf` is read as data;
duplicate batch column names keep the last column.
Regressions: `tests/unit/test_review_followup_regressions.py`.

## Verification

- 55 additional test cases; the final tree collects 2,590 tests.
- Final combined regression-file run: 159 passed.
- Serial offscreen GUI run: 77 passed.
- Focused CUDA run: 97 passed, with `jax.default_backend() == "gpu"` and
  `CudaDevice(id=0)` confirmed. It covers derived/result contracts, KK input
  hardening, and analysis regressions.
- Ruff lint and formatting, strict mypy across all 87 source files, and
  `git diff --check`: passed.
- Sphinx HTML build with warnings treated as errors: passed.
- Final full non-GUI suite: 2,460 passed, 53 skipped in 489.92 seconds.
  Skips cover 51 opt-in performance cases and two device-rule cases.
- Combined CPU/offscreen-GUI result: 2,537 passed, 53 skipped, covering all
  2,590 collected cases. No failures remain.

An earlier full run overlapped source edits and retained the old completion
definitions in its workers: 11 completion regressions failed while 2,438
tests passed. A fresh parallel completion run passed all 81 cases. The final
full run started after all source edits, with the source tree held unchanged,
and passed without failures.

The non-GUI verification command uses `JAX_PLATFORMS=cpu`,
`QT_QPA_PLATFORM=offscreen`, and `uv run --no-sync pytest tests -n 4
--dist loadscope --benchmark-disable --no-cov -q`, with the eight GUI files
listed by `GUI_TESTS` in the Makefile excluded. Those eight files run
separately without xdist, with `--timeout=60`. `--no-sync` preserves the
installed CUDA extras.

The finite-value smoke check passed for five materials, 257 energies spanning
0.03–30 keV, and all ten result arrays. No numerical divergence was observed.

The repository's required `graphify update .` completed using AST extraction,
without a semantic API call. Semantic search was unavailable because the
Windsurf credential was absent; graphify and local searches supplied context.

Limits: focused CUDA regressions do not replace a dedicated GPU
acceptance/benchmark run. PowerShell runtime was unavailable. Opt-in
performance tests were not enabled. No commit, push, or publication was made.

Standards: one documented finding and one heuristic smell, both addressed;
the documented issue was missing public helper documentation. Spec: 20
behavior findings fixed; the most consequential were silent CSV data loss
and lossy complex-input conversion.
