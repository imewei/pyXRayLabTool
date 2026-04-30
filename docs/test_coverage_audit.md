# Test Coverage & Quality Audit — xraylabtool

**Date:** 2026-04-07
**Reviewer role:** Testing
**Scope:** `tests/` directory vs `xraylabtool/` source

---

## Coverage Mapping

| Source Module | Test File(s) | Coverage Status |
|---|---|---|
| `calculators/core.py` | `unit/test_core.py`, `unit/test_numerical_stability.py`, `characterization/test_golden_pipeline.py`, `characterization/test_golden_interpolation.py` | Partial — golden tests strong, unit shallow |
| `calculators/derived_quantities.py` | `characterization/test_golden_derived_quantities.py` | Characterization only |
| `data_handling/atomic_cache.py` | `unit/test_atomic_data.py` (via `utils`), `performance/test_performance_benchmarks.py` | Indirect only |
| `data_handling/batch_processing.py` | `performance/test_memory_management.py` | Partial |
| `data_handling/compound_analysis.py` | None | **UNTESTED** |
| `backend/array_ops.py` | None directly | **UNTESTED** |
| `backend/interpolation.py` | `characterization/test_golden_interpolation.py` | Indirect |
| `analysis/comparator.py` | `unit/test_advanced_analysis.py` | Partial |
| `export/__init__.py`, `io/data_export.py` | `unit/test_export_functionality.py`, `integration/test_export_integration.py` | Moderate |
| `io/file_operations.py` | None | **UNTESTED** |
| `validation/validators.py` | `unit/test_enhanced_validation.py` | Present |
| `exceptions.py` | `unit/test_exceptions.py` | Good |
| `utils.py` | `unit/test_utils.py`, `unit/test_atomic_data.py`, `unit/test_formula_parsing.py` | Good |
| `optimization/vectorized_core.py` | None | **UNTESTED** |
| `optimization/optimized_core.py` | None | **UNTESTED** |
| `optimization/bottleneck_analyzer.py` | None | **UNTESTED** |
| `interfaces/cli.py` | `integration/test_cli.py`, `integration/test_cli_edge_cases.py`, `unit/test_cli_enhanced_error_handling.py` | Moderate |
| `interfaces/completion_v2/` | None | **UNTESTED** |
| `gui/main_window.py` | `test_gui_smoke.py` | Smoke only |
| `gui/workers.py` | `test_gui_progress.py` (partial), `test_gui_smoke.py` | Worker.run() excluded via pragma |
| `gui/services.py` | `test_gui_smoke.py`, `test_gui_progress.py` | Smoke only |
| `gui/widgets/` | None directly | **UNTESTED** |
| `gui/logging_filters.py` | None | **UNTESTED** |
| `gui/theme_manager.py` | None | **UNTESTED** |
| `gui/style.py` | None | **UNTESTED** |
| `gui/protocols.py` | None | **UNTESTED** |
| `constants.py` | `characterization/test_golden_constants.py` | Good |
| `logging_utils.py` | `unit/test_logging_utils.py` | Present |
| `typing_extensions.py` | `unit/test_enhanced_xray_result_typing.py`, `unit/test_type_safety_config.py` | Present |
| `device.py` | None | **UNTESTED** |

---

## Findings

---

### TEST-001
**Severity:** Critical
**Category:** Coverage
**Location:** `xraylabtool/data_handling/compound_analysis.py`
**Description:** `compound_analysis.py` — which contains `COMMON_COMPOUNDS`, `ELEMENT_FAMILIES`, and the cache-warming logic — has zero test coverage. The module is used to intelligently pre-load element data at startup.
**Impact:** Regressions in compound-to-element mapping or cache-warming logic will not be caught. Wrong element counts for complex compounds propagate silently into all downstream calculations.
**Recommendation:** Add `tests/unit/test_compound_analysis.py` testing: correct element decomposition for representative compounds across all families (silicates, carbides, halides, etc.), unknown compound handling, and element family enumeration correctness.
**Effort:** Small

---

### TEST-002
**Severity:** Critical
**Category:** Coverage
**Location:** `xraylabtool/backend/array_ops.py`
**Description:** The `NumpyBackend` and `JaxBackend` classes implementing `ArrayBackend` protocol have no direct unit tests. The backend dispatch path (which determines whether JAX or NumPy is used for all calculations) is exercised only indirectly through higher-level tests.
**Impact:** Switching between backends (e.g., via `set_backend()`) could silently produce numerically different results. Type promotion mismatches between JAX float32 defaults and NumPy float64 would not be caught until end-to-end golden tests fail.
**Recommendation:** Add `tests/unit/test_backend.py` that: instantiates both backends, verifies each operation produces identical float64 results for a set of reference arrays, checks `asarray` dtype promotion, and tests the backend protocol compliance check.
**Effort:** Medium

---

### TEST-003
**Severity:** Critical
**Category:** Coverage
**Location:** `xraylabtool/optimization/vectorized_core.py`, `optimized_core.py`, `bottleneck_analyzer.py`
**Description:** All three files in `optimization/` have zero test coverage. `vectorized_core.py` contains `vectorized_interpolation_batch` and `ensure_c_contiguous`; `optimized_core.py` contains an alternative cache and `OptimizedScatteringData`; `bottleneck_analyzer.py` is entirely unexercised.
**Impact:** The `ensure_c_contiguous` decorator silently skips conversion for JAX arrays; a regression could corrupt memory layout. `OptimizedScatteringData.__getitem__` for unknown column keys has no error-path test.
**Recommendation:** Mark `optimization/` modules as deprecated (they already have deprecation docstrings) and add a single `tests/unit/test_optimization_compatibility.py` verifying that `enable_optimizations()`/`disable_optimizations()` do not change numerical output vs baseline.
**Effort:** Small

---

### TEST-004
**Severity:** High
**Category:** Coverage
**Location:** `xraylabtool/calculators/core.py` — `AtomicScatteringFactor`, `CrystalStructure`
**Description:** `test_core.py` contains only 4 trivial tests for `AtomicScatteringFactor` and `CrystalStructure`. The `calculate_structure_factor` method is tested only with a single atom at origin (trivial phase = 0). `get_scattering_factor` is tested only for shape/type, not values.
**Impact:** Phase calculation errors in `calculate_structure_factor`, wrong scattering factor values, or off-by-one errors in multi-atom structures will not be detected.
**Recommendation:** Add tests: structure factor for BCC/FCC with known systematic absences; scattering factor values cross-checked against NIST reference data; partial occupancy effect on structure factor magnitude.
**Effort:** Medium

---

### TEST-005
**Severity:** High
**Category:** Coverage
**Location:** `xraylabtool/io/file_operations.py`
**Description:** `file_operations.py` has no tests. File I/O operations (reading, writing, format detection) are integration-critical and platform-sensitive.
**Impact:** File path handling bugs, encoding issues, or permission errors in production I/O paths will not be caught in CI.
**Recommendation:** Add `tests/unit/test_file_operations.py` with tests for: valid file round-trip, missing file error, permission-denied handling (mock), unsupported format detection.
**Effort:** Small

---

### TEST-006
**Severity:** High
**Category:** Coverage
**Location:** `xraylabtool/interfaces/completion_v2/` (6 files: `cache.py`, `cli.py`, `environment.py`, `installer.py`, `integration.py`, `shells.py`)
**Description:** The entire `completion_v2/` subsystem has no tests. It handles shell detection, cache management, and installer invocation.
**Impact:** Shell completion installs could fail silently or corrupt shell config files on user machines.
**Recommendation:** Add `tests/unit/test_completion_v2.py` covering: shell detection mocking, cache read/write round-trip, installer dry-run with mocked subprocess, integration entrypoint.
**Effort:** Medium

---

### TEST-007
**Severity:** High
**Category:** EdgeCase
**Location:** `xraylabtool/utils.py` — `parse_formula`
**Description:** `test_formula_parsing.py` does not test: empty string `""`, formula with only numbers `"123"`, deeply nested parentheses `"Ca3(PO4)2"` (the compound analysis table contains this exact formula), unicode lookalike characters, or extremely long formulas.
**Impact:** A malformed formula that crashes the parser could propagate as an unhandled exception through the CLI/GUI without a meaningful error message.
**Recommendation:** Add edge-case tests: empty string, purely numeric input, `Ca3(PO4)2` parentheses, formula with trailing whitespace, single-character invalid input.
**Effort:** Small

---

### TEST-008
**Severity:** High
**Category:** Numerical
**Location:** `tests/unit/test_numerical_stability.py` — `TestPhysicalRealism`
**Description:** `test_critical_angle_monotonicity` asserts strict monotonic decrease using a hand-coded loop with `>=` but uses only 4 energy points. Near absorption edges, the critical angle is non-monotonic in practice. The test will pass trivially far from edges but provides false confidence.
**Impact:** Near-edge calculations that violate the assumed monotonicity will not be detected, and the test assertion itself could start failing for physically correct edge-region data.
**Recommendation:** Replace with a test that verifies the physical formula `θc = sqrt(2δ)` numerically against known SiO2 values at 10 keV to ±0.1%, rather than assuming global monotonicity.
**Effort:** Small

---

### TEST-009
**Severity:** High
**Category:** Quality
**Location:** `tests/unit/test_core.py` — `TestLoadDataFile.test_nonexistent_file`
**Description:** The only test for `load_data_file` checks a nonexistent file. There are no tests for: a valid file load, a malformed/corrupted data file, a file with wrong column count, or a file with mixed NaN/Inf rows.
**Impact:** The positive path (loading a real data file) is completely untested at the unit level. Regressions in data file parsing will only surface through high-level integration tests.
**Recommendation:** Add tests using `tmp_path` fixture for: valid 3-column NIST-format file, file with header rows, file with NaN values in f1/f2 columns.
**Effort:** Small

---

### TEST-010
**Severity:** High
**Category:** Flaky
**Location:** `tests/performance/test_performance_benchmarks.py`, `test_speed_optimization_benchmarks.py`
**Description:** Performance tests assert hard wall-clock time limits (e.g., `avg_time < 0.005`, `throughput > 200`). These run without `@pytest.mark.skip` guards for CI resource constraints and are highly environment-dependent. The `test_throughput_benchmark` runs for 1 full second unconditionally. `test_speed_optimization_benchmarks.py` imports `BasePerformanceTest` which has its own timing infrastructure, leading to duplicated measurement with different baselines.
**Impact:** Spurious CI failures on resource-constrained runners; time wasted investigating non-bugs. Conversely, truly regressed performance may be masked by environment variance.
**Recommendation:** Mark all timing-based tests `@pytest.mark.performance` and exclude from default test run (add `-m "not performance"` to default pytest invocation). Replace absolute time limits with relative regression checks (e.g., "no more than 3× slower than baseline on same machine").
**Effort:** Small

---

### TEST-011
**Severity:** Medium
**Category:** GUI
**Location:** `xraylabtool/gui/widgets/material_form.py`, `material_table.py`, `plot_canvas.py`, `sweep_plots.py`
**Description:** Widget-level GUI code has no unit tests. `test_gui_smoke.py` tests only the happy path through `MainWindow` at a high level. Widget signals, slot connections, and input validation in `material_form.py` are not individually tested.
**Impact:** Input validation regressions (e.g., accepting negative densities in the form, accepting empty formula strings) will not be caught without a full smoke test run that requires PySide6.
**Recommendation:** Add `tests/unit/test_gui_widgets.py` with `@pytest.mark.skipif(no_pyside6)` guard, testing `material_form.py` input validation: empty formula rejection, density range validation, formula parsing error display.
**Effort:** Medium

---

### TEST-012
**Severity:** Medium
**Category:** GUI
**Location:** `xraylabtool/gui/workers.py` — `CalculationWorker.run`
**Description:** `CalculationWorker.run` is marked `# pragma: no cover`. The error signal emission path (exception in `fn`) is completely untested. Signal/slot wiring is not verified.
**Impact:** If a calculation exception is swallowed rather than emitted as an `error` signal, the GUI will silently freeze with no user feedback.
**Recommendation:** Remove the `pragma: no cover` exemption. Add a unit test (with offscreen Qt) that: (1) creates a worker with a function that raises, (2) connects the `error` signal to a recorder, (3) runs the worker, (4) asserts the error signal was emitted with a non-empty message.
**Effort:** Small

---

### TEST-013
**Severity:** Medium
**Category:** Performance
**Location:** `tests/performance/test_memory_management.py` — `test_force_gc_with_cache_clearing`
**Description:** The test contains `assert True` as its only assertion — this is a tautological no-op. The test passes regardless of whether `force_gc()` actually clears caches or whether it silently raises internally.
**Impact:** A regression in `force_gc` (e.g., cache not cleared, exception suppressed) will not be detected.
**Recommendation:** Replace `assert True` with: call `force_gc()`, then call `get_atomic_data_fast` again and measure that the cache miss path is re-entered (check `cache_info().misses` increments).
**Effort:** Small

---

### TEST-014
**Severity:** Medium
**Category:** ErrorPath
**Location:** `xraylabtool/data_handling/batch_processing.py` — `calculate_batch_properties`
**Description:** Error handling in batch processing (partial failures, mixed valid/invalid formulas, memory limit exceeded mid-batch) is not tested. `test_memory_management.py` tests `MemoryMonitor` in isolation but not the full batch pipeline under error conditions.
**Impact:** A single bad formula in a 100-material batch could either crash the entire batch or silently produce incomplete results depending on the exception handling path.
**Recommendation:** Add integration test: batch with one invalid formula among valid ones; verify `BatchProcessingError.failed_items` is populated, valid results are still returned, and no silent data loss occurs.
**Effort:** Small

---

### TEST-015
**Severity:** Medium
**Category:** Numerical
**Location:** `tests/integration/test_integration.py`
**Description:** Integration tests access deprecated `sio2.Dispersion`, `sio2.f1`, etc. (CamelCase aliases) rather than the current snake_case fields. This means the snake_case field correctness is not validated at the integration level — only the deprecated aliases are checked.
**Impact:** A regression in the snake_case fields would not be caught by integration tests if the deprecated-alias wrapper caches or copies values separately.
**Recommendation:** Add parallel assertions using `sio2.dispersion_delta`, `sio2.scattering_factor_f1`, etc. to verify both accessor paths produce identical results.
**Effort:** Small

---

### TEST-016
**Severity:** Medium
**Category:** EdgeCase
**Location:** `xraylabtool/calculators/core.py` — `calculate_single_material_properties`
**Description:** No tests cover: a formula with an element not in the preloaded `_ATOMIC_DATA_PRELOADED` dict (falls through to Mendeleev lookup), concurrent calls from multiple threads, or a formula with isotope notation.
**Impact:** Thread safety of the `lru_cache`/`@cache` decorated functions is assumed but not verified. Concurrent GUI workers could race on cache population.
**Recommendation:** Add a threading test that calls `calculate_single_material_properties` from 8 threads simultaneously for different materials and asserts all results are numerically identical to serial results.
**Effort:** Medium

---

### TEST-017
**Severity:** Low
**Category:** Coverage
**Location:** `xraylabtool/device.py`
**Description:** `device.py` has no tests.
**Impact:** Low risk if this is a thin configuration module, but depends on its actual content.
**Recommendation:** Read the module; if it contains device/backend selection logic, add a unit test.
**Effort:** Small

---

### TEST-018
**Severity:** Low
**Category:** Quality
**Location:** `tests/test_ci_cd_integration.py`, `tests/test_code_quality.py`
**Description:** These top-level test files run linting/CI checks as tests (e.g., subprocess calls to `ruff`, `mypy`). These are process-level checks masquerading as pytest tests, making CI output confusing and the test suite slower.
**Impact:** Failures in these tests produce confusing "test failed" output instead of standard CI step failures. They also add subprocess overhead to every pytest run.
**Recommendation:** Move these checks to dedicated CI pipeline steps (already present in `.github/workflows/ci.yml`) rather than running them via pytest.
**Effort:** Small

---

## Testing Pyramid Assessment

```
E2E / GUI smoke:     2 tests   (2%)   — very thin
Integration:        ~30 tests  (20%)  — moderate
Unit:              ~200 tests  (65%)  — reasonable
Characterization:   ~40 tests  (13%)  — golden values
Performance:        ~40 tests  (separate) — timing-based, fragile
```

**Assessment:** The pyramid is inverted toward characterization/integration tests for the numerical core, which is appropriate for a scientific computing package where correctness of golden values matters. However, unit coverage of supporting modules (optimization/, backend/, gui/widgets/, completion_v2/, io/, device.py) is thin to nonexistent. Performance tests are structurally fragile (hard wall-clock assertions).

---

## Priority Recommendations (ordered)

1. **TEST-001** — Test `compound_analysis.py` (compound→element decomposition is foundational)
2. **TEST-002** — Test `backend/array_ops.py` (JAX/NumPy dispatch is the highest-risk migration surface)
3. **TEST-013** — Fix tautological `assert True` in memory management test
4. **TEST-010** — Guard performance tests with marker; remove from default run
5. **TEST-012** — Test `CalculationWorker` error path; remove `pragma: no cover`
6. **TEST-004** — Expand `CrystalStructure` / `AtomicScatteringFactor` unit tests
7. **TEST-014** — Test batch processing partial failure paths
8. **TEST-007** — Add formula parser edge cases (empty, parentheses, unicode)
