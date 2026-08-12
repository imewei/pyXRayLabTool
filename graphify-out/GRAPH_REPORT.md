# Graph Report - pyXRayLabTool  (2026-08-12)

## Corpus Check
- 139 files · ~123,948 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 3484 nodes · 5891 edges · 176 communities (141 shown, 35 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 433 edges (avg confidence: 0.6)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6f558db5`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- xraylabtool/__init__.py
- FormulaError
- interfaces/cli.py
- tests/conftest.py
- StyleGuideValidator
- EnvironmentDetector
- CompletionInstaller
- parse_formula
- Any
- timer
- test_cli_edge_cases.py
- device.py
- _calculate_single_material_xray_properties
- MainWindow
- patch
- ndarray
- MemoryMonitor
- EnvironmentInfo
- validate_chemical_formula
- BatchConfig
- TestBashCompletionScript
- TestPerformanceBenchmarks
- EnergyError
- BaseUnitTest
- batch_processing.py
- main_window.py
- EnvironmentType
- completion_v2/cli.py
- create_scattering_factor_interpolators
- MockXRayResult
- test_backend_dispatch.py
- TestCICDIntegration
- calculate_single_material_properties
- TestEnergySweepSi
- TestFormatCalculationSummary
- typing_extensions.py
- normalize_intensity
- _calc
- MaterialInputForm
- TestBenchmarkComparison
- test_utilities.py
- TestGoldenAu
- BasePerformanceTest
- Findings
- BaseIntegrationTest
- ADR-004: Host-Device Transfer Minimization Strategy
- Utilities Module
- AtomicDataProvider
- TestFullPipelineSiO2
- TestBasicAnalysis
- TestTypeSafetyConfig
- atomic_cache.py
- current_palette
- test_docs.py
- TestNumericalStabilityChecks
- configure_logging
- CalculationWorker
- MaterialComparator
- process_single_calculation
- TestCalculationSpeedBenchmarks
- TestCIWorkflowSimulation
- TestNumpyBackend
- ._build_multi_tab
- ColorPalette
- test_formula_parsing.py
- legacy_install_completion_main
- format_xray_result
- core.py
- calculate_transmission
- arrays_close
- MemoryProfiler
- PerformanceBenchmark
- parse_energy_string
- CompletionManager
- test_concurrent_cache.py
- _make_cubic
- BaseXRayLabToolTest
- test_file_operations.py
- .__init__
- CI/CD Security Audit Report
- export_to_json
- main
- calculate_critical_angle
- clear_scattering_factor_cache
- TestCLICompatibility
- JaxBackend
- ThemeManager
- PlotCanvas
- CompletionGenerator
- Backend Abstraction
- TestGoldenSi
- TestGoldenFe2O3
- TestGoldenSiO2
- TestGoldenScatteringFactors
- TestPhysicalRealism
- TestParseFormulaDecimal
- TestEnvironmentDetection
- AtomicScatteringFactor
- TestExtremeConditions
- TestParseFormulaErrors
- TestParseFormulaParentheses
- ADR-003: float64 Preservation Policy
- ParametrizedTestMixin
- OverlayScrollbarMarginHelper
- `test_docs.py` - Documentation Testing
- TestUnitConversionConstants
- ndarray
- TestComplexFormula
- PerformanceTimer
- NumpyBackend
- analyze_element_associations
- apply_palette_to_item
- _handle_mendeleev_error
- xraylabtool — Unified Audit & Remediation Plan
- .get_scattering_factors
- test_golden_derived_quantities.py
- calculate_attenuation_length
- Shell Completion Guide
- TestCompletionScriptGeneration
- uninstall_completion_main
- BaseIntegrationTest
- ADR-001: JAX vs NumPy Computation Backend
- TestGoldenScatteringLengthDensity
- BashCompletionGenerator
- TestBatchConfig
- get_compound_family
- parse_chemical_formula
- cache.py
- TableFormatter
- ADR-002: PyQtGraph vs Matplotlib for GUI Plotting
- TestMultiMaterialEquivalence
- TestBatchProcessingMemoryManagement
- RegressionTracker
- Files
- TestGoldenDerivedQuantities
- find_similar_compounds
- get_compound_complexity_score
- get_compound_frequency_score
- get_elements_for_compound
- FastXRayCalculationEngine
- PerformanceMetrics
- ResultValidator
- performance/conftest.py
- test_compound_analysis.py
- test_core.py
- XRayResult
- Complex refractive index n=1-delta-i*beta
- Testing Guide
- conf.py
- _auto_select_backend
- X-ray Calculations Guide
- pyXRayLabTool Logo (Main SVG)
- Test Coverage Audit
- characterization/__init__.py
- fixtures/__init__.py
- tests/__init__.py
- integration/__init__.py
- performance/__init__.py
- unit/__init__.py
- BasePerformanceTest
- BaseUnitTest
- BaseXRayLabToolTest
- Analysis API
- Backend API
- Constants API
- Data Handling API
- IO Operations API
- Validation API
- Rollback Procedures
- Examples Index
- Migration Guide v0.4
- pyXRayLabTool Favicon
- GUI Main Window Screenshot

## God Nodes (most connected - your core abstractions)
1. `CompletionInstaller` - 72 edges
2. `calculate_single_material_properties()` - 67 edges
3. `MainWindow` - 55 edges
4. `FormulaError` - 52 edges
5. `EnergyError` - 52 edges
6. `parse_formula()` - 52 edges
7. `BatchConfig` - 49 edges
8. `XRayResult` - 47 edges
9. `EnvironmentInfo` - 39 edges
10. `calculate_xray_properties()` - 36 edges

## Surprising Connections (you probably didn't know these)
- `_single()` --calls--> `clear_scattering_factor_cache()`  [INFERRED]
  tests/characterization/test_golden_integration.py → xraylabtool/calculators/cache.py
- `TestFullPipelineSiO2` --uses--> `BatchConfig`  [INFERRED]
  tests/characterization/test_golden_integration.py → xraylabtool/data_handling/batch_processing.py
- `TestMultiMaterialEquivalence` --uses--> `BatchConfig`  [INFERRED]
  tests/characterization/test_golden_integration.py → xraylabtool/data_handling/batch_processing.py
- `TestEnergySweepSi` --uses--> `BatchConfig`  [INFERRED]
  tests/characterization/test_golden_integration.py → xraylabtool/data_handling/batch_processing.py
- `cold_cache()` --calls--> `clear_scattering_factor_cache()`  [INFERRED]
  tests/characterization/test_golden_interpolation.py → xraylabtool/calculators/cache.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Modular Architecture v0.4.0** — xraylabtool_calculators, xraylabtool_data_handling, xraylabtool_backend, xraylabtool_gui, xraylabtool_interfaces_cli [EXTRACTED 1.00]
- **CLI Command Suite** — xraylabtool_interfaces_cli, xraylabtool_interfaces_completion_v2 [EXTRACTED 0.90]
- **v0.4.0 Architecture Overhaul** — xraylabtool_backend, xraylabtool_gui, xraylabtool_calculators [EXTRACTED 1.00]
- **JAX Architecture Decision Records** — docs_architecture_001_adr_jax_vs_numpy_jax_decision, docs_architecture_003_adr_dtype_policy_dtype_strategy, architecture_004_adr_host_device_transfers_transfer_policy, docs_architecture_jax_architecture_jax_design [INFERRED 0.85]

## Communities (176 total, 35 thin omitted)

### Community 0 - "xraylabtool/__init__.py"
Cohesion: 0.03
Nodes (81): Tests for atomic data lookup functions with LRU caching., Test that element symbols are case-sensitive., Test atomic data lookup with caching., Test atomic number lookup for valid elements., Test atomic weight lookup for valid elements., Test comprehensive atomic data lookup., Test proper exceptions for unknown elements., Test that LRU caching works correctly. (+73 more)

### Community 1 - "FormulaError"
Cohesion: 0.05
Nodes (62): Tests for custom exception classes. This module tests the behavior of domain-…, Test DataFileError class., Test basic error message., Test error with filename context., Test ValidationError class., Test basic error message., Test error with parameter name., Test error with parameter and value. (+54 more)

### Community 2 - "interfaces/cli.py"
Cohesion: 0.04
Nodes (106): add_atomic_command(), add_batch_command(), add_bragg_command(), add_calc_command(), add_compare_command(), add_completion_command(), add_convert_command(), add_formula_command() (+98 more)

### Community 3 - "tests/conftest.py"
Cohesion: 0.04
Nodes (66): assert_calculation_result_valid(), assert_memory_usage_reasonable(), assert_performance_within_threshold(), batch_config_default(), batch_config_performance(), batch_config_small(), benchmark_baseline(), clean_cache() (+58 more)

### Community 4 - "StyleGuideValidator"
Cohesion: 0.07
Nodes (30): Colors, main(), Path, Get all Python files in the project., Validate import patterns according to style guide., Validate naming conventions according to style guide., Validate type hint usage according to style guide., Validate docstring patterns according to style guide. (+22 more)

### Community 5 - "EnvironmentDetector"
Cohesion: 0.07
Nodes (27): EnvironmentDetector, Any, Path, Discover all available Python environments., Detect the type of the currently active environment., Check if the current environment is using mamba., Detect whether a conda environment is actually mamba-managed., Get the path of the current environment. (+19 more)

### Community 6 - "CompletionInstaller"
Cohesion: 0.05
Nodes (28): Test install-completion command shows help correctly., Integration tests for CompletionInstaller (item 4.1)., Calling _modify_activation_script twice does not duplicate blocks., Uninstall strips the sentinel-delimited block but leaves surrounding content., Tests that install/uninstall work with real temp directories., Path written into activate script is properly shell-quoted., Tests for XRAYLABTOOL_COMPLETION_BEGIN/END sentinel markers., TestCompletionInstallToTempDir (+20 more)

### Community 7 - "parse_formula"
Cohesion: 0.18
Nodes (6): Test basic formula parsing., CO (carbon monoxide) vs Co (cobalt)., Repeated elements in flat formulas are aggregated., TestParseFormulaBasic, parse_formula(), Parse a chemical formula string into element symbols and their counts.…

### Community 8 - "Any"
Cohesion: 0.10
Nodes (4): Array, ArrayBackend, Any, Protocol

### Community 9 - "timer"
Cohesion: 0.07
Nodes (24): Sample Materials Test Data, performance, Performance benchmark tests for optimization validation. This test module…, Benchmark batch calculation performance., Benchmark single vs multi-element performance difference., Benchmark performance with large energy arrays., Benchmark overall throughput (calculations per second)., Performance regression tests. (+16 more)

### Community 10 - "test_cli_edge_cases.py"
Cohesion: 0.06
Nodes (24): Path, Edge case tests for XRayLabTool CLI. This module tests boundary conditions,…, Test CLI file operations edge cases., Test file operations with various path conditions., Test batch processing file edge cases., Test handling of malformed batch input files., Test CLI input validation edge cases., Test CLI error handling edge cases. (+16 more)

### Community 11 - "device.py"
Cohesion: 0.05
Nodes (31): Tests for GPU/device detection in xraylabtool/device.py., Parses CUDA version string from nvcc --version output., Returns (None, None) when stdout contains no recognizable release line., Returns False gracefully when no GPU hardware is detected., Returns False (not an exception) when JAX is not importable., Returns (None, None) gracefully when nvidia-smi is not on PATH., check_gpu_availability always returns a bool., get_device_info returns a dict containing all required keys. (+23 more)

### Community 12 - "_calculate_single_material_xray_properties"
Cohesion: 0.14
Nodes (18): calculate_multiple_xray_properties(), _calculate_single_material_xray_properties(), _convert_energy_input(), Any, EnergyArray, FloatLike, ndarray, Validate inputs for single material calculation. (+10 more)

### Community 13 - "MainWindow"
Cohesion: 0.10
Nodes (7): QCloseEvent, QMainWindow, MainWindow, Export a PyQtGraph plot widget to a PNG file. The plot widgets wrap either a…, Force update of all plots to match new theme., Drop queued-but-unstarted compute jobs so the app shuts down promptly instead…, Replace characters unsafe for filenames with underscores.

### Community 14 - "patch"
Cohesion: 0.08
Nodes (13): patch, Test main function with no arguments., Test main function with help argument., Test main function with version argument., Test main function with invalid command., Test that all expected commands are available in CLI., Test full calculation workflow through CLI., Test full batch processing workflow. (+5 more)

### Community 15 - "ndarray"
Cohesion: 0.08
Nodes (13): Any, ndarray, Deprecated: Use 'energy_kev' instead., Deprecated: Use 'wavelength_angstrom' instead., Deprecated: Use 'dispersion_delta' instead., Deprecated: Use 'absorption_beta' instead., Deprecated: Use 'scattering_factor_f1' instead., Deprecated: Use 'scattering_factor_f2' instead. (+5 more)

### Community 16 - "MemoryMonitor"
Cohesion: 0.06
Nodes (24): MemoryTracker, Context manager for tracking memory usage., performance, Tests for memory management and batch processing optimizations. This test…, Test the MemoryMonitor class and its optimizations., Test cache management optimizations., Test MemoryMonitor initialization., Test cache statistics for monitoring. (+16 more)

### Community 17 - "EnvironmentInfo"
Cohesion: 0.07
Nodes (23): Test EnvironmentInfo with Python version., Test EnvironmentInfo equality comparison., Test EnvironmentInfo string representation., Tests for EnvironmentInfo class., Test creating EnvironmentInfo instance., Test that EnvironmentInfo has required attributes., Test EnvironmentInfo with specific type., Test EnvironmentInfo with environment path. (+15 more)

### Community 18 - "validate_chemical_formula"
Cohesion: 0.03
Nodes (58): Tests for validation/validators.py module. This module tests validation…, Test that empty string raises FormulaError., Test that whitespace-only string raises FormulaError., Test that None input raises FormulaError., Test that invalid characters raise FormulaError., Test that unknown element raises FormulaError., Test that formula with spaces is handled., Tests for validate_density function. (+50 more)

### Community 19 - "BatchConfig"
Cohesion: 0.14
Nodes (13): Characterization tests for end-to-end integration paths (Plan section 3).…, calculate_batch_properties parallel result must match single-material calls to…, TestBatchProcessorEquivalence, Test memory monitoring during batch processing., The result dict has an entry for every formula submitted., Valid formulas yield XRayResult even when invalid ones co-exist., TestBatchInputValidation, TestBatchPartialFailure (+5 more)

### Community 20 - "TestBashCompletionScript"
Cohesion: 0.05
Nodes (24): Tests for interfaces/completion.py bridge module. This module tests the…, Tests for backward compatibility exports., Test that __all__ exports are correctly defined., Test that all exported items are accessible., Test that integration functions are exported., Tests for script fallback behavior., Test that generated script is valid bash., Test that script has graceful fallback. (+16 more)

### Community 21 - "TestPerformanceBenchmarks"
Cohesion: 0.24
Nodes (7): benchmark, Performance benchmark tests using pytest-benchmark., Benchmark single SubRefrac calculation., Benchmark multi-material Refrac calculation., Benchmark calculation with large energy sweep., Benchmark calculation with complex formula., TestPerformanceBenchmarks

### Community 22 - "EnergyError"
Cohesion: 0.04
Nodes (51): Integration tests for XRayLabTool Python package. This test module is a…, Test H2O dispersion values., Test H2O reSLD values., Test SubRefrac Silicon properties matching Julia test values., Test edge cases and error handling., Test with empty materials., Test with mismatched array lengths., Test accessing non-existent material. (+43 more)

### Community 23 - "BaseUnitTest"
Cohesion: 0.06
Nodes (27): BaseUnitTest, Base class for unit tests., Setup for each test method., Teardown for each test method., Test CI/CD pipeline integration for style guide enforcement. This module…, Consolidated code quality tests for pyXRayLabTool. This module validates that…, Test that public functions have adequate docstring coverage., Check for naming convention violations. (+19 more)

### Community 24 - "batch_processing.py"
Cohesion: 0.10
Nodes (30): chunk_iterator(), _filter_dataframe_fields(), _get_energy_point_data(), _initialize_progress_bar(), load_batch_input(), _prepare_energies_array(), _prepare_result_data(), process_batch_chunk() (+22 more)

### Community 25 - "main_window.py"
Cohesion: 0.13
Nodes (24): Headless smoke check for the Qt GUI. This script exercises key Single and Multi…, _run_smoke(), skipif, test_compute_multiple_reports_progress_to_100(), test_headless_gui_smoke_runs_offscreen(), _ensure_app(), QApplication, Regression: a CSV export to an unwritable location must report the error… (+16 more)

### Community 26 - "EnvironmentType"
Cohesion: 0.07
Nodes (21): Tests for interfaces/completion_v2/environment.py module. Tests environment…, Tests for environment property detection., Test getting Python executable path., Tests for EnvironmentType constants., Test Python version format detection., Test site-packages directory detection., Test activation script path resolution., Test that environment types are defined. (+13 more)

### Community 27 - "completion_v2/cli.py"
Cohesion: 0.09
Nodes (25): Tests for install_completion_main function., Test that install_completion_main function exists., Test that function accepts argparse.Namespace., TestInstallCompletionMain, completion_main(), create_completion_parser(), handle_completion_command(), install_completion_main() (+17 more)

### Community 28 - "create_scattering_factor_interpolators"
Cohesion: 0.08
Nodes (20): _check_element(), cold_cache(), fixture, ndarray, Characterization tests: PchipInterpolator golden values (P1-C). This file is…, Golden PCHIP samples for Oxygen at [100, 1000, 5000, 10000, 20000] eV., Golden PCHIP samples for Gold at [100, 1000, 5000, 10000, 20000] eV. Heavy…, Loading three elements must return three distinct interpolator pairs. (+12 more)

### Community 29 - "MockXRayResult"
Cohesion: 0.09
Nodes (17): MockXRayResult, Test table formatting of result., Test table format shows range for arrays., Test table format respects precision., Test that default format is table., Test that format type is case-insensitive., Test that unknown format defaults to table., Test formatting with empty fields list. (+9 more)

### Community 30 - "test_backend_dispatch.py"
Cohesion: 0.13
Nodes (12): Unit tests for xraylabtool.backend dispatch layer., TestBackendSwitching, TestInterpolationFactory, TestOpsProxy, get_backend(), _OpsProxy, Proxy that delegates to the active backend with method caching., Check whether the active backend is JAX. (+4 more)

### Community 31 - "TestCICDIntegration"
Cohesion: 0.10
Nodes (16): integration, Test that Ruff linting integrates correctly with CI., Test that MyPy type checking integrates correctly with CI., Test that the test suite integrates correctly with CI., Test that coverage reporting works in CI environment., Test CI/CD pipeline integration for style guide enforcement., Test that JSON reports are generated for CI consumption., Test that tests can run in parallel for faster CI. (+8 more)

### Community 32 - "calculate_single_material_properties"
Cohesion: 0.09
Nodes (14): fixture, Test field filtering with edge cases., Test CLI output formatting edge cases., Test output formatting with extreme precision values., Test formatting of large datasets., TestCLIOutputFormatting, Test Silicon property values from SubRefrac., Test invalid chemical formula. (+6 more)

### Community 33 - "TestEnergySweepSi"
Cohesion: 0.07
Nodes (3): 500-point energy sweep for Si. Lock snapshot values at indices 0, 249, 499 and…, Higher energy → shorter wavelength., TestEnergySweepSi

### Community 34 - "TestFormatCalculationSummary"
Cohesion: 0.10
Nodes (11): Tests for format_calculation_summary function., Test formatting empty results list., Test JSON formatting of single result., Test JSON formatting of multiple results., Test CSV formatting of single result., Test CSV formatting of multiple results., Test table formatting of summary., Test that default format is table. (+3 more)

### Community 35 - "typing_extensions.py"
Cohesion: 0.09
Nodes (26): Test that typing extension validation helpers work correctly., Test type guard functionality for XRayResult arrays., Test NumPy dtype validation for performance-critical arrays., XRayResult dataclass for X-ray optical property calculations., ensure_complex128_array(), ensure_float64_array(), get_optimal_chunk_size(), is_complex_array() (+18 more)

### Community 36 - "normalize_intensity"
Cohesion: 0.12
Nodes (16): Tests for data processing functions., Test smoothing with invalid window size., Test intensity normalization by maximum., Test intensity normalization by area., Test standard score normalization., TestDataProcessing, background_subtraction(), normalize_intensity() (+8 more)

### Community 37 - "_calc"
Cohesion: 0.11
Nodes (15): _calc(), _clear_cache(), fixture, parametrize, Characterization tests for _calculate_molecular_properties (Plan section P2-E).…, Lock the exact float64 values produced by v0.3.0. rtol=1e-3 is tight enough to…, Return (molecular_weight, total_electrons) for *formula*., Ensure a cold-path call for every test. (+7 more)

### Community 38 - "MaterialInputForm"
Cohesion: 0.07
Nodes (13): form(), _get_app(), fixture, QApplication, qt_app(), GUI widget tests for MaterialInputForm (item 4.2). Requires PySide6 — skipped…, QDoubleSpinBox minimum is 0.001 — cannot go below., TestDensityValidation (+5 more)

### Community 39 - "TestBenchmarkComparison"
Cohesion: 0.23
Nodes (7): Benchmark comparison tests for optimization validation., Overall test of optimization effectiveness., Benchmark single element calculation., Benchmark multi-element calculation., Benchmark batch calculation., Benchmark cache access., TestBenchmarkComparison

### Community 40 - "test_utilities.py"
Cohesion: 0.08
Nodes (25): assert_values_close(), clean_cache_around(), expect_no_warnings(), expect_warnings(), memory_test(), performance_test(), Test utilities and helper functions for xraylabtool tests. This module provides…, Decorator to mark and validate performance tests. (+17 more)

### Community 42 - "BasePerformanceTest"
Cohesion: 0.08
Nodes (27): BasePerformanceTest, Base class for performance tests., Teardown for performance tests., performance, Consolidated performance optimization validation tests. This module tests all…, Test the element path pre-computation optimization., Test that _AVAILABLE_ELEMENTS is populated at module load., Test array optimization features. (+19 more)

### Community 43 - "Findings"
Cohesion: 0.08
Nodes (23): Coverage Mapping, Findings, Priority Recommendations (ordered), TEST-001, TEST-002, TEST-003, TEST-004, TEST-005 (+15 more)

### Community 44 - "BaseIntegrationTest"
Cohesion: 0.02
Nodes (60): BaseIntegrationTest, generate_test_materials(), Base test classes and utilities for xraylabtool tests. This module provides…, Base class for integration tests., Setup for each integration test., Teardown for each integration test., Generate random test materials for stress testing., Test table formatting of results. (+52 more)

### Community 45 - "ADR-004: Host-Device Transfer Minimization Strategy"
Cohesion: 0.09
Nodes (22): ADR-004: Host-Device Transfer Minimization Strategy, Appendix: Transfer Cost Reference (CPU Backend), Array Sizes in This Workload, Batch Processing Optimization, Consequences, Context, Decision, Implementation Pattern (+14 more)

### Community 46 - "Utilities Module"
Cohesion: 0.10
Nodes (12): angle_from_q(), bragg_angle(), d_spacing_cubic(), xraylabtool.device, energy_to_wavelength(), get_atomic_data(), load_atomic_data(), xraylabtool.logging_utils (+4 more)

### Community 47 - "AtomicDataProvider"
Cohesion: 0.13
Nodes (14): RealArray, AtomicDataProvider, CalculationEngine, ComplexArray, EnergyArray, OpticalConstantArray, Protocol, Calculate dispersion and absorption coefficients. Returns -------… (+6 more)

### Community 49 - "TestBasicAnalysis"
Cohesion: 0.11
Nodes (15): Tests for basic analysis functions. This module tests the simplified analysis…, Test the basic analysis functionality., Test basic absorption edge detection., Test edge detection with insufficient data., Test edge detection with no edges present., Test basic material comparison., Test material comparison with empty results., Test material comparison with different property. (+7 more)

### Community 50 - "TestTypeSafetyConfig"
Cohesion: 0.09
Nodes (13): performance, Test that NumPy typing support is available., Test that TYPE_CHECKING flag is available for import optimization., Test that modern Python 3.12+ typing features are available., Test that MyPy cache directory can be created and used., Test that type checking infrastructure has minimal performance impact., Test suite for type safety configuration validation., Test that required type stub packages are available. (+5 more)

### Community 51 - "atomic_cache.py"
Cohesion: 0.18
Nodes (14): get_atomic_data_fast(), get_cache_stats(), is_element_preloaded(), MappingProxyType, High-performance atomic data cache system. This module provides a pre-populated…, Fast atomic data lookup with preloaded cache and fallback to Mendeleev. This…, Pre-warm the cache with specific elements. Args: elements: List of element…, Get cache statistics for monitoring. Returns: Dictionary with cache statistics (+6 more)

### Community 52 - "current_palette"
Cohesion: 0.18
Nodes (11): current_palette(), Detect the active palette by checking the application background color., F1F2Plot, MultiF1F2Plot, Any, QWidget, Scattering factor plot widgets backed by PyQtGraph. This module contains small…, Render f1 and f2 vs energy for multiple materials. Parameters ----------… (+3 more)

### Community 53 - "test_docs.py"
Cohesion: 0.19
Nodes (21): check_accessibility(), check_documentation_coverage(), Colors, main(), print_section(), print_status(), Test code examples in RST documentation files., Test code examples in README.md. (+13 more)

### Community 54 - "TestNumericalStabilityChecks"
Cohesion: 0.07
Nodes (16): Tests for numerical stability improvements. This test module validates the…, Test safe critical angle calculation with np.maximum., Test safe attenuation length calculation with epsilon handling., Test handling of zero absorption values., Test numerical stability checks in calculate_derived_quantities., Test NaN detection in dispersion coefficients., Test calculations at boundary conditions., Test energy boundary validation. (+8 more)

### Community 55 - "configure_logging"
Cohesion: 0.13
Nodes (23): Logger, PathLike, Path, test_configure_logging_writes_file(), test_get_logger_child_inherits_config(), GUI entry for XRayLabTool. This package provides a lightweight Qt-based desktop…, build_parser(), main() (+15 more)

### Community 56 - "CalculationWorker"
Cohesion: 0.17
Nodes (16): QRunnable, fixture, qt_app(), Tests for CalculationWorker error and success signal paths., Run worker synchronously (no thread pool needed for unit tests)., _run_worker(), test_worker_does_not_emit_finished_on_exception(), test_worker_emits_error_on_exception() (+8 more)

### Community 57 - "MaterialComparator"
Cohesion: 0.13
Nodes (12): DataFrame, ComparisonResult, MaterialComparator, Any, Material comparison functionality for X-ray properties analysis., Create a pandas DataFrame from comparison results. Args: result:…, Result container for material comparisons., Generate a detailed text report from comparison results. Args: result:… (+4 more)

### Community 58 - "process_single_calculation"
Cohesion: 0.24
Nodes (7): no_progress_config(), fixture, Batch processing partial-failure tests (item 4.3). Verifies that…, BatchConfig with progress bar disabled to keep test output clean., TestProcessSingleCalculation, process_single_calculation(), Process a single X-ray calculation. Args: formula: Chemical formula energies:…

### Community 59 - "TestCalculationSpeedBenchmarks"
Cohesion: 0.22
Nodes (8): performance, Benchmark batch processing performance for multiple materials., Benchmark memory allocation patterns and garbage collection impact., Comprehensive benchmarks for calculation speed optimization., Benchmark scattering factor interpolator creation speed., Benchmark single material calculation speed across energy ranges., Benchmark concurrent calculation performance with threading., TestCalculationSpeedBenchmarks

### Community 60 - "TestCIWorkflowSimulation"
Cohesion: 0.11
Nodes (10): Set up test fixtures., Test complete CI workflow simulation., Set up test fixtures., Test complete CI workflow from start to finish., Test that dependencies can be imported., Test code quality checks., Test that tests can be executed., Test style guide validation. (+2 more)

### Community 62 - "._build_multi_tab"
Cohesion: 0.13
Nodes (10): QLabel, QTableWidget, Any, QScrollArea, QWidget, Avoid overlay scrollbars clipping the scroll area viewport. Some Qt styles…, Lightweight, non-blocking toast overlay., Toast (+2 more)

### Community 63 - "ColorPalette"
Cohesion: 0.18
Nodes (14): QPalette, apply_pyqtgraph_theme(), apply_styles(), apply_theme(), ColorPalette, get_qss(), QApplication, Shared Qt styling for the XRayLabTool GUI. (+6 more)

### Community 64 - "test_formula_parsing.py"
Cohesion: 0.20
Nodes (7): parametrize, Tests for the canonical chemical formula parser and its delegating wrappers.…, Verify number-format handling matches the Julia regex., TestDelegatingWrappers, TestRegexCompatibility, _parse_formula(), Parse chemical formula into elements and quantities. Delegates to the canonical…

### Community 65 - "legacy_install_completion_main"
Cohesion: 0.13
Nodes (14): patch, Test that successful operation returns 0., Test that function delegates to completion_main., Tests for legacy_install_completion_main function., Test that legacy_install_completion_main function exists., Test that function accepts argparse.Namespace., Test that function handles legacy argument formats., Test handling of uninstall mode flag. (+6 more)

### Community 66 - "format_xray_result"
Cohesion: 0.18
Nodes (16): Tests for io/data_export.py module. This module tests data export and…, fixture, _format_as_csv(), _format_as_json(), _format_as_table(), format_calculation_summary(), format_xray_result(), Any (+8 more)

### Community 67 - "core.py"
Cohesion: 0.11
Nodes (26): WavelengthArray, _prepare_element_data(), Core functionality for XRayLabTool. This module is the orchestration layer for…, Prepare element data with interpolators., Validate input formulas and densities., _validate_xray_inputs(), calculate_derived_quantities(), calculate_scattering_factors() (+18 more)

### Community 68 - "calculate_transmission"
Cohesion: 0.16
Nodes (10): exp(-t/Lambda) must be bit-for-bit identical; atol=1e-14., t=0.1 cm (1 mm) through SiO2 at 10 keV., t=0.001 cm (10 um) — thin-film regime., Direct formula check: T = exp(-t/Lambda)., Zero thickness must give T=1 exactly., Very thick sample (100x Lambda) must give T~0., Transmission must decrease strictly with thickness., TestGoldenTransmission (+2 more)

### Community 69 - "arrays_close"
Cohesion: 0.50
Nodes (4): arrays_close(), assert_arrays_close(), Check if arrays are close within tolerance., Assert that arrays are close within tolerance.

### Community 70 - "MemoryProfiler"
Cohesion: 0.14
Nodes (9): MemoryProfiler, Start memory profiling., Stop profiling and record measurement., Context manager for memory profiling., Get average memory increase in MB., Get maximum memory increase in MB., Assert that memory usage is within threshold., Context manager for measuring execution time. (+1 more)

### Community 71 - "PerformanceBenchmark"
Cohesion: 0.12
Nodes (9): PerformanceBenchmark, Utility class for performance benchmarking., Start timing measurement., Stop timing and record measurement., Get average execution time in milliseconds., Get minimum execution time in milliseconds., Get maximum execution time in milliseconds., Reset all measurements. (+1 more)

### Community 72 - "parse_energy_string"
Cohesion: 0.15
Nodes (8): Test parsing single energy value., Test parsing comma-separated energy values., Test parsing linear energy range., Test parsing logarithmic energy range., Test handling of invalid energy strings., parse_energy_string(), ndarray, Parse energy string and return numpy array.

### Community 73 - "CompletionManager"
Cohesion: 0.09
Nodes (21): Test that get_xraylabtool_commands returns a list., Test that commands list is not empty., Tests for completion integration functions., Test that install_completion_main is available., Test that uninstall_completion_main is available., Test that get_xraylabtool_commands is available., TestCompletionFunctions, _generate_bash_completion_script() (+13 more)

### Community 74 - "test_concurrent_cache.py"
Cohesion: 0.11
Nodes (15): clear_lru_caches(), _mock_get_element(), fixture, Tests for thread-safety of lru_cache-decorated functions in utils.py., Clearing the cache while threads are accessing it does not cause crashes., Multiple threads calling get_atomic_weight simultaneously return correct values., Clearing get_atomic_weight cache while threads are active does not crash., LRU cache returns the same cached object for repeated calls. (+7 more)

### Community 75 - "_make_cubic"
Cohesion: 0.05
Nodes (30): Test that _initialize_element_paths works correctly., Tests for CrystalStructure class., Test CrystalStructure initialization., Test adding atoms to crystal structure., Test structure factor calculation., TestCrystalStructure, _make_cubic(), _phase() (+22 more)

### Community 76 - "BaseXRayLabToolTest"
Cohesion: 0.08
Nodes (21): BaseXRayLabToolTest, Any, Validate that result has expected XRayResult structure. Args: result: Result…, Validate result against reference values from Julia implementation. Args:…, Base class for all XRayLabTool tests. Provides common test utilities, assertion…, Assert that operation performance meets threshold requirements. Args:…, Time an operation and return result and execution time. Args: func: Function to…, performance (+13 more)

### Community 77 - "test_file_operations.py"
Cohesion: 0.14
Nodes (23): Tests for xraylabtool.io.file_operations., Energy column (col 0) should be strictly increasing., test_load_absolute_path(), test_load_csv_file(), test_load_csv_space_separated_fallback(), test_load_empty_nff_raises_data_file_error(), test_load_malformed_nff_raises_data_file_error(), test_load_nff_energy_column_monotone() (+15 more)

### Community 78 - ".__init__"
Cohesion: 0.12
Nodes (9): Any, Initialize DataFileError with message and optional filename context., Initialize ValidationError with message and optional parameter context., Initialize AtomicDataError with message and optional element context., Initialize UnknownElementError with element symbol., Initialize BatchProcessingError with message and optional failure context., Initialize CalculationError with message and optional context. Args: message:…, Initialize FormulaError with message and optional formula context. Args:… (+1 more)

### Community 79 - "CI/CD Security Audit Report"
Cohesion: 0.12
Nodes (15): CI/CD Security Audit Report, CICD-001, CICD-002, CICD-003, CICD-004, CICD-005, CICD-006, CICD-007 (+7 more)

### Community 80 - "export_to_json"
Cohesion: 0.08
Nodes (21): Test integration between calculations and basic export functionality., Set up test fixtures with real calculation data., Test CSV export with real calculation results., Test JSON export with real calculation results., Test export functions handle empty results gracefully., TestBasicExportIntegration, Test that exported data maintains scientific accuracy., Test basic export functionality. (+13 more)

### Community 81 - "main"
Cohesion: 0.29
Nodes (14): analyze_mypy_output(), check_core_modules(), check_type_definitions(), main(), Any, Path, Analyze MyPy output to extract metrics and insights. Parameters ----------…, Check core modules with enhanced strict mode. Parameters ----------… (+6 more)

### Community 82 - "calculate_critical_angle"
Cohesion: 0.18
Nodes (9): Two canonical delta values from the test plan., delta=1e-6 -> ~0.08103 degrees., delta=5e-6 -> ~0.18119 degrees., Verify formula: theta = sqrt(2*delta) * (180/pi)., Array of delta values returns array of the same length., Critical angle must increase monotonically with delta., TestGoldenCriticalAngle, calculate_critical_angle() (+1 more)

### Community 83 - "clear_scattering_factor_cache"
Cohesion: 0.15
Nodes (9): _clear_cache(), fixture, Cold-path for every test — prevents cache artifacts from polluting values., Setup for performance tests., Test that cache clearing works properly., Test that optimized code produces same results as before., Test that optimizations provide measurable speedup., clear_scattering_factor_cache() (+1 more)

### Community 84 - "TestCLICompatibility"
Cohesion: 0.25
Nodes (5): Test CLI compatibility and backward compatibility., Test compatibility of different output formats., Test field name compatibility across versions., Test version information consistency., TestCLICompatibility

### Community 86 - "ThemeManager"
Cohesion: 0.16
Nodes (9): QApplication, QObject, Manages application theme state and persistence., Get the active ColorPalette object., Set the active theme mode., Get current theme mode string., Toggle between light and dark modes., Apply current theme to application and global resources. (+1 more)

### Community 87 - "PlotCanvas"
Cohesion: 0.19
Nodes (8): apply_palette_to_widget(), Apply background and foreground colors from palette to a PlotWidget., PlotCanvas, Any, QWidget, PyQtGraph-based plot canvas widget., Re-apply colors from the currently active palette., Single-panel plot widget backed by PyQtGraph.

### Community 88 - "CompletionGenerator"
Cohesion: 0.06
Nodes (25): ABC, CompletionGenerator, FishCompletionGenerator, PowerShellCompletionGenerator, Any, Base class for shell completion generators., Generate command-specific completion logic., Generates native Zsh completion scripts. (+17 more)

### Community 89 - "Backend Abstraction"
Cohesion: 0.24
Nodes (10): Dev Suite Progress, Science Suite Progress, Changelog, XRayLabTool Documentation, Backend Abstraction, set_backend, calculate_single_material_properties, calculate_xray_properties (+2 more)

### Community 91 - "TestGoldenFe2O3"
Cohesion: 0.11
Nodes (5): cold_cache(), Characterization tests: core pipeline golden values (P1-A, P1-B, P1-D). Golden…, Ensure every test runs against the cold code-path (no cached state)., Fe2O3 at [8, 10, 12] keV, density 5.24 g/cm³ — full array assertions., TestGoldenFe2O3

### Community 94 - "TestPhysicalRealism"
Cohesion: 0.14
Nodes (8): Test that results maintain physical realism., Test that critical angles are always positive and physically reasonable.…, Test monotonic decrease in energy regions free of absorption edges. Away from…, Test that a discontinuity exists across the Si K-edge at 1.839 keV. The real…, Test that attenuation length is always positive., Test that electron density is consistent with formula and density., Test that scattering factors are physically reasonable., TestPhysicalRealism

### Community 96 - "TestEnvironmentDetection"
Cohesion: 0.17
Nodes (8): dict, Tests for environment detection functionality., Test detecting current Python environment., Test markers for venv detection., Test markers for conda environment detection., Test detection of virtual environment via env var., Test detection of conda environment via env var., TestEnvironmentDetection

### Community 97 - "AtomicScatteringFactor"
Cohesion: 0.10
Nodes (16): Tests for AtomicScatteringFactor class., Test AtomicScatteringFactor initialization., Test scattering factor calculation., TestAtomicScatteringFactor, AtomicScatteringFactor, load_data_file(), load_scattering_factor_data(), Any (+8 more)

### Community 98 - "TestExtremeConditions"
Cohesion: 0.14
Nodes (8): Test calculations under extreme conditions., Test calculation at very low energies (near the lower bound)., Test calculation at very high energies (near the upper bound)., Test calculation with very low density materials., Test calculation with very high density materials., Test calculation with large energy arrays., Test calculation with logarithmic energy spacing., TestExtremeConditions

### Community 101 - "ADR-003: float64 Preservation Policy"
Cohesion: 0.17
Nodes (11): ADR-003: float64 Preservation Policy, Appendix: Numerical Precision Requirements by Function, Consequences, Context, Current Codebase float64 Usage, Decision, Enforcement Strategy, JAX float32 Default (+3 more)

### Community 102 - "ParametrizedTestMixin"
Cohesion: 0.18
Nodes (9): param, generate_test_energies(), ParametrizedTestMixin, ndarray, Mixin providing utilities for parametrized tests., Create pytest parameters for material testing., Create pytest parameters for energy testing., Generate test energy arrays. (+1 more)

### Community 103 - "OverlayScrollbarMarginHelper"
Cohesion: 0.24
Nodes (6): QEvent, OverlayScrollbarMarginHelper, QObject, QScrollArea, Overlay scrollbar margin helper for QScrollArea widgets., Adjusts viewport margins so overlay scrollbars don't cover content. Install on…

### Community 104 - "`test_docs.py` - Documentation Testing"
Cohesion: 0.20
Nodes (11): Available Scripts, Example Output:, Integration:, Maintenance Notes, Output:, Requirements:, `run_type_check.py` - Type Checking, Scripts Directory (+3 more)

### Community 105 - "TestUnitConversionConstants"
Cohesion: 0.05
Nodes (17): parametrize, Characterization tests for physical constants and unit conversions (Plan P2-F).…, Lock energy_to_wavelength_angstrom() at specific energies to atol=1e-8 Å. This…, Round-trip: wavelength_angstrom_to_energy(energy_to_wavelength_angstrom(E)) ≈ E., Assert float64 is preserved through constant arithmetic., Lock the hardcoded float literals in constants.py., Lock derived constants to their exact captured float64 values., SCATTERING_FACTOR must equal THOMPSON * AVOGADRO * 1e6 / (2π). (+9 more)

### Community 106 - "ndarray"
Cohesion: 0.15
Nodes (12): create_batch_test_data(), ndarray, Factory for generating test data., Get list of test materials., Get list of material formulas., Get list of material densities., Generate linear energy array., Generate logarithmic energy array. (+4 more)

### Community 107 - "TestComplexFormula"
Cohesion: 0.40
Nodes (3): Test parenthesis-containing formula through the molecular properties path., Ca5(PO4)3OH: verify MW matches textbook value (~502.31 g/mol)., TestComplexFormula

### Community 110 - "analyze_element_associations"
Cohesion: 0.39
Nodes (3): TestAnalyzeElementAssociations, analyze_element_associations(), Analyze which elements are frequently associated together. Args: compound_list:…

### Community 111 - "apply_palette_to_item"
Cohesion: 0.40
Nodes (4): apply_palette_to_item(), Any, Apply background/foreground colors from palette to a PlotItem., Re-apply colors from the currently active palette.

### Community 112 - "_handle_mendeleev_error"
Cohesion: 0.50
Nodes (4): NoReturn, _handle_mendeleev_error(), Exception, Handle errors from mendeleev package.

### Community 113 - "xraylabtool — Unified Audit & Remediation Plan"
Cohesion: 0.18
Nodes (10): Cross-Referenced Root Causes, Executive Summary, Findings Cross-Reference Index, Prioritized Remediation Plan, Tier 1 — Immediate (Week 1) — Correctness & Quick Security Wins, Tier 2 — Short-term (Weeks 2-3) — Security Hardening & CI, Tier 3 — Medium-term (Weeks 4-6) — Architecture & Coverage, Tier 4 — Long-term (Backlog) — Polish & Hardening (+2 more)

### Community 114 - ".get_scattering_factors"
Cohesion: 0.50
Nodes (3): ComplexArray, EnergyArray, Get atomic scattering factors for element at given energies. This method loads…

### Community 115 - "test_golden_derived_quantities.py"
Cohesion: 0.20
Nodes (7): fixture, Characterization tests: derived_quantities.py module (P2-A through P2-D). The…, Verify exact formula: 2*pi * delta / wl^2 (wl in Angstroms)., calculate_scattering_length_density(), ndarray, Derived quantities calculator for X-ray optical properties. This module…, Calculate real and imaginary scattering length densities. Args:…

### Community 116 - "calculate_attenuation_length"
Cohesion: 0.25
Nodes (6): Known wavelength/beta inputs -> known output, locked at rtol=1e-10., Verify Λ = lambda_m / (4*pi*beta) * 100., More absorbing material -> shorter attenuation length., TestGoldenAttenuationLength, calculate_attenuation_length(), Calculate X-ray attenuation length (1/e length). Args: wavelength_angstrom:…

### Community 117 - "Shell Completion Guide"
Cohesion: 0.67
Nodes (3): CLI Reference, Shell Completion Guide, Shell Completion v2

### Community 118 - "TestCompletionScriptGeneration"
Cohesion: 0.18
Nodes (4): Tests that completion scripts are generated correctly., Generated bash script paths use shlex.quote-safe quoting., Installer uses shlex.quote when writing activation scripts., TestCompletionScriptGeneration

### Community 119 - "uninstall_completion_main"
Cohesion: 0.22
Nodes (8): Tests for uninstall_completion_main function., Test that uninstall_completion_main function exists., Test that function accepts argparse.Namespace., Test that function delegates to completion_main., TestUninstallCompletionMain, Any, Legacy uninstall completion main function., uninstall_completion_main()

### Community 121 - "ADR-001: JAX vs NumPy Computation Backend"
Cohesion: 0.20
Nodes (9): ADR-001: JAX vs NumPy Computation Backend, Appendix: Functions that MUST NOT be JIT-compiled, Appendix: Functions to JIT-Compile (Priority Order), Consequences, Context, Decision, Negative, Positive (+1 more)

### Community 122 - "TestGoldenScatteringLengthDensity"
Cohesion: 0.20
Nodes (4): Scalar inputs -> known SLD, locked at atol=1e-12., real_sld = 2*pi*delta/(...) is positive for positive delta., im_sld = 2*pi*beta/(...) is always positive for positive beta., TestGoldenScatteringLengthDensity

### Community 123 - "BashCompletionGenerator"
Cohesion: 0.08
Nodes (17): Tests for interfaces/completion_v2/integration.py module. Tests integration…, Tests for backward compatibility with existing CLI., Test handling of various legacy argument formats., Test that optional arguments are handled., Tests for integration flow between new and legacy systems., Test that new completion system imports correctly., Test that CompletionInstaller is available., Test that circular imports are avoided. (+9 more)

### Community 124 - "TestBatchConfig"
Cohesion: 0.20
Nodes (6): Test BatchConfig optimizations., Test default BatchConfig initialization., Test automatic memory limit adjustment based on system memory., Test worker count calculation., Test custom configuration parameters., TestBatchConfig

### Community 125 - "get_compound_family"
Cohesion: 0.33
Nodes (3): TestGetCompoundFamily, get_compound_family(), Determine which compound family a formula belongs to. Args: formula: Chemical…

### Community 126 - "parse_chemical_formula"
Cohesion: 0.33
Nodes (3): TestParseChemicalFormula, parse_chemical_formula(), Parse a chemical formula into element counts. Delegates to the canonical…

### Community 128 - "cache.py"
Cohesion: 0.13
Nodes (16): Test basic bulk atomic data loading., Test that bulk atomic data loading uses caching effectively., get_bulk_atomic_data(), get_cached_elements(), is_element_cached(), MappingProxyType, Cache infrastructure for scattering factor data and interpolators., Smart cache warming that only loads elements needed for the specific… (+8 more)

### Community 129 - "TableFormatter"
Cohesion: 0.21
Nodes (7): Any, Shared table-cell formatting for X-ray result rows., Formats X-ray calculation results into display-ready cell strings., Format one energy point of a single-material result. Returns a list of strings…, Format one energy point of a multi-material result. Returns a list of strings…, Format the single-material summary row. Returns: [formula, molecular_weight,…, TableFormatter

### Community 132 - "ADR-002: PyQtGraph vs Matplotlib for GUI Plotting"
Cohesion: 0.22
Nodes (8): ADR-002: PyQtGraph vs Matplotlib for GUI Plotting, Appendix: Feature Parity Checklist, Consequences, Context, Decision, Migration Details, Negative, Positive

### Community 133 - "TestMultiMaterialEquivalence"
Cohesion: 0.28
Nodes (3): parametrize, calculate_multiple_xray_properties with 3 materials must return values…, TestMultiMaterialEquivalence

### Community 134 - "TestBatchProcessingMemoryManagement"
Cohesion: 0.22
Nodes (6): skipif, Test memory management in batch processing., Test memory cleanup in single calculations., Test cache clearing integration with memory management., Test memory efficiency with large batches., TestBatchProcessingMemoryManagement

### Community 135 - "RegressionTracker"
Cohesion: 0.28
Nodes (5): Track performance regressions across test runs., Load performance baselines from file., Save performance baselines to file., Check for performance regression., RegressionTracker

### Community 138 - "Files"
Cohesion: 0.25
Nodes (7): Files, Maintenance, `performance_baselines.json`, `sample_materials.json`, Test Data Directory, `test_energies.json`, Usage in Tests

### Community 141 - "find_similar_compounds"
Cohesion: 0.25
Nodes (6): TestFindSimilarCompounds, Any, Intelligently warm cache for compounds and their related elements. This…, warm_cache_for_compounds(), find_similar_compounds(), Find compounds with similar element composition. Args: formula: Target chemical…

### Community 142 - "get_compound_complexity_score"
Cohesion: 0.39
Nodes (3): TestGetCompoundComplexityScore, get_compound_complexity_score(), Calculate a complexity score for a compound (more complex = higher score).…

### Community 143 - "get_compound_frequency_score"
Cohesion: 0.39
Nodes (3): TestGetCompoundFrequencyScore, get_compound_frequency_score(), Get a frequency score for a compound based on its commonality in materials…

### Community 144 - "get_elements_for_compound"
Cohesion: 0.39
Nodes (3): TestGetElementsForCompound, get_elements_for_compound(), Get the list of elements for a given compound formula. Args: formula: Chemical…

### Community 147 - "FastXRayCalculationEngine"
Cohesion: 0.15
Nodes (11): FastXRayCalculationEngine, get_calculation_engine(), High-performance X-ray calculation engine implementing CalculationEngine…, Initialize the calculation engine., Pre-warm caches for improved performance., Get the global calculation engine instance., FastAtomicDataProvider, get_atomic_data_provider() (+3 more)

### Community 148 - "PerformanceMetrics"
Cohesion: 0.25
Nodes (5): PerformanceMetrics, Protocol for performance measurement and validation., Get current calculation rate in calculations/second., Get current memory usage in MB., Validate that performance meets target calculations/second.

### Community 150 - "ResultValidator"
Cohesion: 0.15
Nodes (10): get_system_info(), log_system_info(), Any, Utility class for validating calculation results., Validate XRayResult object., Validate batch calculation results., Validate numerical stability of calculated values., Get system information for test context. (+2 more)

### Community 152 - "performance/conftest.py"
Cohesion: 0.29
Nodes (6): ci_threshold_multiplier(), fixture, pytest_collection_modifyitems(), Performance test configuration. Skips all performance tests unless --run-…, Skip performance tests unless --run-performance is passed., Return a multiplier for wall-clock thresholds (3x in CI, 1x locally).

### Community 153 - "test_compound_analysis.py"
Cohesion: 0.27
Nodes (5): Tests for xraylabtool.data_handling.compound_analysis., TestGetRecommendedElementsForWarming, get_recommended_elements_for_warming(), Element combination analysis for intelligent cache warming. This module…, Get recommended elements for cache warming based on recent compound usage.…

### Community 157 - "test_core.py"
Cohesion: 0.33
Nodes (4): Tests for the core module., Tests for load_data_file function., Test loading nonexistent file raises error., TestLoadDataFile

### Community 159 - "XRayResult"
Cohesion: 0.06
Nodes (17): _single(), Test XRayResult array conversion optimization., Test that XRayResult properly converts input arrays to numpy arrays., Test that XRayResult arrays maintain consistent dtypes for performance., Test that XRayResult can be used in protocol-based contexts., Test that XRayResult arrays support standard array operations., Test XRayResult creation with properly typed inputs., _process_formulas_parallel() (+9 more)

### Community 163 - "Complex refractive index n=1-delta-i*beta"
Cohesion: 0.83
Nodes (4): Complex refractive index n=1-delta-i*beta, Critical angle formula theta_c=sqrt(2*delta), Linear absorption coefficient mu=4pi*beta/lambda, Calculators Module

### Community 164 - "Testing Guide"
Cohesion: 0.50
Nodes (4): Audit Remediation Plan, Contributing Guide, Testing Guide, Scripts README

### Community 165 - "conf.py"
Cohesion: 0.50
Nodes (3): Configuration file for the Sphinx documentation builder. For the full list of…, Set up the Sphinx application configuration., setup()

### Community 167 - "_auto_select_backend"
Cohesion: 0.50
Nodes (4): _auto_select_backend(), _has_nvidia_gpu(), Fast check for NVIDIA GPU without importing JAX (~1ms)., Select the best available backend: JAX GPU > JAX CPU > NumPy. JAX is preferred…

### Community 168 - "X-ray Calculations Guide"
Cohesion: 0.67
Nodes (3): Atomic Data Reference, X-ray Calculations Guide, X-ray Optics Guide

### Community 169 - "pyXRayLabTool Logo (Main SVG)"
Cohesion: 0.67
Nodes (3): pyXRayLabTool Logo (Dark), pyXRayLabTool Logo (Light JPG), pyXRayLabTool Logo (Main SVG)

## Knowledge Gaps
- **116 isolated node(s):** `Colors`, `Colors`, `Context`, `Decision`, `Positive` (+111 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **35 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CompletionInstaller` connect `CompletionInstaller` to `legacy_install_completion_main`, `interfaces/cli.py`, `EnvironmentDetector`, `CompletionManager`, `BashCompletionGenerator`, `patch`, `EnvironmentInfo`, `TestCompletionScriptGeneration`, `uninstall_completion_main`, `EnvironmentType`, `completion_v2/cli.py`?**
  _High betweenness centrality (0.166) - this node is a cross-community bridge._
- **Why does `calculate_single_material_properties()` connect `calculate_single_material_properties` to `cache.py`, `xraylabtool/__init__.py`, `interfaces/cli.py`, `timer`, `test_cli_edge_cases.py`, `_calculate_single_material_xray_properties`, `BatchConfig`, `EnergyError`, `batch_processing.py`, `main_window.py`, `XRayResult`, `TestBenchmarkComparison`, `BasePerformanceTest`, `TestBasicAnalysis`, `process_single_calculation`, `TestCalculationSpeedBenchmarks`, `core.py`, `export_to_json`, `clear_scattering_factor_cache`, `TestCLICompatibility`, `TestGoldenFe2O3`?**
  _High betweenness centrality (0.158) - this node is a cross-community bridge._
- **Why does `parse_formula()` connect `parse_formula` to `test_formula_parsing.py`, `cache.py`, `xraylabtool/__init__.py`, `TestParseFormulaErrors`, `TestParseFormulaParentheses`, `_calc`, `core.py`, `MaterialInputForm`, `interfaces/cli.py`, `FormulaError`, `_calculate_single_material_xray_properties`, `validate_chemical_formula`, `main_window.py`, `test_compound_analysis.py`, `parse_chemical_formula`, `TestParseFormulaDecimal`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `CompletionInstaller` (e.g. with `.test_completion_installer_methods()` and `.test_install_completion_help()`) actually correct?**
  _`CompletionInstaller` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `MainWindow` (e.g. with `EnergyConfig` and `TableFormatter`) actually correct?**
  _`MainWindow` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 32 inferred relationships involving `FormulaError` (e.g. with `TestBasicSetupAndInitialization` and `TestDuplicatedFormulasIndependence`) actually correct?**
  _`FormulaError` has 32 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Colors`, `Colors`, `Context` to the rest of the system?**
  _116 weakly-connected nodes found - possible documentation gaps or missing edges._
