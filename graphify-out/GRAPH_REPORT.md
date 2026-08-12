# Graph Report - pyXRayLabTool  (2026-08-12)

## Corpus Check
- 126 files · ~123,051 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 3490 nodes · 5893 edges · 173 communities (137 shown, 36 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 431 edges (avg confidence: 0.6)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `87dc3a1e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- EnergyError
- interfaces/cli.py
- BaseIntegrationTest
- CompletionInstaller
- test_compound_analysis.py
- _make_cubic
- Any
- EnvironmentDetector
- MockXRayResult
- StyleGuideValidator
- xraylabtool/__init__.py
- calculate_single_material_properties
- EnvironmentInfo
- TestBashCompletionScript
- BasePerformanceTest
- create_scattering_factor_interpolators
- test_cli_edge_cases.py
- device.py
- MaterialInputForm
- TestInstallCompletionCommand
- installer.py
- MaterialComparator
- test_integration.py
- MainWindow
- timer
- core.py
- configure_logging
- BatchConfig
- legacy_install_completion_main
- main_window.py
- atomic_cache.py
- test_backend_dispatch.py
- Utilities Module
- ._build_multi_tab
- AtomicScatteringFactor
- batch_processing.py
- TestCICDIntegration
- TestEnergySweepSi
- typing_extensions.py
- TestBackwardCompatibility
- XRayResult
- Complex refractive index n=1-delta-i*beta
- validate_density
- test_utilities.py
- validate_chemical_formula
- _calc
- test_concurrent_cache.py
- TestPhysicalRealism
- TestFullPipelineSiO2
- test_docs.py
- uninstall_completion_main
- calculate_scattering_factors
- test_file_operations.py
- TestTypeSafetyConfig
- TestValidateEnergyRange
- ndarray
- BaseXRayLabToolTest
- TestNumpyBackend
- current_palette
- ColorPalette
- TestBasicExportFunctionality
- TestNumericalStabilityChecks
- TestCIWorkflowSimulation
- BaseUnitTest
- parse_formula
- .__init__
- calculate_transmission
- ndarray
- MemoryProfiler
- PerformanceBenchmark
- calculate_critical_angle
- test_golden_pipeline.py
- JaxBackend
- ThemeManager
- PlotCanvas
- TestGoldenSi
- TestGoldenAu
- TestGoldenFe2O3
- InterpolatorProtocol
- TestGoldenSiO2
- TestInputValidationAndErrorHandling
- test_memory_management.py
- TestBasicAnalysis
- test_numerical_stability.py
- TestValidateCalculationParameters
- TestEnvironmentDetection
- ParametrizedTestMixin
- CalculationWorker
- OverlayScrollbarMarginHelper
- main
- TestCalculationSpeedBenchmarks
- normalize_intensity
- cache.py
- TestUnitConversionConstants
- TestDataFactory
- TestBenchmarkComparison
- NumpyBackend
- GUI Module
- TableFormatter
- TestPerformanceBenchmarks
- Findings
- constants.py
- test_golden_derived_quantities.py
- calculate_attenuation_length
- test_formula_parsing.py
- TestGoldenScatteringLengthDensity
- TestCompletionIntegrationFlow
- TestBatchConfig
- FastXRayCalculationEngine
- .uninstall
- TestMultiMaterialEquivalence
- RegressionTracker
- find_absorption_edges
- TestDerivedConstants
- CI/CD Security Audit Report
- `test_docs.py` - Documentation Testing
- ADR-004: Host-Device Transfer Minimization Strategy
- ADR-003: float64 Preservation Policy
- Documentation Root
- TestCLIOutputFormatting
- TestCLICompatibility
- API Reference Index
- TestEdgeCasesAndErrorHandling
- TestMemoryLeakPrevention
- PerformanceMetrics
- TestFundamentalConstants
- get_system_info
- TestH2OProperties
- performance/conftest.py
- tests/conftest.py
- TestParseFormulaDecimal
- TestParseFormulaErrors
- TestXRayResultProtocolCompliance
- TestParseFormulaParentheses
- TestDtypePreservation
- TestComplexFormula
- xraylabtool — Unified Audit & Remediation Plan
- apply_palette_to_item
- _handle_mendeleev_error
- TestMathematicalConstants
- _auto_select_backend
- characterization/__init__.py
- fixtures/__init__.py
- integration/__init__.py
- performance/__init__.py
- BashCompletionGenerator
- ADR-001: JAX vs NumPy Computation Backend
- unit/__init__.py
- TestSiO2Properties
- .Attenuation_Length
- ADR-002: PyQtGraph vs Matplotlib for GUI Plotting
- MemoryMonitor
- Testing Guide
- conf.py
- .imSLD
- pyXRayLabTool Logo (Main SVG)
- GitHub Token Direct Push Risk (CICD-004)
- Test Coverage Audit
- Files
- PerformanceTimer
- tests/__init__.py
- pyXRayLabTool Favicon
- GUI Main Window Screenshot
- BaseIntegrationTest
- BasePerformanceTest
- BaseUnitTest
- BaseXRayLabToolTest
- Test Data README
- Changelog
- Rollback Procedures

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
- `TestFullPipelineSiO2` --uses--> `BatchConfig`  [INFERRED]
  tests/characterization/test_golden_integration.py → xraylabtool/data_handling/batch_processing.py
- `TestMultiMaterialEquivalence` --uses--> `BatchConfig`  [INFERRED]
  tests/characterization/test_golden_integration.py → xraylabtool/data_handling/batch_processing.py
- `TestBatchProcessorEquivalence` --uses--> `BatchConfig`  [INFERRED]
  tests/characterization/test_golden_integration.py → xraylabtool/data_handling/batch_processing.py
- `TestEnergySweepSi` --uses--> `BatchConfig`  [INFERRED]
  tests/characterization/test_golden_integration.py → xraylabtool/data_handling/batch_processing.py
- `cold_cache()` --calls--> `clear_scattering_factor_cache()`  [INFERRED]
  tests/characterization/test_golden_interpolation.py → xraylabtool/calculators/cache.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **JAX Architecture Decision Records** — docs_architecture_001_adr_jax_vs_numpy_jax_decision, docs_architecture_003_adr_dtype_policy_dtype_strategy, architecture_004_adr_host_device_transfers_transfer_policy, docs_architecture_jax_architecture_jax_design [INFERRED 0.85]
- **Physics Documentation Cluster** — docs_physics_atomic_data_atomic_data_guide, docs_physics_calculations_calculations_guide, docs_physics_xray_optics_xray_optics_guide [INFERRED 0.90]

## Communities (173 total, 36 thin omitted)

### Community 0 - "EnergyError"
Cohesion: 0.04
Nodes (79): Tests for custom exception classes. This module tests the behavior of domain-…, Test EnergyError class., Test basic error message., Test error with energy value., Test error with valid range information., Test error with both energy and range., Test DataFileError class., Test basic error message. (+71 more)

### Community 1 - "interfaces/cli.py"
Cohesion: 0.04
Nodes (109): add_atomic_command(), add_batch_command(), add_bragg_command(), add_calc_command(), add_compare_command(), add_completion_command(), add_convert_command(), add_formula_command() (+101 more)

### Community 2 - "BaseIntegrationTest"
Cohesion: 0.03
Nodes (56): BaseIntegrationTest, Base class for integration tests., Setup for each integration test., Teardown for each integration test., Test table formatting of results., Test formatting with custom field selection., Test the 'calc' command functionality., Test basic calculation command. (+48 more)

### Community 3 - "CompletionInstaller"
Cohesion: 0.04
Nodes (38): Integration tests for CompletionInstaller (item 4.1)., Calling _modify_activation_script twice does not duplicate blocks., Uninstall strips the sentinel-delimited block but leaves surrounding content., Tests that install/uninstall work with real temp directories., Tests that completion scripts are generated correctly., Path written into activate script is properly shell-quoted., Generated bash script paths use shlex.quote-safe quoting., Installer uses shlex.quote when writing activation scripts. (+30 more)

### Community 4 - "test_compound_analysis.py"
Cohesion: 0.05
Nodes (26): Tests for xraylabtool.data_handling.compound_analysis., TestAnalyzeElementAssociations, TestFindSimilarCompounds, TestGetCompoundComplexityScore, TestGetCompoundFamily, TestGetCompoundFrequencyScore, TestGetElementsForCompound, TestGetRecommendedElementsForWarming (+18 more)

### Community 5 - "_make_cubic"
Cohesion: 0.05
Nodes (30): Test that _initialize_element_paths works correctly., Tests for CrystalStructure class., Test CrystalStructure initialization., Test adding atoms to crystal structure., Test structure factor calculation., TestCrystalStructure, _make_cubic(), _phase() (+22 more)

### Community 6 - "Any"
Cohesion: 0.10
Nodes (4): Array, ArrayBackend, Any, Protocol

### Community 7 - "EnvironmentDetector"
Cohesion: 0.07
Nodes (27): EnvironmentDetector, Any, Path, Discover all available Python environments., Detect the type of the currently active environment., Check if the current environment is using mamba., Detect whether a conda environment is actually mamba-managed., Get the path of the current environment. (+19 more)

### Community 8 - "MockXRayResult"
Cohesion: 0.05
Nodes (28): MockXRayResult, Test table formatting of result., Test table format shows range for arrays., Test table format respects precision., Test that default format is table., Test that format type is case-insensitive., Test that unknown format defaults to table., Test formatting with empty fields list. (+20 more)

### Community 9 - "StyleGuideValidator"
Cohesion: 0.07
Nodes (30): Colors, main(), Path, Get all Python files in the project., Validate import patterns according to style guide., Validate naming conventions according to style guide., Validate type hint usage according to style guide., Validate docstring patterns according to style guide. (+22 more)

### Community 10 - "xraylabtool/__init__.py"
Cohesion: 0.03
Nodes (96): Test integration between calculations and basic export functionality., Test CSV export with real calculation results., Test JSON export with real calculation results., Test export functions handle empty results gracefully., TestBasicExportIntegration, Tests for atomic data lookup functions with LRU caching., Test that element symbols are case-sensitive., Test atomic data lookup with caching. (+88 more)

### Community 11 - "calculate_single_material_properties"
Cohesion: 0.06
Nodes (25): _clear_cache(), fixture, Characterization tests for end-to-end integration paths (Plan section 3).…, calculate_batch_properties parallel result must match single-material calls to…, Cold-path for every test — prevents cache artifacts from polluting values., _single(), TestBatchProcessorEquivalence, cold_cache() (+17 more)

### Community 12 - "EnvironmentInfo"
Cohesion: 0.07
Nodes (23): Test EnvironmentInfo with Python version., Test EnvironmentInfo equality comparison., Test EnvironmentInfo string representation., Tests for EnvironmentInfo class., Test creating EnvironmentInfo instance., Test that EnvironmentInfo has required attributes., Test EnvironmentInfo with specific type., Test EnvironmentInfo with environment path. (+15 more)

### Community 13 - "TestBashCompletionScript"
Cohesion: 0.05
Nodes (24): Tests for interfaces/completion.py bridge module. This module tests the…, Tests for backward compatibility exports., Test that __all__ exports are correctly defined., Test that all exported items are accessible., Test that integration functions are exported., Tests for script fallback behavior., Test that generated script is valid bash., Test that script has graceful fallback. (+16 more)

### Community 14 - "BasePerformanceTest"
Cohesion: 0.07
Nodes (29): BasePerformanceTest, Base class for performance tests., Teardown for performance tests., performance, Consolidated performance optimization validation tests. This module tests all…, Test the element path pre-computation optimization., Test that _AVAILABLE_ELEMENTS is populated at module load., Test array optimization features. (+21 more)

### Community 15 - "create_scattering_factor_interpolators"
Cohesion: 0.07
Nodes (23): _check_element(), cold_cache(), fixture, ndarray, Characterization tests: PchipInterpolator golden values (P1-C). This file is…, Golden PCHIP samples for Oxygen at [100, 1000, 5000, 10000, 20000] eV., Golden PCHIP samples for Gold at [100, 1000, 5000, 10000, 20000] eV. Heavy…, Loading three elements must return three distinct interpolator pairs. (+15 more)

### Community 16 - "test_cli_edge_cases.py"
Cohesion: 0.06
Nodes (24): Path, Edge case tests for XRayLabTool CLI. This module tests boundary conditions,…, Test CLI file operations edge cases., Test file operations with various path conditions., Test batch processing file edge cases., Test handling of malformed batch input files., Test CLI input validation edge cases., Test CLI error handling edge cases. (+16 more)

### Community 17 - "device.py"
Cohesion: 0.05
Nodes (31): Tests for GPU/device detection in xraylabtool/device.py., Parses CUDA version string from nvcc --version output., Returns (None, None) when stdout contains no recognizable release line., Returns False gracefully when no GPU hardware is detected., Returns False (not an exception) when JAX is not importable., Returns (None, None) gracefully when nvidia-smi is not on PATH., check_gpu_availability always returns a bool., get_device_info returns a dict containing all required keys. (+23 more)

### Community 18 - "MaterialInputForm"
Cohesion: 0.07
Nodes (13): form(), _get_app(), fixture, QApplication, qt_app(), GUI widget tests for MaterialInputForm (item 4.2). Requires PySide6 — skipped…, QDoubleSpinBox minimum is 0.001 — cannot go below., TestDensityValidation (+5 more)

### Community 19 - "TestInstallCompletionCommand"
Cohesion: 0.06
Nodes (20): patch, Test main function with no arguments., Test main function with help argument., Test main function with version argument., Test main function with invalid command., Test that all expected commands are available in CLI., Test full calculation workflow through CLI., Test full batch processing workflow. (+12 more)

### Community 20 - "installer.py"
Cohesion: 0.07
Nodes (22): Tests for interfaces/completion_v2/environment.py module. Tests environment…, Tests for environment property detection., Test getting Python executable path., Tests for EnvironmentType constants., Test Python version format detection., Test site-packages directory detection., Test activation script path resolution., Test that environment types are defined. (+14 more)

### Community 21 - "MaterialComparator"
Cohesion: 0.13
Nodes (12): DataFrame, ComparisonResult, MaterialComparator, Any, Material comparison functionality for X-ray properties analysis., Create a pandas DataFrame from comparison results. Args: result:…, Result container for material comparisons., Generate a detailed text report from comparison results. Args: result:… (+4 more)

### Community 22 - "test_integration.py"
Cohesion: 0.07
Nodes (21): generate_test_materials(), Base test classes and utilities for xraylabtool tests. This module provides…, Generate random test materials for stress testing., Centralized test configuration and constants. This module provides centralized…, Integration tests for XRayLabTool Python package. This test module is a…, Test SubRefrac Silicon properties matching Julia test values., Test property consistency across materials., Test that all materials have the same energy array length. (+13 more)

### Community 23 - "MainWindow"
Cohesion: 0.09
Nodes (8): QCloseEvent, QMainWindow, MainWindow, Any, Export a PyQtGraph plot widget to a PNG file. The plot widgets wrap either a…, Force update of all plots to match new theme., Drop queued-but-unstarted compute jobs so the app shuts down promptly instead…, Replace characters unsafe for filenames with underscores.

### Community 24 - "timer"
Cohesion: 0.08
Nodes (22): performance, Performance benchmark tests for optimization validation. This test module…, Benchmark batch calculation performance., Benchmark single vs multi-element performance difference., Benchmark performance with large energy arrays., Benchmark overall throughput (calculations per second)., Performance regression tests., Test that array conversion optimization reduces overhead. (+14 more)

### Community 25 - "core.py"
Cohesion: 0.11
Nodes (31): calculate_multiple_xray_properties(), _calculate_single_material_xray_properties(), calculate_xray_properties(), _convert_energy_input(), _create_process_formula_function(), _prepare_element_data(), _process_formulas_parallel(), Any (+23 more)

### Community 26 - "configure_logging"
Cohesion: 0.13
Nodes (23): Logger, PathLike, Path, test_configure_logging_writes_file(), test_get_logger_child_inherits_config(), GUI entry for XRayLabTool. This package provides a lightweight Qt-based desktop…, build_parser(), main() (+15 more)

### Community 27 - "BatchConfig"
Cohesion: 0.12
Nodes (18): skipif, Test memory monitoring during batch processing., Test memory efficiency with large batches., no_progress_config(), fixture, Batch processing partial-failure tests (item 4.3). Verifies that…, The result dict has an entry for every formula submitted., BatchConfig with progress bar disabled to keep test output clean. (+10 more)

### Community 28 - "legacy_install_completion_main"
Cohesion: 0.06
Nodes (40): patch, Tests for interfaces/completion_v2/integration.py module. Tests integration…, Test that successful operation returns 0., Tests for install_completion_main function., Test that install_completion_main function exists., Test that function accepts argparse.Namespace., Test that function delegates to completion_main., Tests for legacy_install_completion_main function. (+32 more)

### Community 29 - "main_window.py"
Cohesion: 0.13
Nodes (24): Headless smoke check for the Qt GUI. This script exercises key Single and Multi…, _run_smoke(), skipif, test_compute_multiple_reports_progress_to_100(), test_headless_gui_smoke_runs_offscreen(), _ensure_app(), QApplication, Regression: a CSV export to an unwritable location must report the error… (+16 more)

### Community 30 - "atomic_cache.py"
Cohesion: 0.11
Nodes (26): FastAtomicDataProvider, get_atomic_data_fast(), get_atomic_data_provider(), get_bulk_atomic_data_fast(), get_cache_stats(), is_element_preloaded(), Any, MappingProxyType (+18 more)

### Community 31 - "test_backend_dispatch.py"
Cohesion: 0.13
Nodes (12): Unit tests for xraylabtool.backend dispatch layer., TestBackendSwitching, TestInterpolationFactory, TestOpsProxy, get_backend(), _OpsProxy, Proxy that delegates to the active backend with method caching., Check whether the active backend is JAX. (+4 more)

### Community 32 - "Utilities Module"
Cohesion: 0.07
Nodes (21): xraylabtool.interfaces.completion (legacy), completion_v2 (venv-centric completion), completion_v2.cache, completion_v2.cli, completion_v2.environment, completion_v2.installer, completion_v2.integration, completion_v2.shells (+13 more)

### Community 33 - "._build_multi_tab"
Cohesion: 0.20
Nodes (9): QLabel, QTableWidget, QScrollArea, QWidget, Avoid overlay scrollbars clipping the scroll area viewport. Some Qt styles…, Lightweight, non-blocking toast overlay., Toast, MaterialTable (+1 more)

### Community 34 - "AtomicScatteringFactor"
Cohesion: 0.08
Nodes (20): Tests for the core module., Tests for AtomicScatteringFactor class., Test AtomicScatteringFactor initialization., Test scattering factor calculation., Tests for load_data_file function., Test loading nonexistent file raises error., TestAtomicScatteringFactor, TestLoadDataFile (+12 more)

### Community 35 - "batch_processing.py"
Cohesion: 0.11
Nodes (26): chunk_iterator(), _filter_dataframe_fields(), _get_energy_point_data(), _prepare_energies_array(), _prepare_result_data(), process_batch_chunk(), _process_chunks(), Any (+18 more)

### Community 36 - "TestCICDIntegration"
Cohesion: 0.10
Nodes (16): integration, Test that Ruff linting integrates correctly with CI., Test that MyPy type checking integrates correctly with CI., Test that the test suite integrates correctly with CI., Test that coverage reporting works in CI environment., Test CI/CD pipeline integration for style guide enforcement., Test that JSON reports are generated for CI consumption., Test that tests can run in parallel for faster CI. (+8 more)

### Community 37 - "TestEnergySweepSi"
Cohesion: 0.07
Nodes (3): 500-point energy sweep for Si. Lock snapshot values at indices 0, 249, 499 and…, Higher energy → shorter wavelength., TestEnergySweepSi

### Community 38 - "typing_extensions.py"
Cohesion: 0.07
Nodes (33): performance, Tests for enhanced XRayResult dataclass typing and type safety. This module…, Test that XRayResult maintains memory efficiency with type annotations., Test that typing extension validation helpers work correctly., Test type guard functionality for XRayResult arrays., Test that type annotations don't impact XRayResult performance., Test suite for XRayResult dataclass typing enhancements., Helper method to create test XRayResult instances. (+25 more)

### Community 39 - "TestBackwardCompatibility"
Cohesion: 0.33
Nodes (4): Tests for backward compatibility with existing CLI., Test handling of various legacy argument formats., Test that optional arguments are handled., TestBackwardCompatibility

### Community 40 - "XRayResult"
Cohesion: 0.07
Nodes (12): Test XRayResult array conversion optimization., Test that XRayResult properly converts input arrays to numpy arrays., Test that XRayResult arrays maintain consistent dtypes for performance., Test XRayResult creation with properly typed inputs., Post-initialization to handle any setup after object creation., Dataclass containing complete X-ray optical property calculations for a…, Deprecated: Use 'formula' instead., Deprecated: Use 'molecular_weight_g_mol' instead. (+4 more)

### Community 41 - "Complex refractive index n=1-delta-i*beta"
Cohesion: 0.83
Nodes (4): Complex refractive index n=1-delta-i*beta, Critical angle formula theta_c=sqrt(2*delta), Linear absorption coefficient mu=4pi*beta/lambda, Calculators Module

### Community 42 - "validate_density"
Cohesion: 0.10
Nodes (15): Tests for validate_density function., Test validation of valid density value., Test validation with custom min/max range., Test density at minimum boundary., Test density at maximum boundary., Test that density below minimum raises ValidationError., Test that density above maximum raises ValidationError., Test that negative density raises ValidationError. (+7 more)

### Community 43 - "test_utilities.py"
Cohesion: 0.08
Nodes (25): assert_values_close(), clean_cache_around(), expect_no_warnings(), expect_warnings(), memory_test(), performance_test(), Test utilities and helper functions for xraylabtool tests. This module provides…, Decorator to mark and validate performance tests. (+17 more)

### Community 44 - "validate_chemical_formula"
Cohesion: 0.10
Nodes (15): Test that empty string raises FormulaError., Test that whitespace-only string raises FormulaError., Test that None input raises FormulaError., Test that invalid characters raise FormulaError., Test that unknown element raises FormulaError., Test that formula with spaces is handled., Tests for validate_chemical_formula function., Test validation of simple chemical formula. (+7 more)

### Community 45 - "_calc"
Cohesion: 0.11
Nodes (15): _calc(), _clear_cache(), fixture, parametrize, Characterization tests for _calculate_molecular_properties (Plan section P2-E).…, Lock the exact float64 values produced by v0.3.0. rtol=1e-3 is tight enough to…, Return (molecular_weight, total_electrons) for *formula*., Ensure a cold-path call for every test. (+7 more)

### Community 46 - "test_concurrent_cache.py"
Cohesion: 0.11
Nodes (15): clear_lru_caches(), _mock_get_element(), fixture, Tests for thread-safety of lru_cache-decorated functions in utils.py., Clearing the cache while threads are accessing it does not cause crashes., Multiple threads calling get_atomic_weight simultaneously return correct values., Clearing get_atomic_weight cache while threads are active does not crash., LRU cache returns the same cached object for repeated calls. (+7 more)

### Community 47 - "TestPhysicalRealism"
Cohesion: 0.14
Nodes (8): Test that results maintain physical realism., Test that critical angles are always positive and physically reasonable.…, Test monotonic decrease in energy regions free of absorption edges. Away from…, Test that a discontinuity exists across the Si K-edge at 1.839 keV. The real…, Test that attenuation length is always positive., Test that electron density is consistent with formula and density., Test that scattering factors are physically reasonable., TestPhysicalRealism

### Community 49 - "test_docs.py"
Cohesion: 0.19
Nodes (21): check_accessibility(), check_documentation_coverage(), Colors, main(), print_section(), print_status(), Test code examples in RST documentation files., Test code examples in README.md. (+13 more)

### Community 50 - "uninstall_completion_main"
Cohesion: 0.09
Nodes (15): Test that get_xraylabtool_commands returns a list., Test that commands list is not empty., Tests for completion integration functions., Test that install_completion_main is available., Test that uninstall_completion_main is available., Test that get_xraylabtool_commands is available., TestCompletionFunctions, Tests for uninstall_completion_main function. (+7 more)

### Community 51 - "calculate_scattering_factors"
Cohesion: 0.15
Nodes (18): WavelengthArray, calculate_derived_quantities(), calculate_scattering_factors(), _derived_quantities_kernel(), _get_derived_kernel(), _get_scattering_kernel(), Any, EnergyArray (+10 more)

### Community 52 - "test_file_operations.py"
Cohesion: 0.15
Nodes (22): Tests for xraylabtool.io.file_operations., Energy column (col 0) should be strictly increasing., test_load_absolute_path(), test_load_csv_file(), test_load_csv_space_separated_fallback(), test_load_empty_nff_raises_data_file_error(), test_load_malformed_nff_raises_data_file_error(), test_load_nff_energy_column_monotone() (+14 more)

### Community 53 - "TestTypeSafetyConfig"
Cohesion: 0.09
Nodes (13): performance, Test that NumPy typing support is available., Test that TYPE_CHECKING flag is available for import optimization., Test that modern Python 3.12+ typing features are available., Test that MyPy cache directory can be created and used., Test that type checking infrastructure has minimal performance impact., Test suite for type safety configuration validation., Test that required type stub packages are available. (+5 more)

### Community 54 - "TestValidateEnergyRange"
Cohesion: 0.10
Nodes (11): Tests for validate_energy_range function., Test validation of single valid energy value., Test validation of array of valid energy values., Test validation with custom min/max range., Test that energy below minimum raises EnergyError., Test that energy above maximum raises EnergyError., Test that negative energy raises EnergyError., Test that zero energy raises EnergyError. (+3 more)

### Community 55 - "ndarray"
Cohesion: 0.10
Nodes (11): Any, ndarray, Deprecated: Use 'energy_kev' instead., Deprecated: Use 'wavelength_angstrom' instead., Deprecated: Use 'dispersion_delta' instead., Deprecated: Use 'absorption_beta' instead., Deprecated: Use 'scattering_factor_f1' instead., Deprecated: Use 'scattering_factor_f2' instead. (+3 more)

### Community 56 - "BaseXRayLabToolTest"
Cohesion: 0.12
Nodes (12): BaseXRayLabToolTest, Any, Validate that result has expected XRayResult structure. Args: result: Result…, Validate result against reference values from Julia implementation. Args:…, Base class for all XRayLabTool tests. Provides common test utilities, assertion…, Assert that operation performance meets threshold requirements. Args:…, Time an operation and return result and execution time. Args: func: Function to…, Tests for type safety configuration and MyPy setup validation. This module… (+4 more)

### Community 58 - "current_palette"
Cohesion: 0.18
Nodes (11): current_palette(), Detect the active palette by checking the application background color., F1F2Plot, MultiF1F2Plot, Any, QWidget, Scattering factor plot widgets backed by PyQtGraph. This module contains small…, Render f1 and f2 vs energy for multiple materials. Parameters ----------… (+3 more)

### Community 59 - "ColorPalette"
Cohesion: 0.18
Nodes (14): QPalette, apply_pyqtgraph_theme(), apply_styles(), apply_theme(), ColorPalette, get_qss(), QApplication, Shared Qt styling for the XRayLabTool GUI. (+6 more)

### Community 60 - "TestBasicExportFunctionality"
Cohesion: 0.11
Nodes (10): Test that exported data maintains scientific accuracy., Test basic export functionality., Set up test fixtures., Test basic CSV export functionality., Test CSV export with custom fields., Test CSV export with empty results., Test basic JSON export functionality., Test JSON export with multiple results. (+2 more)

### Community 61 - "TestNumericalStabilityChecks"
Cohesion: 0.11
Nodes (10): Test safe critical angle calculation with np.maximum., Test safe attenuation length calculation with epsilon handling., Test handling of zero absorption values., Test numerical stability checks in calculate_derived_quantities., Test NaN detection in dispersion coefficients., Test NaN detection in absorption coefficients., Test infinity detection in dispersion coefficients., Test infinity detection in absorption coefficients. (+2 more)

### Community 62 - "TestCIWorkflowSimulation"
Cohesion: 0.11
Nodes (10): Set up test fixtures., Test complete CI workflow simulation., Set up test fixtures., Test complete CI workflow from start to finish., Test that dependencies can be imported., Test code quality checks., Test that tests can be executed., Test style guide validation. (+2 more)

### Community 63 - "BaseUnitTest"
Cohesion: 0.06
Nodes (27): BaseUnitTest, Base class for unit tests., Setup for each test method., Teardown for each test method., Test CI/CD pipeline integration for style guide enforcement. This module…, Consolidated code quality tests for pyXRayLabTool. This module validates that…, Test that public functions have adequate docstring coverage., Check for naming convention violations. (+19 more)

### Community 64 - "parse_formula"
Cohesion: 0.18
Nodes (6): Test basic formula parsing., CO (carbon monoxide) vs Co (cobalt)., Repeated elements in flat formulas are aggregated., TestParseFormulaBasic, parse_formula(), Parse a chemical formula string into element symbols and their counts.…

### Community 65 - ".__init__"
Cohesion: 0.12
Nodes (9): Any, Initialize DataFileError with message and optional filename context., Initialize ValidationError with message and optional parameter context., Initialize AtomicDataError with message and optional element context., Initialize UnknownElementError with element symbol., Initialize BatchProcessingError with message and optional failure context., Initialize CalculationError with message and optional context. Args: message:…, Initialize FormulaError with message and optional formula context. Args:… (+1 more)

### Community 66 - "calculate_transmission"
Cohesion: 0.16
Nodes (10): exp(-t/Lambda) must be bit-for-bit identical; atol=1e-14., t=0.1 cm (1 mm) through SiO2 at 10 keV., t=0.001 cm (10 um) — thin-film regime., Direct formula check: T = exp(-t/Lambda)., Zero thickness must give T=1 exactly., Very thick sample (100x Lambda) must give T~0., Transmission must decrease strictly with thickness., TestGoldenTransmission (+2 more)

### Community 67 - "ndarray"
Cohesion: 0.13
Nodes (12): arrays_close(), assert_arrays_close(), ndarray, Generate logarithmic energy array., Generate random energy array., Generate energy values that test edge cases., Utility class for validating calculation results., Validate XRayResult object. (+4 more)

### Community 68 - "MemoryProfiler"
Cohesion: 0.14
Nodes (9): MemoryProfiler, Start memory profiling., Stop profiling and record measurement., Context manager for memory profiling., Get average memory increase in MB., Get maximum memory increase in MB., Assert that memory usage is within threshold., Context manager for measuring execution time. (+1 more)

### Community 69 - "PerformanceBenchmark"
Cohesion: 0.12
Nodes (9): PerformanceBenchmark, Utility class for performance benchmarking., Start timing measurement., Stop timing and record measurement., Get average execution time in milliseconds., Get minimum execution time in milliseconds., Get maximum execution time in milliseconds., Reset all measurements. (+1 more)

### Community 70 - "calculate_critical_angle"
Cohesion: 0.18
Nodes (9): Two canonical delta values from the test plan., delta=1e-6 -> ~0.08103 degrees., delta=5e-6 -> ~0.18119 degrees., Verify formula: theta = sqrt(2*delta) * (180/pi)., Array of delta values returns array of the same length., Critical angle must increase monotonically with delta., TestGoldenCriticalAngle, calculate_critical_angle() (+1 more)

### Community 71 - "test_golden_pipeline.py"
Cohesion: 0.13
Nodes (5): Characterization tests: core pipeline golden values (P1-A, P1-B, P1-D). Golden…, Lock intermediate values of calculate_scattering_factors for Si. These values…, Isolate the derived-quantities formula from the interpolation layer. Inputs are…, TestGoldenDerivedQuantities, TestGoldenScatteringFactors

### Community 73 - "ThemeManager"
Cohesion: 0.16
Nodes (9): QApplication, QObject, Manages application theme state and persistence., Get the active ColorPalette object., Set the active theme mode., Get current theme mode string., Toggle between light and dark modes., Apply current theme to application and global resources. (+1 more)

### Community 74 - "PlotCanvas"
Cohesion: 0.19
Nodes (8): apply_palette_to_widget(), Apply background and foreground colors from palette to a PlotWidget., PlotCanvas, Any, QWidget, PyQtGraph-based plot canvas widget., Re-apply colors from the currently active palette., Single-panel plot widget backed by PyQtGraph.

### Community 78 - "InterpolatorProtocol"
Cohesion: 0.11
Nodes (17): RealArray, AtomicDataProvider, CalculationEngine, InterpolatorProtocol, ComplexArray, EnergyArray, OpticalConstantArray, Protocol (+9 more)

### Community 80 - "TestInputValidationAndErrorHandling"
Cohesion: 0.14
Nodes (8): Test comprehensive input validation and error handling., Test energy below 0.03 keV., Test energy above 30 keV., Test mismatched formulaList & massDensityList lengths., Test non-vector formula input., Test empty formula list., Test empty energy vector., TestInputValidationAndErrorHandling

### Community 81 - "test_memory_management.py"
Cohesion: 0.18
Nodes (8): Tests for memory management and batch processing optimizations. This test…, Test memory management in batch processing., Test memory cleanup in single calculations., Test cache clearing integration with memory management., TestBatchProcessingMemoryManagement, TestProcessSingleCalculation, process_single_calculation(), Process a single X-ray calculation. Args: formula: Chemical formula energies:…

### Community 82 - "TestBasicAnalysis"
Cohesion: 0.19
Nodes (9): Tests for basic analysis functions. This module tests the simplified analysis…, Test the basic analysis functionality., Test basic material comparison., Test material comparison with empty results., Test material comparison with different property., Test material comparison with non-existent property., TestBasicAnalysis, compare_materials() (+1 more)

### Community 83 - "test_numerical_stability.py"
Cohesion: 0.08
Nodes (14): Tests for numerical stability improvements. This test module validates the…, Test calculations under extreme conditions., Test calculation at very low energies (near the lower bound)., Test calculation at very high energies (near the upper bound)., Test calculation with very low density materials., Test calculation with very high density materials., Test calculation with large energy arrays., Test calculation with logarithmic energy spacing. (+6 more)

### Community 84 - "TestValidateCalculationParameters"
Cohesion: 0.14
Nodes (8): Tests for validate_calculation_parameters function., Test validation of valid calculation parameters., Test validation with array of energies., Test that invalid formula raises error., Test that invalid energy raises error., Test that invalid density raises error., Test that first invalid parameter is caught., TestValidateCalculationParameters

### Community 85 - "TestEnvironmentDetection"
Cohesion: 0.17
Nodes (8): dict, Tests for environment detection functionality., Test detecting current Python environment., Test markers for venv detection., Test markers for conda environment detection., Test detection of virtual environment via env var., Test detection of conda environment via env var., TestEnvironmentDetection

### Community 86 - "ParametrizedTestMixin"
Cohesion: 0.18
Nodes (9): param, generate_test_energies(), ParametrizedTestMixin, ndarray, Mixin providing utilities for parametrized tests., Create pytest parameters for material testing., Create pytest parameters for energy testing., Generate test energy arrays. (+1 more)

### Community 87 - "CalculationWorker"
Cohesion: 0.17
Nodes (16): QRunnable, fixture, qt_app(), Tests for CalculationWorker error and success signal paths., Run worker synchronously (no thread pool needed for unit tests)., _run_worker(), test_worker_does_not_emit_finished_on_exception(), test_worker_emits_error_on_exception() (+8 more)

### Community 88 - "OverlayScrollbarMarginHelper"
Cohesion: 0.24
Nodes (6): QEvent, OverlayScrollbarMarginHelper, QObject, QScrollArea, Overlay scrollbar margin helper for QScrollArea widgets., Adjusts viewport margins so overlay scrollbars don't cover content. Install on…

### Community 89 - "main"
Cohesion: 0.29
Nodes (14): analyze_mypy_output(), check_core_modules(), check_type_definitions(), main(), Any, Path, Analyze MyPy output to extract metrics and insights. Parameters ----------…, Check core modules with enhanced strict mode. Parameters ----------… (+6 more)

### Community 90 - "TestCalculationSpeedBenchmarks"
Cohesion: 0.22
Nodes (8): performance, Benchmark batch processing performance for multiple materials., Benchmark memory allocation patterns and garbage collection impact., Comprehensive benchmarks for calculation speed optimization., Benchmark scattering factor interpolator creation speed., Benchmark single material calculation speed across energy ranges., Benchmark concurrent calculation performance with threading., TestCalculationSpeedBenchmarks

### Community 91 - "normalize_intensity"
Cohesion: 0.12
Nodes (16): Tests for data processing functions., Test smoothing with invalid window size., Test intensity normalization by maximum., Test intensity normalization by area., Test standard score normalization., TestDataProcessing, background_subtraction(), normalize_intensity() (+8 more)

### Community 92 - "cache.py"
Cohesion: 0.17
Nodes (12): get_bulk_atomic_data(), get_cached_elements(), is_element_cached(), MappingProxyType, Cache infrastructure for scattering factor data and interpolators., Smart cache warming that only loads elements needed for the specific…, Check if scattering factor data for an element is already cached. Args:…, Get list of elements currently cached in the scattering factor cache. Returns:… (+4 more)

### Community 94 - "TestDataFactory"
Cohesion: 0.21
Nodes (8): create_batch_test_data(), Factory for generating test data., Get list of test materials., Get list of material formulas., Get list of material densities., Generate linear energy array., Create batch test data with specified counts., TestDataFactory

### Community 95 - "TestBenchmarkComparison"
Cohesion: 0.23
Nodes (7): Benchmark comparison tests for optimization validation., Overall test of optimization effectiveness., Benchmark single element calculation., Benchmark multi-element calculation., Benchmark batch calculation., Benchmark cache access., TestBenchmarkComparison

### Community 98 - "TableFormatter"
Cohesion: 0.21
Nodes (7): Any, Shared table-cell formatting for X-ray result rows., Formats X-ray calculation results into display-ready cell strings., Format one energy point of a single-material result. Returns a list of strings…, Format one energy point of a multi-material result. Returns a list of strings…, Format the single-material summary row. Returns: [formula, molecular_weight,…, TableFormatter

### Community 99 - "TestPerformanceBenchmarks"
Cohesion: 0.24
Nodes (7): benchmark, Performance benchmark tests using pytest-benchmark., Benchmark single SubRefrac calculation., Benchmark multi-material Refrac calculation., Benchmark calculation with large energy sweep., Benchmark calculation with complex formula., TestPerformanceBenchmarks

### Community 100 - "Findings"
Cohesion: 0.08
Nodes (23): Coverage Mapping, Findings, Priority Recommendations (ordered), TEST-001, TEST-002, TEST-003, TEST-004, TEST-005 (+15 more)

### Community 101 - "constants.py"
Cohesion: 0.12
Nodes (17): parametrize, Characterization tests for physical constants and unit conversions (Plan P2-F).…, Lock energy_to_wavelength_angstrom() at specific energies to atol=1e-8 Å. This…, Round-trip: wavelength_angstrom_to_energy(energy_to_wavelength_angstrom(E)) ≈ E., TestEnergyToWavelength, attenuation_length_cm(), energy_to_wavelength_angstrom(), _isclose() (+9 more)

### Community 102 - "test_golden_derived_quantities.py"
Cohesion: 0.20
Nodes (7): fixture, Characterization tests: derived_quantities.py module (P2-A through P2-D). The…, Verify exact formula: 2*pi * delta / wl^2 (wl in Angstroms)., calculate_scattering_length_density(), ndarray, Derived quantities calculator for X-ray optical properties. This module…, Calculate real and imaginary scattering length densities. Args:…

### Community 103 - "calculate_attenuation_length"
Cohesion: 0.25
Nodes (6): Known wavelength/beta inputs -> known output, locked at rtol=1e-10., Verify Λ = lambda_m / (4*pi*beta) * 100., More absorbing material -> shorter attenuation length., TestGoldenAttenuationLength, calculate_attenuation_length(), Calculate X-ray attenuation length (1/e length). Args: wavelength_angstrom:…

### Community 104 - "test_formula_parsing.py"
Cohesion: 0.20
Nodes (7): parametrize, Tests for the canonical chemical formula parser and its delegating wrappers.…, Verify number-format handling matches the Julia regex., TestDelegatingWrappers, TestRegexCompatibility, _parse_formula(), Parse chemical formula into elements and quantities. Delegates to the canonical…

### Community 105 - "TestGoldenScatteringLengthDensity"
Cohesion: 0.20
Nodes (4): Scalar inputs -> known SLD, locked at atol=1e-12., real_sld = 2*pi*delta/(...) is positive for positive delta., im_sld = 2*pi*beta/(...) is always positive for positive beta., TestGoldenScatteringLengthDensity

### Community 106 - "TestCompletionIntegrationFlow"
Cohesion: 0.20
Nodes (6): Tests for integration flow between new and legacy systems., Test that new completion system imports correctly., Test that CompletionInstaller is available., Test that circular imports are avoided., Test that cli module is available for delegation., TestCompletionIntegrationFlow

### Community 107 - "TestBatchConfig"
Cohesion: 0.20
Nodes (6): Test BatchConfig optimizations., Test default BatchConfig initialization., Test automatic memory limit adjustment based on system memory., Test worker count calculation., Test custom configuration parameters., TestBatchConfig

### Community 108 - "FastXRayCalculationEngine"
Cohesion: 0.22
Nodes (7): FastXRayCalculationEngine, get_calculation_engine(), High-performance X-ray calculation engine implementing CalculationEngine…, Initialize the calculation engine., Pre-warm caches for improved performance., Get performance information about the calculation engine., Get the global calculation engine instance.

### Community 109 - ".uninstall"
Cohesion: 0.22
Nodes (5): Uninstall completion from a specific environment., Uninstall completion from all environments., Uninstall bash completion (backward compatibility)., Uninstall completion (backward compatibility)., Uninstall completion from environment(s). Args: target_env: Target environment…

### Community 110 - "TestMultiMaterialEquivalence"
Cohesion: 0.28
Nodes (3): parametrize, calculate_multiple_xray_properties with 3 materials must return values…, TestMultiMaterialEquivalence

### Community 111 - "RegressionTracker"
Cohesion: 0.28
Nodes (5): Track performance regressions across test runs., Load performance baselines from file., Save performance baselines to file., Check for performance regression., RegressionTracker

### Community 112 - "find_absorption_edges"
Cohesion: 0.22
Nodes (6): Test basic absorption edge detection., Test edge detection with insufficient data., Test edge detection with no edges present., find_absorption_edges(), ndarray, Simple absorption edge detection using f2 derivative. Args: energies: Energy…

### Community 113 - "TestDerivedConstants"
Cohesion: 0.25
Nodes (4): Lock derived constants to their exact captured float64 values., SCATTERING_FACTOR must equal THOMPSON * AVOGADRO * 1e6 / (2π)., ENERGY_TO_WAVELENGTH_FACTOR must equal (h·c/e) / 1000., TestDerivedConstants

### Community 114 - "CI/CD Security Audit Report"
Cohesion: 0.13
Nodes (14): CI/CD Security Audit Report, CICD-001, CICD-002, CICD-003, CICD-004, CICD-005, CICD-006, CICD-007 (+6 more)

### Community 115 - "`test_docs.py` - Documentation Testing"
Cohesion: 0.14
Nodes (13): Available Scripts, Example Output:, Integration:, Maintenance Notes, Output:, Requirements:, `run_type_check.py` - Type Checking, Scripts Directory (+5 more)

### Community 116 - "ADR-004: Host-Device Transfer Minimization Strategy"
Cohesion: 0.09
Nodes (22): ADR-004: Host-Device Transfer Minimization Strategy, Appendix: Transfer Cost Reference (CPU Backend), Array Sizes in This Workload, Batch Processing Optimization, Consequences, Context, Decision, Implementation Pattern (+14 more)

### Community 117 - "ADR-003: float64 Preservation Policy"
Cohesion: 0.17
Nodes (11): ADR-003: float64 Preservation Policy, Appendix: Numerical Precision Requirements by Function, Consequences, Context, Current Codebase float64 Usage, Decision, Enforcement Strategy, JAX float32 Default (+3 more)

### Community 118 - "Documentation Root"
Cohesion: 0.28
Nodes (9): Examples Index, CLI Reference, Shell Completion Guide, Getting Started Guide, Migration Guide v0.4, Documentation Root, Atomic Data Reference, X-ray Calculations Guide (+1 more)

### Community 119 - "TestCLIOutputFormatting"
Cohesion: 0.25
Nodes (5): Test field filtering with edge cases., Test CLI output formatting edge cases., Test output formatting with extreme precision values., Test formatting of large datasets., TestCLIOutputFormatting

### Community 120 - "TestCLICompatibility"
Cohesion: 0.25
Nodes (5): Test CLI compatibility and backward compatibility., Test compatibility of different output formats., Test field name compatibility across versions., Test version information consistency., TestCLICompatibility

### Community 121 - "API Reference Index"
Cohesion: 0.29
Nodes (7): Analysis API, Backend API, Constants API, Data Handling API, API Reference Index, IO Operations API, Validation API

### Community 122 - "TestEdgeCasesAndErrorHandling"
Cohesion: 0.25
Nodes (5): Test edge cases and error handling., Test with empty materials., Test with mismatched array lengths., Test accessing non-existent material., TestEdgeCasesAndErrorHandling

### Community 123 - "TestMemoryLeakPrevention"
Cohesion: 0.25
Nodes (5): Test prevention of memory leaks., Test that repeated calculations don't cause memory leaks., Test that batch calculations clean up memory properly., Test memory management with large arrays., TestMemoryLeakPrevention

### Community 124 - "PerformanceMetrics"
Cohesion: 0.25
Nodes (5): PerformanceMetrics, Protocol for performance measurement and validation., Get current calculation rate in calculations/second., Get current memory usage in MB., Validate that performance meets target calculations/second.

### Community 126 - "get_system_info"
Cohesion: 0.29
Nodes (6): get_system_info(), log_system_info(), Any, Validate batch calculation results., Get system information for test context., Log system information for debugging.

### Community 127 - "TestH2OProperties"
Cohesion: 0.29
Nodes (4): Test H2O dispersion values., Test H2O reSLD values., Test H2O properties matching Julia test values., TestH2OProperties

### Community 128 - "performance/conftest.py"
Cohesion: 0.29
Nodes (6): ci_threshold_multiplier(), fixture, pytest_collection_modifyitems(), Performance test configuration. Skips all performance tests unless --run-…, Skip performance tests unless --run-performance is passed., Return a multiplier for wall-clock thresholds (3x in CI, 1x locally).

### Community 129 - "tests/conftest.py"
Cohesion: 0.04
Nodes (65): assert_calculation_result_valid(), assert_memory_usage_reasonable(), assert_performance_within_threshold(), batch_config_default(), batch_config_performance(), batch_config_small(), benchmark_baseline(), clean_cache() (+57 more)

### Community 132 - "TestXRayResultProtocolCompliance"
Cohesion: 0.33
Nodes (4): Test that XRayResult can work with protocol-based interfaces., Test that XRayResult can be used in protocol-based contexts., Test that XRayResult arrays support standard array operations., TestXRayResultProtocolCompliance

### Community 135 - "TestComplexFormula"
Cohesion: 0.40
Nodes (3): Test parenthesis-containing formula through the molecular properties path., Ca5(PO4)3OH: verify MW matches textbook value (~502.31 g/mol)., TestComplexFormula

### Community 136 - "xraylabtool — Unified Audit & Remediation Plan"
Cohesion: 0.18
Nodes (10): Cross-Referenced Root Causes, Executive Summary, Findings Cross-Reference Index, Prioritized Remediation Plan, Tier 1 — Immediate (Week 1) — Correctness & Quick Security Wins, Tier 2 — Short-term (Weeks 2-3) — Security Hardening & CI, Tier 3 — Medium-term (Weeks 4-6) — Architecture & Coverage, Tier 4 — Long-term (Backlog) — Polish & Hardening (+2 more)

### Community 137 - "apply_palette_to_item"
Cohesion: 0.40
Nodes (4): apply_palette_to_item(), Any, Apply background/foreground colors from palette to a PlotItem., Re-apply colors from the currently active palette.

### Community 138 - "_handle_mendeleev_error"
Cohesion: 0.50
Nodes (4): NoReturn, _handle_mendeleev_error(), Exception, Handle errors from mendeleev package.

### Community 140 - "_auto_select_backend"
Cohesion: 0.50
Nodes (4): _auto_select_backend(), _has_nvidia_gpu(), Fast check for NVIDIA GPU without importing JAX (~1ms)., Select the best available backend: JAX GPU > JAX CPU > NumPy. JAX is preferred…

### Community 145 - "BashCompletionGenerator"
Cohesion: 0.05
Nodes (32): ABC, Regression: generated bash for subcommand-bearing commands., The subcommand-completion block was a plain (non-f) string, so it emitted…, TestBashCompletionSubstitution, BashCompletionGenerator, CompletionGenerator, FishCompletionGenerator, PowerShellCompletionGenerator (+24 more)

### Community 146 - "ADR-001: JAX vs NumPy Computation Backend"
Cohesion: 0.20
Nodes (9): ADR-001: JAX vs NumPy Computation Backend, Appendix: Functions that MUST NOT be JIT-compiled, Appendix: Functions to JIT-Compile (Priority Order), Consequences, Context, Decision, Negative, Positive (+1 more)

### Community 148 - "TestSiO2Properties"
Cohesion: 0.29
Nodes (4): Test SiO2 properties matching Julia test values., Test SiO2 dispersion values., Test SiO2 reSLD values., TestSiO2Properties

### Community 150 - "ADR-002: PyQtGraph vs Matplotlib for GUI Plotting"
Cohesion: 0.22
Nodes (8): ADR-002: PyQtGraph vs Matplotlib for GUI Plotting, Appendix: Feature Parity Checklist, Consequences, Context, Decision, Migration Details, Negative, Positive

### Community 152 - "MemoryMonitor"
Cohesion: 0.08
Nodes (18): MemoryTracker, Context manager for tracking memory usage., performance, Test the MemoryMonitor class and its optimizations., Test cache management optimizations., Test MemoryMonitor initialization., Test cache statistics for monitoring., Test cache behavior under memory pressure. (+10 more)

### Community 154 - "Testing Guide"
Cohesion: 0.50
Nodes (4): Audit Remediation Plan, Contributing Guide, Testing Guide, Scripts README

### Community 155 - "conf.py"
Cohesion: 0.50
Nodes (3): Configuration file for the Sphinx documentation builder. For the full list of…, Set up the Sphinx application configuration., setup()

### Community 158 - "pyXRayLabTool Logo (Main SVG)"
Cohesion: 0.67
Nodes (3): pyXRayLabTool Logo (Dark), pyXRayLabTool Logo (Light JPG), pyXRayLabTool Logo (Main SVG)

### Community 168 - "Files"
Cohesion: 0.25
Nodes (7): Files, Maintenance, `performance_baselines.json`, `sample_materials.json`, Test Data Directory, `test_energies.json`, Usage in Tests

## Knowledge Gaps
- **122 isolated node(s):** `Colors`, `Colors`, `Context`, `Decision`, `Positive` (+117 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **36 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CompletionInstaller` connect `CompletionInstaller` to `interfaces/cli.py`, `TestBackwardCompatibility`, `EnvironmentDetector`, `TestCompletionIntegrationFlow`, `EnvironmentInfo`, `.uninstall`, `BashCompletionGenerator`, `uninstall_completion_main`, `TestInstallCompletionCommand`, `installer.py`, `legacy_install_completion_main`?**
  _High betweenness centrality (0.164) - this node is a cross-community bridge._
- **Why does `calculate_single_material_properties()` connect `calculate_single_material_properties` to `interfaces/cli.py`, `xraylabtool/__init__.py`, `BasePerformanceTest`, `test_cli_edge_cases.py`, `test_integration.py`, `timer`, `core.py`, `main_window.py`, `batch_processing.py`, `XRayResult`, `TestBasicExportFunctionality`, `test_golden_pipeline.py`, `test_memory_management.py`, `TestBasicAnalysis`, `TestCalculationSpeedBenchmarks`, `cache.py`, `TestBenchmarkComparison`, `TestCLIOutputFormatting`, `TestCLICompatibility`?**
  _High betweenness centrality (0.156) - this node is a cross-community bridge._
- **Why does `parse_formula()` connect `parse_formula` to `EnergyError`, `interfaces/cli.py`, `TestParseFormulaDecimal`, `TestParseFormulaErrors`, `test_compound_analysis.py`, `TestParseFormulaParentheses`, `test_formula_parsing.py`, `xraylabtool/__init__.py`, `_calc`, `MaterialInputForm`, `core.py`, `cache.py`, `main_window.py`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `CompletionInstaller` (e.g. with `.test_completion_installer_methods()` and `.test_install_completion_help()`) actually correct?**
  _`CompletionInstaller` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `MainWindow` (e.g. with `EnergyConfig` and `TableFormatter`) actually correct?**
  _`MainWindow` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 32 inferred relationships involving `FormulaError` (e.g. with `TestBasicSetupAndInitialization` and `TestDuplicatedFormulasIndependence`) actually correct?**
  _`FormulaError` has 32 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Colors`, `Colors`, `Context` to the rest of the system?**
  _122 weakly-connected nodes found - possible documentation gaps or missing edges._
