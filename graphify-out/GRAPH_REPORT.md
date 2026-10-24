# Graph Report - pyXRayLabTool  (2026-10-23)

## Corpus Check
- 156 files · ~124,579 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 102 file(s) not represented in the graph (top: .nff 93, .log 2, (none) 2)

## Summary
- 3721 nodes · 6729 edges · 232 communities (87 shown, 145 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 459 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `9bcc5a5b`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- bragg_angle
- test_exceptions.py
- interfaces/__init__.py
- tests/conftest.py
- StyleGuideValidator
- EnvironmentDetector
- CompletionInstaller
- parse_formula
- Any
- timer
- test_cli_edge_cases.py
- environment.py
- _calculate_single_material_xray_properties
- SharedHelpersMixin
- main
- formatting.py
- MemoryMonitor
- EnvironmentInfo
- validate_chemical_formula
- BatchConfig
- xraylabtool/__init__.py
- TestPerformanceBenchmarks
- calculate_xray_properties
- TestCodeQuality
- batch_processing.py
- EnergyConfig
- TestEnvironmentProperties
- format_xray_result
- create_scattering_factor_interpolators
- format_xray_result
- test_backend_dispatch.py
- TestCICDIntegration
- TestCLIOutputFormatting
- TestEnergySweepSi
- parser.py
- typing_extensions.py
- test_utils.py
- _calc
- TestFormulaValidation
- xraylabtool/__init__.py
- pytest
- TestGoldenAu
- BasePerformanceTest
- Findings
- test_cli.py
- ADR-004: Host-Device Transfer Minimization Strategy
- Utilities Module
- AtomicDataProvider
- TestFullPipelineSiO2
- MaterialComparator
- TestTypeSafetyConfig
- atomic_cache.py
- PlotCanvas
- test_docs.py
- .calculate_derived_quantities
- single_tab.py
- CalculationWorker
- MaterialComparator
- process_single_calculation
- TestCalculationSpeedBenchmarks
- TestCIWorkflowSimulation
- NumpyBackend
- MainWindowProtocol
- ColorPalette
- TestRegexCompatibility
- legacy_install_completion_main
- get_atomic_number
- core.py
- calculate_transmission
- validate_density
- PerformanceBenchmark
- test_jax_backend_parity.py
- TestEnergyParsing
- installer.py
- get_atomic_number
- CrystalStructure
- TestXRayResultTyping
- test_file_operations.py
- ValidationError
- CI/CD Security Audit Report
- export_to_csv
- main
- calculate_critical_angle
- calculate_single_material_properties
- get_atomic_weight
- cmd_calc
- validate_energy_range
- Any
- Changelog
- TestGoldenSi
- TestGoldenFe2O3
- TestGoldenSiO2
- cli/__init__.py
- TestPhysicalRealism
- TestParseFormulaDecimal
- TestEnvironmentDetection
- AtomicScatteringFactor
- TestExtremeConditions
- test_formula_parsing.py
- TestParseFormulaParentheses
- ADR-003: float64 Preservation Policy
- test_base.py
- OverlayScrollbarMarginHelper
- `test_docs.py` - Documentation Testing
- test_golden_constants.py
- ndarray
- CalculationError
- validate_calculation_parameters
- formula.py
- test_compound_analysis.py
- TestInstallCompletionCommand
- FormulaError
- xraylabtool — Unified Audit & Remediation Plan
- file_operations.py
- calculators/__init__.py
- calculate_attenuation_length
- Shell Completion Guide
- energy_to_wavelength
- uninstall_completion_main
- ADR-001: JAX vs NumPy Computation Backend
- TestGoldenScatteringLengthDensity
- BashCompletionGenerator
- test_memory_management.py
- get_compound_family
- parse_chemical_formula
- fixture
- get_bulk_atomic_data
- .format_multi_row
- xraylabtool/utils.py
- validate_calculation_parameters
- ADR-002: PyQtGraph vs Matplotlib for GUI Plotting
- TestMultiMaterialEquivalence
- compare.py
- RegressionTracker
- FishCompletionGenerator
- cmd_convert
- Files
- .uninstall
- load_scattering_factor_data
- find_similar_compounds
- get_compound_complexity_score
- get_compound_frequency_score
- get_elements_for_compound
- json
- EnergyError
- FastXRayCalculationEngine
- PerformanceMetrics
- BaseUnitTest
- get_gpu_info
- TestUnitConversionConstants
- performance/conftest.py
- compound_analysis.py
- BatchProcessingError
- BaseXRayLabToolTest
- MaterialInputForm
- numpy
- _collect_compound_elements
- XRayResult
- PowerShellCompletionGenerator
- BaseIntegrationTest
- Complex refractive index n=1-delta-i*beta
- Testing Guide
- TestCompletionIntegrationFlow
- test_integration.py
- _auto_select_backend
- X-ray Calculations Guide
- pyXRayLabTool Logo (Main SVG)
- Toast
- cmd_atomic
- check_gpu_availability
- Test Coverage Audit
- TestTypeValidationHelpers
- CompletionGenerator
- TestDerivedConstants
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
- TestCLIFileOperations
- TestCLIErrorHandling
- TestEdgeCasesAndErrorHandling
- TestEdgeCaseInputValues
- TestEnvironmentType
- TestEnvironmentIntegration
- TestBoundaryConditions
- gc
- TestH2OProperties
- TestSiO2Properties
- unit
- ensure_complex128_array
- TestBackwardCompatibility
- TestSubRefracErrorHandling
- TestXRayResultProtocolCompliance
- compound_analysis.py
- MemoryTracker
- test_data_generator
- TestExactCapturedValues
- TestPropertyConsistency
- assert_calculation_result_valid
- mock_low_memory
- skip_if_low_memory
- benchmark_baseline
- material_and_density
- memory_monitor
- test_energies_small
- test_energies_medium
- Sample Materials Test Data
- test_headless_gui_smoke_runs_offscreen
- _validate_balanced_parentheses

## God Nodes (most connected - your core abstractions)
1. `MainWindowProtocol` - 85 edges
2. `calculate_single_material_properties()` - 75 edges
3. `CompletionInstaller` - 69 edges
4. `parse_formula()` - 56 edges
5. `XRayResult` - 41 edges
6. `BatchConfig` - 41 edges
7. `calculate_xray_properties()` - 37 edges
8. `EnvironmentInfo` - 35 edges
9. `JaxBackend` - 33 edges
10. `clear_scattering_factor_cache()` - 31 edges

## Surprising Connections (you probably didn't know these)
- `wrapper()` --calls--> `clear_scattering_factor_cache()`  [INFERRED]
  tests/fixtures/test_utilities.py → xraylabtool/calculators/cache.py
- `cold_cache()` --calls--> `clear_scattering_factor_cache()`  [INFERRED]
  tests/characterization/test_golden_interpolation.py → xraylabtool/calculators/cache.py
- `_calc()` --calls--> `get_bulk_atomic_data()`  [INFERRED]
  tests/characterization/test_golden_molecular.py → xraylabtool/calculators/cache.py
- `_clear_cache()` --calls--> `clear_scattering_factor_cache()`  [INFERRED]
  tests/characterization/test_golden_molecular.py → xraylabtool/calculators/cache.py
- `setup_test_environment()` --calls--> `clear_scattering_factor_cache()`  [INFERRED]
  tests/conftest.py → xraylabtool/calculators/cache.py

## Import Cycles
- 4-file cycle: `xraylabtool/__init__.py -> xraylabtool/analysis/__init__.py -> xraylabtool/calculators/core.py -> xraylabtool/logging_utils.py -> xraylabtool/__init__.py`
- 4-file cycle: `xraylabtool/__init__.py -> xraylabtool/calculators/__init__.py -> xraylabtool/calculators/core.py -> xraylabtool/logging_utils.py -> xraylabtool/__init__.py`
- 5-file cycle: `xraylabtool/__init__.py -> xraylabtool/analysis/__init__.py -> xraylabtool/analysis/comparator.py -> xraylabtool/calculators/core.py -> xraylabtool/logging_utils.py -> xraylabtool/__init__.py`
- 5-file cycle: `xraylabtool/__init__.py -> xraylabtool/data_handling/__init__.py -> xraylabtool/data_handling/batch_processing.py -> xraylabtool/calculators/core.py -> xraylabtool/logging_utils.py -> xraylabtool/__init__.py`

## Hyperedges (group relationships)
- **v0.4.0 Architecture Overhaul** — xraylabtool_backend, xraylabtool_gui, xraylabtool_calculators [EXTRACTED 1.00]
- **JAX Architecture Decision Records** — docs_architecture_001_adr_jax_vs_numpy_jax_decision, docs_architecture_003_adr_dtype_policy_dtype_strategy, architecture_004_adr_host_device_transfers_transfer_policy, docs_architecture_jax_architecture_jax_design [INFERRED 0.85]

## Communities (232 total, 145 thin omitted)

### Community 0 - "bragg_angle"
Cohesion: 0.09
Nodes (7): TestCrystallographicCalculations, angle_from_q(), bragg_angle(), d_spacing_cubic(), d_spacing_orthorhombic(), d_spacing_tetragonal(), q_from_angle()

### Community 1 - "test_exceptions.py"
Cohesion: 0.08
Nodes (7): TestAtomicDataError, TestExceptionHierarchy, TestUnknownElementError, AtomicDataError, UnknownElementError, XRayLabToolError, _handle_mendeleev_error()

### Community 3 - "tests/conftest.py"
Cohesion: 0.09
Nodes (9): assert_memory_usage_reasonable(), assert_performance_within_threshold(), energy_arrays(), pytest_addoption(), pytest_collection_modifyitems(), pytest_configure(), sample_calculation_result(), test_energies_large() (+1 more)

### Community 4 - "StyleGuideValidator"
Cohesion: 0.06
Nodes (5): Colors, main(), StyleGuideValidator, StyleViolation, ValidationReport

### Community 6 - "CompletionInstaller"
Cohesion: 0.08
Nodes (3): TestSentinelMarkers, CompletionInstaller, test_completion_installation()

### Community 7 - "parse_formula"
Cohesion: 0.14
Nodes (4): TestParseFormulaBasic, _aggregate_element_counts(), _expand_parentheses(), parse_formula()

### Community 9 - "timer"
Cohesion: 0.08
Nodes (4): TestPerformanceBenchmarks, TestPerformanceRegression, TestScalabilityBenchmarks, timer()

### Community 10 - "test_cli_edge_cases.py"
Cohesion: 0.07
Nodes (4): TestBatchRowResilience, TestCLIInputValidation, TestCLIPerformanceAndMemory, TestFieldNameValidation

### Community 11 - "environment.py"
Cohesion: 0.10
Nodes (6): TestGetDeviceInfo, TestGetSystemCudaVersion, check_plugin_conflicts(), get_device_info(), get_recommended_package(), get_system_cuda_version()

### Community 12 - "_calculate_single_material_xray_properties"
Cohesion: 0.14
Nodes (7): calculate_multiple_xray_properties(), _calculate_single_material_xray_properties(), _create_process_formula_function(), process_formula(), _MaterialPropertiesDict, _restore_energy_order(), _validate_single_material_inputs()

### Community 14 - "main"
Cohesion: 0.09
Nodes (6): TestCLIIntegration, TestMainFunction, cmd_completion(), cmd_install_completion(), cmd_uninstall_completion(), main()

### Community 15 - "formatting.py"
Cohesion: 0.07
Nodes (15): TestResultFormatting, _format_array_field_single(), _format_as_csv(), _format_as_json(), _format_filtered_table(), _format_material_properties(), _format_multiple_energies(), _format_multiple_energy_section() (+7 more)

### Community 16 - "MemoryMonitor"
Cohesion: 0.11
Nodes (3): TestMemoryMonitor, MemoryMonitor, process_batch_chunk()

### Community 17 - "EnvironmentInfo"
Cohesion: 0.09
Nodes (3): TestEnvironmentInfo, EnvironmentInfo, EnvironmentType

### Community 18 - "validate_chemical_formula"
Cohesion: 0.10
Nodes (3): TestValidateChemicalFormula, _get_valid_element_symbols(), validate_chemical_formula()

### Community 19 - "BatchConfig"
Cohesion: 0.14
Nodes (5): TestBatchProcessorEquivalence, TestBatchInputValidation, TestBatchPartialFailure, BatchConfig, calculate_batch_properties()

### Community 20 - "xraylabtool/__init__.py"
Cohesion: 0.11
Nodes (9): TestEnergyToWavelength, attenuation_length_cm(), critical_angle_degrees(), energy_to_wavelength_angstrom(), _isclose(), validate_constants(), wavelength_angstrom_to_energy(), _configure_jax_float64() (+1 more)

### Community 21 - "TestPerformanceBenchmarks"
Cohesion: 0.14
Nodes (5): TestPerformanceBenchmarks, energy_sweep(), complex_formula(), multi_calculation(), single_calculation()

### Community 22 - "calculate_xray_properties"
Cohesion: 0.13
Nodes (4): TestInputValidationAndErrorHandling, calculate_xray_properties(), _process_formulas_parallel(), _validate_xray_inputs()

### Community 24 - "batch_processing.py"
Cohesion: 0.11
Nodes (11): chunk_iterator(), _filter_dataframe_fields(), _get_energy_point_data(), _initialize_progress_bar(), load_batch_input(), _prepare_energies_array(), _prepare_result_data(), _process_chunks() (+3 more)

### Community 25 - "EnergyConfig"
Cohesion: 0.10
Nodes (13): _run_smoke(), test_compute_multiple_reports_progress_to_100(), _ensure_app(), test_export_csv_handles_io_error(), test_gui_smoke_offscreen(), test_gui_smoke_threaded_offscreen(), test_duplicate_formula_different_density_not_dropped(), enable_offscreen_quiet_env() (+5 more)

### Community 27 - "format_xray_result"
Cohesion: 0.08
Nodes (7): TestInstallCompletionMain, completion_main(), create_completion_parser(), handle_completion_command(), install_completion_main(), show_completion_info(), handle_new_completion_command()

### Community 28 - "create_scattering_factor_interpolators"
Cohesion: 0.07
Nodes (7): _check_element(), cold_cache(), test_si_o_au_interpolators_are_independent(), TestPchipAu, TestPchipO, TestPchipSi, create_scattering_factor_interpolators()

### Community 29 - "format_xray_result"
Cohesion: 0.05
Nodes (9): MockXRayResult, TestFormatCalculationSummary, TestFormatXRayResult, _format_as_csv(), _format_as_json(), _format_as_table(), format_calculation_summary(), format_xray_result() (+1 more)

### Community 30 - "test_backend_dispatch.py"
Cohesion: 0.09
Nodes (7): TestBackendSwitching, TestInterpolationFactory, TestOpsProxy, get_backend(), _OpsProxy, set_backend(), InterpolationFactory

### Community 34 - "parser.py"
Cohesion: 0.12
Nodes (12): add_atomic_command(), add_batch_command(), add_bragg_command(), add_calc_command(), add_compare_command(), add_completion_command(), add_convert_command(), add_formula_command() (+4 more)

### Community 35 - "typing_extensions.py"
Cohesion: 0.13
Nodes (7): get_optimal_chunk_size(), is_complex_array(), is_energy_array(), is_real_array(), optimize_array_memory_layout(), validate_complex_array(), validate_energy_array()

### Community 36 - "test_utils.py"
Cohesion: 0.15
Nodes (3): TestDataProcessing, normalize_intensity(), smooth_data()

### Community 37 - "_calc"
Cohesion: 0.11
Nodes (6): _calc(), _clear_cache(), TestComplexFormula, TestElectronCountGolden, TestMolecularWeightGolden, _calculate_molecular_properties()

### Community 38 - "TestFormulaValidation"
Cohesion: 0.09
Nodes (6): form(), _get_app(), qt_app(), TestDensityValidation, TestEnergyGridValidation, TestFormulaValidation

### Community 40 - "pytest"
Cohesion: 0.07
Nodes (12): arrays_close(), assert_arrays_close(), assert_values_close(), expect_no_warnings(), expect_warnings(), get_system_info(), log_system_info(), skip_if_insufficient_cpu() (+4 more)

### Community 42 - "BasePerformanceTest"
Cohesion: 0.07
Nodes (10): BasePerformanceTest, generate_test_materials(), TestArrayOptimization, TestBulkAtomicDataOptimization, TestDeprecationWarningOptimization, TestElementPathOptimization, TestInterpolatorCaching, TestNumericalStabilityOptimizations (+2 more)

### Community 43 - "Findings"
Cohesion: 0.08
Nodes (23): Coverage Mapping, Findings, Priority Recommendations (ordered), TEST-001, TEST-002, TEST-003, TEST-004, TEST-005 (+15 more)

### Community 44 - "test_cli.py"
Cohesion: 0.12
Nodes (3): TestBatchCommand, TestListCommand, cmd_list()

### Community 45 - "ADR-004: Host-Device Transfer Minimization Strategy"
Cohesion: 0.09
Nodes (19): ADR-004: Host-Device Transfer Minimization Strategy, Appendix: Transfer Cost Reference (CPU Backend), Array Sizes in This Workload, Batch Processing Optimization, Consequences, Context, Decision, Implementation Pattern (+11 more)

### Community 46 - "Utilities Module"
Cohesion: 0.10
Nodes (12): angle_from_q(), bragg_angle(), d_spacing_cubic(), xraylabtool.device, energy_to_wavelength(), get_atomic_data(), load_atomic_data(), xraylabtool.logging_utils (+4 more)

### Community 49 - "MaterialComparator"
Cohesion: 0.11
Nodes (3): TestBasicAnalysis, compare_materials(), find_absorption_edges()

### Community 51 - "atomic_cache.py"
Cohesion: 0.09
Nodes (13): _smart_cache_warming(), _warm_priority_cache(), _background_cache_warming(), FastAtomicDataProvider, get_atomic_data_fast(), get_atomic_data_provider(), get_bulk_atomic_data_fast(), get_cache_stats() (+5 more)

### Community 52 - "PlotCanvas"
Cohesion: 0.09
Nodes (6): apply_palette_to_item(), apply_palette_to_widget(), current_palette(), PlotCanvas, F1F2Plot, MultiF1F2Plot

### Community 53 - "test_docs.py"
Cohesion: 0.18
Nodes (11): check_accessibility(), check_documentation_coverage(), Colors, main(), print_section(), print_status(), run_style_checks(), test_doctests() (+3 more)

### Community 55 - "single_tab.py"
Cohesion: 0.10
Nodes (12): setup(), test_configure_logging_writes_file(), test_get_logger_child_inherits_config(), build_parser(), main(), _bool_env(), _coerce_int_env(), configure_logging() (+4 more)

### Community 56 - "CalculationWorker"
Cohesion: 0.14
Nodes (10): qt_app(), _run_worker(), test_worker_does_not_emit_finished_on_exception(), test_worker_emits_error_on_exception(), bad_fn(), test_worker_emits_result_on_success(), test_worker_injects_progress_cb_when_accepted(), test_worker_passes_kwargs() (+2 more)

### Community 58 - "process_single_calculation"
Cohesion: 0.24
Nodes (3): no_progress_config(), TestProcessSingleCalculation, process_single_calculation()

### Community 61 - "NumpyBackend"
Cohesion: 0.09
Nodes (3): TestEdgeCases, TestNumpyBackend, NumpyBackend

### Community 62 - "MainWindowProtocol"
Cohesion: 0.05
Nodes (4): MultiTabMixin, MainWindowProtocol, SingleTabMixin, MaterialTable

### Community 63 - "ColorPalette"
Cohesion: 0.10
Nodes (6): apply_pyqtgraph_theme(), apply_styles(), apply_theme(), ColorPalette, get_qss(), ThemeManager

### Community 64 - "TestRegexCompatibility"
Cohesion: 0.20
Nodes (3): TestDelegatingWrappers, TestRegexCompatibility, _parse_formula()

### Community 66 - "get_atomic_number"
Cohesion: 0.11
Nodes (5): TestAtomicDataLookup, worker(), get_atomic_data(), get_atomic_number(), get_atomic_weight()

### Community 67 - "core.py"
Cohesion: 0.09
Nodes (10): get_cached_elements(), is_element_cached(), _prepare_element_data(), calculate_derived_quantities(), calculate_scattering_factors(), _derived_quantities_kernel(), _get_derived_kernel(), _get_scattering_kernel() (+2 more)

### Community 70 - "PerformanceBenchmark"
Cohesion: 0.05
Nodes (12): clean_cache_around(), wrapper(), memory_test(), decorator(), wrapper(), MemoryProfiler, performance_test(), decorator() (+4 more)

### Community 71 - "test_jax_backend_parity.py"
Cohesion: 0.11
Nodes (5): restore_numpy_backend(), TestJaxBackendParity, TestScalarEnergyFallback, _convert_energy_input(), _validate_and_process_energies()

### Community 73 - "installer.py"
Cohesion: 0.10
Nodes (5): TestCompletionInstallToTempDir, TestCompletionScriptGeneration, CompletionManager, get_global_options(), get_xraylabtool_commands()

### Community 74 - "get_atomic_number"
Cohesion: 0.09
Nodes (11): clear_lru_caches(), _mock_get_element(), TestCacheResultConsistency, TestConcurrentGetAtomicNumber, worker(), cache_clearer(), reader(), worker() (+3 more)

### Community 75 - "CrystalStructure"
Cohesion: 0.06
Nodes (6): TestCrystalStructure, _make_cubic(), _phase(), TestCrystalStructureAPI, TestStructureFactorPhysics, CrystalStructure

### Community 77 - "test_file_operations.py"
Cohesion: 0.12
Nodes (14): TestDataFileError, test_load_absolute_path(), test_load_csv_file(), test_load_csv_space_separated_fallback(), test_load_empty_nff_raises_data_file_error(), test_load_malformed_nff_raises_data_file_error(), test_load_nff_energy_column_monotone(), test_load_nff_values_are_finite() (+6 more)

### Community 79 - "CI/CD Security Audit Report"
Cohesion: 0.12
Nodes (15): CI/CD Security Audit Report, CICD-001, CICD-002, CICD-003, CICD-004, CICD-005, CICD-006, CICD-007 (+7 more)

### Community 80 - "export_to_csv"
Cohesion: 0.08
Nodes (4): TestBasicExportIntegration, TestBasicExportFunctionality, export_to_csv(), export_to_json()

### Community 81 - "main"
Cohesion: 0.25
Nodes (6): analyze_mypy_output(), check_core_modules(), check_type_definitions(), main(), run_mypy_check(), validate_cache_performance()

### Community 83 - "calculate_single_material_properties"
Cohesion: 0.09
Nodes (5): _clear_cache(), _single(), cold_cache(), clear_scattering_factor_cache(), calculate_single_material_properties()

### Community 85 - "cmd_calc"
Cohesion: 0.13
Nodes (6): TestCalcCommand, cmd_calc(), _determine_output_format(), _print_calc_verbose_info(), _save_or_print_output(), _validate_calc_inputs()

### Community 89 - "Changelog"
Cohesion: 0.24
Nodes (10): Dev Suite Progress, Science Suite Progress, Changelog, XRayLabTool Documentation, Backend Abstraction, set_backend, calculate_single_material_properties, calculate_xray_properties (+2 more)

### Community 93 - "cli/__init__.py"
Cohesion: 0.17
Nodes (7): cmd_batch(), _convert_result_to_dict(), _parse_batch_data(), _process_batch_materials(), _save_batch_results(), _validate_batch_input(), create_batch_progress_tracker()

### Community 97 - "AtomicScatteringFactor"
Cohesion: 0.08
Nodes (6): TestAtomicScatteringFactor, AtomicScatteringFactor, _initialize_element_paths(), load_data_file(), load_scattering_factor_data(), ScatteringData

### Community 101 - "ADR-003: float64 Preservation Policy"
Cohesion: 0.17
Nodes (11): ADR-003: float64 Preservation Policy, Appendix: Numerical Precision Requirements by Function, Consequences, Context, Current Codebase float64 Usage, Decision, Enforcement Strategy, JAX float32 Default (+3 more)

### Community 104 - "`test_docs.py` - Documentation Testing"
Cohesion: 0.20
Nodes (11): Available Scripts, Example Output:, Integration:, Maintenance Notes, Output:, Requirements:, `run_type_check.py` - Type Checking, Scripts Directory (+3 more)

### Community 105 - "test_golden_constants.py"
Cohesion: 0.11
Nodes (3): TestDtypePreservation, TestFundamentalConstants, TestMathematicalConstants

### Community 106 - "ndarray"
Cohesion: 0.09
Nodes (3): create_batch_test_data(), ResultValidator, TestDataFactory

### Community 107 - "CalculationError"
Cohesion: 0.12
Nodes (3): TestCalculationError, TestExceptionUsage, CalculationError

### Community 109 - "formula.py"
Cohesion: 0.14
Nodes (6): TestFormulaCommand, cmd_formula(), _get_atomic_data(), _output_formula_results(), _print_formula_results(), _process_formula()

### Community 113 - "xraylabtool — Unified Audit & Remediation Plan"
Cohesion: 0.18
Nodes (10): Cross-Referenced Root Causes, Executive Summary, Findings Cross-Reference Index, Prioritized Remediation Plan, Tier 1 — Immediate (Week 1) — Correctness & Quick Security Wins, Tier 2 — Short-term (Weeks 2-3) — Security Hardening & CI, Tier 3 — Medium-term (Weeks 4-6) — Architecture & Coverage, Tier 4 — Long-term (Backlog) — Polish & Hardening (+2 more)

### Community 114 - "file_operations.py"
Cohesion: 0.17
Nodes (9): test_save_and_reload_numpy_array(), test_save_dict_of_arrays_csv(), test_save_json(), test_save_unsupported_format_raises(), _json_numpy_default(), save_calculation_results(), _save_dict_as_csv(), _save_results_as_csv() (+1 more)

### Community 117 - "Shell Completion Guide"
Cohesion: 0.67
Nodes (3): CLI Reference, Shell Completion Guide, Shell Completion v2

### Community 118 - "energy_to_wavelength"
Cohesion: 0.14
Nodes (6): TestUnitConversions, energy_to_wavelength(), get_electron_charge(), get_planck_constant(), get_speed_of_light(), wavelength_to_energy()

### Community 121 - "ADR-001: JAX vs NumPy Computation Backend"
Cohesion: 0.20
Nodes (9): ADR-001: JAX vs NumPy Computation Backend, Appendix: Functions that MUST NOT be JIT-compiled, Appendix: Functions to JIT-Compile (Priority Order), Consequences, Context, Decision, Negative, Positive (+1 more)

### Community 124 - "test_memory_management.py"
Cohesion: 0.08
Nodes (3): TestBatchConfig, TestCacheManagement, TestMemoryLeakPrevention

### Community 127 - "fixture"
Cohesion: 0.12
Nodes (8): batch_config_default(), batch_config_performance(), batch_config_small(), cpu_count(), memory_tracker(), performance_thresholds(), performance_timer(), temp_directory()

### Community 130 - "xraylabtool/utils.py"
Cohesion: 0.15
Nodes (6): background_subtraction(), _convert_atomic_number_to_int(), find_peaks(), get_avogadro_number(), progress_bar(), save_processed_data()

### Community 132 - "ADR-002: PyQtGraph vs Matplotlib for GUI Plotting"
Cohesion: 0.22
Nodes (8): ADR-002: PyQtGraph vs Matplotlib for GUI Plotting, Appendix: Feature Parity Checklist, Consequences, Context, Decision, Migration Details, Negative, Positive

### Community 138 - "Files"
Cohesion: 0.25
Nodes (7): Files, Maintenance, `performance_baselines.json`, `sample_materials.json`, Test Data Directory, `test_energies.json`, Usage in Tests

### Community 145 - "json"
Cohesion: 0.23
Nodes (5): cmd_compare(), _parse_materials(), _result_to_dict(), _write_report(), _write_table()

### Community 157 - "numpy"
Cohesion: 0.13
Nodes (3): large_batch_data(), sample_batch_data(), TestLoadDataFile

### Community 158 - "_collect_compound_elements"
Cohesion: 0.20
Nodes (5): _collect_compound_elements(), _collect_family_compound_elements(), _collect_similar_compound_elements(), warm_cache_for_compounds(), _warm_interpolators()

### Community 163 - "Complex refractive index n=1-delta-i*beta"
Cohesion: 0.83
Nodes (4): Complex refractive index n=1-delta-i*beta, Critical angle formula theta_c=sqrt(2*delta), Linear absorption coefficient mu=4pi*beta/lambda, Calculators Module

### Community 164 - "Testing Guide"
Cohesion: 0.50
Nodes (4): Audit Remediation Plan, Contributing Guide, Testing Guide, Scripts README

### Community 168 - "X-ray Calculations Guide"
Cohesion: 0.67
Nodes (3): Atomic Data Reference, X-ray Calculations Guide, X-ray Optics Guide

### Community 169 - "pyXRayLabTool Logo (Main SVG)"
Cohesion: 0.67
Nodes (3): pyXRayLabTool Logo (Dark), pyXRayLabTool Logo (Light JPG), pyXRayLabTool Logo (Main SVG)

### Community 207 - "gc"
Cohesion: 0.29
Nodes (3): clean_cache(), enforce_test_isolation(), setup_test_environment()

## Knowledge Gaps
- **35 isolated node(s):** `xraylabtool.device`, `xraylabtool.logging_utils`, `xraylabtool.typing_extensions`, `Analysis API`, `Backend API` (+30 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1775 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **145 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `calculate_single_material_properties()` connect `calculate_single_material_properties` to `timer`, `test_cli_edge_cases.py`, `_calculate_single_material_xray_properties`, `BatchConfig`, `xraylabtool/__init__.py`, `TestPerformanceBenchmarks`, `batch_processing.py`, `EnergyConfig`, `XRayResult`, `TestCLIOutputFormatting`, `test_integration.py`, `xraylabtool/__init__.py`, `pytest`, `BasePerformanceTest`, `MaterialComparator`, `atomic_cache.py`, `process_single_calculation`, `TestCalculationSpeedBenchmarks`, `core.py`, `test_jax_backend_parity.py`, `TestCLIFileOperations`, `TestEdgeCaseInputValues`, `export_to_csv`, `get_atomic_weight`, `TestSubRefracErrorHandling`, `cmd_calc`, `TestGoldenSiO2`, `cli/__init__.py`, `calculators/__init__.py`?**
  _High betweenness centrality (0.083) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `MainWindowProtocol` (e.g. with `MultiTabMixin` and `Toast`) actually correct?**
  _`MainWindowProtocol` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `xraylabtool.device`, `xraylabtool.logging_utils`, `xraylabtool.typing_extensions` to the rest of the system?**
  _35 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `bragg_angle` be split into smaller, more focused modules?**
  _Cohesion score 0.09116809116809117 - nodes in this community are weakly interconnected._
- **Why does `CompletionInstaller` connect `CompletionInstaller` to `legacy_install_completion_main`, `interfaces/__init__.py`, `TestCompletionIntegrationFlow`, `EnvironmentDetector`, `installer.py`, `.uninstall`, `test_cli.py`, `main`, `TestInstallCompletionCommand`, `EnvironmentInfo`, `uninstall_completion_main`, `format_xray_result`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `CompletionInstaller` (e.g. with `TestInstallCompletionCommand` and `TestCompletionInstallToTempDir`) actually correct?**
  _`CompletionInstaller` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Should `test_exceptions.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08403361344537816 - nodes in this community are weakly interconnected._
