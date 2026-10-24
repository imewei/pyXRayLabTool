# Graph Report - pyXRayLabTool  (2026-10-23)

## Corpus Check
- 190 files · ~127,529 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 3974 nodes · 7106 edges · 228 communities (94 shown, 134 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 471 edges (avg confidence: 0.87)
- Token cost: 337,103 input · 0 output

## Community Hubs (Navigation)
- Utilities & Formula Parsing
- Exception Hierarchy
- cmd_bragg
- tests/conftest.py
- Style Guide Validator
- Completion Environment Detection
- Completion Installer
- parse_formula
- Array Backend Abstraction
- timer
- CLI Edge-Case Tests
- test_device.py
- Core Calculation Pipeline
- SharedHelpersMixin
- CLI Main & Atomic Command
- Result Formatting (CLI)
- MemoryMonitor
- Environment Info Types
- validate_chemical_formula
- BatchConfig
- constants.py
- Performance Benchmarks
- Integration Tests: Properties
- Code Quality Tests
- Batch Processing
- GUI Services & Window
- TestEnvironmentProperties
- Completion Integration Tests
- create_scattering_factor_interpolators
- Data Export Formatting Tests
- Backend Switching & Dispatch
- CI/CD Integration Tests
- Energy Input & CLI Compatibility Tests
- TestEnergySweepSi
- parser.py
- typing_extensions.py
- test_utils.py
- _calc
- GUI Widget Validation Tests
- format_calculation_summary
- Test Utilities & Fixtures
- TestGoldenAu
- Optimization Performance Tests
- Findings
- cmd_list
- ADR-004: Host-Device Transfer Minimization Strategy
- Utilities Module
- AtomicDataProvider
- TestFullPipelineSiO2
- MaterialComparator
- TestTypeSafetyConfig
- data_handling/__init__.py
- GUI Plotting & Palette
- test_docs.py
- .calculate_derived_quantities
- single_tab.py
- CalculationWorker
- MaterialComparator
- process_single_calculation
- TestCalculationSpeedBenchmarks
- API Reference Index
- NumPy Backend & Tests
- MainWindowProtocol
- ColorPalette
- TestRegexCompatibility
- legacy_install_completion_main
- get_atomic_number
- kernels.py
- calculate_transmission
- validate_density
- Performance Benchmark Fixtures
- X-ray Optics Fundamentals
- TestEnergyParsing
- Completion Installer Tests
- get_atomic_number
- Crystal Structure Tests
- TestXRayResultTyping
- test_file_operations.py
- ValidationError
- CI/CD Security Audit Report
- TestBasicExportFunctionality
- main
- calculate_critical_angle
- clear_scattering_factor_cache
- SingleTabMixin
- cmd_calc
- validate_energy_range
- audit_remediation_plan.md
- Migration Guide v0.3.0 to v0.4.0
- TestGoldenSi
- TestGoldenFe2O3
- TestGoldenSiO2
- CLI Batch Command
- TestPhysicalRealism
- .install
- TestEnvironmentDetection
- AtomicScatteringFactor
- TestExtremeConditions
- test_formula_parsing.py
- TestParseFormulaParentheses
- ADR-003: float64 Preservation Policy
- test_base.py
- OverlayScrollbarMarginHelper
- `test_docs.py` - Documentation Testing
- TestFundamentalConstants
- ndarray
- CalculationError
- validate_calculation_parameters
- formula.py
- test_compound_analysis.py
- TestInstallCompletionCommand
- FormulaError
- xraylabtool — Unified Audit & Remediation Plan
- CSV & JSON Export
- calculators.core (calculate_single_material_properties, calculate_xray_properties)
- calculate_attenuation_length
- CLI Reference
- MultiTabMixin
- Calculation Methods
- energy_to_wavelength_angstrom
- ADR-001: JAX vs NumPy Computation Backend
- TestGoldenScatteringLengthDensity
- BashCompletionGenerator
- TestBatchConfig
- get_compound_family
- parse_chemical_formula
- fixture
- Deep-Debug Regression Tests
- .format_multi_row
- JAX Architecture & Design (v0.4.0)
- test_validators.py
- ADR-002: PyQtGraph vs Matplotlib for GUI Plotting
- TestMultiMaterialEquivalence
- Unreleased changes (post v0.4.6)
- RegressionTracker
- Shell Completion Generators
- cmd_convert
- Files
- .uninstall
- test_golden_pipeline.py
- find_similar_compounds
- get_compound_complexity_score
- get_compound_frequency_score
- get_elements_for_compound
- Performance and Optimization guide
- EnergyError
- FastAtomicDataProvider
- PerformanceMetrics
- TestGuiErrorHandling
- TestGetGpuInfo
- TestUnitConversionConstants
- performance/conftest.py
- compound_analysis.py
- BatchProcessingError
- BaseXRayLabToolTest
- DataFileError
- large_batch_data
- atomic_cache.py
- XRayResult Dataclass
- Getting Started guide
- BaseIntegrationTest
- XRayLabTool Documentation index
- TestBatchProcessing
- Scripts README
- TestCompletionIntegrationFlow
- data_export.py
- _auto_select_backend
- AtomicDataProvider
- pyXRayLabTool Logo (Main SVG)
- Toast
- Decision: replace matplotlib with PyQtGraph for interactive plots
- TestCheckGpuAvailability
- Test Coverage Audit
- test_type_safety_config.py
- validators.py
- TestDerivedConstants
- TestBasicExportIntegration
- Analysis API
- Backend API
- TestMemoryLeakPrevention
- Interfaces Module API Doc
- Testing Guide
- get_system_info
- IO Operations API
- MaterialTable
- TestPerformanceRegression
- Examples Index
- pyXRayLabTool Favicon
- GUI Main Window Screenshot
- conf.py
- psutil
- TestDtypePreservation
- TestEdgeCaseInputValues
- .__init__
- TestEnvironmentIntegration
- TestBoundaryConditions
- gc
- batch_config_small
- cpu_count
- performance_timer
- ensure_complex128_array
- temp_directory
- _configure_jax_float64
- TestXRayResultProtocolCompliance
- compound_analysis.py
- MemoryTracker
- test_data_generator
- _aggregate_element_counts
- assert_calculation_result_valid
- mock_low_memory
- skip_if_low_memory
- material_and_density
- test_energies_medium
- Sample Materials Test Data
- subprocess
- _validate_balanced_parentheses

## God Nodes (most connected - your core abstractions)
1. `calculate_single_material_properties()` - 87 edges
2. `MainWindowProtocol` - 80 edges
3. `CompletionInstaller` - 61 edges
4. `parse_formula()` - 59 edges
5. `BatchConfig` - 44 edges
6. `XRayResult` - 42 edges
7. `calculate_xray_properties()` - 40 edges
8. `EnvironmentInfo` - 35 edges
9. `clear_scattering_factor_cache()` - 35 edges
10. `JaxBackend` - 33 edges

## Surprising Connections (you probably didn't know these)
- `wrapper()` --calls--> `clear_scattering_factor_cache()`  [INFERRED]
  tests/fixtures/test_utilities.py → xraylabtool/calculators/cache.py
- `TestExceptionHierarchy` --uses--> `BatchProcessingError`  [INFERRED]
  tests/unit/test_exceptions.py → xraylabtool/exceptions.py
- `TestExceptionHierarchy` --uses--> `CalculationError`  [INFERRED]
  tests/unit/test_exceptions.py → xraylabtool/exceptions.py
- `TestExceptionHierarchy` --uses--> `DataFileError`  [INFERRED]
  tests/unit/test_exceptions.py → xraylabtool/exceptions.py
- `TestExceptionHierarchy` --uses--> `EnergyError`  [INFERRED]
  tests/unit/test_exceptions.py → xraylabtool/exceptions.py

## Import Cycles
- 4-file cycle: `xraylabtool/__init__.py -> xraylabtool/calculators/__init__.py -> xraylabtool/calculators/core.py -> xraylabtool/logging_utils.py -> xraylabtool/__init__.py`

## Hyperedges (group relationships)
- **JAX Architecture Decision Records** — docs_architecture_001_adr_jax_vs_numpy_jax_decision, architecture_004_adr_host_device_transfers_transfer_policy [INFERRED 0.85]
- **JAX migration architecture decisions (float64, host-device transfers, JAX design)** — docs_architecture_003_adr_dtype_policy_decision, docs_architecture_004_adr_host_device_transfers_decision, docs_architecture_jax_architecture_design_philosophy, docs_architecture_jax_architecture_jit [INFERRED 0.85]
- **CI/CD supply-chain hardening findings (unpinned actions, curl|sh, unpinned third-party action, unused uv pin)** — docs_security_audit_cicd_cicd_001, docs_security_audit_cicd_cicd_002, docs_security_audit_cicd_cicd_009, docs_security_audit_cicd_cicd_010 [EXTRACTED 1.00]
- **XRayLabTool exception hierarchy rooted at XRayLabToolError** — docs_api_validation_xraylabtoolerror, docs_api_validation_validationerror, docs_api_validation_formulaerror, docs_api_validation_energyerror, docs_api_validation_calculationerror [EXTRACTED 1.00]
- **Scientific background documentation set** — docs_physics_xray_optics, docs_physics_atomic_data, docs_physics_calculations [EXTRACTED 1.00]
- **Atomic factors to refractive index to derived optical quantities** — docs_physics_atomic_data_scattering_factors, docs_physics_calculations_number_density, docs_physics_calculations_delta_beta_sum, docs_physics_calculations_critical_angle, docs_physics_calculations_attenuation [EXTRACTED 1.00]
- **v0.4.0 JAX migration, rollback and release documentation** — docs_guides_migration_guide_v0_4, docs_development_rollback_procedures, docs_development_changelog_v0_4_0 [INFERRED 0.85]

## Communities (228 total, 134 thin omitted)

### Community 0 - "Utilities & Formula Parsing"
Cohesion: 0.05
Nodes (20): TestCrystallographicCalculations, TestUnitConversions, __getattr__(), angle_from_q(), bragg_angle(), _check_lattice(), _convert_atomic_number_to_int(), d_spacing_cubic() (+12 more)

### Community 1 - "Exception Hierarchy"
Cohesion: 0.09
Nodes (6): TestAtomicDataError, TestExceptionHierarchy, TestUnknownElementError, AtomicDataError, UnknownElementError, XRayLabToolError

### Community 3 - "tests/conftest.py"
Cohesion: 0.10
Nodes (9): assert_memory_usage_reasonable(), assert_performance_within_threshold(), benchmark_baseline(), energy_arrays(), pytest_addoption(), pytest_collection_modifyitems(), pytest_configure(), sample_batch_data() (+1 more)

### Community 4 - "Style Guide Validator"
Cohesion: 0.06
Nodes (5): Colors, main(), StyleGuideValidator, StyleViolation, ValidationReport

### Community 6 - "Completion Installer"
Cohesion: 0.08
Nodes (3): TestSentinelMarkers, CompletionInstaller, test_completion_installation()

### Community 7 - "parse_formula"
Cohesion: 0.13
Nodes (3): TestParseFormulaBasic, TestParseFormulaDecimal, parse_formula()

### Community 9 - "timer"
Cohesion: 0.06
Nodes (5): TestBenchmarkComparison, TestPerformanceBenchmarks, TestPerformanceRegression, TestScalabilityBenchmarks, timer()

### Community 10 - "CLI Edge-Case Tests"
Cohesion: 0.05
Nodes (6): TestBatchRowResilience, TestCLIErrorHandling, TestCLIFileOperations, TestCLIInputValidation, TestCLIPerformanceAndMemory, TestFieldNameValidation

### Community 12 - "Core Calculation Pipeline"
Cohesion: 0.06
Nodes (18): TestScalarEnergyFallback, _calculate_molecular_properties(), calculate_multiple_xray_properties(), _calculate_single_material_xray_properties(), _coerce_energy_array(), _convert_energy_input(), _create_process_formula_function(), process_formula() (+10 more)

### Community 14 - "CLI Main & Atomic Command"
Cohesion: 0.07
Nodes (6): TestAtomicCommand, TestBatchCommand, TestCLIIntegration, TestMainFunction, cmd_atomic(), main()

### Community 15 - "Result Formatting (CLI)"
Cohesion: 0.08
Nodes (15): TestResultFormatting, _format_array_field_single(), _format_as_csv(), _format_as_json(), _format_filtered_table(), _format_material_properties(), _format_multiple_energies(), _format_multiple_energy_section() (+7 more)

### Community 16 - "MemoryMonitor"
Cohesion: 0.10
Nodes (4): TestBatchProcessingMemoryManagement, TestCacheManagement, TestMemoryMonitor, MemoryMonitor

### Community 17 - "Environment Info Types"
Cohesion: 0.07
Nodes (4): TestEnvironmentInfo, TestEnvironmentType, EnvironmentInfo, EnvironmentType

### Community 19 - "BatchConfig"
Cohesion: 0.12
Nodes (5): TestBatchProcessorEquivalence, TestBatchInputValidation, TestBatchPartialFailure, BatchConfig, calculate_batch_properties()

### Community 20 - "constants.py"
Cohesion: 0.13
Nodes (6): TestMathematicalConstants, attenuation_length_cm(), critical_angle_degrees(), _isclose(), validate_constants(), wavelength_angstrom_to_energy()

### Community 21 - "Performance Benchmarks"
Cohesion: 0.14
Nodes (5): TestPerformanceBenchmarks, energy_sweep(), complex_formula(), multi_calculation(), single_calculation()

### Community 22 - "Integration Tests: Properties"
Cohesion: 0.05
Nodes (9): TestDuplicatedFormulasIndependence, TestEdgeCasesAndErrorHandling, TestH2OProperties, TestInputValidationAndErrorHandling, TestPropertyConsistency, TestSiO2Properties, TestSubRefracSiliconProperties, calculate_xray_properties() (+1 more)

### Community 23 - "Code Quality Tests"
Cohesion: 0.08
Nodes (3): TestCodeQuality, TestDocumentationPatterns, TestStyleCompliance

### Community 24 - "Batch Processing"
Cohesion: 0.10
Nodes (13): expect_warnings(), chunk_iterator(), _filter_dataframe_fields(), _get_energy_point_data(), _initialize_progress_bar(), _prepare_energies_array(), _prepare_result_data(), process_batch_chunk() (+5 more)

### Community 25 - "GUI Services & Window"
Cohesion: 0.10
Nodes (13): _run_smoke(), test_compute_multiple_reports_progress_to_100(), _ensure_app(), test_export_csv_handles_io_error(), test_gui_smoke_offscreen(), test_gui_smoke_threaded_offscreen(), test_duplicate_formula_different_density_not_dropped(), enable_offscreen_quiet_env() (+5 more)

### Community 27 - "Completion Integration Tests"
Cohesion: 0.07
Nodes (8): TestBackwardCompatibility, TestInstallCompletionMain, TestUninstallCompletionMain, completion_main(), create_completion_parser(), install_completion_main(), uninstall_completion_main(), handle_new_completion_command()

### Community 28 - "create_scattering_factor_interpolators"
Cohesion: 0.09
Nodes (6): _check_element(), test_si_o_au_interpolators_are_independent(), TestPchipAu, TestPchipO, TestPchipSi, create_scattering_factor_interpolators()

### Community 29 - "Data Export Formatting Tests"
Cohesion: 0.10
Nodes (3): MockXRayResult, TestFormatXRayResult, format_xray_result()

### Community 30 - "Backend Switching & Dispatch"
Cohesion: 0.07
Nodes (9): TestBackendSwitching, TestInterpolationFactory, TestOpsProxy, restore_numpy_backend(), TestJaxBackendParity, get_backend(), _OpsProxy, set_backend() (+1 more)

### Community 32 - "Energy Input & CLI Compatibility Tests"
Cohesion: 0.06
Nodes (6): TestCLICompatibility, TestCLIOutputFormatting, TestSubRefracErrorHandling, TestAnomalousDispersion, TestEnergyInputContract, calculate_single_material_properties()

### Community 34 - "parser.py"
Cohesion: 0.13
Nodes (12): add_atomic_command(), add_batch_command(), add_bragg_command(), add_calc_command(), add_compare_command(), add_completion_command(), add_convert_command(), add_formula_command() (+4 more)

### Community 35 - "typing_extensions.py"
Cohesion: 0.13
Nodes (7): get_optimal_chunk_size(), is_complex_array(), is_energy_array(), is_real_array(), optimize_array_memory_layout(), validate_complex_array(), validate_energy_array()

### Community 36 - "test_utils.py"
Cohesion: 0.12
Nodes (5): TestDataProcessing, background_subtraction(), normalize_intensity(), save_processed_data(), smooth_data()

### Community 37 - "_calc"
Cohesion: 0.10
Nodes (6): _calc(), _clear_cache(), TestComplexFormula, TestElectronCountGolden, TestExactCapturedValues, TestMolecularWeightGolden

### Community 38 - "GUI Widget Validation Tests"
Cohesion: 0.07
Nodes (8): form(), _get_app(), qt_app(), TestDensityValidation, TestEnergyGridValidation, TestFormulaValidation, TestFormValues, MaterialInputForm

### Community 40 - "Test Utilities & Fixtures"
Cohesion: 0.06
Nodes (7): arrays_close(), assert_arrays_close(), assert_values_close(), expect_no_warnings(), validate_batch_performance(), values_close(), TestLoadDataFile

### Community 42 - "Optimization Performance Tests"
Cohesion: 0.07
Nodes (9): BasePerformanceTest, generate_test_materials(), TestArrayOptimization, TestBulkAtomicDataOptimization, TestDeprecationWarningOptimization, TestElementPathOptimization, TestInterpolatorCaching, TestNumericalStabilityOptimizations (+1 more)

### Community 43 - "Findings"
Cohesion: 0.08
Nodes (23): Coverage Mapping, Findings, Priority Recommendations (ordered), TEST-001, TEST-002, TEST-003, TEST-004, TEST-005 (+15 more)

### Community 45 - "ADR-004: Host-Device Transfer Minimization Strategy"
Cohesion: 0.11
Nodes (17): ADR-004: Host-Device Transfer Minimization Strategy, Appendix: Transfer Cost Reference (CPU Backend), Array Sizes in This Workload, Batch Processing Optimization, Consequences, Context, Decision, Implementation Pattern (+9 more)

### Community 46 - "Utilities Module"
Cohesion: 0.10
Nodes (12): angle_from_q(), bragg_angle(), d_spacing_cubic(), xraylabtool.device, energy_to_wavelength(), get_atomic_data(), load_atomic_data(), xraylabtool.logging_utils (+4 more)

### Community 49 - "MaterialComparator"
Cohesion: 0.10
Nodes (3): TestBasicAnalysis, compare_materials(), find_absorption_edges()

### Community 51 - "data_handling/__init__.py"
Cohesion: 0.12
Nodes (10): _smart_cache_warming(), _warm_priority_cache(), _background_cache_warming(), get_atomic_data_fast(), get_bulk_atomic_data_fast(), get_cache_stats(), is_element_preloaded(), warm_up_cache() (+2 more)

### Community 52 - "GUI Plotting & Palette"
Cohesion: 0.09
Nodes (6): apply_palette_to_item(), apply_palette_to_widget(), current_palette(), PlotCanvas, F1F2Plot, MultiF1F2Plot

### Community 53 - "test_docs.py"
Cohesion: 0.17
Nodes (11): check_accessibility(), check_documentation_coverage(), Colors, main(), print_section(), print_status(), run_style_checks(), test_doctests() (+3 more)

### Community 55 - "single_tab.py"
Cohesion: 0.14
Nodes (11): test_configure_logging_writes_file(), test_get_logger_child_inherits_config(), build_parser(), main(), _bool_env(), _coerce_int_env(), configure_logging(), get_log_file_path() (+3 more)

### Community 56 - "CalculationWorker"
Cohesion: 0.16
Nodes (9): qt_app(), _run_worker(), test_worker_does_not_emit_finished_on_exception(), test_worker_emits_error_on_exception(), bad_fn(), test_worker_emits_result_on_success(), test_worker_injects_progress_cb_when_accepted(), test_worker_passes_kwargs() (+1 more)

### Community 58 - "process_single_calculation"
Cohesion: 0.19
Nodes (3): no_progress_config(), TestProcessSingleCalculation, process_single_calculation()

### Community 60 - "API Reference Index"
Cohesion: 0.13
Nodes (20): Constants Module API Doc, CODATA 2018 standards, Unit conversion factors (angstrom, cm, keV, rad), ENERGY_TO_WAVELENGTH_FACTOR (hc/e / 1000), Physical constants table (PLANCK, SPEED_OF_LIGHT, AVOGADRO, THOMPSON, ...), SCATTERING_FACTOR prefactor (THOMPSON*AVOGADRO*1e6/2pi), API Reference Index, Exception hierarchy (index overview) (+12 more)

### Community 61 - "NumPy Backend & Tests"
Cohesion: 0.09
Nodes (3): TestEdgeCases, TestNumpyBackend, NumpyBackend

### Community 63 - "ColorPalette"
Cohesion: 0.10
Nodes (6): apply_pyqtgraph_theme(), apply_styles(), apply_theme(), ColorPalette, get_qss(), ThemeManager

### Community 64 - "TestRegexCompatibility"
Cohesion: 0.17
Nodes (3): TestDelegatingWrappers, TestRegexCompatibility, _parse_formula()

### Community 66 - "get_atomic_number"
Cohesion: 0.10
Nodes (5): TestAtomicDataLookup, worker(), get_atomic_data(), get_atomic_number(), _handle_mendeleev_error()

### Community 67 - "kernels.py"
Cohesion: 0.16
Nodes (6): calculate_derived_quantities(), calculate_scattering_factors(), _derived_quantities_kernel(), _get_derived_kernel(), _get_scattering_kernel(), _scattering_math_kernel()

### Community 70 - "Performance Benchmark Fixtures"
Cohesion: 0.05
Nodes (12): clean_cache_around(), wrapper(), memory_test(), decorator(), wrapper(), MemoryProfiler, performance_test(), decorator() (+4 more)

### Community 71 - "X-ray Optics Fundamentals"
Cohesion: 0.17
Nodes (20): Negative delta (anomalous dispersion) accepted near edges, Contributing atomic data (citation, range, precision, format), Atomic Scattering Factor Data, Absorption edges (K, L, M) and XANES features, CXRO database, Henke et al. 1993, Atomic Data and Nuclear Data Tables 54, 181-342, Henke tables (Z=1-92, nff format), NIST XCOM photon cross sections (+12 more)

### Community 72 - "TestEnergyParsing"
Cohesion: 0.10
Nodes (7): TestEnergyParsing, cmd_compare(), _parse_materials(), _result_to_dict(), _write_report(), _write_table(), parse_energy_string()

### Community 73 - "Completion Installer Tests"
Cohesion: 0.09
Nodes (5): TestCompletionInstallToTempDir, TestCompletionScriptGeneration, CompletionManager, get_global_options(), get_xraylabtool_commands()

### Community 74 - "get_atomic_number"
Cohesion: 0.09
Nodes (11): clear_lru_caches(), _mock_get_element(), TestCacheResultConsistency, TestConcurrentGetAtomicNumber, worker(), cache_clearer(), reader(), worker() (+3 more)

### Community 75 - "Crystal Structure Tests"
Cohesion: 0.06
Nodes (6): TestCrystalStructure, _make_cubic(), _phase(), TestCrystalStructureAPI, TestStructureFactorPhysics, CrystalStructure

### Community 77 - "test_file_operations.py"
Cohesion: 0.18
Nodes (12): test_load_absolute_path(), test_load_csv_file(), test_load_csv_space_separated_fallback(), test_load_empty_nff_raises_data_file_error(), test_load_malformed_nff_raises_data_file_error(), test_load_nff_energy_column_monotone(), test_load_nff_values_are_finite(), test_load_nonexistent_file_raises_file_not_found() (+4 more)

### Community 79 - "CI/CD Security Audit Report"
Cohesion: 0.21
Nodes (19): Make CI security gates blocking (CICD-005, CICD-006), Tier 2 short-term (completion hardening, SHA pinning, Dependabot, backend tests), Tier 4 backlog (completion tests, GUI tests, license check, release bypass), CI/CD Security Audit Report, CICD-001, CICD-002, CICD-003, CICD-004 (+11 more)

### Community 81 - "main"
Cohesion: 0.25
Nodes (6): analyze_mypy_output(), check_core_modules(), check_type_definitions(), main(), run_mypy_check(), validate_cache_performance()

### Community 83 - "clear_scattering_factor_cache"
Cohesion: 0.12
Nodes (4): _clear_cache(), cold_cache(), cold_cache(), clear_scattering_factor_cache()

### Community 85 - "cmd_calc"
Cohesion: 0.13
Nodes (6): TestCalcCommand, cmd_calc(), _determine_output_format(), _print_calc_verbose_info(), _save_or_print_output(), _validate_calc_inputs()

### Community 88 - "audit_remediation_plan.md"
Cohesion: 0.15
Nodes (16): Legacy CamelCase accessors (DeprecationWarning), calculators submodules (derived_quantities, xray_result, cache, scattering_data, kernels), XRayResult fields table, validate_chemical_formula, interpax device-side interpolation, Transfer policy by boundary (load, cache, interpolator, compute, XRayResult, GUI, CLI, export), XRayResult as numpy public API boundary (bulk device-to-host in __post_init__), ArrayBackend protocol and proxy pattern (NumPy/JAX switching) (+8 more)

### Community 89 - "Migration Guide v0.3.0 to v0.4.0"
Cohesion: 0.15
Nodes (15): Dev Suite Progress, Science Suite Progress, Changelog (Keep a Changelog format), Backend abstraction layer (NumPy/JAX, InterpolationFactory), v0.4.0 release (JAX backend, PyQtGraph), v0.4.6 release (2026-09-17), Rollback: v0.4.0 to v0.3.0, Quick rollback procedure (pip uninstall, install 0.3.0, verify) (+7 more)

### Community 93 - "CLI Batch Command"
Cohesion: 0.10
Nodes (10): cmd_batch(), _convert_result_to_dict(), _parse_batch_data(), _process_batch_materials(), _save_batch_results(), _validate_batch_input(), cmd_completion(), cmd_install_completion() (+2 more)

### Community 97 - "AtomicScatteringFactor"
Cohesion: 0.07
Nodes (7): TestAtomicScatteringFactor, get_cached_elements(), is_element_cached(), AtomicScatteringFactor, _initialize_element_paths(), load_scattering_factor_data(), ScatteringData

### Community 101 - "ADR-003: float64 Preservation Policy"
Cohesion: 0.12
Nodes (15): ADR-003: float64 Preservation Policy, Appendix: Numerical Precision Requirements by Function, Consequences, Context, Current Codebase float64 Usage, Decision, Enforcement Strategy, JAX float32 Default (+7 more)

### Community 104 - "`test_docs.py` - Documentation Testing"
Cohesion: 0.20
Nodes (11): Available Scripts, Example Output:, Integration:, Maintenance Notes, Output:, Requirements:, `run_type_check.py` - Type Checking, Scripts Directory (+3 more)

### Community 106 - "ndarray"
Cohesion: 0.11
Nodes (3): create_batch_test_data(), ResultValidator, TestDataFactory

### Community 107 - "CalculationError"
Cohesion: 0.12
Nodes (3): TestCalculationError, TestExceptionUsage, CalculationError

### Community 109 - "formula.py"
Cohesion: 0.14
Nodes (6): TestFormulaCommand, cmd_formula(), _get_atomic_data(), _output_formula_results(), _print_formula_results(), _process_formula()

### Community 113 - "xraylabtool — Unified Audit & Remediation Plan"
Cohesion: 0.20
Nodes (10): Cross-Referenced Root Causes, Executive Summary, Findings Cross-Reference Index, Prioritized Remediation Plan, Tier 1 — Immediate (Week 1) — Correctness & Quick Security Wins, Tier 2 — Short-term (Weeks 2-3) — Security Hardening & CI, Tier 3 — Medium-term (Weeks 4-6) — Architecture & Coverage, Tier 4 — Long-term (Backlog) — Polish & Hardening (+2 more)

### Community 114 - "CSV & JSON Export"
Cohesion: 0.11
Nodes (12): test_save_and_reload_numpy_array(), test_save_dict_of_arrays_csv(), test_save_json(), test_save_unsupported_format_raises(), export_to_csv(), export_to_json(), _json_numpy_default(), save_calculation_results() (+4 more)

### Community 115 - "calculators.core (calculate_single_material_properties, calculate_xray_properties)"
Cohesion: 0.16
Nodes (14): Calculators Module API Doc, Linear absorption coefficient mu = 4 pi beta / lambda, calculators.core (calculate_single_material_properties, calculate_xray_properties), Critical angle theta_c = sqrt(2 delta), X-ray complex refractive index n = 1 - delta - i beta, Data Handling Module API Doc, atomic_cache (92-element preload, warm_up_cache, get_cache_stats), batch_processing (BatchConfig, calculate_batch_properties, CSV I/O) (+6 more)

### Community 116 - "calculate_attenuation_length"
Cohesion: 0.15
Nodes (3): TestGoldenAttenuationLength, calculate_attenuation_length(), calculate_scattering_length_density()

### Community 117 - "CLI Reference"
Cohesion: 0.12
Nodes (20): Energy grid design (log spacing, dense near absorption edges), CLI Reference, atomic command, bragg command, calc command, compare command, completion command (virtual-environment-centric), convert command (energy/wavelength) (+12 more)

### Community 119 - "Calculation Methods"
Cohesion: 0.22
Nodes (11): First calculation: calculate_single_material_properties(formula, keV, density), XRayResult (per-energy NumPy array fields), Interpolation methods (linear, logarithmic, spline), Calculation Methods, Attenuation coefficients (mu = 4 pi beta / lambda, mass coefficient, length), Per-element delta/beta contributions summed over compound, Chemical formula parser (grammar, errors), Linear interpolation of f1/f2 (np.interp) (+3 more)

### Community 121 - "ADR-001: JAX vs NumPy Computation Backend"
Cohesion: 0.20
Nodes (9): ADR-001: JAX vs NumPy Computation Backend, Appendix: Functions that MUST NOT be JIT-compiled, Appendix: Functions to JIT-Compile (Priority Order), Consequences, Context, Decision, Negative, Positive (+1 more)

### Community 127 - "fixture"
Cohesion: 0.12
Nodes (8): batch_config_default(), batch_config_performance(), memory_monitor(), memory_tracker(), performance_thresholds(), sample_calculation_result(), test_energies_large(), test_materials()

### Community 128 - "Deep-Debug Regression Tests"
Cohesion: 0.05
Nodes (13): stability_test(), test_whitespace_delimited_loader_works_with_current_pandas(), TestAtomicDataCacheIsImmutable, TestBackendState, TestExports, TestFormulaParsing, TestLazySubmoduleAccess, TestRepeatedMaterials (+5 more)

### Community 130 - "JAX Architecture & Design (v0.4.0)"
Cohesion: 0.27
Nodes (8): vmap-based batch processing (single bulk transfer), JAX Architecture & Design (v0.4.0), Automatic differentiation (future capability), JAX gotchas (Python control flow, non-array inputs), GPU acceleration and device placement, JIT compilation (cost-benefit, warm-up pattern), Shape-based recompilation gotcha, vmap vectorization (future v0.5+ automatic batching)

### Community 132 - "ADR-002: PyQtGraph vs Matplotlib for GUI Plotting"
Cohesion: 0.13
Nodes (14): ADR-002: PyQtGraph vs Matplotlib for GUI Plotting, Appendix: Feature Parity Checklist, Consequences, Context, Decision, Migration Details, Negative, Positive (+6 more)

### Community 134 - "Unreleased changes (post v0.4.6)"
Cohesion: 0.20
Nodes (9): Batch API fixes (ordering, timeouts, process-pool fallback), Energy input contract (scalar or 1-D real, else EnergyError), Machine-readable exports keep full precision, Lazy package loader RecursionError fix (importlib), safety dev dependency removed; pip-audit sole gate, Unreleased changes (post v0.4.6), Development workflow (branching, ruff, mypy, pytest, make verify), Coverage policy (22% CI gate, Codecov project and patch checks) (+1 more)

### Community 136 - "Shell Completion Generators"
Cohesion: 0.06
Nodes (4): CompletionGenerator, FishCompletionGenerator, PowerShellCompletionGenerator, ZshCompletionGenerator

### Community 138 - "Files"
Cohesion: 0.25
Nodes (7): Files, Maintenance, `performance_baselines.json`, `sample_materials.json`, Test Data Directory, `test_energies.json`, Usage in Tests

### Community 145 - "Performance and Optimization guide"
Cohesion: 0.20
Nodes (9): Performance and Optimization guide, Batch tiers (sequential vs chunked Process/ThreadPoolExecutor), Cache warming (warm_up_cache, warm_cache_for_compounds), Energy arrays instead of loops, batch command, Batch CSV input (formula,density,energy), JIT warm-up on first call, Multi-level caching strategy (preloaded, LRU, disk) (+1 more)

### Community 149 - "TestGuiErrorHandling"
Cohesion: 0.22
Nodes (3): numpy_backend_after(), TestGuiErrorHandling, task()

### Community 158 - "atomic_cache.py"
Cohesion: 0.17
Nodes (7): _collect_compound_elements(), _collect_family_compound_elements(), _collect_similar_compound_elements(), _warm_atomic_data(), _warm_bulk_combinations(), warm_cache_for_compounds(), _warm_interpolators()

### Community 160 - "Getting Started guide"
Cohesion: 0.25
Nodes (8): JAX/jaxlib upper bound removed (>=0.8.0), Characterization (golden-value) tests, 202 assertions, Getting Started guide, Modular architecture overview (backend, calculators, data_handling, interfaces, io, validation, gui), Installation (PyPI, GPU extras gpu_cuda12/13, uv dev install), Use cases (mirror design, beamline planning), Optional GPU acceleration (CUDA 11/12, TPU), Key features (performance, physics, architecture, CLI, data handling)

### Community 162 - "XRayLabTool Documentation index"
Cohesion: 0.25
Nodes (9): Contributing Guide, Code standards (PEP 8, 88 cols, NumPy docstrings, type hints), Performance requirements (<0.1 ms single, >100k calc/s batch, >90% cache hit), Review process (automated checks, reviewer, maintainer, squash merge), Claimed speedups (7.5x single, 10x batch, 15x GPU), XRayLabTool Documentation index, Citation (BibTeX, Wei Chen, v0.4.6), Performance highlights table (+1 more)

### Community 166 - "data_export.py"
Cohesion: 0.28
Nodes (3): _format_as_csv(), _format_as_json(), _format_as_table()

### Community 169 - "pyXRayLabTool Logo (Main SVG)"
Cohesion: 0.67
Nodes (3): pyXRayLabTool Logo (Dark), pyXRayLabTool Logo (Light JPG), pyXRayLabTool Logo (Main SVG)

### Community 171 - "Decision: replace matplotlib with PyQtGraph for interactive plots"
Cohesion: 0.25
Nodes (7): ColorPalette dataclass (to_qpalette), F1F2Plot widget, MultiF1F2Plot widget, PlotCanvas widget, PlotWidget protocol (clear, plot_single, plot_multi, update_theme, set_scales), export_publication_plot (matplotlib vector export), PyQtGraphPlotCanvas / PyQtGraphF1F2Plot / PyQtGraphMultiF1F2Plot

### Community 189 - "Interfaces Module API Doc"
Cohesion: 0.29
Nodes (7): Interfaces Module API Doc, CLI error handling messages, CLI modules (parser, formatting, utils), completion_v2 virtual-environment-centric shell completion, Supported shells (bash, zsh, fish, PowerShell), FormulaError, Harden completion/installer module (SEC-001, SEC-003..006)

### Community 190 - "Testing Guide"
Cohesion: 0.33
Nodes (7): Platform and Python version compatibility matrix, Testing Guide, CI test matrix (Python 3.12-3.14, Ubuntu/macOS/Windows), Performance tests and benchmarks, Test categories (unit, integration, performance, characterization, physics), Kramers-Kronig relations between f' and f'', Physics validation tests (Kramers-Kronig, sum rules, literature values)

### Community 207 - "gc"
Cohesion: 0.29
Nodes (3): clean_cache(), enforce_test_isolation(), setup_test_environment()

### Community 229 - "subprocess"
Cohesion: 0.19
Nodes (7): test_headless_gui_smoke_runs_offscreen(), check_gpu_availability(), check_plugin_conflicts(), get_device_info(), get_gpu_info(), get_recommended_package(), get_system_cuda_version()

## Ambiguous Edges - Review These
- `XRayResult as numpy public API boundary (bulk device-to-host in __post_init__)` → `Layered architecture (API, calculation, data, XRayResult)`  [AMBIGUOUS]
  docs/architecture/004-adr-host-device-transfers.md · relation: conceptually_related_to
- `Backend abstraction layer (NumPy/JAX, InterpolationFactory)` → `Interpolation methods (linear, logarithmic, spline)`  [AMBIGUOUS]
  docs/physics/atomic_data.rst · relation: conceptually_related_to
- `Backend auto-selection (JAX only with NVIDIA GPU, else NumPy)` → `Claimed speedups (7.5x single, 10x batch, 15x GPU)`  [AMBIGUOUS]
  docs/development/performance.rst · relation: conceptually_related_to
- `XRayResult (per-energy NumPy array fields)` → `Results as jax.Array vs numpy.ndarray`  [AMBIGUOUS]
  docs/guides/getting_started.rst · relation: conceptually_related_to

## Knowledge Gaps
- **66 isolated node(s):** `Scripts README`, `pyXRayLabTool Logo (Dark)`, `pyXRayLabTool Logo (Light JPG)`, `Untested Modules (compound_analysis, array_ops, optimization)`, `Test Coverage Audit` (+61 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1806 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **134 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `calculate_single_material_properties()` connect `Energy Input & CLI Compatibility Tests` to `Deep-Debug Regression Tests`, `timer`, `CLI Edge-Case Tests`, `test_golden_pipeline.py`, `Core Calculation Pipeline`, `BatchConfig`, `Performance Benchmarks`, `Integration Tests: Properties`, `Batch Processing`, `GUI Services & Window`, `Backend Switching & Dispatch`, `XRayResult Dataclass`, `Test Utilities & Fixtures`, `Optimization Performance Tests`, `MaterialComparator`, `data_handling/__init__.py`, `process_single_calculation`, `TestCalculationSpeedBenchmarks`, `TestEdgeCaseInputValues`, `TestBasicExportFunctionality`, `clear_scattering_factor_cache`, `cmd_calc`, `CLI Batch Command`, `CSV & JSON Export`, `calculate_attenuation_length`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `MainWindowProtocol` (e.g. with `MultiTabMixin` and `Toast`) actually correct?**
  _`MainWindowProtocol` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Scripts README`, `pyXRayLabTool Logo (Dark)`, `pyXRayLabTool Logo (Light JPG)` to the rest of the system?**
  _66 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Utilities & Formula Parsing` be split into smaller, more focused modules?**
  _Cohesion score 0.048513302034428794 - nodes in this community are weakly interconnected._
- **Why does `CompletionInstaller` connect `Completion Installer` to `legacy_install_completion_main`, `TestCompletionIntegrationFlow`, `Completion Environment Detection`, `Completion Installer Tests`, `.uninstall`, `Environment Info Types`, `Completion Integration Tests`, `.install`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `XRayResult as numpy public API boundary (bulk device-to-host in __post_init__)` and `Layered architecture (API, calculation, data, XRayResult)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Backend abstraction layer (NumPy/JAX, InterpolationFactory)` and `Interpolation methods (linear, logarithmic, spline)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
