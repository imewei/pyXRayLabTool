# Graph Report - pyXRayLabTool  (2026-10-23)

## Corpus Check
- 172 files · ~121,106 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 3934 nodes · 7199 edges · 240 communities (100 shown, 140 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 512 edges (avg confidence: 0.87)
- Token cost: 92,353 input · 0 output

## Community Hubs (Navigation)
- test_utils.py
- FormulaError
- cmd_bragg
- tests/conftest.py
- Path
- Completion Environment Detection
- Completion Installer
- parse_formula
- Array Backend Abstraction
- timer
- test_cli_edge_cases.py
- environment.py
- _validate_single_material_inputs
- energy_to_wavelength
- main
- Result Formatting (CLI)
- MemoryMonitor
- Environment Info Types
- validate_chemical_formula
- BatchConfig
- constants.py
- Performance Benchmarks
- calculate_xray_properties
- Code Quality Tests
- Batch Processing
- GUI Services & Window
- test_completion_environment.py
- Completion Integration Tests
- create_scattering_factor_interpolators
- Data Export Formatting Tests
- _validate_and_process_energies
- CI/CD Integration Tests
- Energy Input & CLI Compatibility Tests
- TestEnergySweepSi
- parser.py
- typing_extensions.py
- test_utils.py
- _calc
- GUI Widget Validation Tests
- PerformanceBenchmark
- test_utilities.py
- TestGoldenAu
- test_optimization_validation.py
- Any
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
- xraylabtool/__init__.py
- legacy_install_completion_main
- test_base.py
- core.py
- calculate_transmission
- validate_density
- MemoryProfiler
- Calculation Methods
- TestEnergyParsing
- Completion Installer Tests
- get_atomic_number
- _make_cubic
- TestXRayResultTyping
- test_file_operations.py
- ValidationError
- XRayLabTool Changelog
- export_to_csv
- main
- calculate_critical_angle
- clear_scattering_factor_cache
- test_performance_benchmarks.py
- cmd_calc
- validate_energy_range
- typing
- calculators.core (calculate_single_material_properties, calculate_xray_properties)
- Migration Guide v0.3.0 to v0.4.0
- TestGoldenSi
- TestGoldenFe2O3
- TestGoldenSiO2
- CLI Batch Command
- TestPhysicalRealism
- TestCIWorkflowSimulation
- StyleGuideValidator
- scattering_data.py
- TestExtremeConditions
- validate_style_guide.py
- ADR-003: float64 Preservation Policy
- test_base.py
- SharedHelpersMixin
- `test_docs.py` - Documentation Testing
- TestFundamentalConstants
- ndarray
- Toast
- validate_calculation_parameters
- formula.py
- test_compound_analysis.py
- TestInstallCompletionCommand
- test_integration.py
- compare.py
- file_operations.py
- Data Handling Module API Doc
- calculate_attenuation_length
- CLI Reference
- TestBenchmarkComparison
- X-ray Optics Fundamentals
- energy_to_wavelength_angstrom
- ADR-001: JAX vs NumPy Computation Backend
- TestGoldenScatteringLengthDensity
- BashCompletionGenerator
- TestBatchConfig
- get_compound_family
- parse_chemical_formula
- fixture
- Deep-Debug Regression Tests
- TestEdgeCasesAndErrorHandling
- JAX Architecture & Design (v0.4.0)
- validate_calculation_parameters
- ADR-002: PyQtGraph vs Matplotlib for GUI Plotting
- TestMultiMaterialEquivalence
- Deep-debug regression tests
- RegressionTracker
- FishCompletionGenerator
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
- get_gpu_info
- TestUnitConversionConstants
- performance/conftest.py
- compound_analysis.py
- BatchProcessingError
- BaseXRayLabToolTest
- save_calculation_results
- large_batch_data
- atomic_cache.py
- XRayResult Dataclass
- TestMemoryLeakPrevention
- BaseIntegrationTest
- cache.py
- TestBatchProcessing
- bragg_angle
- TestCompletionIntegrationFlow
- TestDataFactory
- set_backend
- pyXRayLabTool Logo (Main SVG)
- wrapper
- Decision: replace matplotlib with PyQtGraph for interactive plots
- check_gpu_availability
- TestH2OProperties
- test_type_safety_config.py
- numpy
- TestDerivedConstants
- .results
- Analysis API
- Backend API
- v0.4.6 release (2026-09-17)
- XRayLabToolError (base exception)
- XRayLabTool Documentation index
- get_system_info
- IO Operations API
- test_cli.py
- Examples Index
- MaterialInputForm
- pyXRayLabTool Favicon
- GUI Main Window Screenshot
- uninstall_completion_main
- PowerShellCompletionGenerator
- CompletionGenerator
- TestDtypePreservation
- TestEdgeCaseInputValues
- TestMathematicalConstants
- test_numerical_stability.py
- gc
- batch_config_small
- cpu_count
- performance_timer
- _auto_select_backend
- temp_directory
- pip-audit dependency vulnerability gate
- test_enhanced_xray_result_typing.py
- calculate_multiple_xray_properties
- MemoryTracker
- test_data_generator
- get_bulk_atomic_data
- assert_calculation_result_valid
- mock_low_memory
- skip_if_low_memory
- TestCLIFileOperations
- material_and_density
- test_headless_gui_smoke_runs_offscreen
- TestCLICompatibility
- test_energies_medium
- Sample Materials Test Data
- device.py
- TestScalabilityBenchmarks
- Decision
- TestSiO2Properties
- Decision: enforce float64 globally; float32 is a correctness bug
- TestBackwardCompatibility
- test_core.py
- FastXRayCalculationEngine

## God Nodes (most connected - your core abstractions)
1. `calculate_single_material_properties()` - 88 edges
2. `MainWindowProtocol` - 85 edges
3. `CompletionInstaller` - 69 edges
4. `parse_formula()` - 60 edges
5. `BatchConfig` - 44 edges
6. `XRayResult` - 42 edges
7. `calculate_xray_properties()` - 41 edges
8. `EnvironmentInfo` - 35 edges
9. `clear_scattering_factor_cache()` - 35 edges
10. `EnergyError` - 33 edges

## Surprising Connections (you probably didn't know these)
- `wrapper()` --calls--> `clear_scattering_factor_cache()`  [INFERRED]
  tests/fixtures/test_utilities.py → xraylabtool/calculators/cache.py
- `TestExceptionHierarchy` --uses--> `BatchProcessingError`  [INFERRED]
  tests/unit/test_exceptions.py → xraylabtool/exceptions.py
- `TestExceptionHierarchy` --uses--> `DataFileError`  [INFERRED]
  tests/unit/test_exceptions.py → xraylabtool/exceptions.py
- `TestExceptionHierarchy` --uses--> `EnergyError`  [INFERRED]
  tests/unit/test_exceptions.py → xraylabtool/exceptions.py
- `TestExceptionHierarchy` --uses--> `ValidationError`  [INFERRED]
  tests/unit/test_exceptions.py → xraylabtool/exceptions.py

## Import Cycles
- 4-file cycle: `xraylabtool/__init__.py -> xraylabtool/calculators/__init__.py -> xraylabtool/calculators/core.py -> xraylabtool/logging_utils.py -> xraylabtool/__init__.py`

## Hyperedges (group relationships)
- **Defects covered by deep-debug regression tests** — docs_development_changelog_deep_debug_regression_tests, docs_development_changelog_energy_input_contract, docs_development_changelog_anomalous_dispersion, docs_development_changelog_export_fixes, docs_development_changelog_batch_api_fixes, docs_development_changelog_parse_formula [EXTRACTED 1.00]
- **Scientific background documentation set** — docs_physics_xray_optics, docs_physics_atomic_data, docs_physics_calculations [EXTRACTED 1.00]
- **Atomic factors to refractive index to derived optical quantities** — docs_physics_atomic_data_scattering_factors, docs_physics_calculations_number_density, docs_physics_calculations_delta_beta_sum, docs_physics_calculations_critical_angle, docs_physics_calculations_attenuation [EXTRACTED 1.00]
- **Development quality scripts** — scripts_test_docs, scripts_run_type_check, scripts_validate_style_guide [EXTRACTED 1.00]
- **XRayLabTool exception hierarchy rooted at XRayLabToolError** — docs_api_validation_xraylabtoolerror, docs_api_validation_validationerror, docs_api_validation_formulaerror, docs_api_validation_energyerror, docs_api_validation_calculationerror [EXTRACTED 1.00]
- **NumPy/JAX backend abstraction stack** — docs_development_changelog_backend_abstraction, docs_development_changelog_set_backend, docs_development_changelog_interpolation_factory, docs_development_changelog_jax_parity_tests [INFERRED 0.85]
- **JAX migration architecture decisions (float64, host-device transfers, JAX design)** — docs_architecture_003_adr_dtype_policy_decision, docs_architecture_004_adr_host_device_transfers_decision, docs_architecture_jax_architecture_design_philosophy, docs_architecture_jax_architecture_jit [INFERRED 0.85]
- **JAX Architecture Decision Records** — docs_architecture_001_adr_jax_vs_numpy_jax_decision, architecture_004_adr_host_device_transfers_transfer_policy [INFERRED 0.85]
- **v0.4.0 JAX migration, rollback and release documentation** — docs_guides_migration_guide_v0_4, docs_development_rollback_procedures, docs_development_changelog_v0_4_0 [INFERRED 0.85]

## Communities (240 total, 140 thin omitted)

### Community 0 - "test_utils.py"
Cohesion: 0.16
Nodes (3): TestCrystallographicCalculations, angle_from_q(), q_from_angle()

### Community 1 - "FormulaError"
Cohesion: 0.05
Nodes (12): TestAtomicDataError, TestCalculationError, TestExceptionHierarchy, TestExceptionUsage, TestFormulaError, TestUnknownElementError, AtomicDataError, CalculationError (+4 more)

### Community 3 - "tests/conftest.py"
Cohesion: 0.10
Nodes (9): assert_memory_usage_reasonable(), assert_performance_within_threshold(), benchmark_baseline(), energy_arrays(), pytest_addoption(), pytest_collection_modifyitems(), pytest_configure(), sample_batch_data() (+1 more)

### Community 6 - "Completion Installer"
Cohesion: 0.08
Nodes (3): TestSentinelMarkers, CompletionInstaller, test_completion_installation()

### Community 7 - "parse_formula"
Cohesion: 0.06
Nodes (10): TestDelegatingWrappers, TestParseFormulaBasic, TestParseFormulaDecimal, TestParseFormulaErrors, TestParseFormulaParentheses, TestRegexCompatibility, _aggregate_element_counts(), parse_formula() (+2 more)

### Community 8 - "Array Backend Abstraction"
Cohesion: 0.10
Nodes (3): ArrayBackend, JaxBackend, _configure_jax_float64()

### Community 10 - "test_cli_edge_cases.py"
Cohesion: 0.06
Nodes (5): TestBatchRowResilience, TestCLIErrorHandling, TestCLIInputValidation, TestCLIPerformanceAndMemory, TestFieldNameValidation

### Community 12 - "_validate_single_material_inputs"
Cohesion: 0.16
Nodes (5): _create_process_formula_function(), process_formula(), _require_finite_energies(), _restore_energy_order(), _validate_single_material_inputs()

### Community 13 - "energy_to_wavelength"
Cohesion: 0.13
Nodes (6): TestUnitConversions, energy_to_wavelength(), get_electron_charge(), get_planck_constant(), get_speed_of_light(), wavelength_to_energy()

### Community 14 - "main"
Cohesion: 0.09
Nodes (6): TestCLIIntegration, TestMainFunction, cmd_completion(), cmd_install_completion(), cmd_uninstall_completion(), main()

### Community 15 - "Result Formatting (CLI)"
Cohesion: 0.07
Nodes (15): TestResultFormatting, _format_array_field_single(), _format_as_csv(), _format_as_json(), _format_filtered_table(), _format_material_properties(), _format_multiple_energies(), _format_multiple_energy_section() (+7 more)

### Community 16 - "MemoryMonitor"
Cohesion: 0.09
Nodes (4): TestBatchProcessingMemoryManagement, TestCacheManagement, TestMemoryMonitor, MemoryMonitor

### Community 18 - "validate_chemical_formula"
Cohesion: 0.09
Nodes (3): TestValidateChemicalFormula, _get_valid_element_symbols(), validate_chemical_formula()

### Community 19 - "BatchConfig"
Cohesion: 0.12
Nodes (5): TestBatchProcessorEquivalence, TestBatchInputValidation, TestBatchPartialFailure, BatchConfig, calculate_batch_properties()

### Community 20 - "constants.py"
Cohesion: 0.14
Nodes (5): attenuation_length_cm(), critical_angle_degrees(), _isclose(), validate_constants(), wavelength_angstrom_to_energy()

### Community 21 - "Performance Benchmarks"
Cohesion: 0.14
Nodes (5): TestPerformanceBenchmarks, energy_sweep(), complex_formula(), multi_calculation(), single_calculation()

### Community 22 - "calculate_xray_properties"
Cohesion: 0.14
Nodes (3): TestInputValidationAndErrorHandling, calculate_xray_properties(), _validate_xray_inputs()

### Community 24 - "Batch Processing"
Cohesion: 0.12
Nodes (10): chunk_iterator(), _filter_dataframe_fields(), _get_energy_point_data(), _initialize_progress_bar(), _prepare_energies_array(), _prepare_result_data(), process_batch_chunk(), _process_chunks() (+2 more)

### Community 25 - "GUI Services & Window"
Cohesion: 0.11
Nodes (12): _run_smoke(), test_compute_multiple_reports_progress_to_100(), _ensure_app(), test_export_csv_handles_io_error(), test_gui_smoke_offscreen(), test_gui_smoke_threaded_offscreen(), test_duplicate_formula_different_density_not_dropped(), suppress_qt_noise() (+4 more)

### Community 26 - "test_completion_environment.py"
Cohesion: 0.05
Nodes (5): TestEnvironmentDetection, TestEnvironmentIntegration, TestEnvironmentProperties, TestEnvironmentType, EnvironmentType

### Community 27 - "Completion Integration Tests"
Cohesion: 0.12
Nodes (6): TestInstallCompletionMain, completion_main(), create_completion_parser(), handle_completion_command(), install_completion_main(), handle_new_completion_command()

### Community 28 - "create_scattering_factor_interpolators"
Cohesion: 0.08
Nodes (7): _check_element(), cold_cache(), test_si_o_au_interpolators_are_independent(), TestPchipAu, TestPchipO, TestPchipSi, create_scattering_factor_interpolators()

### Community 29 - "Data Export Formatting Tests"
Cohesion: 0.05
Nodes (8): MockXRayResult, TestFormatCalculationSummary, TestFormatXRayResult, _format_as_csv(), _format_as_json(), _format_as_table(), format_calculation_summary(), format_xray_result()

### Community 30 - "_validate_and_process_energies"
Cohesion: 0.12
Nodes (5): TestJaxBackendParity, TestScalarEnergyFallback, _coerce_energy_array(), _convert_energy_input(), _validate_and_process_energies()

### Community 32 - "Energy Input & CLI Compatibility Tests"
Cohesion: 0.10
Nodes (4): TestCLIOutputFormatting, TestSubRefracErrorHandling, TestEnergyInputContract, calculate_single_material_properties()

### Community 34 - "parser.py"
Cohesion: 0.12
Nodes (12): add_atomic_command(), add_batch_command(), add_bragg_command(), add_calc_command(), add_compare_command(), add_completion_command(), add_convert_command(), add_formula_command() (+4 more)

### Community 35 - "typing_extensions.py"
Cohesion: 0.10
Nodes (9): ensure_complex128_array(), ensure_float64_array(), get_optimal_chunk_size(), is_complex_array(), is_energy_array(), is_real_array(), optimize_array_memory_layout(), validate_complex_array() (+1 more)

### Community 37 - "_calc"
Cohesion: 0.09
Nodes (7): _calc(), _clear_cache(), TestComplexFormula, TestElectronCountGolden, TestExactCapturedValues, TestMolecularWeightGolden, _calculate_molecular_properties()

### Community 38 - "GUI Widget Validation Tests"
Cohesion: 0.09
Nodes (6): form(), _get_app(), qt_app(), TestDensityValidation, TestEnergyGridValidation, TestFormulaValidation

### Community 40 - "test_utilities.py"
Cohesion: 0.10
Nodes (10): assert_values_close(), clean_cache_around(), wrapper(), expect_no_warnings(), expect_warnings(), skip_if_insufficient_cpu(), skip_if_insufficient_memory(), stability_test() (+2 more)

### Community 42 - "test_optimization_validation.py"
Cohesion: 0.08
Nodes (8): BasePerformanceTest, TestArrayOptimization, TestBulkAtomicDataOptimization, TestDeprecationWarningOptimization, TestElementPathOptimization, TestInterpolatorCaching, TestNumericalStabilityOptimizations, TestPerformanceRegression

### Community 45 - "ADR-004: Host-Device Transfer Minimization Strategy"
Cohesion: 0.17
Nodes (11): ADR-004: Host-Device Transfer Minimization Strategy, Appendix: Transfer Cost Reference (CPU Backend), Array Sizes in This Workload, Consequences, Context, Monitoring, Negative, Positive (+3 more)

### Community 46 - "Utilities Module"
Cohesion: 0.10
Nodes (12): angle_from_q(), bragg_angle(), d_spacing_cubic(), xraylabtool.device, energy_to_wavelength(), get_atomic_data(), load_atomic_data(), xraylabtool.logging_utils (+4 more)

### Community 49 - "MaterialComparator"
Cohesion: 0.11
Nodes (3): TestBasicAnalysis, compare_materials(), find_absorption_edges()

### Community 51 - "data_handling/__init__.py"
Cohesion: 0.16
Nodes (7): get_cache_stats(), is_element_preloaded(), warm_up_cache(), load_batch_input(), save_batch_results(), _save_dataframe(), __getattr__()

### Community 52 - "GUI Plotting & Palette"
Cohesion: 0.07
Nodes (4): current_palette(), PlotCanvas, F1F2Plot, MultiF1F2Plot

### Community 53 - "test_docs.py"
Cohesion: 0.16
Nodes (13): Pre-commit and CI integration, Sphinx doctest builder, check_accessibility(), check_documentation_coverage(), Colors, main(), print_section(), print_status() (+5 more)

### Community 55 - "single_tab.py"
Cohesion: 0.08
Nodes (13): setup(), test_configure_logging_writes_file(), test_get_logger_child_inherits_config(), enable_offscreen_quiet_env(), build_parser(), main(), _bool_env(), _coerce_int_env() (+5 more)

### Community 56 - "CalculationWorker"
Cohesion: 0.13
Nodes (10): qt_app(), _run_worker(), test_worker_does_not_emit_finished_on_exception(), test_worker_emits_error_on_exception(), bad_fn(), test_worker_emits_result_on_success(), test_worker_injects_progress_cb_when_accepted(), test_worker_passes_kwargs() (+2 more)

### Community 58 - "process_single_calculation"
Cohesion: 0.19
Nodes (3): no_progress_config(), TestProcessSingleCalculation, process_single_calculation()

### Community 59 - "TestCalculationSpeedBenchmarks"
Cohesion: 0.12
Nodes (3): TestCalculationSpeedBenchmarks, calculation_worker(), TestPerformanceRegressionDetection

### Community 60 - "API Reference Index"
Cohesion: 0.24
Nodes (10): Constants Module API Doc, CODATA 2018 standards, Unit conversion factors (angstrom, cm, keV, rad), ENERGY_TO_WAVELENGTH_FACTOR (hc/e / 1000), Physical constants table (PLANCK, SPEED_OF_LIGHT, AVOGADRO, THOMPSON, ...), API Reference Index, XRayLabTool sub-package layout (calculators, data_handling, interfaces, validation, io, analysis, gui, backend, utils), utils helpers (parse_formula, energy_to_wavelength, wavelength_to_energy) (+2 more)

### Community 61 - "NumPy Backend & Tests"
Cohesion: 0.10
Nodes (3): TestEdgeCases, TestNumpyBackend, NumpyBackend

### Community 62 - "MainWindowProtocol"
Cohesion: 0.05
Nodes (4): MultiTabMixin, MainWindowProtocol, SingleTabMixin, MaterialTable

### Community 63 - "ColorPalette"
Cohesion: 0.10
Nodes (6): apply_pyqtgraph_theme(), apply_styles(), apply_theme(), ColorPalette, get_qss(), ThemeManager

### Community 64 - "xraylabtool/__init__.py"
Cohesion: 0.10
Nodes (7): TestAtomicDataLookup, worker(), __getattr__(), _convert_atomic_number_to_int(), get_atomic_data(), get_atomic_number(), get_atomic_weight()

### Community 66 - "test_base.py"
Cohesion: 0.12
Nodes (3): BaseUnitTest, generate_test_materials(), TestStyleCompliance

### Community 67 - "core.py"
Cohesion: 0.10
Nodes (10): _calculate_single_material_xray_properties(), _MaterialPropertiesDict, _prepare_element_data(), calculate_derived_quantities(), calculate_scattering_factors(), _derived_quantities_kernel(), _get_derived_kernel(), _get_scattering_kernel() (+2 more)

### Community 70 - "MemoryProfiler"
Cohesion: 0.15
Nodes (3): decorator(), wrapper(), MemoryProfiler

### Community 71 - "Calculation Methods"
Cohesion: 0.15
Nodes (19): Contributing atomic data (citation, range, precision, format), Energy grid design (log spacing, dense near absorption edges), Atomic Scattering Factor Data, Absorption edges (K, L, M) and XANES features, CXRO database, Henke et al. 1993, Atomic Data and Nuclear Data Tables 54, 181-342, Henke tables (Z=1-92, nff format), Interpolation methods (linear, logarithmic, spline) (+11 more)

### Community 73 - "Completion Installer Tests"
Cohesion: 0.08
Nodes (6): TestCompletionInstallToTempDir, TestCompletionScriptGeneration, show_completion_info(), CompletionManager, get_global_options(), get_xraylabtool_commands()

### Community 74 - "get_atomic_number"
Cohesion: 0.09
Nodes (11): clear_lru_caches(), _mock_get_element(), TestCacheResultConsistency, TestConcurrentGetAtomicNumber, worker(), cache_clearer(), reader(), worker() (+3 more)

### Community 75 - "_make_cubic"
Cohesion: 0.06
Nodes (6): TestCrystalStructure, _make_cubic(), _phase(), TestCrystalStructureAPI, TestStructureFactorPhysics, CrystalStructure

### Community 77 - "test_file_operations.py"
Cohesion: 0.13
Nodes (15): TestDataFileError, test_load_absolute_path(), test_load_csv_file(), test_load_csv_space_separated_fallback(), test_load_empty_nff_raises_data_file_error(), test_load_malformed_nff_raises_data_file_error(), test_load_nff_energy_column_monotone(), test_load_nff_values_are_finite() (+7 more)

### Community 79 - "XRayLabTool Changelog"
Cohesion: 0.12
Nodes (20): XRayLabTool Changelog, Backend abstraction layer (NumPy/JAX), Golden-value characterization tests, Platform and Python compatibility matrix, GUI theme engine (light/dark palettes), InterpolationFactory (scipy to interpax PCHIP), JAX-vs-NumPy parity tests, Keep a Changelog format (+12 more)

### Community 80 - "export_to_csv"
Cohesion: 0.07
Nodes (5): TestBasicExportIntegration, TestBasicExportFunctionality, export_to_csv(), export_to_json(), __getattr__()

### Community 81 - "main"
Cohesion: 0.19
Nodes (9): mypy clean on 2.4.0 and 1.20.0, ruff 0.17.2 pin, MyPy static type analysis, analyze_mypy_output(), check_core_modules(), check_type_definitions(), main(), run_mypy_check() (+1 more)

### Community 85 - "cmd_calc"
Cohesion: 0.13
Nodes (6): TestCalcCommand, cmd_calc(), _determine_output_format(), _print_calc_verbose_info(), _save_or_print_output(), _validate_calc_inputs()

### Community 87 - "typing"
Cohesion: 0.11
Nodes (3): TableFormatter, apply_palette_to_item(), apply_palette_to_widget()

### Community 88 - "calculators.core (calculate_single_material_properties, calculate_xray_properties)"
Cohesion: 0.20
Nodes (11): Calculators Module API Doc, Linear absorption coefficient mu = 4 pi beta / lambda, calculators.core (calculate_single_material_properties, calculate_xray_properties), Critical angle theta_c = sqrt(2 delta), Legacy CamelCase accessors (DeprecationWarning), X-ray complex refractive index n = 1 - delta - i beta, calculators submodules (derived_quantities, xray_result, cache, scattering_data, kernels), XRayResult fields table (+3 more)

### Community 89 - "Migration Guide v0.3.0 to v0.4.0"
Cohesion: 0.29
Nodes (8): Rollback: v0.4.0 to v0.3.0, Quick rollback procedure (pip uninstall, install 0.3.0, verify), Rollback criteria (results differ >1e-12, GUI fails, JAX import errors), Migration Guide v0.3.0 to v0.4.0, Computation stack change: NumPy eager to JAX JIT, Results as jax.Array vs numpy.ndarray, Rollback to v0.3.0 (migration guide section), Migration troubleshooting (jax import, slow first call, blank GUI plots, GPU unused)

### Community 93 - "CLI Batch Command"
Cohesion: 0.17
Nodes (7): cmd_batch(), _convert_result_to_dict(), _parse_batch_data(), _process_batch_materials(), _save_batch_results(), _validate_batch_input(), create_batch_progress_tracker()

### Community 97 - "scattering_data.py"
Cohesion: 0.08
Nodes (6): TestAtomicScatteringFactor, AtomicScatteringFactor, _initialize_element_paths(), load_data_file(), load_scattering_factor_data(), ScatteringData

### Community 99 - "validate_style_guide.py"
Cohesion: 0.18
Nodes (4): NumPy-style docstring validation, Colors, main(), ValidationReport

### Community 101 - "ADR-003: float64 Preservation Policy"
Cohesion: 0.15
Nodes (12): ADR-003: float64 Preservation Policy, Appendix: Numerical Precision Requirements by Function, Consequences, Context, Current Codebase float64 Usage, Decision, Enforcement Strategy, JAX float32 Default (+4 more)

### Community 104 - "`test_docs.py` - Documentation Testing"
Cohesion: 0.18
Nodes (11): Available Scripts, Example Output:, Integration:, Maintenance Notes, Output:, Requirements:, `run_type_check.py` - Type Checking, Scripts Directory (+3 more)

### Community 106 - "ndarray"
Cohesion: 0.13
Nodes (3): arrays_close(), assert_arrays_close(), ResultValidator

### Community 109 - "formula.py"
Cohesion: 0.14
Nodes (6): TestFormulaCommand, cmd_formula(), _get_atomic_data(), _output_formula_results(), _print_formula_results(), _process_formula()

### Community 112 - "test_integration.py"
Cohesion: 0.12
Nodes (3): TestDuplicatedFormulasIndependence, TestPropertyConsistency, TestSubRefracSiliconProperties

### Community 113 - "compare.py"
Cohesion: 0.24
Nodes (5): cmd_compare(), _parse_materials(), _result_to_dict(), _write_report(), _write_table()

### Community 114 - "file_operations.py"
Cohesion: 0.23
Nodes (4): _json_numpy_default(), _save_dict_as_csv(), _save_results_as_csv(), _save_results_as_json()

### Community 115 - "Data Handling Module API Doc"
Cohesion: 0.20
Nodes (11): Data Handling Module API Doc, atomic_cache (92-element preload, warm_up_cache, get_cache_stats), batch_processing (BatchConfig, calculate_batch_properties, CSV I/O), compound_analysis (compound families, recommended warming elements), CXRO / NIST / Henke .nff scattering factor data (0.03-30 keV, PCHIP), Interfaces Module API Doc, CLI commands (calc, batch, compare, convert, formula, atomic, bragg, list, completion, install-completion, uninstall-completion), CLI modules (parser, formatting, utils) (+3 more)

### Community 117 - "CLI Reference"
Cohesion: 0.09
Nodes (26): CLI Reference, atomic command, batch command, bragg command, calc command, compare command, completion command (virtual-environment-centric), convert command (energy/wavelength) (+18 more)

### Community 119 - "X-ray Optics Fundamentals"
Cohesion: 0.24
Nodes (13): First calculation: calculate_single_material_properties(formula, keV, density), XRayResult (per-energy NumPy array fields), Attenuation coefficients (mu = 4 pi beta / lambda, mass coefficient, length), Critical angle theta_c = sqrt(2 delta) with unit conversions, X-ray Optics Fundamentals, Attenuation length and linear/mass absorption coefficients, Beer-Lambert law, Complex refractive index n = 1 - delta - i beta (+5 more)

### Community 121 - "ADR-001: JAX vs NumPy Computation Backend"
Cohesion: 0.20
Nodes (9): ADR-001: JAX vs NumPy Computation Backend, Appendix: Functions that MUST NOT be JIT-compiled, Appendix: Functions to JIT-Compile (Priority Order), Consequences, Context, Decision, Negative, Positive (+1 more)

### Community 125 - "get_compound_family"
Cohesion: 0.20
Nodes (3): TestGetCompoundFamily, _classify_family_by_elements(), get_compound_family()

### Community 127 - "fixture"
Cohesion: 0.12
Nodes (8): batch_config_default(), batch_config_performance(), memory_monitor(), memory_tracker(), performance_thresholds(), sample_calculation_result(), test_energies_large(), test_materials()

### Community 128 - "Deep-Debug Regression Tests"
Cohesion: 0.10
Nodes (6): test_whitespace_delimited_loader_works_with_current_pandas(), TestAnomalousDispersion, TestFormulaParsing, TestLazySubmoduleAccess, TestScatteringDataIntegrity, TestUtilityValidation

### Community 130 - "JAX Architecture & Design (v0.4.0)"
Cohesion: 0.27
Nodes (8): vmap-based batch processing (single bulk transfer), JAX Architecture & Design (v0.4.0), Automatic differentiation (future capability), JAX gotchas (Python control flow, non-array inputs), GPU acceleration and device placement, JIT compilation (cost-benefit, warm-up pattern), Shape-based recompilation gotcha, vmap vectorization (future v0.5+ automatic batching)

### Community 132 - "ADR-002: PyQtGraph vs Matplotlib for GUI Plotting"
Cohesion: 0.14
Nodes (13): ADR-002: PyQtGraph vs Matplotlib for GUI Plotting, Appendix: Feature Parity Checklist, Consequences, Context, Decision, Migration Details, Negative, Positive (+5 more)

### Community 134 - "Deep-debug regression tests"
Cohesion: 0.25
Nodes (9): Anomalous dispersion (negative delta) handling, Batch API ordering, timeout and fallback fixes, Deep-debug regression tests, Energy input contract (scalar or 1-D real, EnergyError), CSV/JSON export fixes and full precision, Lazy loader RecursionError fix, f.nff fluorine data correction, parse_formula validation fixes (+1 more)

### Community 138 - "Files"
Cohesion: 0.25
Nodes (7): Files, Maintenance, `performance_baselines.json`, `sample_materials.json`, Test Data Directory, `test_energies.json`, Usage in Tests

### Community 145 - "Performance and Optimization guide"
Cohesion: 0.25
Nodes (7): Performance and Optimization guide, Batch tiers (sequential vs chunked Process/ThreadPoolExecutor), Cache warming (warm_up_cache, warm_cache_for_compounds), Energy arrays instead of loops, JIT warm-up on first call, Multi-level caching strategy (preloaded, LRU, disk), Performance optimizations (LRU cache, precomputed grids, chunked batches)

### Community 149 - "TestGuiErrorHandling"
Cohesion: 0.22
Nodes (3): numpy_backend_after(), TestGuiErrorHandling, task()

### Community 156 - "save_calculation_results"
Cohesion: 0.20
Nodes (5): TestExports, _two_results(), test_save_and_reload_numpy_array(), test_save_json(), save_calculation_results()

### Community 158 - "atomic_cache.py"
Cohesion: 0.17
Nodes (7): _collect_compound_elements(), _collect_family_compound_elements(), _collect_similar_compound_elements(), _warm_atomic_data(), _warm_bulk_combinations(), warm_cache_for_compounds(), _warm_interpolators()

### Community 162 - "cache.py"
Cohesion: 0.21
Nodes (6): get_cached_elements(), is_element_cached(), _smart_cache_warming(), _warm_priority_cache(), _background_cache_warming(), get_bulk_atomic_data_fast()

### Community 167 - "set_backend"
Cohesion: 0.07
Nodes (9): TestBackendSwitching, TestInterpolationFactory, TestOpsProxy, TestBackendState, restore_numpy_backend(), get_backend(), _OpsProxy, set_backend() (+1 more)

### Community 170 - "wrapper"
Cohesion: 0.18
Nodes (6): memory_test(), performance_test(), decorator(), wrapper(), requires_memory(), decorator()

### Community 171 - "Decision: replace matplotlib with PyQtGraph for interactive plots"
Cohesion: 0.20
Nodes (9): ColorPalette dataclass (to_qpalette), F1F2Plot widget, MultiF1F2Plot widget, PlotCanvas widget, PlotWidget protocol (clear, plot_single, plot_multi, update_theme, set_scales), export_publication_plot (matplotlib vector export), PyQtGraphPlotCanvas / PyQtGraphF1F2Plot / PyQtGraphMultiF1F2Plot, interpax device-side interpolation (+1 more)

### Community 181 - "numpy"
Cohesion: 0.12
Nodes (12): test_save_dict_of_arrays_csv(), background_subtraction(), _check_lattice(), d_spacing_cubic(), d_spacing_orthorhombic(), d_spacing_tetragonal(), _expand_parentheses(), find_peaks() (+4 more)

### Community 188 - "v0.4.6 release (2026-09-17)"
Cohesion: 0.40
Nodes (5): CLI package refactor (parser, formatting, commands), completion_v2 shell completion (legacy completion removed), GUI main_window per-tab mixin package, publish.yml manual release workflow, v0.4.6 release (2026-09-17)

### Community 189 - "XRayLabToolError (base exception)"
Cohesion: 0.24
Nodes (11): Exception hierarchy (index overview), CLI error handling messages, CalculationError, EnergyError, FormulaError, validate_calculation_parameters, validate_chemical_formula, validate_density (+3 more)

### Community 190 - "XRayLabTool Documentation index"
Cohesion: 0.14
Nodes (18): Contributing Guide, Code standards (PEP 8, 88 cols, NumPy docstrings, type hints), Performance requirements (<0.1 ms single, >100k calc/s batch, >90% cache hit), Review process (automated checks, reviewer, maintainer, squash merge), Development workflow (branching, ruff, mypy, pytest, make verify), Testing Guide, Characterization (golden-value) tests, 202 assertions, Coverage policy (22% CI gate, Codecov project and patch checks) (+10 more)

### Community 193 - "test_cli.py"
Cohesion: 0.14
Nodes (3): TestAtomicCommand, TestBatchCommand, cmd_atomic()

### Community 207 - "gc"
Cohesion: 0.29
Nodes (3): clean_cache(), enforce_test_isolation(), setup_test_environment()

### Community 215 - "calculate_multiple_xray_properties"
Cohesion: 0.17
Nodes (4): TestRepeatedMaterials, calculate_multiple_xray_properties(), _process_formulas_parallel(), unique_result_key()

### Community 229 - "device.py"
Cohesion: 0.16
Nodes (4): TestGetDeviceInfo, check_plugin_conflicts(), get_device_info(), get_recommended_package()

### Community 233 - "Decision"
Cohesion: 0.33
Nodes (6): Batch Processing Optimization, Decision, Implementation Pattern, Transfer Policy, JIT Compilation Strategy, Architecture Overview

### Community 238 - "Decision: enforce float64 globally; float32 is a correctness bug"
Cohesion: 0.40
Nodes (5): SCATTERING_FACTOR prefactor (THOMPSON*AVOGADRO*1e6/2pi), jax_enable_x64 / JAX_ENABLE_X64 configuration, SCATTERING_FACTOR constant and delta computation example, _verify_float64 import-time check, JAX configuration options (jax_disable_jit, jax_platforms, default float dtype)

## Ambiguous Edges - Review These
- `XRayResult (per-energy NumPy array fields)` → `Results as jax.Array vs numpy.ndarray`  [AMBIGUOUS]
  docs/guides/getting_started.rst · relation: conceptually_related_to
- `Claimed speedups (7.5x single, 10x batch, 15x GPU)` → `Backend auto-selection (JAX only with NVIDIA GPU, else NumPy)`  [AMBIGUOUS]
  docs/development/performance.rst · relation: conceptually_related_to
- `XRayResult as numpy public API boundary (bulk device-to-host in __post_init__)` → `Layered architecture (API, calculation, data, XRayResult)`  [AMBIGUOUS]
  docs/architecture/004-adr-host-device-transfers.md · relation: conceptually_related_to

## Knowledge Gaps
- **78 isolated node(s):** `Numerical precision requirements by quantity and function`, `CLI modules (parser, formatting, utils)`, `CLI output formats (table, csv, json)`, `Supported shells (bash, zsh, fish, PowerShell)`, `atomic command` (+73 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1797 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **140 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `calculate_single_material_properties()` connect `Energy Input & CLI Compatibility Tests` to `Deep-Debug Regression Tests`, `timer`, `test_cli_edge_cases.py`, `test_golden_pipeline.py`, `_validate_single_material_inputs`, `BatchConfig`, `Performance Benchmarks`, `Batch Processing`, `GUI Services & Window`, `save_calculation_results`, `_validate_and_process_energies`, `XRayResult Dataclass`, `cache.py`, `set_backend`, `MaterialComparator`, `.results`, `process_single_calculation`, `TestCalculationSpeedBenchmarks`, `xraylabtool/__init__.py`, `core.py`, `TestEdgeCaseInputValues`, `export_to_csv`, `clear_scattering_factor_cache`, `test_performance_benchmarks.py`, `cmd_calc`, `calculate_multiple_xray_properties`, `CLI Batch Command`, `TestCLIFileOperations`, `TestCLICompatibility`, `TestScalabilityBenchmarks`, `test_integration.py`, `TestBenchmarkComparison`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `MainWindowProtocol` (e.g. with `MultiTabMixin` and `Toast`) actually correct?**
  _`MainWindowProtocol` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Numerical precision requirements by quantity and function`, `CLI modules (parser, formatting, utils)`, `CLI output formats (table, csv, json)` to the rest of the system?**
  _78 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `FormulaError` be split into smaller, more focused modules?**
  _Cohesion score 0.04680365296803653 - nodes in this community are weakly interconnected._
- **Why does `calculate_xray_properties()` connect `calculate_xray_properties` to `Deep-Debug Regression Tests`, `BaseIntegrationTest`, `TestEdgeCasesAndErrorHandling`, `core.py`, `xraylabtool/__init__.py`, `TestEdgeCaseInputValues`, `TestSiO2Properties`, `TestH2OProperties`, `_validate_single_material_inputs`, `test_integration.py`, `Performance Benchmarks`, `calculate_multiple_xray_properties`, `MaterialComparator`, `TestCalculationSpeedBenchmarks`, `save_calculation_results`, `_validate_and_process_energies`, `XRayResult Dataclass`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `XRayResult (per-energy NumPy array fields)` and `Results as jax.Array vs numpy.ndarray`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Claimed speedups (7.5x single, 10x batch, 15x GPU)` and `Backend auto-selection (JAX only with NVIDIA GPU, else NumPy)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._