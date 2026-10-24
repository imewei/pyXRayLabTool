# Graph Report - pyXRayLabTool  (2026-10-24)

## Corpus Check
- 175 files · ~122,635 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 100 file(s) not represented in the graph (top: .nff 93, .ipynb 2, (none) 1)

## Summary
- 3973 nodes · 7316 edges · 239 communities (107 shown, 132 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 516 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- MainWindowProtocol
- numpy
- Data Export Formatting Tests
- MaterialComparator
- Path
- Completion Environment Detection
- set_backend
- Array Backend Abstraction
- _make_cubic
- calculate_xray_properties
- NumPy Backend & Tests
- XRayResult Dataclass
- Completion Integration Tests
- Result Formatting (CLI)
- Completion Installer
- timer
- MemoryMonitor
- Batch Processing
- Completion Installer Tests
- device.py
- GUI Services & Window
- clear_scattering_factor_cache
- BatchConfig
- _validate_single_material_inputs
- single_tab.py
- GUI Plotting & Palette
- get_atomic_number
- FormulaError
- ColorPalette
- Energy Input & CLI Compatibility Tests
- core.py
- test_utilities.py
- CI/CD Integration Tests
- ndarray
- parser.py
- XRayLabTool Documentation index
- test_docs.py
- CLI Batch Command
- TestEnergySweepSi
- TestEnergyParsing
- export_to_csv
- xraylabtool/__init__.py
- FormulaError
- GUI Widget Validation Tests
- create_scattering_factor_interpolators
- test_cli.py
- CalculationWorker
- validate_chemical_formula
- validate_calculation_parameters
- test_cli_edge_cases.py
- main
- cache.py
- test_completion_environment.py
- Environment Info Types
- validate_density
- Utilities Module
- SharedHelpersMixin
- TestFullPipelineSiO2
- test_optimization_validation.py
- cmd_calc
- EnergyError
- TestTypeSafetyConfig
- validate_energy_range
- XRayLabTool Changelog
- _calc
- Deep-Debug Regression Tests
- FormulaError
- test_file_operations.py
- test_utils.py
- GUI Plotting & Palette
- CLI Reference
- formula.py
- constants.py
- calculate_attenuation_length
- PerformanceBenchmark
- parse_formula
- .calculate_derived_quantities
- scattering_data.py
- atomic_cache.py
- Calculation Methods
- typing_extensions.py
- TestInstallCompletionCommand
- legacy_install_completion_main
- Code Quality Tests
- TestXRayResultTyping
- Deep-debug regression tests
- XRayLabTool Documentation index
- main
- calculate_transmission
- tests/conftest.py
- MemoryProfiler
- BaseIntegrationTest
- test_base.py
- scattering_data.py
- FishCompletionGenerator
- X-ray Optics Fundamentals
- XRayLabTool Documentation index
- calculate_critical_angle
- fixture
- cmd_list
- TestCIWorkflowSimulation
- ADR-002: PyQtGraph vs Matplotlib for GUI Plotting
- TestGoldenScatteringLengthDensity
- TestGoldenSi
- TestGoldenAu
- TestGoldenFe2O3
- TestGoldenSiO2
- FormulaError
- Performance Benchmarks
- TestCalculationSpeedBenchmarks
- ValidationError
- TestExtremeConditions
- TestPhysicalRealism
- ADR-003: float64 Preservation Policy
- TestBenchmarkComparison
- test_completion_environment.py
- `test_docs.py` - Documentation Testing
- SharedHelpersMixin
- cmd_convert
- BashCompletionGenerator
- process_single_calculation
- test_optimization_validation.py
- save_calculation_results
- Any
- ADR-004: Host-Device Transfer Minimization Strategy
- test_base.py
- Data Handling Module API Doc
- test_base.py
- TestUnitConversionConstants
- FastAtomicDataProvider
- .uninstall
- file_operations.py
- calculators.core (calculate_single_material_properties, calculate_xray_properties)
- XRayLabToolError (base exception)
- Performance and Optimization guide
- Migration Guide v0.3.0 to v0.4.0
- AtomicDataProvider
- `test_docs.py` - Documentation Testing
- energy_to_wavelength_angstrom
- BaseXRayLabToolTest
- Deep-Debug Regression Tests
- get_gpu_info
- parse_formula
- MaterialInputForm
- CompletionGenerator
- ADR-001: JAX vs NumPy Computation Backend
- API Reference Index
- Decision: replace matplotlib with PyQtGraph for interactive plots
- JAX Architecture & Design (v0.4.0)
- TestCompletionIntegrationFlow
- export_to_csv
- TestBatchConfig
- test_completion_environment.py
- get_compound_family
- parse_chemical_formula
- TestGuiErrorHandling
- FastXRayCalculationEngine
- PowerShellCompletionGenerator
- Toast
- TestMultiMaterialEquivalence
- wrapper
- RegressionTracker
- compound_analysis.py
- TestBatchProcessing
- typing_extensions.py
- test_type_safety_config.py
- Files
- v0.4.6 release (2026-09-17)
- TestDerivedConstants
- TestCLIFileOperations
- test_cli_edge_cases.py
- TestCLICompatibility
- TestEdgeCaseInputValues
- TestMemoryLeakPrevention
- Code Quality Tests
- test_compound_analysis.py
- find_similar_compounds
- get_compound_complexity_score
- get_compound_frequency_score
- get_elements_for_compound
- AtomicDataProvider
- PerformanceMetrics
- Decision
- gc
- TestFundamentalConstants
- performance/conftest.py
- parse_formula
- parse_formula
- _validate_and_process_energies
- Decision: enforce float64 globally; float32 is a correctness bug
- test_cli_edge_cases.py
- test_cli_edge_cases.py
- TestBackwardCompatibility
- create_scattering_factor_interpolators
- test_enhanced_xray_result_typing.py
- parse_formula
- get_compound_family
- TestDtypePreservation
- MemoryTracker
- validate_calculation_parameters
- test_data_generator
- test_optimization_validation.py
- _calc
- _auto_select_backend
- _calc
- mock_low_memory
- skip_if_low_memory
- typing_extensions.py
- pyXRayLabTool Logo (Main SVG)
- batch_config_small
- large_batch_data
- material_and_density
- fixture
- fixture
- fixture
- tests/conftest.py
- fixture
- parse_formula
- parse_formula
- Analysis API
- Backend API
- IO Operations API
- Examples Index
- pyXRayLabTool Favicon
- GUI Main Window Screenshot
- Sample Materials Test Data

## God Nodes (most connected - your core abstractions)
1. `calculate_single_material_properties()` - 88 edges
2. `MainWindowProtocol` - 85 edges
3. `CompletionInstaller` - 69 edges
4. `parse_formula()` - 60 edges
5. `BatchConfig` - 44 edges
6. `XRayResult` - 42 edges
7. `calculate_xray_properties()` - 41 edges
8. `clear_scattering_factor_cache()` - 35 edges
9. `EnvironmentInfo` - 35 edges
10. `JaxBackend` - 33 edges

## Surprising Connections (you probably didn't know these)
- `wrapper()` --calls--> `clear_scattering_factor_cache()`  [INFERRED]
  tests/fixtures/test_utilities.py → xraylabtool/calculators/cache.py
- `_single()` --calls--> `clear_scattering_factor_cache()`  [INFERRED]
  tests/characterization/test_golden_integration.py → xraylabtool/calculators/cache.py
- `cold_cache()` --calls--> `clear_scattering_factor_cache()`  [INFERRED]
  tests/characterization/test_golden_interpolation.py → xraylabtool/calculators/cache.py
- `_calc()` --calls--> `get_bulk_atomic_data()`  [INFERRED]
  tests/characterization/test_golden_molecular.py → xraylabtool/calculators/cache.py
- `_clear_cache()` --calls--> `clear_scattering_factor_cache()`  [INFERRED]
  tests/characterization/test_golden_molecular.py → xraylabtool/calculators/cache.py

## Import Cycles
- 4-file cycle: `xraylabtool/__init__.py -> xraylabtool/calculators/__init__.py -> xraylabtool/calculators/core.py -> xraylabtool/logging_utils.py -> xraylabtool/__init__.py`

## Hyperedges (group relationships)
- **Atomic factors to refractive index to derived optical quantities** — docs_physics_atomic_data_scattering_factors, docs_physics_calculations_number_density, docs_physics_calculations_delta_beta_sum, docs_physics_calculations_critical_angle, docs_physics_calculations_attenuation [EXTRACTED 1.00]
- **XRayLabTool exception hierarchy rooted at XRayLabToolError** — docs_api_validation_xraylabtoolerror, docs_api_validation_validationerror, docs_api_validation_formulaerror, docs_api_validation_energyerror, docs_api_validation_calculationerror [EXTRACTED 1.00]
- **Release and install verification scripts** — scripts_check_version, scripts_test_install, scripts_gpu_check [INFERRED 0.75]
- **JAX migration architecture decisions (float64, host-device transfers, JAX design)** — docs_architecture_003_adr_dtype_policy_decision, docs_architecture_004_adr_host_device_transfers_decision, docs_architecture_jax_architecture_design_philosophy, docs_architecture_jax_architecture_jit [INFERRED 0.85]
- **JAX Architecture Decision Records** — docs_architecture_001_adr_jax_vs_numpy_jax_decision, architecture_004_adr_host_device_transfers_transfer_policy [INFERRED 0.85]
- **v0.4.0 JAX migration, rollback and release documentation** — docs_guides_migration_guide_v0_4, docs_development_rollback_procedures [INFERRED 0.85]
- **Files kept in sync by check_version.py** — scripts_check_version, docs_index_citation, docs_development_changelog_version_0_4_7 [INFERRED 0.85]

## Communities (239 total, 132 thin omitted)

### Community 0 - "MainWindowProtocol"
Cohesion: 0.05
Nodes (4): MultiTabMixin, MainWindowProtocol, SingleTabMixin, MaterialTable

### Community 1 - "numpy"
Cohesion: 0.05
Nodes (20): assert_calculation_result_valid(), TestCrystallographicCalculations, TestUnitConversions, __getattr__(), angle_from_q(), bragg_angle(), _check_lattice(), _convert_atomic_number_to_int() (+12 more)

### Community 2 - "Data Export Formatting Tests"
Cohesion: 0.06
Nodes (8): MockXRayResult, TestFormatCalculationSummary, TestFormatXRayResult, _format_as_csv(), _format_as_json(), _format_as_table(), format_calculation_summary(), format_xray_result()

### Community 3 - "MaterialComparator"
Cohesion: 0.05
Nodes (5): TestBasicAnalysis, ComparisonResult, MaterialComparator, compare_materials(), find_absorption_edges()

### Community 4 - "Path"
Cohesion: 0.06
Nodes (6): NumPy-style docstring validation, Colors, main(), StyleGuideValidator, StyleViolation, ValidationReport

### Community 6 - "set_backend"
Cohesion: 0.06
Nodes (10): TestBackendSwitching, TestInterpolationFactory, TestOpsProxy, TestBackendState, restore_numpy_backend(), TestJaxBackendParity, get_backend(), _OpsProxy (+2 more)

### Community 8 - "_make_cubic"
Cohesion: 0.06
Nodes (6): TestCrystalStructure, _make_cubic(), _phase(), TestCrystalStructureAPI, TestStructureFactorPhysics, CrystalStructure

### Community 9 - "calculate_xray_properties"
Cohesion: 0.05
Nodes (8): TestDuplicatedFormulasIndependence, TestEdgeCasesAndErrorHandling, TestH2OProperties, TestInputValidationAndErrorHandling, TestPropertyConsistency, TestSiO2Properties, calculate_xray_properties(), _validate_xray_inputs()

### Community 10 - "NumPy Backend & Tests"
Cohesion: 0.09
Nodes (3): TestEdgeCases, TestNumpyBackend, NumpyBackend

### Community 12 - "Completion Integration Tests"
Cohesion: 0.07
Nodes (8): TestInstallCompletionMain, TestUninstallCompletionMain, completion_main(), create_completion_parser(), handle_completion_command(), install_completion_main(), uninstall_completion_main(), handle_new_completion_command()

### Community 13 - "Result Formatting (CLI)"
Cohesion: 0.08
Nodes (15): TestResultFormatting, _format_array_field_single(), _format_as_csv(), _format_as_json(), _format_filtered_table(), _format_material_properties(), _format_multiple_energies(), _format_multiple_energy_section() (+7 more)

### Community 14 - "Completion Installer"
Cohesion: 0.08
Nodes (3): TestSentinelMarkers, CompletionInstaller, test_completion_installation()

### Community 15 - "timer"
Cohesion: 0.07
Nodes (5): TestPerformanceBenchmarks, TestPerformanceRegression, TestScalabilityBenchmarks, timer(), get_atomic_data_fast()

### Community 16 - "MemoryMonitor"
Cohesion: 0.08
Nodes (6): TestCacheManagement, TestMemoryMonitor, get_cache_stats(), warm_up_cache(), MemoryMonitor, __getattr__()

### Community 17 - "Batch Processing"
Cohesion: 0.08
Nodes (13): setup(), chunk_iterator(), _filter_dataframe_fields(), _get_energy_point_data(), _initialize_progress_bar(), load_batch_input(), _prepare_result_data(), process_batch_chunk() (+5 more)

### Community 18 - "Completion Installer Tests"
Cohesion: 0.08
Nodes (6): TestCompletionInstallToTempDir, TestCompletionScriptGeneration, show_completion_info(), CompletionManager, get_global_options(), get_xraylabtool_commands()

### Community 19 - "device.py"
Cohesion: 0.08
Nodes (10): test_headless_gui_smoke_runs_offscreen(), TestCheckGpuAvailability, TestGetDeviceInfo, TestGetSystemCudaVersion, check_gpu_availability(), check_plugin_conflicts(), get_device_info(), get_gpu_info() (+2 more)

### Community 20 - "GUI Services & Window"
Cohesion: 0.10
Nodes (13): _run_smoke(), test_compute_multiple_reports_progress_to_100(), _ensure_app(), test_export_csv_handles_io_error(), test_gui_smoke_offscreen(), test_gui_smoke_threaded_offscreen(), test_duplicate_formula_different_density_not_dropped(), enable_offscreen_quiet_env() (+5 more)

### Community 21 - "clear_scattering_factor_cache"
Cohesion: 0.07
Nodes (5): _clear_cache(), cold_cache(), TestGoldenDerivedQuantities, TestGoldenScatteringFactors, clear_scattering_factor_cache()

### Community 22 - "BatchConfig"
Cohesion: 0.11
Nodes (7): _single(), TestBatchProcessorEquivalence, no_progress_config(), TestBatchInputValidation, TestBatchPartialFailure, BatchConfig, calculate_batch_properties()

### Community 23 - "_validate_single_material_inputs"
Cohesion: 0.09
Nodes (13): _calculate_molecular_properties(), calculate_multiple_xray_properties(), _calculate_single_material_xray_properties(), _coerce_energy_array(), _convert_energy_input(), _create_process_formula_function(), process_formula(), _MaterialPropertiesDict (+5 more)

### Community 24 - "single_tab.py"
Cohesion: 0.12
Nodes (12): cpu_count(), test_configure_logging_writes_file(), test_get_logger_child_inherits_config(), build_parser(), main(), _bool_env(), _coerce_int_env(), configure_logging() (+4 more)

### Community 25 - "GUI Plotting & Palette"
Cohesion: 0.12
Nodes (5): apply_palette_to_item(), apply_palette_to_widget(), current_palette(), F1F2Plot, MultiF1F2Plot

### Community 26 - "get_atomic_number"
Cohesion: 0.09
Nodes (11): clear_lru_caches(), _mock_get_element(), TestCacheResultConsistency, TestConcurrentGetAtomicNumber, worker(), cache_clearer(), reader(), worker() (+3 more)

### Community 27 - "FormulaError"
Cohesion: 0.09
Nodes (6): TestAtomicDataError, TestExceptionHierarchy, TestUnknownElementError, AtomicDataError, UnknownElementError, _handle_mendeleev_error()

### Community 28 - "ColorPalette"
Cohesion: 0.10
Nodes (6): apply_pyqtgraph_theme(), apply_styles(), apply_theme(), ColorPalette, get_qss(), ThemeManager

### Community 29 - "Energy Input & CLI Compatibility Tests"
Cohesion: 0.08
Nodes (5): TestCLIOutputFormatting, TestSubRefracErrorHandling, TestSubRefracSiliconProperties, TestEnergyInputContract, calculate_single_material_properties()

### Community 30 - "core.py"
Cohesion: 0.11
Nodes (10): _prepare_element_data(), _process_formulas_parallel(), unique_result_key(), calculate_derived_quantities(), calculate_scattering_factors(), _derived_quantities_kernel(), _get_derived_kernel(), _get_scattering_kernel() (+2 more)

### Community 31 - "test_utilities.py"
Cohesion: 0.09
Nodes (11): assert_values_close(), clean_cache_around(), wrapper(), expect_warnings(), get_system_info(), log_system_info(), skip_if_insufficient_cpu(), skip_if_insufficient_memory() (+3 more)

### Community 33 - "ndarray"
Cohesion: 0.09
Nodes (5): arrays_close(), assert_arrays_close(), create_batch_test_data(), ResultValidator, TestDataFactory

### Community 34 - "parser.py"
Cohesion: 0.12
Nodes (12): add_atomic_command(), add_batch_command(), add_bragg_command(), add_calc_command(), add_compare_command(), add_completion_command(), add_convert_command(), add_formula_command() (+4 more)

### Community 35 - "XRayLabTool Documentation index"
Cohesion: 0.08
Nodes (28): XRayLabTool Documentation (index), api/index, architecture/index, calculate_single_material_properties, Citation (BibTeX, v0.4.7), CXRO/NIST atomic scattering databases, development/changelog, development/contributing (+20 more)

### Community 36 - "test_docs.py"
Cohesion: 0.14
Nodes (14): CI/CD pipelines integration, Pre-commit and CI integration, Sphinx doctest builder, check_accessibility(), check_documentation_coverage(), Colors, main(), print_section() (+6 more)

### Community 37 - "CLI Batch Command"
Cohesion: 0.13
Nodes (7): cmd_batch(), _convert_result_to_dict(), _parse_batch_data(), _process_batch_materials(), _save_batch_results(), _validate_batch_input(), create_batch_progress_tracker()

### Community 39 - "TestEnergyParsing"
Cohesion: 0.10
Nodes (7): TestEnergyParsing, cmd_compare(), _parse_materials(), _result_to_dict(), _write_report(), _write_table(), parse_energy_string()

### Community 40 - "export_to_csv"
Cohesion: 0.10
Nodes (4): TestBasicExportFunctionality, export_to_csv(), export_to_json(), __getattr__()

### Community 41 - "xraylabtool/__init__.py"
Cohesion: 0.11
Nodes (6): TestAtomicDataLookup, worker(), _get_atomic_data(), get_atomic_data(), get_atomic_number(), get_atomic_weight()

### Community 42 - "FormulaError"
Cohesion: 0.12
Nodes (5): TestBatchProcessingError, TestDataFileError, BatchProcessingError, DataFileError, XRayLabToolError

### Community 43 - "GUI Widget Validation Tests"
Cohesion: 0.09
Nodes (6): form(), _get_app(), qt_app(), TestDensityValidation, TestEnergyGridValidation, TestFormulaValidation

### Community 44 - "create_scattering_factor_interpolators"
Cohesion: 0.12
Nodes (7): _check_element(), cold_cache(), test_si_o_au_interpolators_are_independent(), TestPchipAu, TestPchipO, TestPchipSi, create_scattering_factor_interpolators()

### Community 45 - "test_cli.py"
Cohesion: 0.09
Nodes (5): TestAtomicCommand, TestBatchCommand, TestBraggCommand, cmd_atomic(), cmd_bragg()

### Community 46 - "CalculationWorker"
Cohesion: 0.13
Nodes (10): qt_app(), _run_worker(), test_worker_does_not_emit_finished_on_exception(), test_worker_emits_error_on_exception(), bad_fn(), test_worker_emits_result_on_success(), test_worker_injects_progress_cb_when_accepted(), test_worker_passes_kwargs() (+2 more)

### Community 49 - "validate_calculation_parameters"
Cohesion: 0.11
Nodes (3): TestValidateCalculationParameters, _get_valid_element_symbols(), validate_calculation_parameters()

### Community 50 - "test_cli_edge_cases.py"
Cohesion: 0.09
Nodes (3): temp_directory(), TestBatchRowResilience, TestFieldNameValidation

### Community 51 - "main"
Cohesion: 0.11
Nodes (5): TestMainFunction, cmd_completion(), cmd_install_completion(), cmd_uninstall_completion(), main()

### Community 52 - "cache.py"
Cohesion: 0.11
Nodes (8): TestAtomicDataCacheIsImmutable, get_bulk_atomic_data(), get_cached_elements(), is_element_cached(), _smart_cache_warming(), _warm_priority_cache(), _background_cache_warming(), get_bulk_atomic_data_fast()

### Community 53 - "test_completion_environment.py"
Cohesion: 0.09
Nodes (3): TestEnvironmentIntegration, TestEnvironmentType, EnvironmentType

### Community 56 - "Utilities Module"
Cohesion: 0.10
Nodes (12): angle_from_q(), bragg_angle(), d_spacing_cubic(), xraylabtool.device, energy_to_wavelength(), get_atomic_data(), load_atomic_data(), xraylabtool.logging_utils (+4 more)

### Community 59 - "test_optimization_validation.py"
Cohesion: 0.11
Nodes (5): BasePerformanceTest, generate_test_materials(), TestBulkAtomicDataOptimization, TestElementPathOptimization, TestPerformanceRegressionDetection

### Community 60 - "cmd_calc"
Cohesion: 0.13
Nodes (6): TestCalcCommand, cmd_calc(), _determine_output_format(), _print_calc_verbose_info(), _save_or_print_output(), _validate_calc_inputs()

### Community 61 - "EnergyError"
Cohesion: 0.11
Nodes (3): TestEnergyError, TestBoundaryConditions, EnergyError

### Community 64 - "XRayLabTool Changelog"
Cohesion: 0.14
Nodes (21): Changelog (docs), Backend abstraction layer (NumPy/JAX), CHANGELOG.md (v0.2.x history), Characterization tests (202 golden-value), GUI theme engine (light/dark palettes), InterpolationFactory, JAX-vs-NumPy parity tests, Keep a Changelog format (+13 more)

### Community 65 - "_calc"
Cohesion: 0.13
Nodes (4): _calc(), TestComplexFormula, TestElectronCountGolden, TestMolecularWeightGolden

### Community 66 - "Deep-Debug Regression Tests"
Cohesion: 0.11
Nodes (6): test_whitespace_delimited_loader_works_with_current_pandas(), TestAnomalousDispersion, TestLazySubmoduleAccess, TestRepeatedMaterials, TestScatteringDataIntegrity, load_data_file()

### Community 67 - "FormulaError"
Cohesion: 0.12
Nodes (3): TestCalculationError, TestExceptionUsage, CalculationError

### Community 68 - "test_file_operations.py"
Cohesion: 0.18
Nodes (14): expect_no_warnings(), test_load_absolute_path(), test_load_csv_file(), test_load_csv_space_separated_fallback(), test_load_empty_nff_raises_data_file_error(), test_load_malformed_nff_raises_data_file_error(), test_load_nff_energy_column_monotone(), test_load_nff_values_are_finite() (+6 more)

### Community 69 - "test_utils.py"
Cohesion: 0.12
Nodes (5): TestDataProcessing, background_subtraction(), normalize_intensity(), save_processed_data(), smooth_data()

### Community 71 - "CLI Reference"
Cohesion: 0.12
Nodes (20): Energy grid design (log spacing, dense near absorption edges), CLI Reference, atomic command, bragg command, calc command, compare command, completion command (virtual-environment-centric), convert command (energy/wavelength) (+12 more)

### Community 72 - "formula.py"
Cohesion: 0.15
Nodes (5): TestFormulaCommand, cmd_formula(), _output_formula_results(), _print_formula_results(), _process_formula()

### Community 73 - "constants.py"
Cohesion: 0.13
Nodes (6): TestMathematicalConstants, attenuation_length_cm(), critical_angle_degrees(), _isclose(), validate_constants(), wavelength_angstrom_to_energy()

### Community 74 - "calculate_attenuation_length"
Cohesion: 0.15
Nodes (3): TestGoldenAttenuationLength, calculate_attenuation_length(), calculate_scattering_length_density()

### Community 76 - "parse_formula"
Cohesion: 0.16
Nodes (3): TestParseFormulaBasic, _expand_parentheses(), parse_formula()

### Community 78 - "scattering_data.py"
Cohesion: 0.11
Nodes (3): TestAtomicScatteringFactor, TestLoadDataFile, AtomicScatteringFactor

### Community 79 - "atomic_cache.py"
Cohesion: 0.15
Nodes (8): _collect_compound_elements(), _collect_family_compound_elements(), _collect_similar_compound_elements(), is_element_preloaded(), _warm_atomic_data(), _warm_bulk_combinations(), warm_cache_for_compounds(), _warm_interpolators()

### Community 80 - "Calculation Methods"
Cohesion: 0.16
Nodes (18): Contributing atomic data (citation, range, precision, format), Atomic Scattering Factor Data, Absorption edges (K, L, M) and XANES features, CXRO database, Henke et al. 1993, Atomic Data and Nuclear Data Tables 54, 181-342, Henke tables (Z=1-92, nff format), Interpolation methods (linear, logarithmic, spline), Kramers-Kronig relations between f' and f'' (+10 more)

### Community 81 - "typing_extensions.py"
Cohesion: 0.17
Nodes (5): get_optimal_chunk_size(), is_complex_array(), is_energy_array(), is_real_array(), validate_complex_array()

### Community 86 - "Deep-debug regression tests"
Cohesion: 0.13
Nodes (17): Anomalous dispersion (negative delta) handling, Anomalous dispersion (negative delta) fix, Batch API ordering, timeout and fallback fixes, Deep-debug regression tests, Energy input contract (scalar or 1-D real, EnergyError), CSV/JSON export fixes and full precision, Export full-precision and CSV fixes, jax/jaxlib upper bound removal (+9 more)

### Community 87 - "XRayLabTool Documentation index"
Cohesion: 0.13
Nodes (15): Compatibility Matrix, Contributing Guide, Code standards (PEP 8, 88 cols, NumPy docstrings, type hints), Performance requirements (<0.1 ms single, >100k calc/s batch, >90% cache hit), Review process (automated checks, reviewer, maintainer, squash merge), Development workflow (branching, ruff, mypy, pytest, make verify), Testing Guide, Characterization (golden-value) tests, 202 assertions (+7 more)

### Community 88 - "main"
Cohesion: 0.23
Nodes (7): MyPy static type analysis, analyze_mypy_output(), check_core_modules(), check_type_definitions(), main(), run_mypy_check(), validate_cache_performance()

### Community 90 - "tests/conftest.py"
Cohesion: 0.12
Nodes (7): assert_memory_usage_reasonable(), assert_performance_within_threshold(), benchmark_baseline(), performance_thresholds(), pytest_addoption(), pytest_collection_modifyitems(), pytest_configure()

### Community 91 - "MemoryProfiler"
Cohesion: 0.15
Nodes (3): decorator(), wrapper(), MemoryProfiler

### Community 92 - "BaseIntegrationTest"
Cohesion: 0.12
Nodes (3): BaseIntegrationTest, TestCLIIntegration, TestBasicSetupAndInitialization

### Community 94 - "scattering_data.py"
Cohesion: 0.15
Nodes (3): _initialize_element_paths(), load_scattering_factor_data(), ScatteringData

### Community 96 - "X-ray Optics Fundamentals"
Cohesion: 0.21
Nodes (15): Getting Started guide, First calculation: calculate_single_material_properties(formula, keV, density), Use cases (mirror design, beamline planning), XRayResult (per-energy NumPy array fields), Attenuation coefficients (mu = 4 pi beta / lambda, mass coefficient, length), Critical angle theta_c = sqrt(2 delta) with unit conversions, X-ray Optics Fundamentals, Attenuation length and linear/mass absorption coefficients (+7 more)

### Community 97 - "XRayLabTool Documentation index"
Cohesion: 0.15
Nodes (15): Backend abstraction (set_backend), CLI interface, JAX JIT and GPU extras (gpu_cuda12/13), Key Features, Performance claims (150,000+ calc/s), Virtual-environment-centric shell completion, analysis sub-package, backend sub-package (+7 more)

### Community 99 - "fixture"
Cohesion: 0.13
Nodes (7): batch_config_default(), batch_config_performance(), energy_arrays(), memory_monitor(), performance_timer(), sample_batch_data(), test_energies_medium()

### Community 102 - "ADR-002: PyQtGraph vs Matplotlib for GUI Plotting"
Cohesion: 0.14
Nodes (13): ADR-002: PyQtGraph vs Matplotlib for GUI Plotting, Appendix: Feature Parity Checklist, Consequences, Context, Decision, Migration Details, Negative, Positive (+5 more)

### Community 108 - "FormulaError"
Cohesion: 0.18
Nodes (3): TestFormulaError, FormulaError, _configure_jax_float64()

### Community 109 - "Performance Benchmarks"
Cohesion: 0.14
Nodes (5): TestPerformanceBenchmarks, energy_sweep(), complex_formula(), multi_calculation(), single_calculation()

### Community 114 - "ADR-003: float64 Preservation Policy"
Cohesion: 0.15
Nodes (12): ADR-003: float64 Preservation Policy, Appendix: Numerical Precision Requirements by Function, Consequences, Context, Current Codebase float64 Usage, Decision, Enforcement Strategy, JAX float32 Default (+4 more)

### Community 117 - "`test_docs.py` - Documentation Testing"
Cohesion: 0.18
Nodes (5): main(), jax-cuda12/cuda13 plugin conflict check, Makefile targets (version-check, test-install-*, gpu-*), main(), _venv_python()

### Community 121 - "process_single_calculation"
Cohesion: 0.18
Nodes (3): TestBatchProcessingMemoryManagement, TestProcessSingleCalculation, process_single_calculation()

### Community 122 - "test_optimization_validation.py"
Cohesion: 0.15
Nodes (3): TestArrayOptimization, TestDeprecationWarningOptimization, TestPerformanceRegression

### Community 123 - "save_calculation_results"
Cohesion: 0.18
Nodes (6): TestExports, _two_results(), test_save_and_reload_numpy_array(), test_save_dict_of_arrays_csv(), test_save_json(), save_calculation_results()

### Community 125 - "ADR-004: Host-Device Transfer Minimization Strategy"
Cohesion: 0.17
Nodes (11): ADR-004: Host-Device Transfer Minimization Strategy, Appendix: Transfer Cost Reference (CPU Backend), Array Sizes in This Workload, Consequences, Context, Monitoring, Negative, Positive (+3 more)

### Community 126 - "test_base.py"
Cohesion: 0.23
Nodes (4): collect(), main(), _search(), check-version pre-commit hook

### Community 127 - "Data Handling Module API Doc"
Cohesion: 0.20
Nodes (11): Data Handling Module API Doc, atomic_cache (92-element preload, warm_up_cache, get_cache_stats), batch_processing (BatchConfig, calculate_batch_properties, CSV I/O), compound_analysis (compound families, recommended warming elements), CXRO / NIST / Henke .nff scattering factor data (0.03-30 keV, PCHIP), Interfaces Module API Doc, CLI commands (calc, batch, compare, convert, formula, atomic, bragg, list, completion, install-completion, uninstall-completion), CLI modules (parser, formatting, utils) (+3 more)

### Community 132 - "file_operations.py"
Cohesion: 0.24
Nodes (4): _json_numpy_default(), _save_dict_as_csv(), _save_results_as_csv(), _save_results_as_json()

### Community 133 - "calculators.core (calculate_single_material_properties, calculate_xray_properties)"
Cohesion: 0.20
Nodes (11): Calculators Module API Doc, Linear absorption coefficient mu = 4 pi beta / lambda, calculators.core (calculate_single_material_properties, calculate_xray_properties), Critical angle theta_c = sqrt(2 delta), Legacy CamelCase accessors (DeprecationWarning), X-ray complex refractive index n = 1 - delta - i beta, calculators submodules (derived_quantities, xray_result, cache, scattering_data, kernels), XRayResult fields table (+3 more)

### Community 134 - "XRayLabToolError (base exception)"
Cohesion: 0.24
Nodes (11): Exception hierarchy (index overview), CLI error handling messages, CalculationError, EnergyError, FormulaError, validate_calculation_parameters, validate_chemical_formula, validate_density (+3 more)

### Community 135 - "Performance and Optimization guide"
Cohesion: 0.18
Nodes (10): Performance and Optimization guide, Batch tiers (sequential vs chunked Process/ThreadPoolExecutor), Cache warming (warm_up_cache, warm_cache_for_compounds), Energy arrays instead of loops, batch command, Modular architecture overview (backend, calculators, data_handling, interfaces, io, validation, gui), Batch CSV input (formula,density,energy), JIT warm-up on first call (+2 more)

### Community 136 - "Migration Guide v0.3.0 to v0.4.0"
Cohesion: 0.20
Nodes (10): Rollback: v0.4.0 to v0.3.0, Quick rollback procedure (pip uninstall, install 0.3.0, verify), Rollback criteria (results differ >1e-12, GUI fails, JAX import errors), Installation (PyPI, GPU extras gpu_cuda12/13, uv dev install), Migration Guide v0.3.0 to v0.4.0, Computation stack change: NumPy eager to JAX JIT, Optional GPU acceleration (CUDA 11/12, TPU), Results as jax.Array vs numpy.ndarray (+2 more)

### Community 138 - "`test_docs.py` - Documentation Testing"
Cohesion: 0.22
Nodes (11): Available Scripts, Example Output:, Integration:, Maintenance Notes, Output:, Requirements:, `run_type_check.py` - Type Checking, Scripts Directory (+3 more)

### Community 143 - "parse_formula"
Cohesion: 0.20
Nodes (3): TestDelegatingWrappers, TestRegexCompatibility, _parse_formula()

### Community 146 - "ADR-001: JAX vs NumPy Computation Backend"
Cohesion: 0.20
Nodes (9): ADR-001: JAX vs NumPy Computation Backend, Appendix: Functions that MUST NOT be JIT-compiled, Appendix: Functions to JIT-Compile (Priority Order), Consequences, Context, Decision, Negative, Positive (+1 more)

### Community 147 - "API Reference Index"
Cohesion: 0.24
Nodes (10): Constants Module API Doc, CODATA 2018 standards, Unit conversion factors (angstrom, cm, keV, rad), ENERGY_TO_WAVELENGTH_FACTOR (hc/e / 1000), Physical constants table (PLANCK, SPEED_OF_LIGHT, AVOGADRO, THOMPSON, ...), API Reference Index, XRayLabTool sub-package layout (calculators, data_handling, interfaces, validation, io, analysis, gui, backend, utils), utils helpers (parse_formula, energy_to_wavelength, wavelength_to_energy) (+2 more)

### Community 148 - "Decision: replace matplotlib with PyQtGraph for interactive plots"
Cohesion: 0.20
Nodes (9): ColorPalette dataclass (to_qpalette), F1F2Plot widget, MultiF1F2Plot widget, PlotCanvas widget, PlotWidget protocol (clear, plot_single, plot_multi, update_theme, set_scales), export_publication_plot (matplotlib vector export), PyQtGraphPlotCanvas / PyQtGraphF1F2Plot / PyQtGraphMultiF1F2Plot, interpax device-side interpolation (+1 more)

### Community 149 - "JAX Architecture & Design (v0.4.0)"
Cohesion: 0.27
Nodes (8): vmap-based batch processing (single bulk transfer), JAX Architecture & Design (v0.4.0), Automatic differentiation (future capability), JAX gotchas (Python control flow, non-array inputs), GPU acceleration and device placement, JIT compilation (cost-benefit, warm-up pattern), Shape-based recompilation gotcha, vmap vectorization (future v0.5+ automatic batching)

### Community 156 - "TestGuiErrorHandling"
Cohesion: 0.22
Nodes (3): numpy_backend_after(), TestGuiErrorHandling, task()

### Community 157 - "FastXRayCalculationEngine"
Cohesion: 0.20
Nodes (3): FastXRayCalculationEngine, get_calculation_engine(), get_atomic_data_provider()

### Community 161 - "wrapper"
Cohesion: 0.22
Nodes (5): memory_test(), performance_test(), decorator(), requires_memory(), decorator()

### Community 165 - "typing_extensions.py"
Cohesion: 0.25
Nodes (3): ensure_complex128_array(), ensure_float64_array(), validate_energy_array()

### Community 167 - "Files"
Cohesion: 0.25
Nodes (7): Files, Maintenance, `performance_baselines.json`, `sample_materials.json`, Test Data Directory, `test_energies.json`, Usage in Tests

### Community 168 - "v0.4.6 release (2026-09-17)"
Cohesion: 0.29
Nodes (8): CLI package refactor, completion_v2 shell completion (legacy completion removed), GUI main_window mixin package, pip-audit dependency vulnerability gate, publish.yml manual release workflow, safety dependency removal (pip-audit only), v0.4.6 release (2026-09-17), Version 0.4.6 (2026-09-17)

### Community 184 - "Decision"
Cohesion: 0.33
Nodes (6): Batch Processing Optimization, Decision, Implementation Pattern, Transfer Policy, JIT Compilation Strategy, Architecture Overview

### Community 185 - "gc"
Cohesion: 0.29
Nodes (3): clean_cache(), enforce_test_isolation(), setup_test_environment()

### Community 191 - "Decision: enforce float64 globally; float32 is a correctness bug"
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
- **101 isolated node(s):** `Numerical precision requirements by quantity and function`, `CLI modules (parser, formatting, utils)`, `CLI output formats (table, csv, json)`, `Supported shells (bash, zsh, fish, PowerShell)`, `atomic command` (+96 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1775 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **132 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `calculate_single_material_properties()` connect `Energy Input & CLI Compatibility Tests` to `MaterialComparator`, `set_backend`, `XRayResult Dataclass`, `timer`, `Batch Processing`, `GUI Services & Window`, `clear_scattering_factor_cache`, `BatchConfig`, `export_to_csv`, `_validate_single_material_inputs`, `core.py`, `CLI Batch Command`, `export_to_csv`, `TestCLIFileOperations`, `TestCLICompatibility`, `TestEdgeCaseInputValues`, `test_cli_edge_cases.py`, `cache.py`, `test_optimization_validation.py`, `cmd_calc`, `Deep-Debug Regression Tests`, `calculate_attenuation_length`, `FormulaError`, `Performance Benchmarks`, `TestCalculationSpeedBenchmarks`, `TestBenchmarkComparison`, `process_single_calculation`, `save_calculation_results`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `MainWindowProtocol` (e.g. with `MultiTabMixin` and `Toast`) actually correct?**
  _`MainWindowProtocol` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Numerical precision requirements by quantity and function`, `CLI modules (parser, formatting, utils)`, `CLI output formats (table, csv, json)` to the rest of the system?**
  _101 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `MainWindowProtocol` be split into smaller, more focused modules?**
  _Cohesion score 0.05081081081081081 - nodes in this community are weakly interconnected._
- **Why does `calculators.core (calculate_single_material_properties, calculate_xray_properties)` connect `calculators.core (calculate_single_material_properties, calculate_xray_properties)` to `core.py`, `Data Handling Module API Doc`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `XRayResult (per-energy NumPy array fields)` and `Results as jax.Array vs numpy.ndarray`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Claimed speedups (7.5x single, 10x batch, 15x GPU)` and `Backend auto-selection (JAX only with NVIDIA GPU, else NumPy)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._