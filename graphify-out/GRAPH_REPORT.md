# Graph Report - pyXRayLabTool  (2026-10-24)

## Corpus Check
- 182 files · ~161,989 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 110 file(s) not represented in the graph (top: .nff 93, .npz 10, .ipynb 2)

## Summary
- 4604 nodes · 9357 edges · 267 communities (128 shown, 139 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 668 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e715d6f3`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- MainWindowProtocol
- format_xray_result
- services.py
- StyleGuideValidator
- set_backend
- EnvironmentDetector
- Any
- _make_cubic
- constants.py
- NumpyBackend
- test_exceptions.py
- XRayResult
- CompletionInstaller
- calculate_xray_properties
- current_palette
- cli.py
- test_performance_benchmarks.py
- parse_formula
- single_tab.py
- MemoryMonitor
- core.py
- os
- validate_chemical_formula
- export_to_json
- test_utilities.py
- style.py
- MaterialComparator
- scripts/test_docs.py
- test_concurrent_cache.py
- formatting.py
- batch_processing.py
- typing
- BatchConfig
- _calc
- main
- test_deep_debug_regressions.py
- TestCICDIntegration
- EnvironmentInfo
- parser.py
- kernels.py
- TestEnergySweepSi
- create_scattering_factor_interpolators
- test_gui_widgets.py
- calculate_single_material_properties
- test_file_operations.py
- test_cli_edge_cases.py
- CalculationWorker
- installer.py
- collections_abc
- test_utils.py
- validate_density
- test_optimization_validation.py
- EnergyError
- .calculate_derived_quantities
- validators.py
- Utilities Module
- SharedHelpersMixin
- device.py
- TestFullPipelineSiO2
- cmd_calc
- TestTypeSafetyConfig
- argparse
- data_handling/__init__.py
- d_spacing_cubic
- ndarray
- batch.py
- tests/conftest.py
- TestInstallCompletionCommand
- Changelog (docs)
- xraylabtool/__init__.py
- get_atomic_number
- environment.py
- scattering_data.py
- FastAtomicDataProvider
- typing_extensions.py
- legacy_install_completion_main
- TestCodeQuality
- TestXRayResultTyping
- TestCrystallographicCalculations
- pytest
- run_type_check.py
- calculate_transmission
- TestInterpolatorCaching
- fixture
- MemoryProfiler
- PerformanceBenchmark
- test_compound_analysis.py
- test_formula_parsing.py
- atomic_cache.py
- Version 0.4.7 (2026-10-23)
- Contributing Guide
- XRayLabTool Documentation (index)
- X-ray Optics Fundamentals
- test_base.py
- xraylabtool/utils.py
- Any
- compound_analysis.py
- file_operations.py
- calculate_critical_angle
- clear_scattering_factor_cache
- test_cli.py
- parse_energy_string
- cmd_list
- FormulaError
- TestCIWorkflowSimulation
- ADR-002: PyQtGraph vs Matplotlib for GUI Plotting
- OverlayScrollbarMarginHelper
- Shell Completion Guide
- TestGoldenScatteringLengthDensity
- TestGoldenSi
- TestGoldenAu
- TestGoldenFe2O3
- TestGoldenSiO2
- BashCompletionGenerator
- TestPerformanceBenchmarks
- test_speed_optimization_benchmarks.py
- numpy
- XRayLabToolError
- CalculationError
- TestExtremeConditions
- TestPhysicalRealism
- ZshCompletionGenerator
- ADR-003: float64 Preservation Policy
- TestBenchmarkComparison
- TestEnvironmentDetection
- TestMultiMaterialEquivalence
- cmd_convert
- cmd_bragg
- test_memory_management.py
- compare.py
- ADR-004: Host-Device Transfer Minimization Strategy
- Data Handling Module API Doc
- Atomic Scattering Factor Data
- pathlib
- TestUnitConversionConstants
- memory_test
- test_transform.py
- .uninstall
- calculators.core (calculate_single_material_properties, calculate_xray_properties)
- XRayLabToolError (base exception)
- Migration Guide v0.3.0 to v0.4.0
- CLI Reference
- Getting Started
- AtomicDataProvider
- `test_docs.py` - Documentation Testing
- test_golden_derived_quantities.py
- BaseXRayLabToolTest
- kk_calculate_real
- cache.py
- get_gpu_info
- test_type_safety_config.py
- shells.py
- ADR-001: JAX vs NumPy Computation Backend
- TestKKCli
- API Reference Index
- Decision: replace matplotlib with PyQtGraph for interactive plots
- JAX Architecture & Design (v0.4.0)
- test_kk_plot.py
- TestCompletionIntegrationFlow
- TestBatchConfig
- TestEnvironmentProperties
- parse_chemical_formula
- test_gui_kk_tab.py
- v0.4.0 release (2026-04-07)
- test_nearedge_golden.py
- RegressionTracker
- export_to_csv
- TestBatchProcessing
- ensure_complex128_array
- ValidationError
- TableFormatter
- PowerShellCompletionGenerator
- Files
- Performance and Optimization guide
- batch command
- calc command
- TestDerivedConstants
- BaseUnitTest
- compute_kk
- TestCLIFileOperations
- TestCLIErrorHandling
- TestCLICompatibility
- xraylabtool/kk/__init__.py
- TestMemoryLeakPrevention
- analyze_element_associations
- find_similar_compounds
- get_compound_complexity_score
- kk/test_docs.py
- gen_kk_golden.py
- subprocess
- uninstall_completion_main
- PerformanceMetrics
- Decision
- gc
- ResultValidator
- TestJaxBackendParity
- performance/conftest.py
- Decision: enforce float64 globally; float32 is a correctness bug
- Install from conda (imewei channel, noarch, CPU only)
- test_backend_kk_ops.py
- test_kk_packaging.py
- TestXRayResultProtocolCompliance
- _OpsProxy
- TestBasicExportIntegration
- test_completion_aliases.py
- MemoryTracker
- PerformanceTimer
- test_data_generator
- DataFileError
- MaterialInputForm
- get_atomic_data_fast
- v0.4.6 release (2026-09-17)
- build_kk_data.py
- skip_if_low_memory
- save_calculation_results
- optimize_array_memory_layout
- pyXRayLabTool Logo (Dark)
- TestFundamentalConstants
- TestParseFormulaDecimal
- large_batch_data
- material_and_density
- performance_timer
- TestParseFormulaErrors
- TestCLIInputValidation
- Analysis API
- Backend API
- IO Operations API
- Examples Index
- io/
- validation/
- pyXRayLabTool Favicon
- GUI Main Window Screenshot
- Sample Materials Test Data
- TestCLIPerformanceAndMemory
- TestParseFormulaParentheses
- TestDtypePreservation
- _convert_atomic_number_to_int
- TestMathematicalConstants
- assert_arrays_close
- _auto_select_backend
- _clear_cache
- clean_cache_around
- batch_config_default
- benchmark_baseline
- memory_monitor
- memory_tracker
- sample_calculation_result
- sample_batch_data
- test_materials
- test_energies_large
- _configure_jax_float64

## God Nodes (most connected - your core abstractions)
1. `MainWindowProtocol` - 99 edges
2. `calculate_single_material_properties()` - 95 edges
3. `parse_formula()` - 75 edges
4. `CompletionInstaller` - 69 edges
5. `kk_calculate_real()` - 58 edges
6. `JaxBackend` - 51 edges
7. `XRayResult` - 47 edges
8. `ValidationError` - 47 edges
9. `kk_real_spectrum()` - 46 edges
10. `BatchConfig` - 44 edges

## Surprising Connections (you probably didn't know these)
- `wrapper()` --calls--> `clear_scattering_factor_cache()`  [INFERRED]
  tests/fixtures/test_utilities.py → xraylabtool/calculators/cache.py
- `truncation_deficit()` --calls--> `load_asf()`  [INFERRED]
  docs/_static/kk/make_truncation_figure.py → xraylabtool/kk/database.py
- `truncation_deficit()` --calls--> `compound_spectrum()`  [INFERRED]
  docs/_static/kk/make_truncation_figure.py → xraylabtool/kk/spectrum.py
- `truncation_deficit()` --calls--> `kk_real_spectrum()`  [INFERRED]
  docs/_static/kk/make_truncation_figure.py → xraylabtool/kk/transform.py
- `truncation_deficit()` --calls--> `relativistic_correction()`  [INFERRED]
  docs/_static/kk/make_truncation_figure.py → xraylabtool/kk/transform.py

## Import Cycles
- 4-file cycle: `xraylabtool/__init__.py -> xraylabtool/calculators/__init__.py -> xraylabtool/calculators/core.py -> xraylabtool/logging_utils.py -> xraylabtool/__init__.py`

## Hyperedges (group relationships)
- **xraylabtool CLI commands** — docs_guides_cli_reference_calc, docs_guides_cli_reference_batch, docs_guides_cli_reference_compare, docs_guides_cli_reference_convert, docs_guides_cli_reference_formula, docs_guides_cli_reference_atomic, docs_guides_cli_reference_bragg, docs_guides_cli_reference_list, docs_guides_cli_reference_completion [EXTRACTED 1.00]
- **XRayLabTool modular sub-package architecture** — docs_guides_getting_started_arch_backend, docs_guides_getting_started_arch_calculators, docs_guides_getting_started_arch_data_handling, docs_guides_getting_started_arch_interfaces, docs_guides_getting_started_arch_io, docs_guides_getting_started_arch_validation, docs_guides_getting_started_arch_gui [EXTRACTED 1.00]
- **Atomic factors to refractive index to derived optical quantities** — docs_physics_atomic_data_scattering_factors, docs_physics_calculations_number_density, docs_physics_calculations_delta_beta_sum, docs_physics_calculations_critical_angle, docs_physics_calculations_attenuation [EXTRACTED 1.00]
- **XRayLabTool exception hierarchy rooted at XRayLabToolError** — docs_api_validation_xraylabtoolerror, docs_api_validation_validationerror, docs_api_validation_formulaerror, docs_api_validation_energyerror, docs_api_validation_calculationerror [EXTRACTED 1.00]
- **Release and install verification scripts** — scripts_check_version, scripts_test_install, scripts_gpu_check [INFERRED 0.75]
- **JAX migration architecture decisions (float64, host-device transfers, JAX design)** — docs_architecture_003_adr_dtype_policy_decision, docs_architecture_004_adr_host_device_transfers_decision, docs_architecture_jax_architecture_design_philosophy, docs_architecture_jax_architecture_jit [INFERRED 0.85]
- **JAX Architecture Decision Records** — docs_architecture_001_adr_jax_vs_numpy_jax_decision, architecture_004_adr_host_device_transfers_transfer_policy [INFERRED 0.85]
- **v0.4.0 JAX migration, rollback and release documentation** — docs_guides_migration_guide_v0_4, docs_development_rollback_procedures [INFERRED 0.85]
- **Virtual environment-centric completion system** — docs_guides_completion_guide_venv_isolation, docs_guides_completion_guide_env_detection, docs_guides_completion_guide_multi_shell, docs_guides_completion_guide_no_sudo, docs_guides_completion_guide_environment_detector, docs_guides_completion_guide_completion_installer [INFERRED 0.85]
- **Files kept in sync by check_version.py** — scripts_check_version, docs_index_citation, docs_development_changelog_version_0_4_7 [INFERRED 0.85]

## Communities (267 total, 139 thin omitted)

### Community 0 - "MainWindowProtocol"
Cohesion: 0.05
Nodes (4): KKTabMixin, MultiTabMixin, MainWindowProtocol, SingleTabMixin

### Community 1 - "format_xray_result"
Cohesion: 0.05
Nodes (9): formula(), MockXRayResult, TestFormatCalculationSummary, TestFormatXRayResult, _format_as_csv(), _format_as_json(), _format_as_table(), format_calculation_summary() (+1 more)

### Community 2 - "services.py"
Cohesion: 0.12
Nodes (12): _run_smoke(), test_compute_multiple_reports_progress_to_100(), _ensure_app(), test_export_csv_handles_io_error(), test_gui_smoke_offscreen(), test_gui_smoke_threaded_offscreen(), test_duplicate_formula_different_density_not_dropped(), suppress_qt_noise() (+4 more)

### Community 3 - "StyleGuideValidator"
Cohesion: 0.07
Nodes (4): main(), StyleGuideValidator, StyleViolation, ValidationReport

### Community 4 - "set_backend"
Cohesion: 0.06
Nodes (13): backend_name(), backend_name(), backend_name(), backend_name(), jax_backend(), TestBackendSwitching, TestEdgeCases, TestInterpolationFactory (+5 more)

### Community 7 - "_make_cubic"
Cohesion: 0.06
Nodes (6): TestCrystalStructure, _make_cubic(), _phase(), TestCrystalStructureAPI, TestStructureFactorPhysics, CrystalStructure

### Community 8 - "constants.py"
Cohesion: 0.11
Nodes (7): TestEnergyToWavelength, attenuation_length_cm(), critical_angle_degrees(), energy_to_wavelength_angstrom(), _isclose(), validate_constants(), wavelength_angstrom_to_energy()

### Community 10 - "test_exceptions.py"
Cohesion: 0.09
Nodes (6): TestAtomicDataError, TestExceptionHierarchy, TestUnknownElementError, AtomicDataError, UnknownElementError, _handle_mendeleev_error()

### Community 12 - "CompletionInstaller"
Cohesion: 0.08
Nodes (3): TestSentinelMarkers, CompletionInstaller, test_completion_installation()

### Community 13 - "calculate_xray_properties"
Cohesion: 0.05
Nodes (9): TestBasicSetupAndInitialization, TestDuplicatedFormulasIndependence, TestEdgeCasesAndErrorHandling, TestH2OProperties, TestInputValidationAndErrorHandling, TestPropertyConsistency, TestSiO2Properties, calculate_xray_properties() (+1 more)

### Community 14 - "current_palette"
Cohesion: 0.08
Nodes (6): apply_palette_to_item(), apply_palette_to_widget(), current_palette(), PlotCanvas, F1F2Plot, MultiF1F2Plot

### Community 15 - "cli.py"
Cohesion: 0.08
Nodes (8): TestBackwardCompatibility, TestInstallCompletionMain, completion_main(), create_completion_parser(), handle_completion_command(), install_completion_main(), show_completion_info(), handle_new_completion_command()

### Community 16 - "test_performance_benchmarks.py"
Cohesion: 0.07
Nodes (4): TestPerformanceBenchmarks, TestPerformanceRegression, TestScalabilityBenchmarks, timer()

### Community 17 - "parse_formula"
Cohesion: 0.14
Nodes (4): TestParseFormulaBasic, _aggregate_element_counts(), parse_formula(), _validate_balanced_parentheses()

### Community 19 - "MemoryMonitor"
Cohesion: 0.11
Nodes (3): TestCacheManagement, TestMemoryMonitor, MemoryMonitor

### Community 20 - "core.py"
Cohesion: 0.07
Nodes (18): calculate_multiple_xray_properties(), _calculate_single_material_xray_properties(), _coerce_energy_array(), _convert_energy_input(), _create_process_formula_function(), process_formula(), FastXRayCalculationEngine, get_calculation_engine() (+10 more)

### Community 21 - "os"
Cohesion: 0.09
Nodes (13): setup(), kk_log(), test_configure_logging_writes_file(), test_get_logger_child_inherits_config(), build_parser(), main(), _bool_env(), _coerce_int_env() (+5 more)

### Community 22 - "validate_chemical_formula"
Cohesion: 0.10
Nodes (3): TestValidateChemicalFormula, _get_valid_element_symbols(), validate_chemical_formula()

### Community 24 - "test_utilities.py"
Cohesion: 0.10
Nodes (10): assert_values_close(), expect_no_warnings(), expect_warnings(), get_system_info(), log_system_info(), skip_if_insufficient_cpu(), skip_if_insufficient_memory(), stability_test() (+2 more)

### Community 25 - "style.py"
Cohesion: 0.10
Nodes (6): apply_pyqtgraph_theme(), apply_styles(), apply_theme(), ColorPalette, get_qss(), ThemeManager

### Community 27 - "scripts/test_docs.py"
Cohesion: 0.15
Nodes (14): CI/CD pipelines integration, Pre-commit and CI integration, Sphinx doctest builder, check_accessibility(), check_documentation_coverage(), Colors, main(), print_section() (+6 more)

### Community 28 - "test_concurrent_cache.py"
Cohesion: 0.08
Nodes (12): clear_lru_caches(), _mock_get_element(), TestCacheResultConsistency, worker(), TestConcurrentGetAtomicNumber, worker(), cache_clearer(), reader() (+4 more)

### Community 29 - "formatting.py"
Cohesion: 0.07
Nodes (15): TestResultFormatting, _format_array_field_single(), _format_as_csv(), _format_as_json(), _format_filtered_table(), _format_material_properties(), _format_multiple_energies(), _format_multiple_energy_section() (+7 more)

### Community 30 - "batch_processing.py"
Cohesion: 0.12
Nodes (10): chunk_iterator(), _filter_dataframe_fields(), _get_energy_point_data(), _initialize_progress_bar(), _prepare_energies_array(), _prepare_result_data(), process_batch_chunk(), _process_chunks() (+2 more)

### Community 31 - "typing"
Cohesion: 0.12
Nodes (8): cmd_completion(), cmd_install_completion(), cmd_uninstall_completion(), cmd_formula(), _get_atomic_data(), _output_formula_results(), _print_formula_results(), _process_formula()

### Community 32 - "BatchConfig"
Cohesion: 0.12
Nodes (7): _single(), TestBatchProcessorEquivalence, no_progress_config(), TestBatchInputValidation, TestBatchPartialFailure, BatchConfig, calculate_batch_properties()

### Community 33 - "_calc"
Cohesion: 0.11
Nodes (6): _calc(), TestComplexFormula, TestElectronCountGolden, TestExactCapturedValues, TestMolecularWeightGolden, _calculate_molecular_properties()

### Community 34 - "main"
Cohesion: 0.12
Nodes (3): TestCLIIntegration, TestMainFunction, main()

### Community 35 - "test_deep_debug_regressions.py"
Cohesion: 0.07
Nodes (12): numpy_backend_after(), test_whitespace_delimited_loader_works_with_current_pandas(), TestExports, TestFormulaParsing, TestGuiErrorHandling, task(), TestLazySubmoduleAccess, TestRepeatedMaterials (+4 more)

### Community 38 - "parser.py"
Cohesion: 0.11
Nodes (13): add_atomic_command(), add_batch_command(), add_bragg_command(), add_calc_command(), add_compare_command(), add_completion_command(), add_convert_command(), add_formula_command() (+5 more)

### Community 39 - "kernels.py"
Cohesion: 0.12
Nodes (7): calculate_derived_quantities(), calculate_scattering_factors(), _derived_quantities_kernel(), _get_derived_kernel(), _get_scattering_kernel(), _scattering_math_kernel(), InterpolatorProtocol

### Community 41 - "create_scattering_factor_interpolators"
Cohesion: 0.08
Nodes (7): _check_element(), cold_cache(), test_si_o_au_interpolators_are_independent(), TestPchipAu, TestPchipO, TestPchipSi, create_scattering_factor_interpolators()

### Community 42 - "test_gui_widgets.py"
Cohesion: 0.09
Nodes (6): form(), _get_app(), qt_app(), TestDensityValidation, TestEnergyGridValidation, TestFormulaValidation

### Community 43 - "calculate_single_material_properties"
Cohesion: 0.06
Nodes (7): TestCLIOutputFormatting, TestEdgeCaseInputValues, TestSubRefracErrorHandling, TestSubRefracSiliconProperties, TestAnomalousDispersion, TestEnergyInputContract, calculate_single_material_properties()

### Community 44 - "test_file_operations.py"
Cohesion: 0.20
Nodes (12): test_load_absolute_path(), test_load_csv_file(), test_load_csv_space_separated_fallback(), test_load_empty_nff_raises_data_file_error(), test_load_malformed_nff_raises_data_file_error(), test_load_nff_energy_column_monotone(), test_load_nff_values_are_finite(), test_load_nonexistent_file_raises_file_not_found() (+4 more)

### Community 46 - "CalculationWorker"
Cohesion: 0.13
Nodes (10): qt_app(), _run_worker(), test_worker_does_not_emit_finished_on_exception(), test_worker_emits_error_on_exception(), bad_fn(), test_worker_emits_result_on_success(), test_worker_injects_progress_cb_when_accepted(), test_worker_passes_kwargs() (+2 more)

### Community 47 - "installer.py"
Cohesion: 0.09
Nodes (5): TestCompletionInstallToTempDir, TestCompletionScriptGeneration, CompletionManager, get_global_options(), get_xraylabtool_commands()

### Community 48 - "collections_abc"
Cohesion: 0.10
Nodes (3): TestBasicAnalysis, compare_materials(), find_absorption_edges()

### Community 49 - "test_utils.py"
Cohesion: 0.11
Nodes (6): TestDataProcessing, background_subtraction(), find_peaks(), normalize_intensity(), save_processed_data(), smooth_data()

### Community 51 - "test_optimization_validation.py"
Cohesion: 0.06
Nodes (6): TestArrayOptimization, TestBulkAtomicDataOptimization, TestDeprecationWarningOptimization, TestElementPathOptimization, TestNumericalStabilityOptimizations, TestPerformanceRegression

### Community 52 - "EnergyError"
Cohesion: 0.06
Nodes (16): _manual(), test_negative_f1_near_edge_reaches_xray_result(), test_result_module_doctests(), test_to_xray_result_accepts_range_ends(), test_to_xray_result_rejects_out_of_range_energy(), test_to_xray_result_uses_kk_f1(), TestEnergyError, TestBoundaryConditions (+8 more)

### Community 54 - "validators.py"
Cohesion: 0.07
Nodes (4): TestValidateCalculationParameters, TestValidateEnergyRange, validate_calculation_parameters(), validate_energy_range()

### Community 55 - "Utilities Module"
Cohesion: 0.10
Nodes (12): angle_from_q(), bragg_angle(), d_spacing_cubic(), xraylabtool.device, energy_to_wavelength(), get_atomic_data(), load_atomic_data(), xraylabtool.logging_utils (+4 more)

### Community 57 - "device.py"
Cohesion: 0.09
Nodes (8): TestCheckGpuAvailability, TestGetDeviceInfo, TestGetSystemCudaVersion, check_gpu_availability(), check_plugin_conflicts(), get_device_info(), get_recommended_package(), get_system_cuda_version()

### Community 59 - "cmd_calc"
Cohesion: 0.13
Nodes (6): TestCalcCommand, cmd_calc(), _determine_output_format(), _print_calc_verbose_info(), _save_or_print_output(), _validate_calc_inputs()

### Community 61 - "argparse"
Cohesion: 0.09
Nodes (14): Citation (BibTeX, v0.4.7), collect(), main(), _search(), main(), jax-cuda12/cuda13 plugin conflict check, Makefile targets (version-check, test-install-*, gpu-*), NumPy-style docstring validation (+6 more)

### Community 62 - "data_handling/__init__.py"
Cohesion: 0.16
Nodes (7): get_cache_stats(), is_element_preloaded(), warm_up_cache(), load_batch_input(), save_batch_results(), _save_dataframe(), __getattr__()

### Community 63 - "d_spacing_cubic"
Cohesion: 0.22
Nodes (4): _check_lattice(), d_spacing_cubic(), d_spacing_orthorhombic(), d_spacing_tetragonal()

### Community 66 - "batch.py"
Cohesion: 0.15
Nodes (7): cmd_batch(), _convert_result_to_dict(), _parse_batch_data(), _process_batch_materials(), _save_batch_results(), _validate_batch_input(), create_batch_progress_tracker()

### Community 67 - "tests/conftest.py"
Cohesion: 0.10
Nodes (8): assert_calculation_result_valid(), assert_memory_usage_reasonable(), assert_performance_within_threshold(), batch_config_performance(), batch_config_small(), pytest_addoption(), pytest_collection_modifyitems(), pytest_configure()

### Community 69 - "Changelog (docs)"
Cohesion: 0.18
Nodes (12): Changelog (docs), CHANGELOG.md (v0.2.x history), GUI theme engine (light/dark palettes), Keep a Changelog format, Root CHANGELOG.md (v0.2.x history), Semantic Versioning, v0.1.0 release (2024), v0.3.0 release (2025-12-15) (+4 more)

### Community 70 - "xraylabtool/__init__.py"
Cohesion: 0.06
Nodes (25): _cxro_deviation(), test_30kev_truncation_matches_paper_trend(), test_default_grid_matches_upstream_golden_bit_for_bit_inputs(), test_end_to_end_matches_upstream_golden(), test_tabulated_kk_f1_close_to_cxro_f1(), _asf_f2(), test_asf_grid_spans_10ev_to_500kev(), test_asf_has_92_elements() (+17 more)

### Community 71 - "get_atomic_number"
Cohesion: 0.12
Nodes (4): TestAtomicDataLookup, get_atomic_data(), get_atomic_number(), get_atomic_weight()

### Community 72 - "environment.py"
Cohesion: 0.09
Nodes (3): TestEnvironmentIntegration, TestEnvironmentType, EnvironmentType

### Community 73 - "scattering_data.py"
Cohesion: 0.07
Nodes (6): TestAtomicScatteringFactor, TestLoadDataFile, AtomicScatteringFactor, _initialize_element_paths(), load_scattering_factor_data(), ScatteringData

### Community 75 - "typing_extensions.py"
Cohesion: 0.18
Nodes (6): get_optimal_chunk_size(), is_complex_array(), is_energy_array(), is_real_array(), validate_complex_array(), validate_energy_array()

### Community 79 - "TestCrystallographicCalculations"
Cohesion: 0.14
Nodes (3): TestCrystallographicCalculations, angle_from_q(), q_from_angle()

### Community 80 - "pytest"
Cohesion: 0.13
Nodes (36): _edge_spectrum(), test_asf_input_is_unchanged_and_unknown_type_raises(), test_beta_round_trip_and_density_cancels_at_merge(), test_default_merge_points_are_the_overlap_of_both_ranges(), test_directly_constructed_spectrum_is_validated(), test_docstring_examples_run(), test_file_in_kev_declared_ev_raises_no_overlap(), test_flat_endpoints_raise_not_inf() (+28 more)

### Community 81 - "run_type_check.py"
Cohesion: 0.23
Nodes (7): MyPy static type analysis, analyze_mypy_output(), check_core_modules(), check_type_definitions(), main(), run_mypy_check(), validate_cache_performance()

### Community 84 - "fixture"
Cohesion: 0.12
Nodes (7): cpu_count(), energy_arrays(), mock_low_memory(), performance_thresholds(), temp_directory(), test_energies_medium(), test_energies_small()

### Community 85 - "MemoryProfiler"
Cohesion: 0.15
Nodes (3): decorator(), wrapper(), MemoryProfiler

### Community 87 - "test_compound_analysis.py"
Cohesion: 0.19
Nodes (4): TestGetCompoundFrequencyScore, TestGetRecommendedElementsForWarming, get_compound_frequency_score(), get_recommended_elements_for_warming()

### Community 88 - "test_formula_parsing.py"
Cohesion: 0.17
Nodes (3): TestDelegatingWrappers, TestRegexCompatibility, _parse_formula()

### Community 89 - "atomic_cache.py"
Cohesion: 0.13
Nodes (9): TestGetElementsForCompound, _collect_compound_elements(), _collect_family_compound_elements(), _collect_similar_compound_elements(), _warm_atomic_data(), _warm_bulk_combinations(), warm_cache_for_compounds(), _warm_interpolators() (+1 more)

### Community 90 - "Version 0.4.7 (2026-10-23)"
Cohesion: 0.15
Nodes (15): Anomalous dispersion (negative delta) handling, Anomalous dispersion (negative delta) fix, Batch API ordering, timeout and fallback fixes, Deep-debug regression tests, Energy input contract (scalar or 1-D real, EnergyError), CSV/JSON export fixes and full precision, Export full-precision and CSV fixes, Lazy loader RecursionError fix (+7 more)

### Community 91 - "Contributing Guide"
Cohesion: 0.15
Nodes (14): Compatibility Matrix, Contributing Guide, Code standards (PEP 8, 88 cols, NumPy docstrings, type hints), Performance requirements (<0.1 ms single, >100k calc/s batch, >90% cache hit), Review process (automated checks, reviewer, maintainer, squash merge), Development workflow (branching, ruff, mypy, pytest, make verify), Testing Guide, Characterization (golden-value) tests, 202 assertions (+6 more)

### Community 92 - "XRayLabTool Documentation (index)"
Cohesion: 0.20
Nodes (12): backend/ (NumPy/JAX abstraction), set_backend(), XRayLabTool Documentation (index), Backend abstraction (NumPy/JAX set_backend), 202 characterization tests with golden values, CLI interface feature summary, CXRO/NIST atomic scattering databases, Performance features (cache, vectorization, JAX JIT) (+4 more)

### Community 93 - "X-ray Optics Fundamentals"
Cohesion: 0.19
Nodes (17): Interpolation methods (linear, logarithmic, spline), Calculation Methods, Attenuation coefficients (mu = 4 pi beta / lambda, mass coefficient, length), Critical angle theta_c = sqrt(2 delta) with unit conversions, Per-element delta/beta contributions summed over compound, Chemical formula parser (grammar, errors), Linear interpolation of f1/f2 (np.interp), Number density n_total = rho N_A / M (+9 more)

### Community 94 - "test_base.py"
Cohesion: 0.14
Nodes (3): generate_test_energies(), generate_test_materials(), ParametrizedTestMixin

### Community 95 - "xraylabtool/utils.py"
Cohesion: 0.11
Nodes (9): TestUnitConversions, __getattr__(), bragg_angle(), energy_to_wavelength(), get_avogadro_number(), get_electron_charge(), get_planck_constant(), get_speed_of_light() (+1 more)

### Community 97 - "compound_analysis.py"
Cohesion: 0.20
Nodes (3): TestGetCompoundFamily, _classify_family_by_elements(), get_compound_family()

### Community 98 - "file_operations.py"
Cohesion: 0.24
Nodes (4): _json_numpy_default(), _save_dict_as_csv(), _save_results_as_csv(), _save_results_as_json()

### Community 100 - "clear_scattering_factor_cache"
Cohesion: 0.08
Nodes (5): _clear_cache(), cold_cache(), TestGoldenDerivedQuantities, TestGoldenScatteringFactors, clear_scattering_factor_cache()

### Community 101 - "test_cli.py"
Cohesion: 0.09
Nodes (4): TestAtomicCommand, TestBatchCommand, TestFormulaCommand, cmd_atomic()

### Community 104 - "FormulaError"
Cohesion: 0.20
Nodes (3): TestFormulaError, FormulaError, _expand_parentheses()

### Community 106 - "ADR-002: PyQtGraph vs Matplotlib for GUI Plotting"
Cohesion: 0.14
Nodes (13): ADR-002: PyQtGraph vs Matplotlib for GUI Plotting, Appendix: Feature Parity Checklist, Consequences, Context, Decision, Migration Details, Negative, Positive (+5 more)

### Community 108 - "Shell Completion Guide"
Cohesion: 0.22
Nodes (12): Shell Completion Guide, Completion caching system (~/.xraylabtool/cache), Completion features (command, option, path, value), CompletionInstaller, xraylabtool.interfaces.completion_v2, Environment detection (venv, conda, mamba, Poetry, Pipenv), EnvironmentDetector, completion install / status / list / uninstall workflow (+4 more)

### Community 115 - "TestPerformanceBenchmarks"
Cohesion: 0.14
Nodes (5): TestPerformanceBenchmarks, energy_sweep(), complex_formula(), multi_calculation(), single_calculation()

### Community 116 - "test_speed_optimization_benchmarks.py"
Cohesion: 0.10
Nodes (4): BasePerformanceTest, TestCalculationSpeedBenchmarks, calculation_worker(), TestPerformanceRegressionDetection

### Community 117 - "numpy"
Cohesion: 0.11
Nodes (33): test_matches_upstream_improve_accuracy_golden(), _max_midpoint_error(), _refine(), test_debug_log_per_pass_and_info_total(), test_docstring_examples_run(), test_exhaustion_flagged_and_logged(), test_heavy_element_segment_grid_against_upstream_emulation(), test_invalid_refine_arguments_raise_validation_error() (+25 more)

### Community 118 - "XRayLabToolError"
Cohesion: 0.15
Nodes (3): TestBatchProcessingError, BatchProcessingError, XRayLabToolError

### Community 119 - "CalculationError"
Cohesion: 0.12
Nodes (3): TestCalculationError, TestExceptionUsage, CalculationError

### Community 123 - "ADR-003: float64 Preservation Policy"
Cohesion: 0.15
Nodes (12): ADR-003: float64 Preservation Policy, Appendix: Numerical Precision Requirements by Function, Consequences, Context, Current Codebase float64 Usage, Decision, Enforcement Strategy, JAX float32 Default (+4 more)

### Community 129 - "test_memory_management.py"
Cohesion: 0.13
Nodes (3): TestBatchProcessingMemoryManagement, TestProcessSingleCalculation, process_single_calculation()

### Community 130 - "compare.py"
Cohesion: 0.24
Nodes (5): cmd_compare(), _parse_materials(), _result_to_dict(), _write_report(), _write_table()

### Community 131 - "ADR-004: Host-Device Transfer Minimization Strategy"
Cohesion: 0.17
Nodes (11): ADR-004: Host-Device Transfer Minimization Strategy, Appendix: Transfer Cost Reference (CPU Backend), Array Sizes in This Workload, Consequences, Context, Monitoring, Negative, Positive (+3 more)

### Community 132 - "Data Handling Module API Doc"
Cohesion: 0.20
Nodes (11): Data Handling Module API Doc, atomic_cache (92-element preload, warm_up_cache, get_cache_stats), batch_processing (BatchConfig, calculate_batch_properties, CSV I/O), compound_analysis (compound families, recommended warming elements), CXRO / NIST / Henke .nff scattering factor data (0.03-30 keV, PCHIP), Interfaces Module API Doc, CLI commands (calc, batch, compare, convert, formula, atomic, bragg, list, completion, install-completion, uninstall-completion), CLI modules (parser, formatting, utils) (+3 more)

### Community 133 - "Atomic Scattering Factor Data"
Cohesion: 0.21
Nodes (12): Contributing atomic data (citation, range, precision, format), Energy grid design (log spacing, dense near absorption edges), Atomic Scattering Factor Data, Absorption edges (K, L, M) and XANES features, CXRO database, Henke et al. 1993, Atomic Data and Nuclear Data Tables 54, 181-342, Henke tables (Z=1-92, nff format), NIST XCOM photon cross sections (+4 more)

### Community 134 - "pathlib"
Cohesion: 0.13
Nodes (6): enable_offscreen_quiet_env(), cmd_kk(), _fail(), _format_output(), _parse_energies(), _parse_merge()

### Community 136 - "memory_test"
Cohesion: 0.22
Nodes (5): memory_test(), performance_test(), decorator(), requires_memory(), decorator()

### Community 137 - "test_transform.py"
Cohesion: 0.11
Nodes (32): _args(), _eq10_corrected(), _f2_poly(), _golden(), _linear_segments(), _quadrature_f1(), test_chunk_invariance(), test_constant_band_closed_form() (+24 more)

### Community 139 - "calculators.core (calculate_single_material_properties, calculate_xray_properties)"
Cohesion: 0.20
Nodes (11): Calculators Module API Doc, Linear absorption coefficient mu = 4 pi beta / lambda, calculators.core (calculate_single_material_properties, calculate_xray_properties), Critical angle theta_c = sqrt(2 delta), Legacy CamelCase accessors (DeprecationWarning), X-ray complex refractive index n = 1 - delta - i beta, calculators submodules (derived_quantities, xray_result, cache, scattering_data, kernels), XRayResult fields table (+3 more)

### Community 140 - "XRayLabToolError (base exception)"
Cohesion: 0.24
Nodes (11): Exception hierarchy (index overview), CLI error handling messages, CalculationError, EnergyError, FormulaError, validate_calculation_parameters, validate_chemical_formula, validate_density (+3 more)

### Community 141 - "Migration Guide v0.3.0 to v0.4.0"
Cohesion: 0.20
Nodes (10): Rollback: v0.4.0 to v0.3.0, Quick rollback procedure (pip uninstall, install 0.3.0, verify), Rollback criteria (results differ >1e-12, GUI fails, JAX import errors), Migration Guide v0.3.0 to v0.4.0, Computation stack change: NumPy eager to JAX JIT, Optional GPU acceleration (CUDA 11/12, TPU), Results as jax.Array vs numpy.ndarray, Claimed speedups (7.5x single, 10x batch, 15x GPU) (+2 more)

### Community 142 - "CLI Reference"
Cohesion: 0.20
Nodes (11): CLI Reference, atomic command, bragg command, completion command, convert command, CLI error handling and --debug, formula command, install-completion / uninstall-completion (legacy) (+3 more)

### Community 143 - "Getting Started"
Cohesion: 0.18
Nodes (12): xtool / xlt CLI aliases, xraylabtool-gui launcher, Completion for xtool alias, Getting Started, data_handling/, gui/ (PySide6 + PyQtGraph), Development installation with uv, Desktop GUI (python -m xraylabtool.gui) (+4 more)

### Community 145 - "`test_docs.py` - Documentation Testing"
Cohesion: 0.22
Nodes (11): Available Scripts, Example Output:, Integration:, Maintenance Notes, Output:, Requirements:, `run_type_check.py` - Type Checking, Scripts Directory (+3 more)

### Community 146 - "test_golden_derived_quantities.py"
Cohesion: 0.16
Nodes (3): TestGoldenAttenuationLength, calculate_attenuation_length(), calculate_scattering_length_density()

### Community 148 - "kk_calculate_real"
Cohesion: 0.13
Nodes (19): test_backend_is_recorded_and_chunk_size_does_not_matter(), test_default_grid_clipped_to_0p03_30_kev_and_logged(), test_docstring_examples_run(), test_energies_are_copied_not_aliased(), test_energy_out_of_data_range_raises_energy_error(), test_explicit_energy_between_10ev_and_500kev_ok(), test_import_xraylabtool_does_not_import_kk(), test_invalid_chunk_size_raises_validation_error() (+11 more)

### Community 149 - "cache.py"
Cohesion: 0.13
Nodes (5): TestAtomicDataCacheIsImmutable, get_bulk_atomic_data(), get_cached_elements(), is_element_cached(), _smart_cache_warming()

### Community 153 - "ADR-001: JAX vs NumPy Computation Backend"
Cohesion: 0.20
Nodes (9): ADR-001: JAX vs NumPy Computation Backend, Appendix: Functions that MUST NOT be JIT-compiled, Appendix: Functions to JIT-Compile (Priority Order), Consequences, Context, Decision, Negative, Positive (+1 more)

### Community 154 - "TestKKCli"
Cohesion: 0.13
Nodes (4): _parse_csv(), _run(), TestKKCli, _write_edge()

### Community 155 - "API Reference Index"
Cohesion: 0.24
Nodes (10): Constants Module API Doc, CODATA 2018 standards, Unit conversion factors (angstrom, cm, keV, rad), ENERGY_TO_WAVELENGTH_FACTOR (hc/e / 1000), Physical constants table (PLANCK, SPEED_OF_LIGHT, AVOGADRO, THOMPSON, ...), API Reference Index, XRayLabTool sub-package layout (calculators, data_handling, interfaces, validation, io, analysis, gui, backend, utils), utils helpers (parse_formula, energy_to_wavelength, wavelength_to_energy) (+2 more)

### Community 156 - "Decision: replace matplotlib with PyQtGraph for interactive plots"
Cohesion: 0.20
Nodes (9): ColorPalette dataclass (to_qpalette), F1F2Plot widget, MultiF1F2Plot widget, PlotCanvas widget, PlotWidget protocol (clear, plot_single, plot_multi, update_theme, set_scales), export_publication_plot (matplotlib vector export), PyQtGraphPlotCanvas / PyQtGraphF1F2Plot / PyQtGraphMultiF1F2Plot, interpax device-side interpolation (+1 more)

### Community 157 - "JAX Architecture & Design (v0.4.0)"
Cohesion: 0.27
Nodes (8): vmap-based batch processing (single bulk transfer), JAX Architecture & Design (v0.4.0), Automatic differentiation (future capability), JAX gotchas (Python control flow, non-array inputs), GPU acceleration and device placement, JIT compilation (cost-benefit, warm-up pattern), Shape-based recompilation gotcha, vmap vectorization (future v0.5+ automatic batching)

### Community 158 - "test_kk_plot.py"
Cohesion: 0.13
Nodes (11): _computation(), _pen_colors(), plot(), qt_app(), test_plot_draws_curves_on_unequal_grids(), test_plot_follows_theme_toggle(), test_plot_hides_missing_curves(), test_plot_log_x_rule() (+3 more)

### Community 163 - "test_gui_kk_tab.py"
Cohesion: 0.20
Nodes (15): app(), _small_run(), test_energy_range_controls_reach_the_service(), test_export_csv_columns(), test_export_csv_io_error_is_reported(), test_invalid_inputs_are_reported_without_a_worker(), test_nearedge_options_reach_the_service(), test_run_without_file_populates_plot_and_table() (+7 more)

### Community 164 - "v0.4.0 release (2026-04-07)"
Cohesion: 0.42
Nodes (9): Backend abstraction layer (NumPy/JAX), Characterization tests (202 golden-value), InterpolationFactory, JAX-vs-NumPy parity tests, Migration guide v0.3 to v0.4, GUI plotting migration to PyQtGraph, set_backend(), v0.4.0 release (2026-04-07) (+1 more)

### Community 165 - "test_nearedge_golden.py"
Cohesion: 0.15
Nodes (9): backend_name(), _load_golden(), _merged(), test_kernel_on_merged_spectrum_matches_golden_f1_points(), test_merged_spectrum_matches_upstream_golden(), test_nearedge_result_matches_upstream_golden(), test_matches_upstream_kk_pp_golden(), assert_kk_close() (+1 more)

### Community 170 - "ValidationError"
Cohesion: 0.06
Nodes (17): TestValidationError, ValidationError, _check_refine_arguments(), _merge_points_ev(), _check_positive(), _fmt_range(), _merge(), _Merged (+9 more)

### Community 173 - "Files"
Cohesion: 0.25
Nodes (7): Files, Maintenance, `performance_baselines.json`, `sample_materials.json`, Test Data Directory, `test_energies.json`, Usage in Tests

### Community 174 - "Performance and Optimization guide"
Cohesion: 0.25
Nodes (7): Performance and Optimization guide, Batch tiers (sequential vs chunked Process/ThreadPoolExecutor), Cache warming (warm_up_cache, warm_cache_for_compounds), Energy arrays instead of loops, JIT warm-up on first call, Multi-level caching strategy (preloaded, LRU, disk), Performance optimizations (LRU cache, precomputed grids, chunked batches)

### Community 175 - "batch command"
Cohesion: 0.29
Nodes (8): batch command, Batch CSV format (formula,density,energy), calculators/, Batch processing from materials.csv, calculate_xray_properties, Result properties (critical angle, attenuation length, delta, beta), XRayResult (per-energy NumPy arrays), Modular sub-packages (calculators, data_handling, interfaces, io, validation, analysis, export, backend, gui)

### Community 176 - "calc command"
Cohesion: 0.36
Nodes (7): calc command, compare command, Energy specification (keV, ranges, log), Output formats (table, csv, json), calculate_single_material_properties, Use case: beamline planning, Use case: mirror design

### Community 178 - "BaseUnitTest"
Cohesion: 0.10
Nodes (3): BaseUnitTest, TestDocumentationPatterns, TestStyleCompliance

### Community 179 - "compute_kk"
Cohesion: 0.24
Nodes (12): test_compute_kk_default_grid_cxro_matches_result_energies(), test_compute_kk_forwards_energies_kev(), test_compute_kk_forwards_refine_tolerance(), test_compute_kk_merge_points_follow_energy_unit(), test_compute_kk_rejects_unknown_energy_unit(), test_compute_kk_service_matches_api_with_nearedge(), test_compute_kk_service_matches_api_without_nearedge(), test_cxro_comparison_empty_outside_tabulated_range() (+4 more)

### Community 183 - "xraylabtool/kk/__init__.py"
Cohesion: 0.16
Nodes (3): main(), truncation_deficit(), __getattr__()

### Community 188 - "kk/test_docs.py"
Cohesion: 0.23
Nodes (7): _public_objects(), test_docs_cite_the_paper_and_upstream_licence(), test_docs_pages_toctree_linked(), test_docs_state_the_eq10_erratum(), test_every_public_kk_name_is_documented(), test_public_docstrings_have_examples_and_references(), test_transform_docstrings_cite_paper_equations()

### Community 189 - "gen_kk_golden.py"
Cohesion: 0.32
Nodes (8): _checkout(), _eval_points(), _finite(), _git(), main(), _poly(), _save(), _transform_case()

### Community 190 - "subprocess"
Cohesion: 0.24
Nodes (5): _sphinx(), test_kk_docs_build_adds_no_warnings(), test_kk_docs_doctests_pass(), test_headless_gui_smoke_runs_offscreen(), slow()

### Community 193 - "Decision"
Cohesion: 0.33
Nodes (6): Batch Processing Optimization, Decision, Implementation Pattern, Transfer Policy, JIT Compilation Strategy, Architecture Overview

### Community 194 - "gc"
Cohesion: 0.29
Nodes (3): clean_cache(), enforce_test_isolation(), setup_test_environment()

### Community 198 - "Decision: enforce float64 globally; float32 is a correctness bug"
Cohesion: 0.40
Nodes (5): SCATTERING_FACTOR prefactor (THOMPSON*AVOGADRO*1e6/2pi), jax_enable_x64 / JAX_ENABLE_X64 configuration, SCATTERING_FACTOR constant and delta computation example, _verify_float64 import-time check, JAX configuration options (jax_disable_jit, jax_platforms, default float dtype)

### Community 199 - "Install from conda (imewei channel, noarch, CPU only)"
Cohesion: 0.40
Nodes (5): Conda/Mamba install and completion workflow, Install from conda (imewei channel, noarch, CPU only), GPU extras gpu_cuda12 / gpu_cuda13, Install from PyPI, Installation options (pip, GPU extras, conda)

### Community 201 - "test_backend_kk_ops.py"
Cohesion: 0.23
Nodes (5): backend_name(), test_ensure_float64_casts_float32(), test_ensure_float64_enables_x64_under_jax(), test_log_abs_parity_numpy_jax(), ensure_float64()

### Community 212 - "get_atomic_data_fast"
Cohesion: 0.27
Nodes (4): _warm_priority_cache(), _background_cache_warming(), get_atomic_data_fast(), get_bulk_atomic_data_fast()

### Community 213 - "v0.4.6 release (2026-09-17)"
Cohesion: 0.29
Nodes (8): CLI package refactor, completion_v2 shell completion (legacy completion removed), GUI main_window mixin package, pip-audit dependency vulnerability gate, publish.yml manual release workflow, safety dependency removal (pip-audit only), v0.4.6 release (2026-09-17), Version 0.4.6 (2026-09-17)

### Community 216 - "save_calculation_results"
Cohesion: 0.25
Nodes (5): test_save_and_reload_numpy_array(), test_save_dict_of_arrays_csv(), test_save_json(), test_save_unsupported_format_raises(), save_calculation_results()

## Ambiguous Edges - Review These
- `XRayResult as numpy public API boundary (bulk device-to-host in __post_init__)` → `Layered architecture (API, calculation, data, XRayResult)`  [AMBIGUOUS]
  docs/architecture/004-adr-host-device-transfers.md · relation: conceptually_related_to
- `Claimed speedups (7.5x single, 10x batch, 15x GPU)` → `Backend auto-selection (JAX only with NVIDIA GPU, else NumPy)`  [AMBIGUOUS]
  docs/development/performance.rst · relation: conceptually_related_to

## Knowledge Gaps
- **87 isolated node(s):** `gui.main_window (single_tab, multi_tab, shared, toast)`, `gui.services and gui.workers`, `gui theme_manager, style, table_formatter, logging_filters`, `Completion caching system (~/.xraylabtool/cache)`, `Completion features (command, option, path, value)` (+82 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1994 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **139 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `calculate_single_material_properties()` connect `calculate_single_material_properties` to `test_memory_management.py`, `services.py`, `set_backend`, `XRayResult`, `calculate_xray_properties`, `test_performance_benchmarks.py`, `core.py`, `cache.py`, `export_to_json`, `batch_processing.py`, `BatchConfig`, `test_deep_debug_regressions.py`, `export_to_csv`, `test_cli_edge_cases.py`, `collections_abc`, `compute_kk`, `TestCLIFileOperations`, `EnergyError`, `TestCLICompatibility`, `cmd_calc`, `batch.py`, `TestJaxBackendParity`, `xraylabtool/__init__.py`, `TestBasicExportIntegration`, `clear_scattering_factor_cache`, `TestPerformanceBenchmarks`, `test_speed_optimization_benchmarks.py`, `TestBenchmarkComparison`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `MainWindowProtocol` (e.g. with `KKTabMixin` and `MultiTabMixin`) actually correct?**
  _`MainWindowProtocol` has 12 INFERRED edges - model-reasoned connections that need verification._
- **What connects `gui.main_window (single_tab, multi_tab, shared, toast)`, `gui.services and gui.workers`, `gui theme_manager, style, table_formatter, logging_filters` to the rest of the system?**
  _87 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `MainWindowProtocol` be split into smaller, more focused modules?**
  _Cohesion score 0.05092276144907724 - nodes in this community are weakly interconnected._
- **Why does `CompletionInstaller` connect `CompletionInstaller` to `.install`, `uninstall_completion_main`, `TestInstallCompletionCommand`, `test_cli.py`, `EnvironmentDetector`, `EnvironmentInfo`, `environment.py`, `.uninstall`, `legacy_install_completion_main`, `cli.py`, `installer.py`, `typing`, `TestCompletionIntegrationFlow`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `XRayResult as numpy public API boundary (bulk device-to-host in __post_init__)` and `Layered architecture (API, calculation, data, XRayResult)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Claimed speedups (7.5x single, 10x batch, 15x GPU)` and `Backend auto-selection (JAX only with NVIDIA GPU, else NumPy)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._