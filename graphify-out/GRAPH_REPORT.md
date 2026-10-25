# Graph Report - pyXRayLabTool  (2026-10-25)

## Corpus Check
- 183 files · ~162,871 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 110 file(s) not represented in the graph (top: .nff 93, .npz 10, .ipynb 2)

## Summary
- 4788 nodes · 9674 edges · 236 communities (107 shown, 129 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 713 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `13524040`
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
- argparse
- test_performance_benchmarks.py
- parse_formula
- Toast
- MemoryMonitor
- load_asf
- typing
- validate_chemical_formula
- TestJaxBackendParity
- test_utilities.py
- style.py
- TestBasicExportFunctionality
- scripts/test_docs.py
- test_concurrent_cache.py
- formatting.py
- batch_processing.py
- formula.py
- BatchConfig
- _calc
- test_cli.py
- test_deep_debug_regressions.py
- TestCICDIntegration
- EnvironmentInfo
- xraylabtool/kk/__init__.py
- kernels.py
- TestEnergySweepSi
- create_scattering_factor_interpolators
- test_gui_widgets.py
- calculate_single_material_properties
- test_file_operations.py
- atomic_cache.py
- CalculationWorker
- installer.py
- MaterialComparator
- test_utils.py
- validate_density
- test_optimization_validation.py
- fixture
- .calculate_derived_quantities
- validators.py
- Utilities Module
- SharedHelpersMixin
- device.py
- TestFullPipelineSiO2
- cmd_calc
- TestTypeSafetyConfig
- argparse
- unit
- xraylabtool/utils.py
- TestSubRefracSiliconProperties
- ndarray
- batch.py
- tests/conftest.py
- TestInstallCompletionCommand
- Changelog (docs)
- xraylabtool/__init__.py
- get_bulk_atomic_data
- _OpsProxy
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
- validate_calculation_parameters
- calculate_properties_from_factors
- MemoryProfiler
- PerformanceBenchmark
- test_compound_analysis.py
- FishCompletionGenerator
- TestEnvironmentProperties
- Version 0.4.7 (2026-10-23)
- Contributing Guide
- XRayLabTool Documentation (index)
- X-ray Optics Fundamentals
- test_base.py
- xraylabtool/utils.py
- Any
- compound_analysis.py
- TableFormatter
- calculate_critical_angle
- clear_scattering_factor_cache
- EnergyError
- parse_energy_string
- cmd_list
- FormulaError
- TestCIWorkflowSimulation
- ADR-002: PyQtGraph vs Matplotlib for GUI Plotting
- FastXRayCalculationEngine
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
- test_golden_pipeline.py
- ADR-003: float64 Preservation Policy
- TestBenchmarkComparison
- TestEnvironmentDetection
- TestMultiMaterialEquivalence
- cmd_convert
- test_cli.py
- test_memory_management.py
- json
- ADR-004: Host-Device Transfer Minimization Strategy
- Data Handling Module API Doc
- kk.py
- TestDerivedConstants
- memory_test
- test_transform.py
- .uninstall
- calculators.core (calculate_single_material_properties, calculate_xray_properties)
- XRayLabToolError (base exception)
- Backend auto-selection (JAX only with NVIDIA GPU, else NumPy)
- CLI Reference
- Getting Started
- typing_extensions.py
- `test_docs.py` - Documentation Testing
- test_golden_derived_quantities.py
- BaseXRayLabToolTest
- kk_calculate_real
- TestFundamentalConstants
- get_gpu_info
- test_type_safety_config.py
- _parse_formula
- ADR-001: JAX vs NumPy Computation Backend
- TestKKCli
- API Reference Index
- Decision: replace matplotlib with PyQtGraph for interactive plots
- JAX Architecture & Design (v0.4.0)
- test_kk_plot.py
- TestCompletionIntegrationFlow
- Atomic Scattering Factor Data
- environment.py
- parse_chemical_formula
- test_gui_kk_tab.py
- v0.4.0 release (2026-04-07)
- TestUnitConversionConstants
- RegressionTracker
- __getattr__
- TestBatchProcessing
- Testing Guide
- api.py
- TestBackwardCompatibility
- TestNumericalStabilityOptimizations
- Files
- Performance and Optimization guide
- get_elements_for_compound
- TestDtypePreservation
- BaseUnitTest
- compute_kk
- TestCLIFileOperations
- TestMathematicalConstants
- TestCLICompatibility
- xraylabtool/kk/__init__.py
- TestPropertyConsistency
- analyze_element_associations
- find_similar_compounds
- get_compound_complexity_score
- kk/test_docs.py
- gen_kk_golden.py
- subprocess
- uninstall_completion_main
- TestBasicSetupAndInitialization
- Decision
- calculate_multiple_xray_properties
- ResultValidator
- _calculate_single_material_xray_properties
- TestRegexCompatibility
- Decision: enforce float64 globally; float32 is a correctness bug
- .__init__
- PowerShellCompletionGenerator
- _auto_select_backend
- test_kk_packaging.py
- TestXRayResultProtocolCompliance
- test_core.py
- MemoryTracker
- PerformanceTimer
- large_batch_data
- test_memory_management.py
- TestGuiErrorHandling
- memory_tracker
- v0.4.6 release (2026-09-17)
- sample_calculation_result
- batch command
- calc command
- test_energies_large
- integration/__init__.py
- performance/__init__.py
- unit/__init__.py
- BaseIntegrationTest
- BasePerformanceTest
- BaseUnitTest

## God Nodes (most connected - your core abstractions)
1. `MainWindowProtocol` - 99 edges
2. `calculate_single_material_properties()` - 95 edges
3. `parse_formula()` - 76 edges
4. `CompletionInstaller` - 69 edges
5. `kk_calculate_real()` - 64 edges
6. `ValidationError` - 56 edges
7. `JaxBackend` - 51 edges
8. `XRayResult` - 47 edges
9. `EnergyError` - 46 edges
10. `kk_real_spectrum()` - 46 edges

## Surprising Connections (you probably didn't know these)
- `wrapper()` --calls--> `clear_scattering_factor_cache()`  [INFERRED]
  tests/fixtures/test_utilities.py → xraylabtool/calculators/cache.py
- `truncation_deficit()` --calls--> `kk_real_spectrum()`  [INFERRED]
  docs/_static/kk/make_truncation_figure.py → xraylabtool/kk/transform.py
- `truncation_deficit()` --calls--> `relativistic_correction()`  [INFERRED]
  docs/_static/kk/make_truncation_figure.py → xraylabtool/kk/transform.py
- `format_calculation_summary()` --indirect_call--> `formula()`  [INFERRED]
  xraylabtool/io/data_export.py → scripts/gen_kk_golden.py
- `backend_name()` --uses--> `JaxBackend`  [INFERRED]
  tests/characterization/kk/test_end_to_end_golden.py → xraylabtool/backend/array_ops.py

## Import Cycles
- 4-file cycle: `xraylabtool/__init__.py -> xraylabtool/calculators/__init__.py -> xraylabtool/calculators/core.py -> xraylabtool/logging_utils.py -> xraylabtool/__init__.py`

## Hyperedges (group relationships)
- **Watts piecewise-Laurent KK algorithm** — docs_watts_2014_calculation_of_the_kramers_kronig_transform_published_version_piecewise_laurent_polynomial, docs_watts_2014_calculation_of_the_kramers_kronig_transform_published_version_analytic_interval_integration, docs_watts_2014_calculation_of_the_kramers_kronig_transform_published_version_double_interval_integration, docs_watts_2014_calculation_of_the_kramers_kronig_transform_published_version_kronecker_delta_log_handling, docs_watts_2014_calculation_of_the_kramers_kronig_transform_published_version_f1_closed_form [EXTRACTED 0.95]
- **GUI main window tab mixins composed into MainWindow** — docs_api_gui_main_window, docs_api_gui_single_tab, docs_api_gui_multi_tab, docs_api_gui_kk_tab, docs_api_gui_shared, docs_api_gui_toast [EXTRACTED 1.00]
- **XRayLabTool layered architecture (API, calculation, data, output)** — docs_architecture_jax_architecture_api_layer, docs_architecture_jax_architecture_calculation_layer, docs_architecture_jax_architecture_data_layer, docs_architecture_jax_architecture_xrayresult [EXTRACTED 1.00]
- **Python API, CLI and GUI all call kk_calculate_real** — docs_guides_kk_guide_kk_calculate_real, docs_guides_kk_guide_cli_kk, docs_guides_kk_guide_gui_tab [EXTRACTED 1.00]
- **Kramers-Kronig f1 pipeline (compound spectrum, near-edge merge, transform, refine)** — docs_api_kk_compound_spectrum, docs_api_kk_to_asf_scale, docs_api_kk_merge_spectra, docs_api_kk_kk_real_spectrum, docs_api_kk_refine_spectrum, docs_api_kk_kk_calculate_real [EXTRACTED 1.00]
- **KK six-step f1 computation pipeline** — docs_physics_kramers_kronig_six_step_workflow, docs_physics_kramers_kronig_relativistic_correction, docs_physics_kramers_kronig_kk_piecewise_poly, docs_guides_kk_guide_low_level_functions [EXTRACTED 1.00]
- **Atomic factors to refractive index to derived optical quantities** — docs_physics_atomic_data_scattering_factors, docs_physics_calculations_number_density, docs_physics_calculations_delta_beta_sum, docs_physics_calculations_critical_angle, docs_physics_calculations_attenuation [EXTRACTED 1.00]
- **Truncation error compared across C, Fe, Pb** — docs_static_kk_truncation_carbon, docs_static_kk_truncation_iron, docs_static_kk_truncation_lead [EXTRACTED 1.00]
- **Refractive index to absorption to attenuation length chain** — docs_physics_xray_optics_complex_refractive_index, docs_physics_xray_optics_absorption_coefficient, docs_physics_xray_optics_attenuation_length, docs_physics_xray_optics_beer_lambert [EXTRACTED 1.00]
- **XRayLabTool exception hierarchy rooted at XRayLabToolError** — docs_api_validation_xraylabtoolerror, docs_api_validation_validationerror, docs_api_validation_formulaerror, docs_api_validation_energyerror, docs_api_validation_calculationerror [EXTRACTED 1.00]
- **Single Material compute workflow: input, plot, results table** — docs_static_gui_main_offscreen_material_input_panel, docs_static_gui_main_offscreen_attenuation_length_plot, docs_static_gui_main_offscreen_results_table [INFERRED 0.85]
- **JAX migration architecture decisions (float64, host-device transfers, JAX design)** — docs_architecture_003_adr_dtype_policy_decision, docs_architecture_004_adr_host_device_transfers_decision, docs_architecture_jax_architecture_design_philosophy [INFERRED 0.85]
- **JAX Architecture Decision Records** — docs_architecture_001_adr_jax_vs_numpy_jax_decision, architecture_004_adr_host_device_transfers_transfer_policy [INFERRED 0.85]
- **Virtual environment-centric completion system** — docs_guides_completion_guide_venv_isolation, docs_guides_completion_guide_env_detection, docs_guides_completion_guide_multi_shell, docs_guides_completion_guide_no_sudo, docs_guides_completion_guide_environment_detector, docs_guides_completion_guide_completion_installer [INFERRED 0.85]

## Communities (236 total, 129 thin omitted)

### Community 0 - "MainWindowProtocol"
Cohesion: 0.04
Nodes (6): _kk_energy_grid(), KKTabMixin, MultiTabMixin, MainWindowProtocol, SingleTabMixin, MaterialTable

### Community 1 - "format_xray_result"
Cohesion: 0.04
Nodes (25): TestCrystallographicCalculations, TestDataProcessing, TestUnitConversions, _aggregate_element_counts(), angle_from_q(), background_subtraction(), bragg_angle(), _check_lattice() (+17 more)

### Community 2 - "services.py"
Cohesion: 0.04
Nodes (20): backend_name(), backend_name(), backend_name(), backend_name(), backend_name(), jax_backend(), TestBackendSwitching, TestInterpolationFactory (+12 more)

### Community 3 - "StyleGuideValidator"
Cohesion: 0.05
Nodes (33): main(), truncation_deficit(), collect(), main(), _search(), _cxro_deviation(), test_30kev_truncation_matches_paper_trend(), test_default_grid_matches_upstream_golden_bit_for_bit_inputs() (+25 more)

### Community 4 - "set_backend"
Cohesion: 0.05
Nodes (13): TestAtomicDataError, TestDataFileError, TestExceptionHierarchy, TestExceptionUsage, TestFormulaError, TestUnknownElementError, AtomicDataError, CalculationError (+5 more)

### Community 5 - "EnvironmentDetector"
Cohesion: 0.04
Nodes (7): TestBashCompletionSubstitution, test_cli_aliases_registered(), BashCompletionGenerator, CompletionGenerator, FishCompletionGenerator, PowerShellCompletionGenerator, ZshCompletionGenerator

### Community 6 - "Any"
Cohesion: 0.05
Nodes (18): TestValidationError, ValidationError, _check_refine_arguments(), _merge_points_ev(), _check_positive(), _fmt_range(), _merge(), _Merged (+10 more)

### Community 7 - "_make_cubic"
Cohesion: 0.04
Nodes (10): TestBasicSetupAndInitialization, TestDuplicatedFormulasIndependence, TestEdgeCaseInputValues, TestEdgeCasesAndErrorHandling, TestH2OProperties, TestInputValidationAndErrorHandling, TestPropertyConsistency, TestSiO2Properties (+2 more)

### Community 8 - "constants.py"
Cohesion: 0.07
Nodes (10): apply_palette_to_item(), apply_palette_to_widget(), current_palette(), legend_kwargs(), _recolor_axis_label(), _restyle_legend(), KKPlot, PlotCanvas (+2 more)

### Community 9 - "NumpyBackend"
Cohesion: 0.06
Nodes (20): TestScalarEnergyFallback, _calculate_molecular_properties(), calculate_multiple_xray_properties(), _calculate_single_material_xray_properties(), _coerce_energy_array(), _convert_energy_input(), _create_process_formula_function(), process_formula() (+12 more)

### Community 10 - "test_exceptions.py"
Cohesion: 0.05
Nodes (17): TestAtomicDataLookup, clear_lru_caches(), _mock_get_element(), TestCacheResultConsistency, worker(), TestConcurrentGetAtomicNumber, worker(), cache_clearer() (+9 more)

### Community 11 - "XRayResult"
Cohesion: 0.13
Nodes (39): _edge_spectrum(), test_asf_input_is_unchanged_and_unknown_type_raises(), test_beta_round_trip_and_density_cancels_at_merge(), test_default_merge_points_are_the_overlap_of_both_ranges(), test_directly_constructed_spectrum_is_validated(), test_docstring_examples_run(), test_file_in_kev_declared_ev_raises_no_overlap(), test_flat_endpoints_raise_not_inf() (+31 more)

### Community 12 - "CompletionInstaller"
Cohesion: 0.05
Nodes (7): TestValidateChemicalFormula, TestValidateDensity, attenuation_length_cm(), __getattr__(), _get_valid_element_symbols(), validate_chemical_formula(), validate_density()

### Community 13 - "calculate_xray_properties"
Cohesion: 0.05
Nodes (9): _clear_cache(), _single(), TestCLIOutputFormatting, TestSubRefracErrorHandling, TestSubRefracSiliconProperties, TestPerformanceRegression, TestEnergyInputContract, clear_scattering_factor_cache() (+1 more)

### Community 14 - "current_palette"
Cohesion: 0.06
Nodes (9): TestFormulaParsing, TestDelegatingWrappers, TestParseFormulaBasic, TestParseFormulaDecimal, TestParseFormulaErrors, TestParseFormulaParentheses, TestRegexCompatibility, parse_formula() (+1 more)

### Community 15 - "argparse"
Cohesion: 0.05
Nodes (14): numpy_backend_after(), test_whitespace_delimited_loader_works_with_current_pandas(), TestAnomalousDispersion, TestBackendState, TestExports, TestGuiErrorHandling, task(), TestLazySubmoduleAccess (+6 more)

### Community 16 - "test_performance_benchmarks.py"
Cohesion: 0.09
Nodes (29): test_backend_is_recorded_and_chunk_size_does_not_matter(), test_boundary_energies_are_accepted_and_f2_matches_the_table(), test_default_grid_clipped_to_0p03_30_kev_and_logged(), test_docstring_examples_run(), test_empty_default_grid_after_merge_raises(), test_energies_are_copied_not_aliased(), test_energy_one_ulp_outside_the_data_range_raises(), test_energy_out_of_data_range_raises_energy_error() (+21 more)

### Community 18 - "Toast"
Cohesion: 0.07
Nodes (4): main(), StyleGuideValidator, StyleViolation, ValidationReport

### Community 19 - "MemoryMonitor"
Cohesion: 0.06
Nodes (5): TestBenchmarkComparison, TestPerformanceBenchmarks, TestPerformanceRegression, TestScalabilityBenchmarks, timer()

### Community 20 - "load_asf"
Cohesion: 0.11
Nodes (32): _args(), _eq10_corrected(), _f2_poly(), _golden(), _linear_segments(), _quadrature_f1(), test_chunk_invariance(), test_constant_band_closed_form() (+24 more)

### Community 21 - "typing"
Cohesion: 0.09
Nodes (35): _max_midpoint_error(), _refine(), si(), _spy_chunk_sizes(), test_debug_log_per_pass_and_info_total(), test_docstring_examples_run(), test_exhaustion_flagged_and_logged(), test_heavy_element_segment_grid_against_upstream_emulation() (+27 more)

### Community 22 - "validate_chemical_formula"
Cohesion: 0.07
Nodes (15): no_progress_config(), TestProcessSingleCalculation, chunk_iterator(), _filter_dataframe_fields(), _get_energy_point_data(), _initialize_progress_bar(), _prepare_energies_array(), _prepare_result_data() (+7 more)

### Community 23 - "TestJaxBackendParity"
Cohesion: 0.07
Nodes (15): TestResultFormatting, _format_array_field_single(), _format_as_csv(), _format_as_json(), _format_filtered_table(), _format_material_properties(), _format_multiple_energies(), _format_multiple_energy_section() (+7 more)

### Community 24 - "test_utilities.py"
Cohesion: 0.07
Nodes (14): setup(), ci_threshold_multiplier(), pytest_collection_modifyitems(), kk_log(), test_configure_logging_writes_file(), test_get_logger_child_inherits_config(), build_parser(), main() (+6 more)

### Community 26 - "TestBasicExportFunctionality"
Cohesion: 0.05
Nodes (36): ADR-002: PyQtGraph vs Matplotlib for GUI Plotting, Appendix: Feature Parity Checklist, Consequences, Context, Decision, Migration Details, Negative, Positive (+28 more)

### Community 27 - "scripts/test_docs.py"
Cohesion: 0.08
Nodes (18): Data Handling Module API Doc, atomic_cache (92-element preload, warm_up_cache, get_cache_stats), batch_processing (BatchConfig, calculate_batch_properties, CSV I/O), compound_analysis (compound families, recommended warming elements), CXRO / NIST / Henke .nff scattering factor data (0.03-30 keV, PCHIP), _collect_compound_elements(), _collect_family_compound_elements(), get_atomic_data_fast() (+10 more)

### Community 28 - "test_concurrent_cache.py"
Cohesion: 0.08
Nodes (10): main(), TestCheckGpuAvailability, TestGetDeviceInfo, TestGetSystemCudaVersion, check_gpu_availability(), check_plugin_conflicts(), get_device_info(), get_gpu_info() (+2 more)

### Community 29 - "formatting.py"
Cohesion: 0.07
Nodes (5): TestBasicExportIntegration, TestBasicExportFunctionality, export_to_csv(), export_to_json(), __getattr__()

### Community 30 - "batch_processing.py"
Cohesion: 0.07
Nodes (4): TestValidateCalculationParameters, TestValidateEnergyRange, validate_calculation_parameters(), validate_energy_range()

### Community 31 - "formula.py"
Cohesion: 0.09
Nodes (6): TestBatchProcessorEquivalence, TestBatchConfig, TestBatchInputValidation, TestBatchPartialFailure, BatchConfig, calculate_batch_properties()

### Community 34 - "test_cli.py"
Cohesion: 0.07
Nodes (9): form(), _get_app(), qt_app(), TestDensityValidation, TestEnergyGridValidation, TestFormulaValidation, TestFormValues, TestPlotAxisLabelTheming (+1 more)

### Community 35 - "test_deep_debug_regressions.py"
Cohesion: 0.11
Nodes (22): _computation(), _pen_colors(), plot(), qt_app(), test_plot_draws_curves_on_unequal_grids(), test_plot_follows_theme_toggle(), test_plot_hides_missing_curves(), test_plot_log_x_rule() (+14 more)

### Community 36 - "TestCICDIntegration"
Cohesion: 0.10
Nodes (13): _run_smoke(), test_compute_multiple_reports_progress_to_100(), _ensure_app(), test_export_csv_handles_io_error(), test_gui_smoke_offscreen(), test_gui_smoke_threaded_offscreen(), test_duplicate_formula_different_density_not_dropped(), enable_offscreen_quiet_env() (+5 more)

### Community 37 - "EnvironmentInfo"
Cohesion: 0.07
Nodes (8): TestCLIIntegration, TestMainFunction, TestInstallCompletionMain, cmd_completion(), cmd_install_completion(), cmd_uninstall_completion(), main(), install_completion_main()

### Community 38 - "xraylabtool/kk/__init__.py"
Cohesion: 0.12
Nodes (5): _parse_csv(), _run(), test_empty_default_grid_after_merge_exits_1(), TestKKCli, _write_edge()

### Community 40 - "TestEnergySweepSi"
Cohesion: 0.10
Nodes (4): MockXRayResult, TestFormatXRayResult, _format_as_json(), format_xray_result()

### Community 41 - "create_scattering_factor_interpolators"
Cohesion: 0.08
Nodes (35): Calculators Module (API docs), xraylabtool.calculators.cache, calculate_single_material_properties, calculate_xray_properties, xraylabtool.calculators.core, Critical angle theta_c = sqrt(2 delta), xraylabtool.calculators.derived_quantities, xraylabtool.calculators.kernels (+27 more)

### Community 42 - "test_gui_widgets.py"
Cohesion: 0.06
Nodes (4): TestEnvironmentIntegration, TestEnvironmentProperties, TestEnvironmentType, EnvironmentType

### Community 43 - "calculate_single_material_properties"
Cohesion: 0.12
Nodes (7): cmd_batch(), _convert_result_to_dict(), _parse_batch_data(), _process_batch_materials(), _save_batch_results(), _validate_batch_input(), create_batch_progress_tracker()

### Community 44 - "test_file_operations.py"
Cohesion: 0.09
Nodes (5): TestCompletionInstallToTempDir, TestCompletionScriptGeneration, CompletionManager, get_global_options(), get_xraylabtool_commands()

### Community 45 - "atomic_cache.py"
Cohesion: 0.10
Nodes (4): TestBatchProcessingMemoryManagement, TestCacheManagement, TestMemoryMonitor, MemoryMonitor

### Community 46 - "CalculationWorker"
Cohesion: 0.08
Nodes (14): arrays_close(), assert_arrays_close(), assert_values_close(), clean_cache_around(), wrapper(), expect_no_warnings(), expect_warnings(), get_system_info() (+6 more)

### Community 47 - "installer.py"
Cohesion: 0.10
Nodes (6): apply_pyqtgraph_theme(), apply_styles(), apply_theme(), ColorPalette, get_qss(), ThemeManager

### Community 48 - "MaterialComparator"
Cohesion: 0.08
Nodes (7): _check_element(), cold_cache(), test_si_o_au_interpolators_are_independent(), TestPchipAu, TestPchipO, TestPchipSi, create_scattering_factor_interpolators()

### Community 49 - "test_utils.py"
Cohesion: 0.08
Nodes (7): BasePerformanceTest, generate_test_materials(), TestArrayOptimization, TestBulkAtomicDataOptimization, TestDeprecationWarningOptimization, TestElementPathOptimization, TestInterpolatorCaching

### Community 50 - "validate_density"
Cohesion: 0.11
Nodes (13): add_atomic_command(), add_batch_command(), add_bragg_command(), add_calc_command(), add_compare_command(), add_completion_command(), add_convert_command(), add_formula_command() (+5 more)

### Community 51 - "test_optimization_validation.py"
Cohesion: 0.08
Nodes (6): completion_main(), create_completion_parser(), handle_completion_command(), show_completion_info(), handle_new_completion_command(), test_completion_installation()

### Community 52 - "fixture"
Cohesion: 0.09
Nodes (30): Linear absorption coefficient mu = 4 pi beta / lambda, Contributing atomic data (citation, range, precision, format), Performance and Optimization guide, Batch tiers (sequential vs chunked Process/ThreadPoolExecutor), Cache warming (warm_up_cache, warm_cache_for_compounds), Energy arrays instead of loops, Energy grid design (log spacing, dense near absorption edges), Atomic Scattering Factor Data (+22 more)

### Community 53 - ".calculate_derived_quantities"
Cohesion: 0.09
Nodes (30): Kramers-Kronig (kk) API docs, xraylabtool.kk.api (high-level API), AsfDatabase, compound_spectrum, xraylabtool.kk.database, Eq. (10) a_-3 sign erratum (Table 1), evaluate_f2, kk_calculate_real (+22 more)

### Community 54 - "validators.py"
Cohesion: 0.06
Nodes (4): cold_cache(), TestGoldenAu, TestGoldenDerivedQuantities, TestGoldenScatteringFactors

### Community 56 - "SharedHelpersMixin"
Cohesion: 0.09
Nodes (3): TestBackwardCompatibility, TestLegacyInstallCompletionMain, legacy_install_completion_main()

### Community 57 - "device.py"
Cohesion: 0.19
Nodes (20): app(), _small_run(), test_energies_outside_cxro_range_reach_the_service(), test_energy_range_controls_reach_the_service(), spy(), test_export_csv_columns(), test_export_csv_io_error_is_reported(), test_export_file_name_uses_computed_formula() (+12 more)

### Community 58 - "TestFullPipelineSiO2"
Cohesion: 0.12
Nodes (14): _public_objects(), test_docs_cite_the_paper_and_upstream_licence(), test_docs_pages_toctree_linked(), test_docs_state_the_eq10_erratum(), test_every_public_kk_name_is_documented(), test_public_docstrings_have_examples_and_references(), test_transform_docstrings_cite_paper_equations(), test_helper_delta_beta_match_closed_form() (+6 more)

### Community 62 - "unit"
Cohesion: 0.10
Nodes (6): _calc(), _clear_cache(), TestComplexFormula, TestElectronCountGolden, TestExactCapturedValues, TestMolecularWeightGolden

### Community 63 - "xraylabtool/utils.py"
Cohesion: 0.09
Nodes (5): TestAtomicCommand, TestBatchCommand, TestBraggCommand, cmd_atomic(), cmd_bragg()

### Community 65 - "ndarray"
Cohesion: 0.16
Nodes (12): check_accessibility(), check_documentation_coverage(), Colors, main(), print_section(), print_status(), run_style_checks(), test_doctests() (+4 more)

### Community 67 - "tests/conftest.py"
Cohesion: 0.13
Nodes (10): qt_app(), _run_worker(), test_worker_does_not_emit_finished_on_exception(), test_worker_emits_error_on_exception(), bad_fn(), test_worker_emits_result_on_success(), test_worker_injects_progress_cb_when_accepted(), test_worker_passes_kwargs() (+2 more)

### Community 69 - "Changelog (docs)"
Cohesion: 0.11
Nodes (3): TestBasicAnalysis, compare_materials(), find_absorption_edges()

### Community 71 - "get_bulk_atomic_data"
Cohesion: 0.09
Nodes (21): ADR-003: float64 Preservation Policy, Appendix: Numerical Precision Requirements by Function, Consequences, Context, Current Codebase float64 Usage, Decision, Enforcement Strategy, JAX float32 Default (+13 more)

### Community 72 - "_OpsProxy"
Cohesion: 0.12
Nodes (22): energy_to_wavelength, parse_formula, utils sub-package, wavelength_to_energy, CLI Reference, atomic command, batch command, batch CSV format (formula, density, energy) (+14 more)

### Community 73 - "scattering_data.py"
Cohesion: 0.10
Nodes (12): angle_from_q(), bragg_angle(), d_spacing_cubic(), xraylabtool.device, energy_to_wavelength(), get_atomic_data(), load_atomic_data(), xraylabtool.logging_utils (+4 more)

### Community 75 - "typing_extensions.py"
Cohesion: 0.13
Nodes (6): TestCalcCommand, cmd_calc(), _determine_output_format(), _print_calc_verbose_info(), _save_or_print_output(), _validate_calc_inputs()

### Community 77 - "TestCodeQuality"
Cohesion: 0.12
Nodes (7): calculate_derived_quantities(), calculate_scattering_factors(), _derived_quantities_kernel(), _get_derived_kernel(), _get_scattering_kernel(), _scattering_math_kernel(), InterpolatorProtocol

### Community 78 - "TestXRayResultTyping"
Cohesion: 0.10
Nodes (22): GUI (API docs), xraylabtool.gui.logging_filters, xraylabtool.gui.main_window, widgets.material_form, widgets.material_table, main_window.multi_tab, widgets.plot_canvas, widgets.scrollbar_helper (+14 more)

### Community 79 - "TestCrystallographicCalculations"
Cohesion: 0.11
Nodes (21): backend sub-package (NumPy/JAX abstraction), Backend abstraction layer (NumPy/JAX, xraylabtool.backend), Characterization (golden-value) tests, Platform and Python version compatibility matrix, InterpolationFactory (scipy PCHIP to interpax), GUI plotting migrated from matplotlib to PyQtGraph, set_backend, v0.4.0 (2026-04-07) (+13 more)

### Community 80 - "pytest"
Cohesion: 0.10
Nodes (22): Changelog, Batch API fixes (ordering, timeouts, process-pool fallback, formula@density keys), CLI aliases xtool / xlt and GUI aliases xtool-gui / xlt-gui, conda-recipe/meta.yaml (imewei channel), Export full-precision and CSV/JSON fixes, jax/jaxlib version bounds (<0.11 added in 0.4.5, removed in 0.4.7), Keep a Changelog / Semantic Versioning format, Lazy loader RecursionError fix (importlib) (+14 more)

### Community 81 - "run_type_check.py"
Cohesion: 0.11
Nodes (20): Analytic per-interval integration (Eqs. 3-4), Biggs and Lighthill high-energy cross-sections (to 500 keV), Cauchy principal value integral, Conventional KK numerical methods (FFT, Euler-Maclaurin, Gaussian, TANH), Zero-to-infinity convergence conditions on polynomial coefficients, Cross et al. difference-spectrum KK method, Double-interval integration across x=E singularity (Eq. 5), Closed-form f1(E) for piecewise Laurent f2 (Eq. 7, Eq. 10) (+12 more)

### Community 82 - "calculate_transmission"
Cohesion: 0.10
Nodes (7): TestAtomicDataCacheIsImmutable, get_bulk_atomic_data(), get_cached_elements(), is_element_cached(), _smart_cache_warming(), _warm_priority_cache(), _background_cache_warming()

### Community 83 - "validate_calculation_parameters"
Cohesion: 0.11
Nodes (3): TestEnergyError, TestBoundaryConditions, EnergyError

### Community 84 - "calculate_properties_from_factors"
Cohesion: 0.10
Nodes (8): assert_calculation_result_valid(), assert_memory_usage_reasonable(), assert_performance_within_threshold(), batch_config_performance(), batch_config_small(), pytest_addoption(), pytest_collection_modifyitems(), pytest_configure()

### Community 86 - "PerformanceBenchmark"
Cohesion: 0.14
Nodes (6): TestFormulaCommand, cmd_formula(), _get_atomic_data(), _output_formula_results(), _print_formula_results(), _process_formula()

### Community 87 - "test_compound_analysis.py"
Cohesion: 0.15
Nodes (16): test_load_absolute_path(), test_load_csv_file(), test_load_csv_space_separated_fallback(), test_load_empty_nff_raises_data_file_error(), test_load_malformed_nff_raises_data_file_error(), test_load_nff_energy_column_monotone(), test_load_nff_values_are_finite(), test_load_nonexistent_file_raises_file_not_found() (+8 more)

### Community 88 - "FishCompletionGenerator"
Cohesion: 0.18
Nodes (8): _load_golden(), _merged(), test_kernel_on_merged_spectrum_matches_golden_f1_points(), test_merged_spectrum_matches_upstream_golden(), test_nearedge_result_matches_upstream_golden(), test_matches_upstream_kk_pp_golden(), assert_kk_close(), kk_rounding_scale()

### Community 91 - "Contributing Guide"
Cohesion: 0.12
Nodes (19): widgets.kk_plot (Kramers-Kronig Plot), main_window.kk_tab, CalculationError, EnergyError, FormulaError, kk sub-package (Kramers-Kronig, kkcalc port), ValidationError, XRayLabToolError (+11 more)

### Community 92 - "XRayLabTool Documentation (index)"
Cohesion: 0.15
Nodes (18): Kramers-Kronig Guide, Near-edge input types (nexafs, beta, asf), kkcalc attribution (zlib/libpng, Watts and Lauk), Near-edge file format, Watts 2014, Opt. Express 22(19), 23628 (piecewise Laurent polynomial KK), Documentation toctree (user guide, scientific background, API, architecture, development), Kramers-Kronig Transform (physics page), Accuracy validation (upstream parity, quadrature, Lorentz oscillator, CXRO comparison, conditioning) (+10 more)

### Community 93 - "X-ray Optics Fundamentals"
Cohesion: 0.18
Nodes (6): get_optimal_chunk_size(), is_complex_array(), is_energy_array(), is_real_array(), validate_complex_array(), validate_energy_array()

### Community 95 - "xraylabtool/utils.py"
Cohesion: 0.13
Nodes (18): Interfaces Module API Doc, CLI commands (calc, batch, compare, convert, formula, atomic, bragg, list, completion, install-completion, uninstall-completion), CLI error handling messages, CLI modules (parser, formatting, utils), completion_v2 virtual-environment-centric shell completion, CLI output formats (table, csv, json), Supported shells (bash, zsh, fish, PowerShell), Validation Module API Doc (+10 more)

### Community 97 - "compound_analysis.py"
Cohesion: 0.14
Nodes (3): TestCalculationSpeedBenchmarks, calculation_worker(), TestPerformanceRegressionDetection

### Community 99 - "calculate_critical_angle"
Cohesion: 0.16
Nodes (4): build(), main(), main(), _venv_python()

### Community 100 - "clear_scattering_factor_cache"
Cohesion: 0.20
Nodes (7): _outcome(), _result(), test_kk_result_arrays_are_read_only_copies(), test_kk_result_rejects_inconsistent_state(), test_refine_outcome_arrays_are_read_only_copies(), test_refine_outcome_rejects_inconsistent_state(), __getattr__()

### Community 104 - "FormulaError"
Cohesion: 0.15
Nodes (3): decorator(), wrapper(), MemoryProfiler

### Community 106 - "ADR-002: PyQtGraph vs Matplotlib for GUI Plotting"
Cohesion: 0.17
Nodes (14): Shell Completion Guide, Completion caching system (~/.xraylabtool/cache), Completion for xtool alias, Completion features (command, option, path, value), CompletionInstaller, xraylabtool.interfaces.completion_v2, Conda/Mamba install and completion workflow, Environment detection (venv, conda, mamba, Poetry, Pipenv) (+6 more)

### Community 107 - "FastXRayCalculationEngine"
Cohesion: 0.25
Nodes (6): analyze_mypy_output(), check_core_modules(), check_type_definitions(), main(), run_mypy_check(), validate_cache_performance()

### Community 108 - "Shell Completion Guide"
Cohesion: 0.12
Nodes (7): cpu_count(), energy_arrays(), mock_low_memory(), performance_thresholds(), temp_directory(), test_energies_medium(), test_energies_small()

### Community 115 - "TestPerformanceBenchmarks"
Cohesion: 0.18
Nodes (14): Attenuation length vs energy plot for SiO2 (8-12 keV, 50 points, rising curve ~13 to ~44 cm x0.001), Compute primary action button, Energy to Wavelength converter widget (keV and Angstrom, E to lambda buttons), Footer buttons: Log Path and Light Mode (theme toggle), Inline validation feedback (Formula valid, Composition Si:1 O:2, Density valid, grid summary), Material input panel (formula, density, energy grid, log-spaced option, Compute button), Material summary table (formula, MW, density, electron density, total electrons), Plot toolbar (property selector, Log X/Log Y, Save Plot PNG, Export CSV) (+6 more)

### Community 119 - "CalculationError"
Cohesion: 0.14
Nodes (5): TestPerformanceBenchmarks, energy_sweep(), complex_formula(), multi_calculation(), single_calculation()

### Community 122 - "test_golden_pipeline.py"
Cohesion: 0.23
Nodes (5): _json_numpy_default(), save_calculation_results(), _save_dict_as_csv(), _save_results_as_csv(), _save_results_as_json()

### Community 124 - "TestBenchmarkComparison"
Cohesion: 0.15
Nodes (11): Adaptive refinement (refine_tolerance), xraylabtool kk CLI command, kk_calculate_real, KKResult, load_nearedge, Merge points (affine match of near-edge data to tabulated f2), NearEdgeDataError, NearEdgeSpectrum (+3 more)

### Community 125 - "TestEnvironmentDetection"
Cohesion: 0.24
Nodes (5): cmd_kk(), _fail(), _format_output(), _parse_energies(), _parse_merge()

### Community 127 - "cmd_convert"
Cohesion: 0.32
Nodes (8): _checkout(), _eval_points(), _finite(), _git(), main(), _poly(), _save(), _transform_case()

### Community 134 - "kk.py"
Cohesion: 0.24
Nodes (5): cmd_compare(), _parse_materials(), _result_to_dict(), _write_report(), _write_table()

### Community 136 - "memory_test"
Cohesion: 0.20
Nodes (12): Getting Started Guide, xraylabtool batch CSV processing, calculate_single_material_properties (usage), calculate_xray_properties (usage), gpu_cuda12 / gpu_cuda13 extras, Installation (pip, conda, uv, GPU extras), Performance tips (preloaded elements, batching, energy arrays), Virtual-environment-centric shell completion (+4 more)

### Community 137 - "test_transform.py"
Cohesion: 0.18
Nodes (9): Low-level kk functions (compound_spectrum, kk_real_spectrum, evaluate_f2, to_asf_scale, merge_spectra, refine_spectrum), Closed-form interval integral (corrected Eq. 10), Kramers-Kronig dispersion relation f1 from f2 (Eq. 1), kk_piecewise_poly, Singularities at knots and end points (decision D5), Piecewise Laurent polynomial representation of f2 (Eq. 8), Relativistic correction Z* (Eq. 9), Six-step KK workflow mapped to implementation (+1 more)

### Community 140 - "XRayLabToolError (base exception)"
Cohesion: 0.18
Nodes (6): memory_test(), performance_test(), decorator(), wrapper(), requires_memory(), decorator()

### Community 144 - "typing_extensions.py"
Cohesion: 0.22
Nodes (11): Available Scripts, Example Output:, Integration:, Maintenance Notes, Output:, Requirements:, `run_type_check.py` - Type Checking, Scripts Directory (+3 more)

### Community 149 - "TestFundamentalConstants"
Cohesion: 0.22
Nodes (4): critical_angle_degrees(), _isclose(), validate_constants(), wavelength_angstrom_to_energy()

### Community 150 - "get_gpu_info"
Cohesion: 0.20
Nodes (9): ADR-001: JAX vs NumPy Computation Backend, Appendix: Functions that MUST NOT be JIT-compiled, Appendix: Functions to JIT-Compile (Priority Order), Consequences, Context, Decision, Negative, Positive (+1 more)

### Community 151 - "test_type_safety_config.py"
Cohesion: 0.22
Nodes (9): JAX Architecture & Design, JAX array immutability and lazy evaluation, Automatic differentiation (grad), Debugging and testing JAX code (disable jit, jax.debug.print, JIT-vs-eager parity), Future JAX features for v0.5+ (autodiff, vmap, distributed), GPU acceleration and device placement, JAX library, JAX configuration (jax_disable_jit, jax_platforms, JAX_PLATFORMS env) (+1 more)

### Community 154 - "TestKKCli"
Cohesion: 0.33
Nodes (4): _sphinx(), test_kk_docs_build_adds_no_warnings(), test_kk_docs_doctests_pass(), slow()

### Community 156 - "Decision: replace matplotlib with PyQtGraph for interactive plots"
Cohesion: 0.29
Nodes (3): TestFindSimilarCompounds, _collect_similar_compound_elements(), find_similar_compounds()

### Community 161 - "environment.py"
Cohesion: 0.22
Nodes (9): Runtime NumPy/JAX backend switching (set_backend), Modular package architecture (backend, calculators, data_handling, interfaces, io, validation, gui), XRayLabTool Documentation (master index), Citation (BibTeX entry), CXRO/NIST atomic scattering databases, Key features (performance, physics, architecture, CLI, data handling), Target audience (synchrotron scientists, materials researchers, optics developers), Data sources (Henke, CXRO, NIST XCOM, theory) (+1 more)

### Community 162 - "parse_chemical_formula"
Cohesion: 0.28
Nodes (8): PySide6 GUI (Single, Multiple, Kramers-Kronig tabs), GUI Kramers-Kronig tab, scripts/check_version.py (version consistency), scripts/gpu_check.py (GPU diagnostics), scripts/gui_headless_smoke.py (offscreen GUI smoke check), Makefile targets and CI integration of scripts, scripts/run_type_check.py (mypy wrapper), scripts/test_install.py (install smoke test)

### Community 168 - "TestBatchProcessing"
Cohesion: 0.25
Nodes (7): Files, Maintenance, `performance_baselines.json`, `sample_materials.json`, Test Data Directory, `test_energies.json`, Usage in Tests

### Community 169 - "Testing Guide"
Cohesion: 0.29
Nodes (8): 30 keV tabulation cut (dashed vertical line), C (carbon) curve, Figure: f1 overestimated by truncating f2 integration at 30 keV, Fe (iron) curve, Kramers-Kronig f2-to-f1 integral truncated to tabulated range, Pb (lead) curve, f1 truncation error (f1_trunc - f1_full) vs photon energy, log-log, Result: truncation error grows with Z (C ~1e-4, Fe ~0.1, Pb ~2.5 e/atom), flat at low energy, rises sharply near 30 keV

### Community 185 - "analyze_element_associations"
Cohesion: 0.29
Nodes (3): clean_cache(), enforce_test_isolation(), setup_test_environment()

### Community 186 - "find_similar_compounds"
Cohesion: 0.29
Nodes (3): formula(), _format_as_csv(), _format_as_table()

### Community 196 - "_calculate_single_material_xray_properties"
Cohesion: 0.40
Nodes (3): kk module limits (no tail, conditioning, pole skipping, wrong energy-unit hazard), Biggs-Lighthill photoabsorption fits (30-500 keV), Integration limits and 10 eV-500 keV data range

## Knowledge Gaps
- **123 isolated node(s):** `Completion caching system (~/.xraylabtool/cache)`, `Completion for xtool alias`, `Completion features (command, option, path, value)`, `Conda/Mamba install and completion workflow`, `Migration from legacy system-wide completion` (+118 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 2045 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **129 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `v0.4.7 (2026-10-23)` connect `pytest` to `StyleGuideValidator`, `create_scattering_factor_interpolators`, `Contributing Guide`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `MainWindowProtocol` (e.g. with `KKTabMixin` and `MultiTabMixin`) actually correct?**
  _`MainWindowProtocol` has 12 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Completion caching system (~/.xraylabtool/cache)`, `Completion for xtool alias`, `Completion features (command, option, path, value)` to the rest of the system?**
  _123 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `MainWindowProtocol` be split into smaller, more focused modules?**
  _Cohesion score 0.04085045389393215 - nodes in this community are weakly interconnected._
- **Why does `calculate_single_material_properties()` connect `calculate_xray_properties` to `services.py`, `set_backend`, `_make_cubic`, `NumpyBackend`, `argparse`, `MemoryMonitor`, `validate_chemical_formula`, `style.py`, `formatting.py`, `formula.py`, `test_deep_debug_regressions.py`, `TestCICDIntegration`, `kernels.py`, `TestBackwardCompatibility`, `calculate_single_material_properties`, `Files`, `test_utils.py`, `validators.py`, `TestFullPipelineSiO2`, `Changelog (docs)`, `typing_extensions.py`, `calculate_transmission`, `compound_analysis.py`, `CalculationError`, `TestMultiMaterialEquivalence`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Should `format_xray_result` be split into smaller, more focused modules?**
  _Cohesion score 0.036114911080711354 - nodes in this community are weakly interconnected._
- **Why does `Changelog` connect `pytest` to `Contributing Guide`, `TestXRayResultTyping`, `TestCrystallographicCalculations`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._