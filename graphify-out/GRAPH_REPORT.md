# Graph Report - pyXRayLabTool  (2026-10-26)

## Corpus Check
- 207 files · ~198,195 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 112 file(s) not represented in the graph (top: .nff 93, .npz 12, .ipynb 2)

## Summary
- 5457 nodes · 11988 edges · 267 communities (139 shown, 128 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 869 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `80ca8564`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- MainWindowProtocol
- xraylabtool/utils.py
- test_jax_backend_parity.py
- compound_spectrum
- ValidationError
- BashCompletionGenerator
- nearedge.py
- _make_cubic
- constants.py
- _calculate_single_material_xray_properties
- test_concurrent_cache.py
- XRayResult
- validate_density
- clear_scattering_factor_cache
- parse_formula
- test_deep_debug_regressions.py
- test_api.py
- parse_formula
- Toast
- timer
- load_asf
- typing
- validate_chemical_formula
- TestJaxBackendParity
- test_utilities.py
- style.py
- TestBasicExportFunctionality
- data_handling/__init__.py
- test_concurrent_cache.py
- TestBasicExportFunctionality
- batch_processing.py
- BatchConfig
- CompletionInstaller
- OverlayScrollbarMarginHelper
- test_cli.py
- test_deep_debug_regressions.py
- TestCICDIntegration
- main
- TestKKCli
- single_tab.py
- TestEnergySweepSi
- create_scattering_factor_interpolators
- TestEnvironmentIntegration
- typing
- installer.py
- atomic_cache.py
- CalculationWorker
- installer.py
- MaterialComparator
- test_utils.py
- test_device_spectrum.py
- main
- fixture
- .calculate_derived_quantities
- TestGoldenSi
- Utilities Module
- legacy_install_completion_main
- device.py
- kk/test_docs.py
- cmd_calc
- validate_chemical_formula
- argparse
- unit
- load_asf
- Any
- ndarray
- load_asf
- tests/conftest.py
- TestInstallCompletionCommand
- MaterialComparator
- xraylabtool/__init__.py
- get_bulk_atomic_data
- _OpsProxy
- scattering_data.py
- FastAtomicDataProvider
- set_backend
- legacy_install_completion_main
- _device.py
- TestXRayResultTyping
- test_device_kernel.py
- pytest
- run_type_check.py
- comparator.py
- EnergyError
- tests/conftest.py
- MemoryProfiler
- PerformanceBenchmark
- test_compound_analysis.py
- kk_gpu_check.py
- TestEnvironmentProperties
- Version 0.4.7 (2026-10-23)
- kk_calculate_real
- XRayLabTool Documentation (index)
- typing_extensions.py
- refine.py
- xraylabtool/utils.py
- test_exceptions.py
- compound_analysis.py
- TableFormatter
- build_kk_data.py
- clear_scattering_factor_cache
- EnergyError
- parse_energy_string
- .uninstall
- FormulaError
- TestCIWorkflowSimulation
- ADR-002: PyQtGraph vs Matplotlib for GUI Plotting
- FastXRayCalculationEngine
- test_device_merge.py
- _make_cubic
- test_performance_benchmarks.py
- TestGoldenAu
- TestGoldenFe2O3
- TestGoldenSiO2
- BashCompletionGenerator
- TestPerformanceBenchmarks
- test_device_refine.py
- numpy
- XRayLabToolError
- TestPerformanceBenchmarks
- TestExtremeConditions
- api.py
- JIT compilation
- ADR-003: float64 Preservation Policy
- xraylabtool/kk/__init__.py
- json
- test_cli_edge_cases.py
- cmd_convert
- test_cli.py
- test_oracle.py
- atomic_cache.py
- __getattr__
- core.py
- bucket
- BaseIntegrationTest
- bragg_angle
- Getting Started Guide
- Closed-form interval integral (corrected Eq. 10)
- generate_test_energies
- calculators.core (calculate_single_material_properties, calculate_xray_properties)
- XRayLabToolError (base exception)
- CalculationError
- OverlayScrollbarMarginHelper
- kernels.py
- typing_extensions.py
- TestKKCliEndToEnd
- constants.py
- fixture
- kk_calculate_real
- validate_chemical_formula
- get_gpu_info
- test_type_safety_config.py
- _parse_formula
- ADR-001: JAX vs NumPy Computation Backend
- tempfile
- test_warm_up.py
- Decision: replace matplotlib with PyQtGraph for interactive plots
- JAX Architecture & Design (v0.4.0)
- test_kk_plot.py
- BatchProcessingError
- pytest
- kk_compiles.py
- time
- compute_kk
- v0.4.0 release (2026-04-07)
- TestUnitConversionConstants
- test_compound_analysis.py
- xraylabtool/__init__.py
- TestBatchProcessing
- Testing Guide
- api.py
- test_midpoint_loop_never_evaluates_more_than_block_per_call
- test_validators.py
- TestBenchmarkComparison
- ResultValidator
- stray_conversions
- get_elements_for_compound
- TestDtypePreservation
- TestGoldenAu
- .sum
- TestBatchConfig
- cmd_list
- TestCLICompatibility
- parametrize
- get_gpu_info
- TestPhysicalRealism
- compare.py
- TestFundamentalConstants
- gen_kk_prejax_baseline.py
- calculators/__init__.py
- argparse
- kk/test_docs.py
- test_formula_parsing.py
- normalize_intensity
- calculate_multiple_xray_properties
- transfer_counts
- TestEnvironmentProperties
- TestRegexCompatibility
- Decision: enforce float64 globally; float32 is a correctness bug
- refine_spectrum
- evaluate_f2
- test_kk_docs_build.py
- typing
- TestBatchConfig
- test_core.py
- TestGuiErrorHandling
- ADR-005: The Kramers-Kronig Package Is JAX-Only
- TestFormulaValidation
- test_numerical_stability.py
- re
- gc
- TestScalarEnergyFallback
- TestBackwardCompatibility
- get_bulk_atomic_data
- Data Handling Module API Doc
- check_gpu_availability
- test_gui_kk_worker_end_to_end.py
- integration/__init__.py
- performance/__init__.py
- unit/__init__.py
- BaseIntegrationTest
- BasePerformanceTest
- BaseUnitTest
- test_data_generator
- transform.py
- TestCLIOutputFormatting
- TestCLIFileOperations
- TestCLICompatibility
- TestEdgeCaseInputValues
- skip_if_low_memory
- TestMemoryLeakPrevention
- kk/conftest.py
- TestGoldenScatteringFactors
- large_batch_data
- material_and_density
- TestCLIInputValidation
- TestCLIPerformanceAndMemory
- performance_timer
- TestSubRefracErrorHandling
- test_threaded_calls_agree
- TestDtypePreservation
- _handle_mendeleev_error
- assert_arrays_close
- _auto_select_backend
- assert_calculation_result_valid
- mock_low_memory
- clean_cache_around
- test_compile_recorder_sees_a_fresh_function_at_two_shapes
- batch_config_small
- cpu_count
- temp_directory
- test_energies_medium

## God Nodes (most connected - your core abstractions)
1. `kk_calculate_real()` - 116 edges
2. `MainWindowProtocol` - 101 edges
3. `calculate_single_material_properties()` - 95 edges
4. `parse_formula()` - 91 edges
5. `CompletionInstaller` - 69 edges
6. `kk_real_spectrum()` - 69 edges
7. `ValidationError` - 61 edges
8. `compound_spectrum()` - 54 edges
9. `Spectrum` - 51 edges
10. `merge_spectra()` - 50 edges

## Surprising Connections (you probably didn't know these)
- `Alternatives considered` --references--> `ArrayBackend`  [INFERRED]
  docs/architecture/005-adr-kk-jax-only.md → xraylabtool/backend/array_ops.py
- `Context` --references--> `set_backend()`  [INFERRED]
  docs/architecture/005-adr-kk-jax-only.md → xraylabtool/backend/array_ops.py
- `Evidence` --references--> `kk_calculate_real()`  [INFERRED]
  docs/architecture/005-adr-kk-jax-only.md → xraylabtool/kk/api.py
- `Consequences` --references--> `set_backend()`  [INFERRED]
  docs/architecture/005-adr-kk-jax-only.md → xraylabtool/backend/array_ops.py
- `Decision` --references--> `enable_persistent_cache()`  [INFERRED]
  docs/architecture/005-adr-kk-jax-only.md → xraylabtool/kk/_device.py

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

## Communities (267 total, 128 thin omitted)

### Community 0 - "MainWindowProtocol"
Cohesion: 0.05
Nodes (4): KKTabMixin, MultiTabMixin, MainWindowProtocol, SingleTabMixin

### Community 1 - "xraylabtool/utils.py"
Cohesion: 0.05
Nodes (19): TestCrystallographicCalculations, TestUnitConversions, angle_from_q(), bragg_angle(), _check_lattice(), _convert_atomic_number_to_int(), d_spacing_cubic(), d_spacing_orthorhombic() (+11 more)

### Community 2 - "test_jax_backend_parity.py"
Cohesion: 0.07
Nodes (9): TestBackendSwitching, TestInterpolationFactory, TestOpsProxy, restore_numpy_backend(), TestJaxBackendParity, get_backend(), _OpsProxy, set_backend() (+1 more)

### Community 3 - "compound_spectrum"
Cohesion: 0.16
Nodes (14): si(), _numpy_assembly(), test_compound_coefficients_equal_the_pre_jax_baseline(), test_compound_coefficients_round_like_a_separate_multiply_and_add(), test_compound_grid_is_union_of_boundaries(), test_compound_spectrum_rejects_bad_input(), test_docstring_examples_run(), test_element_order_does_not_change_the_result() (+6 more)

### Community 4 - "ValidationError"
Cohesion: 0.05
Nodes (12): TestAtomicDataError, TestDataFileError, TestExceptionHierarchy, TestFormulaError, TestUnknownElementError, TestValidationError, AtomicDataError, DataFileError (+4 more)

### Community 5 - "BashCompletionGenerator"
Cohesion: 0.04
Nodes (7): TestBashCompletionSubstitution, test_cli_aliases_registered(), BashCompletionGenerator, CompletionGenerator, FishCompletionGenerator, PowerShellCompletionGenerator, ZshCompletionGenerator

### Community 6 - "nearedge.py"
Cohesion: 0.10
Nodes (13): _asf_scale_arguments(), _check_positive(), _fmt_range(), _interp(), merge_device(), MergeResult, MergeStatus, raise_or_log() (+5 more)

### Community 7 - "_make_cubic"
Cohesion: 0.05
Nodes (9): TestDuplicatedFormulasIndependence, TestEdgeCasesAndErrorHandling, TestH2OProperties, TestInputValidationAndErrorHandling, TestPropertyConsistency, TestSiO2Properties, TestSubRefracSiliconProperties, calculate_xray_properties() (+1 more)

### Community 8 - "constants.py"
Cohesion: 0.06
Nodes (9): apply_palette_to_item(), apply_palette_to_widget(), current_palette(), legend_kwargs(), _recolor_axis_label(), _restyle_legend(), PlotCanvas, F1F2Plot (+1 more)

### Community 9 - "_calculate_single_material_xray_properties"
Cohesion: 0.08
Nodes (11): _calculate_single_material_xray_properties(), _coerce_energy_array(), _convert_energy_input(), process_formula(), FastXRayCalculationEngine, get_calculation_engine(), _MaterialPropertiesDict, _properties_from_totals() (+3 more)

### Community 10 - "test_concurrent_cache.py"
Cohesion: 0.06
Nodes (16): TestAtomicDataLookup, clear_lru_caches(), _mock_get_element(), TestCacheResultConsistency, worker(), TestConcurrentGetAtomicNumber, worker(), cache_clearer() (+8 more)

### Community 11 - "XRayResult"
Cohesion: 0.09
Nodes (44): test_host_scale_conversion_and_merge_of_tiny_data_match_scale_one(), _edge_spectrum(), test_asf_input_is_unchanged_and_unknown_type_raises(), test_beta_round_trip_and_density_cancels_at_merge(), test_default_merge_points_are_the_overlap_of_both_ranges(), test_directly_constructed_spectrum_is_validated(), test_docstring_examples_run(), test_file_in_kev_declared_ev_raises_no_overlap() (+36 more)

### Community 13 - "clear_scattering_factor_cache"
Cohesion: 0.10
Nodes (4): _clear_cache(), cold_cache(), TestPerformanceRegression, clear_scattering_factor_cache()

### Community 14 - "parse_formula"
Cohesion: 0.07
Nodes (8): TestFormulaParsing, TestParseFormulaBasic, TestParseFormulaDecimal, TestParseFormulaErrors, TestParseFormulaParentheses, _aggregate_element_counts(), _expand_parentheses(), parse_formula()

### Community 15 - "test_deep_debug_regressions.py"
Cohesion: 0.06
Nodes (13): test_whitespace_delimited_loader_works_with_current_pandas(), TestAnomalousDispersion, TestBackendState, TestEnergyInputContract, TestExports, TestLazySubmoduleAccess, TestRepeatedMaterials, TestScatteringDataIntegrity (+5 more)

### Community 16 - "test_api.py"
Cohesion: 0.08
Nodes (27): test_backend_is_jax_and_chunk_size_does_not_matter(), test_boundary_energies_are_accepted_and_f2_matches_the_table(), test_default_grid_clipped_to_0p03_30_kev_and_logged(), test_docstring_examples_run(), test_empty_default_grid_after_merge_raises(), test_energies_are_copied_not_aliased(), test_energy_one_ulp_outside_the_data_range_raises(), test_energy_out_of_data_range_raises_energy_error() (+19 more)

### Community 18 - "Toast"
Cohesion: 0.06
Nodes (5): Colors, main(), StyleGuideValidator, StyleViolation, ValidationReport

### Community 20 - "load_asf"
Cohesion: 0.13
Nodes (29): _args(), _eq10_corrected(), _f2_poly(), _golden(), _linear_segments(), _quadrature_f1(), test_chunk_invariance(), test_constant_band_closed_form() (+21 more)

### Community 21 - "typing"
Cohesion: 0.11
Nodes (29): _max_midpoint_error(), _refine(), _spy_blocks(), wrap(), test_debug_log_per_pass_and_info_total(), test_docstring_examples_run(), test_exhaustion_flagged_and_logged(), test_invalid_refine_arguments_raise_validation_error() (+21 more)

### Community 22 - "validate_chemical_formula"
Cohesion: 0.08
Nodes (13): TestProcessSingleCalculation, chunk_iterator(), _filter_dataframe_fields(), _get_energy_point_data(), _initialize_progress_bar(), _prepare_energies_array(), _prepare_result_data(), process_batch_chunk() (+5 more)

### Community 23 - "TestJaxBackendParity"
Cohesion: 0.07
Nodes (15): TestResultFormatting, _format_array_field_single(), _format_as_csv(), _format_as_json(), _format_filtered_table(), _format_material_properties(), _format_multiple_energies(), _format_multiple_energy_section() (+7 more)

### Community 24 - "test_utilities.py"
Cohesion: 0.10
Nodes (12): setup(), test_configure_logging_writes_file(), test_get_logger_child_inherits_config(), build_parser(), main(), _bool_env(), _coerce_int_env(), configure_logging() (+4 more)

### Community 26 - "TestBasicExportFunctionality"
Cohesion: 0.05
Nodes (35): ADR-002: PyQtGraph vs Matplotlib for GUI Plotting, Appendix: Feature Parity Checklist, Consequences, Context, Decision, Migration Details, Negative, Positive (+27 more)

### Community 27 - "data_handling/__init__.py"
Cohesion: 0.06
Nodes (18): _warm_priority_cache(), _background_cache_warming(), _collect_compound_elements(), _collect_family_compound_elements(), _collect_similar_compound_elements(), FastAtomicDataProvider, get_atomic_data_fast(), get_atomic_data_provider() (+10 more)

### Community 28 - "test_concurrent_cache.py"
Cohesion: 0.10
Nodes (7): main(), TestGetDeviceInfo, TestGetSystemCudaVersion, check_plugin_conflicts(), get_device_info(), get_recommended_package(), get_system_cuda_version()

### Community 29 - "TestBasicExportFunctionality"
Cohesion: 0.07
Nodes (5): TestBasicExportIntegration, TestBasicExportFunctionality, export_to_csv(), export_to_json(), __getattr__()

### Community 31 - "BatchConfig"
Cohesion: 0.11
Nodes (6): TestBatchProcessorEquivalence, no_progress_config(), TestBatchInputValidation, TestBatchPartialFailure, BatchConfig, calculate_batch_properties()

### Community 32 - "CompletionInstaller"
Cohesion: 0.08
Nodes (3): TestSentinelMarkers, CompletionInstaller, test_completion_installation()

### Community 34 - "test_cli.py"
Cohesion: 0.10
Nodes (8): form(), _get_app(), qt_app(), TestDensityValidation, TestEnergyGridValidation, TestFormValues, TestPlotAxisLabelTheming, MaterialInputForm

### Community 35 - "test_deep_debug_regressions.py"
Cohesion: 0.13
Nodes (11): _computation(), _pen_colors(), plot(), qt_app(), test_plot_draws_curves_on_unequal_grids(), test_plot_follows_theme_toggle(), test_plot_hides_missing_curves(), test_plot_log_x_rule() (+3 more)

### Community 36 - "TestCICDIntegration"
Cohesion: 0.11
Nodes (13): _run_smoke(), test_compute_multiple_reports_progress_to_100(), _ensure_app(), test_export_csv_handles_io_error(), test_gui_smoke_offscreen(), test_gui_smoke_threaded_offscreen(), test_duplicate_formula_different_density_not_dropped(), enable_offscreen_quiet_env() (+5 more)

### Community 37 - "main"
Cohesion: 0.09
Nodes (5): TestAtomicCommand, TestBatchCommand, TestBraggCommand, cmd_atomic(), cmd_bragg()

### Community 38 - "TestKKCli"
Cohesion: 0.06
Nodes (18): _parse_csv(), _run(), test_empty_default_grid_after_merge_exits_1(), TestKKCli, _write_edge(), add_atomic_command(), add_batch_command(), add_bragg_command() (+10 more)

### Community 39 - "single_tab.py"
Cohesion: 0.11
Nodes (3): Toast, TableFormatter, MaterialTable

### Community 40 - "TestEnergySweepSi"
Cohesion: 0.08
Nodes (7): formula(), MockXRayResult, TestFormatXRayResult, _format_as_csv(), _format_as_json(), _format_as_table(), format_xray_result()

### Community 41 - "create_scattering_factor_interpolators"
Cohesion: 0.06
Nodes (40): Calculators Module (API docs), xraylabtool.calculators.cache, calculate_xray_properties, xraylabtool.calculators.core, Critical angle theta_c = sqrt(2 delta), xraylabtool.calculators.derived_quantities, xraylabtool.calculators.kernels, Legacy CamelCase accessors (DeprecationWarning) (+32 more)

### Community 43 - "typing"
Cohesion: 0.17
Nodes (7): cmd_batch(), _convert_result_to_dict(), _parse_batch_data(), _process_batch_materials(), _save_batch_results(), _validate_batch_input(), create_batch_progress_tracker()

### Community 44 - "installer.py"
Cohesion: 0.09
Nodes (5): TestCompletionInstallToTempDir, TestCompletionScriptGeneration, CompletionManager, get_global_options(), get_xraylabtool_commands()

### Community 45 - "atomic_cache.py"
Cohesion: 0.09
Nodes (5): TestBatchProcessingMemoryManagement, TestCacheManagement, TestMemoryMonitor, MemoryMonitor, _process_chunks()

### Community 46 - "CalculationWorker"
Cohesion: 0.10
Nodes (10): assert_values_close(), expect_no_warnings(), expect_warnings(), get_system_info(), log_system_info(), skip_if_insufficient_cpu(), skip_if_insufficient_memory(), stability_test() (+2 more)

### Community 47 - "installer.py"
Cohesion: 0.10
Nodes (6): apply_pyqtgraph_theme(), apply_styles(), apply_theme(), ColorPalette, get_qss(), ThemeManager

### Community 48 - "MaterialComparator"
Cohesion: 0.07
Nodes (7): _check_element(), cold_cache(), test_si_o_au_interpolators_are_independent(), TestPchipAu, TestPchipO, TestPchipSi, create_scattering_factor_interpolators()

### Community 49 - "test_utils.py"
Cohesion: 0.08
Nodes (7): BasePerformanceTest, TestArrayOptimization, TestBulkAtomicDataOptimization, TestDeprecationWarningOptimization, TestElementPathOptimization, TestInterpolatorCaching, TestNumericalStabilityOptimizations

### Community 50 - "test_device_spectrum.py"
Cohesion: 0.08
Nodes (29): Decision, _evaluate_f2_device(), test_undersized_merge_capacity_reports_the_true_count(), test_device_loop_runs_under_transfer_guard(), _poly(), _reference_f2(), test_evaluate_f2_device_edges(), test_evaluate_f2_device_is_jittable_and_bucket_independent() (+21 more)

### Community 51 - "main"
Cohesion: 0.08
Nodes (6): TestCLIIntegration, TestMainFunction, cmd_completion(), cmd_install_completion(), cmd_uninstall_completion(), main()

### Community 52 - "fixture"
Cohesion: 0.06
Nodes (44): Linear absorption coefficient mu = 4 pi beta / lambda, Platform and Python version compatibility matrix, Contributing Guide, Contributing atomic data (citation, range, precision, format), Code standards (PEP 8, 88 cols, NumPy docstrings, type hints), Performance requirements (<0.1 ms single, >100k calc/s batch, >90% cache hit), Review process (automated checks, reviewer, maintainer, squash merge), Development workflow (branching, ruff, mypy, pytest, make verify) (+36 more)

### Community 53 - ".calculate_derived_quantities"
Cohesion: 0.09
Nodes (34): Kramers-Kronig (kk) API docs, xraylabtool.kk.api (high-level API), AsfDatabase, compound_spectrum, xraylabtool.kk.database, Eq. (10) a_-3 sign erratum (Table 1), evaluate_f2, kk_calculate_real (+26 more)

### Community 55 - "Utilities Module"
Cohesion: 0.08
Nodes (3): TestEdgeCases, TestNumpyBackend, NumpyBackend

### Community 57 - "device.py"
Cohesion: 0.18
Nodes (20): app(), _small_run(), test_energies_outside_cxro_range_reach_the_service(), test_energy_range_controls_reach_the_service(), spy(), test_export_csv_columns(), test_export_csv_io_error_is_reported(), test_export_file_name_uses_computed_formula() (+12 more)

### Community 58 - "kk/test_docs.py"
Cohesion: 0.21
Nodes (8): test_helper_delta_beta_match_closed_form(), test_helper_rejects_complex_input(), test_helper_rejects_mismatched_nonfinite_or_bad_density(), test_helper_reproduces_lookup_path(), test_negative_f1_preserved_critical_angle_zero(), test_negative_f2_kept_and_attenuation_clamped(), calculate_properties_from_factors(), _xray_result_from_properties()

### Community 60 - "validate_chemical_formula"
Cohesion: 0.06
Nodes (4): TestEnvironmentInfo, TestEnvironmentType, EnvironmentInfo, EnvironmentType

### Community 62 - "unit"
Cohesion: 0.09
Nodes (7): _calc(), _clear_cache(), TestComplexFormula, TestElectronCountGolden, TestExactCapturedValues, TestMolecularWeightGolden, _calculate_molecular_properties()

### Community 63 - "load_asf"
Cohesion: 0.14
Nodes (20): test_asf_grid_spans_10ev_to_500kev(), test_asf_has_92_elements(), test_cached_tables_are_read_only(), test_im_coeffs_shape(), test_load_asf_is_cached(), test_nff_divergent_elements(), _assemble(), test_capacity_independent() (+12 more)

### Community 65 - "ndarray"
Cohesion: 0.16
Nodes (12): check_accessibility(), check_documentation_coverage(), Colors, main(), print_section(), print_status(), run_style_checks(), test_doctests() (+4 more)

### Community 66 - "load_asf"
Cohesion: 0.13
Nodes (6): _asf_f2(), _compound_segments(), AsfDatabase, _check_z(), _load_asf_cached(), ensure_x64()

### Community 67 - "tests/conftest.py"
Cohesion: 0.09
Nodes (12): qt_app(), _run_worker(), test_worker_does_not_emit_finished_on_exception(), test_worker_emits_error_on_exception(), bad_fn(), test_worker_emits_result_on_success(), test_worker_injects_progress_cb_when_accepted(), test_worker_passes_kwargs() (+4 more)

### Community 71 - "get_bulk_atomic_data"
Cohesion: 0.09
Nodes (21): ADR-003: float64 Preservation Policy, Appendix: Numerical Precision Requirements by Function, Consequences, Context, Current Codebase float64 Usage, Decision, Enforcement Strategy, JAX float32 Default (+13 more)

### Community 72 - "_OpsProxy"
Cohesion: 0.11
Nodes (23): calculate_single_material_properties, energy_to_wavelength, parse_formula, utils sub-package, wavelength_to_energy, CLI Reference, atomic command, batch command (+15 more)

### Community 73 - "scattering_data.py"
Cohesion: 0.10
Nodes (12): angle_from_q(), bragg_angle(), d_spacing_cubic(), xraylabtool.device, energy_to_wavelength(), get_atomic_data(), load_atomic_data(), xraylabtool.logging_utils (+4 more)

### Community 75 - "set_backend"
Cohesion: 0.13
Nodes (6): TestCalcCommand, cmd_calc(), _determine_output_format(), _print_calc_verbose_info(), _save_or_print_output(), _validate_calc_inputs()

### Community 77 - "_device.py"
Cohesion: 0.07
Nodes (16): report_union_sort_cost(), padded(), _log_runtime_once(), DeviceTables, _flatten_tables(), _unflatten_tables(), check_capacity(), _flatten() (+8 more)

### Community 78 - "TestXRayResultTyping"
Cohesion: 0.10
Nodes (24): GUI (API docs), widgets.kk_plot (Kramers-Kronig Plot), main_window.kk_tab, xraylabtool.gui.logging_filters, xraylabtool.gui.main_window, widgets.material_form, widgets.material_table, main_window.multi_tab (+16 more)

### Community 79 - "test_device_kernel.py"
Cohesion: 0.19
Nodes (16): _case(), _eval(), test_kernel_without_x64_raises_the_documented_error_without_a_warning(), test_kk_f1_jit_vmap(), test_kk_f1_keeps_input_shape_and_rejects_bad_block(), test_output_float64_and_shapes(), test_padding_at_a_pad_knot_is_finite_and_matches(), test_padding_neutral_at_knots() (+8 more)

### Community 80 - "pytest"
Cohesion: 0.07
Nodes (30): CalculationError, EnergyError, FormulaError, ValidationError, XRayLabToolError, Changelog, Batch API fixes (ordering, timeouts, process-pool fallback, formula@density keys), CLI aliases xtool / xlt and GUI aliases xtool-gui / xlt-gui (+22 more)

### Community 81 - "run_type_check.py"
Cohesion: 0.10
Nodes (23): Low-level kk functions (compound_spectrum, kk_real_spectrum, evaluate_f2, to_asf_scale, merge_spectra, refine_spectrum), Relativistic correction Z* (Eq. 9), Six-step KK workflow mapped to implementation, Analytic per-interval integration (Eqs. 3-4), Biggs and Lighthill high-energy cross-sections (to 500 keV), Cauchy principal value integral, Conventional KK numerical methods (FFT, Euler-Maclaurin, Gaussian, TANH), Zero-to-infinity convergence conditions on polynomial coefficients (+15 more)

### Community 82 - "comparator.py"
Cohesion: 0.10
Nodes (3): TestBasicAnalysis, compare_materials(), find_absorption_edges()

### Community 84 - "tests/conftest.py"
Cohesion: 0.10
Nodes (9): assert_memory_usage_reasonable(), assert_performance_within_threshold(), benchmark_baseline(), energy_arrays(), pytest_addoption(), pytest_collection_modifyitems(), pytest_configure(), sample_batch_data() (+1 more)

### Community 86 - "PerformanceBenchmark"
Cohesion: 0.14
Nodes (6): TestFormulaCommand, cmd_formula(), _get_atomic_data(), _output_formula_results(), _print_formula_results(), _process_formula()

### Community 87 - "test_compound_analysis.py"
Cohesion: 0.15
Nodes (17): test_load_absolute_path(), test_load_csv_file(), test_load_csv_space_separated_fallback(), test_load_empty_nff_raises_data_file_error(), test_load_malformed_nff_raises_data_file_error(), test_load_nff_energy_column_monotone(), test_load_nff_values_are_finite(), test_load_nonexistent_file_raises_file_not_found() (+9 more)

### Community 88 - "kk_gpu_check.py"
Cohesion: 0.06
Nodes (24): check_additivity(), check_bench(), check_f2_si_edge(), check_oracle(), check_padding(), check_transfers(), _f2_numpy(), guarded() (+16 more)

### Community 89 - "TestEnvironmentProperties"
Cohesion: 0.13
Nodes (3): BaseUnitTest, TestDocumentationPatterns, TestStyleCompliance

### Community 91 - "kk_calculate_real"
Cohesion: 0.11
Nodes (23): _colliding_pair(), _run(), _subprocess(), test_backend_field_is_jax(), test_cold_concurrent_first_calls_upload_the_tables_once(), test_conversion_probe_sees_a_stray_conversion(), test_default_grid_is_exact_ev_over_1000(), test_ev_collision_first_representative_kept() (+15 more)

### Community 92 - "XRayLabTool Documentation (index)"
Cohesion: 0.08
Nodes (30): Runtime NumPy/JAX backend switching (set_backend), Modular package architecture (backend, calculators, data_handling, interfaces, io, validation, gui), Kramers-Kronig Guide, Near-edge input types (nexafs, beta, asf), kkcalc attribution (zlib/libpng, Watts and Lauk), kk module limits (no tail, conditioning, pole skipping, wrong energy-unit hazard), Near-edge file format, Watts 2014, Opt. Express 22(19), 23628 (piecewise Laurent polynomial KK) (+22 more)

### Community 93 - "typing_extensions.py"
Cohesion: 0.11
Nodes (9): ensure_complex128_array(), ensure_float64_array(), get_optimal_chunk_size(), is_complex_array(), is_energy_array(), is_real_array(), optimize_array_memory_layout(), validate_complex_array() (+1 more)

### Community 94 - "refine.py"
Cohesion: 0.08
Nodes (23): _accumulator_need(), _finalize(), _first_pass(), _fit_active(), _fit_columns(), _grow_accumulator(), _Knobs, _midpoint_f1() (+15 more)

### Community 95 - "xraylabtool/utils.py"
Cohesion: 0.15
Nodes (16): Interfaces Module API Doc, CLI error handling messages, CLI modules (parser, formatting, utils), completion_v2 virtual-environment-centric shell completion, Supported shells (bash, zsh, fish, PowerShell), Validation Module API Doc, Validation best practices (validate first, catch specific exceptions), CalculationError (+8 more)

### Community 97 - "compound_analysis.py"
Cohesion: 0.13
Nodes (3): TestCalculationSpeedBenchmarks, calculation_worker(), TestPerformanceRegressionDetection

### Community 100 - "clear_scattering_factor_cache"
Cohesion: 0.38
Nodes (6): _outcome(), _result(), test_kk_result_arrays_are_read_only_copies(), test_kk_result_rejects_inconsistent_state(), test_refine_outcome_arrays_are_read_only_copies(), test_refine_outcome_rejects_inconsistent_state()

### Community 104 - "FormulaError"
Cohesion: 0.15
Nodes (3): decorator(), wrapper(), MemoryProfiler

### Community 106 - "ADR-002: PyQtGraph vs Matplotlib for GUI Plotting"
Cohesion: 0.17
Nodes (14): Shell Completion Guide, Completion caching system (~/.xraylabtool/cache), Completion for xtool alias, Completion features (command, option, path, value), CompletionInstaller, xraylabtool.interfaces.completion_v2, Conda/Mamba install and completion workflow, Environment detection (venv, conda, mamba, Poetry, Pipenv) (+6 more)

### Community 107 - "FastXRayCalculationEngine"
Cohesion: 0.25
Nodes (6): analyze_mypy_output(), check_core_modules(), check_type_definitions(), main(), run_mypy_check(), validate_cache_performance()

### Community 108 - "test_device_merge.py"
Cohesion: 0.15
Nodes (31): _assert_matches_reference(), _explicit_scale(), _poly(), _reference(), _run(), _status_of(), test_adjacent_float_knots_use_lower_boundary(), test_decimal_scale_of_the_near_edge_values_matches_scale_one() (+23 more)

### Community 109 - "_make_cubic"
Cohesion: 0.06
Nodes (6): TestCrystalStructure, _make_cubic(), _phase(), TestCrystalStructureAPI, TestStructureFactorPhysics, CrystalStructure

### Community 115 - "TestPerformanceBenchmarks"
Cohesion: 0.18
Nodes (14): Attenuation length vs energy plot for SiO2 (8-12 keV, 50 points, rising curve ~13 to ~44 cm x0.001), Compute primary action button, Energy to Wavelength converter widget (keV and Angstrom, E to lambda buttons), Footer buttons: Log Path and Light Mode (theme toggle), Inline validation feedback (Formula valid, Composition Si:1 O:2, Density valid, grid summary), Material input panel (formula, density, energy grid, log-spaced option, Compute button), Material summary table (formula, MW, density, electron density, total electrons), Plot toolbar (property selector, Log X/Log Y, Save Plot PNG, Export CSV) (+6 more)

### Community 116 - "test_device_refine.py"
Cohesion: 0.14
Nodes (28): _device_run(), _host_reference(), _logged(), _narrow_last_segment_table(), _refine(), _same_result(), _spacing_grid(), test_accumulator_growth_path_is_bit_identical() (+20 more)

### Community 119 - "TestPerformanceBenchmarks"
Cohesion: 0.14
Nodes (5): TestPerformanceBenchmarks, energy_sweep(), complex_formula(), multi_calculation(), single_calculation()

### Community 121 - "api.py"
Cohesion: 0.08
Nodes (19): _check_refine_arguments(), _checked_merge(), _compound_capacity(), _default_grid_size(), _explicit_energies(), _host_inputs(), _merge_points_ev(), _near_edge_call() (+11 more)

### Community 122 - "JIT compilation"
Cohesion: 0.23
Nodes (4): _json_numpy_default(), _save_dict_as_csv(), _save_results_as_csv(), _save_results_as_json()

### Community 124 - "xraylabtool/kk/__init__.py"
Cohesion: 0.10
Nodes (11): main(), truncation_deficit(), _load_golden(), _merged(), test_kernel_on_merged_spectrum_matches_golden_f1_points(), test_merged_spectrum_matches_upstream_golden(), test_nearedge_result_matches_upstream_golden(), _random_call() (+3 more)

### Community 125 - "json"
Cohesion: 0.24
Nodes (5): cmd_kk(), _fail(), _format_output(), _parse_energies(), _parse_merge()

### Community 126 - "test_cli_edge_cases.py"
Cohesion: 0.10
Nodes (3): TestBatchRowResilience, TestCLIErrorHandling, TestFieldNameValidation

### Community 127 - "cmd_convert"
Cohesion: 0.32
Nodes (8): _checkout(), _eval_points(), _finite(), _git(), main(), _poly(), _save(), _transform_case()

### Community 129 - "test_oracle.py"
Cohesion: 0.13
Nodes (12): f1_reference(), _gauss_legendre(), _h(), _pv_integral(), _quadrature(), _f2_mismatch(), oracle_energies(), _reference() (+4 more)

### Community 131 - "__getattr__"
Cohesion: 0.07
Nodes (6): TestAtomicScatteringFactor, TestLoadDataFile, AtomicScatteringFactor, _initialize_element_paths(), load_scattering_factor_data(), ScatteringData

### Community 132 - "core.py"
Cohesion: 0.07
Nodes (13): get_cached_elements(), is_element_cached(), _smart_cache_warming(), _prepare_element_data(), _process_formulas_parallel(), unique_result_key(), _validate_xray_inputs(), calculate_derived_quantities() (+5 more)

### Community 133 - "bucket"
Cohesion: 0.11
Nodes (22): _budget(), platform(), test_block_for_examples(), test_block_for_never_exceeds_the_request(), test_bucket_is_multiple_of_8_monotone_and_bounded(), test_bucket_ladder_head(), test_bucket_pow2(), test_bucket_rejects_bad_minimum() (+14 more)

### Community 136 - "Getting Started Guide"
Cohesion: 0.08
Nodes (28): Getting Started Guide, xraylabtool batch CSV processing, calculate_single_material_properties (usage), calculate_xray_properties (usage), gpu_cuda12 / gpu_cuda13 extras, PySide6 GUI (Single, Multiple, Kramers-Kronig tabs), Installation (pip, conda, uv, GPU extras), Performance tips (preloaded elements, batching, energy arrays) (+20 more)

### Community 137 - "Closed-form interval integral (corrected Eq. 10)"
Cohesion: 0.17
Nodes (9): Critical angle theta_c = sqrt(2 delta) with unit conversions, Closed-form interval integral (corrected Eq. 10), Kramers-Kronig dispersion relation f1 from f2 (Eq. 1), kk_piecewise_poly, Singularities at knots and end points (decision D5), Piecewise Laurent polynomial representation of f2 (Eq. 8), Critical angle for total external reflection, delta and beta from atomic scattering factors f1, f2 (+1 more)

### Community 140 - "XRayLabToolError (base exception)"
Cohesion: 0.22
Nodes (5): memory_test(), performance_test(), decorator(), requires_memory(), decorator()

### Community 141 - "CalculationError"
Cohesion: 0.12
Nodes (3): TestCalculationError, TestExceptionUsage, CalculationError

### Community 143 - "kernels.py"
Cohesion: 0.10
Nodes (3): AtomicDataProvider, CalculationEngine, InterpolatorProtocol

### Community 144 - "typing_extensions.py"
Cohesion: 0.22
Nodes (11): Available Scripts, Example Output:, Integration:, Maintenance Notes, Output:, Requirements:, `run_type_check.py` - Type Checking, Scripts Directory (+3 more)

### Community 145 - "TestKKCliEndToEnd"
Cohesion: 0.15
Nodes (7): _assert_matches(), _in_process(), _subprocess(), _table(), test_cli_writes_the_persistent_compile_cache(), test_version_unaffected(), TestKKCliEndToEnd

### Community 146 - "constants.py"
Cohesion: 0.11
Nodes (6): TestEnergyToWavelength, TestMathematicalConstants, energy_to_wavelength_angstrom(), _isclose(), validate_constants(), wavelength_angstrom_to_energy()

### Community 147 - "fixture"
Cohesion: 0.12
Nodes (8): batch_config_default(), batch_config_performance(), memory_monitor(), memory_tracker(), performance_thresholds(), sample_calculation_result(), test_energies_large(), test_materials()

### Community 149 - "validate_chemical_formula"
Cohesion: 0.10
Nodes (3): TestValidateChemicalFormula, _get_valid_element_symbols(), validate_chemical_formula()

### Community 150 - "get_gpu_info"
Cohesion: 0.20
Nodes (9): ADR-001: JAX vs NumPy Computation Backend, Appendix: Functions that MUST NOT be JIT-compiled, Appendix: Functions to JIT-Compile (Priority Order), Consequences, Context, Decision, Negative, Positive (+1 more)

### Community 151 - "test_type_safety_config.py"
Cohesion: 0.14
Nodes (16): backend sub-package (NumPy/JAX abstraction), JAX Architecture & Design, JAX array immutability and lazy evaluation, Automatic differentiation (grad), Debugging and testing JAX code (disable jit, jax.debug.print, JIT-vs-eager parity), Future JAX features for v0.5+ (autodiff, vmap, distributed), GPU acceleration and device placement, JAX library (+8 more)

### Community 155 - "test_warm_up.py"
Cohesion: 0.15
Nodes (11): cache_config(), test_cache_dir_is_os_appropriate(), test_cache_helper_enables_and_sets_threshold(), test_cache_helper_keeps_user_threshold(), test_cache_helper_respects_opt_out_and_existing_config(), test_cli_does_not_call_warm_up(), test_library_import_does_not_enable_cache(), test_warm_up_exported_lazily() (+3 more)

### Community 157 - "JAX Architecture & Design (v0.4.0)"
Cohesion: 0.22
Nodes (3): TestGetCompoundFamily, _classify_family_by_elements(), get_compound_family()

### Community 160 - "pytest"
Cohesion: 0.13
Nodes (17): ci_threshold_multiplier(), pytest_collection_modifyitems(), test_database_rejects_non_integer_atomic_number(), test_evaluate_f2_rejects_complex(), test_evaluate_f2_rejects_nan_segment_edge(), test_kernel_rejects_complex_instead_of_dropping_the_imaginary_part(), test_kk_result_stores_merge_points_as_an_immutable_tuple(), test_merge_spectra_rejects_complex_inputs() (+9 more)

### Community 161 - "kk_compiles.py"
Cohesion: 0.16
Nodes (7): _Handler, kk_programs(), numerical_programs(), record_compiles(), test_refined_sio2_compiles_few_top_level_programs(), test_energy_length_bucket_recompiles_only_the_evaluation(), test_second_call_no_recompile()

### Community 162 - "time"
Cohesion: 0.24
Nodes (10): _call_worker(), call(), _gate(), _kernel_worker(), main(), _recorder(), _run_worker(), _timed_ms() (+2 more)

### Community 163 - "compute_kk"
Cohesion: 0.24
Nodes (12): test_compute_kk_default_grid_cxro_matches_result_energies(), test_compute_kk_forwards_energies_kev(), test_compute_kk_forwards_refine_tolerance(), test_compute_kk_merge_points_follow_energy_unit(), test_compute_kk_rejects_unknown_energy_unit(), test_compute_kk_service_matches_api_with_nearedge(), test_compute_kk_service_matches_api_without_nearedge(), test_cxro_comparison_empty_outside_tabulated_range() (+4 more)

### Community 166 - "test_compound_analysis.py"
Cohesion: 0.19
Nodes (4): TestGetCompoundFrequencyScore, TestGetRecommendedElementsForWarming, get_compound_frequency_score(), get_recommended_elements_for_warming()

### Community 167 - "xraylabtool/__init__.py"
Cohesion: 0.27
Nodes (4): attenuation_length_cm(), critical_angle_degrees(), _configure_jax_float64(), __getattr__()

### Community 168 - "TestBatchProcessing"
Cohesion: 0.25
Nodes (7): Files, Maintenance, `performance_baselines.json`, `sample_materials.json`, Test Data Directory, `test_energies.json`, Usage in Tests

### Community 169 - "Testing Guide"
Cohesion: 0.29
Nodes (8): 30 keV tabulation cut (dashed vertical line), C (carbon) curve, Figure: f1 overestimated by truncating f2 integration at 30 keV, Fe (iron) curve, Kramers-Kronig f2-to-f1 integral truncated to tabulated range, Pb (lead) curve, f1 truncation error (f1_trunc - f1_full) vs photon energy, log-log, Result: truncation error grows with Z (C ~1e-4, Fe ~0.1, Pb ~2.5 e/atom), flat at low energy, rises sharply near 30 keV

### Community 171 - "test_midpoint_loop_never_evaluates_more_than_block_per_call"
Cohesion: 0.16
Nodes (8): dlog(), order(), pass_trace(), wrap(), spy(), si(), test_midpoint_loop_never_evaluates_more_than_block_per_call(), test_refine_buffer_growth_and_shrink_between_passes_is_bit_identical()

### Community 175 - "stray_conversions"
Cohesion: 0.24
Nodes (7): backend_restored(), _outside_device_helpers(), stray_conversions(), patch_method(), wrapper(), patch_numpy(), wrapper()

### Community 183 - "parametrize"
Cohesion: 0.17
Nodes (5): test_accelerator_block_rule_gives_the_same_result(), test_chunk_size_is_upper_bound(), test_kernel_evaluates_only_blocks_holding_real_energies(), test_no_host_conversion_outside_the_device_helpers(), test_nonfinite_nearedge_conversion_raises()

### Community 186 - "compare.py"
Cohesion: 0.24
Nodes (5): cmd_compare(), _parse_materials(), _result_to_dict(), _write_report(), _write_table()

### Community 188 - "gen_kk_prejax_baseline.py"
Cohesion: 0.23
Nodes (4): _finite(), main(), _rel_corr(), probe_energies()

### Community 190 - "argparse"
Cohesion: 0.08
Nodes (7): TestInstallCompletionMain, completion_main(), create_completion_parser(), handle_completion_command(), install_completion_main(), show_completion_info(), handle_new_completion_command()

### Community 191 - "kk/test_docs.py"
Cohesion: 0.23
Nodes (7): _public_objects(), test_docs_cite_the_paper_and_upstream_licence(), test_docs_pages_toctree_linked(), test_docs_state_the_eq10_erratum(), test_every_public_kk_name_is_documented(), test_public_docstrings_have_examples_and_references(), test_transform_docstrings_cite_paper_equations()

### Community 192 - "test_formula_parsing.py"
Cohesion: 0.17
Nodes (3): TestDelegatingWrappers, TestRegexCompatibility, _parse_formula()

### Community 193 - "normalize_intensity"
Cohesion: 0.12
Nodes (5): TestDataProcessing, background_subtraction(), normalize_intensity(), save_processed_data(), smooth_data()

### Community 195 - "transfer_counts"
Cohesion: 0.21
Nodes (7): _passes(), test_cold_table_upload_is_counted_once_and_separately(), test_jax_array_energies_are_fetched_through_the_counted_helper(), test_runtime_is_logged_once_per_process(), test_transfer_budget_warm_call(), reset_transfer_counts(), transfer_counts()

### Community 199 - "refine_spectrum"
Cohesion: 0.24
Nodes (6): test_refine_rejects_bad_arguments(), test_type_hints_resolve_at_runtime(), _checked_inputs(), _grid_array(), refine_spectrum(), RefineOutcome

### Community 200 - "evaluate_f2"
Cohesion: 0.20
Nodes (6): main(), _run(), test_refine_matches_golden_on_device(), test_heavy_element_segment_grid_against_upstream_emulation(), _upstream_like_added(), evaluate_f2()

### Community 201 - "test_kk_docs_build.py"
Cohesion: 0.33
Nodes (4): _sphinx(), test_kk_docs_build_adds_no_warnings(), test_kk_docs_doctests_pass(), slow()

### Community 207 - "TestGuiErrorHandling"
Cohesion: 0.22
Nodes (3): numpy_backend_after(), TestGuiErrorHandling, task()

### Community 208 - "ADR-005: The Kramers-Kronig Package Is JAX-Only"
Cohesion: 0.22
Nodes (8): ADR-005: The Kramers-Kronig Package Is JAX-Only, Alternatives considered, Consequences, Context, Evidence, Known deviations from the acceptance gates, References, x64 requirement

### Community 211 - "re"
Cohesion: 0.33
Nodes (4): collect(), main(), _search(), test_pyproject_package_data_lists_kk_files()

### Community 212 - "gc"
Cohesion: 0.29
Nodes (3): clean_cache(), enforce_test_isolation(), setup_test_environment()

### Community 216 - "Data Handling Module API Doc"
Cohesion: 0.32
Nodes (7): Data Handling Module API Doc, atomic_cache (92-element preload, warm_up_cache, get_cache_stats), batch_processing (BatchConfig, calculate_batch_properties, CSV I/O), compound_analysis (compound families, recommended warming elements), CXRO / NIST / Henke .nff scattering factor data (0.03-30 keV, PCHIP), CLI commands (calc, batch, compare, convert, formula, atomic, bragg, list, completion, install-completion, uninstall-completion), CLI output formats (table, csv, json)

### Community 219 - "test_gui_kk_worker_end_to_end.py"
Cohesion: 0.23
Nodes (11): _check_table(), _setup(), test_close_during_warm_up_does_not_hang(), test_error_path_shows_dialog_and_stays_responsive(), test_launcher_schedules_the_warm_up(), test_nearedge_run_matches_api(), test_plain_run_matches_api(), test_refined_run_matches_api() (+3 more)

### Community 237 - "transform.py"
Cohesion: 0.14
Nodes (12): test_kk_f1_grad(), loss(), _crop(), _f64(), kk_f1(), _kk_f1_blocks(), one(), kk_piecewise_poly() (+4 more)

### Community 252 - "test_threaded_calls_agree"
Cohesion: 0.33
Nodes (4): test_threaded_calls_agree(), _threaded(), work(), fn()

## Knowledge Gaps
- **123 isolated node(s):** `Henke, Gullikson, Davis 1993 X-ray interactions tables`, `Completion caching system (~/.xraylabtool/cache)`, `Completion for xtool alias`, `Completion features (command, option, path, value)`, `Conda/Mamba install and completion workflow` (+118 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 2235 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **128 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `v0.4.7 (2026-10-23)` connect `pytest` to `create_scattering_factor_interpolators`, `re`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Are the 46 inferred relationships involving `kk_calculate_real()` (e.g. with `Decision` and `Evidence`) actually correct?**
  _`kk_calculate_real()` has 46 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Henke, Gullikson, Davis 1993 X-ray interactions tables`, `Completion caching system (~/.xraylabtool/cache)`, `Completion for xtool alias` to the rest of the system?**
  _123 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `MainWindowProtocol` be split into smaller, more focused modules?**
  _Cohesion score 0.05257312106627175 - nodes in this community are weakly interconnected._
- **Why does `Changelog` connect `pytest` to `fixture`, `.calculate_derived_quantities`, `TestXRayResultTyping`, `test_type_safety_config.py`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `MainWindowProtocol` (e.g. with `KKTabMixin` and `MultiTabMixin`) actually correct?**
  _`MainWindowProtocol` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Should `xraylabtool/utils.py` be split into smaller, more focused modules?**
  _Cohesion score 0.04941724941724942 - nodes in this community are weakly interconnected._