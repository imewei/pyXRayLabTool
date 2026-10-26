# Graph Report - pyXRayLabTool  (2026-10-26)

## Corpus Check
- 207 files · ~195,065 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 112 file(s) not represented in the graph (top: .nff 93, .npz 12, .ipynb 2)

## Summary
- 5405 nodes · 11799 edges · 242 communities (132 shown, 110 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 859 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c22f1767`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- MainWindowProtocol
- parametrize
- test_jax_backend_parity.py
- numpy
- EnergyError
- EnvironmentDetector
- ValidationError
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
- MemoryMonitor
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
- evaluate_f2
- TestEnergySweepSi
- create_scattering_factor_interpolators
- test_completion_environment.py
- typing
- installer.py
- atomic_cache.py
- CalculationWorker
- installer.py
- MaterialComparator
- test_utils.py
- Spectrum
- argparse
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
- cmd_bragg
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
- nearedge.py
- TestXRayResultTyping
- transform.py
- pytest
- run_type_check.py
- cache.py
- EnergyError
- tests/conftest.py
- MemoryProfiler
- PerformanceBenchmark
- test_compound_analysis.py
- kk_gpu_check.py
- TestEnvironmentProperties
- Version 0.4.7 (2026-10-23)
- test_pipeline_device.py
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
- FastAtomicDataProvider
- TestGoldenAu
- TestGoldenFe2O3
- TestGoldenSiO2
- BashCompletionGenerator
- TestPerformanceBenchmarks
- test_device_refine.py
- numpy
- XRayLabToolError
- calculate_single_material_properties
- test_numerical_stability.py
- kk_calculate_real
- JIT compilation
- ADR-003: float64 Preservation Policy
- xraylabtool/kk/__init__.py
- json
- test_cli_edge_cases.py
- cmd_convert
- test_cli.py
- test_oracle.py
- atomic_cache.py
- scattering_data.py
- cmd_calc
- bucket
- test_integration.py
- bragg_angle
- Getting Started Guide
- test_transform.py
- generate_test_energies
- calculators.core (calculate_single_material_properties, calculate_xray_properties)
- XRayLabToolError (base exception)
- xraylabtool/__init__.py
- get_atomic_weight
- Getting Started
- typing_extensions.py
- TestDtypePreservation
- constants.py
- TestInstallCompletionCommand
- kk_calculate_real
- validate_chemical_formula
- get_gpu_info
- test_type_safety_config.py
- _parse_formula
- ADR-001: JAX vs NumPy Computation Backend
- pathlib
- test_warm_up.py
- Decision: replace matplotlib with PyQtGraph for interactive plots
- JAX Architecture & Design (v0.4.0)
- test_kk_plot.py
- BasePerformanceTest
- pytest
- kk_compiles.py
- os
- compute_kk
- v0.4.0 release (2026-04-07)
- TestUnitConversionConstants
- test_compound_analysis.py
- FormulaError
- TestBatchProcessing
- Testing Guide
- api.py
- calculate_properties_from_factors
- test_validators.py
- atomic_cache.py
- BaseXRayLabToolTest
- test_kernel_evaluates_only_blocks_holding_real_energies
- get_elements_for_compound
- TestDtypePreservation
- TestGoldenAu
- .sum
- TestBatchConfig
- Toast
- TestCLICompatibility
- BashCompletionGenerator
- TestEnvironmentProperties
- test_heavy_element_segment_grid_against_upstream_emulation
- calculators/__init__.py
- test_golden_constants.py
- ResultValidator
- TestBatchProcessingMemoryManagement
- kk/test_docs.py
- test_formula_parsing.py
- TestMemoryLeakPrevention
- calculate_multiple_xray_properties
- test_midpoint_loop_never_evaluates_more_than_block_per_call
- TestEnvironmentProperties
- TestRegexCompatibility
- Decision: enforce float64 globally; float32 is a correctness bug
- refine_spectrum
- FormulaError
- process_single_calculation
- TestEnvironmentIntegration
- typing_extensions.py
- test_core.py
- _outcome
- ADR-005: The Kramers-Kronig Package Is JAX-Only
- validators.py
- performance/conftest.py
- TestParseFormulaDecimal
- TestParseFormulaErrors
- TestScalarEnergyFallback
- TestBackwardCompatibility
- TestParseFormulaParentheses
- conf.py
- TestFormulaParsing
- test_gui_kk_worker_end_to_end.py
- TestMathematicalConstants
- integration/__init__.py
- performance/__init__.py
- unit/__init__.py
- BaseIntegrationTest
- BasePerformanceTest
- BaseUnitTest
- _resize
- body
- _clear_cache
- assert_calculation_result_valid

## God Nodes (most connected - your core abstractions)
1. `kk_calculate_real()` - 106 edges
2. `MainWindowProtocol` - 101 edges
3. `calculate_single_material_properties()` - 95 edges
4. `parse_formula()` - 90 edges
5. `CompletionInstaller` - 69 edges
6. `kk_real_spectrum()` - 65 edges
7. `ValidationError` - 61 edges
8. `compound_spectrum()` - 51 edges
9. `Spectrum` - 50 edges
10. `EnergyError` - 48 edges

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

## Communities (242 total, 110 thin omitted)

### Community 0 - "MainWindowProtocol"
Cohesion: 0.04
Nodes (6): _kk_energy_grid(), KKTabMixin, MultiTabMixin, MainWindowProtocol, SingleTabMixin, MaterialTable

### Community 1 - "parametrize"
Cohesion: 0.04
Nodes (25): TestCrystallographicCalculations, TestDataProcessing, TestUnitConversions, __getattr__(), angle_from_q(), background_subtraction(), bragg_angle(), _check_lattice() (+17 more)

### Community 2 - "test_jax_backend_parity.py"
Cohesion: 0.06
Nodes (11): TestBackendSwitching, TestInterpolationFactory, TestOpsProxy, restore_numpy_backend(), TestJaxBackendParity, _auto_select_backend(), get_backend(), _has_nvidia_gpu() (+3 more)

### Community 3 - "numpy"
Cohesion: 0.07
Nodes (31): _cxro_deviation(), test_30kev_truncation_matches_paper_trend(), test_default_grid_matches_upstream_golden_bit_for_bit_inputs(), test_end_to_end_matches_upstream_golden(), test_tabulated_kk_f1_close_to_cxro_f1(), _load_golden(), _merged(), test_kernel_on_merged_spectrum_matches_golden_f1_points() (+23 more)

### Community 4 - "EnergyError"
Cohesion: 0.05
Nodes (12): TestAtomicDataError, TestCalculationError, TestDataFileError, TestExceptionHierarchy, TestExceptionUsage, TestUnknownElementError, AtomicDataError, CalculationError (+4 more)

### Community 5 - "EnvironmentDetector"
Cohesion: 0.05
Nodes (5): test_cli_aliases_registered(), CompletionGenerator, FishCompletionGenerator, PowerShellCompletionGenerator, ZshCompletionGenerator

### Community 6 - "ValidationError"
Cohesion: 0.11
Nodes (3): TestValidationError, ValidationError, _readonly_1d()

### Community 7 - "_make_cubic"
Cohesion: 0.06
Nodes (6): TestEdgeCasesAndErrorHandling, TestH2OProperties, TestInputValidationAndErrorHandling, TestSiO2Properties, calculate_xray_properties(), _validate_xray_inputs()

### Community 8 - "constants.py"
Cohesion: 0.09
Nodes (9): apply_palette_to_item(), apply_palette_to_widget(), current_palette(), legend_kwargs(), _recolor_axis_label(), _restyle_legend(), PlotCanvas, F1F2Plot (+1 more)

### Community 9 - "_calculate_single_material_xray_properties"
Cohesion: 0.05
Nodes (24): _calculate_molecular_properties(), calculate_multiple_xray_properties(), _calculate_single_material_xray_properties(), _coerce_energy_array(), _convert_energy_input(), _create_process_formula_function(), process_formula(), _MaterialPropertiesDict (+16 more)

### Community 10 - "test_concurrent_cache.py"
Cohesion: 0.06
Nodes (16): TestAtomicDataLookup, clear_lru_caches(), _mock_get_element(), TestCacheResultConsistency, worker(), TestConcurrentGetAtomicNumber, worker(), cache_clearer() (+8 more)

### Community 11 - "XRayResult"
Cohesion: 0.12
Nodes (37): _edge_spectrum(), test_asf_input_is_unchanged_and_unknown_type_raises(), test_beta_round_trip_and_density_cancels_at_merge(), test_default_merge_points_are_the_overlap_of_both_ranges(), test_directly_constructed_spectrum_is_validated(), test_docstring_examples_run(), test_file_in_kev_declared_ev_raises_no_overlap(), test_flat_endpoints_raise_not_inf() (+29 more)

### Community 13 - "clear_scattering_factor_cache"
Cohesion: 0.10
Nodes (4): _clear_cache(), cold_cache(), cold_cache(), clear_scattering_factor_cache()

### Community 14 - "parse_formula"
Cohesion: 0.14
Nodes (4): TestParseFormulaBasic, _aggregate_element_counts(), parse_formula(), _validate_balanced_parentheses()

### Community 15 - "test_deep_debug_regressions.py"
Cohesion: 0.05
Nodes (12): numpy_backend_after(), test_whitespace_delimited_loader_works_with_current_pandas(), TestAtomicDataCacheIsImmutable, TestBackendState, TestGuiErrorHandling, task(), TestLazySubmoduleAccess, TestRepeatedMaterials (+4 more)

### Community 16 - "test_api.py"
Cohesion: 0.18
Nodes (16): test_default_grid_clipped_to_0p03_30_kev_and_logged(), test_docstring_examples_run(), test_empty_default_grid_after_merge_raises(), test_energies_are_copied_not_aliased(), test_energy_one_ulp_outside_the_data_range_raises(), test_energy_out_of_data_range_raises_energy_error(), test_explicit_energy_between_10ev_and_500kev_ok(), test_import_xraylabtool_does_not_import_kk() (+8 more)

### Community 18 - "Toast"
Cohesion: 0.07
Nodes (4): main(), StyleGuideValidator, StyleViolation, ValidationReport

### Community 19 - "MemoryMonitor"
Cohesion: 0.06
Nodes (5): TestBenchmarkComparison, TestPerformanceBenchmarks, TestPerformanceRegression, TestScalabilityBenchmarks, timer()

### Community 20 - "load_asf"
Cohesion: 0.12
Nodes (29): _args(), _eq10_corrected(), _f2_poly(), _golden(), _linear_segments(), _quadrature_f1(), test_chunk_invariance(), test_constant_band_closed_form() (+21 more)

### Community 21 - "typing"
Cohesion: 0.14
Nodes (25): _max_midpoint_error(), _refine(), test_debug_log_per_pass_and_info_total(), test_docstring_examples_run(), test_exhaustion_flagged_and_logged(), test_loose_tolerance_adds_nothing(), test_lower_range_end_through_kk_calculate_real(), test_midpoint_stagnation_terminates() (+17 more)

### Community 22 - "validate_chemical_formula"
Cohesion: 0.10
Nodes (12): chunk_iterator(), _filter_dataframe_fields(), _get_energy_point_data(), _initialize_progress_bar(), _prepare_energies_array(), _prepare_result_data(), process_batch_chunk(), _process_chunks() (+4 more)

### Community 23 - "TestJaxBackendParity"
Cohesion: 0.07
Nodes (15): TestResultFormatting, _format_array_field_single(), _format_as_csv(), _format_as_json(), _format_filtered_table(), _format_material_properties(), _format_multiple_energies(), _format_multiple_energy_section() (+7 more)

### Community 24 - "test_utilities.py"
Cohesion: 0.10
Nodes (12): kk_log(), test_configure_logging_writes_file(), test_get_logger_child_inherits_config(), build_parser(), main(), _bool_env(), _coerce_int_env(), configure_logging() (+4 more)

### Community 26 - "TestBasicExportFunctionality"
Cohesion: 0.05
Nodes (36): ADR-002: PyQtGraph vs Matplotlib for GUI Plotting, Appendix: Feature Parity Checklist, Consequences, Context, Decision, Migration Details, Negative, Positive (+28 more)

### Community 27 - "data_handling/__init__.py"
Cohesion: 0.13
Nodes (9): _warm_priority_cache(), _background_cache_warming(), get_atomic_data_fast(), get_bulk_atomic_data_fast(), get_cache_stats(), is_element_preloaded(), warm_up_cache(), load_batch_input() (+1 more)

### Community 28 - "test_concurrent_cache.py"
Cohesion: 0.08
Nodes (9): main(), TestCheckGpuAvailability, TestGetDeviceInfo, TestGetSystemCudaVersion, check_gpu_availability(), check_plugin_conflicts(), get_device_info(), get_recommended_package() (+1 more)

### Community 29 - "TestBasicExportFunctionality"
Cohesion: 0.07
Nodes (5): TestBasicExportIntegration, TestBasicExportFunctionality, export_to_csv(), export_to_json(), __getattr__()

### Community 31 - "BatchConfig"
Cohesion: 0.08
Nodes (7): TestBatchProcessorEquivalence, TestBatchConfig, no_progress_config(), TestBatchInputValidation, TestBatchPartialFailure, BatchConfig, calculate_batch_properties()

### Community 32 - "CompletionInstaller"
Cohesion: 0.08
Nodes (3): TestSentinelMarkers, CompletionInstaller, test_completion_installation()

### Community 34 - "test_cli.py"
Cohesion: 0.07
Nodes (9): form(), _get_app(), qt_app(), TestDensityValidation, TestEnergyGridValidation, TestFormulaValidation, TestFormValues, TestPlotAxisLabelTheming (+1 more)

### Community 35 - "test_deep_debug_regressions.py"
Cohesion: 0.13
Nodes (11): _computation(), _pen_colors(), plot(), qt_app(), test_plot_draws_curves_on_unequal_grids(), test_plot_follows_theme_toggle(), test_plot_hides_missing_curves(), test_plot_log_x_rule() (+3 more)

### Community 36 - "TestCICDIntegration"
Cohesion: 0.11
Nodes (12): _run_smoke(), test_compute_multiple_reports_progress_to_100(), _ensure_app(), test_export_csv_handles_io_error(), test_gui_smoke_offscreen(), test_gui_smoke_threaded_offscreen(), test_duplicate_formula_different_density_not_dropped(), suppress_qt_noise() (+4 more)

### Community 37 - "main"
Cohesion: 0.05
Nodes (8): TestAtomicCommand, TestBatchCommand, TestCLIIntegration, TestListCommand, TestMainFunction, cmd_atomic(), cmd_list(), main()

### Community 38 - "TestKKCli"
Cohesion: 0.06
Nodes (18): _parse_csv(), _run(), test_empty_default_grid_after_merge_exits_1(), TestKKCli, _write_edge(), add_atomic_command(), add_batch_command(), add_bragg_command() (+10 more)

### Community 40 - "TestEnergySweepSi"
Cohesion: 0.10
Nodes (3): MockXRayResult, TestFormatXRayResult, format_xray_result()

### Community 41 - "create_scattering_factor_interpolators"
Cohesion: 0.06
Nodes (40): Calculators Module (API docs), xraylabtool.calculators.cache, calculate_xray_properties, xraylabtool.calculators.core, Critical angle theta_c = sqrt(2 delta), xraylabtool.calculators.derived_quantities, xraylabtool.calculators.kernels, Legacy CamelCase accessors (DeprecationWarning) (+32 more)

### Community 43 - "typing"
Cohesion: 0.08
Nodes (11): cmd_batch(), _convert_result_to_dict(), _parse_batch_data(), _process_batch_materials(), _save_batch_results(), _validate_batch_input(), cmd_completion(), cmd_install_completion() (+3 more)

### Community 44 - "installer.py"
Cohesion: 0.08
Nodes (5): TestCompletionInstallToTempDir, TestCompletionScriptGeneration, CompletionManager, get_global_options(), get_xraylabtool_commands()

### Community 45 - "atomic_cache.py"
Cohesion: 0.09
Nodes (4): TestBatchProcessingMemoryManagement, TestCacheManagement, TestMemoryMonitor, MemoryMonitor

### Community 46 - "CalculationWorker"
Cohesion: 0.09
Nodes (10): arrays_close(), assert_arrays_close(), assert_values_close(), expect_no_warnings(), expect_warnings(), skip_if_insufficient_cpu(), skip_if_insufficient_memory(), stability_test() (+2 more)

### Community 47 - "installer.py"
Cohesion: 0.10
Nodes (6): apply_pyqtgraph_theme(), apply_styles(), apply_theme(), ColorPalette, get_qss(), ThemeManager

### Community 48 - "MaterialComparator"
Cohesion: 0.08
Nodes (6): _check_element(), test_si_o_au_interpolators_are_independent(), TestPchipAu, TestPchipO, TestPchipSi, create_scattering_factor_interpolators()

### Community 49 - "test_utils.py"
Cohesion: 0.07
Nodes (9): BasePerformanceTest, TestArrayOptimization, TestBulkAtomicDataOptimization, TestDeprecationWarningOptimization, TestElementPathOptimization, TestInterpolatorCaching, TestNumericalStabilityOptimizations, TestPerformanceRegression (+1 more)

### Community 50 - "Spectrum"
Cohesion: 0.07
Nodes (30): Decision, _evaluate_f2_device(), test_device_loop_runs_under_transfer_guard(), _poly(), _reference_f2(), test_evaluate_f2_device_edges(), test_evaluate_f2_device_is_jittable_and_bucket_independent(), test_evaluate_f2_device_rounds_like_upstream_at_cancellation() (+22 more)

### Community 51 - "argparse"
Cohesion: 0.09
Nodes (10): collect(), main(), _search(), TestInstallCompletionMain, test_pyproject_package_data_lists_kk_files(), completion_main(), create_completion_parser(), handle_completion_command() (+2 more)

### Community 52 - "fixture"
Cohesion: 0.07
Nodes (38): Platform and Python version compatibility matrix, Contributing Guide, Contributing atomic data (citation, range, precision, format), Code standards (PEP 8, 88 cols, NumPy docstrings, type hints), Performance requirements (<0.1 ms single, >100k calc/s batch, >90% cache hit), Review process (automated checks, reviewer, maintainer, squash merge), Development workflow (branching, ruff, mypy, pytest, make verify), Performance and Optimization guide (+30 more)

### Community 53 - ".calculate_derived_quantities"
Cohesion: 0.07
Nodes (39): widgets.kk_plot (Kramers-Kronig Plot), main_window.kk_tab, kk sub-package (Kramers-Kronig, kkcalc port), Kramers-Kronig (kk) API docs, xraylabtool.kk.api (high-level API), AsfDatabase, compound_spectrum, xraylabtool.kk.database (+31 more)

### Community 54 - "TestGoldenSi"
Cohesion: 0.07
Nodes (3): TestGoldenDerivedQuantities, TestGoldenScatteringFactors, TestGoldenSi

### Community 55 - "Utilities Module"
Cohesion: 0.08
Nodes (3): TestEdgeCases, TestNumpyBackend, NumpyBackend

### Community 57 - "device.py"
Cohesion: 0.19
Nodes (20): app(), _small_run(), test_energies_outside_cxro_range_reach_the_service(), test_energy_range_controls_reach_the_service(), spy(), test_export_csv_columns(), test_export_csv_io_error_is_reported(), test_export_file_name_uses_computed_formula() (+12 more)

### Community 58 - "kk/test_docs.py"
Cohesion: 0.25
Nodes (7): test_helper_delta_beta_match_closed_form(), test_helper_rejects_complex_input(), test_helper_rejects_mismatched_nonfinite_or_bad_density(), test_helper_reproduces_lookup_path(), test_negative_f1_preserved_critical_angle_zero(), test_negative_f2_kept_and_attenuation_clamped(), calculate_properties_from_factors()

### Community 62 - "unit"
Cohesion: 0.12
Nodes (5): _calc(), TestComplexFormula, TestElectronCountGolden, TestExactCapturedValues, TestMolecularWeightGolden

### Community 64 - "Any"
Cohesion: 0.09
Nodes (3): ArrayBackend, JaxBackend, _configure_jax_float64()

### Community 65 - "ndarray"
Cohesion: 0.16
Nodes (12): check_accessibility(), check_documentation_coverage(), Colors, main(), print_section(), print_status(), run_style_checks(), test_doctests() (+4 more)

### Community 66 - "load_asf"
Cohesion: 0.07
Nodes (28): _kernel_worker(), _asf_f2(), test_asf_grid_spans_10ev_to_500kev(), test_asf_has_92_elements(), test_cached_tables_are_read_only(), test_im_coeffs_shape(), test_load_asf_is_cached(), test_nff_divergent_elements() (+20 more)

### Community 67 - "tests/conftest.py"
Cohesion: 0.12
Nodes (12): work(), qt_app(), _run_worker(), test_worker_does_not_emit_finished_on_exception(), test_worker_emits_error_on_exception(), bad_fn(), test_worker_emits_result_on_success(), test_worker_injects_progress_cb_when_accepted() (+4 more)

### Community 69 - "MaterialComparator"
Cohesion: 0.05
Nodes (5): TestBasicAnalysis, ComparisonResult, MaterialComparator, compare_materials(), find_absorption_edges()

### Community 71 - "get_bulk_atomic_data"
Cohesion: 0.09
Nodes (21): ADR-003: float64 Preservation Policy, Appendix: Numerical Precision Requirements by Function, Consequences, Context, Current Codebase float64 Usage, Decision, Enforcement Strategy, JAX float32 Default (+13 more)

### Community 72 - "_OpsProxy"
Cohesion: 0.13
Nodes (21): calculate_single_material_properties, energy_to_wavelength, parse_formula, utils sub-package, wavelength_to_energy, CLI Reference, atomic command, bragg command (+13 more)

### Community 73 - "scattering_data.py"
Cohesion: 0.10
Nodes (12): angle_from_q(), bragg_angle(), d_spacing_cubic(), xraylabtool.device, energy_to_wavelength(), get_atomic_data(), load_atomic_data(), xraylabtool.logging_utils (+4 more)

### Community 75 - "set_backend"
Cohesion: 0.13
Nodes (6): TestCalcCommand, cmd_calc(), _determine_output_format(), _print_calc_verbose_info(), _save_or_print_output(), _validate_calc_inputs()

### Community 77 - "nearedge.py"
Cohesion: 0.06
Nodes (21): _check_positive(), _fmt_range(), _interp(), _merge(), merge_device(), _Merged, MergeResult, MergeStatus (+13 more)

### Community 78 - "TestXRayResultTyping"
Cohesion: 0.10
Nodes (22): GUI (API docs), xraylabtool.gui.logging_filters, xraylabtool.gui.main_window, widgets.material_form, widgets.material_table, main_window.multi_tab, widgets.plot_canvas, widgets.scrollbar_helper (+14 more)

### Community 79 - "transform.py"
Cohesion: 0.08
Nodes (28): main(), _run(), _case(), _eval(), test_kk_f1_grad(), loss(), test_kk_f1_jit_vmap(), test_kk_f1_keeps_input_shape_and_rejects_bad_block() (+20 more)

### Community 80 - "pytest"
Cohesion: 0.07
Nodes (30): CalculationError, EnergyError, FormulaError, ValidationError, XRayLabToolError, Changelog, Batch API fixes (ordering, timeouts, process-pool fallback, formula@density keys), CLI aliases xtool / xlt and GUI aliases xtool-gui / xlt-gui (+22 more)

### Community 81 - "run_type_check.py"
Cohesion: 0.11
Nodes (20): Analytic per-interval integration (Eqs. 3-4), Biggs and Lighthill high-energy cross-sections (to 500 keV), Cauchy principal value integral, Conventional KK numerical methods (FFT, Euler-Maclaurin, Gaussian, TANH), Zero-to-infinity convergence conditions on polynomial coefficients, Cross et al. difference-spectrum KK method, Double-interval integration across x=E singularity (Eq. 5), Closed-form f1(E) for piecewise Laurent f2 (Eq. 7, Eq. 10) (+12 more)

### Community 82 - "cache.py"
Cohesion: 0.22
Nodes (3): get_cached_elements(), is_element_cached(), _smart_cache_warming()

### Community 83 - "EnergyError"
Cohesion: 0.11
Nodes (9): _manual(), test_negative_f1_near_edge_reaches_xray_result(), test_result_module_doctests(), test_to_xray_result_accepts_range_ends(), test_to_xray_result_rejects_out_of_range_energy(), test_to_xray_result_uses_kk_f1(), TestEnergyError, EnergyError (+1 more)

### Community 84 - "tests/conftest.py"
Cohesion: 0.04
Nodes (30): assert_memory_usage_reasonable(), assert_performance_within_threshold(), batch_config_default(), batch_config_performance(), batch_config_small(), benchmark_baseline(), clean_cache(), cpu_count() (+22 more)

### Community 86 - "PerformanceBenchmark"
Cohesion: 0.14
Nodes (6): TestFormulaCommand, cmd_formula(), _get_atomic_data(), _output_formula_results(), _print_formula_results(), _process_formula()

### Community 87 - "test_compound_analysis.py"
Cohesion: 0.11
Nodes (19): TestExports, _two_results(), test_load_absolute_path(), test_load_csv_file(), test_load_csv_space_separated_fallback(), test_load_empty_nff_raises_data_file_error(), test_load_malformed_nff_raises_data_file_error(), test_load_nff_energy_column_monotone() (+11 more)

### Community 88 - "kk_gpu_check.py"
Cohesion: 0.11
Nodes (18): check_additivity(), check_bench(), check_f2_si_edge(), check_oracle(), check_padding(), check_transfers(), _f2_numpy(), guarded() (+10 more)

### Community 89 - "TestEnvironmentProperties"
Cohesion: 0.10
Nodes (3): BaseUnitTest, TestDocumentationPatterns, TestStyleCompliance

### Community 90 - "Version 0.4.7 (2026-10-23)"
Cohesion: 0.08
Nodes (3): TestNumericalStabilityChecks, FastXRayCalculationEngine, get_calculation_engine()

### Community 91 - "test_pipeline_device.py"
Cohesion: 0.08
Nodes (28): _colliding_pair(), _passes(), _run(), _subprocess(), test_backend_field_is_jax(), test_chunk_size_is_upper_bound(), test_cold_concurrent_first_calls_upload_the_tables_once(), test_cold_table_upload_is_counted_once_and_separately() (+20 more)

### Community 92 - "XRayLabTool Documentation (index)"
Cohesion: 0.08
Nodes (29): Runtime NumPy/JAX backend switching (set_backend), Modular package architecture (backend, calculators, data_handling, interfaces, io, validation, gui), Kramers-Kronig Guide, Near-edge input types (nexafs, beta, asf), kkcalc attribution (zlib/libpng, Watts and Lauk), kk module limits (no tail, conditioning, pole skipping, wrong energy-unit hazard), Near-edge file format, Watts 2014, Opt. Express 22(19), 23628 (piecewise Laurent polynomial KK) (+21 more)

### Community 93 - "typing_extensions.py"
Cohesion: 0.20
Nodes (5): is_complex_array(), is_energy_array(), is_real_array(), validate_complex_array(), validate_energy_array()

### Community 94 - "refine.py"
Cohesion: 0.10
Nodes (18): _accumulator_need(), _finalize(), _first_pass(), _fit_active(), _fit_columns(), _grow_accumulator(), _Knobs, _midpoint_f1() (+10 more)

### Community 95 - "xraylabtool/utils.py"
Cohesion: 0.10
Nodes (23): Data Handling Module API Doc, atomic_cache (92-element preload, warm_up_cache, get_cache_stats), batch_processing (BatchConfig, calculate_batch_properties, CSV I/O), compound_analysis (compound families, recommended warming elements), CXRO / NIST / Henke .nff scattering factor data (0.03-30 keV, PCHIP), Interfaces Module API Doc, CLI commands (calc, batch, compare, convert, formula, atomic, bragg, list, completion, install-completion, uninstall-completion), CLI error handling messages (+15 more)

### Community 100 - "clear_scattering_factor_cache"
Cohesion: 0.38
Nodes (6): _outcome(), _result(), test_kk_result_arrays_are_read_only_copies(), test_kk_result_rejects_inconsistent_state(), test_refine_outcome_arrays_are_read_only_copies(), test_refine_outcome_rejects_inconsistent_state()

### Community 104 - "FormulaError"
Cohesion: 0.15
Nodes (3): decorator(), wrapper(), MemoryProfiler

### Community 105 - "TestCIWorkflowSimulation"
Cohesion: 0.10
Nodes (4): clean_cache_around(), wrapper(), wrapper(), PerformanceBenchmark

### Community 106 - "ADR-002: PyQtGraph vs Matplotlib for GUI Plotting"
Cohesion: 0.17
Nodes (14): Shell Completion Guide, Completion caching system (~/.xraylabtool/cache), Completion for xtool alias, Completion features (command, option, path, value), CompletionInstaller, xraylabtool.interfaces.completion_v2, Conda/Mamba install and completion workflow, Environment detection (venv, conda, mamba, Poetry, Pipenv) (+6 more)

### Community 107 - "FastXRayCalculationEngine"
Cohesion: 0.25
Nodes (6): analyze_mypy_output(), check_core_modules(), check_type_definitions(), main(), run_mypy_check(), validate_cache_performance()

### Community 108 - "test_device_merge.py"
Cohesion: 0.11
Nodes (31): _assert_matches_reference(), _explicit_scale(), _poly(), _reference(), _run(), _status_of(), test_adjacent_float_knots_use_lower_boundary(), test_interp_exact_knot_with_overflowing_slope() (+23 more)

### Community 111 - "TestGoldenAu"
Cohesion: 0.16
Nodes (3): TestGoldenAttenuationLength, calculate_attenuation_length(), calculate_scattering_length_density()

### Community 113 - "TestGoldenSiO2"
Cohesion: 0.10
Nodes (7): TestEnergyParsing, cmd_compare(), _parse_materials(), _result_to_dict(), _write_report(), _write_table(), parse_energy_string()

### Community 115 - "TestPerformanceBenchmarks"
Cohesion: 0.18
Nodes (14): Attenuation length vs energy plot for SiO2 (8-12 keV, 50 points, rising curve ~13 to ~44 cm x0.001), Compute primary action button, Energy to Wavelength converter widget (keV and Angstrom, E to lambda buttons), Footer buttons: Log Path and Light Mode (theme toggle), Inline validation feedback (Formula valid, Composition Si:1 O:2, Density valid, grid summary), Material input panel (formula, density, energy grid, log-spaced option, Compute button), Material summary table (formula, MW, density, electron density, total electrons), Plot toolbar (property selector, Log X/Log Y, Save Plot PNG, Export CSV) (+6 more)

### Community 116 - "test_device_refine.py"
Cohesion: 0.13
Nodes (29): _device_run(), _host_reference(), _logged(), _narrow_last_segment_table(), _refine(), _same_result(), _spacing_grid(), test_accumulator_growth_path_is_bit_identical() (+21 more)

### Community 119 - "calculate_single_material_properties"
Cohesion: 0.04
Nodes (11): TestCLICompatibility, TestCLIOutputFormatting, TestEdgeCaseInputValues, TestPerformanceBenchmarks, energy_sweep(), complex_formula(), multi_calculation(), single_calculation() (+3 more)

### Community 120 - "test_numerical_stability.py"
Cohesion: 0.05
Nodes (3): TestBoundaryConditions, TestExtremeConditions, TestPhysicalRealism

### Community 121 - "kk_calculate_real"
Cohesion: 0.08
Nodes (19): _check_refine_arguments(), _checked_merge(), _compound_capacity(), _compound_segments(), _default_grid_size(), _explicit_energies(), _host_inputs(), kk_calculate_real() (+11 more)

### Community 122 - "JIT compilation"
Cohesion: 0.24
Nodes (4): _json_numpy_default(), _save_dict_as_csv(), _save_results_as_csv(), _save_results_as_json()

### Community 124 - "xraylabtool/kk/__init__.py"
Cohesion: 0.08
Nodes (10): main(), truncation_deficit(), _finite(), main(), _rel_corr(), probe_energies(), _random_call(), test_fuzz_covers_every_input_type() (+2 more)

### Community 125 - "json"
Cohesion: 0.09
Nodes (11): _in_process(), _subprocess(), _table(), test_version_unaffected(), TestKKCliEndToEnd, enable_offscreen_quiet_env(), cmd_kk(), _fail() (+3 more)

### Community 126 - "test_cli_edge_cases.py"
Cohesion: 0.07
Nodes (5): TestBatchRowResilience, TestCLIErrorHandling, TestCLIInputValidation, TestCLIPerformanceAndMemory, TestFieldNameValidation

### Community 127 - "cmd_convert"
Cohesion: 0.30
Nodes (8): _checkout(), _eval_points(), _finite(), _git(), main(), _poly(), _save(), _transform_case()

### Community 129 - "test_oracle.py"
Cohesion: 0.13
Nodes (12): f1_reference(), _gauss_legendre(), _h(), _pv_integral(), _quadrature(), _f2_mismatch(), oracle_energies(), _reference() (+4 more)

### Community 133 - "bucket"
Cohesion: 0.15
Nodes (17): test_block_for_examples(), test_block_for_never_exceeds_the_request(), test_bucket_is_multiple_of_8_monotone_and_bounded(), test_bucket_ladder_head(), test_bucket_pow2(), test_bucket_rejects_bad_minimum(), test_bucket_small_minimum_still_advances(), test_energy_blocks_have_few_distinct_shapes() (+9 more)

### Community 134 - "test_integration.py"
Cohesion: 0.08
Nodes (6): BaseIntegrationTest, generate_test_materials(), TestBasicSetupAndInitialization, TestDuplicatedFormulasIndependence, TestPropertyConsistency, TestSubRefracSiliconProperties

### Community 136 - "Getting Started Guide"
Cohesion: 0.08
Nodes (27): Getting Started Guide, xraylabtool batch CSV processing, calculate_single_material_properties (usage), calculate_xray_properties (usage), gpu_cuda12 / gpu_cuda13 extras, PySide6 GUI (Single, Multiple, Kramers-Kronig tabs), Installation (pip, conda, uv, GPU extras), Performance tips (preloaded elements, batching, energy arrays) (+19 more)

### Community 137 - "test_transform.py"
Cohesion: 0.09
Nodes (18): Adaptive refinement (refine_tolerance), xraylabtool kk CLI command, kk_calculate_real, load_nearedge, Low-level kk functions (compound_spectrum, kk_real_spectrum, evaluate_f2, to_asf_scale, merge_spectra, refine_spectrum), Merge points (affine match of near-edge data to tabulated f2), NearEdgeDataError, NearEdgeSpectrum (+10 more)

### Community 140 - "XRayLabToolError (base exception)"
Cohesion: 0.22
Nodes (5): memory_test(), performance_test(), decorator(), requires_memory(), decorator()

### Community 144 - "typing_extensions.py"
Cohesion: 0.22
Nodes (11): Available Scripts, Example Output:, Integration:, Maintenance Notes, Output:, Requirements:, `run_type_check.py` - Type Checking, Scripts Directory (+3 more)

### Community 146 - "constants.py"
Cohesion: 0.11
Nodes (7): TestEnergyToWavelength, attenuation_length_cm(), critical_angle_degrees(), energy_to_wavelength_angstrom(), _isclose(), validate_constants(), wavelength_angstrom_to_energy()

### Community 150 - "get_gpu_info"
Cohesion: 0.20
Nodes (9): ADR-001: JAX vs NumPy Computation Backend, Appendix: Functions that MUST NOT be JIT-compiled, Appendix: Functions to JIT-Compile (Priority Order), Consequences, Context, Decision, Negative, Positive (+1 more)

### Community 151 - "test_type_safety_config.py"
Cohesion: 0.14
Nodes (16): backend sub-package (NumPy/JAX abstraction), JAX Architecture & Design, JAX array immutability and lazy evaluation, Automatic differentiation (grad), Debugging and testing JAX code (disable jit, jax.debug.print, JIT-vs-eager parity), Future JAX features for v0.5+ (autodiff, vmap, distributed), GPU acceleration and device placement, JAX library (+8 more)

### Community 154 - "pathlib"
Cohesion: 0.11
Nodes (7): main(), _venv_python(), Colors, _sphinx(), test_kk_docs_build_adds_no_warnings(), test_kk_docs_doctests_pass(), slow()

### Community 155 - "test_warm_up.py"
Cohesion: 0.15
Nodes (11): cache_config(), test_cache_dir_is_os_appropriate(), test_cache_helper_enables_and_sets_threshold(), test_cache_helper_keeps_user_threshold(), test_cache_helper_respects_opt_out_and_existing_config(), test_cli_does_not_call_warm_up(), test_library_import_does_not_enable_cache(), test_warm_up_exported_lazily() (+3 more)

### Community 157 - "JAX Architecture & Design (v0.4.0)"
Cohesion: 0.20
Nodes (3): TestGetCompoundFamily, _classify_family_by_elements(), get_compound_family()

### Community 160 - "pytest"
Cohesion: 0.20
Nodes (15): test_database_rejects_non_integer_atomic_number(), test_evaluate_f2_rejects_complex(), test_evaluate_f2_rejects_nan_segment_edge(), test_kernel_rejects_complex_instead_of_dropping_the_imaginary_part(), test_kk_result_stores_merge_points_as_an_immutable_tuple(), test_merge_spectra_rejects_complex_inputs(), test_non_string_formula_raises_formula_error(), test_refine_rejects_complex_inputs() (+7 more)

### Community 161 - "kk_compiles.py"
Cohesion: 0.13
Nodes (8): _Handler, kk_programs(), numerical_programs(), record_compiles(), test_compile_recorder_sees_a_fresh_function_at_two_shapes(), test_refined_sio2_compiles_few_top_level_programs(), test_energy_length_bucket_recompiles_only_the_evaluation(), test_second_call_no_recompile()

### Community 162 - "os"
Cohesion: 0.18
Nodes (8): _call_worker(), call(), _gate(), main(), _recorder(), _run_worker(), _timed_ms(), _wait()

### Community 163 - "compute_kk"
Cohesion: 0.24
Nodes (12): test_compute_kk_default_grid_cxro_matches_result_energies(), test_compute_kk_forwards_energies_kev(), test_compute_kk_forwards_refine_tolerance(), test_compute_kk_merge_points_follow_energy_unit(), test_compute_kk_rejects_unknown_energy_unit(), test_compute_kk_service_matches_api_with_nearedge(), test_compute_kk_service_matches_api_without_nearedge(), test_cxro_comparison_empty_outside_tabulated_range() (+4 more)

### Community 166 - "test_compound_analysis.py"
Cohesion: 0.19
Nodes (4): TestGetCompoundFrequencyScore, TestGetRecommendedElementsForWarming, get_compound_frequency_score(), get_recommended_elements_for_warming()

### Community 167 - "FormulaError"
Cohesion: 0.12
Nodes (3): TestSubRefracErrorHandling, TestFormulaError, FormulaError

### Community 168 - "TestBatchProcessing"
Cohesion: 0.25
Nodes (7): Files, Maintenance, `performance_baselines.json`, `sample_materials.json`, Test Data Directory, `test_energies.json`, Usage in Tests

### Community 169 - "Testing Guide"
Cohesion: 0.29
Nodes (8): 30 keV tabulation cut (dashed vertical line), C (carbon) curve, Figure: f1 overestimated by truncating f2 integration at 30 keV, Fe (iron) curve, Kramers-Kronig f2-to-f1 integral truncated to tabulated range, Pb (lead) curve, f1 truncation error (f1_trunc - f1_full) vs photon energy, log-log, Result: truncation error grows with Z (C ~1e-4, Fe ~0.1, Pb ~2.5 e/atom), flat at low energy, rises sharply near 30 keV

### Community 173 - "atomic_cache.py"
Cohesion: 0.17
Nodes (7): _collect_compound_elements(), _collect_family_compound_elements(), _collect_similar_compound_elements(), _warm_atomic_data(), _warm_bulk_combinations(), warm_cache_for_compounds(), _warm_interpolators()

### Community 175 - "test_kernel_evaluates_only_blocks_holding_real_energies"
Cohesion: 0.14
Nodes (9): backend_restored(), _outside_device_helpers(), stray_conversions(), patch_method(), wrapper(), patch_numpy(), wrapper(), test_kernel_evaluates_only_blocks_holding_real_energies() (+1 more)

### Community 185 - "test_heavy_element_segment_grid_against_upstream_emulation"
Cohesion: 0.18
Nodes (8): _spy_blocks(), wrap(), test_heavy_element_segment_grid_against_upstream_emulation(), test_invalid_refine_arguments_raise_validation_error(), test_kk_calculate_real_block_never_exceeds_chunk_size(), test_refine_block_never_exceeds_chunk_size(), test_refine_rejects_non_increasing_input(), _upstream_like_added()

### Community 186 - "calculators/__init__.py"
Cohesion: 0.24
Nodes (4): formula(), _format_as_csv(), _format_as_json(), _format_as_table()

### Community 188 - "ResultValidator"
Cohesion: 0.18
Nodes (3): get_system_info(), log_system_info(), ResultValidator

### Community 191 - "kk/test_docs.py"
Cohesion: 0.23
Nodes (7): _public_objects(), test_docs_cite_the_paper_and_upstream_licence(), test_docs_pages_toctree_linked(), test_docs_state_the_eq10_erratum(), test_every_public_kk_name_is_documented(), test_public_docstrings_have_examples_and_references(), test_transform_docstrings_cite_paper_equations()

### Community 192 - "test_formula_parsing.py"
Cohesion: 0.17
Nodes (3): TestDelegatingWrappers, TestRegexCompatibility, _parse_formula()

### Community 195 - "test_midpoint_loop_never_evaluates_more_than_block_per_call"
Cohesion: 0.20
Nodes (6): dlog(), order(), pass_trace(), wrap(), spy(), test_midpoint_loop_never_evaluates_more_than_block_per_call()

### Community 199 - "refine_spectrum"
Cohesion: 0.33
Nodes (4): test_refine_rejects_bad_arguments(), _checked_inputs(), _grid_array(), refine_spectrum()

### Community 207 - "_outcome"
Cohesion: 0.25
Nodes (3): _log_slivers(), _outcome(), _RefineInfo

### Community 208 - "ADR-005: The Kramers-Kronig Package Is JAX-Only"
Cohesion: 0.29
Nodes (6): ADR-005: The Kramers-Kronig Package Is JAX-Only, Alternatives considered, Consequences, Context, Evidence, References

### Community 219 - "test_gui_kk_worker_end_to_end.py"
Cohesion: 0.18
Nodes (11): _check_table(), _setup(), test_close_during_warm_up_does_not_hang(), test_error_path_shows_dialog_and_stays_responsive(), test_nearedge_run_matches_api(), test_plain_run_matches_api(), test_refined_run_matches_api(), test_warm_up_failure_is_logged_not_raised() (+3 more)

### Community 236 - "_resize"
Cohesion: 0.50
Nodes (3): _resize(), head(), tail()

### Community 237 - "body"
Cohesion: 0.67
Nodes (3): padded(), trips(), body()

## Knowledge Gaps
- **123 isolated node(s):** `Henke, Gullikson, Davis 1993 X-ray interactions tables`, `Completion caching system (~/.xraylabtool/cache)`, `Completion for xtool alias`, `Completion features (command, option, path, value)`, `Conda/Mamba install and completion workflow` (+118 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 2220 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **110 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `v0.4.7 (2026-10-23)` connect `pytest` to `create_scattering_factor_interpolators`, `argparse`?**
  _High betweenness centrality (0.088) - this node is a cross-community bridge._
- **Are the 40 inferred relationships involving `kk_calculate_real()` (e.g. with `Evidence` and `main()`) actually correct?**
  _`kk_calculate_real()` has 40 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Henke, Gullikson, Davis 1993 X-ray interactions tables`, `Completion caching system (~/.xraylabtool/cache)`, `Completion for xtool alias` to the rest of the system?**
  _123 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `MainWindowProtocol` be split into smaller, more focused modules?**
  _Cohesion score 0.039473684210526314 - nodes in this community are weakly interconnected._
- **Why does `Changelog` connect `pytest` to `fixture`, `.calculate_derived_quantities`, `TestXRayResultTyping`, `test_type_safety_config.py`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `MainWindowProtocol` (e.g. with `KKTabMixin` and `MultiTabMixin`) actually correct?**
  _`MainWindowProtocol` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Should `parametrize` be split into smaller, more focused modules?**
  _Cohesion score 0.038850889192886456 - nodes in this community are weakly interconnected._