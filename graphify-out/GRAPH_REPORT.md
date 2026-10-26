# Graph Report - pyXRayLabTool  (2026-10-26)

## Corpus Check
- 208 files · ~200,008 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 112 file(s) not represented in the graph (top: .nff 93, .npz 12, .ipynb 2)

## Summary
- 5503 nodes · 12118 edges · 252 communities (141 shown, 111 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 879 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ba764fb5`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- MainWindowProtocol
- xraylabtool/utils.py
- test_jax_backend_parity.py
- compound_spectrum
- ValidationError
- BashCompletionGenerator
- api.py
- _make_cubic
- constants.py
- JaxPchipInterpolator
- test_concurrent_cache.py
- XRayResult
- validate_density
- calculate_single_material_properties
- parse_formula
- test_deep_debug_regressions.py
- test_api.py
- parse_formula
- Toast
- test_performance_benchmarks.py
- load_asf
- typing
- clear_scattering_factor_cache
- TestJaxBackendParity
- test_utilities.py
- style.py
- TestBasicExportFunctionality
- data_handling/__init__.py
- get_gpu_info
- TestBasicExportFunctionality
- validate_energy_range
- BatchConfig
- CompletionInstaller
- OverlayScrollbarMarginHelper
- test_cli.py
- test_deep_debug_regressions.py
- TestCICDIntegration
- test_integration.py
- TestKKCli
- single_tab.py
- MockXRayResult
- create_scattering_factor_interpolators
- pathlib
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
- clear_scattering_factor_cache
- Utilities Module
- legacy_install_completion_main
- device.py
- KKResult
- cmd_calc
- validate_chemical_formula
- argparse
- unit
- load_asf
- Any
- pathlib
- test_file_operations.py
- tests/conftest.py
- TestInstallCompletionCommand
- MaterialComparator
- typing
- get_bulk_atomic_data
- _OpsProxy
- scattering_data.py
- FastAtomicDataProvider
- set_backend
- legacy_install_completion_main
- format_calculation_summary
- TestXRayResultTyping
- transform.py
- pytest
- run_type_check.py
- FastXRayCalculationEngine
- test_device_kernel.py
- tests/conftest.py
- compute_kk
- PerformanceBenchmark
- export_to_csv
- kk_gpu_check.py
- BaseUnitTest
- Version 0.4.7 (2026-10-23)
- kk_calculate_real
- XRayLabTool Documentation (index)
- typing_extensions.py
- refine.py
- xraylabtool/utils.py
- test_exceptions.py
- compound_analysis.py
- TableFormatter
- get_atomic_number
- ValidationError
- EnergyError
- parse_energy_string
- CrystalStructure
- FormulaError
- TestCIWorkflowSimulation
- ADR-002: PyQtGraph vs Matplotlib for GUI Plotting
- FastXRayCalculationEngine
- test_device_merge.py
- _make_cubic
- cache.py
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
- energy_to_wavelength
- TestGoldenSi
- ADR-003: float64 Preservation Policy
- assert_kk_close
- test_kk_cli_end_to_end.py
- test_cli_edge_cases.py
- cmd_convert
- test_cli.py
- test_oracle.py
- atomic_cache.py
- scattering_data.py
- core.py
- bucket
- atomic_cache.py
- TestCrystallographicCalculations
- Getting Started Guide
- kk_calculate_real
- generate_test_energies
- calculators.core (calculate_single_material_properties, calculate_xray_properties)
- XRayLabToolError (base exception)
- format_xray_result
- test_container_validation.py
- kernels.py
- typing_extensions.py
- TestKKCliEndToEnd
- xraylabtool/__init__.py
- fixture
- kk_calculate_real
- validate_chemical_formula
- get_gpu_info
- test_type_safety_config.py
- _parse_formula
- ADR-001: JAX vs NumPy Computation Backend
- batch_processing.py
- test_warm_up.py
- Decision: replace matplotlib with PyQtGraph for interactive plots
- JAX Architecture & Design (v0.4.0)
- test_kk_plot.py
- math
- pytest
- kk_compiles.py
- json
- validators.py
- v0.4.0 release (2026-04-07)
- TestUnitConversionConstants
- test_compound_analysis.py
- device.py
- TestBatchProcessing
- Testing Guide
- api.py
- BaseXRayLabToolTest
- cmd_bragg
- TestBasicExportIntegration
- ndarray
- stray_conversions
- get_elements_for_compound
- TestDtypePreservation
- TestGuiErrorHandling
- .sum
- TestBatchConfig
- cmd_list
- TestCLICompatibility
- test_numerical_stability.py
- TestCLIErrorHandling
- TestPhysicalRealism
- _pchip_slopes
- AtomicDataProvider
- get_system_cuda_version
- check_gpu_availability
- cli.py
- kk/test_docs.py
- TestCLIFileOperations
- normalize_intensity
- calculate_multiple_xray_properties
- Data Handling Module API Doc
- test_fuzz.py
- TestRegexCompatibility
- Decision: enforce float64 globally; float32 is a correctness bug
- TestEdgeCaseInputValues
- TestFundamentalConstants
- test_kk_docs_build.py
- test_device.py
- TestBatchProcessingMemoryManagement
- test_core.py
- ResultValidator
- ADR-005: The Kramers-Kronig Package Is JAX-Only
- performance/conftest.py
- conf.py
- ensure_complex128_array
- gc
- TestJaxBackendParity
- test_kk_cli_end_to_end.py
- TestRegexCompatibility
- TestFormulaParsing
- test_gui_kk_worker_end_to_end.py
- batch_config_performance
- integration/__init__.py
- performance/__init__.py
- unit/__init__.py
- BaseIntegrationTest
- BasePerformanceTest
- BaseUnitTest
- test_data_generator
- benchmark_baseline
- optimize_array_memory_layout
- sample_batch_data
- TestCLICompatibility
- test_materials
- skip_if_low_memory
- test_energies_large
- _configure_jax_float64
- large_batch_data
- material_and_density
- calculate_single_material_properties
- assert_calculation_result_valid
- mock_low_memory

## God Nodes (most connected - your core abstractions)
1. `kk_calculate_real()` - 119 edges
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

## Communities (252 total, 111 thin omitted)

### Community 0 - "MainWindowProtocol"
Cohesion: 0.04
Nodes (5): KKTabMixin, MultiTabMixin, MainWindowProtocol, SingleTabMixin, MaterialTable

### Community 1 - "xraylabtool/utils.py"
Cohesion: 0.12
Nodes (9): _check_lattice(), _convert_atomic_number_to_int(), d_spacing_cubic(), d_spacing_orthorhombic(), d_spacing_tetragonal(), _expand_parentheses(), get_avogadro_number(), progress_bar() (+1 more)

### Community 2 - "test_jax_backend_parity.py"
Cohesion: 0.07
Nodes (12): backend_restored(), TestBackendSwitching, TestInterpolationFactory, TestOpsProxy, TestBackendState, restore_numpy_backend(), _auto_select_backend(), get_backend() (+4 more)

### Community 3 - "compound_spectrum"
Cohesion: 0.11
Nodes (18): si(), _numpy_assembly(), test_compound_coefficients_equal_the_pre_jax_baseline(), test_compound_coefficients_round_like_a_separate_multiply_and_add(), test_compound_grid_is_union_of_boundaries(), test_compound_spectrum_rejects_bad_input(), test_docstring_examples_run(), test_element_order_does_not_change_the_result() (+10 more)

### Community 4 - "ValidationError"
Cohesion: 0.04
Nodes (16): TestAtomicDataError, TestBatchProcessingError, TestCalculationError, TestEnergyError, TestExceptionHierarchy, TestExceptionUsage, TestFormulaError, TestUnknownElementError (+8 more)

### Community 5 - "BashCompletionGenerator"
Cohesion: 0.05
Nodes (5): test_cli_aliases_registered(), CompletionGenerator, FishCompletionGenerator, PowerShellCompletionGenerator, ZshCompletionGenerator

### Community 6 - "api.py"
Cohesion: 0.05
Nodes (31): _checked_merge(), _compound_capacity(), _explicit_energies(), _host_inputs(), _merge_points_ev(), _near_edge_call(), _NearEdgeCall, _Plan (+23 more)

### Community 7 - "_make_cubic"
Cohesion: 0.05
Nodes (8): TestDuplicatedFormulasIndependence, TestEdgeCasesAndErrorHandling, TestH2OProperties, TestInputValidationAndErrorHandling, TestPropertyConsistency, TestSiO2Properties, calculate_xray_properties(), _validate_xray_inputs()

### Community 8 - "constants.py"
Cohesion: 0.06
Nodes (11): ColorPalette, apply_palette_to_item(), apply_palette_to_widget(), current_palette(), legend_kwargs(), _recolor_axis_label(), _restyle_legend(), KKPlot (+3 more)

### Community 9 - "JaxPchipInterpolator"
Cohesion: 0.15
Nodes (9): _query(), restore_x64(), _scipy(), TestMatchesScipy, TestPrecision, TestQueryHandling, TestShapePreservation, TestValidation (+1 more)

### Community 10 - "test_concurrent_cache.py"
Cohesion: 0.09
Nodes (11): clear_lru_caches(), _mock_get_element(), TestCacheResultConsistency, worker(), TestConcurrentGetAtomicNumber, worker(), cache_clearer(), reader() (+3 more)

### Community 11 - "XRayResult"
Cohesion: 0.12
Nodes (37): _edge_spectrum(), test_asf_input_is_unchanged_and_unknown_type_raises(), test_beta_round_trip_and_density_cancels_at_merge(), test_default_merge_points_are_the_overlap_of_both_ranges(), test_directly_constructed_spectrum_is_validated(), test_docstring_examples_run(), test_file_in_kev_declared_ev_raises_no_overlap(), test_flat_endpoints_raise_not_inf() (+29 more)

### Community 13 - "calculate_single_material_properties"
Cohesion: 0.07
Nodes (18): calculate_multiple_xray_properties(), _calculate_single_material_xray_properties(), _coerce_energy_array(), _convert_energy_input(), _create_process_formula_function(), process_formula(), FastXRayCalculationEngine, get_calculation_engine() (+10 more)

### Community 14 - "parse_formula"
Cohesion: 0.07
Nodes (8): si(), TestParseFormulaBasic, TestParseFormulaDecimal, TestParseFormulaErrors, TestParseFormulaParentheses, _aggregate_element_counts(), parse_formula(), _validate_balanced_parentheses()

### Community 15 - "test_deep_debug_regressions.py"
Cohesion: 0.09
Nodes (8): test_whitespace_delimited_loader_works_with_current_pandas(), TestExports, TestLazySubmoduleAccess, TestRepeatedMaterials, TestScatteringDataIntegrity, TestUtilityValidation, _two_results(), load_data_file()

### Community 16 - "test_api.py"
Cohesion: 0.17
Nodes (18): test_backend_is_jax_and_chunk_size_does_not_matter(), test_boundary_energies_are_accepted_and_f2_matches_the_table(), test_default_grid_clipped_to_0p03_30_kev_and_logged(), test_docstring_examples_run(), test_empty_default_grid_after_merge_raises(), test_energies_are_copied_not_aliased(), test_energy_one_ulp_outside_the_data_range_raises(), test_energy_out_of_data_range_raises_energy_error() (+10 more)

### Community 18 - "Toast"
Cohesion: 0.07
Nodes (3): StyleGuideValidator, StyleViolation, ValidationReport

### Community 19 - "test_performance_benchmarks.py"
Cohesion: 0.06
Nodes (5): TestBenchmarkComparison, TestPerformanceBenchmarks, TestPerformanceRegression, TestScalabilityBenchmarks, timer()

### Community 20 - "load_asf"
Cohesion: 0.14
Nodes (28): _args(), _eq10_corrected(), _f2_poly(), _golden(), _linear_segments(), _quadrature_f1(), test_chunk_invariance(), test_constant_band_closed_form() (+20 more)

### Community 21 - "typing"
Cohesion: 0.09
Nodes (34): _max_midpoint_error(), _refine(), _spy_blocks(), wrap(), test_debug_log_per_pass_and_info_total(), test_docstring_examples_run(), test_exhaustion_flagged_and_logged(), test_heavy_element_segment_grid_against_upstream_emulation() (+26 more)

### Community 22 - "clear_scattering_factor_cache"
Cohesion: 0.09
Nodes (6): _clear_cache(), _single(), TestBatchProcessorEquivalence, cold_cache(), TestPerformanceRegression, clear_scattering_factor_cache()

### Community 23 - "TestJaxBackendParity"
Cohesion: 0.07
Nodes (15): TestResultFormatting, _format_array_field_single(), _format_as_csv(), _format_as_json(), _format_filtered_table(), _format_material_properties(), _format_multiple_energies(), _format_multiple_energy_section() (+7 more)

### Community 24 - "test_utilities.py"
Cohesion: 0.09
Nodes (13): kk_log(), x64_restored(), test_configure_logging_writes_file(), test_get_logger_child_inherits_config(), build_parser(), main(), _bool_env(), _coerce_int_env() (+5 more)

### Community 26 - "TestBasicExportFunctionality"
Cohesion: 0.05
Nodes (36): ADR-002: PyQtGraph vs Matplotlib for GUI Plotting, Appendix: Feature Parity Checklist, Consequences, Context, Decision, Migration Details, Negative, Positive (+28 more)

### Community 27 - "data_handling/__init__.py"
Cohesion: 0.14
Nodes (8): get_atomic_data_fast(), get_cache_stats(), is_element_preloaded(), warm_up_cache(), load_batch_input(), save_batch_results(), _save_dataframe(), __getattr__()

### Community 30 - "validate_energy_range"
Cohesion: 0.07
Nodes (4): TestValidateCalculationParameters, TestValidateEnergyRange, validate_calculation_parameters(), validate_energy_range()

### Community 31 - "BatchConfig"
Cohesion: 0.10
Nodes (6): TestBatchConfig, TestBatchInputValidation, TestBatchPartialFailure, BatchConfig, calculate_batch_properties(), _initialize_progress_bar()

### Community 32 - "CompletionInstaller"
Cohesion: 0.06
Nodes (3): TestSentinelMarkers, CompletionInstaller, test_completion_installation()

### Community 34 - "test_cli.py"
Cohesion: 0.07
Nodes (9): form(), _get_app(), qt_app(), TestDensityValidation, TestEnergyGridValidation, TestFormulaValidation, TestFormValues, TestPlotAxisLabelTheming (+1 more)

### Community 35 - "test_deep_debug_regressions.py"
Cohesion: 0.21
Nodes (10): _computation(), _pen_colors(), plot(), qt_app(), test_plot_draws_curves_on_unequal_grids(), test_plot_follows_theme_toggle(), test_plot_hides_missing_curves(), test_plot_log_x_rule() (+2 more)

### Community 36 - "TestCICDIntegration"
Cohesion: 0.11
Nodes (13): _run_smoke(), test_compute_multiple_reports_progress_to_100(), test_headless_gui_smoke_runs_offscreen(), _ensure_app(), test_export_csv_handles_io_error(), test_gui_smoke_offscreen(), test_gui_smoke_threaded_offscreen(), test_duplicate_formula_different_density_not_dropped() (+5 more)

### Community 38 - "TestKKCli"
Cohesion: 0.06
Nodes (18): _parse_csv(), _run(), test_empty_default_grid_after_merge_exits_1(), TestKKCli, _write_edge(), add_atomic_command(), add_batch_command(), add_bragg_command() (+10 more)

### Community 40 - "MockXRayResult"
Cohesion: 0.10
Nodes (3): MockXRayResult, TestFormatXRayResult, format_xray_result()

### Community 41 - "create_scattering_factor_interpolators"
Cohesion: 0.06
Nodes (40): Calculators Module (API docs), xraylabtool.calculators.cache, calculate_xray_properties, xraylabtool.calculators.core, Critical angle theta_c = sqrt(2 delta), xraylabtool.calculators.derived_quantities, xraylabtool.calculators.kernels, Legacy CamelCase accessors (DeprecationWarning) (+32 more)

### Community 42 - "pathlib"
Cohesion: 0.06
Nodes (4): TestEnvironmentIntegration, TestEnvironmentProperties, TestEnvironmentType, EnvironmentType

### Community 43 - "typing"
Cohesion: 0.17
Nodes (7): cmd_batch(), _convert_result_to_dict(), _parse_batch_data(), _process_batch_materials(), _save_batch_results(), _validate_batch_input(), create_batch_progress_tracker()

### Community 44 - "installer.py"
Cohesion: 0.08
Nodes (7): TestCompletionInstallToTempDir, TestCompletionScriptGeneration, TestBashCompletionSubstitution, BashCompletionGenerator, CompletionManager, get_global_options(), get_xraylabtool_commands()

### Community 46 - "CalculationWorker"
Cohesion: 0.13
Nodes (8): assert_values_close(), get_system_info(), log_system_info(), skip_if_insufficient_cpu(), skip_if_insufficient_memory(), stability_test(), validate_batch_performance(), values_close()

### Community 47 - "installer.py"
Cohesion: 0.09
Nodes (5): apply_pyqtgraph_theme(), apply_styles(), apply_theme(), get_qss(), ThemeManager

### Community 48 - "MaterialComparator"
Cohesion: 0.08
Nodes (7): _check_element(), cold_cache(), test_si_o_au_interpolators_are_independent(), TestPchipAu, TestPchipO, TestPchipSi, create_scattering_factor_interpolators()

### Community 49 - "test_utils.py"
Cohesion: 0.08
Nodes (8): BasePerformanceTest, TestArrayOptimization, TestBulkAtomicDataOptimization, TestDeprecationWarningOptimization, TestElementPathOptimization, TestInterpolatorCaching, TestNumericalStabilityOptimizations, TestPerformanceRegressionDetection

### Community 50 - "test_device_spectrum.py"
Cohesion: 0.08
Nodes (24): Decision, _evaluate_f2_device(), test_transfer_helpers_are_counted(), test_device_loop_runs_under_transfer_guard(), test_compound_spectrum_device_under_strict_guard(), test_evaluate_f2_device_under_strict_guard(), test_host_to_asf_scale_beta_passes_the_strict_guard(), call() (+16 more)

### Community 51 - "main"
Cohesion: 0.06
Nodes (8): TestAtomicCommand, TestBatchCommand, TestCLIIntegration, TestMainFunction, cmd_atomic(), cmd_install_completion(), cmd_uninstall_completion(), main()

### Community 52 - "fixture"
Cohesion: 0.07
Nodes (38): Platform and Python version compatibility matrix, Contributing Guide, Contributing atomic data (citation, range, precision, format), Code standards (PEP 8, 88 cols, NumPy docstrings, type hints), Performance requirements (<0.1 ms single, >100k calc/s batch, >90% cache hit), Review process (automated checks, reviewer, maintainer, squash merge), Development workflow (branching, ruff, mypy, pytest, make verify), Performance and Optimization guide (+30 more)

### Community 53 - ".calculate_derived_quantities"
Cohesion: 0.07
Nodes (39): widgets.kk_plot (Kramers-Kronig Plot), main_window.kk_tab, kk sub-package (Kramers-Kronig, kkcalc port), Kramers-Kronig (kk) API docs, xraylabtool.kk.api (high-level API), AsfDatabase, compound_spectrum, xraylabtool.kk.database (+31 more)

### Community 54 - "clear_scattering_factor_cache"
Cohesion: 0.07
Nodes (3): TestGoldenAu, TestGoldenDerivedQuantities, TestGoldenScatteringFactors

### Community 55 - "Utilities Module"
Cohesion: 0.08
Nodes (3): TestEdgeCases, TestNumpyBackend, NumpyBackend

### Community 57 - "device.py"
Cohesion: 0.19
Nodes (21): app(), _small_run(), test_energies_outside_cxro_range_reach_the_service(), test_energy_range_controls_reach_the_service(), spy(), test_exhausted_refinement_shows_a_warning_toast(), test_export_csv_columns(), test_export_csv_io_error_is_reported() (+13 more)

### Community 58 - "KKResult"
Cohesion: 0.10
Nodes (14): _manual(), test_negative_f1_near_edge_reaches_xray_result(), test_result_module_doctests(), test_to_xray_result_accepts_range_ends(), test_to_xray_result_rejects_out_of_range_energy(), test_to_xray_result_uses_kk_f1(), test_helper_delta_beta_match_closed_form(), test_helper_rejects_complex_input() (+6 more)

### Community 62 - "unit"
Cohesion: 0.09
Nodes (7): _calc(), _clear_cache(), TestComplexFormula, TestElectronCountGolden, TestExactCapturedValues, TestMolecularWeightGolden, _calculate_molecular_properties()

### Community 63 - "load_asf"
Cohesion: 0.07
Nodes (33): _asf_f2(), test_asf_grid_spans_10ev_to_500kev(), test_asf_has_92_elements(), test_cached_tables_are_read_only(), test_im_coeffs_shape(), test_load_asf_is_cached(), test_nff_divergent_elements(), _assemble() (+25 more)

### Community 65 - "pathlib"
Cohesion: 0.08
Nodes (20): collect(), main(), _search(), check_accessibility(), check_documentation_coverage(), Colors, main(), print_section() (+12 more)

### Community 66 - "test_file_operations.py"
Cohesion: 0.10
Nodes (18): TestDataFileError, test_load_absolute_path(), test_load_csv_file(), test_load_csv_space_separated_fallback(), test_load_empty_nff_raises_data_file_error(), test_load_malformed_nff_raises_data_file_error(), test_load_nff_energy_column_monotone(), test_load_nff_values_are_finite() (+10 more)

### Community 67 - "tests/conftest.py"
Cohesion: 0.12
Nodes (12): work(), qt_app(), _run_worker(), test_worker_does_not_emit_finished_on_exception(), test_worker_emits_error_on_exception(), bad_fn(), test_worker_emits_result_on_success(), test_worker_injects_progress_cb_when_accepted() (+4 more)

### Community 70 - "typing"
Cohesion: 0.09
Nodes (3): TestBasicAnalysis, compare_materials(), find_absorption_edges()

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

### Community 78 - "TestXRayResultTyping"
Cohesion: 0.10
Nodes (22): GUI (API docs), xraylabtool.gui.logging_filters, xraylabtool.gui.main_window, widgets.material_form, widgets.material_table, main_window.multi_tab, widgets.plot_canvas, widgets.scrollbar_helper (+14 more)

### Community 79 - "transform.py"
Cohesion: 0.06
Nodes (19): main(), truncation_deficit(), _finite(), main(), _rel_corr(), main(), _run(), probe_energies() (+11 more)

### Community 80 - "pytest"
Cohesion: 0.07
Nodes (30): CalculationError, EnergyError, FormulaError, ValidationError, XRayLabToolError, Changelog, Batch API fixes (ordering, timeouts, process-pool fallback, formula@density keys), CLI aliases xtool / xlt and GUI aliases xtool-gui / xlt-gui (+22 more)

### Community 81 - "run_type_check.py"
Cohesion: 0.11
Nodes (20): Analytic per-interval integration (Eqs. 3-4), Biggs and Lighthill high-energy cross-sections (to 500 keV), Cauchy principal value integral, Conventional KK numerical methods (FFT, Euler-Maclaurin, Gaussian, TANH), Zero-to-infinity convergence conditions on polynomial coefficients, Cross et al. difference-spectrum KK method, Double-interval integration across x=E singularity (Eq. 5), Closed-form f1(E) for piecewise Laurent f2 (Eq. 7, Eq. 10) (+12 more)

### Community 83 - "test_device_kernel.py"
Cohesion: 0.17
Nodes (18): _case(), _eval(), test_kernel_without_x64_raises_the_documented_error_without_a_warning(), test_kk_f1_grad(), loss(), test_kk_f1_jit_vmap(), test_kk_f1_keeps_input_shape_and_rejects_bad_block(), test_output_float64_and_shapes() (+10 more)

### Community 84 - "tests/conftest.py"
Cohesion: 0.11
Nodes (8): assert_memory_usage_reasonable(), assert_performance_within_threshold(), energy_arrays(), performance_thresholds(), performance_timer(), pytest_addoption(), pytest_collection_modifyitems(), pytest_configure()

### Community 85 - "compute_kk"
Cohesion: 0.15
Nodes (14): test_compute_kk_default_grid_cxro_matches_result_energies(), test_compute_kk_forwards_energies_kev(), test_compute_kk_forwards_refine_tolerance(), test_compute_kk_merge_points_follow_energy_unit(), test_compute_kk_rejects_unknown_energy_unit(), test_compute_kk_service_matches_api_with_nearedge(), test_compute_kk_service_matches_api_without_nearedge(), test_cxro_comparison_empty_outside_tabulated_range() (+6 more)

### Community 86 - "PerformanceBenchmark"
Cohesion: 0.14
Nodes (6): TestFormulaCommand, cmd_formula(), _get_atomic_data(), _output_formula_results(), _print_formula_results(), _process_formula()

### Community 87 - "export_to_csv"
Cohesion: 0.13
Nodes (8): export_to_csv(), export_to_json(), _json_numpy_default(), save_calculation_results(), _save_dict_as_csv(), _save_results_as_csv(), _save_results_as_json(), __getattr__()

### Community 88 - "kk_gpu_check.py"
Cohesion: 0.09
Nodes (21): check_additivity(), check_bench(), check_f2_si_edge(), check_oracle(), check_padding(), check_transfers(), _f2_numpy(), guarded() (+13 more)

### Community 89 - "BaseUnitTest"
Cohesion: 0.09
Nodes (3): BaseUnitTest, TestDocumentationPatterns, TestStyleCompliance

### Community 91 - "kk_calculate_real"
Cohesion: 0.07
Nodes (36): _colliding_pair(), _passes(), _run(), test_accelerator_block_rule_gives_the_same_result(), test_backend_field_is_jax(), test_chunk_size_is_upper_bound(), test_cold_concurrent_first_calls_upload_the_tables_once(), test_cold_table_upload_is_counted_once_and_separately() (+28 more)

### Community 92 - "XRayLabTool Documentation (index)"
Cohesion: 0.08
Nodes (29): Runtime NumPy/JAX backend switching (set_backend), Modular package architecture (backend, calculators, data_handling, interfaces, io, validation, gui), Kramers-Kronig Guide, Near-edge input types (nexafs, beta, asf), kkcalc attribution (zlib/libpng, Watts and Lauk), kk module limits (no tail, conditioning, pole skipping, wrong energy-unit hazard), Near-edge file format, Watts 2014, Opt. Express 22(19), 23628 (piecewise Laurent polynomial KK) (+21 more)

### Community 93 - "typing_extensions.py"
Cohesion: 0.18
Nodes (6): get_optimal_chunk_size(), is_complex_array(), is_energy_array(), is_real_array(), validate_complex_array(), validate_energy_array()

### Community 94 - "refine.py"
Cohesion: 0.06
Nodes (31): padded(), _check_refine_arguments(), _accumulator_need(), _checked_inputs(), _finalize(), _first_pass(), _fit_active(), _fit_columns() (+23 more)

### Community 95 - "xraylabtool/utils.py"
Cohesion: 0.21
Nodes (12): CLI error handling messages, Validation Module API Doc, Validation best practices (validate first, catch specific exceptions), CalculationError, EnergyError, FormulaError, validate_calculation_parameters, validate_chemical_formula (+4 more)

### Community 99 - "get_atomic_number"
Cohesion: 0.12
Nodes (5): TestAtomicDataLookup, worker(), get_atomic_data(), get_atomic_number(), get_atomic_weight()

### Community 100 - "ValidationError"
Cohesion: 0.10
Nodes (5): TestValidationError, ValidationError, _asf_scale_arguments(), _check_positive(), _readonly_1d()

### Community 103 - "CrystalStructure"
Cohesion: 0.09
Nodes (3): TestCrystalStructure, TestLoadDataFile, CrystalStructure

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
Cohesion: 0.10
Nodes (40): _assert_matches_reference(), _explicit_scale(), _poly(), _reference(), _run(), _status_of(), test_adjacent_float_knots_use_lower_boundary(), test_decimal_scale_of_the_near_edge_values_matches_scale_one() (+32 more)

### Community 109 - "_make_cubic"
Cohesion: 0.10
Nodes (4): _make_cubic(), _phase(), TestCrystalStructureAPI, TestStructureFactorPhysics

### Community 110 - "cache.py"
Cohesion: 0.12
Nodes (7): TestAtomicDataCacheIsImmutable, get_bulk_atomic_data(), is_element_cached(), _smart_cache_warming(), _warm_priority_cache(), _background_cache_warming(), get_bulk_atomic_data_fast()

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
Cohesion: 0.09
Nodes (37): _device_run(), dlog(), _host_reference(), _logged(), _narrow_last_segment_table(), order(), pass_trace(), wrap() (+29 more)

### Community 119 - "TestPerformanceBenchmarks"
Cohesion: 0.14
Nodes (5): TestPerformanceBenchmarks, energy_sweep(), complex_formula(), multi_calculation(), single_calculation()

### Community 121 - "energy_to_wavelength"
Cohesion: 0.13
Nodes (6): TestUnitConversions, energy_to_wavelength(), get_electron_charge(), get_planck_constant(), get_speed_of_light(), wavelength_to_energy()

### Community 124 - "assert_kk_close"
Cohesion: 0.09
Nodes (14): _cxro_deviation(), test_30kev_truncation_matches_paper_trend(), test_default_grid_matches_upstream_golden_bit_for_bit_inputs(), test_end_to_end_matches_upstream_golden(), test_tabulated_kk_f1_close_to_cxro_f1(), _load_golden(), _merged(), test_kernel_on_merged_spectrum_matches_golden_f1_points() (+6 more)

### Community 125 - "test_kk_cli_end_to_end.py"
Cohesion: 0.17
Nodes (6): enable_offscreen_quiet_env(), cmd_kk(), _fail(), _format_output(), _parse_energies(), _parse_merge()

### Community 126 - "test_cli_edge_cases.py"
Cohesion: 0.08
Nodes (4): TestBatchRowResilience, TestCLIInputValidation, TestCLIPerformanceAndMemory, TestFieldNameValidation

### Community 127 - "cmd_convert"
Cohesion: 0.32
Nodes (8): _checkout(), _eval_points(), _finite(), _git(), main(), _poly(), _save(), _transform_case()

### Community 129 - "test_oracle.py"
Cohesion: 0.13
Nodes (12): f1_reference(), _gauss_legendre(), _h(), _pv_integral(), _quadrature(), _f2_mismatch(), oracle_energies(), _reference() (+4 more)

### Community 130 - "atomic_cache.py"
Cohesion: 0.10
Nodes (3): TestUninstallCompletionMain, uninstall_completion_main(), legacy_uninstall_completion_main()

### Community 131 - "scattering_data.py"
Cohesion: 0.08
Nodes (5): TestAtomicScatteringFactor, AtomicScatteringFactor, _initialize_element_paths(), load_scattering_factor_data(), ScatteringData

### Community 132 - "core.py"
Cohesion: 0.12
Nodes (7): calculate_derived_quantities(), calculate_scattering_factors(), _derived_quantities_kernel(), _get_derived_kernel(), _get_scattering_kernel(), _scattering_math_kernel(), InterpolatorProtocol

### Community 133 - "bucket"
Cohesion: 0.13
Nodes (19): _budget(), platform(), test_block_for_examples(), test_block_for_never_exceeds_the_request(), test_bucket_is_multiple_of_8_monotone_and_bounded(), test_bucket_ladder_head(), test_bucket_pow2(), test_bucket_rejects_bad_minimum() (+11 more)

### Community 134 - "atomic_cache.py"
Cohesion: 0.17
Nodes (7): _collect_compound_elements(), _collect_family_compound_elements(), _collect_similar_compound_elements(), _warm_atomic_data(), _warm_bulk_combinations(), warm_cache_for_compounds(), _warm_interpolators()

### Community 136 - "Getting Started Guide"
Cohesion: 0.08
Nodes (27): Getting Started Guide, xraylabtool batch CSV processing, calculate_single_material_properties (usage), calculate_xray_properties (usage), gpu_cuda12 / gpu_cuda13 extras, PySide6 GUI (Single, Multiple, Kramers-Kronig tabs), Installation (pip, conda, uv, GPU extras), Performance tips (preloaded elements, batching, energy arrays) (+19 more)

### Community 137 - "kk_calculate_real"
Cohesion: 0.09
Nodes (18): Adaptive refinement (refine_tolerance), xraylabtool kk CLI command, kk_calculate_real, load_nearedge, Low-level kk functions (compound_spectrum, kk_real_spectrum, evaluate_f2, to_asf_scale, merge_spectra, refine_spectrum), Merge points (affine match of near-edge data to tabulated f2), NearEdgeDataError, NearEdgeSpectrum (+10 more)

### Community 138 - "generate_test_energies"
Cohesion: 0.14
Nodes (3): generate_test_energies(), generate_test_materials(), ParametrizedTestMixin

### Community 140 - "XRayLabToolError (base exception)"
Cohesion: 0.22
Nodes (5): memory_test(), performance_test(), decorator(), requires_memory(), decorator()

### Community 141 - "format_xray_result"
Cohesion: 0.24
Nodes (4): formula(), _format_as_csv(), _format_as_json(), _format_as_table()

### Community 142 - "test_container_validation.py"
Cohesion: 0.38
Nodes (6): _outcome(), _result(), test_kk_result_arrays_are_read_only_copies(), test_kk_result_rejects_inconsistent_state(), test_refine_outcome_arrays_are_read_only_copies(), test_refine_outcome_rejects_inconsistent_state()

### Community 144 - "typing_extensions.py"
Cohesion: 0.22
Nodes (11): Available Scripts, Example Output:, Integration:, Maintenance Notes, Output:, Requirements:, `run_type_check.py` - Type Checking, Scripts Directory (+3 more)

### Community 145 - "TestKKCliEndToEnd"
Cohesion: 0.25
Nodes (4): _assert_matches(), _in_process(), _table(), TestKKCliEndToEnd

### Community 146 - "xraylabtool/__init__.py"
Cohesion: 0.11
Nodes (9): TestEnergyToWavelength, get_cached_elements(), attenuation_length_cm(), critical_angle_degrees(), energy_to_wavelength_angstrom(), _isclose(), validate_constants(), wavelength_angstrom_to_energy() (+1 more)

### Community 147 - "fixture"
Cohesion: 0.12
Nodes (8): batch_config_default(), batch_config_small(), cpu_count(), memory_monitor(), memory_tracker(), sample_calculation_result(), test_energies_medium(), test_energies_small()

### Community 150 - "get_gpu_info"
Cohesion: 0.20
Nodes (9): ADR-001: JAX vs NumPy Computation Backend, Appendix: Functions that MUST NOT be JIT-compiled, Appendix: Functions to JIT-Compile (Priority Order), Consequences, Context, Decision, Negative, Positive (+1 more)

### Community 151 - "test_type_safety_config.py"
Cohesion: 0.14
Nodes (16): backend sub-package (NumPy/JAX abstraction), JAX Architecture & Design, JAX array immutability and lazy evaluation, Automatic differentiation (grad), Debugging and testing JAX code (disable jit, jax.debug.print, JIT-vs-eager parity), Future JAX features for v0.5+ (autodiff, vmap, distributed), GPU acceleration and device placement, JAX library (+8 more)

### Community 154 - "batch_processing.py"
Cohesion: 0.08
Nodes (14): expect_no_warnings(), expect_warnings(), no_progress_config(), TestProcessSingleCalculation, chunk_iterator(), _filter_dataframe_fields(), _get_energy_point_data(), _prepare_energies_array() (+6 more)

### Community 155 - "test_warm_up.py"
Cohesion: 0.13
Nodes (12): test_cold_concurrent_first_call_uploads_the_tables_once(), cache_config(), test_cache_dir_is_os_appropriate(), test_cache_helper_enables_and_sets_threshold(), test_cache_helper_keeps_user_threshold(), test_cache_helper_respects_opt_out_and_existing_config(), test_cli_does_not_call_warm_up(), test_library_import_does_not_enable_cache() (+4 more)

### Community 157 - "JAX Architecture & Design (v0.4.0)"
Cohesion: 0.20
Nodes (3): TestGetCompoundFamily, _classify_family_by_elements(), get_compound_family()

### Community 159 - "math"
Cohesion: 0.14
Nodes (3): TestDtypePreservation, TestMathematicalConstants, _atomic_numbers()

### Community 160 - "pytest"
Cohesion: 0.20
Nodes (15): test_database_rejects_non_integer_atomic_number(), test_evaluate_f2_rejects_complex(), test_evaluate_f2_rejects_nan_segment_edge(), test_kernel_rejects_complex_instead_of_dropping_the_imaginary_part(), test_kk_result_stores_merge_points_as_an_immutable_tuple(), test_merge_spectra_rejects_complex_inputs(), test_non_string_formula_raises_formula_error(), test_refine_rejects_complex_inputs() (+7 more)

### Community 161 - "kk_compiles.py"
Cohesion: 0.12
Nodes (11): _call_worker(), call(), _recorder(), _Handler, kk_programs(), numerical_programs(), record_compiles(), test_refined_sio2_compiles_few_top_level_programs() (+3 more)

### Community 162 - "json"
Cohesion: 0.16
Nodes (8): build(), main(), _gate(), _kernel_worker(), main(), _run_worker(), _timed_ms(), _wait()

### Community 163 - "validators.py"
Cohesion: 0.17
Nodes (3): TestDelegatingWrappers, _get_valid_element_symbols(), _parse_formula()

### Community 166 - "test_compound_analysis.py"
Cohesion: 0.19
Nodes (4): TestGetCompoundFrequencyScore, TestGetRecommendedElementsForWarming, get_compound_frequency_score(), get_recommended_elements_for_warming()

### Community 167 - "device.py"
Cohesion: 0.23
Nodes (4): main(), check_plugin_conflicts(), get_device_info(), get_recommended_package()

### Community 168 - "TestBatchProcessing"
Cohesion: 0.25
Nodes (7): Files, Maintenance, `performance_baselines.json`, `sample_materials.json`, Test Data Directory, `test_energies.json`, Usage in Tests

### Community 169 - "Testing Guide"
Cohesion: 0.29
Nodes (8): 30 keV tabulation cut (dashed vertical line), C (carbon) curve, Figure: f1 overestimated by truncating f2 integration at 30 keV, Fe (iron) curve, Kramers-Kronig f2-to-f1 integral truncated to tabulated range, Pb (lead) curve, f1 truncation error (f1_trunc - f1_full) vs photon energy, log-log, Result: truncation error grows with Z (C ~1e-4, Fe ~0.1, Pb ~2.5 e/atom), flat at low energy, rises sharply near 30 keV

### Community 172 - "cmd_bragg"
Cohesion: 0.16
Nodes (3): TestBraggCommand, cmd_bragg(), bragg_angle()

### Community 174 - "ndarray"
Cohesion: 0.11
Nodes (4): arrays_close(), assert_arrays_close(), create_batch_test_data(), TestDataFactory

### Community 175 - "stray_conversions"
Cohesion: 0.20
Nodes (7): _outside_device_helpers(), stray_conversions(), patch_method(), wrapper(), patch_numpy(), wrapper(), test_kernel_evaluates_only_blocks_holding_real_energies()

### Community 178 - "TestGuiErrorHandling"
Cohesion: 0.22
Nodes (3): numpy_backend_after(), TestGuiErrorHandling, task()

### Community 190 - "cli.py"
Cohesion: 0.07
Nodes (9): TestBackwardCompatibility, TestInstallCompletionMain, cmd_completion(), completion_main(), create_completion_parser(), handle_completion_command(), install_completion_main(), show_completion_info() (+1 more)

### Community 191 - "kk/test_docs.py"
Cohesion: 0.23
Nodes (7): _public_objects(), test_docs_cite_the_paper_and_upstream_licence(), test_docs_pages_toctree_linked(), test_docs_state_the_eq10_erratum(), test_every_public_kk_name_is_documented(), test_public_docstrings_have_examples_and_references(), test_transform_docstrings_cite_paper_equations()

### Community 193 - "normalize_intensity"
Cohesion: 0.11
Nodes (6): TestDataProcessing, background_subtraction(), find_peaks(), normalize_intensity(), save_processed_data(), smooth_data()

### Community 195 - "Data Handling Module API Doc"
Cohesion: 0.20
Nodes (11): Data Handling Module API Doc, atomic_cache (92-element preload, warm_up_cache, get_cache_stats), batch_processing (BatchConfig, calculate_batch_properties, CSV I/O), compound_analysis (compound families, recommended warming elements), CXRO / NIST / Henke .nff scattering factor data (0.03-30 keV, PCHIP), Interfaces Module API Doc, CLI commands (calc, batch, compare, convert, formula, atomic, bragg, list, completion, install-completion, uninstall-completion), CLI modules (parser, formatting, utils) (+3 more)

### Community 196 - "test_fuzz.py"
Cohesion: 0.47
Nodes (3): _random_call(), test_fuzz_covers_every_input_type(), test_fuzz_valid_calls()

### Community 201 - "test_kk_docs_build.py"
Cohesion: 0.33
Nodes (4): _sphinx(), test_kk_docs_build_adds_no_warnings(), test_kk_docs_doctests_pass(), slow()

### Community 203 - "TestBatchProcessingMemoryManagement"
Cohesion: 0.09
Nodes (3): TestBatchProcessingMemoryManagement, TestCacheManagement, TestMemoryLeakPrevention

### Community 208 - "ADR-005: The Kramers-Kronig Package Is JAX-Only"
Cohesion: 0.22
Nodes (8): ADR-005: The Kramers-Kronig Package Is JAX-Only, Alternatives considered, Consequences, Context, Evidence, Known deviations from the acceptance gates, References, x64 requirement

### Community 212 - "gc"
Cohesion: 0.29
Nodes (3): clean_cache(), enforce_test_isolation(), setup_test_environment()

### Community 214 - "test_kk_cli_end_to_end.py"
Cohesion: 0.14
Nodes (3): _subprocess(), test_cli_writes_the_persistent_compile_cache(), test_version_unaffected()

### Community 219 - "test_gui_kk_worker_end_to_end.py"
Cohesion: 0.16
Nodes (13): _check_table(), _setup(), test_close_during_warm_up_does_not_hang(), test_error_path_shows_dialog_and_stays_responsive(), test_launcher_schedules_the_warm_up(), test_nearedge_run_matches_api(), test_plain_run_matches_api(), test_refined_run_matches_api() (+5 more)

### Community 251 - "calculate_single_material_properties"
Cohesion: 0.07
Nodes (6): TestCLIOutputFormatting, TestSubRefracErrorHandling, TestSubRefracSiliconProperties, TestAnomalousDispersion, TestEnergyInputContract, calculate_single_material_properties()

## Knowledge Gaps
- **123 isolated node(s):** `Henke, Gullikson, Davis 1993 X-ray interactions tables`, `Completion caching system (~/.xraylabtool/cache)`, `Completion for xtool alias`, `Completion features (command, option, path, value)`, `Conda/Mamba install and completion workflow` (+118 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 2245 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **111 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `v0.4.7 (2026-10-23)` connect `pytest` to `pathlib`, `create_scattering_factor_interpolators`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Are the 47 inferred relationships involving `kk_calculate_real()` (e.g. with `Decision` and `Evidence`) actually correct?**
  _`kk_calculate_real()` has 47 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Henke, Gullikson, Davis 1993 X-ray interactions tables`, `Completion caching system (~/.xraylabtool/cache)`, `Completion for xtool alias` to the rest of the system?**
  _123 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `MainWindowProtocol` be split into smaller, more focused modules?**
  _Cohesion score 0.04151404151404151 - nodes in this community are weakly interconnected._
- **Why does `Changelog` connect `pytest` to `fixture`, `.calculate_derived_quantities`, `TestXRayResultTyping`, `test_type_safety_config.py`?**
  _High betweenness centrality (0.054) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `MainWindowProtocol` (e.g. with `KKTabMixin` and `MultiTabMixin`) actually correct?**
  _`MainWindowProtocol` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Should `xraylabtool/utils.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11857707509881422 - nodes in this community are weakly interconnected._