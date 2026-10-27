# Graph Report - pyXRayLabTool  (2026-10-27)

## Corpus Check
- 247 files · ~268,214 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 115 file(s) not represented in the graph (top: .nff 93, .npz 14, .ipynb 2)

## Summary
- 6107 nodes · 14093 edges · 257 communities (155 shown, 102 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 1027 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `abcccc58`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- MainWindowProtocol
- assert_kk_close
- cases.py
- compound_spectrum
- ValidationError
- BashCompletionGenerator
- api.py
- _make_cubic
- constants.py
- JaxPchipInterpolator
- get_atomic_number
- XRayResult
- validate_density
- core.py
- parse_formula
- test_deep_debug_regressions.py
- pytest
- parse_formula
- Toast
- test_performance_benchmarks.py
- kk_real_spectrum
- typing
- test_batch_blocks.py
- TestJaxBackendParity
- typing
- style.py
- TestBasicExportFunctionality
- atomic_cache.py
- device.py
- export_to_csv
- validate_energy_range
- BatchConfig
- CompletionInstaller
- OverlayScrollbarMarginHelper
- test_cli.py
- test_deep_debug_regressions.py
- services.py
- pytest
- TestKKCli
- Toast
- MockXRayResult
- create_scattering_factor_interpolators
- TestEnvironmentProperties
- typing
- installer.py
- MemoryMonitor
- CalculationWorker
- installer.py
- MaterialComparator
- test_utils.py
- Spectrum
- main
- fixture
- .calculate_derived_quantities
- clear_scattering_factor_cache
- test_optics.py
- legacy_install_completion_main
- device.py
- calculate_properties_from_factors
- cmd_calc
- validate_chemical_formula
- argparse
- unit
- load_asf
- optics.py
- scripts/test_docs.py
- test_file_operations.py
- tests/conftest.py
- TestInstallCompletionCommand
- host.py
- MaterialComparator
- get_bulk_atomic_data
- _OpsProxy
- scattering_data.py
- FastAtomicDataProvider
- set_backend
- legacy_install_completion_main
- core_bench.py
- TestXRayResultTyping
- transform.py
- pytest
- run_type_check.py
- FastXRayCalculationEngine
- test_utils_physics.py
- assert_calculation_result_valid
- compute_kk
- PerformanceBenchmark
- jax_numpy
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
- test_advanced_analysis.py
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
- test_tables.py
- TestGoldenAu
- TestGoldenFe2O3
- TestGoldenSiO2
- BashCompletionGenerator
- TestPerformanceBenchmarks
- test_device_refine.py
- numpy
- format_calculation_summary
- TestPerformanceBenchmarks
- TestExtremeConditions
- test_legacy_contracts.py
- TestGoldenSi
- ADR-003: float64 Preservation Policy
- jax_numpy
- record_compiles
- test_cli_edge_cases.py
- cmd_convert
- test_cli.py
- test_oracle.py
- .uninstall
- scattering_data.py
- test_table_interpolator.py
- bucket
- transfer_counts
- energy_to_wavelength
- Getting Started Guide
- kk_calculate_real
- generate_test_energies
- calculators.core (calculate_single_material_properties, calculate_xray_properties)
- XRayLabToolError (base exception)
- test_core_golden.py
- test_batch_screening.py
- kernels.py
- typing_extensions.py
- TestKKCliEndToEnd
- energy_to_wavelength_angstrom
- tests/conftest.py
- kk_calculate_real
- validate_chemical_formula
- get_gpu_info
- test_type_safety_config.py
- _parse_formula
- ADR-001: JAX vs NumPy Computation Backend
- _process_chunks
- test_warm_up.py
- get_elements_for_compound
- JAX Architecture & Design (v0.4.0)
- test_kk_plot.py
- test_cli_jax_discipline.py
- _outcome
- record_compiles
- time
- TestBasicExportFunctionality
- v0.4.0 release (2026-04-07)
- TestUnitConversionConstants
- get_compound_frequency_score
- test_jax_backend_parity.py
- TestBatchProcessing
- Testing Guide
- api.py
- BaseXRayLabToolTest
- test_cli.py
- test_encode_memo.py
- ndarray
- stray_conversions
- get_elements_for_compound
- TestDtypePreservation
- cmd_bragg
- test_core.py
- assert_ulp_close
- cmd_list
- TestCLICompatibility
- test_core_gpu_check.py
- test_midpoint_loop_never_evaluates_more_than_block_per_call
- TestPhysicalRealism
- _pchip_slopes
- AtomicDataProvider
- TestBatchConfig
- TestFormulaValidation
- cli.py
- kk/test_docs.py
- ADR-005: The Kramers-Kronig Package Is JAX-Only
- normalize_intensity
- calculate_multiple_xray_properties
- Data Handling Module API Doc
- TestGoldenAu
- TestRegexCompatibility
- Decision: enforce float64 globally; float32 is a correctness bug
- TestEdgeCaseInputValues
- TestFundamentalConstants
- re
- core_gpu_check.py
- process_single_calculation
- test_core.py
- ResultValidator
- backend/__init__.py
- performance/conftest.py
- gen_scattering_tables.py
- ensure_complex128_array
- TestGuiErrorHandling
- json
- TableFormatter
- TestMemoryLeakPrevention
- ADR-006: The Core Calculators Are JAX-Only
- test_install.py
- test_gui_kk_worker_end_to_end.py
- kk/conftest.py
- integration/__init__.py
- performance/__init__.py
- unit/__init__.py
- BaseIntegrationTest
- BasePerformanceTest
- BaseUnitTest
- test_data_generator
- _subprocess
- test_import_discipline.py
- test_golden_constants.py
- _checked_inputs
- TestDtypePreservation
- _initialize_element_paths
- TestRegexCompatibility
- warnings
- TestUninstallCompletionMain
- pytest_addoption
- pytest_configure
- TestEnvironmentIntegration
- TestBackwardCompatibility
- calculate_single_material_properties
- .error
- TestSubRefracSiliconProperties
- _threaded

## God Nodes (most connected - your core abstractions)
1. `calculate_single_material_properties()` - 130 edges
2. `kk_calculate_real()` - 115 edges
3. `MainWindowProtocol` - 103 edges
4. `parse_formula()` - 97 edges
5. `CompletionInstaller` - 69 edges
6. `kk_real_spectrum()` - 67 edges
7. `calculate_xray_properties()` - 65 edges
8. `ValidationError` - 61 edges
9. `BatchConfig` - 54 edges
10. `compound_spectrum()` - 52 edges

## Surprising Connections (you probably didn't know these)
- `Context` --references--> `set_backend()`  [INFERRED]
  docs/architecture/005-adr-kk-jax-only.md → xraylabtool/backend/__init__.py
- `Known limitations` --references--> `calculate_xray_properties()`  [INFERRED]
  docs/architecture/006-adr-core-jax-only.md → xraylabtool/calculators/core.py
- `Evidence` --references--> `kk_calculate_real()`  [INFERRED]
  docs/architecture/005-adr-kk-jax-only.md → xraylabtool/kk/api.py
- `Decision` --references--> `get_backend()`  [INFERRED]
  docs/architecture/006-adr-core-jax-only.md → xraylabtool/backend/__init__.py
- `Consequences` --references--> `set_backend()`  [INFERRED]
  docs/architecture/005-adr-kk-jax-only.md → xraylabtool/backend/__init__.py

## Import Cycles
- 3-file cycle: `xraylabtool/calculators/host.py -> xraylabtool/calculators/optics.py -> xraylabtool/data_handling/atomic_cache.py -> xraylabtool/calculators/host.py`
- 3-file cycle: `xraylabtool/calculators/__init__.py -> xraylabtool/calculators/derived_quantities.py -> xraylabtool/calculators/host.py -> xraylabtool/calculators/__init__.py`
- 3-file cycle: `xraylabtool/calculators/__init__.py -> xraylabtool/calculators/core.py -> xraylabtool/calculators/host.py -> xraylabtool/calculators/__init__.py`
- 4-file cycle: `xraylabtool/__init__.py -> xraylabtool/calculators/__init__.py -> xraylabtool/calculators/core.py -> xraylabtool/logging_utils.py -> xraylabtool/__init__.py`
- 4-file cycle: `xraylabtool/calculators/__init__.py -> xraylabtool/calculators/optics.py -> xraylabtool/data_handling/atomic_cache.py -> xraylabtool/calculators/host.py -> xraylabtool/calculators/__init__.py`
- 4-file cycle: `xraylabtool/calculators/__init__.py -> xraylabtool/calculators/core.py -> xraylabtool/calculators/derived_quantities.py -> xraylabtool/calculators/host.py -> xraylabtool/calculators/__init__.py`
- 4-file cycle: `xraylabtool/calculators/__init__.py -> xraylabtool/calculators/core.py -> xraylabtool/calculators/cache.py -> xraylabtool/calculators/host.py -> xraylabtool/calculators/__init__.py`
- 4-file cycle: `xraylabtool/calculators/__init__.py -> xraylabtool/calculators/core.py -> xraylabtool/calculators/scattering_data.py -> xraylabtool/calculators/host.py -> xraylabtool/calculators/__init__.py`

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

## Communities (257 total, 102 thin omitted)

### Community 0 - "MainWindowProtocol"
Cohesion: 0.04
Nodes (6): KKTabMixin, MultiTabMixin, MainWindowProtocol, SingleTabMixin, MaterialTable, PlotCanvas

### Community 1 - "assert_kk_close"
Cohesion: 0.08
Nodes (15): _cxro_deviation(), test_30kev_truncation_matches_paper_trend(), test_default_grid_matches_upstream_golden_bit_for_bit_inputs(), test_end_to_end_matches_upstream_golden(), test_tabulated_kk_f1_close_to_cxro_f1(), _load_golden(), _merged(), test_kernel_on_merged_spectrum_matches_golden_f1_points() (+7 more)

### Community 2 - "cases.py"
Cohesion: 0.07
Nodes (29): build_inputs(), compute_family(), _element_data(), element_density(), fam_atomic_provider(), fam_chunk_contracts(), fam_constants_helpers(), fam_crystal_structure() (+21 more)

### Community 3 - "compound_spectrum"
Cohesion: 0.13
Nodes (16): _numpy_assembly(), test_compound_coefficients_equal_the_pre_jax_baseline(), test_compound_coefficients_round_like_a_separate_multiply_and_add(), test_compound_grid_is_union_of_boundaries(), test_compound_spectrum_rejects_bad_input(), test_docstring_examples_run(), test_element_order_does_not_change_the_result(), test_evaluate_f2_edge_conventions() (+8 more)

### Community 4 - "ValidationError"
Cohesion: 0.04
Nodes (18): test_element_without_table_raises_unknown_element(), test_empty_formula_raises_formula_error(), test_parse_errors_are_not_cached_and_repeat(), TestAtomicDataError, TestBatchProcessingError, TestCalculationError, TestDataFileError, TestExceptionHierarchy (+10 more)

### Community 5 - "BashCompletionGenerator"
Cohesion: 0.04
Nodes (7): TestBashCompletionSubstitution, test_cli_aliases_registered(), BashCompletionGenerator, CompletionGenerator, FishCompletionGenerator, PowerShellCompletionGenerator, ZshCompletionGenerator

### Community 6 - "api.py"
Cohesion: 0.05
Nodes (29): critical_angle_degrees(), _isclose(), validate_constants(), _checked_merge(), _compound_capacity(), _compound_segments(), _default_grid_size(), _explicit_energies() (+21 more)

### Community 7 - "_make_cubic"
Cohesion: 0.04
Nodes (15): _compile_counts(), worker(), call(), TestBasicSetupAndInitialization, TestDuplicatedFormulasIndependence, TestEdgeCasesAndErrorHandling, TestH2OProperties, TestInputValidationAndErrorHandling (+7 more)

### Community 8 - "constants.py"
Cohesion: 0.09
Nodes (9): apply_palette_to_item(), apply_palette_to_widget(), current_palette(), legend_kwargs(), _recolor_axis_label(), _restyle_legend(), KKPlot, F1F2Plot (+1 more)

### Community 9 - "JaxPchipInterpolator"
Cohesion: 0.13
Nodes (10): TestJaxPchipInterpolator, _query(), restore_x64(), _scipy(), TestMatchesScipy, TestPrecision, TestQueryHandling, TestShapePreservation (+2 more)

### Community 10 - "get_atomic_number"
Cohesion: 0.05
Nodes (19): TestAtomicDataLookup, clear_lru_caches(), _mock_get_element(), TestCacheResultConsistency, worker(), TestConcurrentGetAtomicNumber, worker(), cache_clearer() (+11 more)

### Community 11 - "XRayResult"
Cohesion: 0.12
Nodes (41): test_merge_spectra_edge_errors_keep_legacy_type_and_text(), test_merge_spectra_nan_coefficients_raise_the_normalisation_error(), _edge_spectrum(), test_asf_input_is_unchanged_and_unknown_type_raises(), test_beta_round_trip_and_density_cancels_at_merge(), test_default_merge_points_are_the_overlap_of_both_ranges(), test_directly_constructed_spectrum_is_validated(), test_docstring_examples_run() (+33 more)

### Community 12 - "validate_density"
Cohesion: 0.09
Nodes (4): TestValidateDensity, attenuation_length_cm(), __getattr__(), validate_density()

### Community 13 - "core.py"
Cohesion: 0.07
Nodes (19): _smart_cache_warming(), _calculate_materials(), calculate_multiple_xray_properties(), _calculate_single_material_xray_properties(), _coerce_energy_array(), _convert_energy_input(), _density_value(), FastXRayCalculationEngine (+11 more)

### Community 14 - "parse_formula"
Cohesion: 0.08
Nodes (6): TestParseFormulaBasic, TestParseFormulaDecimal, TestParseFormulaErrors, TestParseFormulaParentheses, parse_formula(), _parse_formula()

### Community 15 - "test_deep_debug_regressions.py"
Cohesion: 0.05
Nodes (12): test_whitespace_delimited_loader_works_with_current_pandas(), TestAtomicDataCacheIsImmutable, TestDeviceProgramCompilation, TestExports, TestFormulaParsing, TestLazySubmoduleAccess, TestRepeatedMaterials, TestScatteringDataIntegrity (+4 more)

### Community 16 - "pytest"
Cohesion: 0.06
Nodes (27): test_boundary_energies_are_accepted_and_f2_matches_the_table(), test_default_grid_clipped_to_0p03_30_kev_and_logged(), test_docstring_examples_run(), test_empty_default_grid_after_merge_raises(), test_energies_are_copied_not_aliased(), test_energy_one_ulp_outside_the_data_range_raises(), test_energy_out_of_data_range_raises_energy_error(), test_explicit_energy_between_10ev_and_500kev_ok() (+19 more)

### Community 18 - "Toast"
Cohesion: 0.05
Nodes (10): Colors, main(), StyleGuideValidator, StyleViolation, ValidationReport, _banned_module(), forbidden(), test_the_package_has_no_process_or_thread_pool() (+2 more)

### Community 19 - "test_performance_benchmarks.py"
Cohesion: 0.06
Nodes (5): TestBenchmarkComparison, TestPerformanceBenchmarks, TestPerformanceRegression, TestScalabilityBenchmarks, timer()

### Community 20 - "kk_real_spectrum"
Cohesion: 0.14
Nodes (28): _args(), _eq10_corrected(), _f2_poly(), _golden(), _linear_segments(), _quadrature_f1(), test_chunk_invariance(), test_constant_band_closed_form() (+20 more)

### Community 21 - "typing"
Cohesion: 0.09
Nodes (35): _max_midpoint_error(), _refine(), _spy_blocks(), wrap(), test_debug_log_per_pass_and_info_total(), test_docstring_examples_run(), test_exhaustion_flagged_and_logged(), test_heavy_element_segment_grid_against_upstream_emulation() (+27 more)

### Community 22 - "test_batch_blocks.py"
Cohesion: 0.11
Nodes (14): _logged(), _same_bits(), spy(), test_a_bad_energy_fails_every_formula_of_the_multiple_call(), test_a_binding_budget_splits_a_group_and_bounds_the_padded_blocks(), test_a_chunk_with_different_energies_per_item_follows_its_own_energies(), test_a_failing_formula_in_the_middle_is_skipped_and_logged(), test_a_material_is_bit_identical_whatever_its_batch_mates() (+6 more)

### Community 23 - "TestJaxBackendParity"
Cohesion: 0.07
Nodes (15): TestResultFormatting, _format_array_field_single(), _format_as_csv(), _format_as_json(), _format_filtered_table(), _format_material_properties(), _format_multiple_energies(), _format_multiple_energy_section() (+7 more)

### Community 24 - "typing"
Cohesion: 0.07
Nodes (11): test_configure_logging_writes_file(), test_get_logger_child_inherits_config(), build_parser(), main(), _bool_env(), _coerce_int_env(), configure_logging(), get_log_file_path() (+3 more)

### Community 26 - "TestBasicExportFunctionality"
Cohesion: 0.05
Nodes (36): ADR-002: PyQtGraph vs Matplotlib for GUI Plotting, Appendix: Feature Parity Checklist, Consequences, Context, Decision, Migration Details, Negative, Positive (+28 more)

### Community 27 - "atomic_cache.py"
Cohesion: 0.10
Nodes (13): _collect_compound_elements(), _collect_family_compound_elements(), get_atomic_data_fast(), get_bulk_atomic_data_fast(), get_cache_stats(), is_element_preloaded(), _warm_atomic_data(), _warm_bulk_combinations() (+5 more)

### Community 28 - "device.py"
Cohesion: 0.06
Nodes (11): main(), TestCheckGpuAvailability, TestGetDeviceInfo, TestGetGpuInfo, TestGetSystemCudaVersion, check_gpu_availability(), check_plugin_conflicts(), get_device_info() (+3 more)

### Community 29 - "export_to_csv"
Cohesion: 0.09
Nodes (8): TestBasicExportIntegration, export_to_csv(), export_to_json(), _json_numpy_default(), _save_dict_as_csv(), _save_results_as_csv(), _save_results_as_json(), __getattr__()

### Community 30 - "validate_energy_range"
Cohesion: 0.06
Nodes (5): TestValidateCalculationParameters, TestValidateEnergyRange, _get_valid_element_symbols(), validate_calculation_parameters(), validate_energy_range()

### Community 31 - "BatchConfig"
Cohesion: 0.10
Nodes (8): TestBatchProcessorEquivalence, test_max_workers_is_accepted_ignored_and_logged_once_at_debug(), TestBatchInputValidation, TestBatchPartialFailure, TestBatchProcessing, BatchConfig, calculate_batch_properties(), _initialize_progress_bar()

### Community 32 - "CompletionInstaller"
Cohesion: 0.08
Nodes (3): TestSentinelMarkers, CompletionInstaller, test_completion_installation()

### Community 34 - "test_cli.py"
Cohesion: 0.10
Nodes (8): form(), _get_app(), qt_app(), TestDensityValidation, TestEnergyGridValidation, TestFormValues, TestPlotAxisLabelTheming, MaterialInputForm

### Community 35 - "test_deep_debug_regressions.py"
Cohesion: 0.21
Nodes (10): _computation(), _pen_colors(), plot(), qt_app(), test_plot_draws_curves_on_unequal_grids(), test_plot_follows_theme_toggle(), test_plot_hides_missing_curves(), test_plot_log_x_rule() (+2 more)

### Community 36 - "services.py"
Cohesion: 0.07
Nodes (23): _run_smoke(), app(), _fresh(), test_compute_runs_inside_the_worker_guard(), test_core_warm_up_failure_is_logged_not_raised(), test_multiple_materials_on_a_worker_match_direct_calls(), test_start_core_warm_up_runs_one_untracked_worker(), test_the_pool_threads_are_not_the_main_thread() (+15 more)

### Community 37 - "pytest"
Cohesion: 0.17
Nodes (17): test_empty_formula_still_raises_formula_error(), test_element_index(), test_database_rejects_non_integer_atomic_number(), test_evaluate_f2_rejects_complex(), test_evaluate_f2_rejects_nan_segment_edge(), test_kernel_rejects_complex_instead_of_dropping_the_imaginary_part(), test_kk_result_stores_merge_points_as_an_immutable_tuple(), test_merge_spectra_rejects_complex_inputs() (+9 more)

### Community 38 - "TestKKCli"
Cohesion: 0.06
Nodes (18): _parse_csv(), _run(), test_empty_default_grid_after_merge_exits_1(), TestKKCli, _write_edge(), add_atomic_command(), add_batch_command(), add_bragg_command() (+10 more)

### Community 40 - "MockXRayResult"
Cohesion: 0.08
Nodes (6): formula(), MockXRayResult, TestFormatXRayResult, _format_as_csv(), _format_as_table(), format_xray_result()

### Community 41 - "create_scattering_factor_interpolators"
Cohesion: 0.06
Nodes (40): Calculators Module (API docs), xraylabtool.calculators.cache, calculate_xray_properties, xraylabtool.calculators.core, Critical angle theta_c = sqrt(2 delta), xraylabtool.calculators.derived_quantities, xraylabtool.calculators.kernels, Legacy CamelCase accessors (DeprecationWarning) (+32 more)

### Community 43 - "typing"
Cohesion: 0.16
Nodes (9): count(), cmd_batch(), _convert_result_to_dict(), _parse_batch_data(), _process_batch_materials(), _save_batch_results(), _validate_batch_input(), create_batch_progress_tracker() (+1 more)

### Community 44 - "installer.py"
Cohesion: 0.09
Nodes (5): TestCompletionInstallToTempDir, TestCompletionScriptGeneration, CompletionManager, get_global_options(), get_xraylabtool_commands()

### Community 45 - "MemoryMonitor"
Cohesion: 0.08
Nodes (6): TestBatchProcessingMemoryManagement, TestCacheManagement, TestMemoryMonitor, test_force_gc_keeps_the_resident_tables_and_the_encoding_memo(), MemoryMonitor, _validate_batch_inputs()

### Community 46 - "CalculationWorker"
Cohesion: 0.10
Nodes (10): assert_values_close(), expect_no_warnings(), expect_warnings(), get_system_info(), log_system_info(), skip_if_insufficient_cpu(), skip_if_insufficient_memory(), stability_test() (+2 more)

### Community 47 - "installer.py"
Cohesion: 0.09
Nodes (6): apply_pyqtgraph_theme(), apply_styles(), apply_theme(), ColorPalette, get_qss(), ThemeManager

### Community 48 - "MaterialComparator"
Cohesion: 0.08
Nodes (8): _check_element(), cold_cache(), test_si_o_au_interpolators_are_independent(), TestPchipAu, TestPchipO, TestPchipSi, test_insufficient_data_keeps_the_value_error(), create_scattering_factor_interpolators()

### Community 49 - "test_utils.py"
Cohesion: 0.07
Nodes (8): BasePerformanceTest, TestArrayOptimization, TestBulkAtomicDataOptimization, TestDeprecationWarningOptimization, TestElementPathOptimization, TestInterpolatorCaching, TestNumericalStabilityOptimizations, TestPerformanceRegression

### Community 50 - "Spectrum"
Cohesion: 0.04
Nodes (47): Decision, Decision, _evaluate_f2_device(), test_an_extra_upload_inside_the_program_call_is_rejected(), sneaky(), test_thread_counts_ignore_other_threads(), test_kk_device_and_shapes_re_export_the_promoted_names(), test_runtime_and_shapes_import_has_no_side_effect() (+39 more)

### Community 51 - "main"
Cohesion: 0.08
Nodes (6): TestCLIIntegration, TestMainFunction, cmd_completion(), cmd_install_completion(), cmd_uninstall_completion(), main()

### Community 52 - "fixture"
Cohesion: 0.07
Nodes (38): Platform and Python version compatibility matrix, Contributing Guide, Contributing atomic data (citation, range, precision, format), Code standards (PEP 8, 88 cols, NumPy docstrings, type hints), Performance requirements (<0.1 ms single, >100k calc/s batch, >90% cache hit), Review process (automated checks, reviewer, maintainer, squash merge), Development workflow (branching, ruff, mypy, pytest, make verify), Performance and Optimization guide (+30 more)

### Community 53 - ".calculate_derived_quantities"
Cohesion: 0.07
Nodes (39): widgets.kk_plot (Kramers-Kronig Plot), main_window.kk_tab, kk sub-package (Kramers-Kronig, kkcalc port), Kramers-Kronig (kk) API docs, xraylabtool.kk.api (high-level API), AsfDatabase, compound_spectrum, xraylabtool.kk.database (+31 more)

### Community 54 - "clear_scattering_factor_cache"
Cohesion: 0.07
Nodes (3): TestGoldenDerivedQuantities, TestGoldenScatteringFactors, TestGoldenSiO2

### Community 55 - "test_optics.py"
Cohesion: 0.08
Nodes (34): _inputs(), run_program(), run_tail(), test_max_block_bounds_the_materials_per_block(), test_repeated_elements_are_aggregated_in_first_seen_order(), test_host_tables_are_rejected(), test_wrapper_runs_under_an_outer_explicit_transfer_guard(), _assert_golden() (+26 more)

### Community 57 - "device.py"
Cohesion: 0.19
Nodes (21): app(), _small_run(), test_energies_outside_cxro_range_reach_the_service(), test_energy_range_controls_reach_the_service(), spy(), test_exhausted_refinement_shows_a_warning_toast(), test_export_csv_columns(), test_export_csv_io_error_is_reported() (+13 more)

### Community 58 - "calculate_properties_from_factors"
Cohesion: 0.15
Nodes (11): test_non_finite_status_raises_the_legacy_value_error(), test_errors_match_the_task0_catalogue(), test_matches_molecular_properties_bit_for_bit(), test_helper_delta_beta_match_closed_form(), test_helper_rejects_complex_input(), test_helper_rejects_mismatched_nonfinite_or_bad_density(), test_helper_reproduces_lookup_path(), test_negative_f1_preserved_critical_angle_zero() (+3 more)

### Community 60 - "validate_chemical_formula"
Cohesion: 0.06
Nodes (4): TestEnvironmentInfo, TestEnvironmentType, EnvironmentInfo, EnvironmentType

### Community 62 - "unit"
Cohesion: 0.10
Nodes (6): _calc(), _clear_cache(), TestComplexFormula, TestElectronCountGolden, TestExactCapturedValues, TestMolecularWeightGolden

### Community 63 - "load_asf"
Cohesion: 0.07
Nodes (28): _asf_f2(), test_asf_grid_spans_10ev_to_500kev(), test_asf_has_92_elements(), test_cached_tables_are_read_only(), test_im_coeffs_shape(), test_load_asf_is_cached(), test_nff_divergent_elements(), _assemble() (+20 more)

### Community 64 - "optics.py"
Cohesion: 0.11
Nodes (17): Amendments to the design spec, recorded here, one(), derive(), _derived(), _derived_program(), _field_program(), _optical_program(), one() (+9 more)

### Community 65 - "scripts/test_docs.py"
Cohesion: 0.14
Nodes (14): check_accessibility(), check_documentation_coverage(), Colors, main(), print_section(), print_status(), run_style_checks(), test_doctests() (+6 more)

### Community 66 - "test_file_operations.py"
Cohesion: 0.15
Nodes (17): test_load_absolute_path(), test_load_csv_file(), test_load_csv_space_separated_fallback(), test_load_empty_nff_raises_data_file_error(), test_load_malformed_nff_raises_data_file_error(), test_load_nff_energy_column_monotone(), test_load_nff_values_are_finite(), test_load_nonexistent_file_raises_file_not_found() (+9 more)

### Community 67 - "tests/conftest.py"
Cohesion: 0.13
Nodes (10): qt_app(), _run_worker(), test_worker_does_not_emit_finished_on_exception(), test_worker_emits_error_on_exception(), bad_fn(), test_worker_emits_result_on_success(), test_worker_injects_progress_cb_when_accepted(), test_worker_passes_kwargs() (+2 more)

### Community 69 - "host.py"
Cohesion: 0.09
Nodes (24): test_a_material_larger_than_the_budget_runs_as_blocks_of_exactly_one(), test_wrapper_budget_is_one_upload_and_one_fetch_per_call(), test_padding_does_not_change_a_bit_of_the_pchip_programs(), thread_transfer_counts(), bucket_pow2(), warm_up(), broadcast_shape(), check_molecular_weight() (+16 more)

### Community 70 - "MaterialComparator"
Cohesion: 0.10
Nodes (3): test_comparator_fallback_names_the_first_failing_formula(), ComparisonResult, MaterialComparator

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

### Community 77 - "core_bench.py"
Cohesion: 0.12
Nodes (15): _cell(), _cli_times(), _cli_times_warm_cache(), cold_env(), _fmt(), _importtime(), machine_state(), main() (+7 more)

### Community 78 - "TestXRayResultTyping"
Cohesion: 0.10
Nodes (22): GUI (API docs), xraylabtool.gui.logging_filters, xraylabtool.gui.main_window, widgets.material_form, widgets.material_table, main_window.multi_tab, widgets.plot_canvas, widgets.scrollbar_helper (+14 more)

### Community 79 - "transform.py"
Cohesion: 0.06
Nodes (15): main(), truncation_deficit(), _finite(), main(), _rel_corr(), main(), _run(), probe_energies() (+7 more)

### Community 80 - "pytest"
Cohesion: 0.07
Nodes (30): CalculationError, EnergyError, FormulaError, ValidationError, XRayLabToolError, Changelog, Batch API fixes (ordering, timeouts, process-pool fallback, formula@density keys), CLI aliases xtool / xlt and GUI aliases xtool-gui / xlt-gui (+22 more)

### Community 81 - "run_type_check.py"
Cohesion: 0.11
Nodes (20): Analytic per-interval integration (Eqs. 3-4), Biggs and Lighthill high-energy cross-sections (to 500 keV), Cauchy principal value integral, Conventional KK numerical methods (FFT, Euler-Maclaurin, Gaussian, TANH), Zero-to-infinity convergence conditions on polynomial coefficients, Cross et al. difference-spectrum KK method, Double-interval integration across x=E singularity (Eq. 5), Closed-form f1(E) for piecewise Laurent f2 (Eq. 7, Eq. 10) (+12 more)

### Community 83 - "test_utils_physics.py"
Cohesion: 0.10
Nodes (27): _fraction_moving_average(), _mine(), _offset_signal(), _signal(), _smooth_reference(), test_a_constant_series_stays_constant_after_smoothing(), test_a_new_window_is_a_new_program_and_a_repeat_is_not(), test_arrays_larger_than_one_element_still_fail_in_float() (+19 more)

### Community 85 - "compute_kk"
Cohesion: 0.11
Nodes (16): test_compute_kk_default_grid_cxro_matches_result_energies(), test_compute_kk_forwards_energies_kev(), test_compute_kk_forwards_refine_tolerance(), test_compute_kk_merge_points_follow_energy_unit(), test_compute_kk_rejects_unknown_energy_unit(), test_compute_kk_service_matches_api_with_nearedge(), test_compute_kk_service_matches_api_without_nearedge(), test_cxro_comparison_empty_outside_tabulated_range() (+8 more)

### Community 86 - "PerformanceBenchmark"
Cohesion: 0.15
Nodes (5): TestFormulaCommand, cmd_formula(), _output_formula_results(), _print_formula_results(), _process_formula()

### Community 87 - "jax_numpy"
Cohesion: 0.10
Nodes (13): angle_from_q(), bragg_angle(), d_spacing_cubic(), d_spacing_orthorhombic(), d_spacing_tetragonal(), energy_to_wavelength(), linear_background(), normalize_intensity() (+5 more)

### Community 88 - "kk_gpu_check.py"
Cohesion: 0.10
Nodes (22): check_additivity(), check_bench(), check_f2_si_edge(), check_oracle(), check_padding(), check_transfers(), _f2_numpy(), guarded() (+14 more)

### Community 91 - "kk_calculate_real"
Cohesion: 0.08
Nodes (28): _colliding_pair(), _run(), test_accelerator_block_rule_gives_the_same_result(), test_backend_field_is_jax(), test_chunk_size_is_upper_bound(), test_cold_table_upload_is_counted_once_and_separately(), test_conversion_probe_sees_a_stray_conversion(), test_default_grid_is_exact_ev_over_1000() (+20 more)

### Community 92 - "XRayLabTool Documentation (index)"
Cohesion: 0.08
Nodes (29): Runtime NumPy/JAX backend switching (set_backend), Modular package architecture (backend, calculators, data_handling, interfaces, io, validation, gui), Kramers-Kronig Guide, Near-edge input types (nexafs, beta, asf), kkcalc attribution (zlib/libpng, Watts and Lauk), kk module limits (no tail, conditioning, pole skipping, wrong energy-unit hazard), Near-edge file format, Watts 2014, Opt. Express 22(19), 23628 (piecewise Laurent polynomial KK) (+21 more)

### Community 93 - "typing_extensions.py"
Cohesion: 0.14
Nodes (7): get_optimal_chunk_size(), is_complex_array(), is_energy_array(), is_real_array(), optimize_array_memory_layout(), validate_complex_array(), validate_energy_array()

### Community 94 - "refine.py"
Cohesion: 0.09
Nodes (21): _accumulator_need(), _finalize(), _first_pass(), _fit_active(), _fit_columns(), _grow_accumulator(), _Knobs, _midpoint_f1() (+13 more)

### Community 95 - "xraylabtool/utils.py"
Cohesion: 0.13
Nodes (18): Interfaces Module API Doc, CLI commands (calc, batch, compare, convert, formula, atomic, bragg, list, completion, install-completion, uninstall-completion), CLI error handling messages, CLI modules (parser, formatting, utils), completion_v2 virtual-environment-centric shell completion, CLI output formats (table, csv, json), Supported shells (bash, zsh, fish, PowerShell), Validation Module API Doc (+10 more)

### Community 97 - "compound_analysis.py"
Cohesion: 0.15
Nodes (3): TestCalculationSpeedBenchmarks, calculation_worker(), TestPerformanceRegressionDetection

### Community 99 - "test_advanced_analysis.py"
Cohesion: 0.11
Nodes (3): TestBasicAnalysis, compare_materials(), find_absorption_edges()

### Community 100 - "ValidationError"
Cohesion: 0.09
Nodes (11): _outcome(), _result(), test_kk_result_arrays_are_read_only_copies(), test_kk_result_rejects_inconsistent_state(), test_refine_outcome_arrays_are_read_only_copies(), test_refine_outcome_rejects_inconsistent_state(), TestValidationError, ValidationError (+3 more)

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
Cohesion: 0.15
Nodes (7): setup(), analyze_mypy_output(), check_core_modules(), check_type_definitions(), main(), run_mypy_check(), validate_cache_performance()

### Community 108 - "test_device_merge.py"
Cohesion: 0.13
Nodes (33): _assert_matches_reference(), _explicit_scale(), _poly(), _reference(), _run(), _status_of(), test_adjacent_float_knots_use_lower_boundary(), test_decimal_scale_of_the_near_edge_values_matches_scale_one() (+25 more)

### Community 109 - "_make_cubic"
Cohesion: 0.10
Nodes (4): _make_cubic(), _phase(), TestCrystalStructureAPI, TestStructureFactorPhysics

### Community 110 - "test_tables.py"
Cohesion: 0.13
Nodes (8): gen(), _hermite(), shipped(), tables_mod(), test_check_flag_fails_on_drift(), test_check_flag_passes_on_the_shipped_file(), test_padded_rows_evaluate_like_the_unpadded_pchip(), test_padding_knots_strictly_increase_and_pad_values()

### Community 111 - "TestGoldenAu"
Cohesion: 0.09
Nodes (10): TestGoldenAttenuationLength, test_standalone_primitives_match_golden_edges(), calculate_attenuation_length(), calculate_scattering_length_density(), _map(), _scattering_length_density(), __getattr__(), attenuation_length_from_beta() (+2 more)

### Community 112 - "TestGoldenFe2O3"
Cohesion: 0.17
Nodes (3): TestGoldenCriticalAngle, test_helpers_keep_their_standalone_edges(), calculate_critical_angle()

### Community 113 - "TestGoldenSiO2"
Cohesion: 0.10
Nodes (7): TestEnergyParsing, cmd_kk(), _fail(), _format_output(), _parse_energies(), _parse_merge(), parse_energy_string()

### Community 115 - "TestPerformanceBenchmarks"
Cohesion: 0.18
Nodes (14): Attenuation length vs energy plot for SiO2 (8-12 keV, 50 points, rising curve ~13 to ~44 cm x0.001), Compute primary action button, Energy to Wavelength converter widget (keV and Angstrom, E to lambda buttons), Footer buttons: Log Path and Light Mode (theme toggle), Inline validation feedback (Formula valid, Composition Si:1 O:2, Density valid, grid summary), Material input panel (formula, density, energy grid, log-spaced option, Compute button), Material summary table (formula, MW, density, electron density, total electrons), Plot toolbar (property selector, Log X/Log Y, Save Plot PNG, Export CSV) (+6 more)

### Community 116 - "test_device_refine.py"
Cohesion: 0.12
Nodes (31): _device_run(), _host_reference(), _logged(), _narrow_last_segment_table(), _refine(), _same_result(), _spacing_grid(), test_accumulator_growth_path_is_bit_identical() (+23 more)

### Community 119 - "TestPerformanceBenchmarks"
Cohesion: 0.14
Nodes (5): TestPerformanceBenchmarks, energy_sweep(), complex_formula(), multi_calculation(), single_calculation()

### Community 121 - "test_legacy_contracts.py"
Cohesion: 0.11
Nodes (23): _derived(), _exact_reference(), _shapes(), _si(), test_all_scalar_derived_quantities_are_float_and_numpy_scalars(), test_all_zero_formula_is_skipped_by_the_collection_calls(), test_all_zero_formula_raises_zero_division(), test_all_zero_formula_with_a_numpy_density_keeps_the_non_finite_failure() (+15 more)

### Community 124 - "jax_numpy"
Cohesion: 0.08
Nodes (34): padded(), _case(), _eval(), test_kernel_without_x64_raises_the_documented_error_without_a_warning(), test_kk_f1_grad(), loss(), test_kk_f1_jit_vmap(), test_kk_f1_keeps_input_shape_and_rejects_bad_block() (+26 more)

### Community 125 - "record_compiles"
Cohesion: 0.14
Nodes (9): _Handler, kk_programs(), numerical_programs(), record_compiles(), test_two_hundred_formulas_compile_at_most_one_program_and_transfer_once_per_block(), test_refined_sio2_compiles_few_top_level_programs(), test_energy_length_bucket_recompiles_only_the_evaluation(), test_refinement_knobs_do_not_recompile_assembly() (+1 more)

### Community 126 - "test_cli_edge_cases.py"
Cohesion: 0.05
Nodes (6): TestBatchRowResilience, TestCLIErrorHandling, TestCLIFileOperations, TestCLIInputValidation, TestCLIPerformanceAndMemory, TestFieldNameValidation

### Community 127 - "cmd_convert"
Cohesion: 0.32
Nodes (8): _checkout(), _eval_points(), _finite(), _git(), main(), _poly(), _save(), _transform_case()

### Community 129 - "test_oracle.py"
Cohesion: 0.13
Nodes (12): f1_reference(), _gauss_legendre(), _h(), _pv_integral(), _quadrature(), _f2_mismatch(), oracle_energies(), _reference() (+4 more)

### Community 131 - "scattering_data.py"
Cohesion: 0.12
Nodes (5): AtomicScatteringFactor, load_scattering_factor_data(), _ones_like_program(), ScatteringData, _structure_factor_program()

### Community 132 - "test_table_interpolator.py"
Cohesion: 0.08
Nodes (23): test_standalone_scattering_factors_keep_a_tiny_result_for_a_large_molecular_weight(), _budget(), _material(), test_atomic_provider_fills_complex128_from_the_tables_in_one_program_call(), test_atomic_provider_keeps_the_shape_of_its_energies(), test_atomic_provider_unknown_element_keeps_its_error(), test_derived_quantities_are_one_program_call_and_clamped(), test_derived_quantities_keep_the_scalar_electron_density_expression() (+15 more)

### Community 133 - "bucket"
Cohesion: 0.12
Nodes (20): test_the_planner_never_returns_a_block_of_zero(), _budget(), platform(), test_block_for_examples(), test_block_for_never_exceeds_the_request(), test_bucket_is_multiple_of_8_monotone_and_bounded(), test_bucket_ladder_head(), test_bucket_pow2() (+12 more)

### Community 134 - "transfer_counts"
Cohesion: 0.16
Nodes (13): test_a_block_is_one_upload_and_one_fetch(), test_one_program_call_is_one_upload_and_one_fetch(), test_clear_tables_forces_reload(), test_load_tables_single_upload_from_threads(), test_load_tables_uploads_once(), test_a_scalar_conversion_is_one_upload_and_one_fetch(), test_a_signal_helper_is_one_upload_and_one_fetch(), _passes() (+5 more)

### Community 135 - "energy_to_wavelength"
Cohesion: 0.05
Nodes (22): fam_utils_crystal(), test_parse_formula_is_cached_but_public_result_is_fresh_lists(), TestCrystallographicCalculations, TestUnitConversions, _aggregate_element_counts(), angle_from_q(), bragg_angle(), _check_lattice() (+14 more)

### Community 136 - "Getting Started Guide"
Cohesion: 0.08
Nodes (27): Getting Started Guide, xraylabtool batch CSV processing, calculate_single_material_properties (usage), calculate_xray_properties (usage), gpu_cuda12 / gpu_cuda13 extras, PySide6 GUI (Single, Multiple, Kramers-Kronig tabs), Installation (pip, conda, uv, GPU extras), Performance tips (preloaded elements, batching, energy arrays) (+19 more)

### Community 137 - "kk_calculate_real"
Cohesion: 0.09
Nodes (18): Adaptive refinement (refine_tolerance), xraylabtool kk CLI command, kk_calculate_real, load_nearedge, Low-level kk functions (compound_spectrum, kk_real_spectrum, evaluate_f2, to_asf_scale, merge_spectra, refine_spectrum), Merge points (affine match of near-edge data to tabulated f2), NearEdgeDataError, NearEdgeSpectrum (+10 more)

### Community 138 - "generate_test_energies"
Cohesion: 0.11
Nodes (3): generate_test_energies(), generate_test_materials(), ParametrizedTestMixin

### Community 140 - "XRayLabToolError (base exception)"
Cohesion: 0.22
Nodes (5): memory_test(), performance_test(), decorator(), requires_memory(), decorator()

### Community 141 - "test_core_golden.py"
Cohesion: 0.16
Nodes (13): _backend_label(), check_family(), compare_arrays(), _compare_text(), _meta(), _npz(), _structure_problem(), test_active_backend_label() (+5 more)

### Community 142 - "test_batch_screening.py"
Cohesion: 0.17
Nodes (11): test_progress_counts_a_non_finite_item_once(), _chunk(), test_a_complex_energy_does_not_share_a_group_with_real_bytes(), test_a_reshaped_energy_grid_is_rejected_for_every_item(), test_a_sequence_density_in_the_middle_of_a_chunk_keeps_the_rows_aligned(), test_a_sequence_density_is_rejected_up_front_by_calculate_xray_properties(), test_a_sequence_density_is_skipped_by_the_multiple_call(), test_a_sequence_density_warns_and_maps_to_none_in_the_batch() (+3 more)

### Community 144 - "typing_extensions.py"
Cohesion: 0.22
Nodes (11): Available Scripts, Example Output:, Integration:, Maintenance Notes, Output:, Requirements:, `run_type_check.py` - Type Checking, Scripts Directory (+3 more)

### Community 145 - "TestKKCliEndToEnd"
Cohesion: 0.15
Nodes (7): _assert_matches(), _in_process(), _subprocess(), _table(), test_cli_writes_the_persistent_compile_cache(), test_version_unaffected(), TestKKCliEndToEnd

### Community 146 - "energy_to_wavelength_angstrom"
Cohesion: 0.19
Nodes (4): Scope and what follows, TestEnergyToWavelength, energy_to_wavelength_angstrom(), wavelength_angstrom_to_energy()

### Community 147 - "tests/conftest.py"
Cohesion: 0.05
Nodes (27): assert_memory_usage_reasonable(), assert_performance_within_threshold(), batch_config_default(), batch_config_performance(), batch_config_small(), benchmark_baseline(), clean_cache(), cpu_count() (+19 more)

### Community 150 - "get_gpu_info"
Cohesion: 0.20
Nodes (9): ADR-001: JAX vs NumPy Computation Backend, Appendix: Functions that MUST NOT be JIT-compiled, Appendix: Functions to JIT-Compile (Priority Order), Consequences, Context, Decision, Negative, Positive (+1 more)

### Community 151 - "test_type_safety_config.py"
Cohesion: 0.14
Nodes (16): backend sub-package (NumPy/JAX abstraction), JAX Architecture & Design, JAX array immutability and lazy evaluation, Automatic differentiation (grad), Debugging and testing JAX code (disable jit, jax.debug.print, JIT-vs-eager parity), Future JAX features for v0.5+ (autodiff, vmap, distributed), GPU acceleration and device placement, JAX library (+8 more)

### Community 154 - "_process_chunks"
Cohesion: 0.11
Nodes (10): test_chunk_size_bounds_the_blocks_and_progress_counts_every_item(), test_the_collisions_also_hold_through_process_chunks(), chunk_iterator(), _energies_key(), _filter_dataframe_fields(), _get_energy_point_data(), _prepare_result_data(), _process_chunks() (+2 more)

### Community 155 - "test_warm_up.py"
Cohesion: 0.15
Nodes (11): cache_config(), test_cache_dir_is_os_appropriate(), test_cache_helper_enables_and_sets_threshold(), test_cache_helper_keeps_user_threshold(), test_cache_helper_respects_opt_out_and_existing_config(), test_cli_does_not_call_warm_up(), test_library_import_does_not_enable_cache(), test_warm_up_exported_lazily() (+3 more)

### Community 156 - "get_elements_for_compound"
Cohesion: 0.17
Nodes (5): TestFindSimilarCompounds, TestGetElementsForCompound, _collect_similar_compound_elements(), find_similar_compounds(), get_elements_for_compound()

### Community 157 - "JAX Architecture & Design (v0.4.0)"
Cohesion: 0.14
Nodes (5): TestGetCompoundFamily, TestGetRecommendedElementsForWarming, _classify_family_by_elements(), get_compound_family(), get_recommended_elements_for_warming()

### Community 158 - "test_kk_plot.py"
Cohesion: 0.21
Nodes (3): TestParseChemicalFormula, TestDelegatingWrappers, parse_chemical_formula()

### Community 159 - "test_cli_jax_discipline.py"
Cohesion: 0.25
Nodes (12): _batch_argv(), _bragg_argv(), _cache_dir(), _calc_argv(), _compare_argv(), _convert_argv(), _kk_argv(), _run_cli() (+4 more)

### Community 161 - "record_compiles"
Cohesion: 0.16
Nodes (9): _mine(), test_a_second_tables_object_hits_the_same_compiled_program(), test_another_bucket_does_compile(), test_energy_count_inside_the_same_bucket_compiles_nothing(), test_first_call_compiles_then_same_buckets_compile_nothing(), _warm(), clear_tables(), element_index() (+1 more)

### Community 162 - "time"
Cohesion: 0.27
Nodes (9): _call_worker(), call(), _gate(), _kernel_worker(), main(), _recorder(), _run_worker(), _timed_ms() (+1 more)

### Community 168 - "TestBatchProcessing"
Cohesion: 0.25
Nodes (7): Files, Maintenance, `performance_baselines.json`, `sample_materials.json`, Test Data Directory, `test_energies.json`, Usage in Tests

### Community 169 - "Testing Guide"
Cohesion: 0.29
Nodes (8): 30 keV tabulation cut (dashed vertical line), C (carbon) curve, Figure: f1 overestimated by truncating f2 integration at 30 keV, Fe (iron) curve, Kramers-Kronig f2-to-f1 integral truncated to tabulated range, Pb (lead) curve, f1 truncation error (f1_trunc - f1_full) vs photon energy, log-log, Result: truncation error grows with Z (C ~1e-4, Fe ~0.1, Pb ~2.5 e/atom), flat at low energy, rises sharply near 30 keV

### Community 172 - "test_cli.py"
Cohesion: 0.10
Nodes (4): BaseIntegrationTest, TestAtomicCommand, TestBatchCommand, cmd_atomic()

### Community 173 - "test_encode_memo.py"
Cohesion: 0.17
Nodes (8): test_a_failing_formula_raises_every_time(), test_a_formula_is_encoded_once_and_the_arrays_are_read_only(), test_clearing_the_scattering_cache_clears_the_memo_and_refills_the_element_cache(), test_density_screen_accepts_positive_numbers_of_every_kind(), test_density_screen_rejects_non_positive_numbers_of_every_kind(), get_cached_elements(), is_element_cached(), _encode_material()

### Community 174 - "ndarray"
Cohesion: 0.11
Nodes (4): arrays_close(), assert_arrays_close(), create_batch_test_data(), TestDataFactory

### Community 175 - "stray_conversions"
Cohesion: 0.18
Nodes (7): _outside_device_helpers(), stray_conversions(), patch_method(), wrapper(), patch_numpy(), wrapper(), test_kernel_evaluates_only_blocks_holding_real_energies()

### Community 180 - "assert_ulp_close"
Cohesion: 0.22
Nodes (8): assert_ulp_close(), _step(), test_arrays_report_the_worst_entry(), test_infinities_must_be_equal_and_never_close_to_finite(), test_nan_must_match_positions(), test_shape_mismatch_and_empty(), test_within_and_beyond_the_bound(), test_zero_expected_requires_zero()

### Community 183 - "test_core_gpu_check.py"
Cohesion: 0.25
Nodes (5): _fake(), gate(), _p12(), test_p12_fails_when_the_timed_warm_calls_compile(), test_p12_passes_when_nothing_compiles_after_warm_up()

### Community 184 - "test_midpoint_loop_never_evaluates_more_than_block_per_call"
Cohesion: 0.20
Nodes (6): dlog(), order(), pass_trace(), wrap(), spy(), test_midpoint_loop_never_evaluates_more_than_block_per_call()

### Community 190 - "cli.py"
Cohesion: 0.08
Nodes (7): TestInstallCompletionMain, completion_main(), create_completion_parser(), handle_completion_command(), install_completion_main(), show_completion_info(), handle_new_completion_command()

### Community 191 - "kk/test_docs.py"
Cohesion: 0.23
Nodes (7): _public_objects(), test_docs_cite_the_paper_and_upstream_licence(), test_docs_pages_toctree_linked(), test_docs_state_the_eq10_erratum(), test_every_public_kk_name_is_documented(), test_public_docstrings_have_examples_and_references(), test_transform_docstrings_cite_paper_equations()

### Community 192 - "ADR-005: The Kramers-Kronig Package Is JAX-Only"
Cohesion: 0.22
Nodes (8): ADR-005: The Kramers-Kronig Package Is JAX-Only, Alternatives considered, Consequences, Context, Evidence, Known deviations from the acceptance gates, References, x64 requirement

### Community 193 - "normalize_intensity"
Cohesion: 0.09
Nodes (16): Exemptions (host code by design), fam_utils_signal(), test_multi_column_background_compiles_once_per_shape_bucket(), test_one_dimensional_only_helpers_reject_other_ranks(), TestDataProcessing, background_subtraction(), find_peaks(), normalize_intensity() (+8 more)

### Community 195 - "Data Handling Module API Doc"
Cohesion: 0.47
Nodes (5): Data Handling Module API Doc, atomic_cache (92-element preload, warm_up_cache, get_cache_stats), batch_processing (BatchConfig, calculate_batch_properties, CSV I/O), compound_analysis (compound families, recommended warming elements), CXRO / NIST / Henke .nff scattering factor data (0.03-30 keV, PCHIP)

### Community 201 - "re"
Cohesion: 0.17
Nodes (7): collect(), main(), _search(), _sphinx(), test_kk_docs_build_adds_no_warnings(), test_kk_docs_doctests_pass(), test_pyproject_package_data_lists_kk_files()

### Community 202 - "core_gpu_check.py"
Cohesion: 0.31
Nodes (8): check_bench(), check_goldens(), check_padding(), check_transfers(), guarded(), main(), record(), _same()

### Community 203 - "process_single_calculation"
Cohesion: 0.15
Nodes (4): no_progress_config(), TestProcessSingleCalculation, _prepare_energies_array(), process_single_calculation()

### Community 208 - "backend/__init__.py"
Cohesion: 0.12
Nodes (7): _git(), main(), _versions(), test_set_backend_calls_do_not_affect_kk(), TestBackendStubs, get_backend(), set_backend()

### Community 210 - "gen_scattering_tables.py"
Cohesion: 0.22
Nodes (6): build_tables(), _drift(), main(), _row(), _symbols_by_atomic_number(), _unique_files()

### Community 213 - "json"
Cohesion: 0.10
Nodes (8): build(), main(), cmd_compare(), _parse_materials(), _result_to_dict(), _write_report(), _write_table(), _format_as_json()

### Community 216 - "ADR-006: The Core Calculators Are JAX-Only"
Cohesion: 0.29
Nodes (6): Accepted: single-call latency relative to NumPy, ADR-006: The Core Calculators Are JAX-Only, Consequences, Context, Gate evidence, Known limitations

### Community 219 - "test_gui_kk_worker_end_to_end.py"
Cohesion: 0.25
Nodes (10): _check_table(), _setup(), test_close_during_warm_up_does_not_hang(), test_error_path_shows_dialog_and_stays_responsive(), test_launcher_schedules_the_warm_up(), test_nearedge_run_matches_api(), test_plain_run_matches_api(), test_refined_run_matches_api() (+2 more)

### Community 237 - "_subprocess"
Cohesion: 0.29
Nodes (4): test_warm_up_covers_the_first_compute_click(), _subprocess(), test_cold_concurrent_first_calls_upload_the_tables_once(), test_refined_run_program_budget()

### Community 238 - "test_import_discipline.py"
Cohesion: 0.43
Nodes (3): _leaked_modules(), test_no_jax_with_gpu_tools(), test_no_jax_without_gpu_tools()

### Community 244 - "warnings"
Cohesion: 0.36
Nodes (4): _outcome(), _same(), test_every_departure_is_a_real_difference(), test_matches_the_legacy_outcome_or_the_documented_departure()

### Community 251 - "calculate_single_material_properties"
Cohesion: 0.04
Nodes (20): _clear_cache(), cold_cache(), TestCLICompatibility, TestCLIOutputFormatting, TestSubRefracErrorHandling, _budget(), test_calculation_fills_the_element_cache_that_get_cached_elements_reports(), test_clear_cache_drops_the_resident_tables_and_the_next_call_reloads_them() (+12 more)

### Community 258 - "_threaded"
Cohesion: 0.50
Nodes (3): _threaded(), work(), fn()

## Knowledge Gaps
- **123 isolated node(s):** `Henke, Gullikson, Davis 1993 X-ray interactions tables`, `Completion caching system (~/.xraylabtool/cache)`, `Completion for xtool alias`, `Completion features (command, option, path, value)`, `Conda/Mamba install and completion workflow` (+118 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 2482 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **102 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `v0.4.7 (2026-10-23)` connect `pytest` to `create_scattering_factor_interpolators`, `re`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `calculate_single_material_properties()` (e.g. with `fam_single_errors()` and `fam_single_input_types()`) actually correct?**
  _`calculate_single_material_properties()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Henke, Gullikson, Davis 1993 X-ray interactions tables`, `Completion caching system (~/.xraylabtool/cache)`, `Completion for xtool alias` to the rest of the system?**
  _123 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `MainWindowProtocol` be split into smaller, more focused modules?**
  _Cohesion score 0.036657681940700806 - nodes in this community are weakly interconnected._
- **Why does `calculate_single_material_properties()` connect `calculate_single_material_properties` to `TestSubRefracSiliconProperties`, `cases.py`, `ValidationError`, `_make_cubic`, `core.py`, `test_batch_screening.py`, `test_deep_debug_regressions.py`, `test_performance_benchmarks.py`, `test_batch_blocks.py`, `typing`, `style.py`, `export_to_csv`, `BatchConfig`, `TestBasicExportFunctionality`, `services.py`, `test_jax_backend_parity.py`, `typing`, `MemoryMonitor`, `clear_scattering_factor_cache`, `calculate_properties_from_factors`, `TestEdgeCaseInputValues`, `core_gpu_check.py`, `process_single_calculation`, `set_backend`, `core_bench.py`, `compute_kk`, `kk_gpu_check.py`, `compound_analysis.py`, `test_advanced_analysis.py`, `TestGoldenAu`, `TestPerformanceBenchmarks`, `test_legacy_contracts.py`, `test_cli_edge_cases.py`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Are the 47 inferred relationships involving `kk_calculate_real()` (e.g. with `Decision` and `Evidence`) actually correct?**
  _`kk_calculate_real()` has 47 INFERRED edges - model-reasoned connections that need verification._
- **Should `assert_kk_close` be split into smaller, more focused modules?**
  _Cohesion score 0.08250355618776671 - nodes in this community are weakly interconnected._