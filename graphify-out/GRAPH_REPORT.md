# Graph Report - pyXRayLabTool  (2026-10-27)

## Corpus Check
- 245 files · ~267,622 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 115 file(s) not represented in the graph (top: .nff 93, .npz 14, .ipynb 2)

## Summary
- 6094 nodes · 14063 edges · 267 communities (153 shown, 114 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 1027 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `2e873201`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- MainWindowProtocol
- xraylabtool/utils.py
- cases.py
- compound_spectrum
- ValidationError
- shells.py
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
- numpy
- typing
- clear_scattering_factor_cache
- TestJaxBackendParity
- test_utilities.py
- style.py
- TestBasicExportFunctionality
- data_handling/__init__.py
- device.py
- export_to_csv
- validate_energy_range
- BatchConfig
- CompletionInstaller
- OverlayScrollbarMarginHelper
- test_cli.py
- test_deep_debug_regressions.py
- services.py
- TestBasicSetupAndInitialization
- TestKKCli
- single_tab.py
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
- tests/conftest.py
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
- create_parser
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
- runtime.py
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
- test_legacy_contracts.py
- TestGoldenSi
- ADR-003: float64 Preservation Policy
- _kk_f1_blocks
- test_kk_cli_end_to_end.py
- pathlib
- cmd_convert
- test_cli.py
- test_oracle.py
- .uninstall
- scattering_data.py
- test_table_interpolator.py
- bucket
- atomic_cache.py
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
- test_cli_jax_discipline.py
- _outcome
- record_compiles
- time
- Any
- v0.4.0 release (2026-04-07)
- TestUnitConversionConstants
- test_compound_analysis.py
- argparse
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
- FishCompletionGenerator
- .sum
- assert_ulp_close
- cmd_list
- TestCLICompatibility
- EnergyError
- TestCLIErrorHandling
- TestPhysicalRealism
- _pchip_slopes
- AtomicDataProvider
- BashCompletionGenerator
- BatchProcessingError
- cli.py
- kk/test_docs.py
- TestCLIFileOperations
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
- gc
- compare.py
- test_cli_writes_the_persistent_compile_cache
- Array
- ADR-006: The Core Calculators Are JAX-Only
- subprocess
- test_gui_kk_worker_end_to_end.py
- test_no_process_pools.py
- integration/__init__.py
- performance/__init__.py
- unit/__init__.py
- BaseIntegrationTest
- BasePerformanceTest
- BaseUnitTest
- test_data_generator
- benchmark_baseline
- ensure_x64
- sample_batch_data
- TestCLICompatibility
- test_materials
- skip_if_low_memory
- test_energies_large
- warnings
- TestUninstallCompletionMain
- large_batch_data
- material_and_density
- TestEnvironmentIntegration
- TestCLIPerformanceAndMemory
- TestBackwardCompatibility
- calculate_single_material_properties
- TestPerformanceRegression
- .error
- assert_arrays_close
- TestSubRefracSiliconProperties
- test_threaded_calls_agree
- _user_warnings
- clean_cache_around
- batch_config_default
- memory_monitor
- memory_tracker
- performance_timer
- sample_calculation_result

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
- `Known limitations` --references--> `calculate_xray_properties()`  [INFERRED]
  docs/architecture/006-adr-core-jax-only.md → xraylabtool/calculators/core.py
- `Evidence` --references--> `kk_calculate_real()`  [INFERRED]
  docs/architecture/005-adr-kk-jax-only.md → xraylabtool/kk/api.py
- `Decision` --references--> `get_backend()`  [INFERRED]
  docs/architecture/006-adr-core-jax-only.md → xraylabtool/backend/__init__.py
- `Context` --references--> `set_backend()`  [INFERRED]
  docs/architecture/005-adr-kk-jax-only.md → xraylabtool/backend/__init__.py
- `Decision` --references--> `_pchip_slopes()`  [INFERRED]
  docs/architecture/006-adr-core-jax-only.md → xraylabtool/backend/interpolation.py

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

## Communities (267 total, 114 thin omitted)

### Community 0 - "MainWindowProtocol"
Cohesion: 0.04
Nodes (6): _kk_energy_grid(), KKTabMixin, MultiTabMixin, MainWindowProtocol, SingleTabMixin, MaterialTable

### Community 1 - "xraylabtool/utils.py"
Cohesion: 0.08
Nodes (15): test_element_without_table_raises_unknown_element(), test_empty_formula_raises_formula_error(), test_parse_errors_are_not_cached_and_repeat(), test_parse_formula_is_cached_but_public_result_is_fresh_lists(), test_repeated_elements_are_aggregated_in_first_seen_order(), TestFormulaError, FormulaError, _aggregate_element_counts() (+7 more)

### Community 2 - "cases.py"
Cohesion: 0.08
Nodes (27): build_inputs(), compute_family(), _element_data(), element_density(), fam_atomic_provider(), fam_chunk_contracts(), fam_crystal_structure(), fam_derived_quantities() (+19 more)

### Community 3 - "compound_spectrum"
Cohesion: 0.10
Nodes (18): test_matches_upstream_improve_accuracy_golden(), test_merge_wrapper_raises_on_undersized_capacity(), _numpy_assembly(), test_compound_coefficients_equal_the_pre_jax_baseline(), test_compound_coefficients_round_like_a_separate_multiply_and_add(), test_compound_grid_is_union_of_boundaries(), test_compound_spectrum_rejects_bad_input(), test_docstring_examples_run() (+10 more)

### Community 4 - "ValidationError"
Cohesion: 0.05
Nodes (11): TestAtomicDataError, TestCalculationError, TestDataFileError, TestExceptionHierarchy, TestExceptionUsage, TestUnknownElementError, AtomicDataError, CalculationError (+3 more)

### Community 5 - "shells.py"
Cohesion: 0.09
Nodes (3): test_cli_aliases_registered(), CompletionGenerator, PowerShellCompletionGenerator

### Community 6 - "api.py"
Cohesion: 0.05
Nodes (28): _checked_merge(), _compound_capacity(), _compound_segments(), _explicit_energies(), _host_inputs(), _merge_points_ev(), _near_edge_call(), _NearEdgeCall (+20 more)

### Community 7 - "_make_cubic"
Cohesion: 0.03
Nodes (27): TestDuplicatedFormulasIndependence, TestEdgeCasesAndErrorHandling, TestH2OProperties, TestInputValidationAndErrorHandling, TestPropertyConsistency, TestSiO2Properties, _logged(), _same_bits() (+19 more)

### Community 8 - "constants.py"
Cohesion: 0.08
Nodes (9): apply_palette_to_item(), apply_palette_to_widget(), current_palette(), legend_kwargs(), _recolor_axis_label(), _restyle_legend(), PlotCanvas, F1F2Plot (+1 more)

### Community 9 - "JaxPchipInterpolator"
Cohesion: 0.12
Nodes (11): TestJaxPchipInterpolator, _query(), restore_x64(), _scipy(), TestMatchesScipy, TestPrecision, TestQueryHandling, TestShapePreservation (+3 more)

### Community 10 - "get_atomic_number"
Cohesion: 0.05
Nodes (19): TestAtomicDataLookup, clear_lru_caches(), _mock_get_element(), TestCacheResultConsistency, worker(), TestConcurrentGetAtomicNumber, worker(), cache_clearer() (+11 more)

### Community 11 - "XRayResult"
Cohesion: 0.12
Nodes (41): test_host_scale_conversion_and_merge_of_tiny_data_match_scale_one(), test_merge_spectra_nan_coefficients_raise_the_normalisation_error(), _edge_spectrum(), test_asf_input_is_unchanged_and_unknown_type_raises(), test_beta_round_trip_and_density_cancels_at_merge(), test_default_merge_points_are_the_overlap_of_both_ranges(), test_directly_constructed_spectrum_is_validated(), test_docstring_examples_run() (+33 more)

### Community 12 - "validate_density"
Cohesion: 0.10
Nodes (3): TestValidateDensity, __getattr__(), validate_density()

### Community 13 - "core.py"
Cohesion: 0.06
Nodes (20): TestGoldenParity, TestScalarEnergyFallback, _smart_cache_warming(), _calculate_materials(), calculate_multiple_xray_properties(), _calculate_single_material_xray_properties(), _coerce_energy_array(), _convert_energy_input() (+12 more)

### Community 14 - "parse_formula"
Cohesion: 0.06
Nodes (9): TestFormulaParsing, TestDelegatingWrappers, TestParseFormulaBasic, TestParseFormulaDecimal, TestParseFormulaErrors, TestParseFormulaParentheses, TestRegexCompatibility, parse_formula() (+1 more)

### Community 15 - "test_deep_debug_regressions.py"
Cohesion: 0.05
Nodes (13): test_whitespace_delimited_loader_works_with_current_pandas(), TestAnomalousDispersion, TestAtomicDataCacheIsImmutable, TestDeviceProgramCompilation, TestExports, TestGuiErrorHandling, task(), TestLazySubmoduleAccess (+5 more)

### Community 16 - "pytest"
Cohesion: 0.07
Nodes (46): test_non_finite_status_raises_the_legacy_value_error(), test_arrays_larger_than_one_element_still_fail_in_float(), test_backend_is_jax_and_chunk_size_does_not_matter(), test_boundary_energies_are_accepted_and_f2_matches_the_table(), test_default_grid_clipped_to_0p03_30_kev_and_logged(), test_docstring_examples_run(), test_empty_default_grid_after_merge_raises(), test_energies_are_copied_not_aliased() (+38 more)

### Community 18 - "Toast"
Cohesion: 0.07
Nodes (4): main(), StyleGuideValidator, StyleViolation, ValidationReport

### Community 19 - "test_performance_benchmarks.py"
Cohesion: 0.06
Nodes (5): TestBenchmarkComparison, TestPerformanceBenchmarks, TestPerformanceRegression, TestScalabilityBenchmarks, timer()

### Community 20 - "numpy"
Cohesion: 0.06
Nodes (60): check_additivity(), _load_golden(), _merged(), test_kernel_on_merged_spectrum_matches_golden_f1_points(), test_merged_spectrum_matches_upstream_golden(), test_nearedge_result_matches_upstream_golden(), test_matches_upstream_kk_pp_golden(), assert_kk_close() (+52 more)

### Community 21 - "typing"
Cohesion: 0.09
Nodes (35): _max_midpoint_error(), _refine(), _spy_blocks(), wrap(), test_debug_log_per_pass_and_info_total(), test_docstring_examples_run(), test_exhaustion_flagged_and_logged(), test_heavy_element_segment_grid_against_upstream_emulation() (+27 more)

### Community 22 - "clear_scattering_factor_cache"
Cohesion: 0.14
Nodes (3): _clear_cache(), cold_cache(), clear_scattering_factor_cache()

### Community 23 - "TestJaxBackendParity"
Cohesion: 0.07
Nodes (15): TestResultFormatting, _format_array_field_single(), _format_as_csv(), _format_as_json(), _format_filtered_table(), _format_material_properties(), _format_multiple_energies(), _format_multiple_energy_section() (+7 more)

### Community 24 - "test_utilities.py"
Cohesion: 0.08
Nodes (14): setup(), kk_log(), x64_restored(), test_configure_logging_writes_file(), test_get_logger_child_inherits_config(), build_parser(), main(), _bool_env() (+6 more)

### Community 26 - "TestBasicExportFunctionality"
Cohesion: 0.05
Nodes (36): ADR-002: PyQtGraph vs Matplotlib for GUI Plotting, Appendix: Feature Parity Checklist, Consequences, Context, Decision, Migration Details, Negative, Positive (+28 more)

### Community 27 - "data_handling/__init__.py"
Cohesion: 0.17
Nodes (6): get_atomic_data_fast(), get_bulk_atomic_data_fast(), get_cache_stats(), is_element_preloaded(), warm_up_cache(), __getattr__()

### Community 28 - "device.py"
Cohesion: 0.06
Nodes (10): TestCheckGpuAvailability, TestGetDeviceInfo, TestGetGpuInfo, TestGetSystemCudaVersion, check_gpu_availability(), check_plugin_conflicts(), get_device_info(), get_gpu_info() (+2 more)

### Community 29 - "export_to_csv"
Cohesion: 0.07
Nodes (5): TestBasicExportIntegration, TestBasicExportFunctionality, export_to_csv(), export_to_json(), __getattr__()

### Community 30 - "validate_energy_range"
Cohesion: 0.07
Nodes (4): TestValidateCalculationParameters, TestValidateEnergyRange, validate_calculation_parameters(), validate_energy_range()

### Community 31 - "BatchConfig"
Cohesion: 0.07
Nodes (9): TestBatchProcessorEquivalence, TestBatchConfig, test_max_workers_is_accepted_ignored_and_logged_once_at_debug(), no_progress_config(), TestBatchInputValidation, TestBatchPartialFailure, TestBatchProcessing, BatchConfig (+1 more)

### Community 32 - "CompletionInstaller"
Cohesion: 0.08
Nodes (3): TestSentinelMarkers, CompletionInstaller, test_completion_installation()

### Community 34 - "test_cli.py"
Cohesion: 0.07
Nodes (9): form(), _get_app(), qt_app(), TestDensityValidation, TestEnergyGridValidation, TestFormulaValidation, TestFormValues, TestPlotAxisLabelTheming (+1 more)

### Community 35 - "test_deep_debug_regressions.py"
Cohesion: 0.15
Nodes (11): _computation(), _pen_colors(), plot(), qt_app(), test_plot_draws_curves_on_unequal_grids(), test_plot_follows_theme_toggle(), test_plot_hides_missing_curves(), test_plot_log_x_rule() (+3 more)

### Community 36 - "services.py"
Cohesion: 0.06
Nodes (26): _run_smoke(), app(), _fresh(), test_compute_runs_inside_the_worker_guard(), test_core_warm_up_failure_is_logged_not_raised(), test_multiple_materials_on_a_worker_match_direct_calls(), test_start_core_warm_up_runs_one_untracked_worker(), test_the_pool_threads_are_not_the_main_thread() (+18 more)

### Community 38 - "TestKKCli"
Cohesion: 0.11
Nodes (5): _parse_csv(), _run(), test_empty_default_grid_after_merge_exits_1(), TestKKCli, _write_edge()

### Community 40 - "MockXRayResult"
Cohesion: 0.05
Nodes (9): formula(), MockXRayResult, TestFormatCalculationSummary, TestFormatXRayResult, _format_as_csv(), _format_as_json(), _format_as_table(), format_calculation_summary() (+1 more)

### Community 41 - "create_scattering_factor_interpolators"
Cohesion: 0.06
Nodes (40): Calculators Module (API docs), xraylabtool.calculators.cache, calculate_xray_properties, xraylabtool.calculators.core, Critical angle theta_c = sqrt(2 delta), xraylabtool.calculators.derived_quantities, xraylabtool.calculators.kernels, Legacy CamelCase accessors (DeprecationWarning) (+32 more)

### Community 43 - "typing"
Cohesion: 0.12
Nodes (7): cmd_batch(), _convert_result_to_dict(), _parse_batch_data(), _process_batch_materials(), _save_batch_results(), _validate_batch_input(), create_batch_progress_tracker()

### Community 44 - "installer.py"
Cohesion: 0.08
Nodes (5): TestCompletionInstallToTempDir, TestCompletionScriptGeneration, CompletionManager, get_global_options(), get_xraylabtool_commands()

### Community 45 - "MemoryMonitor"
Cohesion: 0.07
Nodes (5): TestBatchProcessingMemoryManagement, TestCacheManagement, TestMemoryLeakPrevention, TestMemoryMonitor, MemoryMonitor

### Community 46 - "CalculationWorker"
Cohesion: 0.10
Nodes (10): assert_values_close(), expect_no_warnings(), expect_warnings(), get_system_info(), log_system_info(), skip_if_insufficient_cpu(), skip_if_insufficient_memory(), stability_test() (+2 more)

### Community 47 - "installer.py"
Cohesion: 0.10
Nodes (6): apply_pyqtgraph_theme(), apply_styles(), apply_theme(), ColorPalette, get_qss(), ThemeManager

### Community 48 - "MaterialComparator"
Cohesion: 0.06
Nodes (14): _check_element(), cold_cache(), test_si_o_au_interpolators_are_independent(), TestPchipAu, TestPchipO, TestPchipSi, test_standalone_scattering_factors_keep_a_tiny_result_for_a_large_molecular_weight(), test_element_without_data_keeps_file_not_found() (+6 more)

### Community 49 - "test_utils.py"
Cohesion: 0.09
Nodes (7): BasePerformanceTest, TestArrayOptimization, TestBulkAtomicDataOptimization, TestDeprecationWarningOptimization, TestElementPathOptimization, TestInterpolatorCaching, TestNumericalStabilityOptimizations

### Community 50 - "Spectrum"
Cohesion: 0.05
Nodes (42): test_device_loop_runs_under_transfer_guard(), _assemble(), _poly(), _reference_f2(), test_capacity_independent(), test_compound_capacity_is_bucketed_segment_sum(), test_count_within_capacity(), test_device_matches_baseline() (+34 more)

### Community 51 - "main"
Cohesion: 0.07
Nodes (9): TestCLIIntegration, TestMainFunction, cmd_completion(), cmd_install_completion(), cmd_uninstall_completion(), main(), install_completion_main(), show_completion_info() (+1 more)

### Community 52 - "fixture"
Cohesion: 0.07
Nodes (38): Platform and Python version compatibility matrix, Contributing Guide, Contributing atomic data (citation, range, precision, format), Code standards (PEP 8, 88 cols, NumPy docstrings, type hints), Performance requirements (<0.1 ms single, >100k calc/s batch, >90% cache hit), Review process (automated checks, reviewer, maintainer, squash merge), Development workflow (branching, ruff, mypy, pytest, make verify), Performance and Optimization guide (+30 more)

### Community 53 - ".calculate_derived_quantities"
Cohesion: 0.07
Nodes (39): widgets.kk_plot (Kramers-Kronig Plot), main_window.kk_tab, kk sub-package (Kramers-Kronig, kkcalc port), Kramers-Kronig (kk) API docs, xraylabtool.kk.api (high-level API), AsfDatabase, compound_spectrum, xraylabtool.kk.database (+31 more)

### Community 55 - "test_optics.py"
Cohesion: 0.07
Nodes (34): run_program(), run_tail(), test_errors_match_the_task0_catalogue(), test_matches_molecular_properties_bit_for_bit(), test_host_tables_are_rejected(), test_wrapper_runs_under_an_outer_explicit_transfer_guard(), _assert_golden(), _assert_material() (+26 more)

### Community 56 - "legacy_install_completion_main"
Cohesion: 0.10
Nodes (3): TestInstallCompletionMain, TestLegacyInstallCompletionMain, legacy_install_completion_main()

### Community 57 - "device.py"
Cohesion: 0.19
Nodes (21): app(), _small_run(), test_energies_outside_cxro_range_reach_the_service(), test_energy_range_controls_reach_the_service(), spy(), test_exhausted_refinement_shows_a_warning_toast(), test_export_csv_columns(), test_export_csv_io_error_is_reported() (+13 more)

### Community 58 - "calculate_properties_from_factors"
Cohesion: 0.23
Nodes (7): test_helper_delta_beta_match_closed_form(), test_helper_rejects_complex_input(), test_helper_rejects_mismatched_nonfinite_or_bad_density(), test_helper_reproduces_lookup_path(), test_negative_f1_preserved_critical_angle_zero(), test_negative_f2_kept_and_attenuation_clamped(), calculate_properties_from_factors()

### Community 62 - "unit"
Cohesion: 0.09
Nodes (7): _calc(), _clear_cache(), TestComplexFormula, TestElectronCountGolden, TestExactCapturedValues, TestMolecularWeightGolden, _calculate_molecular_properties()

### Community 63 - "load_asf"
Cohesion: 0.06
Nodes (20): _cxro_deviation(), test_30kev_truncation_matches_paper_trend(), test_default_grid_matches_upstream_golden_bit_for_bit_inputs(), test_end_to_end_matches_upstream_golden(), test_tabulated_kk_f1_close_to_cxro_f1(), _asf_f2(), test_asf_grid_spans_10ev_to_500kev(), test_asf_has_92_elements() (+12 more)

### Community 64 - "optics.py"
Cohesion: 0.10
Nodes (23): Amendments to the design spec, recorded here, Decision, test_standalone_primitives_match_golden_edges(), critical_angle_from_delta(), derive(), _derived(), _derived_program(), _field_program() (+15 more)

### Community 65 - "scripts/test_docs.py"
Cohesion: 0.17
Nodes (11): check_accessibility(), check_documentation_coverage(), Colors, main(), print_section(), print_status(), run_style_checks(), test_doctests() (+3 more)

### Community 66 - "test_file_operations.py"
Cohesion: 0.10
Nodes (21): test_load_absolute_path(), test_load_csv_file(), test_load_csv_space_separated_fallback(), test_load_empty_nff_raises_data_file_error(), test_load_malformed_nff_raises_data_file_error(), test_load_nff_energy_column_monotone(), test_load_nff_values_are_finite(), test_load_nonexistent_file_raises_file_not_found() (+13 more)

### Community 67 - "tests/conftest.py"
Cohesion: 0.12
Nodes (12): work(), qt_app(), _run_worker(), test_worker_does_not_emit_finished_on_exception(), test_worker_emits_error_on_exception(), bad_fn(), test_worker_emits_result_on_success(), test_worker_injects_progress_cb_when_accepted() (+4 more)

### Community 69 - "host.py"
Cohesion: 0.11
Nodes (21): test_a_block_is_one_upload_and_one_fetch(), test_max_block_bounds_the_materials_per_block(), test_padding_does_not_change_a_bit_of_the_pchip_programs(), broadcast_shape(), check_molecular_weight(), device_tables(), execute(), _jitted() (+13 more)

### Community 70 - "MaterialComparator"
Cohesion: 0.05
Nodes (6): test_comparator_fallback_names_the_first_failing_formula(), TestBasicAnalysis, ComparisonResult, MaterialComparator, compare_materials(), find_absorption_edges()

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
Cohesion: 0.10
Nodes (18): _cell(), _cli_times(), _cli_times_warm_cache(), cold_env(), _compile_counts(), _fmt(), _importtime(), machine_state() (+10 more)

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
Cohesion: 0.12
Nodes (24): _fraction_moving_average(), _mine(), _offset_signal(), _signal(), _smooth_reference(), test_a_new_window_is_a_new_program_and_a_repeat_is_not(), test_a_signal_helper_is_one_upload_and_one_fetch(), test_background_subtraction_matches_the_numpy_formulas() (+16 more)

### Community 84 - "tests/conftest.py"
Cohesion: 0.10
Nodes (8): assert_calculation_result_valid(), assert_memory_usage_reasonable(), assert_performance_within_threshold(), batch_config_performance(), batch_config_small(), pytest_addoption(), pytest_collection_modifyitems(), pytest_configure()

### Community 85 - "compute_kk"
Cohesion: 0.26
Nodes (12): test_compute_kk_default_grid_cxro_matches_result_energies(), test_compute_kk_forwards_energies_kev(), test_compute_kk_forwards_refine_tolerance(), test_compute_kk_merge_points_follow_energy_unit(), test_compute_kk_rejects_unknown_energy_unit(), test_compute_kk_service_matches_api_with_nearedge(), test_compute_kk_service_matches_api_without_nearedge(), test_cxro_comparison_empty_outside_tabulated_range() (+4 more)

### Community 86 - "PerformanceBenchmark"
Cohesion: 0.15
Nodes (5): TestFormulaCommand, cmd_formula(), _output_formula_results(), _print_formula_results(), _process_formula()

### Community 87 - "jax_numpy"
Cohesion: 0.11
Nodes (14): _pad_edge(), angle_from_q(), bragg_angle(), d_spacing_cubic(), d_spacing_orthorhombic(), d_spacing_tetragonal(), energy_to_wavelength(), linear_background() (+6 more)

### Community 88 - "kk_gpu_check.py"
Cohesion: 0.14
Nodes (18): check_bench(), check_f2_si_edge(), check_oracle(), check_padding(), check_transfers(), _evaluate_f2_device(), _f2_numpy(), guarded() (+10 more)

### Community 89 - "BaseUnitTest"
Cohesion: 0.11
Nodes (3): BaseUnitTest, TestDocumentationPatterns, TestStyleCompliance

### Community 91 - "kk_calculate_real"
Cohesion: 0.06
Nodes (31): test_warm_up_covers_the_first_compute_click(), _colliding_pair(), _passes(), _run(), _subprocess(), test_accelerator_block_rule_gives_the_same_result(), test_backend_field_is_jax(), test_chunk_size_is_upper_bound() (+23 more)

### Community 92 - "XRayLabTool Documentation (index)"
Cohesion: 0.08
Nodes (29): Runtime NumPy/JAX backend switching (set_backend), Modular package architecture (backend, calculators, data_handling, interfaces, io, validation, gui), Kramers-Kronig Guide, Near-edge input types (nexafs, beta, asf), kkcalc attribution (zlib/libpng, Watts and Lauk), kk module limits (no tail, conditioning, pole skipping, wrong energy-unit hazard), Near-edge file format, Watts 2014, Opt. Express 22(19), 23628 (piecewise Laurent polynomial KK) (+21 more)

### Community 93 - "typing_extensions.py"
Cohesion: 0.14
Nodes (7): get_optimal_chunk_size(), is_complex_array(), is_energy_array(), is_real_array(), optimize_array_memory_layout(), validate_complex_array(), validate_energy_array()

### Community 94 - "refine.py"
Cohesion: 0.13
Nodes (15): Decision, _accumulator_need(), _finalize(), _first_pass(), _fit_active(), _fit_columns(), _grow_accumulator(), _Knobs (+7 more)

### Community 95 - "xraylabtool/utils.py"
Cohesion: 0.24
Nodes (11): Validation Module API Doc, Validation best practices (validate first, catch specific exceptions), CalculationError, EnergyError, FormulaError, validate_calculation_parameters, validate_chemical_formula, validate_density (+3 more)

### Community 97 - "compound_analysis.py"
Cohesion: 0.15
Nodes (3): TestCalculationSpeedBenchmarks, calculation_worker(), TestPerformanceRegressionDetection

### Community 99 - "create_parser"
Cohesion: 0.11
Nodes (13): add_atomic_command(), add_batch_command(), add_bragg_command(), add_calc_command(), add_compare_command(), add_completion_command(), add_convert_command(), add_formula_command() (+5 more)

### Community 100 - "ValidationError"
Cohesion: 0.09
Nodes (13): _outcome(), _result(), test_kk_result_arrays_are_read_only_copies(), test_kk_result_rejects_inconsistent_state(), test_refine_outcome_arrays_are_read_only_copies(), test_refine_outcome_rejects_inconsistent_state(), TestValidationError, ValidationError (+5 more)

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
Cohesion: 0.08
Nodes (38): _assert_matches_reference(), _explicit_scale(), _poly(), _reference(), _run(), _status_of(), test_adjacent_float_knots_use_lower_boundary(), test_decimal_scale_of_the_near_edge_values_matches_scale_one() (+30 more)

### Community 109 - "_make_cubic"
Cohesion: 0.08
Nodes (5): _make_cubic(), _phase(), TestCrystalStructureAPI, TestStructureFactorPhysics, CrystalStructure

### Community 110 - "runtime.py"
Cohesion: 0.05
Nodes (28): test_two_hundred_formulas_compile_at_most_one_program_and_transfer_once_per_block(), test_an_extra_upload_inside_the_program_call_is_rejected(), sneaky(), test_thread_counts_ignore_other_threads(), test_wrapper_budget_is_one_upload_and_one_fetch_per_call(), test_kk_device_and_shapes_re_export_the_promoted_names(), test_runtime_and_shapes_import_has_no_side_effect(), test_transfer_counters_live_in_runtime_only() (+20 more)

### Community 111 - "TestGoldenAu"
Cohesion: 0.11
Nodes (8): TestGoldenAttenuationLength, test_scattering_length_density_of_scalars_returns_floats(), calculate_attenuation_length(), calculate_scattering_length_density(), _map(), _scattering_length_density(), __getattr__(), attenuation_length_from_beta()

### Community 112 - "TestGoldenFe2O3"
Cohesion: 0.17
Nodes (3): TestGoldenCriticalAngle, test_helpers_keep_their_standalone_edges(), calculate_critical_angle()

### Community 115 - "TestPerformanceBenchmarks"
Cohesion: 0.18
Nodes (14): Attenuation length vs energy plot for SiO2 (8-12 keV, 50 points, rising curve ~13 to ~44 cm x0.001), Compute primary action button, Energy to Wavelength converter widget (keV and Angstrom, E to lambda buttons), Footer buttons: Log Path and Light Mode (theme toggle), Inline validation feedback (Formula valid, Composition Si:1 O:2, Density valid, grid summary), Material input panel (formula, density, energy grid, log-spaced option, Compute button), Material summary table (formula, MW, density, electron density, total electrons), Plot toolbar (property selector, Log X/Log Y, Save Plot PNG, Export CSV) (+6 more)

### Community 116 - "test_device_refine.py"
Cohesion: 0.08
Nodes (40): kk_programs(), numerical_programs(), _device_run(), dlog(), _host_reference(), _logged(), _narrow_last_segment_table(), order() (+32 more)

### Community 119 - "TestPerformanceBenchmarks"
Cohesion: 0.14
Nodes (5): TestPerformanceBenchmarks, energy_sweep(), complex_formula(), multi_calculation(), single_calculation()

### Community 121 - "test_legacy_contracts.py"
Cohesion: 0.12
Nodes (22): _derived(), _exact_reference(), _shapes(), _si(), test_all_scalar_derived_quantities_are_float_and_numpy_scalars(), test_all_zero_formula_raises_zero_division(), test_all_zero_formula_with_a_numpy_density_keeps_the_non_finite_failure(), test_derived_field_shapes_follow_their_own_operands() (+14 more)

### Community 124 - "_kk_f1_blocks"
Cohesion: 0.13
Nodes (12): padded(), _midpoint_f1(), trips(), body(), _trip_sizes(), _crop(), _f64(), _kk_f1_blocks() (+4 more)

### Community 125 - "test_kk_cli_end_to_end.py"
Cohesion: 0.21
Nodes (6): enable_persistent_cache(), cmd_kk(), _fail(), _format_output(), _parse_energies(), _parse_merge()

### Community 126 - "pathlib"
Cohesion: 0.06
Nodes (8): TestBatchRowResilience, TestCLIInputValidation, TestFieldNameValidation, _leaked_modules(), test_no_jax_with_gpu_tools(), test_no_jax_without_gpu_tools(), TestEnvironmentType, EnvironmentType

### Community 127 - "cmd_convert"
Cohesion: 0.32
Nodes (8): _checkout(), _eval_points(), _finite(), _git(), main(), _poly(), _save(), _transform_case()

### Community 129 - "test_oracle.py"
Cohesion: 0.13
Nodes (12): f1_reference(), _gauss_legendre(), _h(), _pv_integral(), _quadrature(), _f2_mismatch(), oracle_energies(), _reference() (+4 more)

### Community 131 - "scattering_data.py"
Cohesion: 0.07
Nodes (6): TestAtomicScatteringFactor, AtomicScatteringFactor, _initialize_element_paths(), _ones_like_program(), ScatteringData, _structure_factor_program()

### Community 132 - "test_table_interpolator.py"
Cohesion: 0.10
Nodes (16): _budget(), _material(), test_atomic_provider_fills_complex128_from_the_tables_in_one_program_call(), test_atomic_provider_keeps_the_shape_of_its_energies(), test_atomic_provider_unknown_element_keeps_its_error(), test_derived_quantities_are_one_program_call_and_clamped(), test_derived_quantities_keep_the_scalar_electron_density_expression(), test_interpolator_accepts_scalars_host_and_device_arrays() (+8 more)

### Community 133 - "bucket"
Cohesion: 0.09
Nodes (24): test_the_planner_never_returns_a_block_of_zero(), _budget(), platform(), test_block_for_examples(), test_block_for_never_exceeds_the_request(), test_bucket_is_multiple_of_8_monotone_and_bounded(), test_bucket_ladder_head(), test_bucket_pow2() (+16 more)

### Community 134 - "atomic_cache.py"
Cohesion: 0.17
Nodes (7): _collect_compound_elements(), _collect_family_compound_elements(), _collect_similar_compound_elements(), _warm_atomic_data(), _warm_bulk_combinations(), warm_cache_for_compounds(), _warm_interpolators()

### Community 135 - "energy_to_wavelength"
Cohesion: 0.07
Nodes (15): fam_utils_crystal(), TestCrystallographicCalculations, TestUnitConversions, angle_from_q(), bragg_angle(), _check_lattice(), d_spacing_cubic(), d_spacing_orthorhombic() (+7 more)

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

### Community 141 - "test_core_golden.py"
Cohesion: 0.16
Nodes (14): _backend_label(), check_family(), compare_arrays(), _compare_text(), _inputs(), _meta(), _npz(), _structure_problem() (+6 more)

### Community 142 - "test_batch_screening.py"
Cohesion: 0.12
Nodes (15): test_progress_counts_a_non_finite_item_once(), _chunk(), test_a_complex_energy_does_not_share_a_group_with_real_bytes(), test_a_material_larger_than_the_budget_runs_as_blocks_of_exactly_one(), test_a_reshaped_energy_grid_is_rejected_for_every_item(), test_a_sequence_density_in_the_middle_of_a_chunk_keeps_the_rows_aligned(), test_a_sequence_density_is_rejected_up_front_by_calculate_xray_properties(), test_a_sequence_density_is_skipped_by_the_multiple_call() (+7 more)

### Community 144 - "typing_extensions.py"
Cohesion: 0.22
Nodes (11): Available Scripts, Example Output:, Integration:, Maintenance Notes, Output:, Requirements:, `run_type_check.py` - Type Checking, Scripts Directory (+3 more)

### Community 145 - "TestKKCliEndToEnd"
Cohesion: 0.18
Nodes (6): _assert_matches(), _in_process(), _subprocess(), _table(), test_version_unaffected(), TestKKCliEndToEnd

### Community 146 - "xraylabtool/__init__.py"
Cohesion: 0.08
Nodes (11): Scope and what follows, fam_constants_helpers(), TestDtypePreservation, TestEnergyToWavelength, TestMathematicalConstants, attenuation_length_cm(), critical_angle_degrees(), energy_to_wavelength_angstrom() (+3 more)

### Community 147 - "fixture"
Cohesion: 0.12
Nodes (7): cpu_count(), energy_arrays(), mock_low_memory(), performance_thresholds(), temp_directory(), test_energies_medium(), test_energies_small()

### Community 149 - "validate_chemical_formula"
Cohesion: 0.10
Nodes (3): TestValidateChemicalFormula, _get_valid_element_symbols(), validate_chemical_formula()

### Community 150 - "get_gpu_info"
Cohesion: 0.20
Nodes (9): ADR-001: JAX vs NumPy Computation Backend, Appendix: Functions that MUST NOT be JIT-compiled, Appendix: Functions to JIT-Compile (Priority Order), Consequences, Context, Decision, Negative, Positive (+1 more)

### Community 151 - "test_type_safety_config.py"
Cohesion: 0.14
Nodes (16): backend sub-package (NumPy/JAX abstraction), JAX Architecture & Design, JAX array immutability and lazy evaluation, Automatic differentiation (grad), Debugging and testing JAX code (disable jit, jax.debug.print, JIT-vs-eager parity), Future JAX features for v0.5+ (autodiff, vmap, distributed), GPU acceleration and device placement, JAX library (+8 more)

### Community 154 - "batch_processing.py"
Cohesion: 0.09
Nodes (13): test_chunk_size_bounds_the_blocks_and_progress_counts_every_item(), chunk_iterator(), _energies_key(), _filter_dataframe_fields(), _get_energy_point_data(), _initialize_progress_bar(), load_batch_input(), _prepare_energies_array() (+5 more)

### Community 155 - "test_warm_up.py"
Cohesion: 0.18
Nodes (9): cache_config(), test_cache_dir_is_os_appropriate(), test_cache_helper_enables_and_sets_threshold(), test_cache_helper_keeps_user_threshold(), test_cache_helper_respects_opt_out_and_existing_config(), test_cli_does_not_call_warm_up(), test_library_import_does_not_enable_cache(), test_warm_up_exported_lazily() (+1 more)

### Community 157 - "JAX Architecture & Design (v0.4.0)"
Cohesion: 0.22
Nodes (3): TestGetCompoundFamily, _classify_family_by_elements(), get_compound_family()

### Community 159 - "test_cli_jax_discipline.py"
Cohesion: 0.25
Nodes (12): _batch_argv(), _bragg_argv(), _cache_dir(), _calc_argv(), _compare_argv(), _convert_argv(), _kk_argv(), _run_cli() (+4 more)

### Community 160 - "_outcome"
Cohesion: 0.25
Nodes (3): _log_slivers(), _outcome(), _RefineInfo

### Community 161 - "record_compiles"
Cohesion: 0.18
Nodes (8): _Handler, record_compiles(), _mine(), test_a_second_tables_object_hits_the_same_compiled_program(), test_another_bucket_does_compile(), test_energy_count_inside_the_same_bucket_compiles_nothing(), test_first_call_compiles_then_same_buckets_compile_nothing(), clear_tables()

### Community 162 - "time"
Cohesion: 0.27
Nodes (9): _call_worker(), call(), _gate(), _kernel_worker(), main(), _recorder(), _run_worker(), _timed_ms() (+1 more)

### Community 166 - "test_compound_analysis.py"
Cohesion: 0.19
Nodes (4): TestGetCompoundFrequencyScore, TestGetRecommendedElementsForWarming, get_compound_frequency_score(), get_recommended_elements_for_warming()

### Community 167 - "argparse"
Cohesion: 0.10
Nodes (7): build(), main(), _git(), main(), _versions(), main(), Colors

### Community 168 - "TestBatchProcessing"
Cohesion: 0.25
Nodes (7): Files, Maintenance, `performance_baselines.json`, `sample_materials.json`, Test Data Directory, `test_energies.json`, Usage in Tests

### Community 169 - "Testing Guide"
Cohesion: 0.29
Nodes (8): 30 keV tabulation cut (dashed vertical line), C (carbon) curve, Figure: f1 overestimated by truncating f2 integration at 30 keV, Fe (iron) curve, Kramers-Kronig f2-to-f1 integral truncated to tabulated range, Pb (lead) curve, f1 truncation error (f1_trunc - f1_full) vs photon energy, log-log, Result: truncation error grows with Z (C ~1e-4, Fe ~0.1, Pb ~2.5 e/atom), flat at low energy, rises sharply near 30 keV

### Community 172 - "test_cli.py"
Cohesion: 0.09
Nodes (5): TestAtomicCommand, TestBatchCommand, TestBraggCommand, cmd_atomic(), cmd_bragg()

### Community 173 - "test_encode_memo.py"
Cohesion: 0.22
Nodes (9): test_a_failing_formula_raises_every_time(), test_a_formula_is_encoded_once_and_the_arrays_are_read_only(), test_clearing_the_scattering_cache_clears_the_memo_and_refills_the_element_cache(), test_density_screen_accepts_positive_numbers_of_every_kind(), test_density_screen_rejects_non_positive_numbers_of_every_kind(), test_empty_formula_still_raises_formula_error(), get_cached_elements(), _encode_material() (+1 more)

### Community 175 - "stray_conversions"
Cohesion: 0.28
Nodes (6): _outside_device_helpers(), stray_conversions(), patch_method(), wrapper(), patch_numpy(), wrapper()

### Community 180 - "assert_ulp_close"
Cohesion: 0.22
Nodes (8): assert_ulp_close(), _step(), test_arrays_report_the_worst_entry(), test_infinities_must_be_equal_and_never_close_to_finite(), test_nan_must_match_positions(), test_shape_mismatch_and_empty(), test_within_and_beyond_the_bound(), test_zero_expected_requires_zero()

### Community 183 - "EnergyError"
Cohesion: 0.10
Nodes (3): TestEnergyError, TestBoundaryConditions, EnergyError

### Community 190 - "cli.py"
Cohesion: 0.11
Nodes (4): completion_main(), create_completion_parser(), handle_completion_command(), handle_new_completion_command()

### Community 191 - "kk/test_docs.py"
Cohesion: 0.23
Nodes (7): _public_objects(), test_docs_cite_the_paper_and_upstream_licence(), test_docs_pages_toctree_linked(), test_docs_state_the_eq10_erratum(), test_every_public_kk_name_is_documented(), test_public_docstrings_have_examples_and_references(), test_transform_docstrings_cite_paper_equations()

### Community 193 - "normalize_intensity"
Cohesion: 0.10
Nodes (13): fam_utils_signal(), test_a_constant_series_stays_constant_after_smoothing(), test_linear_background_is_column_wise_on_a_2d_array(), test_normalize_max_and_standard_are_global_over_any_shape(), test_one_dimensional_only_helpers_reject_other_ranks(), TestDataProcessing, background_subtraction(), find_peaks() (+5 more)

### Community 195 - "Data Handling Module API Doc"
Cohesion: 0.18
Nodes (12): Data Handling Module API Doc, atomic_cache (92-element preload, warm_up_cache, get_cache_stats), batch_processing (BatchConfig, calculate_batch_properties, CSV I/O), compound_analysis (compound families, recommended warming elements), CXRO / NIST / Henke .nff scattering factor data (0.03-30 keV, PCHIP), Interfaces Module API Doc, CLI commands (calc, batch, compare, convert, formula, atomic, bragg, list, completion, install-completion, uninstall-completion), CLI error handling messages (+4 more)

### Community 201 - "re"
Cohesion: 0.17
Nodes (7): collect(), main(), _search(), _sphinx(), test_kk_docs_build_adds_no_warnings(), test_kk_docs_doctests_pass(), test_pyproject_package_data_lists_kk_files()

### Community 202 - "core_gpu_check.py"
Cohesion: 0.31
Nodes (8): check_bench(), check_goldens(), check_padding(), check_transfers(), guarded(), main(), record(), _same()

### Community 208 - "backend/__init__.py"
Cohesion: 0.11
Nodes (12): ADR-005: The Kramers-Kronig Package Is JAX-Only, Alternatives considered, Consequences, Context, Evidence, Known deviations from the acceptance gates, References, x64 requirement (+4 more)

### Community 210 - "gen_scattering_tables.py"
Cohesion: 0.24
Nodes (6): build_tables(), _drift(), main(), _row(), _symbols_by_atomic_number(), _unique_files()

### Community 212 - "gc"
Cohesion: 0.29
Nodes (3): clean_cache(), enforce_test_isolation(), setup_test_environment()

### Community 213 - "compare.py"
Cohesion: 0.24
Nodes (5): cmd_compare(), _parse_materials(), _result_to_dict(), _write_report(), _write_table()

### Community 215 - "Array"
Cohesion: 0.18
Nodes (5): _resize(), _sliver_split(), _start_state(), head(), tail()

### Community 216 - "ADR-006: The Core Calculators Are JAX-Only"
Cohesion: 0.17
Nodes (9): Accepted: single-call latency relative to NumPy, ADR-006: The Core Calculators Are JAX-Only, Consequences, Context, Exemptions (host code by design), Gate evidence, Known limitations, progress_bar() (+1 more)

### Community 217 - "subprocess"
Cohesion: 0.23
Nodes (5): main(), _venv_python(), test_headless_gui_smoke_runs_offscreen(), test_backend_lazy_attribute_resolves_after_bare_import(), test_calculators_reexport_the_optics_api_lazily()

### Community 219 - "test_gui_kk_worker_end_to_end.py"
Cohesion: 0.25
Nodes (10): _check_table(), _setup(), test_close_during_warm_up_does_not_hang(), test_error_path_shows_dialog_and_stays_responsive(), test_launcher_schedules_the_warm_up(), test_nearedge_run_matches_api(), test_plain_run_matches_api(), test_refined_run_matches_api() (+2 more)

### Community 223 - "test_no_process_pools.py"
Cohesion: 0.29
Nodes (5): _banned_module(), forbidden(), test_the_package_has_no_process_or_thread_pool(), test_the_scan_flags_every_form_of_a_pool(), test_the_scan_leaves_qt_thread_pools_and_plain_text_alone()

### Community 238 - "ensure_x64"
Cohesion: 0.20
Nodes (4): _warm(), _x64(), _x64(), ensure_x64()

### Community 244 - "warnings"
Cohesion: 0.33
Nodes (4): _outcome(), _same(), test_every_departure_is_a_real_difference(), test_matches_the_legacy_outcome_or_the_documented_departure()

### Community 251 - "calculate_single_material_properties"
Cohesion: 0.07
Nodes (16): TestCLIOutputFormatting, TestSubRefracErrorHandling, _budget(), test_calculation_fills_the_element_cache_that_get_cached_elements_reports(), test_clear_cache_drops_the_resident_tables_and_the_next_call_reloads_them(), test_element_with_atomic_data_but_no_table_keeps_file_not_found(), test_from_factors_call_is_one_upload_and_one_fetch(), test_public_entries_give_float64_without_the_environment_variable() (+8 more)

## Knowledge Gaps
- **123 isolated node(s):** `Henke, Gullikson, Davis 1993 X-ray interactions tables`, `Completion caching system (~/.xraylabtool/cache)`, `Completion for xtool alias`, `Completion features (command, option, path, value)`, `Conda/Mamba install and completion workflow` (+118 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 2478 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **114 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `v0.4.7 (2026-10-23)` connect `pytest` to `create_scattering_factor_interpolators`, `re`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `calculate_single_material_properties()` (e.g. with `fam_single_errors()` and `fam_single_input_types()`) actually correct?**
  _`calculate_single_material_properties()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Henke, Gullikson, Davis 1993 X-ray interactions tables`, `Completion caching system (~/.xraylabtool/cache)`, `Completion for xtool alias` to the rest of the system?**
  _123 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `MainWindowProtocol` be split into smaller, more focused modules?**
  _Cohesion score 0.03663366336633663 - nodes in this community are weakly interconnected._
- **Why does `calculate_single_material_properties()` connect `calculate_single_material_properties` to `TestSubRefracSiliconProperties`, `cases.py`, `ValidationError`, `_make_cubic`, `core.py`, `test_batch_screening.py`, `test_deep_debug_regressions.py`, `test_performance_benchmarks.py`, `clear_scattering_factor_cache`, `style.py`, `batch_processing.py`, `export_to_csv`, `BatchConfig`, `services.py`, `typing`, `clear_scattering_factor_cache`, `calculate_properties_from_factors`, `TestCLIFileOperations`, `MaterialComparator`, `TestEdgeCaseInputValues`, `core_gpu_check.py`, `process_single_calculation`, `set_backend`, `core_bench.py`, `compute_kk`, `kk_gpu_check.py`, `compound_analysis.py`, `TestGoldenAu`, `TestCLICompatibility`, `TestPerformanceBenchmarks`, `test_legacy_contracts.py`, `pathlib`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Are the 47 inferred relationships involving `kk_calculate_real()` (e.g. with `Decision` and `Evidence`) actually correct?**
  _`kk_calculate_real()` has 47 INFERRED edges - model-reasoned connections that need verification._
- **Should `xraylabtool/utils.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07899159663865546 - nodes in this community are weakly interconnected._