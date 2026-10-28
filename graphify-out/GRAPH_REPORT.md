# Graph Report - pyXRayLabTool  (2026-10-28)

## Corpus Check
- 30 files · ~307,901 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 6098 nodes · 13731 edges · 264 communities (170 shown, 94 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 898 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Community 54
- Golden Case Builders
- Community 42
- KK Public API
- Community 126
- Community 52
- Shell Completion Tests
- Plot Palette Widgets
- Community 23
- Community 66
- GUI Headless Smoke
- KK Refinement Loop
- Community 26
- Bulk Atomic Data Tests
- Completion Env Detector
- KK Merge Spectra Tests
- KK Pipeline Device Tests
- JAX Backend Parity Tests
- Community 20
- Community 70
- Community 183
- Community 39
- Community 48
- Community 40
- Style Guide Validation
- Derived Quantities Goldens
- Community 47
- Community 27
- Community 22
- KK Transform Parity
- Community 25
- Community 138
- Community 87
- Community 29
- Community 33
- Community 51
- Community 37
- Community 36
- Community 79
- Community 67
- Compile Count Gates
- Community 38
- Community 62
- PCHIP Interpolation Goldens
- Community 41
- Community 68
- Community 59
- Community 61
- Community 21
- Community 32
- Community 24
- Community 30
- Community 83
- Community 20 (2)
- Community 131
- Community 34
- Community 53
- Community 95
- Community 113
- Community 57
- Community 78
- Community 56
- Community 127
- Community 188
- Community 45
- Community 24 (2)
- Community 31
- Crystal Structure Tests
- Community 85
- Community 63
- Community 69
- Community 58
- Community 72
- Community 77
- Community 73
- Community 74
- Community 50
- Community 129
- Community 75
- Community 65
- Community 43
- Community 71
- Community 81
- Community 86
- Core Calculator Tests
- Community 92
- Community 84
- Community 89
- Community 157
- Core Calculator Tests (2)
- Community 20 (3)
- Community 88
- Community 93
- Community 125
- Community 91
- Community 107
- Community 199
- Community 96
- Community 97
- Community 114
- PCHIP Interpolation Goldens (2)
- Community 102
- Community 98
- Community 94
- Community 99
- Community 100
- Community 106
- Community 119
- Community 103
- Community 105
- Community 109
- Community 55
- Community 35
- Community 133
- Community 130
- Community 115
- Community 110
- Community 187
- Community 82
- CLI Formula Tests
- Community 121
- Community 108
- Community 122
- JAX Backend Parity Tests (2)
- Community 104
- Community 128
- Community 204
- Community 46
- Community 116
- Community 132
- Community 117
- Community 123
- Community 167
- Community 60
- Community 134
- Community 28
- Community 135
- Community 137
- Community 150
- Community 190
- Derived Quantities Goldens (2)
- Community 142
- Community 80
- Runtime Transfer Guard Tests
- Community 148
- Community 149
- Community 140
- Community 145
- Community 146
- Community 151
- Community 152
- Community 90
- Community 159
- Crystal Structure Tests (2)
- Community 153
- Community 64
- Community 179
- Community 222
- Community 35 (2)
- Community 34 (2)
- Community 162
- Core Calculator Tests (3)
- Community 28 (2)
- Community 55 (2)
- Community 143
- Community 185
- Community 206
- Community 61 (2)
- Community 164
- Community 165
- Community 156
- Style Guide Validation (2)
- Community 166
- Community 171
- Community 172
- Community 44
- Community 174
- Community 236
- Community 21 (2)
- Community 178
- Community 168
- Community 31 (2)
- Community 163
- Community 144
- Community 189
- Community 76
- Community 124
- Community 191
- Community 192
- Community 86 (2)
- Community 196
- Community 197
- Community 43 (2)
- Community 216
- Community 201
- Community 139
- Community 203
- Community 182
- Community 205
- Community 50 (2)
- Community 231
- Community 212
- Community 213
- Community 214
- Community 101
- Community 215
- Community 217
- Community 219
- Community 158
- Community 221
- Community 223
- Compile Count Gates (2)
- Core Calculator Tests (4)
- Community 28 (3)
- Community 80 (2)
- Community 39 (2)
- Community 225
- Community 226
- Community 227
- Community 228
- Community 157 (2)
- Community 229
- Community 139 (2)
- Community 181
- Community 218
- Community 82 (2)
- Community 58 (2)
- Core Calculator Tests (5)
- Community 35 (3)
- Community 238
- KK Transform Parity (2)
- Community 242
- Community 244
- Community 245
- Community 248
- Community 29 (2)
- KK Transform Parity (3)
- Community 240
- Community 250
- Community 251
- Community 104 (2)
- Community 255
- Community 257
- Community 274
- Community 279
- Community 280
- Community 281
- Community 282
- Community 283

## God Nodes (most connected - your core abstractions)
1. `calculate_single_material_properties()` - 128 edges
2. `kk_calculate_real()` - 114 edges
3. `MainWindowProtocol` - 97 edges
4. `parse_formula()` - 90 edges
5. `CompletionInstaller` - 69 edges
6. `kk_real_spectrum()` - 67 edges
7. `calculate_xray_properties()` - 62 edges
8. `BatchConfig` - 56 edges
9. `compound_spectrum()` - 52 edges
10. `Spectrum` - 51 edges

## Surprising Connections (you probably didn't know these)
- `Evidence` --references--> `kk_calculate_real()`  [INFERRED]
  docs/architecture/005-adr-kk-jax-only.md → xraylabtool/kk/api.py
- `Context` --references--> `set_backend()`  [INFERRED]
  docs/architecture/005-adr-kk-jax-only.md → xraylabtool/backend/__init__.py
- `Decision` --references--> `Spectrum`  [INFERRED]
  docs/architecture/005-adr-kk-jax-only.md → xraylabtool/kk/_device.py
- `Consequences` --references--> `KKResult`  [INFERRED]
  docs/architecture/005-adr-kk-jax-only.md → xraylabtool/kk/result.py
- `Exemptions (host code by design)` --references--> `KKResult`  [INFERRED]
  docs/architecture/006-adr-core-jax-only.md → xraylabtool/kk/result.py

## Import Cycles
- 3-file cycle: `xraylabtool/calculators/host.py -> xraylabtool/calculators/optics.py -> xraylabtool/data_handling/atomic_cache.py -> xraylabtool/calculators/host.py`
- 3-file cycle: `xraylabtool/calculators/__init__.py -> xraylabtool/calculators/derived_quantities.py -> xraylabtool/calculators/host.py -> xraylabtool/calculators/__init__.py`
- 3-file cycle: `xraylabtool/calculators/__init__.py -> xraylabtool/calculators/core.py -> xraylabtool/calculators/host.py -> xraylabtool/calculators/__init__.py`
- 4-file cycle: `xraylabtool/calculators/__init__.py -> xraylabtool/calculators/optics.py -> xraylabtool/data_handling/atomic_cache.py -> xraylabtool/calculators/host.py -> xraylabtool/calculators/__init__.py`
- 4-file cycle: `xraylabtool/calculators/__init__.py -> xraylabtool/calculators/core.py -> xraylabtool/calculators/derived_quantities.py -> xraylabtool/calculators/host.py -> xraylabtool/calculators/__init__.py`
- 4-file cycle: `xraylabtool/calculators/__init__.py -> xraylabtool/calculators/core.py -> xraylabtool/calculators/cache.py -> xraylabtool/calculators/host.py -> xraylabtool/calculators/__init__.py`
- 4-file cycle: `xraylabtool/calculators/__init__.py -> xraylabtool/calculators/core.py -> xraylabtool/calculators/scattering_data.py -> xraylabtool/calculators/host.py -> xraylabtool/calculators/__init__.py`
- 4-file cycle: `xraylabtool/__init__.py -> xraylabtool/calculators/__init__.py -> xraylabtool/calculators/core.py -> xraylabtool/logging_utils.py -> xraylabtool/__init__.py`

## Hyperedges (group relationships)
- **Watts piecewise-Laurent KK algorithm** — docs_watts_2014_calculation_of_the_kramers_kronig_transform_published_version_piecewise_laurent_polynomial, docs_watts_2014_calculation_of_the_kramers_kronig_transform_published_version_analytic_interval_integration, docs_watts_2014_calculation_of_the_kramers_kronig_transform_published_version_double_interval_integration, docs_watts_2014_calculation_of_the_kramers_kronig_transform_published_version_kronecker_delta_log_handling, docs_watts_2014_calculation_of_the_kramers_kronig_transform_published_version_f1_closed_form [EXTRACTED 0.95]
- **xraylabtool.backend package modules (runtime, shapes, interpolation)** — docs_api_backend_runtime_module, docs_api_backend_shapes_module, docs_api_backend_interpolation_module [EXTRACTED 1.00]
- **Host validate, one upload, one compiled vmap program, one fetch** — docs_architecture_006_adr_core_jax_only_one_device_program_per_call, docs_architecture_006_adr_core_jax_only_resident_scattering_tables, docs_architecture_006_adr_core_jax_only_runtime_and_shapes, docs_architecture_006_adr_core_jax_only_optical_properties, docs_architecture_jax_architecture_optical_program [EXTRACTED 1.00]
- **GUI main window tab mixins composed into MainWindow** — docs_api_gui_main_window, docs_api_gui_single_tab, docs_api_gui_multi_tab, docs_api_gui_kk_tab, docs_api_gui_shared, docs_api_gui_toast [EXTRACTED 1.00]
- **JAX-only decision: ADR-005, ADR-006 and the JAX architecture guide** — docs_architecture_005_adr_kk_jax_only, docs_architecture_006_adr_core_jax_only, docs_architecture_jax_architecture, docs_development_changelog_breaking_core_jax_only [EXTRACTED 1.00]
- **Atomic factors to refractive index to derived optical quantities** — docs_physics_atomic_data_scattering_factors, docs_physics_calculations_number_density, docs_physics_calculations_delta_beta_sum, docs_physics_calculations_critical_angle, docs_physics_calculations_attenuation [EXTRACTED 1.00]
- **Truncation error compared across C, Fe, Pb** — docs_static_kk_truncation_carbon, docs_static_kk_truncation_iron, docs_static_kk_truncation_lead [EXTRACTED 1.00]
- **Refractive index to absorption to attenuation length chain** — docs_physics_xray_optics_complex_refractive_index, docs_physics_xray_optics_absorption_coefficient, docs_physics_xray_optics_attenuation_length, docs_physics_xray_optics_beer_lambert [EXTRACTED 1.00]
- **XRayLabTool exception hierarchy rooted at XRayLabToolError** — docs_api_validation_xraylabtoolerror, docs_api_validation_validationerror, docs_api_validation_formulaerror, docs_api_validation_energyerror, docs_api_validation_calculationerror [EXTRACTED 1.00]
- **Bounding JAX compiles: shape buckets, explicit warm-up, persistent compile cache** — docs_architecture_005_adr_kk_jax_only_shape_buckets, docs_architecture_006_adr_core_jax_only_explicit_warm_up, docs_architecture_005_adr_kk_jax_only_persistent_compile_cache, docs_development_performance_compile_cache_opt [INFERRED 0.85]
- **Device-resident compute pattern (one upload, bucketed shapes, one fetch)** — docs_architecture_005_adr_kk_jax_only_device_resident_pipeline, docs_architecture_004_adr_host_device_transfers_shipped_flow [INFERRED 0.85]
- **Single Material tab workflow: input, compute, plot and table** — docs_static_gui_main_offscreen_material_input, docs_static_gui_main_offscreen_property_plot, docs_static_gui_main_offscreen_results_table, docs_static_gui_main_offscreen_summary_table [INFERRED 0.85]
- **Kramers-Kronig documentation set (physics, guide, CLI, changelog)** — docs_physics_kramers_kronig, docs_guides_kk_guide, docs_guides_cli_reference_kk_command [INFERRED 0.85]
- **Virtual environment-centric completion system** — docs_guides_completion_guide_venv_isolation, docs_guides_completion_guide_env_detection, docs_guides_completion_guide_multi_shell, docs_guides_completion_guide_no_sudo, docs_guides_completion_guide_environment_detector, docs_guides_completion_guide_completion_installer [INFERRED 0.85]

## Communities (264 total, 94 thin omitted)

### Community 0 - "Community 54"
Cohesion: 0.04
Nodes (5): _kk_energy_grid(), KKTabMixin, MultiTabMixin, MainWindowProtocol, SingleTabMixin

### Community 1 - "Golden Case Builders"
Cohesion: 0.07
Nodes (31): build_inputs(), compute_family(), _element_data(), element_density(), fam_atomic_provider(), fam_chunk_contracts(), fam_constants_helpers(), fam_crystal_structure() (+23 more)

### Community 2 - "Community 42"
Cohesion: 0.06
Nodes (36): _evaluate_f2_device(), _asf_f2(), test_asf_grid_spans_10ev_to_500kev(), test_asf_has_92_elements(), test_cached_tables_are_read_only(), test_im_coeffs_shape(), test_load_asf_is_cached(), test_nff_divergent_elements() (+28 more)

### Community 3 - "KK Public API"
Cohesion: 0.05
Nodes (31): _parse_merge(), _check_refine_arguments(), _checked_merge(), _compound_capacity(), _default_grid_size(), _explicit_energies(), _host_inputs(), _merge_points_ev() (+23 more)

### Community 4 - "Community 126"
Cohesion: 0.04
Nodes (31): assert_calculation_result_valid(), assert_memory_usage_reasonable(), assert_performance_within_threshold(), batch_config_default(), batch_config_performance(), batch_config_small(), benchmark_baseline(), clean_cache() (+23 more)

### Community 5 - "Community 52"
Cohesion: 0.05
Nodes (23): Exemptions (host code by design), fam_utils_crystal(), TestCrystallographicCalculations, TestUnitConversions, angle_from_q(), bragg_angle(), _check_lattice(), _convert_atomic_number_to_int() (+15 more)

### Community 6 - "Shell Completion Tests"
Cohesion: 0.04
Nodes (6): test_cli_aliases_registered(), BashCompletionGenerator, CompletionGenerator, FishCompletionGenerator, PowerShellCompletionGenerator, ZshCompletionGenerator

### Community 7 - "Plot Palette Widgets"
Cohesion: 0.06
Nodes (14): apply_pyqtgraph_theme(), apply_styles(), apply_theme(), ColorPalette, get_qss(), apply_palette_to_item(), apply_palette_to_widget(), current_palette() (+6 more)

### Community 8 - "Community 23"
Cohesion: 0.06
Nodes (28): _cxro_deviation(), test_30kev_truncation_matches_paper_trend(), test_default_grid_matches_upstream_golden_bit_for_bit_inputs(), test_end_to_end_matches_upstream_golden(), test_tabulated_kk_f1_close_to_cxro_f1(), _load_golden(), _merged(), test_kernel_on_merged_spectrum_matches_golden_f1_points() (+20 more)

### Community 9 - "Community 66"
Cohesion: 0.05
Nodes (17): TestAtomicDataLookup, clear_lru_caches(), _mock_get_element(), TestCacheResultConsistency, worker(), TestConcurrentGetAtomicNumber, worker(), cache_clearer() (+9 more)

### Community 10 - "GUI Headless Smoke"
Cohesion: 0.06
Nodes (24): _run_smoke(), app(), _fresh(), test_compute_runs_inside_the_worker_guard(), test_core_warm_up_failure_is_logged_not_raised(), test_multiple_materials_on_a_worker_match_direct_calls(), test_start_core_warm_up_runs_one_untracked_worker(), test_the_pool_threads_are_not_the_main_thread() (+16 more)

### Community 11 - "KK Refinement Loop"
Cohesion: 0.07
Nodes (24): Decision, _accumulator_need(), _checked_inputs(), _finalize(), _first_pass(), _fit_active(), _fit_columns(), _grid_array() (+16 more)

### Community 12 - "Community 26"
Cohesion: 0.07
Nodes (25): test_undersized_merge_capacity_reports_the_true_count(), test_device_loop_runs_under_transfer_guard(), test_cold_concurrent_first_call_uploads_the_tables_once(), test_compound_spectrum_device_under_strict_guard(), test_evaluate_f2_device_under_strict_guard(), test_host_to_asf_scale_beta_passes_the_strict_guard(), call(), test_merge_device_under_strict_guard() (+17 more)

### Community 13 - "Bulk Atomic Data Tests"
Cohesion: 0.05
Nodes (14): test_whitespace_delimited_loader_works_with_current_pandas(), TestAnomalousDispersion, TestAtomicDataCacheIsImmutable, TestDeviceProgramCompilation, TestExports, TestGuiErrorHandling, task(), TestLazySubmoduleAccess (+6 more)

### Community 15 - "KK Merge Spectra Tests"
Cohesion: 0.12
Nodes (38): _edge_spectrum(), test_asf_input_is_unchanged_and_unknown_type_raises(), test_beta_round_trip_and_density_cancels_at_merge(), test_default_merge_points_are_the_overlap_of_both_ranges(), test_directly_constructed_spectrum_is_validated(), test_docstring_examples_run(), test_file_in_kev_declared_ev_raises_no_overlap(), test_flat_endpoints_raise_not_inf() (+30 more)

### Community 16 - "KK Pipeline Device Tests"
Cohesion: 0.07
Nodes (30): _colliding_pair(), _passes(), _run(), _subprocess(), test_accelerator_block_rule_gives_the_same_result(), test_backend_field_is_jax(), test_chunk_size_is_upper_bound(), test_cold_concurrent_first_calls_upload_the_tables_once() (+22 more)

### Community 17 - "JAX Backend Parity Tests"
Cohesion: 0.07
Nodes (20): test_all_zero_formula_is_skipped_by_the_collection_calls(), test_a_valid_input_is_returned_as_a_new_float64_array(), test_invalid_energies_keep_their_exact_errors(), test_scalars_and_integer_input_come_back_as_1d_float64(), test_the_inclusive_limits_apply_after_widening_to_float64(), test_the_range_limits_are_inclusive(), test_xray_result_keeps_float64_arrays_and_converts_lists(), test_xray_result_normalises_array_fields_to_float64() (+12 more)

### Community 18 - "Community 20"
Cohesion: 0.05
Nodes (22): gen(), _hermite(), shipped(), tables_mod(), test_check_flag_fails_on_drift(), test_check_flag_passes_on_the_shipped_file(), test_coef_evaluation_at_the_knots_edges_and_just_outside(), test_element_index() (+14 more)

### Community 19 - "Community 70"
Cohesion: 0.10
Nodes (32): app(), _small_run(), test_energies_outside_cxro_range_reach_the_service(), test_energy_range_controls_reach_the_service(), spy(), test_exhausted_refinement_shows_a_warning_toast(), test_export_csv_columns(), test_export_csv_io_error_is_reported() (+24 more)

### Community 20 - "Community 183"
Cohesion: 0.05
Nodes (11): Scope and what follows, TestDtypePreservation, TestEnergyToWavelength, TestFundamentalConstants, TestMathematicalConstants, TestUnitConversionConstants, critical_angle_from_delta(), attenuation_length_cm() (+3 more)

### Community 21 - "Community 39"
Cohesion: 0.07
Nodes (25): _chunk(), test_a_complex_energy_does_not_share_a_group_with_real_bytes(), test_a_reshaped_energy_grid_is_rejected_for_every_item(), test_a_sequence_density_in_the_middle_of_a_chunk_keeps_the_rows_aligned(), test_a_sequence_density_is_rejected_up_front_by_calculate_xray_properties(), test_a_sequence_density_is_skipped_by_the_multiple_call(), test_a_sequence_density_warns_and_maps_to_none_in_the_batch(), test_an_energy_input_numpy_cannot_arrayify_fails_alone() (+17 more)

### Community 22 - "Community 48"
Cohesion: 0.06
Nodes (11): main(), TestCheckGpuAvailability, TestGetDeviceInfo, TestGetGpuInfo, TestGetSystemCudaVersion, check_gpu_availability(), check_plugin_conflicts(), get_device_info() (+3 more)

### Community 23 - "Community 40"
Cohesion: 0.07
Nodes (8): TestBatchProcessorEquivalence, TestBatchConfig, no_progress_config(), TestBatchInputValidation, TestBatchPartialFailure, TestBatchProcessing, BatchConfig, calculate_batch_properties()

### Community 24 - "Style Guide Validation"
Cohesion: 0.07
Nodes (4): main(), StyleGuideValidator, StyleViolation, ValidationReport

### Community 25 - "Derived Quantities Goldens"
Cohesion: 0.06
Nodes (8): TestGoldenAttenuationLength, TestGoldenCriticalAngle, TestGoldenScatteringLengthDensity, calculate_attenuation_length(), calculate_critical_angle(), calculate_scattering_length_density(), _map(), _scattering_length_density()

### Community 26 - "Community 47"
Cohesion: 0.10
Nodes (35): _assert_matches_reference(), _explicit_scale(), _poly(), _reference(), _run(), _status_of(), test_adjacent_float_knots_use_lower_boundary(), test_decimal_scale_of_the_near_edge_values_matches_scale_one() (+27 more)

### Community 27 - "Community 27"
Cohesion: 0.06
Nodes (5): TestBenchmarkComparison, TestPerformanceBenchmarks, TestPerformanceRegression, TestScalabilityBenchmarks, timer()

### Community 28 - "Community 22"
Cohesion: 0.09
Nodes (25): Amendments to the design spec, recorded here, Decision, total_dispersion(), test_standalone_primitives_match_golden_edges(), one(), attenuation_length_from_beta(), derive(), _derived() (+17 more)

### Community 29 - "KK Transform Parity"
Cohesion: 0.09
Nodes (23): test_matches_upstream_improve_accuracy_golden(), test_matches_upstream_kk_pp_golden(), assert_kk_close(), kk_rounding_scale(), additivity_energies_kev(), test_additivity(), _case(), _eval() (+15 more)

### Community 31 - "Community 138"
Cohesion: 0.08
Nodes (16): setup(), _git(), main(), _versions(), _call_worker(), _gate(), _kernel_worker(), main() (+8 more)

### Community 32 - "Community 87"
Cohesion: 0.06
Nodes (5): TestBatchProcessingMemoryManagement, TestCacheManagement, TestMemoryLeakPrevention, TestMemoryMonitor, MemoryMonitor

### Community 33 - "Community 29"
Cohesion: 0.08
Nodes (7): si(), TestParseFormulaBasic, TestParseFormulaDecimal, TestParseFormulaErrors, TestParseFormulaParentheses, parse_formula(), _parse_formula()

### Community 34 - "Community 33"
Cohesion: 0.11
Nodes (21): test_warm_up_precompiles_the_requested_energy_buckets(), test_padding_does_not_change_a_bit_of_the_pchip_programs(), bucket_pow2(), broadcast_shape(), check_molecular_weight(), device_tables(), execute(), _jitted() (+13 more)

### Community 35 - "Community 51"
Cohesion: 0.09
Nodes (21): main(), _run(), padded(), loss(), test_interp_exact_knot_with_overflowing_slope(), test_relativistic_correction_device(), _interp(), _midpoint_f1() (+13 more)

### Community 36 - "Community 37"
Cohesion: 0.13
Nodes (29): _args(), _eq10_corrected(), _f2_poly(), _golden(), _linear_segments(), _quadrature_f1(), test_chunk_invariance(), test_constant_band_closed_form() (+21 more)

### Community 37 - "Community 36"
Cohesion: 0.08
Nodes (3): TestSentinelMarkers, CompletionInstaller, test_completion_installation()

### Community 38 - "Community 79"
Cohesion: 0.07
Nodes (10): TestXRayResultProtocolCompliance, ensure_complex128_array(), ensure_float64_array(), get_optimal_chunk_size(), is_complex_array(), is_energy_array(), is_real_array(), optimize_array_memory_layout() (+2 more)

### Community 39 - "Community 67"
Cohesion: 0.13
Nodes (30): _device_run(), _host_reference(), _logged(), _narrow_last_segment_table(), _refine(), _same_result(), _spacing_grid(), test_accumulator_growth_path_is_bit_identical() (+22 more)

### Community 40 - "Compile Count Gates"
Cohesion: 0.07
Nodes (17): TestCLIOutputFormatting, TestSubRefracErrorHandling, _budget(), test_calculation_fills_the_element_cache_that_get_cached_elements_reports(), test_clear_cache_drops_the_resident_tables_and_the_next_call_reloads_them(), test_element_with_atomic_data_but_no_table_keeps_file_not_found(), test_from_factors_call_is_one_upload_and_one_fetch(), test_public_calls_run_under_an_outer_explicit_transfer_guard() (+9 more)

### Community 41 - "Community 38"
Cohesion: 0.07
Nodes (5): TestBasicExportIntegration, TestBasicExportFunctionality, export_to_csv(), export_to_json(), __getattr__()

### Community 42 - "Community 62"
Cohesion: 0.08
Nodes (15): _collect_compound_elements(), _collect_family_compound_elements(), _collect_similar_compound_elements(), FastAtomicDataProvider, get_atomic_data_fast(), get_bulk_atomic_data_fast(), get_cache_stats(), is_element_preloaded() (+7 more)

### Community 43 - "PCHIP Interpolation Goldens"
Cohesion: 0.07
Nodes (8): _check_element(), cold_cache(), test_si_o_au_interpolators_are_independent(), TestPchipAu, TestPchipO, TestPchipSi, create_scattering_factor_interpolators(), TableInterpolator

### Community 44 - "Community 41"
Cohesion: 0.11
Nodes (5): _parse_csv(), _run(), test_empty_default_grid_after_merge_exits_1(), TestKKCli, _write_edge()

### Community 45 - "Community 68"
Cohesion: 0.05
Nodes (6): TestBatchRowResilience, TestCLIErrorHandling, TestCLIFileOperations, TestCLIInputValidation, TestCLIPerformanceAndMemory, TestFieldNameValidation

### Community 46 - "Community 59"
Cohesion: 0.10
Nodes (17): _cell(), _cli_times(), _cli_times_warm_cache(), cold_env(), _compile_counts(), _fmt(), _importtime(), machine_state() (+9 more)

### Community 47 - "Community 61"
Cohesion: 0.08
Nodes (8): TestAtomicDataError, TestDataFileError, TestExceptionHierarchy, TestValidationError, AtomicDataError, DataFileError, ValidationError, XRayLabToolError

### Community 48 - "Community 21"
Cohesion: 0.11
Nodes (3): build_parser(), main(), Toast

### Community 49 - "Community 32"
Cohesion: 0.10
Nodes (25): _derived(), _exact_reference(), _shapes(), _si(), test_all_scalar_derived_quantities_are_float_and_numpy_scalars(), test_all_zero_formula_raises_zero_division(), test_all_zero_formula_with_a_numpy_density_keeps_the_non_finite_failure(), test_derived_field_shapes_follow_their_own_operands() (+17 more)

### Community 50 - "Community 24"
Cohesion: 0.08
Nodes (5): TestCompletionInstallToTempDir, TestCompletionScriptGeneration, CompletionManager, get_global_options(), get_xraylabtool_commands()

### Community 51 - "Community 30"
Cohesion: 0.09
Nodes (24): _assert_golden(), _assert_material(), _check_formulas(), _props(), _same(), test_derived_quantities_golden_pins_clamped_rules(), test_energy_outside_the_table_sets_ok_false_not_an_error(), test_every_element_matches_golden() (+16 more)

### Community 53 - "Community 20 (2)"
Cohesion: 0.07
Nodes (23): run_tail(), test_max_block_bounds_the_materials_per_block(), test_a_material_larger_than_the_budget_runs_as_blocks_of_exactly_one(), test_element_without_table_raises_unknown_element(), test_empty_formula_raises_formula_error(), test_errors_match_the_task0_catalogue(), test_matches_molecular_properties_bit_for_bit(), test_parse_errors_are_not_cached_and_repeat() (+15 more)

### Community 54 - "Community 131"
Cohesion: 0.11
Nodes (17): _x64(), test_compute_kk_default_grid_cxro_matches_result_energies(), test_compute_kk_forwards_energies_kev(), test_compute_kk_forwards_refine_tolerance(), test_compute_kk_merge_points_follow_energy_unit(), test_compute_kk_rejects_unknown_energy_unit(), test_compute_kk_service_matches_api_with_nearedge(), test_compute_kk_service_matches_api_without_nearedge() (+9 more)

### Community 55 - "Community 34"
Cohesion: 0.06
Nodes (3): TestGoldenScatteringFactors, TestGoldenSi, TestGoldenSiO2

### Community 56 - "Community 53"
Cohesion: 0.10
Nodes (4): MockXRayResult, TestFormatXRayResult, _format_as_json(), format_xray_result()

### Community 57 - "Community 95"
Cohesion: 0.12
Nodes (16): check_accessibility(), check_documentation_coverage(), Colors, main(), print_section(), print_status(), run_style_checks(), test_doctests() (+8 more)

### Community 58 - "Community 113"
Cohesion: 0.13
Nodes (10): TestJaxPchipInterpolator, _query(), restore_x64(), _scipy(), TestMatchesScipy, TestPrecision, TestQueryHandling, TestShapePreservation (+2 more)

### Community 59 - "Community 57"
Cohesion: 0.06
Nodes (4): TestEnvironmentIntegration, TestEnvironmentProperties, TestEnvironmentType, EnvironmentType

### Community 60 - "Community 78"
Cohesion: 0.07
Nodes (5): BaseIntegrationTest, BaseUnitTest, BaseXRayLabToolTest, generate_test_energies(), generate_test_materials()

### Community 61 - "Community 56"
Cohesion: 0.08
Nodes (3): TestCodeQuality, TestDocumentationPatterns, TestStyleCompliance

### Community 62 - "Community 127"
Cohesion: 0.10
Nodes (16): run_program(), _Handler, kk_programs(), numerical_programs(), record_compiles(), _mine(), test_a_second_tables_object_hits_the_same_compiled_program(), test_another_bucket_does_compile() (+8 more)

### Community 63 - "Community 188"
Cohesion: 0.08
Nodes (31): Calculators Module (API docs), xraylabtool.calculators.cache, calculate_single_material_properties, calculate_xray_properties, xraylabtool.calculators.core, xraylabtool.calculators.derived_quantities, xraylabtool.calculators.optics (device programs), X-ray complex refractive index n = 1 - delta - i beta (+23 more)

### Community 64 - "Community 45"
Cohesion: 0.12
Nodes (20): test_the_planner_never_returns_a_block_of_zero(), _budget(), platform(), test_block_for_examples(), test_block_for_never_exceeds_the_request(), test_bucket_is_multiple_of_8_monotone_and_bounded(), test_bucket_ladder_head(), test_bucket_pow2() (+12 more)

### Community 65 - "Community 24 (2)"
Cohesion: 0.08
Nodes (7): TestInstallCompletionMain, completion_main(), create_completion_parser(), handle_completion_command(), install_completion_main(), show_completion_info(), handle_new_completion_command()

### Community 66 - "Community 31"
Cohesion: 0.11
Nodes (14): _format_array_field_single(), _format_as_csv(), _format_as_json(), _format_filtered_table(), _format_material_properties(), _format_multiple_energies(), _format_multiple_energy_section(), _format_scalar_field() (+6 more)

### Community 67 - "Crystal Structure Tests"
Cohesion: 0.10
Nodes (4): _make_cubic(), _phase(), TestCrystalStructureAPI, TestStructureFactorPhysics

### Community 68 - "Community 85"
Cohesion: 0.12
Nodes (17): check_additivity(), check_bench(), check_f2_si_edge(), check_oracle(), check_padding(), check_transfers(), _f2_numpy(), guarded() (+9 more)

### Community 69 - "Community 63"
Cohesion: 0.08
Nodes (8): BasePerformanceTest, TestArrayOptimization, TestBulkAtomicDataOptimization, TestDeprecationWarningOptimization, TestElementPathOptimization, TestInterpolatorCaching, TestNumericalStabilityOptimizations, TestPerformanceRegression

### Community 70 - "Community 69"
Cohesion: 0.11
Nodes (13): add_atomic_command(), add_batch_command(), add_bragg_command(), add_calc_command(), add_compare_command(), add_completion_command(), add_convert_command(), add_formula_command() (+5 more)

### Community 71 - "Community 58"
Cohesion: 0.11
Nodes (11): kk_log(), x64_restored(), test_configure_logging_writes_file(), test_get_logger_child_inherits_config(), _bool_env(), _coerce_int_env(), configure_logging(), get_log_file_path() (+3 more)

### Community 72 - "Community 72"
Cohesion: 0.13
Nodes (12): f1_reference(), _gauss_legendre(), _h(), _pv_integral(), _quadrature(), _f2_mismatch(), oracle_energies(), _reference() (+4 more)

### Community 73 - "Community 77"
Cohesion: 0.09
Nodes (4): _clear_cache(), cold_cache(), TestGoldenDerivedQuantities, clear_scattering_factor_cache()

### Community 74 - "Community 73"
Cohesion: 0.09
Nodes (7): _calc(), _clear_cache(), TestComplexFormula, TestElectronCountGolden, TestExactCapturedValues, TestMolecularWeightGolden, _calculate_molecular_properties()

### Community 75 - "Community 74"
Cohesion: 0.09
Nodes (6): TestCLIIntegration, TestMainFunction, cmd_completion(), cmd_install_completion(), cmd_uninstall_completion(), main()

### Community 76 - "Community 50"
Cohesion: 0.12
Nodes (23): _fraction_moving_average(), _mine(), _signal(), _smooth_reference(), test_a_constant_series_stays_constant_after_smoothing(), test_a_new_window_is_a_new_program_and_a_repeat_is_not(), test_a_signal_helper_is_one_upload_and_one_fetch(), test_arrays_larger_than_one_element_still_fail_in_float() (+15 more)

### Community 77 - "Community 129"
Cohesion: 0.12
Nodes (15): test_non_finite_status_raises_the_legacy_value_error(), _public_objects(), test_docs_cite_the_paper_and_upstream_licence(), test_docs_pages_toctree_linked(), test_docs_state_the_eq10_erratum(), test_every_public_kk_name_is_documented(), test_public_docstrings_have_examples_and_references(), test_transform_docstrings_cite_paper_equations() (+7 more)

### Community 78 - "Community 75"
Cohesion: 0.16
Nodes (21): call(), test_backend_is_jax_and_chunk_size_does_not_matter(), test_boundary_energies_are_accepted_and_f2_matches_the_table(), test_default_grid_clipped_to_0p03_30_kev_and_logged(), test_docstring_examples_run(), test_empty_default_grid_after_merge_raises(), test_energies_are_copied_not_aliased(), test_energy_one_ulp_outside_the_data_range_raises() (+13 more)

### Community 79 - "Community 65"
Cohesion: 0.16
Nodes (23): _max_midpoint_error(), _refine(), test_debug_log_per_pass_and_info_total(), test_docstring_examples_run(), test_exhaustion_flagged_and_logged(), test_loose_tolerance_adds_nothing(), test_lower_range_end_through_kk_calculate_real(), test_midpoint_stagnation_terminates() (+15 more)

### Community 80 - "Community 43"
Cohesion: 0.08
Nodes (7): form(), _get_app(), qt_app(), TestDensityValidation, TestEnergyGridValidation, TestFormulaValidation, TestPlotAxisLabelTheming

### Community 81 - "Community 71"
Cohesion: 0.10
Nodes (13): angle_from_q(), bragg_angle(), d_spacing_cubic(), d_spacing_orthorhombic(), d_spacing_tetragonal(), energy_to_wavelength(), linear_background(), normalize_intensity() (+5 more)

### Community 83 - "Community 86"
Cohesion: 0.09
Nodes (5): TestAtomicCommand, TestBatchCommand, TestListCommand, cmd_atomic(), cmd_list()

### Community 84 - "Core Calculator Tests"
Cohesion: 0.09
Nodes (9): call(), TestH2OProperties, TestSiO2Properties, test_batch_equals_a_loop_of_single_calls(), test_every_formula_failing_raises_here_and_returns_nothing_there(), test_results_follow_the_input_order_across_element_buckets(), test_the_compile_recorder_sees_a_fresh_shape(), calculate_xray_properties() (+1 more)

### Community 85 - "Community 92"
Cohesion: 0.13
Nodes (6): TestGetCompoundFrequencyScore, TestGetElementsForCompound, TestGetRecommendedElementsForWarming, get_compound_frequency_score(), get_elements_for_compound(), get_recommended_elements_for_warming()

### Community 86 - "Community 84"
Cohesion: 0.15
Nodes (11): _computation(), _pen_colors(), plot(), qt_app(), test_plot_draws_curves_on_unequal_grids(), test_plot_follows_theme_toggle(), test_plot_hides_missing_curves(), test_plot_log_x_rule() (+3 more)

### Community 88 - "Community 157"
Cohesion: 0.10
Nodes (22): Batch transfer optimization: one upload and fetch per F-block (2N to 2), ADR-001 (NumPy default backend), Known deviations from the acceptance gates, Padded Spectrum pytree, Persistent compile cache (CLI/GUI only), Shape buckets (kk compile-count bound), Status pytrees instead of device exceptions, ADR-001 (JAX vs NumPy computation backend) (+14 more)

### Community 89 - "Core Calculator Tests (2)"
Cohesion: 0.11
Nodes (14): _logged(), _same_bits(), test_a_bad_energy_fails_every_formula_of_the_multiple_call(), test_a_binding_budget_splits_a_group_and_bounds_the_padded_blocks(), test_a_chunk_with_different_energies_per_item_follows_its_own_energies(), test_a_failing_formula_in_the_middle_is_skipped_and_logged(), test_a_material_is_bit_identical_whatever_its_batch_mates(), test_batch_keys_values_and_order_with_a_failing_item_in_the_middle() (+6 more)

### Community 90 - "Community 20 (3)"
Cohesion: 0.12
Nodes (14): test_a_bucket_is_never_smaller_than_the_count_and_wastes_at_most_a_factor_two(), test_a_negative_count_is_rejected(), test_above_4096_the_padding_stays_within_the_ladder_bound(), test_few_rungs_cover_the_whole_small_range(), test_powers_of_two_up_to_4096_then_the_ladder(), test_warm_up_compiles_one_program_for_counts_sharing_a_bucket(), test_warm_up_rejects_a_non_integer_energy_count(), test_warm_up_rejects_a_non_positive_energy_count() (+6 more)

### Community 92 - "Community 93"
Cohesion: 0.10
Nodes (3): TestValidateChemicalFormula, _get_valid_element_symbols(), validate_chemical_formula()

### Community 93 - "Community 125"
Cohesion: 0.14
Nodes (8): TestDataProcessing, background_subtraction(), normalize_intensity(), _real(), _run_columns(), _run_series(), _series(), smooth_data()

### Community 94 - "Community 91"
Cohesion: 0.13
Nodes (10): qt_app(), _run_worker(), test_worker_does_not_emit_finished_on_exception(), test_worker_emits_error_on_exception(), bad_fn(), test_worker_emits_result_on_success(), test_worker_injects_progress_cb_when_accepted(), test_worker_passes_kwargs() (+2 more)

### Community 95 - "Community 107"
Cohesion: 0.11
Nodes (3): TestBasicAnalysis, compare_materials(), find_absorption_edges()

### Community 96 - "Community 199"
Cohesion: 0.11
Nodes (5): AtomicScatteringFactor, load_scattering_factor_data(), _ones_like_program(), ScatteringData, _structure_factor_program()

### Community 97 - "Community 96"
Cohesion: 0.10
Nodes (10): assert_values_close(), expect_no_warnings(), expect_warnings(), get_system_info(), log_system_info(), skip_if_insufficient_cpu(), skip_if_insufficient_memory(), stability_test() (+2 more)

### Community 98 - "Community 97"
Cohesion: 0.16
Nodes (14): _backend_label(), check_family(), compare_arrays(), _compare_text(), _inputs(), _meta(), _npz(), _structure_problem() (+6 more)

### Community 100 - "PCHIP Interpolation Goldens (2)"
Cohesion: 0.11
Nodes (10): _budget(), _material(), test_atomic_provider_fills_complex128_from_the_tables_in_one_program_call(), test_atomic_provider_keeps_the_shape_of_its_energies(), test_derived_quantities_are_one_program_call_and_clamped(), test_insufficient_data_keeps_the_value_error(), test_interpolator_crosses_the_boundary_only_through_runtime_upload(), test_interpolator_rejects_host_tables() (+2 more)

### Community 102 - "Community 98"
Cohesion: 0.10
Nodes (3): TestEnergyError, TestBoundaryConditions, EnergyError

### Community 103 - "Community 94"
Cohesion: 0.15
Nodes (8): test_cli_batch_reports_failed_materials_on_stderr_with_progress_on(), cmd_batch(), _convert_result_to_dict(), _parse_batch_data(), _process_batch_materials(), _save_batch_results(), _validate_batch_input(), create_batch_progress_tracker()

### Community 104 - "Community 99"
Cohesion: 0.09
Nodes (21): ADR-003: float64 Preservation Policy, Appendix: Numerical Precision Requirements by Function, Consequences, Context, Current Codebase float64 Usage, Decision, Enforcement Strategy, JAX float32 Default (+13 more)

### Community 105 - "Community 100"
Cohesion: 0.10
Nodes (12): angle_from_q(), bragg_angle(), d_spacing_cubic(), xraylabtool.device, energy_to_wavelength(), get_atomic_data(), load_atomic_data(), xraylabtool.logging_utils (+4 more)

### Community 106 - "Community 106"
Cohesion: 0.15
Nodes (11): cache_config(), test_cache_dir_is_os_appropriate(), test_cache_helper_enables_and_sets_threshold(), test_cache_helper_keeps_user_threshold(), test_cache_helper_respects_opt_out_and_existing_config(), test_cli_does_not_call_warm_up(), test_library_import_does_not_enable_cache(), test_warm_up_exported_lazily() (+3 more)

### Community 107 - "Community 119"
Cohesion: 0.11
Nodes (3): AtomicDataProvider, CalculationEngine, InterpolatorProtocol

### Community 109 - "Community 105"
Cohesion: 0.13
Nodes (6): TestCalcCommand, cmd_calc(), _determine_output_format(), _print_calc_verbose_info(), _save_or_print_output(), _validate_calc_inputs()

### Community 111 - "Community 55"
Cohesion: 0.16
Nodes (3): test_comparator_keeps_the_cause_of_a_per_formula_failure(), ComparisonResult, MaterialComparator

### Community 112 - "Community 35"
Cohesion: 0.13
Nodes (22): Contributing atomic data (citation, range, precision, format), Atomic Scattering Factor Data, Absorption edges (K, L, M) and XANES features, Multi-level caching strategy (preloaded, LRU, disk), CXRO database, Henke et al. 1993, Atomic Data and Nuclear Data Tables 54, 181-342, Henke tables (Z=1-92, nff format), Interpolation methods (linear, logarithmic, spline) (+14 more)

### Community 113 - "Community 133"
Cohesion: 0.11
Nodes (20): Analytic per-interval integration (Eqs. 3-4), Biggs and Lighthill high-energy cross-sections (to 500 keV), Cauchy principal value integral, Conventional KK numerical methods (FFT, Euler-Maclaurin, Gaussian, TANH), Zero-to-infinity convergence conditions on polynomial coefficients, Cross et al. difference-spectrum KK method, Double-interval integration across x=E singularity (Eq. 5), Closed-form f1(E) for piecewise Laurent f2 (Eq. 7, Eq. 10) (+12 more)

### Community 114 - "Community 130"
Cohesion: 0.20
Nodes (15): test_database_rejects_non_integer_atomic_number(), test_evaluate_f2_rejects_complex(), test_evaluate_f2_rejects_nan_segment_edge(), test_kernel_rejects_complex_instead_of_dropping_the_imaginary_part(), test_kk_result_stores_merge_points_as_an_immutable_tuple(), test_merge_spectra_rejects_complex_inputs(), test_non_string_formula_raises_formula_error(), test_refine_rejects_complex_inputs() (+7 more)

### Community 115 - "Community 115"
Cohesion: 0.12
Nodes (3): TestCalculationError, TestExceptionUsage, CalculationError

### Community 117 - "Community 187"
Cohesion: 0.11
Nodes (20): Backend API Reference, xraylabtool.backend.interpolation (JaxPchipInterpolator), xraylabtool.backend.runtime, xraylabtool.backend.shapes, ADR-001: JAX vs NumPy Computation Backend (superseded), ArrayBackend Protocol (NumpyBackend / JaxBackend), InterpolationFactory (scipy vs JaxPchipInterpolator), Functions to JIT vs functions that must not be JIT-compiled (+12 more)

### Community 119 - "CLI Formula Tests"
Cohesion: 0.14
Nodes (6): TestFormulaCommand, cmd_formula(), _get_atomic_data(), _output_formula_results(), _print_formula_results(), _process_formula()

### Community 121 - "Community 108"
Cohesion: 0.15
Nodes (16): test_load_absolute_path(), test_load_csv_file(), test_load_csv_space_separated_fallback(), test_load_empty_nff_raises_data_file_error(), test_load_malformed_nff_raises_data_file_error(), test_load_nff_energy_column_monotone(), test_load_nff_values_are_finite(), test_load_nonexistent_file_raises_file_not_found() (+8 more)

### Community 122 - "Community 122"
Cohesion: 0.12
Nodes (20): GUI (API docs), widgets.kk_plot (Kramers-Kronig Plot), main_window.kk_tab, xraylabtool.gui.logging_filters, xraylabtool.gui.main_window, widgets.material_form, widgets.material_table, main_window.multi_tab (+12 more)

### Community 123 - "JAX Backend Parity Tests (2)"
Cohesion: 0.14
Nodes (4): TestGoldenParity, TestScalarEnergyFallback, _convert_energy_input(), _validate_and_process_energies()

### Community 126 - "Community 204"
Cohesion: 0.10
Nodes (3): TestBackwardCompatibility, TestBashCompletionSubstitution, TestCompletionIntegrationFlow

### Community 128 - "Community 116"
Cohesion: 0.14
Nodes (3): TestCalculationSpeedBenchmarks, calculation_worker(), TestPerformanceRegressionDetection

### Community 129 - "Community 132"
Cohesion: 0.16
Nodes (19): CLI Reference (docs), batch command (--workers ignored), calc command, compare command, completion, install-completion, uninstall-completion commands, convert command, CLI energy specification (single, list, linear and log ranges, keV), formula, atomic, bragg and list commands (+11 more)

### Community 131 - "Community 123"
Cohesion: 0.17
Nodes (7): collect(), main(), _search(), _sphinx(), test_kk_docs_build_adds_no_warnings(), test_kk_docs_doctests_pass(), test_pyproject_package_data_lists_kk_files()

### Community 133 - "Community 167"
Cohesion: 0.12
Nodes (18): Data Handling Module API Reference, data_handling.atomic_cache (92-element host metadata cache), data_handling.batch_processing, BatchConfig (chunk_size, max_workers ignored, memory_limit_gb), data_handling.compound_analysis (compound families, warming recommendations), Atomic scattering data sources (CXRO, NIST, Henke .nff, 0.03-30 keV, PCHIP), Exception hierarchy (XRayLabToolError root), NearEdgeDataError (+10 more)

### Community 134 - "Community 60"
Cohesion: 0.20
Nodes (9): _default_cache_dir(), _run(), test_a_core_call_leaves_the_cache_alone_by_default(), test_a_failed_cache_setup_is_retried_not_hidden(), test_a_relative_xdg_cache_home_is_ignored(), test_a_user_preset_cache_directory_is_respected(), test_only_the_value_1_opts_in(), test_the_opt_in_env_var_enables_the_cache() (+1 more)

### Community 135 - "Community 134"
Cohesion: 0.25
Nodes (12): _batch_argv(), _bragg_argv(), _cache_dir(), _calc_argv(), _compare_argv(), _convert_argv(), _kk_argv(), _run_cli() (+4 more)

### Community 136 - "Community 28"
Cohesion: 0.11
Nodes (4): TestBasicSetupAndInitialization, TestDuplicatedFormulasIndependence, TestPropertyConsistency, TestSubRefracSiliconProperties

### Community 138 - "Community 137"
Cohesion: 0.12
Nodes (15): ADR-002: PyQtGraph vs Matplotlib for GUI Plotting, Appendix: Feature Parity Checklist, Consequences, Context, Decision, Migration Details, Negative, Positive (+7 more)

### Community 139 - "Community 150"
Cohesion: 0.12
Nodes (16): ADR-004: Host-Device Transfer Minimization Strategy, Appendix: Transfer Cost Reference (CPU Backend), Array Sizes in This Workload, Batch Processing Optimization, Consequences, Context, Decision, Implementation Pattern (+8 more)

### Community 140 - "Community 190"
Cohesion: 0.15
Nodes (15): XRayLabTool sub-packages (calculators, data_handling, interfaces, validation, io, analysis, gui, backend, kk, utils), core_bench.py and core_gpu_check.py gates, Frozen golden tests (rtol=1e-12), Host-code exemptions (I/O, parse_formula, validation, GUI, containers), Gate P6 GPU miss (7.29 ms vs 5 ms), Performance gates P1-P13, xraylabtool/_physics.py (JAX utils helpers), Removed names (ops, NumpyBackend, JaxBackend, ArrayBackend, InterpolationFactory, kernels) (+7 more)

### Community 142 - "Community 142"
Cohesion: 0.15
Nodes (3): decorator(), wrapper(), MemoryProfiler

### Community 143 - "Community 80"
Cohesion: 0.25
Nodes (4): _assert_matches(), _in_process(), _table(), TestKKCliEndToEnd

### Community 144 - "Runtime Transfer Guard Tests"
Cohesion: 0.16
Nodes (12): test_a_block_is_one_upload_and_one_fetch(), test_two_hundred_formulas_compile_at_most_one_program_and_transfer_once_per_block(), test_thread_counts_ignore_other_threads(), test_transfer_counters_live_in_runtime_only(), test_clear_tables_forces_reload(), test_load_tables_single_upload_from_threads(), test_load_tables_uploads_once(), test_a_scalar_conversion_is_one_upload_and_one_fetch() (+4 more)

### Community 145 - "Community 148"
Cohesion: 0.20
Nodes (10): test_a_failing_formula_raises_every_time(), test_a_formula_is_encoded_once_and_the_arrays_are_read_only(), test_clearing_the_scattering_cache_clears_the_memo_and_refills_the_element_cache(), test_density_screen_accepts_positive_numbers_of_every_kind(), test_density_screen_rejects_non_positive_numbers_of_every_kind(), test_empty_formula_still_raises_formula_error(), test_run_optical_blocks_rejects_a_non_positive_max_block(), get_cached_elements() (+2 more)

### Community 147 - "Community 140"
Cohesion: 0.20
Nodes (7): build_tables(), _drift(), main(), _row(), _same(), _symbols_by_atomic_number(), _unique_files()

### Community 148 - "Community 145"
Cohesion: 0.17
Nodes (14): Shell Completion Guide, Completion caching system (~/.xraylabtool/cache), Completion for xtool alias, Completion features (command, option, path, value), CompletionInstaller, xraylabtool.interfaces.completion_v2, Conda/Mamba install and completion workflow, Environment detection (venv, conda, mamba, Poetry, Pipenv) (+6 more)

### Community 149 - "Community 146"
Cohesion: 0.25
Nodes (6): analyze_mypy_output(), check_core_modules(), check_type_definitions(), main(), run_mypy_check(), validate_cache_performance()

### Community 150 - "Community 151"
Cohesion: 0.20
Nodes (9): assert_ulp_close(), _step(), test_arrays_report_the_worst_entry(), test_golden_zero_floor_applies_only_to_flagged_arrays(), test_infinities_must_be_equal_and_never_close_to_finite(), test_nan_must_match_positions(), test_shape_mismatch_and_empty(), test_within_and_beyond_the_bound() (+1 more)

### Community 152 - "Community 90"
Cohesion: 0.18
Nodes (4): test_set_backend_calls_do_not_affect_kk(), TestBackendStubs, get_backend(), set_backend()

### Community 153 - "Community 159"
Cohesion: 0.22
Nodes (7): _manual(), test_negative_f1_near_edge_reaches_xray_result(), test_result_module_doctests(), test_to_xray_result_accepts_range_ends(), test_to_xray_result_rejects_out_of_range_energy(), test_to_xray_result_uses_kk_f1(), KKResult

### Community 158 - "Community 222"
Cohesion: 0.20
Nodes (4): build(), main(), main(), _venv_python()

### Community 159 - "Community 35 (2)"
Cohesion: 0.15
Nodes (14): Attenuation coefficients (mu = 4 pi beta / lambda, mass coefficient, length), Critical angle theta_c = sqrt(2 delta) with unit conversions, X-ray Optics Fundamentals, Linear and mass absorption coefficient, Absorption edges (pre-edge, edge jump, EXAFS), Attenuation length (1/e), Beer-Lambert law, Complex refractive index n = 1 - delta - i beta (+6 more)

### Community 163 - "Community 28 (2)"
Cohesion: 0.14
Nodes (5): TestPerformanceBenchmarks, energy_sweep(), complex_formula(), multi_calculation(), single_calculation()

### Community 165 - "Community 143"
Cohesion: 0.16
Nodes (8): dlog(), order(), pass_trace(), wrap(), spy(), si(), test_midpoint_loop_never_evaluates_more_than_block_per_call(), test_refine_buffer_growth_and_shrink_between_passes_is_bit_identical()

### Community 166 - "Community 185"
Cohesion: 0.18
Nodes (8): _spy_blocks(), wrap(), test_heavy_element_segment_grid_against_upstream_emulation(), test_invalid_refine_arguments_raise_validation_error(), test_kk_calculate_real_block_never_exceeds_chunk_size(), test_refine_block_never_exceeds_chunk_size(), test_refine_rejects_non_increasing_input(), _upstream_like_added()

### Community 167 - "Community 206"
Cohesion: 0.21
Nodes (3): TestParseChemicalFormula, TestDelegatingWrappers, parse_chemical_formula()

### Community 171 - "Community 156"
Cohesion: 0.23
Nodes (5): _json_numpy_default(), save_calculation_results(), _save_dict_as_csv(), _save_results_as_csv(), _save_results_as_json()

### Community 172 - "Style Guide Validation (2)"
Cohesion: 0.23
Nodes (5): _banned_module(), forbidden(), test_the_package_has_no_process_or_thread_pool(), test_the_scan_flags_every_form_of_a_pool(), test_the_scan_leaves_qt_thread_pools_and_plain_text_alone()

### Community 174 - "Community 171"
Cohesion: 0.31
Nodes (8): check_bench(), check_goldens(), check_padding(), check_transfers(), guarded(), main(), record(), _same()

### Community 175 - "Community 172"
Cohesion: 0.32
Nodes (8): _checkout(), _eval_points(), _finite(), _git(), main(), _poly(), _save(), _transform_case()

### Community 178 - "Community 236"
Cohesion: 0.17
Nodes (5): test_refine_rejects_bad_arguments(), test_type_hints_resolve_at_runtime(), refine_spectrum(), RefineOutcome, _readonly_1d()

### Community 180 - "Community 178"
Cohesion: 0.24
Nodes (5): cmd_compare(), _parse_materials(), _result_to_dict(), _write_report(), _write_table()

### Community 181 - "Community 168"
Cohesion: 0.26
Nodes (12): Kramers-Kronig (kk) API Reference, load_asf / AsfDatabase (vendored kkcalc ASF tables, 10 eV-500 keV), calculate_properties_from_factors (delta, beta from f1/f2 totals), kk function-to-equation map (Watts 2014), kk_calculate_real / KKResult (high-level entry, keV), kk_f1 (device-level f1, jit/vmap/grad compatible), kk_piecewise_poly (Eqs. 7, 10; a-3 sign erratum), kkcalc upstream (commit f331ca3, zlib licence) (+4 more)

### Community 183 - "Community 163"
Cohesion: 0.26
Nodes (3): TestGetCompoundFamily, _classify_family_by_elements(), get_compound_family()

### Community 185 - "Community 189"
Cohesion: 0.24
Nodes (11): Validation Module API Doc, Validation best practices (validate first, catch specific exceptions), CalculationError, EnergyError, FormulaError, validate_calculation_parameters, validate_chemical_formula, validate_density (+3 more)

### Community 186 - "Community 76"
Cohesion: 0.25
Nodes (5): _fake(), gate(), _p12(), test_p12_fails_when_the_timed_warm_calls_compile(), test_p12_passes_when_nothing_compiles_after_warm_up()

### Community 187 - "Community 124"
Cohesion: 0.29
Nodes (4): cmd_kk(), _fail(), _format_output(), _parse_energies()

### Community 188 - "Community 191"
Cohesion: 0.22
Nodes (11): Available Scripts, Example Output:, Integration:, Maintenance Notes, Output:, Requirements:, `run_type_check.py` - Type Checking, Scripts Directory (+3 more)

### Community 191 - "Community 196"
Cohesion: 0.38
Nodes (6): _outcome(), _result(), test_kk_result_arrays_are_read_only_copies(), test_kk_result_rejects_inconsistent_state(), test_refine_outcome_arrays_are_read_only_copies(), test_refine_outcome_rejects_inconsistent_state()

### Community 195 - "Community 201"
Cohesion: 0.20
Nodes (9): ADR-001: JAX vs NumPy Computation Backend, Appendix: Functions that MUST NOT be JIT-compiled, Appendix: Functions to JIT-Compile (Priority Order), Consequences, Context, Decision, Negative, Positive (+1 more)

### Community 196 - "Community 139"
Cohesion: 0.29
Nodes (9): kkcalc attribution (Zlib licence, Watts 2014), Kramers-Kronig Transform (physics page), Accuracy validation (upstream parity, quadrature, Lorentz oscillator, CXRO comparison, conditioning), Closed-form interval sum (corrected Eq. 10, kk_piecewise_poly), Kramers-Kronig dispersion relation for f1 from f2, kkcalc upstream implementation (KK_PP, improve_accuracy), Piecewise Laurent polynomial representation of f2 (Eq. 8, eV units), Relativistic correction Z* (Eq. 9) and six-step workflow (+1 more)

### Community 197 - "Community 203"
Cohesion: 0.29
Nodes (10): XRayLabTool GUI main window screenshot (Single Material tab, offscreen render), Energy-Wavelength converter widget (E->A, A->E), Footer controls: Log Path, Light Mode (theme toggle), Material input panel (formula SiO2, density 2.2 g/cm3, energy grid 8-12 keV, 50 points, Compute button, live validation feedback), Presets panel (material and energy preset selectors), Property plot: attenuation length vs energy for SiO2, with Log X/Log Y, Save Plot PNG, Export CSV, Results table (energy, wavelength, delta, beta, critical angle, attenuation length, mu, f1, f2, SLD), SiO2 example calculation (60.083 g/mol, 2.2 g/cm3, 30 electrons) (+2 more)

### Community 198 - "Community 182"
Cohesion: 0.27
Nodes (4): _finite(), main(), _rel_corr(), probe_energies()

### Community 199 - "Community 205"
Cohesion: 0.29
Nodes (4): _outcome(), _same(), test_every_departure_is_a_real_difference(), test_matches_the_legacy_outcome_or_the_documented_departure()

### Community 200 - "Community 50 (2)"
Cohesion: 0.22
Nodes (7): _offset_signal(), test_smoothing_offset_data_agrees_across_pad_widths(), test_standard_normalisation_of_offset_data_agrees_across_pad_widths(), test_standard_normalisation_of_offset_data_matches_an_accurate_reference(), test_the_polynomial_background_at_two_pad_widths_agrees(), test_the_same_data_at_two_pad_widths_agrees_up_to_summation_order(), _two_widths()

### Community 201 - "Community 231"
Cohesion: 0.28
Nodes (8): JAX floor >=0.12.0.1, Release v0.4.7 (2026-10-23), Installation (pip, gpu_cuda extras, conda, uv dev), check_version.py version consistency, gpu_check.py GPU diagnostics, gui_headless_smoke.py GUI smoke check, run_type_check.py type checking script, test_install.py install smoke test

### Community 203 - "Community 213"
Cohesion: 0.22
Nodes (5): memory_test(), performance_test(), decorator(), requires_memory(), decorator()

### Community 205 - "Community 101"
Cohesion: 0.22
Nodes (5): test_parse_formula_is_cached_but_public_result_is_fresh_lists(), _aggregate_element_counts(), _expand_parentheses(), _parse_formula_cached(), _validate_balanced_parentheses()

### Community 206 - "Community 215"
Cohesion: 0.28
Nodes (6): _outside_device_helpers(), stray_conversions(), patch_method(), wrapper(), patch_numpy(), wrapper()

### Community 208 - "Community 219"
Cohesion: 0.25
Nodes (7): Files, Maintenance, `performance_baselines.json`, `sample_materials.json`, Test Data Directory, `test_energies.json`, Usage in Tests

### Community 210 - "Community 221"
Cohesion: 0.29
Nodes (8): 30 keV tabulation cut (dashed vertical line), C (carbon) curve, Figure: f1 overestimated by truncating f2 integration at 30 keV, Fe (iron) curve, Kramers-Kronig f2-to-f1 integral truncated to tabulated range, Pb (lead) curve, f1 truncation error (f1_trunc - f1_full) vs photon energy, log-log, Result: truncation error grows with Z (C ~1e-4, Fe ~0.1, Pb ~2.5 e/atom), flat at low energy, rises sharply near 30 keV

### Community 215 - "Community 80 (2)"
Cohesion: 0.25
Nodes (3): _subprocess(), test_cli_writes_the_persistent_compile_cache(), test_version_unaffected()

### Community 221 - "Community 157 (2)"
Cohesion: 0.29
Nodes (7): ADR-005: The Kramers-Kronig Package Is JAX-Only, Alternatives considered, Consequences, Context, Evidence, References, x64 requirement

### Community 222 - "Community 229"
Cohesion: 0.38
Nodes (7): Testing Guide (docs), Characterization golden-value tests (202 assertions), CI test matrix: Python 3.12-3.14, Ubuntu/macOS/Windows, Test contribution guidelines and naming convention, Coverage requirements: 22% CI gate, Codecov patch check, Test categories: unit, integration, performance, characterization, physics, nearedge_example.txt (carbon K-edge NEXAFS sample spectrum)

### Community 223 - "Community 139 (2)"
Cohesion: 0.33
Nodes (7): Behaviour decisions D1-D7, kk limits (range, conditioning, wrong energy unit symptom), kkcalc ASF data and nine elements differing from .nff (Sc, Mo, Ru, Be, Ti, Si, La, Gd, W), Henke et al. 1993 photoabsorption data (10 eV-30 keV), Integration limits: data covers 10 eV to 500 keV, f2=0 outside, Singularities at knots and end points (decision D5 conventions), Truncating the integral at 30 keV changes f1 (C, Fe, Pb)

### Community 225 - "Community 218"
Cohesion: 0.29
Nodes (3): formula(), _format_as_csv(), _format_as_table()

### Community 229 - "Community 35 (3)"
Cohesion: 0.33
Nodes (5): Contributing Guide, Code standards (PEP 8, 88 cols, NumPy docstrings, type hints), Performance requirements (<0.1 ms single, >100k calc/s batch, >90% cache hit), Review process (automated checks, reviewer, maintainer, squash merge), Development workflow (branching, ruff, mypy, pytest, make verify)

### Community 231 - "KK Transform Parity (2)"
Cohesion: 0.47
Nodes (3): _random_call(), test_fuzz_covers_every_input_type(), test_fuzz_valid_calls()

## Knowledge Gaps
- **94 isolated node(s):** `kkcalc Zlib licence (Watts, Lauk 2013)`, `xraylabtool.device`, `xraylabtool.logging_utils`, `xraylabtool.typing_extensions`, `xraylabtool.gui.logging_filters` (+89 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 2472 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **94 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `calculate_xray_properties()` connect `Core Calculator Tests` to `Community 116`, `Golden Case Builders`, `Community 28`, `Bulk Atomic Data Tests`, `Runtime Transfer Guard Tests`, `JAX Backend Parity Tests`, `Community 39`, `Community 22`, `Core Calculator Tests (3)`, `Community 28 (2)`, `Community 55 (2)`, `Community 171`, `Community 32`, `Community 78`, `Core Calculator Tests (4)`, `Community 28 (3)`, `Core Calculator Tests (2)`, `Community 20 (3)`, `Community 114`, `Community 55`, `Community 187`, `JAX Backend Parity Tests (2)`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `calculate_single_material_properties()` (e.g. with `call()` and `fam_single_errors()`) actually correct?**
  _`calculate_single_material_properties()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `kkcalc Zlib licence (Watts, Lauk 2013)`, `xraylabtool.device`, `xraylabtool.logging_utils` to the rest of the system?**
  _94 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 54` be split into smaller, more focused modules?**
  _Cohesion score 0.03793032364460936 - nodes in this community are weakly interconnected._
- **Why does `calculate_single_material_properties()` connect `Compile Count Gates` to `Community 116`, `Golden Case Builders`, `Community 28`, `GUI Headless Smoke`, `Bulk Atomic Data Tests`, `JAX Backend Parity Tests`, `Community 39`, `Community 40`, `Community 27`, `Community 28 (2)`, `Community 38`, `Community 68`, `Community 171`, `Community 32`, `Community 131`, `Community 34`, `Community 78`, `Community 77`, `Community 129`, `Core Calculator Tests`, `Compile Count Gates (2)`, `Community 28 (3)`, `Community 39 (2)`, `Core Calculator Tests (2)`, `Community 20 (3)`, `Community 107`, `Community 114`, `Community 94`, `Community 105`, `JAX Backend Parity Tests (2)`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Are the 47 inferred relationships involving `kk_calculate_real()` (e.g. with `Decision` and `Evidence`) actually correct?**
  _`kk_calculate_real()` has 47 INFERRED edges - model-reasoned connections that need verification._
- **Should `Golden Case Builders` be split into smaller, more focused modules?**
  _Cohesion score 0.0675990675990676 - nodes in this community are weakly interconnected._