# Graph Report - pyXRayLabTool  (2026-10-28)

## Corpus Check
- 285 files · ~269,120 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 6033 nodes · 13698 edges · 287 communities (177 shown, 110 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 921 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Core Calculator Tests
- Golden Case Builders
- KK Merge Spectra Tests
- Shell Completion Tests
- Style Guide Validation
- JAX Backend Parity Tests
- Bulk Atomic Data Tests
- Completion Env Detector
- KK Refinement Loop
- KK Pipeline Device Tests
- Derived Quantities Goldens
- GUI Headless Smoke
- Runtime Transfer Guard Tests
- KK Transform Parity
- PCHIP Interpolation Goldens
- KK Public API
- Crystal Structure Tests
- CLI Formula Tests
- Plot Palette Widgets
- Compile Count Gates
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29
- Community 30
- Community 31
- Community 32
- Community 33
- Community 34
- Community 35
- Community 36
- Community 37
- Community 38
- Community 39
- Community 40
- Community 41
- Community 42
- Community 43
- Community 44
- Community 45
- Community 46
- Community 47
- Community 48
- Community 49
- Community 50
- Community 51
- Community 52
- Community 53
- Community 54
- Community 55
- Community 56
- Community 57
- Community 58
- Community 59
- Community 60
- Community 61
- Community 62
- Community 63
- Community 64
- Community 65
- Community 66
- Community 67
- Community 68
- Community 69
- Community 70
- Community 71
- Community 72
- Community 73
- Community 74
- Community 75
- Community 76
- Community 77
- Community 78
- Community 79
- Community 80
- Community 81
- Community 82
- Community 83
- Community 84
- Community 85
- Community 86
- Community 87
- Community 88
- Community 89
- Community 90
- Community 91
- Community 92
- Community 93
- Community 94
- Community 95
- Community 96
- Community 97
- Community 98
- Community 99
- Community 100
- Community 101
- Community 102
- Community 103
- Community 104
- Community 105
- Community 106
- Community 107
- Community 108
- Community 109
- Community 110
- Community 111
- Community 112
- Community 113
- Community 114
- Community 115
- Community 116
- Community 117
- Community 118
- Community 119
- Community 120
- Community 121
- Community 122
- Community 123
- Community 124
- Community 125
- Community 126
- Community 127
- Community 128
- Community 129
- Community 130
- Community 131
- Community 132
- Community 133
- Community 134
- Community 135
- Community 137
- Community 138
- Community 139
- Community 140
- Community 141
- Community 142
- Community 143
- Community 144
- Community 145
- Community 146
- Community 147
- Community 148
- Community 149
- Community 150
- Community 151
- Community 152
- Community 153
- Community 154
- Community 155
- Community 156
- Community 157
- Community 158
- Community 159
- Community 160
- Community 161
- Community 162
- Community 163
- Community 164
- Community 165
- Community 166
- Community 167
- Community 168
- Community 169
- Community 170
- Community 171
- Community 172
- Community 173
- Community 174
- Community 175
- Community 176
- Community 177
- Community 178
- Community 179
- Community 180
- Community 181
- Community 182
- Community 183
- Community 184
- Community 185
- Community 186
- Community 187
- Community 188
- Community 189
- Community 190
- Community 191
- Community 192
- Community 193
- Community 194
- Community 195
- Community 196
- Community 197
- Community 198
- Community 199
- Community 200
- Community 201
- Community 202
- Community 203
- Community 204
- Community 205
- Community 206
- Community 207
- Community 208
- Community 209
- Community 210
- Community 211
- Community 212
- Community 213
- Community 214
- Community 215
- Community 216
- Community 217
- Community 218
- Community 219
- Community 220
- Community 221
- Community 222
- Community 223
- Community 224
- Community 225
- Community 226
- Community 227
- Community 228
- Community 229
- Community 230
- Community 231
- Community 232
- Community 233
- Community 234
- Community 235
- Community 236
- Community 238
- Community 239
- Community 240
- Community 241
- Community 242
- Community 243
- Community 244
- Community 245
- Community 246
- Community 247
- Community 248
- Community 249
- Community 250
- Community 251
- Community 252
- Community 253
- Community 254
- Community 255
- Community 257
- Community 260
- Community 261
- Community 262
- Community 263
- Community 264
- Community 265
- Community 266
- Community 274
- Community 279
- Community 280
- Community 281
- Community 282
- Community 283

## God Nodes (most connected - your core abstractions)
1. `calculate_single_material_properties()` - 130 edges
2. `kk_calculate_real()` - 114 edges
3. `MainWindowProtocol` - 97 edges
4. `parse_formula()` - 97 edges
5. `CompletionInstaller` - 69 edges
6. `kk_real_spectrum()` - 67 edges
7. `calculate_xray_properties()` - 65 edges
8. `BatchConfig` - 56 edges
9. `compound_spectrum()` - 52 edges
10. `Spectrum` - 51 edges

## Surprising Connections (you probably didn't know these)
- `Evidence` --references--> `kk_calculate_real()`  [INFERRED]
  docs/architecture/005-adr-kk-jax-only.md → xraylabtool/kk/api.py
- `Exemptions (host code by design)` --references--> `XRayResult`  [INFERRED]
  docs/architecture/006-adr-core-jax-only.md → xraylabtool/calculators/xray_result.py
- `Scope and what follows` --references--> `attenuation_length_cm()`  [INFERRED]
  docs/architecture/006-adr-core-jax-only.md → xraylabtool/constants.py
- `Scope and what follows` --references--> `critical_angle_degrees()`  [INFERRED]
  docs/architecture/006-adr-core-jax-only.md → xraylabtool/constants.py
- `Scope and what follows` --references--> `energy_to_wavelength_angstrom()`  [INFERRED]
  docs/architecture/006-adr-core-jax-only.md → xraylabtool/constants.py

## Import Cycles
- 3-file cycle: `xraylabtool/calculators/host.py -> xraylabtool/calculators/optics.py -> xraylabtool/data_handling/atomic_cache.py -> xraylabtool/calculators/host.py`
- 3-file cycle: `xraylabtool/calculators/__init__.py -> xraylabtool/calculators/derived_quantities.py -> xraylabtool/calculators/host.py -> xraylabtool/calculators/__init__.py`
- 3-file cycle: `xraylabtool/calculators/__init__.py -> xraylabtool/calculators/core.py -> xraylabtool/calculators/host.py -> xraylabtool/calculators/__init__.py`
- 4-file cycle: `xraylabtool/calculators/__init__.py -> xraylabtool/calculators/core.py -> xraylabtool/calculators/scattering_data.py -> xraylabtool/calculators/host.py -> xraylabtool/calculators/__init__.py`
- 4-file cycle: `xraylabtool/calculators/__init__.py -> xraylabtool/calculators/optics.py -> xraylabtool/data_handling/atomic_cache.py -> xraylabtool/calculators/host.py -> xraylabtool/calculators/__init__.py`
- 4-file cycle: `xraylabtool/calculators/__init__.py -> xraylabtool/calculators/core.py -> xraylabtool/calculators/cache.py -> xraylabtool/calculators/host.py -> xraylabtool/calculators/__init__.py`
- 4-file cycle: `xraylabtool/calculators/__init__.py -> xraylabtool/calculators/core.py -> xraylabtool/calculators/derived_quantities.py -> xraylabtool/calculators/host.py -> xraylabtool/calculators/__init__.py`
- 4-file cycle: `xraylabtool/__init__.py -> xraylabtool/calculators/__init__.py -> xraylabtool/calculators/core.py -> xraylabtool/logging_utils.py -> xraylabtool/__init__.py`

## Hyperedges (group relationships)
- **Watts piecewise-Laurent KK algorithm** — docs_watts_2014_calculation_of_the_kramers_kronig_transform_published_version_piecewise_laurent_polynomial, docs_watts_2014_calculation_of_the_kramers_kronig_transform_published_version_analytic_interval_integration, docs_watts_2014_calculation_of_the_kramers_kronig_transform_published_version_double_interval_integration, docs_watts_2014_calculation_of_the_kramers_kronig_transform_published_version_kronecker_delta_log_handling, docs_watts_2014_calculation_of_the_kramers_kronig_transform_published_version_f1_closed_form [EXTRACTED 0.95]
- **GUI main window tab mixins composed into MainWindow** — docs_api_gui_main_window, docs_api_gui_single_tab, docs_api_gui_multi_tab, docs_api_gui_kk_tab, docs_api_gui_shared, docs_api_gui_toast [EXTRACTED 1.00]
- **Atomic factors to refractive index to derived optical quantities** — docs_physics_atomic_data_scattering_factors, docs_physics_calculations_number_density, docs_physics_calculations_delta_beta_sum, docs_physics_calculations_critical_angle, docs_physics_calculations_attenuation [EXTRACTED 1.00]
- **Truncation error compared across C, Fe, Pb** — docs_static_kk_truncation_carbon, docs_static_kk_truncation_iron, docs_static_kk_truncation_lead [EXTRACTED 1.00]
- **Refractive index to absorption to attenuation length chain** — docs_physics_xray_optics_complex_refractive_index, docs_physics_xray_optics_absorption_coefficient, docs_physics_xray_optics_attenuation_length, docs_physics_xray_optics_beer_lambert [EXTRACTED 1.00]
- **XRayLabTool exception hierarchy rooted at XRayLabToolError** — docs_api_validation_xraylabtoolerror, docs_api_validation_validationerror, docs_api_validation_formulaerror, docs_api_validation_energyerror, docs_api_validation_calculationerror [EXTRACTED 1.00]
- **Virtual environment-centric completion system** — docs_guides_completion_guide_venv_isolation, docs_guides_completion_guide_env_detection, docs_guides_completion_guide_multi_shell, docs_guides_completion_guide_no_sudo, docs_guides_completion_guide_environment_detector, docs_guides_completion_guide_completion_installer [INFERRED 0.85]
- **JAX-first ADR lineage (ADR-001 superseded by ADR-005 for kk and ADR-006 for core, with ADR-004 transfer policy)** — docs_architecture_001_adr_jax_vs_numpy_adr_001, architecture_004_adr_host_device_transfers_adr_004_host_device_transfer_minimization_strategy, docs_architecture_005_adr_kk_jax_only_adr_005_the_kramers_kronig_package_is_jax_only, docs_architecture_006_adr_core_jax_only_adr_006_the_core_calculators_are_jax_only [EXTRACTED 1.00]
- **Device-resident compute pattern (one upload, bucketed shapes, one fetch)** — docs_architecture_006_adr_core_jax_only_one_device_program, docs_architecture_005_adr_kk_jax_only_device_resident_pipeline, docs_architecture_004_adr_host_device_transfers_shipped_flow, docs_architecture_006_adr_core_jax_only_shape_buckets_runtime [INFERRED 0.85]
- **xraylabtool.backend package modules (runtime, shapes, interpolation)** — docs_api_backend_runtime_module, docs_api_backend_shapes_module, docs_api_backend_interpolation_module [EXTRACTED 1.00]
- **Kramers-Kronig documentation set (physics, guide, CLI, changelog)** — docs_physics_kramers_kronig, docs_guides_kk_guide, docs_guides_cli_reference_kk_command, docs_development_changelog_unreleased_kk_module [INFERRED 0.85]
- **JAX-only core performance narrative across docs** — docs_development_changelog_unreleased_jax_first_core, docs_development_performance_where_time_goes, docs_guides_getting_started_architecture_overview, docs_index_key_features [INFERRED 0.85]
- **Single Material tab workflow: input, compute, plot and table** — docs_static_gui_main_offscreen_material_input, docs_static_gui_main_offscreen_property_plot, docs_static_gui_main_offscreen_results_table, docs_static_gui_main_offscreen_summary_table [INFERRED 0.85]

## Communities (287 total, 110 thin omitted)

### Community 0 - "Core Calculator Tests"
Cohesion: 0.04
Nodes (25): TestEdgeCasesAndErrorHandling, TestH2OProperties, TestInputValidationAndErrorHandling, TestSiO2Properties, _logged(), _same_bits(), spy(), test_a_bad_energy_fails_every_formula_of_the_multiple_call() (+17 more)

### Community 1 - "Golden Case Builders"
Cohesion: 0.07
Nodes (28): build_inputs(), compute_family(), _element_data(), element_density(), fam_atomic_provider(), fam_chunk_contracts(), fam_constants_helpers(), fam_crystal_structure() (+20 more)

### Community 2 - "KK Merge Spectra Tests"
Cohesion: 0.09
Nodes (47): test_host_scale_conversion_and_merge_of_tiny_data_match_scale_one(), test_merge_spectra_edge_errors_keep_legacy_type_and_text(), test_merge_spectra_exact_knot_with_overflowing_slope(), test_merge_spectra_nan_coefficients_raise_the_normalisation_error(), test_merge_spectra_unchecked_coefficients_away_from_the_merge_points_pass(), test_merge_spectra_rejects_complex_inputs(), _edge_spectrum(), test_asf_input_is_unchanged_and_unknown_type_raises() (+39 more)

### Community 3 - "Shell Completion Tests"
Cohesion: 0.04
Nodes (7): TestBashCompletionSubstitution, test_cli_aliases_registered(), BashCompletionGenerator, CompletionGenerator, FishCompletionGenerator, PowerShellCompletionGenerator, ZshCompletionGenerator

### Community 4 - "Style Guide Validation"
Cohesion: 0.05
Nodes (10): Colors, main(), StyleGuideValidator, StyleViolation, ValidationReport, _banned_module(), forbidden(), test_the_package_has_no_process_or_thread_pool() (+2 more)

### Community 5 - "JAX Backend Parity Tests"
Cohesion: 0.06
Nodes (20): TestGoldenParity, TestScalarEnergyFallback, _smart_cache_warming(), _calculate_materials(), calculate_multiple_xray_properties(), _calculate_single_material_xray_properties(), _coerce_energy_array(), _convert_energy_input() (+12 more)

### Community 6 - "Bulk Atomic Data Tests"
Cohesion: 0.05
Nodes (14): test_whitespace_delimited_loader_works_with_current_pandas(), TestAnomalousDispersion, TestAtomicDataCacheIsImmutable, TestDeviceProgramCompilation, TestExports, TestGuiErrorHandling, task(), TestLazySubmoduleAccess (+6 more)

### Community 8 - "KK Refinement Loop"
Cohesion: 0.07
Nodes (25): Decision, test_device_loop_runs_under_transfer_guard(), _accumulator_need(), _finalize(), _first_pass(), _fit_active(), _fit_columns(), _grow_accumulator() (+17 more)

### Community 9 - "KK Pipeline Device Tests"
Cohesion: 0.06
Nodes (33): _colliding_pair(), _passes(), _run(), _subprocess(), test_accelerator_block_rule_gives_the_same_result(), test_backend_field_is_jax(), test_chunk_size_is_upper_bound(), test_cold_concurrent_first_calls_upload_the_tables_once() (+25 more)

### Community 10 - "Derived Quantities Goldens"
Cohesion: 0.06
Nodes (11): fam_primitives(), TestGoldenAttenuationLength, TestGoldenCriticalAngle, TestGoldenTransmission, test_helpers_keep_their_standalone_edges(), calculate_attenuation_length(), calculate_critical_angle(), calculate_scattering_length_density() (+3 more)

### Community 11 - "GUI Headless Smoke"
Cohesion: 0.07
Nodes (23): _run_smoke(), app(), _fresh(), test_compute_runs_inside_the_worker_guard(), test_core_warm_up_failure_is_logged_not_raised(), test_multiple_materials_on_a_worker_match_direct_calls(), test_start_core_warm_up_runs_one_untracked_worker(), test_the_pool_threads_are_not_the_main_thread() (+15 more)

### Community 12 - "Runtime Transfer Guard Tests"
Cohesion: 0.07
Nodes (36): test_a_block_is_one_upload_and_one_fetch(), test_an_extra_upload_inside_the_program_call_is_rejected(), sneaky(), test_thread_counts_ignore_other_threads(), test_kk_device_and_shapes_re_export_the_promoted_names(), test_runtime_and_shapes_import_has_no_side_effect(), test_transfer_counters_live_in_runtime_only(), test_clear_tables_forces_reload() (+28 more)

### Community 13 - "KK Transform Parity"
Cohesion: 0.08
Nodes (27): test_matches_upstream_kk_pp_golden(), assert_kk_close(), kk_rounding_scale(), arrays_close(), assert_arrays_close(), additivity_energies_kev(), test_additivity(), _case() (+19 more)

### Community 14 - "PCHIP Interpolation Goldens"
Cohesion: 0.06
Nodes (25): _check_element(), cold_cache(), test_si_o_au_interpolators_are_independent(), TestPchipAu, TestPchipO, TestPchipSi, _budget(), _material() (+17 more)

### Community 15 - "KK Public API"
Cohesion: 0.06
Nodes (22): _check_refine_arguments(), _checked_merge(), _compound_capacity(), _default_grid_size(), _explicit_energies(), _host_inputs(), _log_runtime_once(), _merge_points_ev() (+14 more)

### Community 16 - "Crystal Structure Tests"
Cohesion: 0.05
Nodes (7): TestCrystalStructure, _make_cubic(), _phase(), TestCrystalStructureAPI, TestStructureFactorPhysics, CrystalStructure, _structure_factor_program()

### Community 17 - "CLI Formula Tests"
Cohesion: 0.06
Nodes (12): TestFormulaCommand, TestAtomicDataLookup, worker(), __getattr__(), cmd_formula(), _get_atomic_data(), _output_formula_results(), _print_formula_results() (+4 more)

### Community 18 - "Plot Palette Widgets"
Cohesion: 0.09
Nodes (9): apply_palette_to_item(), apply_palette_to_widget(), current_palette(), legend_kwargs(), _recolor_axis_label(), _restyle_legend(), PlotCanvas, F1F2Plot (+1 more)

### Community 19 - "Compile Count Gates"
Cohesion: 0.06
Nodes (20): _compile_counts(), worker(), call(), TestCLICompatibility, TestCLIOutputFormatting, _budget(), test_calculation_fills_the_element_cache_that_get_cached_elements_reports(), test_clear_cache_drops_the_resident_tables_and_the_next_call_reloads_them() (+12 more)

### Community 20 - "Community 20"
Cohesion: 0.06
Nodes (23): test_element_without_table_raises_unknown_element(), test_empty_formula_raises_formula_error(), test_errors_match_the_task0_catalogue(), test_matches_molecular_properties_bit_for_bit(), test_parse_errors_are_not_cached_and_repeat(), test_repeated_elements_are_aggregated_in_first_seen_order(), test_host_tables_are_rejected(), test_wrapper_budget_is_one_upload_and_one_fetch_per_call() (+15 more)

### Community 21 - "Community 21"
Cohesion: 0.09
Nodes (3): _isclose(), validate_constants(), get_log_file_path()

### Community 22 - "Community 22"
Cohesion: 0.09
Nodes (24): Amendments to the design spec, recorded here, Decision, Scope and what follows, attenuation_length_from_beta(), critical_angle_from_delta(), derive(), _derived(), _derived_program() (+16 more)

### Community 23 - "Community 23"
Cohesion: 0.07
Nodes (22): _cxro_deviation(), test_30kev_truncation_matches_paper_trend(), test_default_grid_matches_upstream_golden_bit_for_bit_inputs(), test_end_to_end_matches_upstream_golden(), test_tabulated_kk_f1_close_to_cxro_f1(), test_matches_upstream_improve_accuracy_golden(), si(), _numpy_assembly() (+14 more)

### Community 24 - "Community 24"
Cohesion: 0.07
Nodes (9): TestCompletionInstallToTempDir, completion_main(), create_completion_parser(), handle_completion_command(), show_completion_info(), handle_new_completion_command(), CompletionManager, get_global_options() (+1 more)

### Community 26 - "Community 26"
Cohesion: 0.08
Nodes (20): _evaluate_f2_device(), test_cold_concurrent_first_call_uploads_the_tables_once(), test_compound_spectrum_device_under_strict_guard(), test_evaluate_f2_device_under_strict_guard(), test_host_to_asf_scale_beta_passes_the_strict_guard(), call(), test_merge_device_under_strict_guard(), test_public_kk_f1_under_strict_guard() (+12 more)

### Community 27 - "Community 27"
Cohesion: 0.06
Nodes (5): TestBenchmarkComparison, TestPerformanceBenchmarks, TestPerformanceRegression, TestScalabilityBenchmarks, timer()

### Community 28 - "Community 28"
Cohesion: 0.04
Nodes (11): TestBasicSetupAndInitialization, TestDuplicatedFormulasIndependence, TestEdgeCaseInputValues, TestPerformanceBenchmarks, energy_sweep(), complex_formula(), multi_calculation(), single_calculation() (+3 more)

### Community 29 - "Community 29"
Cohesion: 0.08
Nodes (6): TestFormulaParsing, TestParseFormulaBasic, TestParseFormulaDecimal, TestParseFormulaErrors, TestParseFormulaParentheses, parse_formula()

### Community 30 - "Community 30"
Cohesion: 0.10
Nodes (29): _inputs(), run_program(), run_tail(), _assert_golden(), _assert_material(), _check_formulas(), _props(), _same() (+21 more)

### Community 31 - "Community 31"
Cohesion: 0.07
Nodes (15): TestResultFormatting, _format_array_field_single(), _format_as_csv(), _format_as_json(), _format_filtered_table(), _format_material_properties(), _format_multiple_energies(), _format_multiple_energy_section() (+7 more)

### Community 32 - "Community 32"
Cohesion: 0.08
Nodes (29): _derived(), _exact_reference(), _shapes(), _si(), test_all_scalar_derived_quantities_are_float_and_numpy_scalars(), test_all_zero_formula_is_skipped_by_the_collection_calls(), test_all_zero_formula_raises_zero_division(), test_all_zero_formula_with_a_numpy_density_keeps_the_non_finite_failure() (+21 more)

### Community 33 - "Community 33"
Cohesion: 0.12
Nodes (20): test_padding_does_not_change_a_bit_of_the_pchip_programs(), bucket_pow2(), warm_up(), broadcast_shape(), device_tables(), execute(), _jitted(), _pad_last() (+12 more)

### Community 34 - "Community 34"
Cohesion: 0.05
Nodes (4): TestGoldenAu, TestGoldenDerivedQuantities, TestGoldenScatteringFactors, TestGoldenSiO2

### Community 35 - "Community 35"
Cohesion: 0.06
Nodes (41): Contributing Guide, Contributing atomic data (citation, range, precision, format), Code standards (PEP 8, 88 cols, NumPy docstrings, type hints), Performance requirements (<0.1 ms single, >100k calc/s batch, >90% cache hit), Review process (automated checks, reviewer, maintainer, squash merge), Development workflow (branching, ruff, mypy, pytest, make verify), Atomic Scattering Factor Data, Absorption edges (K, L, M) and XANES features (+33 more)

### Community 36 - "Community 36"
Cohesion: 0.08
Nodes (3): TestSentinelMarkers, CompletionInstaller, test_completion_installation()

### Community 37 - "Community 37"
Cohesion: 0.14
Nodes (28): _args(), _eq10_corrected(), _f2_poly(), _golden(), _linear_segments(), _quadrature_f1(), test_chunk_invariance(), test_constant_band_closed_form() (+20 more)

### Community 38 - "Community 38"
Cohesion: 0.07
Nodes (5): TestBasicExportIntegration, TestBasicExportFunctionality, export_to_csv(), export_to_json(), __getattr__()

### Community 39 - "Community 39"
Cohesion: 0.08
Nodes (15): test_chunk_size_bounds_the_blocks_and_progress_counts_every_item(), TestProcessSingleCalculation, chunk_iterator(), _energies_key(), _filter_dataframe_fields(), _get_energy_point_data(), _initialize_progress_bar(), load_batch_input() (+7 more)

### Community 40 - "Community 40"
Cohesion: 0.10
Nodes (8): TestBatchProcessorEquivalence, test_max_workers_is_accepted_ignored_and_logged_once_at_debug(), no_progress_config(), TestBatchInputValidation, TestBatchPartialFailure, TestBatchProcessing, BatchConfig, calculate_batch_properties()

### Community 41 - "Community 41"
Cohesion: 0.11
Nodes (5): _parse_csv(), _run(), test_empty_default_grid_after_merge_exits_1(), TestKKCli, _write_edge()

### Community 42 - "Community 42"
Cohesion: 0.09
Nodes (15): _asf_f2(), test_asf_grid_spans_10ev_to_500kev(), test_asf_has_92_elements(), test_cached_tables_are_read_only(), test_im_coeffs_shape(), test_load_asf_is_cached(), test_nff_divergent_elements(), _compound_segments() (+7 more)

### Community 43 - "Community 43"
Cohesion: 0.07
Nodes (9): form(), _get_app(), qt_app(), TestDensityValidation, TestEnergyGridValidation, TestFormulaValidation, TestFormValues, TestPlotAxisLabelTheming (+1 more)

### Community 44 - "Community 44"
Cohesion: 0.07
Nodes (7): setup(), TestConvertCommand, TestUnitConversions, warm_up_core(), cmd_convert(), energy_to_wavelength(), wavelength_to_energy()

### Community 45 - "Community 45"
Cohesion: 0.11
Nodes (21): _budget(), platform(), test_block_for_examples(), test_block_for_never_exceeds_the_request(), test_bucket_is_multiple_of_8_monotone_and_bounded(), test_bucket_ladder_head(), test_bucket_pow2(), test_bucket_rejects_bad_minimum() (+13 more)

### Community 46 - "Community 46"
Cohesion: 0.07
Nodes (5): TestBackwardCompatibility, TestInstallCompletionMain, TestLegacyInstallCompletionMain, install_completion_main(), legacy_install_completion_main()

### Community 47 - "Community 47"
Cohesion: 0.13
Nodes (29): _assert_matches_reference(), _explicit_scale(), _poly(), _reference(), _run(), _status_of(), test_adjacent_float_knots_use_lower_boundary(), test_decimal_scale_of_the_near_edge_values_matches_scale_one() (+21 more)

### Community 48 - "Community 48"
Cohesion: 0.08
Nodes (9): main(), TestCheckGpuAvailability, TestGetDeviceInfo, TestGetSystemCudaVersion, check_gpu_availability(), check_plugin_conflicts(), get_device_info(), get_recommended_package() (+1 more)

### Community 49 - "Community 49"
Cohesion: 0.09
Nodes (12): check_capacity(), _flatten(), host_spectrum(), _unflatten(), validated_arrays(), _asf_scale_arguments(), _check_positive(), _fmt_range() (+4 more)

### Community 50 - "Community 50"
Cohesion: 0.10
Nodes (27): _fraction_moving_average(), _mine(), _offset_signal(), _signal(), _smooth_reference(), test_a_new_window_is_a_new_program_and_a_repeat_is_not(), test_a_signal_helper_is_one_upload_and_one_fetch(), test_arrays_larger_than_one_element_still_fail_in_float() (+19 more)

### Community 51 - "Community 51"
Cohesion: 0.11
Nodes (17): main(), _run(), padded(), loss(), f1_at(), trips(), body(), _crop() (+9 more)

### Community 52 - "Community 52"
Cohesion: 0.09
Nodes (10): fam_utils_crystal(), TestCrystallographicCalculations, angle_from_q(), bragg_angle(), _check_lattice(), d_spacing_cubic(), d_spacing_orthorhombic(), d_spacing_tetragonal() (+2 more)

### Community 53 - "Community 53"
Cohesion: 0.10
Nodes (3): MockXRayResult, TestFormatXRayResult, format_xray_result()

### Community 55 - "Community 55"
Cohesion: 0.10
Nodes (4): test_comparator_fallback_names_the_first_failing_formula(), test_comparator_keeps_the_cause_of_a_per_formula_failure(), ComparisonResult, MaterialComparator

### Community 56 - "Community 56"
Cohesion: 0.08
Nodes (3): TestCodeQuality, TestDocumentationPatterns, TestStyleCompliance

### Community 57 - "Community 57"
Cohesion: 0.06
Nodes (4): TestEnvironmentIntegration, TestEnvironmentProperties, TestEnvironmentType, EnvironmentType

### Community 58 - "Community 58"
Cohesion: 0.09
Nodes (12): ci_threshold_multiplier(), pytest_collection_modifyitems(), kk_log(), x64_restored(), test_configure_logging_writes_file(), test_get_logger_child_inherits_config(), _bool_env(), _coerce_int_env() (+4 more)

### Community 59 - "Community 59"
Cohesion: 0.12
Nodes (15): _cell(), _cli_times(), _cli_times_warm_cache(), cold_env(), _fmt(), _importtime(), machine_state(), main() (+7 more)

### Community 60 - "Community 60"
Cohesion: 0.07
Nodes (4): TestBatchConfig, TestBatchProcessingMemoryManagement, TestCacheManagement, TestMemoryLeakPrevention

### Community 61 - "Community 61"
Cohesion: 0.09
Nodes (6): TestAtomicDataError, TestExceptionHierarchy, TestUnknownElementError, AtomicDataError, UnknownElementError, XRayLabToolError

### Community 62 - "Community 62"
Cohesion: 0.10
Nodes (13): _collect_compound_elements(), _collect_family_compound_elements(), _collect_similar_compound_elements(), get_atomic_data_fast(), get_bulk_atomic_data_fast(), get_cache_stats(), is_element_preloaded(), _warm_atomic_data() (+5 more)

### Community 63 - "Community 63"
Cohesion: 0.08
Nodes (8): BasePerformanceTest, TestArrayOptimization, TestBulkAtomicDataOptimization, TestDeprecationWarningOptimization, TestElementPathOptimization, TestInterpolatorCaching, TestNumericalStabilityOptimizations, TestPerformanceRegression

### Community 64 - "Community 64"
Cohesion: 0.10
Nodes (6): apply_pyqtgraph_theme(), apply_styles(), apply_theme(), ColorPalette, get_qss(), ThemeManager

### Community 65 - "Community 65"
Cohesion: 0.14
Nodes (26): _max_midpoint_error(), _refine(), test_debug_log_per_pass_and_info_total(), test_docstring_examples_run(), test_exhaustion_flagged_and_logged(), test_invalid_refine_arguments_raise_validation_error(), test_loose_tolerance_adds_nothing(), test_lower_range_end_through_kk_calculate_real() (+18 more)

### Community 66 - "Community 66"
Cohesion: 0.09
Nodes (11): clear_lru_caches(), _mock_get_element(), TestCacheResultConsistency, TestConcurrentGetAtomicNumber, worker(), cache_clearer(), reader(), worker() (+3 more)

### Community 67 - "Community 67"
Cohesion: 0.17
Nodes (22): _host_reference(), _logged(), _narrow_last_segment_table(), _refine(), _spacing_grid(), test_both_endpoint_slivers(), test_device_matches_pre_migration_loop(), test_float_spacing_sticky_reason_and_continues() (+14 more)

### Community 68 - "Community 68"
Cohesion: 0.06
Nodes (5): TestBatchRowResilience, TestCLIErrorHandling, TestCLIInputValidation, TestCLIPerformanceAndMemory, TestFieldNameValidation

### Community 69 - "Community 69"
Cohesion: 0.11
Nodes (13): add_atomic_command(), add_batch_command(), add_bragg_command(), add_calc_command(), add_compare_command(), add_completion_command(), add_convert_command(), add_formula_command() (+5 more)

### Community 70 - "Community 70"
Cohesion: 0.19
Nodes (21): app(), _small_run(), test_energies_outside_cxro_range_reach_the_service(), test_energy_range_controls_reach_the_service(), spy(), test_exhausted_refinement_shows_a_warning_toast(), test_export_csv_columns(), test_export_csv_io_error_is_reported() (+13 more)

### Community 71 - "Community 71"
Cohesion: 0.12
Nodes (13): angle_from_q(), bragg_angle(), d_spacing_cubic(), d_spacing_orthorhombic(), d_spacing_tetragonal(), energy_to_wavelength(), linear_background(), normalize_intensity() (+5 more)

### Community 72 - "Community 72"
Cohesion: 0.13
Nodes (12): f1_reference(), _gauss_legendre(), _h(), _pv_integral(), _quadrature(), _f2_mismatch(), oracle_energies(), _reference() (+4 more)

### Community 73 - "Community 73"
Cohesion: 0.09
Nodes (7): _calc(), _clear_cache(), TestComplexFormula, TestElectronCountGolden, TestExactCapturedValues, TestMolecularWeightGolden, _calculate_molecular_properties()

### Community 74 - "Community 74"
Cohesion: 0.09
Nodes (6): TestCLIIntegration, TestMainFunction, cmd_completion(), cmd_install_completion(), cmd_uninstall_completion(), main()

### Community 75 - "Community 75"
Cohesion: 0.17
Nodes (21): test_backend_is_jax_and_chunk_size_does_not_matter(), test_boundary_energies_are_accepted_and_f2_matches_the_table(), test_default_grid_clipped_to_0p03_30_kev_and_logged(), test_docstring_examples_run(), test_empty_default_grid_after_merge_raises(), test_energies_are_copied_not_aliased(), test_energy_one_ulp_outside_the_data_range_raises(), test_energy_out_of_data_range_raises_energy_error() (+13 more)

### Community 76 - "Community 76"
Cohesion: 0.08
Nodes (13): gen(), _hermite(), shipped(), test_check_flag_fails_on_drift(), test_check_flag_passes_on_the_shipped_file(), test_element_index(), test_padded_rows_evaluate_like_the_unpadded_pchip(), test_padding_knots_strictly_increase_and_pad_values() (+5 more)

### Community 77 - "Community 77"
Cohesion: 0.09
Nodes (3): _clear_cache(), cold_cache(), clear_scattering_factor_cache()

### Community 78 - "Community 78"
Cohesion: 0.08
Nodes (4): BaseIntegrationTest, BaseUnitTest, BaseXRayLabToolTest, generate_test_materials()

### Community 79 - "Community 79"
Cohesion: 0.11
Nodes (9): ensure_complex128_array(), ensure_float64_array(), get_optimal_chunk_size(), is_complex_array(), is_energy_array(), is_real_array(), optimize_array_memory_layout(), validate_complex_array() (+1 more)

### Community 80 - "Community 80"
Cohesion: 0.14
Nodes (7): _assert_matches(), _in_process(), _subprocess(), _table(), test_cli_writes_the_persistent_compile_cache(), test_version_unaffected(), TestKKCliEndToEnd

### Community 82 - "Community 82"
Cohesion: 0.09
Nodes (3): create_batch_test_data(), ResultValidator, TestDataFactory

### Community 84 - "Community 84"
Cohesion: 0.15
Nodes (11): _computation(), _pen_colors(), plot(), qt_app(), test_plot_draws_curves_on_unequal_grids(), test_plot_follows_theme_toggle(), test_plot_hides_missing_curves(), test_plot_log_x_rule() (+3 more)

### Community 85 - "Community 85"
Cohesion: 0.15
Nodes (16): check_additivity(), check_bench(), check_f2_si_edge(), check_oracle(), check_padding(), check_transfers(), _f2_numpy(), guarded() (+8 more)

### Community 86 - "Community 86"
Cohesion: 0.09
Nodes (5): TestAtomicCommand, TestBatchCommand, TestBraggCommand, cmd_atomic(), cmd_bragg()

### Community 90 - "Community 90"
Cohesion: 0.11
Nodes (7): Context, _git(), main(), _versions(), TestBackendStubs, get_backend(), set_backend()

### Community 91 - "Community 91"
Cohesion: 0.13
Nodes (10): qt_app(), _run_worker(), test_worker_does_not_emit_finished_on_exception(), test_worker_emits_error_on_exception(), bad_fn(), test_worker_emits_result_on_success(), test_worker_injects_progress_cb_when_accepted(), test_worker_passes_kwargs() (+2 more)

### Community 92 - "Community 92"
Cohesion: 0.14
Nodes (6): TestGetCompoundFrequencyScore, TestGetElementsForCompound, TestGetRecommendedElementsForWarming, get_compound_frequency_score(), get_elements_for_compound(), get_recommended_elements_for_warming()

### Community 94 - "Community 94"
Cohesion: 0.14
Nodes (7): cmd_batch(), _convert_result_to_dict(), _parse_batch_data(), _process_batch_materials(), _save_batch_results(), _validate_batch_input(), create_batch_progress_tracker()

### Community 95 - "Community 95"
Cohesion: 0.17
Nodes (11): check_accessibility(), check_documentation_coverage(), Colors, main(), print_section(), print_status(), run_style_checks(), test_doctests() (+3 more)

### Community 96 - "Community 96"
Cohesion: 0.10
Nodes (10): assert_values_close(), expect_no_warnings(), expect_warnings(), get_system_info(), log_system_info(), skip_if_insufficient_cpu(), skip_if_insufficient_memory(), stability_test() (+2 more)

### Community 97 - "Community 97"
Cohesion: 0.15
Nodes (14): _backend_label(), check_family(), compare_arrays(), _compare_text(), _meta(), _npz(), _structure_problem(), test_active_backend_label() (+6 more)

### Community 98 - "Community 98"
Cohesion: 0.10
Nodes (3): TestEnergyError, TestBoundaryConditions, EnergyError

### Community 99 - "Community 99"
Cohesion: 0.09
Nodes (21): ADR-003: float64 Preservation Policy, Appendix: Numerical Precision Requirements by Function, Consequences, Context, Current Codebase float64 Usage, Decision, Enforcement Strategy, JAX float32 Default (+13 more)

### Community 100 - "Community 100"
Cohesion: 0.10
Nodes (12): angle_from_q(), bragg_angle(), d_spacing_cubic(), xraylabtool.device, energy_to_wavelength(), get_atomic_data(), load_atomic_data(), xraylabtool.logging_utils (+4 more)

### Community 101 - "Community 101"
Cohesion: 0.10
Nodes (10): test_parse_formula_is_cached_but_public_result_is_fresh_lists(), _aggregate_element_counts(), _expand_parentheses(), get_avogadro_number(), get_electron_charge(), get_planck_constant(), get_speed_of_light(), _handle_mendeleev_error() (+2 more)

### Community 102 - "Community 102"
Cohesion: 0.12
Nodes (3): formula(), TestFormatCalculationSummary, format_calculation_summary()

### Community 104 - "Community 104"
Cohesion: 0.10
Nodes (4): clean_cache_around(), wrapper(), wrapper(), PerformanceBenchmark

### Community 105 - "Community 105"
Cohesion: 0.13
Nodes (6): TestCalcCommand, cmd_calc(), _determine_output_format(), _print_calc_verbose_info(), _save_or_print_output(), _validate_calc_inputs()

### Community 106 - "Community 106"
Cohesion: 0.15
Nodes (11): cache_config(), test_cache_dir_is_os_appropriate(), test_cache_helper_enables_and_sets_threshold(), test_cache_helper_keeps_user_threshold(), test_cache_helper_respects_opt_out_and_existing_config(), test_cli_does_not_call_warm_up(), test_library_import_does_not_enable_cache(), test_warm_up_exported_lazily() (+3 more)

### Community 107 - "Community 107"
Cohesion: 0.11
Nodes (3): TestBasicAnalysis, compare_materials(), find_absorption_edges()

### Community 108 - "Community 108"
Cohesion: 0.15
Nodes (17): test_load_absolute_path(), test_load_csv_file(), test_load_csv_space_separated_fallback(), test_load_empty_nff_raises_data_file_error(), test_load_malformed_nff_raises_data_file_error(), test_load_nff_energy_column_monotone(), test_load_nff_values_are_finite(), test_load_nonexistent_file_raises_file_not_found() (+9 more)

### Community 112 - "Community 112"
Cohesion: 0.13
Nodes (22): Changelog (docs), Platform and Python version compatibility matrix, Removed: NumPy backend ops, executors and multiprocessing, Unreleased: JAX-only core breaking change (suggested 0.5.0), utils signal/conversion helpers moved to _physics.py with edge-case changes, v0.4.0: backend abstraction layer (NumPy/JAX), PyQtGraph GUI, v0.4.7: deep-debugging fixes and repo hygiene, Performance and Optimization (docs) (+14 more)

### Community 113 - "Community 113"
Cohesion: 0.17
Nodes (6): _query(), restore_x64(), _scipy(), TestMatchesScipy, TestPrecision, TestValidation

### Community 115 - "Community 115"
Cohesion: 0.12
Nodes (3): TestCalculationError, TestExceptionUsage, CalculationError

### Community 116 - "Community 116"
Cohesion: 0.13
Nodes (3): TestCalculationSpeedBenchmarks, calculation_worker(), TestPerformanceRegressionDetection

### Community 117 - "Community 117"
Cohesion: 0.12
Nodes (3): TestFormulaError, FormulaError, _get_valid_element_symbols()

### Community 118 - "Community 118"
Cohesion: 0.21
Nodes (11): _check_table(), _setup(), test_close_during_warm_up_does_not_hang(), test_error_path_shows_dialog_and_stays_responsive(), test_launcher_schedules_the_warm_up(), test_nearedge_run_matches_api(), test_plain_run_matches_api(), test_refined_run_matches_api() (+3 more)

### Community 119 - "Community 119"
Cohesion: 0.12
Nodes (3): AtomicDataProvider, CalculationEngine, InterpolatorProtocol

### Community 120 - "Community 120"
Cohesion: 0.12
Nodes (14): test_progress_counts_a_non_finite_item_once(), _chunk(), test_a_complex_energy_does_not_share_a_group_with_real_bytes(), test_a_material_larger_than_the_budget_runs_as_blocks_of_exactly_one(), test_a_reshaped_energy_grid_is_rejected_for_every_item(), test_a_sequence_density_in_the_middle_of_a_chunk_keeps_the_rows_aligned(), test_a_sequence_density_is_rejected_up_front_by_calculate_xray_properties(), test_a_sequence_density_is_skipped_by_the_multiple_call() (+6 more)

### Community 122 - "Community 122"
Cohesion: 0.12
Nodes (20): GUI (API docs), widgets.kk_plot (Kramers-Kronig Plot), main_window.kk_tab, xraylabtool.gui.logging_filters, xraylabtool.gui.main_window, widgets.material_form, widgets.material_table, main_window.multi_tab (+12 more)

### Community 123 - "Community 123"
Cohesion: 0.16
Nodes (7): collect(), main(), _search(), _sphinx(), test_kk_docs_build_adds_no_warnings(), test_kk_docs_doctests_pass(), test_pyproject_package_data_lists_kk_files()

### Community 124 - "Community 124"
Cohesion: 0.16
Nodes (8): _leaked_modules(), test_no_jax_with_gpu_tools(), test_no_jax_without_gpu_tools(), cmd_kk(), _fail(), _format_output(), _parse_energies(), _parse_merge()

### Community 125 - "Community 125"
Cohesion: 0.19
Nodes (11): test_a_constant_series_stays_constant_after_smoothing(), test_multi_column_background_compiles_once_per_shape_bucket(), test_one_dimensional_only_helpers_reject_other_ranks(), background_subtraction(), normalize_intensity(), _program(), _real(), _run_columns() (+3 more)

### Community 126 - "Community 126"
Cohesion: 0.10
Nodes (9): assert_memory_usage_reasonable(), assert_performance_within_threshold(), benchmark_baseline(), energy_arrays(), pytest_addoption(), pytest_collection_modifyitems(), pytest_configure(), sample_batch_data() (+1 more)

### Community 127 - "Community 127"
Cohesion: 0.16
Nodes (10): _Handler, record_compiles(), _mine(), test_a_second_tables_object_hits_the_same_compiled_program(), test_another_bucket_does_compile(), test_energy_count_inside_the_same_bucket_compiles_nothing(), test_first_call_compiles_then_same_buckets_compile_nothing(), test_one_program_call_is_one_upload_and_one_fetch() (+2 more)

### Community 129 - "Community 129"
Cohesion: 0.17
Nodes (9): test_non_finite_status_raises_the_legacy_value_error(), test_helper_delta_beta_match_closed_form(), test_helper_rejects_complex_input(), test_helper_rejects_mismatched_nonfinite_or_bad_density(), test_helper_reproduces_lookup_path(), test_negative_f1_preserved_critical_angle_zero(), test_negative_f2_kept_and_attenuation_clamped(), calculate_properties_from_factors() (+1 more)

### Community 130 - "Community 130"
Cohesion: 0.18
Nodes (14): test_database_rejects_non_integer_atomic_number(), test_evaluate_f2_rejects_complex(), test_evaluate_f2_rejects_nan_segment_edge(), test_kernel_rejects_complex_instead_of_dropping_the_imaginary_part(), test_kk_result_stores_merge_points_as_an_immutable_tuple(), test_non_string_formula_raises_formula_error(), test_refine_rejects_complex_inputs(), test_refinement_accepts_keV_values_that_collide_in_eV() (+6 more)

### Community 131 - "Community 131"
Cohesion: 0.24
Nodes (12): test_compute_kk_default_grid_cxro_matches_result_energies(), test_compute_kk_forwards_energies_kev(), test_compute_kk_forwards_refine_tolerance(), test_compute_kk_merge_points_follow_energy_unit(), test_compute_kk_rejects_unknown_energy_unit(), test_compute_kk_service_matches_api_with_nearedge(), test_compute_kk_service_matches_api_without_nearedge(), test_cxro_comparison_empty_outside_tabulated_range() (+4 more)

### Community 132 - "Community 132"
Cohesion: 0.13
Nodes (18): v0.4.8: conda recipe and xlt/xlt-gui aliases, CLI Reference (docs), batch command (--workers ignored), calc command, compare command, completion, install-completion, uninstall-completion commands, convert command, CLI energy specification (single, list, linear and log ranges, keV) (+10 more)

### Community 133 - "Community 133"
Cohesion: 0.14
Nodes (16): Analytic per-interval integration (Eqs. 3-4), Biggs and Lighthill high-energy cross-sections (to 500 keV), Conventional KK numerical methods (FFT, Euler-Maclaurin, Gaussian, TANH), Zero-to-infinity convergence conditions on polynomial coefficients, Cross et al. difference-spectrum KK method, Double-interval integration across x=E singularity (Eq. 5), Closed-form f1(E) for piecewise Laurent f2 (Eq. 7, Eq. 10), f2 polynomial form a1 x + a0 + a-1/x + a-2/x^2 + a-3/x^3 (Eq. 8) (+8 more)

### Community 134 - "Community 134"
Cohesion: 0.25
Nodes (12): _batch_argv(), _bragg_argv(), _cache_dir(), _calc_argv(), _compare_argv(), _convert_argv(), _kk_argv(), _run_cli() (+4 more)

### Community 137 - "Community 137"
Cohesion: 0.12
Nodes (15): ADR-002: PyQtGraph vs Matplotlib for GUI Plotting, Appendix: Feature Parity Checklist, Consequences, Context, Decision, Migration Details, Negative, Positive (+7 more)

### Community 138 - "Community 138"
Cohesion: 0.21
Nodes (9): _call_worker(), call(), _gate(), _kernel_worker(), main(), _recorder(), _run_worker(), _timed_ms() (+1 more)

### Community 139 - "Community 139"
Cohesion: 0.16
Nodes (16): Behaviour decisions D1-D7, kkcalc attribution (Zlib licence, Watts 2014), kk limits (range, conditioning, wrong energy unit symptom), Kramers-Kronig Transform (physics page), Accuracy validation (upstream parity, quadrature, Lorentz oscillator, CXRO comparison, conditioning), Closed-form interval sum (corrected Eq. 10, kk_piecewise_poly), kkcalc ASF data and nine elements differing from .nff (Sc, Mo, Ru, Be, Ti, Si, La, Gd, W), Kramers-Kronig dispersion relation for f1 from f2 (+8 more)

### Community 140 - "Community 140"
Cohesion: 0.19
Nodes (8): build_tables(), _drift(), main(), _row(), _same(), _symbols_by_atomic_number(), _unique_files(), load_scattering_factor_data()

### Community 141 - "Community 141"
Cohesion: 0.12
Nodes (8): batch_config_default(), batch_config_performance(), memory_monitor(), memory_tracker(), performance_thresholds(), sample_calculation_result(), test_energies_large(), test_materials()

### Community 142 - "Community 142"
Cohesion: 0.15
Nodes (3): decorator(), wrapper(), MemoryProfiler

### Community 143 - "Community 143"
Cohesion: 0.18
Nodes (11): _device_run(), pass_trace(), wrap(), spy(), _same_result(), test_accumulator_growth_path_is_bit_identical(), test_accumulator_growth_with_max_points(), test_accumulator_is_not_sized_from_max_points() (+3 more)

### Community 144 - "Community 144"
Cohesion: 0.12
Nodes (3): TestAtomicScatteringFactor, TestLoadDataFile, AtomicScatteringFactor

### Community 145 - "Community 145"
Cohesion: 0.17
Nodes (14): Shell Completion Guide, Completion caching system (~/.xraylabtool/cache), Completion for xtool alias, Completion features (command, option, path, value), CompletionInstaller, xraylabtool.interfaces.completion_v2, Conda/Mamba install and completion workflow, Environment detection (venv, conda, mamba, Poetry, Pipenv) (+6 more)

### Community 146 - "Community 146"
Cohesion: 0.25
Nodes (6): analyze_mypy_output(), check_core_modules(), check_type_definitions(), main(), run_mypy_check(), validate_cache_performance()

### Community 147 - "Community 147"
Cohesion: 0.16
Nodes (5): main(), _venv_python(), test_headless_gui_smoke_runs_offscreen(), test_backend_lazy_attribute_resolves_after_bare_import(), test_calculators_reexport_the_optics_api_lazily()

### Community 148 - "Community 148"
Cohesion: 0.22
Nodes (9): test_a_failing_formula_raises_every_time(), test_a_formula_is_encoded_once_and_the_arrays_are_read_only(), test_clearing_the_scattering_cache_clears_the_memo_and_refills_the_element_cache(), test_density_screen_accepts_positive_numbers_of_every_kind(), test_density_screen_rejects_non_positive_numbers_of_every_kind(), test_empty_formula_still_raises_formula_error(), get_cached_elements(), _encode_material() (+1 more)

### Community 150 - "Community 150"
Cohesion: 0.13
Nodes (14): ADR-004: Host-Device Transfer Minimization Strategy, Appendix: Transfer Cost Reference (CPU Backend), Array Sizes in This Workload, Batch Processing Optimization, Consequences, Context, Decision, Implementation Pattern (+6 more)

### Community 151 - "Community 151"
Cohesion: 0.22
Nodes (8): assert_ulp_close(), _step(), test_arrays_report_the_worst_entry(), test_infinities_must_be_equal_and_never_close_to_finite(), test_nan_must_match_positions(), test_shape_mismatch_and_empty(), test_within_and_beyond_the_bound(), test_zero_expected_requires_zero()

### Community 155 - "Community 155"
Cohesion: 0.15
Nodes (5): _warm(), _x64(), ensure_x64(), worker_guard(), warm_up_kk()

### Community 156 - "Community 156"
Cohesion: 0.21
Nodes (4): _json_numpy_default(), _save_dict_as_csv(), _save_results_as_csv(), _save_results_as_json()

### Community 157 - "Community 157"
Cohesion: 0.16
Nodes (13): ADR-005: The Kramers-Kronig Package Is JAX-Only, Alternatives considered, kk NumPy vs JAX benchmark evidence (CPU and RTX 4090), Consequences, kk device-resident pipeline (one upload, status fetches, one download), Evidence, JAX floor >=0.12.0.1 for golden tolerances, kk accepted deviations from acceptance gates (+5 more)

### Community 158 - "Community 158"
Cohesion: 0.16
Nodes (3): main(), truncation_deficit(), __getattr__()

### Community 159 - "Community 159"
Cohesion: 0.22
Nodes (6): _manual(), test_negative_f1_near_edge_reaches_xray_result(), test_result_module_doctests(), test_to_xray_result_accepts_range_ends(), test_to_xray_result_rejects_out_of_range_energy(), test_to_xray_result_uses_kk_f1()

### Community 163 - "Community 163"
Cohesion: 0.22
Nodes (3): TestGetCompoundFamily, _classify_family_by_elements(), get_compound_family()

### Community 167 - "Community 167"
Cohesion: 0.17
Nodes (13): Data Handling Module API Reference, data_handling.atomic_cache (92-element host metadata cache), data_handling.batch_processing, BatchConfig (chunk_size, max_workers ignored, memory_limit_gb), data_handling.compound_analysis (compound families, warming recommendations), Atomic scattering data sources (CXRO, NIST, Henke .nff, 0.03-30 keV, PCHIP), kk.warm_up (precompile default and refined programs), Batch transfer optimization: one upload and fetch per F-block (2N to 2) (+5 more)

### Community 168 - "Community 168"
Cohesion: 0.23
Nodes (13): Kramers-Kronig (kk) API Reference, load_asf / AsfDatabase (vendored kkcalc ASF tables, 10 eV-500 keV), kk function-to-equation map (Watts 2014), kk_calculate_real / KKResult (high-level entry, keV), kk_f1 (device-level f1, jit/vmap/grad compatible), kk_piecewise_poly (Eqs. 7, 10; a-3 sign erratum), kkcalc upstream (commit f331ca3, zlib licence), merge_spectra / NearEdgeSpectrum near-edge data splice (+5 more)

### Community 169 - "Community 169"
Cohesion: 0.23
Nodes (7): _public_objects(), test_docs_cite_the_paper_and_upstream_licence(), test_docs_pages_toctree_linked(), test_docs_state_the_eq10_erratum(), test_every_public_kk_name_is_documented(), test_public_docstrings_have_examples_and_references(), test_transform_docstrings_cite_paper_equations()

### Community 171 - "Community 171"
Cohesion: 0.31
Nodes (8): check_bench(), check_goldens(), check_padding(), check_transfers(), guarded(), main(), record(), _same()

### Community 172 - "Community 172"
Cohesion: 0.32
Nodes (8): _checkout(), _eval_points(), _finite(), _git(), main(), _poly(), _save(), _transform_case()

### Community 175 - "Community 175"
Cohesion: 0.26
Nodes (4): TestJaxPchipInterpolator, TestQueryHandling, TestShapePreservation, JaxPchipInterpolator

### Community 178 - "Community 178"
Cohesion: 0.24
Nodes (5): cmd_compare(), _parse_materials(), _result_to_dict(), _write_report(), _write_table()

### Community 180 - "Community 180"
Cohesion: 0.17
Nodes (12): Functions to JIT vs functions that must not be JIT-compiled, Shipped flow: validate, upload, one program, fetch, build XRayResult, Persistent compile cache for CLI and GUI only, CLI JAX policy (persistent cache for calc/batch/compare/kk; light commands never import JAX), One device program per public call (upload, program, fetch), Pure public device API (optics.optical_properties, derive, encode_formula), JAX Architecture & Design guide, Automatic differentiation (optical_properties and kk_f1 grad-compatible) (+4 more)

### Community 182 - "Community 182"
Cohesion: 0.23
Nodes (4): _finite(), main(), _rel_corr(), probe_energies()

### Community 184 - "Community 184"
Cohesion: 0.24
Nodes (6): test_refine_accepts_numpy_integer_limits(), test_type_hints_resolve_at_runtime(), _checked_inputs(), _grid_array(), refine_spectrum(), RefineOutcome

### Community 185 - "Community 185"
Cohesion: 0.21
Nodes (6): _spy_blocks(), wrap(), test_heavy_element_segment_grid_against_upstream_emulation(), test_kk_calculate_real_block_never_exceeds_chunk_size(), test_refine_block_never_exceeds_chunk_size(), _upstream_like_added()

### Community 187 - "Community 187"
Cohesion: 0.24
Nodes (11): xraylabtool.backend.interpolation (JaxPchipInterpolator), ADR-001: JAX vs NumPy Computation Backend (superseded), ArrayBackend Protocol (NumpyBackend / JaxBackend), InterpolationFactory (scipy vs JaxPchipInterpolator), PCHIP parity with SciPy pinned at 1e-12, optimization/vectorized_core.py manual SIMD heuristics, Golden tests rtol=1e-12 replacing NumPy/JAX parity tests, Removed names (ops, NumpyBackend, JaxBackend, ArrayBackend, InterpolationFactory, kernels) (+3 more)

### Community 188 - "Community 188"
Cohesion: 0.24
Nodes (11): Calculators Module API Reference, xraylabtool.calculators.cache (TableInterpolator shims), xraylabtool.calculators.core, xraylabtool.calculators.derived_quantities, xraylabtool.calculators.optics (Device Programs), X-ray refractive index n = 1 - delta - i beta, critical angle, absorption coefficient, xraylabtool.calculators.scattering_data, xraylabtool.calculators.tables (Resident Tables) (+3 more)

### Community 189 - "Community 189"
Cohesion: 0.24
Nodes (11): Validation Module API Doc, Validation best practices (validate first, catch specific exceptions), CalculationError, EnergyError, FormulaError, validate_calculation_parameters, validate_chemical_formula, validate_density (+3 more)

### Community 190 - "Community 190"
Cohesion: 0.22
Nodes (10): Accepted: single-call latency relative to NumPy, ADR-006: The Core Calculators Are JAX-Only, Consequences, Context, Gate evidence, Host-code exemptions (I/O, parse_formula, validation, find_peaks, GUI, containers), Known limitations, Accepted single-call latency trade-off vs NumPy (+2 more)

### Community 191 - "Community 191"
Cohesion: 0.22
Nodes (11): Available Scripts, Example Output:, Integration:, Maintenance Notes, Output:, Requirements:, `run_type_check.py` - Type Checking, Scripts Directory (+3 more)

### Community 194 - "Community 194"
Cohesion: 0.20
Nodes (6): kk_programs(), numerical_programs(), test_refined_sio2_compiles_few_top_level_programs(), test_energy_length_bucket_recompiles_only_the_evaluation(), test_refinement_knobs_do_not_recompile_assembly(), test_second_call_no_recompile()

### Community 196 - "Community 196"
Cohesion: 0.38
Nodes (6): _outcome(), _result(), test_kk_result_arrays_are_read_only_copies(), test_kk_result_rejects_inconsistent_state(), test_refine_outcome_arrays_are_read_only_copies(), test_refine_outcome_rejects_inconsistent_state()

### Community 200 - "Community 200"
Cohesion: 0.24
Nodes (4): _padded_table(), _rescaled(), _validate_arrays(), value_exponent()

### Community 201 - "Community 201"
Cohesion: 0.20
Nodes (9): ADR-001: JAX vs NumPy Computation Backend, Appendix: Functions that MUST NOT be JIT-compiled, Appendix: Functions to JIT-Compile (Priority Order), Consequences, Context, Decision, Negative, Positive (+1 more)

### Community 202 - "Community 202"
Cohesion: 0.22
Nodes (5): Exemptions (host code by design), _convert_atomic_number_to_int(), find_peaks(), progress_bar(), save_processed_data()

### Community 203 - "Community 203"
Cohesion: 0.29
Nodes (10): XRayLabTool GUI main window screenshot (Single Material tab, offscreen render), Energy-Wavelength converter widget (E->A, A->E), Footer controls: Log Path, Light Mode (theme toggle), Material input panel (formula SiO2, density 2.2 g/cm3, energy grid 8-12 keV, 50 points, Compute button, live validation feedback), Presets panel (material and energy preset selectors), Property plot: attenuation length vs energy for SiO2, with Log X/Log Y, Save Plot PNG, Export CSV, Results table (energy, wavelength, delta, beta, critical angle, attenuation length, mu, f1, f2, SLD), SiO2 example calculation (60.083 g/mol, 2.2 g/cm3, 30 electrons) (+2 more)

### Community 205 - "Community 205"
Cohesion: 0.29
Nodes (4): _outcome(), _same(), test_every_departure_is_a_real_difference(), test_matches_the_legacy_outcome_or_the_documented_departure()

### Community 208 - "Community 208"
Cohesion: 0.28
Nodes (9): API Reference Index, Exception hierarchy (XRayLabToolError, ValidationError, FormulaError, EnergyError, CalculationError), High-level entry points calculate_single_material_properties / calculate_xray_properties, XRayLabTool sub-package overview, Interfaces Module API Reference, CLI commands overview (calc, batch, compare, convert, kk, formula, atomic, bragg, list, completion), CLI error handling, CLI output formats (table, csv, json) (+1 more)

### Community 209 - "Community 209"
Cohesion: 0.42
Nodes (9): Unreleased: xraylabtool.kk Kramers-Kronig module added, kk command (CLI Kramers-Kronig f1), Kramers-Kronig Guide (docs), Adaptive refinement (refine_tolerance), xraylabtool kk command-line usage and units, Kramers-Kronig GUI tab workflow, kk_calculate_real (shared by Python, CLI and GUI), Near-edge file format and input types (nexafs, beta, asf) (+1 more)

### Community 211 - "Community 211"
Cohesion: 0.44
Nodes (5): _load_golden(), _merged(), test_kernel_on_merged_spectrum_matches_golden_f1_points(), test_merged_spectrum_matches_upstream_golden(), test_nearedge_result_matches_upstream_golden()

### Community 213 - "Community 213"
Cohesion: 0.22
Nodes (5): memory_test(), performance_test(), decorator(), requires_memory(), decorator()

### Community 215 - "Community 215"
Cohesion: 0.28
Nodes (6): _outside_device_helpers(), stray_conversions(), patch_method(), wrapper(), patch_numpy(), wrapper()

### Community 218 - "Community 218"
Cohesion: 0.28
Nodes (3): _format_as_csv(), _format_as_json(), _format_as_table()

### Community 219 - "Community 219"
Cohesion: 0.25
Nodes (7): Files, Maintenance, `performance_baselines.json`, `sample_materials.json`, Test Data Directory, `test_energies.json`, Usage in Tests

### Community 220 - "Community 220"
Cohesion: 0.25
Nodes (7): Backend API Reference, xraylabtool.backend.runtime, xraylabtool.backend.shapes, runtime transfer counters and transfer_guard test budgets, kk shape buckets (geometric ladder, ratio <= 1.25), Shared runtime.py and shapes.py promoted from kk (bucket, bucket_pow2, block_for), Shape-based recompilation gotcha and padding fix

### Community 221 - "Community 221"
Cohesion: 0.29
Nodes (8): 30 keV tabulation cut (dashed vertical line), C (carbon) curve, Figure: f1 overestimated by truncating f2 integration at 30 keV, Fe (iron) curve, Kramers-Kronig f2-to-f1 integral truncated to tabulated range, Pb (lead) curve, f1 truncation error (f1_trunc - f1_full) vs photon energy, log-log, Result: truncation error grows with Z (C ~1e-4, Fe ~0.1, Pb ~2.5 e/atom), flat at low energy, rises sharply near 30 keV

### Community 229 - "Community 229"
Cohesion: 0.38
Nodes (7): Testing Guide (docs), Characterization golden-value tests (202 assertions), CI test matrix: Python 3.12-3.14, Ubuntu/macOS/Windows, Test contribution guidelines and naming convention, Coverage requirements: 22% CI gate, Codecov patch check, Test categories: unit, integration, performance, characterization, physics, nearedge_example.txt (carbon K-edge NEXAFS sample spectrum)

### Community 230 - "Community 230"
Cohesion: 0.29
Nodes (3): clean_cache(), enforce_test_isolation(), setup_test_environment()

### Community 231 - "Community 231"
Cohesion: 0.38
Nodes (6): scripts/check_version.py (version consistency), scripts/gpu_check.py (GPU diagnostics), Makefile targets and CI integration of scripts, scripts/run_type_check.py (mypy wrapper), scripts/test_docs.py (documentation testing), scripts/test_install.py (install smoke test)

### Community 234 - "Community 234"
Cohesion: 0.29
Nodes (3): attenuation_length_cm(), critical_angle_degrees(), wavelength_angstrom_to_energy()

### Community 247 - "Community 247"
Cohesion: 0.40
Nodes (3): dlog(), order(), si()

### Community 249 - "Community 249"
Cohesion: 0.67
Nodes (4): Cauchy principal value integral, Atomic scattering factors f1, f2 (Henke convention), Kramers-Kronig transform (f2 to f1, Eq. 1), Complex X-ray refractive index n = 1 - delta - i beta

## Ambiguous Edges - Review These
- `Where the time goes: resident tables, one program per call` → `Performance highlights table`  [AMBIGUOUS]
  docs/index.rst · relation: conceptually_related_to

## Knowledge Gaps
- **82 isolated node(s):** `kkcalc Zlib licence (Watts, Lauk 2013)`, `Completion caching system (~/.xraylabtool/cache)`, `Completion for xtool alias`, `Completion features (command, option, path, value)`, `Conda/Mamba install and completion workflow` (+77 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 2437 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **110 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `scripts/test_docs.py (documentation testing)` connect `Community 231` to `Community 112`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `calculate_single_material_properties()` (e.g. with `fam_single_errors()` and `fam_single_input_types()`) actually correct?**
  _`calculate_single_material_properties()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `kkcalc Zlib licence (Watts, Lauk 2013)`, `Completion caching system (~/.xraylabtool/cache)`, `Completion for xtool alias` to the rest of the system?**
  _82 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Core Calculator Tests` be split into smaller, more focused modules?**
  _Cohesion score 0.039244724176231024 - nodes in this community are weakly interconnected._
- **Why does `XRayLabTool Documentation index` connect `Community 112` to `Community 145`, `Community 231`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Are the 47 inferred relationships involving `kk_calculate_real()` (e.g. with `Decision` and `Evidence`) actually correct?**
  _`kk_calculate_real()` has 47 INFERRED edges - model-reasoned connections that need verification._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Where the time goes: resident tables, one program per call` and `Performance highlights table`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._