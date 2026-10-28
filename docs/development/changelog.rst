Changelog
=========

Changes to XRayLabTool are documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Version Categories
------------------

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Now removed features
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes

v0.5.0 (2026-10-28)
--------------------

**Added**

- ``xraylabtool.kk``: Kramers-Kronig transform of the imaginary scattering factor f2 to f1 by the piecewise Laurent polynomial method of Watts (2014), a JAX port of kkcalc's ``KK_PP`` with its tabulated f2 data (10 eV-500 keV, vendored as ``data/kkcalc/asf.npz``), optional merging of measured near-edge spectra (``load_nearedge``, ``to_asf_scale``, ``merge_spectra``), adaptive refinement (``refine_spectrum``) and ``kk_calculate_real`` returning ``KKResult`` (with ``to_xray_result``). Lazily imported; ``import xraylabtool`` does not load it.
- ``xraylabtool kk FORMULA`` CLI subcommand (``--nearedge``, ``--input-type``, ``--energy-unit``, ``--merge``, ``--density``, ``--refine``, ``--energies``, ``-o``, ``--format``) with shell completion.
- GUI "Kramers-Kronig" tab: near-edge file, merge points, refinement, plot of measured/spliced f2, KK f1 and CXRO f1, CSV export.
- ``NearEdgeDataError`` (a ``ValidationError``), also exported from ``xraylabtool``.
- Docs: ``physics/kramers_kronig`` (equations, the Eq. (10) a₋₃ sign erratum, accuracy, 30 keV truncation), ``api/kk`` and ``guides/kk_guide``; ``NOTICE`` with the kkcalc Zlib attribution.
- ``xraylabtool.kk`` is JAX-only: one ``jax.numpy`` implementation, a single device-resident pipeline (one upload, one status fetch after assembly and merge, one per refinement pass, one download) over bucketed shapes, so a default refined run compiles five programs and a second call with the same buckets compiles none. See ADR-005.
- ``xraylabtool.kk.Spectrum`` (padded piecewise-polynomial f2 table, a JAX pytree) and ``xraylabtool.kk.kk_f1``: ``jit``/``vmap``/``grad``-compatible device-level f1.
- ``xraylabtool.kk.warm_up(formulas=("SiO2",), refine_tolerance=0.01)`` precompiles the default programs. The GUI warms the programs of its own Compute click on an idle worker; the CLI never warms up.
- The ``xraylabtool kk`` command and the GUI launcher enable JAX's persistent compile cache under the OS user-cache directory (``xraylabtool/jax``, compile-time threshold 0) unless ``jax_compilation_cache_dir`` is already set or ``XRAYLABTOOL_NO_JAX_CACHE=1``; the library never enables it.
- ``KKResult.refine_exhausted`` (default ``False``) is ``True`` when adaptive refinement stopped on ``refine_max_passes``, ``refine_max_points`` or float spacing before every interval met the tolerance; the ``kk`` CLI prints a warning then.
- ``scripts/kk_gpu_check.py``: manual GPU gate (oracle, additivity, transfer budget with the guard live, padding invariance, f2 at the Si edge, benchmark) for the CUDA machine.
- ``docs/architecture/005-adr-kk-jax-only.md`` (with the section "Known deviations from the acceptance gates": CPU refined run 10-20 % slower than the forced-JAX baseline, GPU small-m overhead of 2-4 ms, bare-library GPU cold start 3.7-4.4 s); ``jax_architecture`` and the kk guide describe the device pipeline.
- ``xraylabtool.calculators.optics``: the pure ``jax.numpy`` device programs behind the calculators, ``optical_properties`` and ``OpticalProps`` (``jit``/``vmap``/``grad``-compatible, x64 precondition), ``interpolate_f``, ``derive``, ``encode_formula``, and the standalone primitives ``critical_angle_from_delta``, ``attenuation_length_from_beta`` and ``transmission``.
- ``xraylabtool.calculators.tables`` (``ScatteringTables``, ``load_tables``) and ``xraylabtool/data/scattering_tables.npz``: the f1, f2 and PCHIP slopes of the 92 elements as one ``[92, 449]`` float64 table, uploaded to the device once and passed to the programs as an argument. ``scripts/gen_scattering_tables.py`` builds it from the ``.nff`` files and ``--check`` (run in the CI test job) fails on drift.
- ``xraylabtool.backend.runtime`` (``ensure_x64``, ``enable_persistent_cache``, ``worker_guard``, counted ``upload``/``fetch``/``fetch_status``) and ``xraylabtool.backend.shapes`` (``bucket``, ``bucket_pow2``, ``block_for``), promoted from ``kk`` (which re-exports them); importing either has no side effect.
- ``xraylabtool.warm_up(formulas=("SiO2",))``: public, explicit precompile of the default core programs (nothing warms up implicitly).
- ``scripts/core_gpu_check.py`` and ``make core-gpu-check``: the manual CUDA gate (goldens, P5 to P7, P11 to P13, a verdict per gate; refuses without a CUDA device).
- ``scripts/core_bench.py`` (CPU gates P1 to P4, P8 to P10, P12, P13) and the frozen golden outputs under ``tests/characterization/core/`` (generated from the NumPy path before it was deleted; ``rtol=1e-12``).
- ``docs/architecture/006-adr-core-jax-only.md`` (supersedes ADR-001; the gate table was measured on a quiet host; GPU gate P6 is recorded there as a miss) and an updated ADR-004.
- ``xraylabtool/_physics.py``: ``energy_to_wavelength``, ``wavelength_to_energy``, ``bragg_angle``, ``d_spacing_*``, ``q_from_angle``, ``angle_from_q``, ``smooth_data``, ``background_subtraction`` and ``normalize_intensity`` are pure ``jax.numpy`` programs behind the validating ``utils`` wrappers (same messages and Python float / NumPy array returns, edge-case departures listed under Changed; signal series are padded to shape buckets so lengths in one bucket share a compile). ``find_peaks``, ``progress_bar``, ``save_processed_data`` and ``parse_formula`` stay host code.
- ``xraylabtool convert`` and ``bragg`` enable the persistent compile cache like ``calc`` (``XRAYLABTOOL_NO_JAX_CACHE=1`` opts out); with a warm cache they take about 0.5 s on the CPU host and 1.0 s on the GPU host. ``--help``, ``--version``, ``formula``, ``atomic``, ``list`` and ``completion`` still import no JAX.
- GUI: the launcher warms the core programs (``gui.services.warm_up_core``) on an idle worker next to the Kramers-Kronig warm-up; the Compute services run under ``runtime.worker_guard()``.
- ``xraylabtool.calculators`` re-exports ``OpticalProps``, ``ScatteringTables``, ``encode_formula``, ``load_tables`` and ``optical_properties`` lazily (importing the package still does not load JAX).

**Changed**

- ``utils`` signal and conversion helpers (``xraylabtool/_physics.py``) keep their results but differ in a few edge cases from the NumPy implementation: float32 input is computed in float64 (results are float64; for ``energy_to_wavelength(np.float32(10))`` this corrects the float32 rounding, 1.23984170 to 1.23984172); ``smooth_data`` accepts one-dimensional data only and raises ``ValueError("Signal data must be one-dimensional")`` otherwise (it raised ``TypeError`` for an unsized input and returned a meaningless result for N-D); ``normalize_intensity(..., "area")`` on other than 1-D data raises that ``ValueError`` (was ``TypeError``); ``background_subtraction(..., "polynomial")`` with ``x`` and ``y`` of different length raises ``ValueError("x and y must have the same length")`` (was a NumPy broadcast ``ValueError`` or ``IndexError``); ``energy_to_wavelength([10.0])`` still raises ``TypeError``, with NumPy's conversion message. Complex input is rejected rather than cast: a complex scalar or array passed to a conversion, Bragg, d-spacing or q helper raises ``TypeError`` (a Python complex scalar already raised it for ``energy_to_wavelength``, ``wavelength_to_energy`` and the cubic ``d_spacing``; in the tetragonal and orthorhombic ``d_spacing`` and the ``q_from_angle`` wavelength argument, and for NumPy complex scalars and arrays, the imaginary part used to be dropped with a ``ComplexWarning``), and a complex signal array raises ``TypeError("complex signal data is not supported")`` (the signal helpers used to compute complex results); no ``ComplexWarning`` is emitted. ``smooth_data`` keeps the old window validation (integers only; ``2.9``, ``3.0`` and ``True`` raise ``TypeError``; a window of at least ``len(y)`` returns the mean of any numeric type). ``normalize_intensity`` ``"standard"`` and the mean returned by ``smooth_data`` (a window of at least ``len(y)``) sum offsets from the first sample, so a constant series is exactly constant (standardising it gives NaN, as NumPy) and data with a large offset keep their spread; windowed smoothing sums the raw values as before. Window types: ``np.uint8`` (and other unsigned NumPy integers) is now accepted, and a 1-element array such as ``np.array([3])`` now raises ``TypeError``; the multi-column linear background runs as one device call. Further edge cases: ``background_subtraction(..., "linear")`` on data of three or more dimensions raises ``ValueError("Signal data must be one-dimensional")`` (it computed an N-D result); ``linear`` on a 0-d signal and on a ``(0, k)`` array, and ``normalize_intensity(..., "area")`` on a 0-d signal, raise that ``ValueError`` (they raised ``IndexError``); ``polynomial`` on an empty ``y`` raises ``IndexError`` (it raised ``ValueError``); ``polynomial`` on N-D ``y`` raises that ``ValueError`` (it raised a broadcast ``ValueError`` for 2-D and ``TypeError`` for 3-D); ``wavelength_to_energy([...])`` raises ``TypeError`` with NumPy's conversion message (like ``energy_to_wavelength``). Unchanged: ``normalize_intensity`` ``"max"``/``"standard"`` on any shape and 0-d, and ``background_subtraction(..., "linear")`` on an ``(N, k)`` array (column-wise).
- The JAX backend's PCHIP interpolation is an in-tree ``jax.numpy`` implementation, ``backend.interpolation.JaxPchipInterpolator`` (SciPy's Fritsch-Carlson slopes, jit-compiled evaluation, NaN outside the table), replacing ``interpax``; it matches SciPy's ``PchipInterpolator`` to float64 round-off.
- Licence expression is ``MIT AND Zlib`` in ``pyproject.toml`` and ``conda-recipe/meta.yaml``; ``LICENSE-kkcalc.txt`` ships in the sdist and wheel.
- ``calculators.core.calculate_properties_from_factors(formula, energy_kev, density, f1_total, f2_total)``: new public helper computing δ, β and the derived quantities from given f1/f2 totals; the lookup path shares its code.
- GUI: the main window opens maximized.
- ``kk_calculate_real(chunk_size=...)``: ``bool``, ``float`` and ``str`` now raise ``ValueError`` (a positive integer, including NumPy integers, is required; it is an upper bound on the energies per block).
- ``refine_spectrum`` validates the spectrum table and ``chunk_size`` up front; one-point grids and sliver-only passes no longer skip validation.
- Near-edge conversion overflow now surfaces after the merge-point validation (precedence deviation from the pre-JAX pipeline: same exception type, different message; also when the data do not overlap the table).
- ``set_backend()`` has no effect on ``xraylabtool.kk``; ``KKResult.backend`` is always ``"jax"`` from ``kk`` (the legacy value ``"numpy"`` is still accepted when constructing a ``KKResult``).
- NumPy integer values are accepted for the point and chunk limits (``refine_max_passes``, ``refine_max_points``, ``chunk_size``).
- Minimum JAX is ``>=0.12.0.1`` (``pyproject.toml``, ``uv.lock``): the golden tolerances were measured on it, and releases 0.8.0 to 0.12.0 differ in the last digits of fused XLA expressions.
- ``tests/test_code_quality.py::test_naming_conventions`` ignores physical unit symbols (``eV``, ``keV``, ...) in its camelCase check; the threshold is unchanged.
- ``docs/api/kk.rst`` no longer documents ``MergeStatus``, ``MergeResult`` and ``DeviceTables`` fields twice (duplicate-object warnings).
- **Breaking:** the core calculators are one JAX float64 implementation with no NumPy twin (ADR-006). A public call does host validation, one upload, one compiled program and one download; shapes are padded to buckets so a repeat call compiles nothing. Results equal the previous NumPy path to ``rtol=1e-12``. A small single call on CPU is slower than before (about 0.45 ms against 0.08 ms; GPU about 0.6 ms), batches of 200 and 2000 formulas are faster on CPU (about 1.8x against the frozen NumPy-era row, measured on a quiet host; see ADR-006).
- ``calculate_xray_properties``, ``calculate_multiple_xray_properties``, ``calculate_batch_properties`` and the comparator run materials as blocks through one device program (a ``vmap`` over materials). ``calculate_xray_properties`` no longer sorts and restores the energy order: the output follows the input order. ``BatchConfig.max_workers`` and the CLI ``--workers`` are accepted and ignored.
- ``set_backend("jax")`` returns ``None``; any other name raises ``ValueError("the NumPy backend was removed; see ADR-006")``; ``get_backend()`` returns ``"jax"``.
- ``import xraylabtool`` no longer sets ``JAX_ENABLE_X64`` or imports JAX; ``runtime.ensure_x64()`` runs at every compute entry. ``--help``, ``--version``, ``completion``, ``formula``, ``atomic`` and ``list`` never import JAX. The ``calc``, ``batch`` and ``compare`` commands enable the persistent compile cache like ``kk`` (``XRAYLABTOOL_NO_JAX_CACHE=1`` opts out).
- ``calculate_scattering_factors(..., element_data)`` accepts ``TableInterpolator`` entries only (anything else raises ``TypeError``); with a tiny molecular weight it returns the finite value (it returned ``inf``). ``calculate_derived_quantities`` accepts plain lists (it raised ``TypeError``).
- The first public call after ``clear_scattering_factor_cache()`` only loads the tables; the hidden first-call warm-up is gone (use ``warm_up``).
- Known limitation: a molecular weight above about 4.5e307 g/mol gives ``delta = beta = 0`` with an accepted status (XLA flushes ``1 / mw`` as a subnormal); a device exception such as out-of-memory aborts the whole batch call, skip-and-log stays per formula.

**Removed**

- ``ensure_float64`` from ``xraylabtool.backend.array_ops`` (its only user was ``kk``, which now establishes x64 itself) and the NumPy twin of the kk kernel, assembly, merge and refinement.
- ``calculators.derived_quantities.RADIANS_TO_DEGREES`` (a re-imported constant; import it from ``xraylabtool.constants``).
- ``xraylabtool.backend.ops``, ``NumpyBackend``, ``JaxBackend``, ``ArrayBackend``, ``InterpolationFactory``, ``calculators.kernels`` and ``backend/array_ops.py``; the SciPy PCHIP is no longer used at runtime. ``scripts/gen_core_golden.py`` and ``scripts/gen_kk_prejax_baseline.py`` need the removed path and say which commit to check out.
- ``ProcessPoolExecutor``, ``ThreadPoolExecutor`` and ``multiprocessing`` from ``xraylabtool/`` (an AST test keeps them out), and the batch chunk stall timeout that only existed for the executors.

**Fixed**

- ``JaxPchipInterpolator`` preserves float64 when x64 is disabled after construction or in a thread-local JAX context, and rejects complex tables instead of discarding their imaginary parts.
- GUI: table headers no longer render uppercase, which turned δ, β and μ into Δ, B and M.
- GUI: the single-material plot now spans the energy-wavelength converter row and, with the multi-material plot, resizes to fit its panel instead of being forced to 720 px; the scrollbar remains for windows too short to fit it.
- GUI: toast notifications are sized after their stylesheet is applied, so the message is no longer clipped ("calculation do…").
- GUI: plot legends use the theme text color on a translucent plot-colored box, stay readable over curves and restyle on a live theme switch; the multi-material results table no longer stretches its last column.
- GUI: plot axis titles use the theme text color instead of pyqtgraph's near-white default, so they stay readable in the Light theme.

v0.4.8 (2026-10-24)
--------------------

**Added**

- Short console-script aliases ``xlt`` (for ``xraylabtool``) and ``xlt-gui`` (for ``xraylabtool-gui``); shell completion registers ``xlt`` alongside ``xtool`` for bash, zsh, fish, and PowerShell.
- ``conda-recipe/meta.yaml``: noarch conda recipe. The package is published on the ``imewei`` channel (``conda install -c imewei -c conda-forge xraylabtool``) with the core runtime dependencies and CPU-only JAX; the CUDA extras are pip/uv only.
- ``tests/unit/test_completion_aliases.py``: checks that every shell completion generator registers both CLI aliases.

**Changed**

- The README and the index, getting-started and completion guides document the conda installation.
- ``check-yaml`` excludes the Jinja-templated ``conda-recipe/meta.yaml``.

v0.4.7 (2026-10-23)
--------------------

**Added**

- ``tests/unit/test_deep_debug_regressions.py``: regression tests for each defect fixed from the deep-debugging review (lazy imports, energy contract, anomalous dispersion, cache immutability, data integrity for all ``.nff`` files, exports, backend state, batch ordering/timeouts/fallback, parser, validators, repeated materials, pandas loader, GUI error slots and worker, utility validation).
- Python 3.12 in the CI test matrix (Ubuntu, macOS, Windows); the 3.12 suite passes (740 tests) against the locked dependencies.
- ``tests/unit/test_jax_backend_parity.py``: JAX-vs-NumPy parity tests (SiO2, H2O, Fe2O3; ``rtol=1e-12``), an ``InterpolationFactory.create_pchip`` check against ``interpax``, and tests for the ``float()`` fallback in the two scalar-energy converters.
- ``scripts/check_version.py`` (checks that the nine files recording the version agree; ``--expect``, ``--installed``, ``--print``), ``scripts/test_install.py`` (installs a wheel or a PyPI/TestPyPI release into a fresh venv and runs one calculation) and ``scripts/gpu_check.py`` (JAX backend, devices and CUDA plugin conflicts); the ``check-version`` pre-commit hook; ``make test-characterization`` and ``make test-gui``.

**Changed**

- ``jax``/``jaxlib`` no longer have an upper bound (``>=0.8.0``); the ``<0.11`` ceiling added in 0.4.5 is removed from ``pyproject.toml`` and from the ``cuda12-local``/``cuda13-local`` Makefile targets. The suite and a JAX-vs-NumPy comparison pass on 0.12.0.1 (maximum relative difference 2.5e-16), and ``uv.lock`` moves ``jax``/``jaxlib`` from 0.10.2 to 0.12.0.1, matching the locked CUDA plugin packages.
- ruff is pinned to 0.17.2 in ``.pre-commit-config.yaml``, ``ci.yml`` and ``security.yml``, matching ``uv.lock``. ``PLR0917`` (stable in ruff 0.17) is ignored alongside ``PLR0913`` instead of making public signatures keyword-only.
- Refreshed ``uv.lock`` dependency versions (including ``typer`` 0.27.3).
- Machine-readable output (``format_xray_result`` CSV/JSON, ``xraylabtool.io``) now keeps full precision; ``precision`` applies to table output only.
- ``export_to_csv`` raises ``ValueError`` for results with different numbers of energies, and a scalar field (for example ``density_g_cm3``) has one column named after the field instead of one per energy.
- ``energy_to_wavelength``, ``wavelength_to_energy`` and the ``d_spacing_*`` functions raise ``ValueError`` for non-positive energies, wavelengths and lattice parameters and for Miller indices (0, 0, 0) instead of returning negative or infinite values; ``parse_energy_string`` raises ``ValueError`` for unknown spacing (only ``linear``, ``lin`` and ``log``) and for extra ``:`` fields.
- ``pyproject.toml`` matches the code: unused dependency groups are gone (API server, scikit-learn/statsmodels, profilers, Jupyter, seaborn/plotly, openpyxl/jinja2, pbr, codecov and others; ``uv.lock`` has 185 packages), ``PySide6``/``pyqtgraph``/``matplotlib`` and ``psutil`` are documented as intentional core dependencies, ``setuptools>=77.0`` is the only build requirement, and the ruff/mypy/pre-commit floors equal the CI pins. Only the 9 markers that tests use are registered (``--strict-markers`` stays on), ``testpaths`` is ``tests``, ``mypy`` checks the package only and replaces the deprecated ``strict_concatenate`` with ``extra_checks``, and the ruff ``per-file-ignores`` list only codes that fire on the files they match.
- ``.gitignore`` rebuilt in sections with anchored directory rules so a source package named ``cache/`` or ``lib/`` cannot be ignored by accident. ``graphify-out/`` tracks ``GRAPH_REPORT.md``, ``graph.html`` and ``graph.json``; the other graphify files (manifest, cost, root/interpreter paths) are untracked.
- ``.pre-commit-config.yaml``: ``pre-commit-hooks`` v6.0.0, ``gitleaks`` v8.30.1, the ``ruff-check`` hook id, generated ``graphify-out/`` and ``docs/_build/`` excluded, and the version-consistency hook.
- Makefile: runs against ``./.venv`` without activation; category test targets pass ``--no-cov`` and ``--run-performance`` where the conftest gate requires it; GUI tests (all four files CI ignores) run serially after the parallel run; ``build`` uses ``uv build`` plus ``twine check``; ``version-check``, ``test-install-*`` and ``gpu-*`` call the new scripts; the GPU install targets use ``uv sync --extra gpu_cuda12|gpu_cuda13``; ``verify`` also checks formatting and fails on type errors; ``release-check`` adds the doctest build and a clean-venv wheel install. ``make claude`` enforces a 27% coverage floor (measured 28.7%) instead of 42%.

**Removed**

- ``safety`` dev dependency, its ``[tool.safety]`` table and its steps in the ``security.yml`` and ``dependencies.yml`` workflows. ``pip-audit`` is now the only dependency vulnerability gate. The lock also drops the 11 packages only ``safety`` required (``nltk``, ``authlib``, ``marshmallow``, ``tenacity``, ...).
- ``repomix-output.xml`` (a 9.6 MB snapshot of the repository from 2026-09-17, now out of date), ``scripts/create_favicon.py`` (unreferenced one-off generator) and the root-level ``.graphify_uncached.txt`` (a graphify temp file that was committed by mistake and contained another machine's absolute paths).
- The 2026-04 audit reports ``docs/audit_remediation_plan.md``, ``docs/security_audit_cicd.md`` and ``docs/test_coverage_audit.md``, which were not part of the documentation build, and their entries in ``docs/conf.py`` ``exclude_patterns``.
- Dead ruff ``per-file-ignores`` for files that no longer exist (``xraylabtool/core/*.py``, ``**/interfaces/completion.py``).
- ``docs/_static/logo-light.jpg``, a copy of the logo that nothing referenced (the docs theme uses ``logo-light.svg`` and ``logo-dark.svg``).

**Fixed**

- OSTI GitLab mirror: tags are force-pushed in a separate command, so a re-tagged release (such as ``v0.4.6``) no longer fails the job with "tag already exists".
- ``mypy xraylabtool/`` reports no errors on mypy 2.4.0 and on the CI-pinned 1.20.0 (23 errors fixed: generic ``dict`` annotations, batch executor typing, a list rebound to an ndarray in ``analysis``, stale ``type: ignore`` comments).
- ``test_documented_import_examples`` executed prose from ``CLAUDE.md`` ("so ``import xraylabtool`` stays fast; ...") and always failed; it now matches only import statements and also scans ``README.md``.
- ``xraylabtool.utils``, ``.constants``, ``.validation``, ``.analysis``, ``.io``, ``.calculators`` and ``.data_handling`` raised ``RecursionError`` when first accessed as attributes of the package in a fresh interpreter (``import xraylabtool; xraylabtool.utils``); the lazy loader now uses ``importlib``, and the lookups for the nonexistent ``cleanup``/``export`` modules are gone.
- Energy input: 2-D arrays silently produced wrong delta/beta/f1/f2, 0-d NumPy arrays crashed, complex energies lost their imaginary part (silently in the batch API for scalars), and NaN raised a generic ``ValueError``. Energies must now be a scalar or 1-D real array; otherwise ``EnergyError``. ``XRayResult.energy_kev`` is a copy and no longer aliases the caller's array.
- Negative delta near absorption edges (anomalous dispersion) is no longer rejected as "physically unrealistic": 20 elements, including C, Na, Ca, Ag and Pb, failed at the 0.03 keV limit. The critical angle is 0 where delta < 0; absorption and attenuation are unaffected.
- ``get_bulk_atomic_data`` returned a shared mutable dict, so editing it changed later calculations in the process and ``clear_scattering_factor_cache()`` did not undo that; the mapping is read-only and the cache clear now includes the inner loader.
- ``data/AtomicScatteringFactor/f.nff``: first row energy was ``129.3`` instead of ``29.3``, which corrupted fluorine f1/f2 near 129 eV after sorting.
- Exports: wide CSV shifted values into the wrong columns for unequal energy grids or scalar fields; dictionary CSV split strings into characters and depended on key order; a dict of ``XRayResult`` was written as its ``repr()`` in CSV and JSON; CSV/JSON formatters rounded to a fixed number of decimals, turning small beta and SLD values into 0.
- ``set_backend()`` left backend-specific interpolators cached, so the JAX backend could keep using scipy interpolators (and the reverse) depending on call order.
- Batch API: results were returned in completion order (thread path), the documented per-calculation timeout could never fire, and a process pool that cannot start (for example without semaphores) aborted the batch instead of falling back to threads.
- JAX scatter kernel recompiled for every distinct density/material because ``common_factor`` was a static argument.
- ``parse_formula``: invalid text inside a group (``Si(O-2)``, ``Si(O2+)``, ``Si()``) was silently dropped, and small fractional counts inside a group (``(H0.00001)O``) failed after expansion.
- Repeated formulas with different densities overwrote each other in ``calculate_xray_properties``, ``calculate_multiple_xray_properties`` and the GUI service; later entries now get a ``formula@density`` key.
- ``validate_chemical_formula`` rejected Ac, Fr, Pa, Ra, Th and U although scattering data exists for them.
- ``scattering_data.load_data_file`` failed on pandas 3 (``delim_whitespace`` was removed).
- GUI: both calculation-error slots raised ``KeyError`` while logging (reserved ``message`` field), so the error dialog never appeared; ``CalculationWorker`` crashed on ``functools.partial``/builtin callables and mistook a local variable named ``progress_cb`` for a parameter.
- The ``perf-baseline``, ``perf-compare`` and ``perf-report`` Makefile targets pointed at ``tests/test_integration.py``, which moved to ``tests/integration/test_integration.py``, so they collected nothing.
- Makefile targets that could not work: ``info`` and ``test-install-*`` held multi-line ``python -c`` strings that the shell split; ``gpu-diagnose`` ran a ``for`` loop after a semicolon (a syntax error hidden by ``2>/dev/null``); ``install-jax-gpu`` read an ``nvidia-smi`` error message as the compute capability; ``type-check*`` and ``claude`` printed "mypy not available" when mypy failed; ``claude`` ignored ruff failures and its performance phase collected nothing without ``--run-performance``; ``test-memory``, ``test-edge``, ``test-ci``, ``test-nightly`` and ``test-smoke`` selected markers that no longer exist; ``upload`` used ``read -p`` under ``sh``; ``clean`` descended into ``.venv``.
- ``mypy xraylabtool/ --strict`` reports no errors (``MaterialComparator.__init__`` annotated; untyped ``jax.config.update`` covered by ``untyped_calls_exclude``).
- Four stale performance tests: ``MemoryMonitor`` only catches ``psutil.Error``, batch results keep repeated formulas under ``formula@density`` keys, and ``get_bulk_atomic_data`` returns a read-only mapping.
- Ruff reported RUF001/RUF003/PLC2401 on ``docs/conf.py`` and the example notebooks when run on the whole repository (as pre-commit does); ``docs/**`` now ignores those typography rules.
- ``scripts/test_docs.py`` (``make docs-test-all``) ran Sphinx on the nonexistent ``docs/source`` and so failed every build step; it now uses ``docs/``. The old Makefile guard (``command -v python scripts/test_docs.py``) was always false, which hid this by running the plain ``docs-test``/``docs-doctest``/``docs-linkcheck`` fallback instead.

**Security**

- ``filelock`` 3.32.7 to 4.2.4 (CVE-2026-112940, fixed in 4.2.1; build/dev tooling only). The update was blocked by ``safety``, which required ``filelock<4``.

**Documentation**

- API reference now covers the CLI ``parser``, ``formatting`` and ``utils`` modules and the GUI main-window tab, widget, style and table-formatter modules; the backend classes, ``get_backend``, ``set_backend`` and ``InterpolationFactory`` have docstrings.
- Testing guide: replaced the unsupported ">95% coverage" targets with the actual CI gate (22%) and the Codecov project/patch behavior, and corrected the CI triggers and Python matrix. Contributing guide: ``make verify`` and where the ruff version is pinned. Getting started: supported JAX versions.
- Python code blocks in Markdown formatted with ruff 0.17.2.

v0.4.6 (2026-09-17)
--------------------

**Changed**

- ``xraylabtool.interfaces.cli`` is now a package (``parser``, ``formatting``, ``utils``, one module per command under ``commands/``); ``main()`` and prior public names are re-exported, so the console-script entry point and existing imports are unaffected
- ``xraylabtool.gui.main_window`` is now a package of per-tab mixins (``single_tab``, ``multi_tab``, ``shared``, ``toast``) combined into ``MainWindow``
- Batch calculation error-path warnings now go through the structured logger instead of ``print()``
- ``psutil`` failures in batch processing are caught as ``psutil.Error`` instead of a bare ``Exception``
- Internal refactors (no behavior change): ``parse_formula``, ``save_calculation_results``, ``warm_cache_for_compounds``, ``get_compound_family``, ``MaterialComparator``, ``cmd_compare``, and the Zsh/Fish completion generators split into smaller helpers

**Removed**

- Legacy ``xraylabtool.interfaces.completion`` bridge module; import from ``xraylabtool.interfaces.completion_v2`` instead
- Automated ``release.yml`` workflow; releases are now bumped/tagged by hand and published via ``publish.yml`` (see ``.github/workflows/README.md``)

**Fixed**

- CI dependency audit no longer fails on the unpatched ``nltk`` CVE (PYSEC-2026-3740) pulled in by ``safety`` itself

**Documentation**

- Correct energy units (keV), argument order ``(formula, energy_keV, density)``, CLI flags, install extras, and removed references to non-existent functions/constants across guides, API reference, notebooks, and developer docs
- CLI reference rewritten from actual ``--help`` output

v0.4.5 (2026-08-12)
--------------------

**Fixed**

- GUI now applies the persisted/default theme immediately on launch instead of staying unstyled until the first theme toggle
- Light theme is now a real palette (``LIGHT_INDUSTRIAL``) tuned for WCAG contrast on white, instead of aliasing the dark palette's neon accents
- GUI layout: consistent control spacing in the converter/presets/energy-range rows, the converter row no longer stretches full-width, and the multi-material table has a minimum height

**Changed**

- ``jax``/``jaxlib`` dependency bounds constrained to ``<0.11`` to avoid untested upstream releases

v0.4.4 (2026-06-25)
--------------------

**Added**

- Python 3.14 support: ``pyproject.toml`` classifier and CI matrix coverage (ubuntu/macOS/windows)

**Fixed**

- ``parse_formula`` now rejects trailing/invalid characters (e.g. ``SiO2junk``, ``Si-O2``) instead of silently ignoring them
- Complex energy inputs are rejected with a clear ``EnergyError`` rather than being coerced to their real part
- Empty energy arrays now raise ``EnergyError`` instead of producing empty results
- Batch processing disambiguates colliding result keys so no result is silently dropped
- ``calculate_scattering_length_density`` reconciled with the pipeline kernel (SLD = 2π·δ/λ²; real part positive for δ > 0)
- ``validate_energy_range`` defaults corrected to the supported 0.03–30 keV range
- GUI: density spinbox range aligned with ``validate_density`` (0.001–30 g/cm³); non-numeric density cells are skipped instead of crashing
- GUI "Save PNG" now exports the plot via pyqtgraph's ``ImageExporter`` (the PyQtGraph widgets have no matplotlib figure, so the previous ``savefig`` path always errored)
- ``xraylabtool batch`` no longer crashes on the default non-progress path, where the progress tracker yielded ``None``

**Changed**

- Removed unused ``EnhancedValidator`` / ``ErrorRecoveryManager`` machinery from the batch CLI path
- Migrated class-scoped pytest fixtures to ``@classmethod`` for pytest 9 compatibility

v0.4.3 (2026-05-08)
--------------------

**Changed**

- Upgraded pip to >=26.1 in CI to resolve CVE-2026-6357
- Expanded ``.gitignore`` with comprehensive uv/venv, CI, and tooling patterns
- Updated Sphinx API docs to cover all public modules (calculators, gui, interfaces, utils)
- Fixed Sphinx build output path in README development commands

**Security**

- CI: pin ``pip>=26.1`` in dependency-audit job to prevent CVE-2026-6357 false positives

v0.4.2 (2026-04-30)
--------------------

**Changed**

- UI/UX Pro Max upgrade applied to PySide6 interface
- Upgraded form validation to use non-blocking toast notifications instead of blocking message boxes
- Enhanced touch target sizes for all inputs and buttons
- Standardized UI spacing and interactive states (loading buttons, disabled opacities)

v0.4.1 (2026-04-07)
--------------------

**Fixed**

- Version bump for patch release

v0.4.0 (2026-04-07)
--------------------

**Added**

- Backend abstraction layer (``xraylabtool.backend``) supporting NumPy and JAX backends
- ``set_backend("jax")`` / ``set_backend("numpy")`` for runtime backend switching
- ``InterpolationFactory`` for backend-agnostic PCHIP interpolation (scipy ↔ interpax)
- PyQtGraph-based interactive plotting widgets (replaces matplotlib in GUI)
- JAX float64 auto-configuration at package import time
- 202 characterization tests with golden-value assertions for migration safety
- Optional ``[jax]`` dependency group: ``pip install xraylabtool[jax]``
- Optional ``[plots]`` dependency group for matplotlib (publication plots)
- Architecture Decision Records in ``docs/architecture/``
- Security scanning workflow and license compliance auditing

**Changed**

- All core computation modules now use backend ops instead of direct NumPy
- GUI plotting migrated from matplotlib to PyQtGraph for interactive performance
- ``ColorPalette.mpl_cycle`` renamed to ``ColorPalette.plot_cycle``
- ``apply_matplotlib_theme()`` replaced with ``apply_pyqtgraph_theme()``
- ``scipy.constants`` eliminated from utils.py (uses constants.py directly)
- matplotlib moved from required to optional dependency
- ReadTheDocs migrated to uv-based builds on Ubuntu 24.04 / Python 3.13
- CI updated to Python 3.13, latest GitHub Actions with SHA-pinned versions

**Deprecated**

- ``optimization/vectorized_core.py`` manual SIMD heuristics (use JAX backend instead)
- ``optimization/optimized_core.py`` monkey-patching (use ``set_backend('jax')``)

**Removed**

- ``ScalarFriendlyArray`` numpy subclass (replaced with plain numpy arrays)
- Direct matplotlib dependency (now optional via ``[plots]`` extra)
- Direct scipy.constants usage in computation path

v0.3.0 (2025-12-15)
--------------------

**Added**

- GUI modernization: dark mode toggle with persistent preferences
- New theme engine supporting light and dark themes

**Security**

- Replaced python-jose with PyJWT to address Minerva attack vulnerability

**Fixed**

- Plot clipping issues with scrollbars
- Single-point plot visibility in energy sweeps
- Log path toggle contrast
- GUI smoke test made non-blocking for CI reliability

**Maintenance**

- Removed obsolete workflows (security.yml, performance-monitoring.yml, dependabot.yml)
- Simplified lint job to use ruff only (removed isort/black)
- Updated tool versions in CI
- Added explicit permissions to all GitHub Actions workflows

v0.2.x (2025)
--------------

See the full history in `CHANGELOG.md <https://github.com/imewei/pyXRayLabTool/blob/main/CHANGELOG.md>`_
for detailed v0.2.0 through v0.2.7 release notes covering:

- Enhanced CLI with shell completion support
- Batch processing from CSV files
- Performance benchmarking suite (150,000+ calculations/sec)
- Cross-platform shell completion (bash, zsh, fish, PowerShell)
- Mamba support and environment isolation

v0.1.0 (2024)
--------------

**Added**

- Core X-ray optical properties calculations
- Support for single materials and compounds
- Complex refractive index calculations (delta, beta)
- Critical angle and attenuation length calculations
- Chemical formula parsing
- Energy-wavelength conversions
- Basic command-line interface
- Atomic scattering factor data for 92 elements

Compatibility Matrix
--------------------

.. list-table:: Platform and Python Version Support
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Version
     - Python 3.12
     - Python 3.13
     - Windows
     - macOS/Linux
   * - v0.4.1
     - ✅
     - ✅
     - ✅
     - ✅
   * - v0.4.0
     - ✅
     - ✅
     - ✅
     - ✅
   * - v0.3.0
     - ✅
     - ✅
     - ✅
     - ✅
   * - v0.2.x
     - ✅
     - ✅
     - ✅
     - ✅
   * - v0.1.0
     - ✅
     - ❌
     - ✅
     - ✅
