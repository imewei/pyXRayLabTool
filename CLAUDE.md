# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

XRayLabTool is a Python package for calculating X-ray optical properties of materials using CXRO/NIST atomic scattering factor databases. It provides both a Python API and CLI for synchrotron/X-ray science applications.

- **JAX-first**: the core calculators, the resident scattering tables, batch and the `utils` physics/signal helpers (`xraylabtool/_physics.py`) run as one `jax.numpy` float64 implementation (no NumPy twin, ADR-006); the GUI warms the core programs on an idle worker (`gui/services.warm_up_core`)
- **Python**: >=3.12
- **Package Manager**: uv (preferred), pip
- **Version**: See `pyproject.toml` and `xraylabtool/__init__.py`

## Common Commands

### Development Setup
```bash
uv sync                            # Full dev install (all runtime + test deps are core deps)
uv sync --extra gpu_cuda13         # or gpu_cuda12 (declared conflicting; syncing one swaps the other)
make dev-setup                     # Same as `make install` + quick-command summary
```

### Testing
```bash
pytest tests/ -v                   # All tests
pytest tests/ -v --no-cov          # Fast (no coverage)
pytest tests/unit/ -v              # Unit tests only
pytest tests/integration/ -v       # Integration tests only
pytest tests/performance/ -v --run-performance --no-cov  # Perf tests are skipped without the flag
pytest -m "unit" -v                # By marker
make test-fast                     # Quick run: no coverage, GUI tests serial
make claude                        # Format+lint (auto-fix), mypy --strict, tests with coverage floor, perf phase
make verify                        # Read-only checks (fails on mypy errors and unformatted code)
make gpu-check                     # JAX backend / CUDA report (scripts/gpu_check.py)
JAX_PLATFORMS=cpu pytest tests/ --no-cov   # Pin JAX to the CPU (CI has no GPU; every test must pass on CPU and on a CUDA host)
python scripts/gen_scattering_tables.py --check   # Fails if data/scattering_tables.npz drifted from the .nff files
```

### Code Quality
```bash
ruff check xraylabtool/ tests/ --fix   # Lint with auto-fix (ruff also handles import sorting)
ruff format xraylabtool/ tests/        # Format code
mypy xraylabtool/ --strict             # Type checking
python scripts/run_type_check.py --target core  # Enhanced type check
```

### Documentation
```bash
sphinx-build -b html docs docs/_build/html   # Build docs
make docs-serve                              # Build and serve locally
```

### CLI Testing
```bash
xraylabtool --version
xraylabtool calc SiO2 -e 10.0 -d 2.2
xraylabtool --help
xraylabtool-gui                   # PySide6 GUI (aliases: xtool-gui, xlt-gui)
```

## Architecture

### Source Layout (`xraylabtool/`)

```
xraylabtool/
├── __init__.py           # Lazy __getattr__ exports, public API
├── constants.py          # Physical constants and unit factors
├── utils.py              # parse_formula(), energy/wavelength conversions
├── exceptions.py         # Exception hierarchy (XRayLabToolError root)
├── device.py             # GPU/CUDA detection, plugin-conflict checks
├── logging_utils.py      # Structured logging; XRAYLABTOOL_LOG_* env vars
├── typing_extensions.py  # FloatLike / ArrayLike aliases
├── backend/              # JAX runtime helpers (the NumPy backend and `ops` were removed, ADR-006)
│   ├── __init__.py       # set_backend()/get_backend() stubs only: "jax" is a no-op, any other name raises ValueError
│   ├── runtime.py        # ensure_x64(), enable_persistent_cache(), worker_guard(), counted upload()/fetch()/fetch_status()
│   ├── shapes.py         # bucket(), bucket_pow2(), block_for(): static-shape buckets that bound compile counts
│   └── interpolation.py  # JaxPchipInterpolator (in-tree jnp PCHIP, pinned to SciPy at 1e-12)
├── calculators/          # Core X-ray physics
│   ├── core.py           # calculate_single_material_properties(), calculate_xray_properties()
│   ├── xray_result.py    # XRayResult dataclass
│   ├── tables.py         # ScatteringTables pytree, load_tables(): the 92 elements' f1/f2/PCHIP slopes, resident on the device
│   ├── optics.py         # Pure jnp device programs: optical_properties(), derive(), encode_formula() (jit/vmap/grad-compatible)
│   ├── host.py           # Host wrappers: pad to buckets, one upload, one program, one fetch, slice padding
│   ├── scattering_data.py# f1/f2 lookup at requested energies
│   ├── cache.py          # TableInterpolator shims over the resident tables; get_cached_elements(), clear_scattering_factor_cache()
│   └── derived_quantities.py  # Critical angle, attenuation, SLD, calculate_scattering_factors()/calculate_derived_quantities()
├── data_handling/
│   ├── atomic_cache.py   # 92-element preload, warm_up_cache(), get_cache_stats()
│   ├── compound_analysis.py  # Compound families, recommended elements for warming
│   └── batch_processing.py   # BatchConfig, calculate_batch_properties(), CSV load/save
├── interfaces/
│   ├── cli/              # parser.py, formatting.py, utils.py, commands/<one file per subcommand>
│   └── completion_v2/    # Virtual-environment-centric shell completion
├── validation/validators.py  # validate_chemical_formula/energy_range/density/calculation_parameters
├── analysis/comparator.py    # MaterialComparator
├── kk/                   # Kramers-Kronig f2→f1 (kkcalc port, Watts 2014; JAX-only): api, _pipeline, _device, _shapes, transform, spectrum, nearedge, refine, result, database
├── io/                   # file_operations.py, data_export.py
├── gui/                  # PySide6 app: main_window/ (per-tab mixins incl. kk_tab), widgets/, services.py (Qt-free compute), workers.py, theme_manager.py, style.py
├── data/AtomicScatteringFactor/  # .nff data files (CXRO), source of truth for the tables
├── data/scattering_tables.npz    # Stacked f1/f2/slope tables [92, 449]; rebuild with scripts/gen_scattering_tables.py (--check fails on drift)
└── data/kkcalc/          # asf.npz (upstream ASF.json, 10 eV-500 keV) + Zlib LICENSE-kkcalc.txt
```

### Key Design Patterns

1. **Lazy Imports**: `__init__.py` uses `__getattr__` so `import xraylabtool` stays fast; heavy modules load on first attribute access
2. **XRayResult Dataclass**: All calculations return this with snake_case fields; every per-energy field is an array (index `[0]` for scalars). Legacy CamelCase accessors still work with `DeprecationWarning`
3. **Argument order**: `calculate_single_material_properties(formula, energy_keV, density)` and `calculate_xray_properties(formulas, energies, densities)` — energy always in keV
4. **JAX-first core (ADR-006)**: one `jax.numpy` float64 implementation on JAX's default device (CPU, or the GPU of a CUDA install); there is no NumPy twin, no `ops` dispatch and no auto-select. `set_backend("jax")` is a no-op, any other name raises `ValueError`, `get_backend()` returns `"jax"`. A public call does host validation, one `runtime.upload`, one compiled program, one `runtime.fetch` (ADR-004); shapes are padded to buckets (`backend/shapes.py`) so compiles stay bounded. `runtime.ensure_x64()` runs at every concrete public compute entry; `import xraylabtool`, `--help`, `--version`, `completion`, `formula`, `atomic`, `list` never import JAX (compute modules import `optics`/`tables`/`runtime` inside functions). `optics.optical_properties` is the public pure device function
5. **Batch is a `vmap` over materials**: `calculate_xray_properties`, `calculate_multiple_xray_properties`, `data_handling.batch_processing.calculate_batch_properties` and the comparator group materials by element-count bucket and run blocks of materials through the same program (`BatchConfig.chunk_size` bounds the block; `max_workers` is accepted and ignored). There are no process or thread pools anywhere in `xraylabtool/` (an AST test enforces it); output order equals input order; a failing formula is logged and skipped, a device error aborts the call
6. **Scattering tables and warming**: `data/scattering_tables.npz` is read and uploaded once (`load_tables()`, under a lock) and passed to programs as an argument. The first public call only loads the tables; `xraylabtool.warm_up(formulas=("SiO2",))` is the explicit precompile helper (no hidden warm-up). `cache.py`'s `TableInterpolator` keeps `create_scattering_factor_interpolators()` working as a thin reader of a table row; `data_handling.atomic_cache.warm_up_cache()` / `warm_cache_for_compounds()` only front-load the host atomic-data cache
7. **Kramers-Kronig (`xraylabtool.kk`)**: lazy, not imported by `import xraylabtool`; **kk is JAX-only** (one `jax.numpy` implementation in float64, no NumPy twin; `set_backend()` does not affect it, ADR-005). `kk_calculate_real` is a device-resident pipeline (`_pipeline.py`, `refine.py`; helpers and `Spectrum` in `_device.py`): one upload, one status fetch after assembly/merge, one per refinement pass, one download, over bucketed shapes (`_shapes.py`). Public API in keV, transform in eV; `Spectrum`/`kk_f1` are jit/vmap/grad-compatible. `kk.warm_up()` is public API (precompiles a formula's default and refined programs); the GUI warms its own Compute path on an idle worker via `gui/services.warm_up_kk`, the CLI never warms up; the CLI `calc`/`batch`/`compare`/`kk` commands and the GUI launcher enable the persistent compile cache via `runtime.enable_persistent_cache()` (`XRAYLABTOOL_NO_JAX_CACHE=1` opts out; the suite sets it in `tests/conftest.py`). Uses its own vendored data (differs from `.nff` for Sc, Mo, Ru, Be, Ti, Si, La, Gd, W); core accepts 10 eV-500 keV, only `KKResult.to_xray_result` enforces 0.03-30 keV. JAX floor `>=0.12.0.1` (golden tolerances are measured on it). `scripts/kk_bench.py` (CPU) and `scripts/kk_gpu_check.py` (manual, CUDA) are standalone gates. Docs: `docs/physics/kramers_kronig.rst` (Eq. 10 a₋₃ sign erratum), `docs/guides/kk_guide.rst`, `docs/architecture/005-adr-kk-jax-only.md`

### Test Structure

```
tests/
├── unit/           # Component tests (core, formula parsing, validation)
├── integration/    # CLI, cross-module, export tests
├── performance/    # Benchmarks, memory
├── characterization/  # Golden-output regression tests
├── fixtures/       # Test utilities and configuration
└── data/           # Test data assets
```

Registered markers (`--strict-markers`; an unregistered one is a collection error): `unit`, `integration`, `performance`, `slow`, `benchmark`, `memory`, `stability`, `regression`, `large_data`. `tests/conftest.py` auto-assigns `performance`, `memory`, `stability`, `large_data` and `regression` from test names, so `-m regression` only selects tests with "regression" in the name; use `-k regression` to be safe.

### Gotchas

- `--cov` is in pytest `addopts`, so every run measures coverage; pass `--no-cov` for single-test runs.
- `filterwarnings = error`: a new warning fails the test.
- GUI tests (`test_gui_smoke.py`, `test_gui_progress.py`, `test_gui_kk_tab.py`, `test_gui_kk_worker_end_to_end.py`, `unit/test_gui_widgets.py`, `unit/test_calculation_worker.py`, `unit/test_kk_plot.py`) run serially, outside xdist; the Makefile (`GUI_TESTS`) and `.github/workflows/ci.yml` (`IGNORE_GUI`) exclude them from the parallel pass, so add a new GUI test file to both lists.
- ruff is pinned to the same version in `.pre-commit-config.yaml`, `.github/workflows/{ci,security}.yml` and `uv.lock`; bump all four together.
- A version bump touches ~10 places; run `python scripts/check_version.py` (also a pre-commit hook; `--expect X.Y.Z` for a release).
- `mypy --strict` is clean locally (`untyped_calls_exclude = ["jax"]`); CI's mypy step is advisory.
- `graphify-out/` tracks only `GRAPH_REPORT.md`, `graph.html`, `graph.json`; the other files are ignored.

## Key Files

- `pyproject.toml` - All project config (deps, pytest, mypy, ruff)
- `Makefile` - Development commands (run `make help`)
- `scripts/run_type_check.py` - Enhanced mypy wrapper
- `scripts/check_version.py` - Cross-file version consistency check
- `scripts/gpu_check.py` - JAX/CUDA diagnostics behind `make gpu-check` / `gpu-diagnose`
- `.pre-commit-config.yaml` - Hooks (ruff, gitleaks, version check); ruff rev must match CI

## Coding Standards

- Use explicit imports
- NumPy-style docstrings
- Type hints for all public APIs
- Snake_case for new field names in XRayResult
- Energy in keV everywhere (tabulated range 0.03-30 keV); never eV in examples or tests

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
