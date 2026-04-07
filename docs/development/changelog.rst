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

Migration Guides
----------------

For upgrading between versions, see:

- :doc:`/guides/migration_guide_v0_4` — Upgrading from v0.3.0 to v0.4.0
