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

Migration Guides
----------------

For upgrading between versions, see:

- :doc:`/guides/migration_guide_v0_4` — Upgrading from v0.3.0 to v0.4.0
