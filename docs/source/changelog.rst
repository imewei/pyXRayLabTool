Changelog
=========

Version 0.1.1 (2025-01-13)
---------------------------

**Enhanced Robustness & Compatibility:**

- Fixed complex number handling in energy conversion functions
- Improved type safety with comprehensive type hints and checking  
- Updated pandas method calls for modern compatibility (``fillna`` → ``bfill``/``ffill``)
- Enhanced atomic data handling with robust type conversions
- Fixed numpy deprecation warnings (``trapz`` → ``trapezoid``)

**New Features:**

- PCHIP interpolation for atomic scattering factors with accuracy validation
- Enhanced caching system for scattering factor data
- Comprehensive error handling and validation with clear error messages
- Improved smooth data function with edge case handling

**Testing & Quality:**

- 100% test suite coverage with 13/13 test suites passing
- Added robustness tests for complex number handling
- Enhanced integration tests matching Julia implementation
- Performance benchmarks and regression testing
- Added type ignore annotations for intentional error testing

**Developer Experience:**

- Improved type annotations for better IDE support
- Enhanced test utilities and robustness testing
- Better documentation and examples with recent updates
- Cross-platform compatibility verified
- Enhanced error message clarity and debugging information

**Bug Fixes:**

- Fixed ``pkg_resources`` import fallback to ``importlib.resources``
- Fixed ArrayLike operator issues by converting to numpy arrays
- Fixed XRayResult dataclass type casting with explicit conversions
- Fixed smooth_data dtype truncation issues
- Fixed integration test regex patterns to match actual error messages
- Fixed Column type handling from mendeleev library

Version 0.1.0 (2025-01-07)
---------------------------

Initial release of XRayLabTool.

**Features:**

- Core X-ray optical property calculations
- Support for single and multiple material calculations
- XRayResult dataclass for structured output
- Atomic scattering factor data loading and caching
- Comprehensive utility functions for X-ray analysis
- Physical constants and conversion functions
- Vectorized NumPy calculations for performance
- Integration with CXRO/NIST data tables
- Full test coverage
- Complete documentation with Sphinx

**API:**

- ``SubRefrac()`` - Calculate properties for single material
- ``Refrac()`` - Calculate properties for multiple materials
- ``XRayResult`` - Dataclass for X-ray optical properties
- Utility functions for energy/wavelength conversion
- Bragg angle calculations
- Chemical formula parsing
- Atomic data access

**Data Sources:**

- CXRO atomic scattering factor tables
- NIST physical constants
- Support for elements H through U

**Dependencies:**

- NumPy >= 1.20.0
- SciPy >= 1.7.0  
- Pandas >= 1.3.0
- Mendeleev >= 0.10.0
- tqdm >= 4.60.0
- matplotlib >= 3.4.0 (optional)
