# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4] - 2025-01-14

### Changed
- **BREAKING**: Renamed main functions for better readability and Python conventions:
  - `SubRefrac()` → `calculate_sub_refraction()`
  - `Refrac()` → `calculate_refraction()`
- Updated all documentation and examples to use new function names
- Updated test suite to use new function names (145 tests passing)
- Improved variable naming in internal functions for better code readability

### Fixed
- Updated installation test script to use new function names
- Updated Sphinx documentation configuration
- Maintained backward compatibility for XRayResult dataclass field names

### Documentation
- Updated README.md with all new function names
- Updated Sphinx documentation examples
- Updated test documentation
- All code examples now use descriptive function names

### Notes
- This is a **breaking change** for users calling `SubRefrac()` or `Refrac()` directly
- XRayResult dataclass fields remain unchanged (MW, f1, f2, etc.) for compatibility
- All numerical results and functionality remain identical

## [0.1.3] - 2025-01-13

### Changed
- Documentation cleanup
- Updated version references across files

## [0.1.2] - 2025-01-13

### Added
- Major performance optimizations
- Enhanced caching system for atomic scattering factor data
- Bulk atomic data loading capabilities
- Interpolator caching for improved performance
- Element path pre-computation
- Comprehensive test suite with 100% coverage
- Performance benchmarking tests

### Changed
- Improved robustness with complex number handling
- Enhanced type safety and error handling
- Updated pandas compatibility for modern versions
- PCHIP interpolation for more accurate scattering factor calculations

## [0.1.1] - 2025-01-12

### Added
- Initial release with core functionality
- X-ray optical property calculations
- Support for single and multiple material calculations
- NumPy-based vectorized calculations
- Built-in atomic scattering factor data

### Features
- Calculate optical constants (δ, β)
- Calculate scattering factors (f1, f2)
- Support for chemical formulas and material densities
- Based on CXRO/NIST data tables