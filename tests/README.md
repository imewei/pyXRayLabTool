# XRayLabTool Test Suite

This directory contains a comprehensive test suite for the XRayLabTool Python package, translated from the original Julia test suite with enhancements for Python-specific functionality.

## Overview

The test suite is designed to ensure 100% compatibility between the Julia and Python implementations of XRayLabTool, with all numerical assertions using `numpy.isclose()` for robust floating-point comparisons.

## Test Structure

### Core Test Files

- **`test_integration.py`** - Main integration tests (translation of `test/runtests.jl`)
  - Basic setup and initialization tests
  - SiO2 and H2O property validation
  - calculate_single_material_properties Silicon properties
  - Edge cases and error handling
  - Input validation
  - Performance benchmarks using `pytest-benchmark`

- **`test_formula_parsing.py`** - Chemical formula parsing tests
  - Single atoms, mixed case correctness
  - Fractional stoichiometry
  - Long formulas (≥10 elements)
  - Complex formulas and validation
  - Invalid input handling
  - Regex compatibility verification

- **`test_utils.py`** - Utility function tests
  - Unit conversions (wavelength ↔ energy)
  - Crystallographic calculations
  - Data processing functions

- **`test_utils_enhanced.py`** - Enhanced utilities (mirrors Julia `test/utils.jl`)
  - Vector comparison utilities (`approx_vec`)
  - Cache management functions
  - Test isolation utilities

- **`test_atomic_data.py`** - Atomic data lookup tests
  - Element symbol validation
  - Atomic numbers and weights
  - LRU caching functionality
  - Error handling for invalid elements

- **`test_scattering_factors.py`** - Scattering factor loading and interpolation
  - .nff file loading and caching
  - PCHIP interpolator functionality
  - Data validation and format verification
  - Performance testing for interpolation

- **`test_core.py`** - Core module functionality
  - AtomicScatteringFactor class
  - CrystalStructure class
  - File loading utilities

- **`test_core_physics.py`** - Core physics calculations
  - X-ray optical properties
  - Mathematical validation

- **`test_robustness.py`** - Robustness improvements testing
  - Complex number handling in energy conversions
  - Type conversion edge cases
  - Error message validation
  - Dataclass field type verification

- **`test_smooth_data.py`** - Enhanced smooth_data function testing
  - Updated pandas method compatibility
  - Edge case handling for small arrays
  - Window size validation

## Test Categories

### 1. Integration Tests
Complete end-to-end testing of the main `calculate_xray_properties` and `calculate_single_material_properties` functions with expected values from the Julia implementation.

### 2. Unit Tests
Individual function and class testing with comprehensive edge case coverage.

### 3. Performance Benchmarks
Timed performance tests using `pytest-benchmark` for:
- Single material calculations
- Multi-material calculations
- Energy sweep calculations
- Formula complexity benchmarks

### 4. Error Handling Tests
Comprehensive validation of error conditions:
- Invalid input validation
- File not found scenarios
- Out-of-range parameters
- Type checking

## Expected Test Values

The test suite includes exact expected values from the Julia implementation for validation:

### SiO2 Properties (Julia indices → Python indices)
- **Dispersion**: `[2] = 9.451484792575434e-6`, `[4] = 5.69919201789506e-06`
- **f1 values**: `[0] = 30.641090313037314`, `[2] = 30.46419063207884`, `[4] = 30.366953850108544`
- **reSLD values**: `[2] = 1.8929689855615698e-5`, `[4] = 1.886926933936152e-5`

### H2O Properties
- **Dispersion**: `[2] = 4.734311949237782e-6`, `[4] = 2.8574954896752405e-6`
- **f1 values**: `[0] = 10.110775776847062`, `[2] = 10.065881494924541`, `[4] = 10.04313810715771`
- **reSLD values**: `[2] = 9.482008260671003e-6`, `[4] = 9.460584107129207e-6`

### Silicon Properties (calculate_single_material_properties)
- **Dispersion**: `1.20966554922812e-06`
- **f1**: `14.048053047106292`
- **f2**: `0.053331074920700626`
- **reSLD**: `1.9777910804587255e-5`
- **imSLD**: `7.508351793358633e-8`

## Running Tests

### Prerequisites
```bash
pip install -e .[dev]  # Install with development dependencies
```

### Run All Tests
```bash
# Using the provided test runner
python run_tests.py

# Or using pytest directly
pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Integration tests only
pytest tests/test_integration.py -v

# Performance benchmarks only  
pytest tests/test_integration.py::TestPerformanceBenchmarks --benchmark-only

# Formula parsing tests
pytest tests/test_formula_parsing.py -v

# With coverage report
pytest tests/ --cov=xraylabtool --cov-report=html
```

### Run Tests with Benchmarks
```bash
# Run benchmarks and generate JSON report
pytest tests/test_integration.py::TestPerformanceBenchmarks --benchmark-only --benchmark-json=benchmark.json

# Compare benchmarks (requires py.test-benchmark)
pytest-benchmark compare benchmark.json
```

## Numerical Precision

All numerical comparisons use `numpy.isclose()` with appropriate tolerances:
- **Default tolerance**: `1e-6` (matching Julia `DEFAULT_TOL`)
- **High precision comparisons**: `1e-10` for interpolation accuracy
- **Custom tolerances**: Specified per test case based on expected precision

## Continuous Integration

The test suite is designed for cross-platform CI using GitHub Actions:
- **Platforms**: Linux (Ubuntu), macOS, Windows
- **Python versions**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Coverage reporting**: Codecov integration
- **Benchmark comparison**: PR-based performance regression detection

## Test Utilities

### Vector Comparison
```python
from tests.test_utils_enhanced import approx_vec
assert approx_vec(array1, array2, atol=1e-6, rtol=1e-5)
```

### Cache Management
```python
from tests.test_utils_enhanced import with_cleared_caches

result = with_cleared_caches(lambda: your_test_function())
```

## Skipped Tests

Some tests are conditionally skipped when required data files are not available:
- Element-specific tests skip when `.nff` files are missing
- Complex element tests (Au, etc.) may be skipped in development environments

## Test Data Requirements

For complete test coverage, the following data files should be available:
- `src/AtomicScatteringFactor/si.nff` (Silicon)
- `src/AtomicScatteringFactor/ge.nff` (Germanium) 
- `src/AtomicScatteringFactor/h.nff` (Hydrogen)
- `src/AtomicScatteringFactor/c.nff` (Carbon)
- `src/AtomicScatteringFactor/n.nff` (Nitrogen)
- `src/AtomicScatteringFactor/o.nff` (Oxygen)

## Performance Benchmarks

The benchmark suite measures:
- **Single calculations**: calculate_single_material_properties with energy arrays
- **Multi-material calculations**: calculate_xray_properties with multiple formulas
- **Energy sweeps**: Large energy range calculations
- **Formula complexity**: Complex chemical formulas

Benchmark results are comparable to Julia performance and track performance regressions over time.

## Contributing

When adding new tests:
1. Use `numpy.isclose()` for numerical assertions
2. Include both positive and negative test cases
3. Add appropriate benchmark tests for performance-critical functions
4. Document expected values and tolerances
5. Ensure cross-platform compatibility

## Recent Improvements

### Version 0.1.1 Test Suite Enhancements

**New Test Coverage:**
- ✅ Complex number handling in energy conversion functions
- ✅ Type safety validation and conversion edge cases  
- ✅ Updated pandas method compatibility (`bfill`/`ffill`)
- ✅ Enhanced atomic data type conversion robustness
- ✅ PCHIP interpolation accuracy and edge cases
- ✅ Smooth data function improvements and edge handling
- ✅ Comprehensive error message validation

**Test Suite Statistics:**
- **Total Test Suites**: 13/13 passing (100% success rate)
- **Test Categories**: Integration, Unit, Performance, Robustness
- **Coverage**: All major components and edge cases
- **Compatibility**: Cross-platform verified (Linux, macOS, Windows)

**Enhanced Test Utilities:**
- Improved type ignore annotations for intentional error testing
- Enhanced robustness testing for edge cases
- Better error message pattern matching
- Comprehensive dataclass field validation

## Validation

The test suite validates:
- ✅ Numerical accuracy vs. Julia implementation
- ✅ Error handling and input validation
- ✅ Performance characteristics
- ✅ Cross-platform compatibility
- ✅ Memory usage and caching behavior
- ✅ API consistency and backward compatibility
- ✅ **NEW**: Type safety and complex number handling
- ✅ **NEW**: Modern pandas compatibility
- ✅ **NEW**: Enhanced robustness for edge cases
