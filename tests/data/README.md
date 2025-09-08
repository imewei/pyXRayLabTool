# Test Data Directory

This directory contains test data fixtures and configuration files for the xraylabtool test suite.

## Files

### `sample_materials.json`
Contains sample material data including:
- Common oxides (SiO2, Al2O3, Fe2O3, TiO2, ZnO)
- Pure elements (Si, Au, C, Cu, Pt)
- Complex compounds (CaCO3, BaF2, CaF2)
- Test configurations for different scenarios
- Reference values for validation

### `test_energies.json`
Contains standard energy arrays for testing:
- Single point, three point, and multi-point arrays
- Linear and logarithmic energy ranges
- Common X-ray edge energies
- Performance and memory testing configurations
- Edge case and error testing values

### `performance_baselines.json`
Contains performance baseline values and configurations:
- Performance thresholds for various operations
- Memory usage baselines
- Optimization validation criteria
- Benchmark configurations
- CI/CD test configurations

## Usage in Tests

These data files are used by the test fixtures in `conftest.py` and utility functions in `utils.py` to provide consistent test data across the test suite.

Example usage:
```python
import json
from pathlib import Path

# Load sample materials
data_dir = Path(__file__).parent / 'data'
with open(data_dir / 'sample_materials.json') as f:
    materials = json.load(f)

# Use in tests
oxide_materials = materials['materials']['common_oxides']
```

## Maintenance

- Update reference values when calculation algorithms change
- Adjust performance baselines when optimizations are implemented
- Add new materials for expanded test coverage
- Keep energy ranges synchronized with calculation limits
