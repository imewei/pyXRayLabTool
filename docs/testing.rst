Testing Guide
=============

Comprehensive guide to testing XRayLabTool code and ensuring quality.

Testing Philosophy
------------------

XRayLabTool follows a comprehensive testing strategy:

**Test Categories:**
- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test complete workflows and CLI commands
- **Performance Tests**: Ensure performance requirements are met
- **Physics Tests**: Validate scientific accuracy
- **Regression Tests**: Prevent reintroduction of fixed bugs

**Testing Principles:**
- **Fast feedback**: Most tests run in milliseconds
- **Comprehensive coverage**: >95% code coverage target
- **Reliable**: Tests pass consistently across platforms
- **Clear failures**: Descriptive error messages
- **Independent**: Tests don't depend on external resources

Test Organization
-----------------

Directory Structure
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   tests/
   ├── conftest.py                 # Pytest configuration and fixtures
   ├── fixtures/                   # Shared test utilities
   │   ├── __init__.py
   │   ├── base_test.py            # Base test classes
   │   ├── test_config.py          # Test constants and configuration
   │   └── test_utilities.py       # Helper functions
   ├── unit/                       # Unit tests
   │   ├── __init__.py
   │   ├── test_core.py            # Core calculation tests
   │   ├── test_utils.py           # Utility function tests
   │   ├── test_exceptions.py      # Exception handling tests
   │   ├── test_formula_parsing.py # Formula parsing tests
   │   └── test_atomic_data.py     # Atomic data cache tests
   ├── integration/                # Integration tests
   │   ├── __init__.py
   │   ├── test_cli.py             # CLI command tests
   │   └── test_workflows.py       # End-to-end workflow tests
   ├── performance/                # Performance tests
   │   ├── __init__.py
   │   ├── test_benchmarks.py      # Performance benchmarks
   │   └── test_memory_usage.py    # Memory usage tests
   └── data/                       # Test data files
       ├── test_materials.csv
       ├── expected_results.json
       └── atomic_test_data.h5

Base Test Classes
~~~~~~~~~~~~~~~~~

All tests inherit from base classes that provide common functionality:

.. code-block:: python

   # tests/fixtures/base_test.py
   import pytest
   import numpy as np
   from xraylabtool.calculators.core import calculate_single_material_properties

   class BaseXRayLabToolTest:
       """Base class for all XRayLabTool tests."""

       @pytest.fixture(autouse=True)
       def setup_test(self):
           """Set up each test with clean state."""
           # Clear caches before each test
           from xraylabtool.data_handling.atomic_cache import clear_cache
           clear_cache()
           yield
           # Cleanup after test if needed

   class BaseUnitTest(BaseXRayLabToolTest):
       """Base class for unit tests."""

       def assert_close(self, actual, expected, rtol=1e-5, atol=1e-8):
           """Assert two values are close within tolerance."""
           np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

       def assert_physics_reasonable(self, result):
           """Assert that physics results are reasonable."""
           assert 0 < result.delta < 1e-3, f"Delta {result.delta} out of range"
           assert 0 < result.beta < result.delta, f"Beta {result.beta} should be < delta"
           assert 0 < result.critical_angle_degrees < 1, "Critical angle out of range"

   class BaseIntegrationTest(BaseXRayLabToolTest):
       """Base class for integration tests."""

       def run_cli_command(self, cmd_args):
           """Run CLI command and return result."""
           import subprocess
           result = subprocess.run(
               ["xraylabtool"] + cmd_args,
               capture_output=True,
               text=True
           )
           return result

Unit Tests
----------

Core Functionality Tests
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # tests/unit/test_core.py
   import pytest
   from tests.fixtures.base_test import BaseUnitTest
   from xraylabtool.calculators.core import (
       calculate_single_material_properties,
       calculate_xray_properties
   )
   from xraylabtool.validation.exceptions import FormulaError, EnergyError

   class TestSingleMaterialCalculations(BaseUnitTest):
       """Test single material calculations."""

       def test_silicon_at_8kev(self):
           """Test silicon properties at 8 keV - reference case."""
           result = calculate_single_material_properties("Si", 2.33, 8000)

           assert result.formula == "Si"
           assert result.density_g_cm3 == 2.33
           assert result.energy_ev == 8000
           assert abs(result.wavelength_angstrom - 1.5498) < 0.0001

           # Critical angle should be ~0.158 degrees
           self.assert_close(result.critical_angle_degrees, 0.158, rtol=0.01)

           # Attenuation length should be reasonable
           assert 5 < result.attenuation_length_cm < 20

           # Physics sanity checks
           self.assert_physics_reasonable(result)

       def test_compound_material(self):
           """Test compound material calculation."""
           result = calculate_single_material_properties("SiO2", 2.20, 8000)

           assert result.formula == "SiO2"
           assert result.density_g_cm3 == 2.20

           # SiO2 should have smaller critical angle than pure Si
           si_result = calculate_single_material_properties("Si", 2.33, 8000)
           assert result.critical_angle_degrees < si_result.critical_angle_degrees

       @pytest.mark.parametrize("formula,density,expected_elements", [
           ("H2O", 1.0, ["H", "O"]),
           ("Ca5(PO4)3F", 3.18, ["Ca", "P", "O", "F"]),
           ("CuSO4·5H2O", 2.29, ["Cu", "S", "O", "H"])
       ])
       def test_complex_formulas(self, formula, density, expected_elements):
           """Test complex formula parsing and calculation."""
           result = calculate_single_material_properties(formula, density, 8000)

           assert result.formula == formula
           self.assert_physics_reasonable(result)

           # Verify formula parsing worked
           from xraylabtool.utils import parse_formula
           composition = parse_formula(formula)
           for element in expected_elements:
               assert element in composition

       def test_energy_array(self):
           """Test calculation with energy array."""
           import numpy as np
           energies = np.array([5000, 8000, 10000])

           results = []
           for energy in energies:
               result = calculate_single_material_properties("Si", 2.33, energy)
               results.append(result)

           # Critical angle should decrease with energy
           critical_angles = [r.critical_angle_degrees for r in results]
           assert critical_angles[0] > critical_angles[1] > critical_angles[2]

           # Attenuation length should increase with energy
           att_lengths = [r.attenuation_length_cm for r in results]
           assert att_lengths[0] < att_lengths[1] < att_lengths[2]

Error Handling Tests
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # tests/unit/test_exceptions.py
   import pytest
   from tests.fixtures.base_test import BaseUnitTest
   from xraylabtool.calculators.core import calculate_single_material_properties
   from xraylabtool.validation.exceptions import (
       FormulaError, EnergyError, ValidationError
   )

   class TestErrorHandling(BaseUnitTest):
       """Test error handling and validation."""

       def test_invalid_formulas(self):
           """Test various invalid formula formats."""
           invalid_formulas = [
               "",           # Empty formula
               "XYZ",        # Unknown element
               "Si-O2",      # Invalid syntax
               "123",        # Numbers only
               "si",         # Wrong capitalization
           ]

           for formula in invalid_formulas:
               with pytest.raises(FormulaError):
                   calculate_single_material_properties(formula, 2.33, 8000)

       def test_invalid_energies(self):
           """Test invalid energy values."""
           invalid_energies = [0, -1000, -1e-10]

           for energy in invalid_energies:
               with pytest.raises(EnergyError, match="Energy must be positive"):
                   calculate_single_material_properties("Si", 2.33, energy)

       def test_invalid_densities(self):
           """Test invalid density values."""
           invalid_densities = [0, -1.0, -1e-10]

           for density in invalid_densities:
               with pytest.raises(ValidationError):
                   calculate_single_material_properties("Si", density, 8000)

       def test_warning_conditions(self):
           """Test conditions that should generate warnings."""
           import warnings

           # Very low energy should warn
           with warnings.catch_warnings(record=True) as w:
               warnings.simplefilter("always")
               calculate_single_material_properties("Si", 2.33, 5)  # 5 eV
               assert len(w) > 0
               assert "unreliable" in str(w[0].message).lower()

           # Very high energy should warn
           with warnings.catch_warnings(record=True) as w:
               warnings.simplefilter("always")
               calculate_single_material_properties("Si", 2.33, 200000)  # 200 keV
               assert len(w) > 0

Utility Function Tests
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # tests/unit/test_utils.py
   import pytest
   import numpy as np
   from tests.fixtures.base_test import BaseUnitTest
   from xraylabtool.utils import (
       parse_formula, energy_to_wavelength, wavelength_to_energy,
       calculate_molecular_weight
   )

   class TestUtilityFunctions(BaseUnitTest):
       """Test utility functions."""

       def test_energy_wavelength_conversion(self):
           """Test energy-wavelength conversion functions."""
           # Test known values
           energy = 8000  # eV
           expected_wavelength = 1.54984  # Angstroms

           wavelength = energy_to_wavelength(energy)
           self.assert_close(wavelength, expected_wavelength, rtol=1e-4)

           # Round trip conversion
           energy_back = wavelength_to_energy(wavelength)
           self.assert_close(energy_back, energy, rtol=1e-10)

       def test_formula_parsing(self):
           """Test chemical formula parsing."""
           test_cases = [
               ("H2O", {"H": 2, "O": 1}),
               ("SiO2", {"Si": 1, "O": 2}),
               ("Ca5(PO4)3F", {"Ca": 5, "P": 3, "O": 12, "F": 1}),
               ("CuSO4·5H2O", {"Cu": 1, "S": 1, "O": 9, "H": 10}),
               ("Al2(SO4)3·18H2O", {"Al": 2, "S": 3, "O": 30, "H": 36})
           ]

           for formula, expected in test_cases:
               result = parse_formula(formula)
               assert result == expected, f"Failed for {formula}"

       def test_molecular_weight(self):
           """Test molecular weight calculations."""
           # Water: 2*1.008 + 15.999 = 18.015
           mw_water = calculate_molecular_weight("H2O")
           self.assert_close(mw_water, 18.015, rtol=1e-3)

           # Silicon dioxide: 28.0855 + 2*15.999 = 60.0835
           mw_sio2 = calculate_molecular_weight("SiO2")
           self.assert_close(mw_sio2, 60.084, rtol=1e-3)

Integration Tests
-----------------

CLI Command Tests
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # tests/integration/test_cli.py
   import json
   import subprocess
   from tests.fixtures.base_test import BaseIntegrationTest

   class TestCLICommands(BaseIntegrationTest):
       """Test CLI command functionality."""

       def test_calc_command_basic(self):
           """Test basic calc command."""
           result = self.run_cli_command([
               "calc", "Si", "--density", "2.33", "--energy", "8000"
           ])

           assert result.returncode == 0
           assert "Si" in result.stdout
           assert "Critical angle" in result.stdout
           assert "Attenuation length" in result.stdout

       def test_calc_command_json_output(self):
           """Test calc command with JSON output."""
           result = self.run_cli_command([
               "calc", "Si", "--density", "2.33", "--energy", "8000",
               "--output", "json"
           ])

           assert result.returncode == 0
           data = json.loads(result.stdout)
           assert len(data) == 1
           assert data[0]["formula"] == "Si"
           assert abs(data[0]["critical_angle_degrees"] - 0.158) < 0.01

       def test_calc_command_multiple_energies(self):
           """Test calc command with multiple energies."""
           result = self.run_cli_command([
               "calc", "Si", "--density", "2.33", "--energy", "5000,8000,10000",
               "--output", "json"
           ])

           assert result.returncode == 0
           data = json.loads(result.stdout)
           assert len(data) == 3

           # Check energy values
           energies = [item["energy_ev"] for item in data]
           assert energies == [5000, 8000, 10000]

       def test_batch_command(self):
           """Test batch processing command."""
           import tempfile
           import csv

           # Create temporary CSV file
           with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
               writer = csv.writer(f)
               writer.writerow(['Formula', 'Density', 'Energy'])
               writer.writerow(['Si', '2.33', '8000'])
               writer.writerow(['Al', '2.70', '8000'])
               temp_file = f.name

           try:
               result = self.run_cli_command([
                   "batch", temp_file, "--format", "json"
               ])

               assert result.returncode == 0
               data = json.loads(result.stdout)
               assert len(data) == 2

               formulas = [item["formula"] for item in data]
               assert "Si" in formulas
               assert "Al" in formulas

           finally:
               import os
               os.unlink(temp_file)

       def test_convert_command(self):
           """Test unit conversion command."""
           result = self.run_cli_command([
               "convert", "--energy", "8000", "--to", "wavelength"
           ])

           assert result.returncode == 0
           assert "1.5498" in result.stdout  # Expected wavelength

       def test_formula_command(self):
           """Test formula parsing command."""
           result = self.run_cli_command([
               "formula", "Ca5(PO4)3F"
           ])

           assert result.returncode == 0
           assert "Ca (5)" in result.stdout
           assert "P (3)" in result.stdout
           assert "O (12)" in result.stdout
           assert "F (1)" in result.stdout

       def test_list_command(self):
           """Test list command."""
           result = self.run_cli_command(["list", "elements"])

           assert result.returncode == 0
           assert "Si" in result.stdout
           assert "Al" in result.stdout
           assert "Fe" in result.stdout

       def test_error_handling(self):
           """Test CLI error handling."""
           # Invalid formula
           result = self.run_cli_command([
               "calc", "XYZ", "--density", "1.0", "--energy", "8000"
           ])
           assert result.returncode != 0
           assert "Unknown element" in result.stderr

           # Missing required argument
           result = self.run_cli_command(["calc", "Si"])
           assert result.returncode != 0
           assert "density" in result.stderr.lower()

Workflow Tests
~~~~~~~~~~~~~~

.. code-block:: python

   # tests/integration/test_workflows.py
   from tests.fixtures.base_test import BaseIntegrationTest
   from xraylabtool.calculators.core import calculate_xray_properties

   class TestWorkflows(BaseIntegrationTest):
       """Test complete analysis workflows."""

       def test_material_comparison_workflow(self):
           """Test comparing multiple materials."""
           materials = [
               {"formula": "Si", "density": 2.33},
               {"formula": "SiO2", "density": 2.20},
               {"formula": "Al", "density": 2.70}
           ]
           energies = [8000]

           results = calculate_xray_properties(materials, energies)

           assert len(results) == 3

           # Sort by critical angle
           results_sorted = sorted(results,
                                 key=lambda x: x.critical_angle_degrees,
                                 reverse=True)

           # Al should have highest critical angle (highest Z*density)
           assert results_sorted[0].formula == "Al"

           # All results should be physically reasonable
           for result in results:
               self.assert_physics_reasonable(result)

       def test_energy_scan_workflow(self):
           """Test energy-dependent analysis."""
           import numpy as np

           energies = np.logspace(3, 4.5, 20)  # 1 keV to ~32 keV
           material = {"formula": "Si", "density": 2.33}

           results = []
           for energy in energies:
               result = calculate_single_material_properties(
                   material["formula"], material["density"], energy
               )
               results.append(result)

           # Extract properties for analysis
           critical_angles = [r.critical_angle_mrad for r in results]
           attenuation_lengths = [r.attenuation_length_cm for r in results]

           # Critical angle should decrease monotonically with energy
           assert all(a > b for a, b in zip(critical_angles[:-1], critical_angles[1:]))

           # Attenuation length should generally increase with energy
           # (may have local variations near absorption edges)
           assert attenuation_lengths[-1] > attenuation_lengths[0]

Performance Tests
-----------------

Benchmark Tests
~~~~~~~~~~~~~~~

.. code-block:: python

   # tests/performance/test_benchmarks.py
   import time
   import pytest
   from tests.fixtures.base_test import BaseXRayLabToolTest
   from xraylabtool.calculators.core import (
       calculate_single_material_properties,
       calculate_xray_properties
   )

   class TestPerformance(BaseXRayLabToolTest):
       """Test performance requirements."""

       def test_single_calculation_speed(self):
           """Test single calculation performance."""
           # Warm up cache
           calculate_single_material_properties("Si", 2.33, 8000)

           # Measure performance
           n_iterations = 1000
           start_time = time.time()

           for _ in range(n_iterations):
               calculate_single_material_properties("Si", 2.33, 8000)

           end_time = time.time()
           avg_time = (end_time - start_time) / n_iterations

           # Should be < 0.1 ms per calculation
           assert avg_time < 0.0001, f"Too slow: {avg_time*1000:.3f} ms per calculation"

       def test_batch_processing_speed(self):
           """Test batch processing performance."""
           materials = [{"formula": "Si", "density": 2.33}] * 1000
           energies = [8000]

           start_time = time.time()
           results = calculate_xray_properties(materials, energies)
           end_time = time.time()

           assert len(results) == 1000

           # Should process 1000 materials in < 50 ms
           processing_time = end_time - start_time
           assert processing_time < 0.05, f"Batch too slow: {processing_time:.3f} s"

           # Calculate throughput
           throughput = len(results) / processing_time
           assert throughput > 20000, f"Low throughput: {throughput:.0f} calc/s"

       @pytest.mark.parametrize("n_materials", [100, 1000, 10000])
       def test_scaling_performance(self, n_materials):
           """Test performance scaling with dataset size."""
           materials = [{"formula": "Si", "density": 2.33}] * n_materials
           energies = [8000]

           start_time = time.time()
           results = calculate_xray_properties(materials, energies)
           end_time = time.time()

           processing_time = end_time - start_time
           time_per_calculation = processing_time / len(results)

           # Should maintain good performance per calculation
           assert time_per_calculation < 0.0001, \
               f"Poor scaling at {n_materials} materials: {time_per_calculation*1000:.3f} ms/calc"

Memory Usage Tests
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # tests/performance/test_memory_usage.py
   import pytest
   import psutil
   import os
   from tests.fixtures.base_test import BaseXRayLabToolTest
   from xraylabtool.calculators.core import calculate_xray_properties

   class TestMemoryUsage(BaseXRayLabToolTest):
       """Test memory usage and management."""

       def get_memory_usage(self):
           """Get current memory usage in MB."""
           process = psutil.Process(os.getpid())
           return process.memory_info().rss / 1024 / 1024

       def test_memory_usage_scaling(self):
           """Test memory usage scales reasonably with dataset size."""
           base_memory = self.get_memory_usage()

           # Test different dataset sizes
           sizes = [100, 1000, 5000]
           memory_usage = []

           for size in sizes:
               materials = [{"formula": "Si", "density": 2.33}] * size
               energies = [8000]

               mem_before = self.get_memory_usage()
               results = calculate_xray_properties(materials, energies)
               mem_after = self.get_memory_usage()

               memory_used = mem_after - mem_before
               memory_usage.append(memory_used)

               # Memory usage should be reasonable
               assert memory_used < size * 0.01, \
                   f"Excessive memory usage: {memory_used:.1f} MB for {size} materials"

           # Memory usage should scale sub-linearly due to caching
           memory_per_item = [mem / size for mem, size in zip(memory_usage, sizes)]
           assert memory_per_item[-1] <= memory_per_item[0], \
               "Memory usage scaling worse than linear"

       def test_memory_cleanup(self):
           """Test memory is properly cleaned up."""
           initial_memory = self.get_memory_usage()

           # Create large dataset
           materials = [{"formula": "Si", "density": 2.33}] * 10000
           energies = [8000]

           results = calculate_xray_properties(materials, energies)
           peak_memory = self.get_memory_usage()

           # Clear references
           del results, materials, energies

           # Force garbage collection
           import gc
           gc.collect()

           final_memory = self.get_memory_usage()

           # Memory should return close to initial level
           memory_increase = final_memory - initial_memory
           assert memory_increase < 50, \
               f"Memory leak detected: {memory_increase:.1f} MB not cleaned up"

Test Configuration
------------------

Pytest Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # conftest.py
   import pytest
   import numpy as np

   # Configure NumPy for consistent results
   np.random.seed(42)

   @pytest.fixture(scope="session")
   def test_config():
       """Test configuration constants."""
       return {
           'default_energy': 8000,
           'default_density': 2.33,
           'tolerance_rtol': 1e-5,
           'tolerance_atol': 1e-8,
           'performance_timeout': 0.1,  # 100ms
           'memory_limit_mb': 100
       }

   @pytest.fixture
   def silicon_result(test_config):
       """Standard silicon calculation for comparison."""
       from xraylabtool.calculators.core import calculate_single_material_properties
       return calculate_single_material_properties(
           "Si", test_config['default_density'], test_config['default_energy']
       )

   # Custom markers
   pytest.mark.slow = pytest.mark.slow
   pytest.mark.integration = pytest.mark.integration
   pytest.mark.performance = pytest.mark.performance

Test Data Management
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # tests/fixtures/test_utilities.py
   import json
   import numpy as np
   from pathlib import Path

   class TestDataManager:
       """Manage test data files and expected results."""

       def __init__(self):
           self.data_dir = Path(__file__).parent.parent / "data"

       def load_expected_results(self, test_name):
           """Load expected results for validation."""
           file_path = self.data_dir / f"{test_name}_expected.json"
           with open(file_path, 'r') as f:
               return json.load(f)

       def save_expected_results(self, test_name, results):
           """Save results as expected values for future tests."""
           file_path = self.data_dir / f"{test_name}_expected.json"

           # Convert numpy arrays to lists for JSON serialization
           serializable_results = self._make_serializable(results)

           with open(file_path, 'w') as f:
               json.dump(serializable_results, f, indent=2)

       def _make_serializable(self, obj):
           """Convert numpy arrays to lists recursively."""
           if isinstance(obj, np.ndarray):
               return obj.tolist()
           elif isinstance(obj, dict):
               return {k: self._make_serializable(v) for k, v in obj.items()}
           elif isinstance(obj, list):
               return [self._make_serializable(item) for item in obj]
           else:
               return obj

Running Tests
-------------

Command Line Usage
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest tests/ -v

   # Run specific test categories
   pytest tests/unit/ -v                    # Unit tests only
   pytest tests/integration/ -v             # Integration tests only
   pytest tests/performance/ -v             # Performance tests only

   # Run tests with coverage
   pytest tests/ --cov=xraylabtool --cov-report=html --cov-report=term

   # Run tests with specific markers
   pytest -m "not slow" -v                  # Skip slow tests
   pytest -m "performance" -v               # Performance tests only
   pytest -m "integration" -v               # Integration tests only

   # Run tests matching pattern
   pytest tests/ -k "silicon" -v            # Tests containing "silicon"
   pytest tests/ -k "cli" -v                # CLI-related tests

   # Run with different verbosity levels
   pytest tests/ -v                         # Verbose
   pytest tests/ -vv                        # Very verbose
   pytest tests/ -q                         # Quiet

   # Run specific test file or function
   pytest tests/unit/test_core.py -v
   pytest tests/unit/test_core.py::TestSingleMaterialCalculations::test_silicon_at_8kev -v

   # Run with custom options
   pytest tests/ --tb=short                 # Short traceback format
   pytest tests/ --maxfail=3                # Stop after 3 failures
   pytest tests/ --pdb                      # Drop into debugger on failure

Continuous Integration
~~~~~~~~~~~~~~~~~~~~~~

GitHub Actions workflow for automated testing:

.. code-block:: yaml

   # .github/workflows/tests.yml
   name: Tests

   on: [push, pull_request]

   jobs:
     test:
       runs-on: ubuntu-latest
       strategy:
         matrix:
           python-version: [3.12, 3.13]

       steps:
       - uses: actions/checkout@v4
       - name: Set up Python ${{ matrix.python-version }}
         uses: actions/setup-python@v4
         with:
           python-version: ${{ matrix.python-version }}

       - name: Install dependencies
         run: |
           python -m pip install --upgrade pip
           pip install -e .[dev]

       - name: Run tests
         run: |
           pytest tests/ -v --cov=xraylabtool --cov-report=xml

       - name: Upload coverage
         uses: codecov/codecov-action@v3
         with:
           file: ./coverage.xml

Test Quality Metrics
---------------------

Coverage Requirements
~~~~~~~~~~~~~~~~~~~~~

- **Overall coverage**: > 95%
- **Critical modules**: > 98%
- **New code**: 100% coverage required
- **Integration tests**: > 80% of features covered

Performance Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

- **Single calculation**: < 0.1 ms (warm cache)
- **Batch processing**: > 100,000 calculations/second
- **Memory usage**: Linear scaling with dataset size
- **Cache hit rate**: > 90% for repeated calculations

Reliability Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

- **Test pass rate**: > 99.5%
- **Flaky tests**: < 0.1% failure rate
- **Cross-platform**: Pass on Windows, macOS, Linux
- **Python versions**: Support 3.12+

Writing Good Tests
------------------

Test Design Principles
~~~~~~~~~~~~~~~~~~~~~~

**1. Clear and Descriptive Names:**

.. code-block:: python

   # Good
   def test_silicon_critical_angle_at_8kev():
       """Test critical angle calculation for silicon at 8 keV."""

   # Less clear
   def test_calc():
       """Test calculation."""

**2. Single Responsibility:**

.. code-block:: python

   # Good - tests one specific behavior
   def test_formula_parsing_with_parentheses():
       """Test formula parsing handles parentheses correctly."""
       result = parse_formula("Ca(OH)2")
       assert result == {"Ca": 1, "O": 2, "H": 2}

   # Bad - tests multiple unrelated things
   def test_everything():
       """Test formula parsing and calculations."""
       # ... tests multiple different functionalities

**3. Arrange-Act-Assert Pattern:**

.. code-block:: python

   def test_critical_angle_calculation():
       """Test critical angle calculation."""
       # Arrange
       formula = "Si"
       density = 2.33
       energy = 8000

       # Act
       result = calculate_single_material_properties(formula, density, energy)

       # Assert
       assert abs(result.critical_angle_degrees - 0.158) < 0.001

**4. Use Fixtures for Setup:**

.. code-block:: python

   @pytest.fixture
   def silicon_material():
       """Standard silicon test case."""
       return {"formula": "Si", "density": 2.33}

   def test_calculation_with_fixture(silicon_material):
       """Test calculation using fixture."""
       result = calculate_single_material_properties(
           silicon_material["formula"],
           silicon_material["density"],
           8000
       )
       assert result.formula == "Si"

**5. Parameterized Tests for Multiple Cases:**

.. code-block:: python

   @pytest.mark.parametrize("formula,expected_mw", [
       ("H2O", 18.015),
       ("SiO2", 60.084),
       ("Al2O3", 101.961)
   ])
   def test_molecular_weights(formula, expected_mw):
       """Test molecular weight calculations."""
       mw = calculate_molecular_weight(formula)
       assert abs(mw - expected_mw) < 0.01

Test Debugging
--------------

When tests fail:

**1. Read the Error Message Carefully:**
- Look for assertion details
- Check file and line numbers
- Understand what was expected vs actual

**2. Use Pytest's Debugging Features:**

.. code-block:: bash

   # Drop into debugger on failure
   pytest tests/unit/test_core.py::test_failing_test --pdb

   # Show local variables in traceback
   pytest tests/ --tb=long

   # Show only the first failure
   pytest tests/ --maxfail=1

**3. Add Temporary Debug Output:**

.. code-block:: python

   def test_debug_example():
       """Test with debug output."""
       result = calculate_properties("Si", 2.33, 8000)

       # Temporary debug output
       print(f"Debug: result.delta = {result.delta}")
       print(f"Debug: result.critical_angle = {result.critical_angle_degrees}")

       assert abs(result.critical_angle_degrees - 0.158) < 0.001

**4. Use pytest-xdist for Parallel Testing:**

.. code-block:: bash

   # Run tests in parallel
   pip install pytest-xdist
   pytest tests/ -n auto  # Use all CPU cores

This comprehensive testing approach ensures XRayLabTool maintains high quality, performance, and reliability across all supported platforms and use cases.
