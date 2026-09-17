Testing Guide
=============

Guide to testing XRayLabTool code and ensuring quality.

Testing Philosophy
------------------

XRayLabTool follows a comprehensive testing strategy:

**Test Categories:**
- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test complete workflows and CLI commands
- **Performance Tests**: Ensure performance requirements are met
- **Characterization Tests**: 202 golden-value assertions for migration safety
- **Physics Tests**: Validate scientific accuracy

**Testing Principles:**
- Quick feedback: Most tests run in milliseconds
- Code coverage: >95% code coverage target
- Reliable: Tests pass consistently across platforms
- Clear failures: Descriptive error messages

Test Organization
-----------------

Directory Structure
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   tests/
   ├── conftest.py                 # Pytest configuration and fixtures
   ├── test_code_quality.py        # Code quality checks (naming, imports, types)
   ├── unit/                       # Unit tests
   │   ├── test_core.py            # Core calculation tests
   │   ├── test_utils.py           # Utility function tests
   │   ├── test_backend_dispatch.py # Backend abstraction tests
   │   └── ...
   ├── integration/                # Integration tests
   │   ├── test_integration.py     # End-to-end workflow tests
   │   └── test_completion_installer.py
   ├── characterization/           # Golden-value migration safety tests
   │   ├── test_golden_constants.py
   │   ├── test_golden_interpolation.py
   │   ├── test_golden_molecular.py
   │   ├── test_golden_pipeline.py
   │   └── ...                     # 202 assertions total
   └── performance/                # Performance regression tests
       ├── test_performance_benchmarks.py
       ├── test_memory_management.py
       └── ...

Running Tests
-------------

Basic Test Execution
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests (using uv)
   uv run pytest tests/ -v

   # Run specific test categories
   uv run pytest tests/unit/ -v              # Unit tests only
   uv run pytest tests/integration/ -v       # Integration tests only
   uv run pytest tests/characterization/ -v  # Golden-value tests
   uv run pytest tests/performance/ -v       # Performance tests only

   # Run with coverage
   uv run pytest tests/ --cov=xraylabtool --cov-report=html

   # Run tests matching pattern
   uv run pytest tests/ -k "test_silicon" -v

   # Or use Makefile shortcuts
   make test          # Tests with coverage
   make test-all      # Full suite

Writing Tests
-------------

Unit Test Example
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pytest
   from xraylabtool.calculators.core import calculate_single_material_properties
   from xraylabtool.exceptions import EnergyError, UnknownElementError

   class TestSingleMaterialCalculations:
       """Test single material property calculations."""

       def test_silicon_properties(self):
           """Test silicon properties at 8 keV."""
           result = calculate_single_material_properties("Si", 8.0, 2.33)

           assert result.formula == "Si"
           assert result.density_g_cm3 == 2.33
           assert result.energy_kev[0] == 8.0
           assert abs(result.critical_angle_degrees[0] - 0.2248) < 0.001

       def test_invalid_formula(self):
           """Test error handling for invalid formulas."""
           with pytest.raises(UnknownElementError, match="Unknown element"):
               calculate_single_material_properties("XYZ", 8.0, 1.0)

       @pytest.mark.parametrize("energy", [0, -1.0])
       def test_invalid_energy(self, energy):
           """Test error handling for invalid energies."""
           with pytest.raises(EnergyError):
               calculate_single_material_properties("Si", energy, 2.33)

Integration Test Example
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import subprocess
   import json

   def test_cli_calc_command():
       """Test the calc CLI command."""
       result = subprocess.run([
           "xraylabtool", "calc", "Si",
           "-e", "8.0",
           "-d", "2.33",
           "--format", "json"
       ], capture_output=True, text=True)

       assert result.returncode == 0
       data = json.loads(result.stdout)
       assert data["formula"] == "Si"
       assert abs(data["critical_angle_degrees"][0] - 0.2248) < 0.001

Performance Test Example
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   import pytest

   def test_batch_processing_performance():
       """Test that batch processing meets performance requirements."""
       formulas = ["Si", "SiO2", "Al2O3"] * 100
       densities = [2.33, 2.20, 3.95] * 100

       start_time = time.time()
       results = calculate_xray_properties(formulas, 8.0, densities)
       end_time = time.time()

       assert (end_time - start_time) < 5.0
       assert set(results) == {"Si", "SiO2", "Al2O3"}  # keyed by formula

Test Configuration
------------------

Pytest Configuration
~~~~~~~~~~~~~~~~~~~~~

The ``conftest.py`` file contains shared test configuration:

.. code-block:: python

   import pytest
   import numpy as np

   @pytest.fixture
   def sample_materials():
       """Common test materials."""
       return [
           {"formula": "Si", "density": 2.33},
           {"formula": "SiO2", "density": 2.20},
           {"formula": "Al2O3", "density": 3.95}
       ]

   @pytest.fixture
   def energy_range():
       """Common energy range for testing."""
       return np.logspace(0, np.log10(30), 10)  # 1 keV to 30 keV

Test Utilities
~~~~~~~~~~~~~~

The ``fixtures/`` directory contains helper functions:

.. code-block:: python

   def assert_result_valid(result):
       """Assert that an XRayResult is valid."""
       assert result.formula is not None
       assert (result.energy_kev > 0).all()
       assert (result.critical_angle_degrees > 0).all()
       assert (result.attenuation_length_cm > 0).all()

   def create_test_material(formula="Si", energy_kev=8.0, density=2.33):
       """Create a test material for consistent testing."""
       return calculate_single_material_properties(formula, energy_kev, density)

Performance Testing
-------------------

Performance Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~

Tests ensure performance standards:

- **Single calculations**: a few ms with a warm atomic-data cache
- **Batch processing**: parallel above 20 materials (``ThreadPoolExecutor``)
- **Memory usage**: Reasonable scaling with dataset size

Benchmarking Code
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   from xraylabtool.calculators.core import calculate_single_material_properties

   def benchmark_single_calculation():
       """Benchmark single material calculation."""
       n_iterations = 1000

       start_time = time.time()
       for _ in range(n_iterations):
           calculate_single_material_properties("Si", 8.0, 2.33)
       end_time = time.time()

       avg_time = (end_time - start_time) / n_iterations
       assert avg_time < 0.01  # a few ms with a warm cache

Test Coverage
-------------

Coverage Requirements
~~~~~~~~~~~~~~~~~~~~~

- **Minimum coverage**: 95%
- **Critical modules**: 100% coverage required
- **Exception paths**: All error conditions tested

Generating Coverage Reports
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Generate HTML coverage report
   pytest tests/ --cov=xraylabtool --cov-report=html

   # Generate terminal coverage report
   pytest tests/ --cov=xraylabtool --cov-report=term-missing

   # Coverage with branch checking
   pytest tests/ --cov=xraylabtool --cov-branch

Continuous Integration
----------------------

All tests run automatically on:

- **Push to main branch**
- **Pull requests**
- **Scheduled nightly builds**

Test Matrix
~~~~~~~~~~~

Tests run across multiple configurations:

- **Python versions**: 3.12, 3.13
- **Operating systems**: Ubuntu, macOS, Windows
- **Toolchain**: ruff (lint + format), mypy (type checking), pytest (testing)
- **CI**: GitHub Actions with SHA-pinned action versions

Contributing Tests
------------------

When contributing code:

1. **Write tests first** (TDD approach)
2. **Ensure all tests pass** before submitting
3. **Maintain coverage** above 95%
4. **Add performance tests** for new features
5. **Include integration tests** for CLI changes

Test Guidelines
~~~~~~~~~~~~~~~

**Good Test Practices:**
- Test one thing per test function
- Use descriptive test names
- Include both positive and negative test cases
- Use appropriate assertions
- Mock external dependencies

**Test Naming Convention:**
- ``test_function_behavior_condition()``
- Example: ``test_calculate_properties_invalid_formula()``

For more testing examples and patterns, see the existing test suite in the ``tests/`` directory.
