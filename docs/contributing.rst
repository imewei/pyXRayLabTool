Contributing Guide
==================

Welcome to XRayLabTool! We appreciate your interest in contributing to this high-performance X-ray optical properties calculator.

Getting Started
---------------

Types of Contributions
~~~~~~~~~~~~~~~~~~~~~~

We welcome various types of contributions:

**Code Contributions:**
- New features and enhancements
- Performance optimizations
- Bug fixes and stability improvements
- Test coverage improvements

**Documentation:**
- Tutorial improvements
- API documentation enhancements
- Example notebooks and use cases
- Translation to other languages

**Scientific Contributions:**
- Validation against experimental data
- New atomic data sources
- Algorithm improvements
- Physics model enhancements

**Community Support:**
- Answering questions in GitHub Discussions
- Reviewing pull requests
- Reporting issues and bugs
- User experience feedback

Development Setup
-----------------

Prerequisites
~~~~~~~~~~~~~

- **Python 3.12+**
- **Git** for version control
- **Basic knowledge** of X-ray physics (helpful but not required)
- **Familiarity** with NumPy and scientific Python

Clone the Repository
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/b80985/pyXRayLabTool.git
   cd pyXRayLabTool

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

Install in development mode with all dependencies:

.. code-block:: bash

   # Create virtual environment
   python -m venv xraylabtool-dev
   source xraylabtool-dev/bin/activate  # On Windows: xraylabtool-dev\Scripts\activate

   # Install in development mode
   pip install -e .[dev]

   # Verify installation
   pytest tests/ -v
   xraylabtool --version

Development Tools
~~~~~~~~~~~~~~~~~

The development environment includes:

- **pytest**: Test framework
- **black**: Code formatting
- **ruff**: Fast linting
- **mypy**: Type checking
- **pre-commit**: Git hooks for code quality

Set up pre-commit hooks:

.. code-block:: bash

   pre-commit install

Development Workflow
--------------------

Branch Strategy
~~~~~~~~~~~~~~~

We use a simple branching strategy:

- **main**: Stable release code
- **feature/description**: New features
- **fix/description**: Bug fixes
- **docs/description**: Documentation updates

Create a Feature Branch
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git checkout main
   git pull origin main
   git checkout -b feature/my-new-feature

Make Your Changes
~~~~~~~~~~~~~~~~~

1. **Write code** following our coding standards
2. **Add tests** for new functionality
3. **Update documentation** as needed
4. **Run quality checks** before committing

.. code-block:: bash

   # Format code
   black xraylabtool tests *.py

   # Lint code
   ruff check xraylabtool tests

   # Type check
   mypy xraylabtool

   # Run tests
   pytest tests/ -v --cov=xraylabtool

Commit Your Changes
~~~~~~~~~~~~~~~~~~~

Write clear, descriptive commit messages:

.. code-block:: bash

   git add .
   git commit -m "feat: add support for custom atomic data sources

   - Add AtomicDataLoader class for custom data
   - Support multiple file formats (HDF5, CSV, JSON)
   - Include validation for custom atomic data
   - Add documentation and examples"

Push and Create Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git push origin feature/my-new-feature

Then create a Pull Request on GitHub with:
- Clear title describing the change
- Detailed description of what was changed and why
- Link to any related issues
- Screenshots or examples if applicable

Code Standards
--------------

Python Style Guide
~~~~~~~~~~~~~~~~~~

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Imports**: Use absolute imports, organize by standard/third-party/local
- **Docstrings**: NumPy style for all public functions
- **Type hints**: Required for all public APIs

**Example function:**

.. code-block:: python

   def calculate_critical_angle(delta: float) -> tuple[float, float, float]:
       """Calculate critical angle from refractive index decrement.

       Parameters
       ----------
       delta : float
           Real part of refractive index decrement.

       Returns
       -------
       tuple[float, float, float]
           Critical angle in (radians, degrees, milliradians).

       Examples
       --------
       >>> theta_rad, theta_deg, theta_mrad = calculate_critical_angle(1e-5)
       >>> print(f"Critical angle: {theta_deg:.3f}°")
       Critical angle: 0.162°
       """
       import numpy as np

       theta_rad = np.sqrt(2 * delta)
       theta_deg = theta_rad * 180 / np.pi
       theta_mrad = theta_rad * 1000

       return theta_rad, theta_deg, theta_mrad

Error Handling
~~~~~~~~~~~~~~

Use specific exceptions with helpful messages:

.. code-block:: python

   from xraylabtool.validation.exceptions import FormulaError, EnergyError

   def validate_inputs(formula: str, energy: float) -> None:
       if not formula.strip():
           raise FormulaError("Formula cannot be empty")

       if energy <= 0:
           raise EnergyError(
               f"Energy must be positive, got {energy} eV",
               suggestion="Use energy values between 10 eV and 100 keV"
           )

       if energy > 100_000:
           warnings.warn(
               f"Energy {energy} eV is above typical range, "
               "results may be unreliable"
           )

Testing Guidelines
------------------

Test Structure
~~~~~~~~~~~~~~

Tests are organized by functionality:

.. code-block:: text

   tests/
   ├── unit/                     # Unit tests for individual components
   │   ├── test_core.py          # Core calculation tests
   │   ├── test_utils.py         # Utility function tests
   │   └── test_validation.py    # Input validation tests
   ├── integration/              # End-to-end tests
   │   ├── test_cli.py           # CLI command tests
   │   └── test_workflows.py     # Complete workflow tests
   ├── performance/              # Performance regression tests
   │   └── test_benchmarks.py    # Benchmark tests
   └── fixtures/                 # Shared test data and utilities

Writing Tests
~~~~~~~~~~~~~

**Unit Test Example:**

.. code-block:: python

   import pytest
   from xraylabtool.calculators.core import calculate_single_material_properties
   from xraylabtool.validation.exceptions import FormulaError

   class TestSingleMaterialCalculations:
       """Test single material property calculations."""

       def test_silicon_properties(self):
           """Test silicon properties at 8 keV."""
           result = calculate_single_material_properties("Si", 2.33, 8000)

           assert result.formula == "Si"
           assert result.density_g_cm3 == 2.33
           assert result.energy_ev == 8000
           assert abs(result.critical_angle_degrees - 0.158) < 0.001
           assert result.attenuation_length_cm > 5  # Reasonable range

       def test_invalid_formula(self):
           """Test error handling for invalid formulas."""
           with pytest.raises(FormulaError, match="Unknown element"):
               calculate_single_material_properties("XYZ", 1.0, 8000)

       @pytest.mark.parametrize("energy", [0, -1000])
       def test_invalid_energy(self, energy):
           """Test error handling for invalid energies."""
           with pytest.raises(EnergyError):
               calculate_single_material_properties("Si", 2.33, energy)

**Integration Test Example:**

.. code-block:: python

   import subprocess
   import json

   def test_cli_calc_command():
       """Test the calc CLI command."""
       result = subprocess.run([
           "xraylabtool", "calc", "Si",
           "--density", "2.33",
           "--energy", "8000",
           "--output", "json"
       ], capture_output=True, text=True)

       assert result.returncode == 0
       data = json.loads(result.stdout)
       assert len(data) == 1
       assert data[0]["formula"] == "Si"
       assert abs(data[0]["critical_angle_degrees"] - 0.158) < 0.001

**Performance Test Example:**

.. code-block:: python

   import time
   import pytest

   def test_batch_processing_performance():
       """Test that batch processing meets performance requirements."""
       materials = [{"formula": "Si", "density": 2.33}] * 1000
       energies = [8000]

       start_time = time.time()
       results = calculate_xray_properties(materials, energies)
       end_time = time.time()

       # Should process 1000 materials in under 50ms
       assert (end_time - start_time) < 0.05
       assert len(results) == 1000

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest tests/ -v

   # Run specific test categories
   pytest tests/unit/ -v          # Unit tests only
   pytest tests/integration/ -v   # Integration tests only
   pytest tests/performance/ -v   # Performance tests only

   # Run with coverage
   pytest tests/ --cov=xraylabtool --cov-report=html

   # Run tests matching pattern
   pytest tests/ -k "test_silicon" -v

Documentation Standards
-----------------------

Docstring Format
~~~~~~~~~~~~~~~~

Use NumPy-style docstrings:

.. code-block:: python

   def complex_function(param1: str, param2: list[float],
                       param3: bool = True) -> dict:
       """One-line summary of the function.

       Longer description explaining the purpose and behavior.
       Can span multiple paragraphs.

       Parameters
       ----------
       param1 : str
           Description of first parameter.
       param2 : list of float
           Description of second parameter.
       param3 : bool, optional
           Description of optional parameter. Default is True.

       Returns
       -------
       dict
           Description of return value with keys and types.

       Raises
       ------
       ValueError
           When parameter validation fails.
       FormulaError
           When chemical formula is invalid.

       Examples
       --------
       >>> result = complex_function("H2O", [1.0, 2.0])
       >>> print(result["success"])
       True

       See Also
       --------
       related_function : Related functionality

       Notes
       -----
       Additional technical notes or implementation details.

       References
       ----------
       .. [1] Author, "Title", Journal, Year.
       """

API Documentation
~~~~~~~~~~~~~~~~~

All public APIs must be documented:

.. code-block:: python

   # Good - documented public function
   def calculate_properties(formula: str) -> XRayResult:
       """Calculate X-ray properties for a material."""

   # Private functions can have simpler docstrings
   def _internal_helper(data: np.ndarray) -> float:
       """Internal helper for data processing."""

Tutorial Guidelines
~~~~~~~~~~~~~~~~~~~

When writing tutorials:

1. **Start with motivation** - why is this useful?
2. **Include complete examples** - copy-pasteable code
3. **Explain the physics** - scientific context
4. **Show visualizations** - plots and graphs
5. **Provide exercises** - hands-on learning

Performance Considerations
--------------------------

Performance Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

New features should maintain performance standards:

- **Single calculations**: < 0.1 ms
- **Batch processing**: > 100,000 calculations/second
- **Memory usage**: Reasonable scaling with dataset size
- **Cache efficiency**: > 90% hit rate for repeated calculations

Benchmarking
~~~~~~~~~~~~

Include benchmarks for performance-critical code:

.. code-block:: python

   import time
   from xraylabtool.calculators.core import calculate_single_material_properties

   def benchmark_single_calculation():
       """Benchmark single material calculation."""
       n_iterations = 1000

       start_time = time.time()
       for _ in range(n_iterations):
           calculate_single_material_properties("Si", 2.33, 8000)
       end_time = time.time()

       avg_time = (end_time - start_time) / n_iterations
       print(f"Average time per calculation: {avg_time*1000:.3f} ms")
       assert avg_time < 0.0001  # < 0.1 ms requirement

Review Process
--------------

Pull Request Review
~~~~~~~~~~~~~~~~~~~

All code changes go through peer review:

**Review Checklist:**
- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] Performance impact is acceptable
- [ ] Breaking changes are justified and documented

**Review Process:**
1. **Automated checks** must pass (CI/CD)
2. **At least one reviewer** must approve
3. **Maintainer approval** for significant changes
4. **Squash and merge** to main branch

Common Review Comments
~~~~~~~~~~~~~~~~~~~~~~

**Code Quality:**
- "Consider using a more descriptive variable name"
- "This function is getting long, consider splitting it"
- "Add error handling for this edge case"
- "This could be more efficient using vectorization"

**Testing:**
- "Please add a test for the error case"
- "Consider testing with different input types"
- "Add a regression test for this bug fix"
- "Performance test would be valuable here"

**Documentation:**
- "Add a docstring example for this function"
- "Update the API documentation for this change"
- "Consider adding this to the tutorial"
- "Physics explanation would be helpful"

Contributing Atomic Data
-------------------------

Data Quality Standards
~~~~~~~~~~~~~~~~~~~~~~

When contributing atomic scattering factor data:

**Requirements:**
- **Source citation**: Primary literature reference
- **Energy range**: Clearly specified
- **Precision**: Known uncertainty estimates
- **Format**: Compatible with existing data structure
- **Validation**: Cross-checked against existing databases

**Submission Process:**
1. Open an issue describing the data source
2. Submit data in HDF5 or CSV format
3. Include validation script comparing to existing data
4. Provide documentation and references
5. Allow time for scientific review

Data Format
~~~~~~~~~~~

.. code-block:: python

   # Atomic data format
   atomic_data = {
       'element': 'Si',
       'atomic_number': 14,
       'atomic_weight': 28.0855,
       'energies': np.array([...]),  # eV
       'f1_values': np.array([...]), # Real scattering factors
       'f2_values': np.array([...]), # Imaginary scattering factors
       'source': 'Henke et al. (1993)',
       'date_created': '2024-01-01',
       'notes': 'Interpolated from tabulated values'
   }

Community Guidelines
--------------------

Code of Conduct
~~~~~~~~~~~~~~~

We follow the Python Community Code of Conduct:

- **Be respectful** and inclusive
- **Focus on constructive** feedback
- **Help create a welcoming** environment
- **Report inappropriate behavior** to maintainers

Communication Channels
~~~~~~~~~~~~~~~~~~~~~~

**GitHub Issues**: Bug reports, feature requests
**GitHub Discussions**: General questions, ideas
**Pull Request Comments**: Code review discussions
**Email**: For private or sensitive matters

Recognition
~~~~~~~~~~~

Contributors are recognized in:
- **AUTHORS.md** file
- **Release notes** for significant contributions
- **Documentation credits**
- **Conference presentations** when appropriate

Getting Help
------------

If you need help with contributing:

1. **Check existing issues** and pull requests
2. **Read the documentation** thoroughly
3. **Ask in GitHub Discussions**
4. **Contact maintainers** for complex questions

**For scientific questions:**
- Provide context about your application
- Include relevant literature references
- Share test cases or examples
- Be specific about physics requirements

**For technical questions:**
- Include your Python and OS versions
- Provide minimal reproducible examples
- Share error messages and stack traces
- Describe expected vs actual behavior

Thank you for contributing to XRayLabTool! Your efforts help advance X-ray science and support the research community.
