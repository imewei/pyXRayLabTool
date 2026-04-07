Contributing Guide
==================

Contributions welcome: code, documentation, scientific validation, and community support.

Development Setup
-----------------

**Prerequisites:** Python 3.12+, Git

.. code-block:: bash

   git clone https://github.com/b80985/pyXRayLabTool.git
   cd pyXRayLabTool
   python -m venv xraylabtool-dev
   source xraylabtool-dev/bin/activate
   pip install -e .[dev]
   pre-commit install
   pytest tests/ -v

Development Workflow
--------------------

**Branch naming:** feature/description, fix/description, docs/description

.. code-block:: bash

   git checkout main && git pull origin main
   git checkout -b feature/my-new-feature
   # Make changes, add tests, update docs
   black xraylabtool tests *.py
   ruff check xraylabtool tests
   mypy xraylabtool
   pytest tests/ -v --cov=xraylabtool
   git add . && git commit -m "feat: descriptive message"
   git push origin feature/my-new-feature

Code Standards
--------------

**Style:** PEP 8, 88 char lines, NumPy docstrings, type hints required

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
       """
       import numpy as np
       theta_rad = np.sqrt(2 * delta)
       return theta_rad, theta_rad * 180 / np.pi, theta_rad * 1000

   # Error handling
   from xraylabtool.exceptions import FormulaError, EnergyError

   def validate_inputs(formula: str, energy: float) -> None:
       if not formula.strip():
           raise FormulaError("Formula cannot be empty")
       if energy <= 0:
           raise EnergyError(f"Energy must be positive, got {energy} eV")

Testing Guidelines
------------------

**Structure:** tests/unit/, tests/integration/, tests/performance/

.. code-block:: python

   # Unit test example
   import pytest
   from xraylabtool.calculators.core import calculate_single_material_properties
   from xraylabtool.exceptions import FormulaError

   def test_silicon_properties():
       result = calculate_single_material_properties("Si", 2.33, 8000)
       assert result.formula == "Si"
       assert abs(result.critical_angle_degrees - 0.158) < 0.001

   def test_invalid_formula():
       with pytest.raises(FormulaError):
           calculate_single_material_properties("XYZ", 1.0, 8000)

.. code-block:: bash

   pytest tests/ -v                    # All tests
   pytest tests/unit/ -v               # Unit tests
   pytest tests/ --cov=xraylabtool     # With coverage

Documentation Standards
-----------------------

**Docstrings:** NumPy style for all public functions

.. code-block:: python

   def calculate_properties(formula: str, energy: float) -> dict:
       """Calculate X-ray properties for a material.

       Parameters
       ----------
       formula : str
           Chemical formula (e.g., "SiO2").
       energy : float
           X-ray energy in eV.

       Returns
       -------
       dict
           Properties including critical angle and attenuation.
       """

Performance Requirements
------------------------

- Single calculations: < 0.1 ms
- Batch processing: > 100,000 calculations/second
- Cache efficiency: > 90% hit rate

.. code-block:: python

   # Benchmark example
   import time
   start_time = time.time()
   for _ in range(1000):
       calculate_single_material_properties("Si", 2.33, 8000)
   avg_time = (time.time() - start_time) / 1000
   assert avg_time < 0.0001

Review Process
--------------

**Requirements:**
- [ ] Code follows style guidelines
- [ ] Tests included and passing
- [ ] Documentation updated
- [ ] Performance acceptable

**Process:** Automated checks → reviewer approval → maintainer approval → squash merge

Contributing Atomic Data
------------------------

**Requirements:** Source citation, energy range, precision estimates, HDF5/CSV format

.. code-block:: python

   atomic_data = {
       'element': 'Si',
       'atomic_number': 14,
       'energies': np.array([...]),     # eV
       'f1_values': np.array([...]),    # Real factors
       'f2_values': np.array([...]),    # Imaginary factors
       'source': 'Henke et al. (1993)'
   }

Getting Help
------------

**Communication:** GitHub Issues (bugs/features), GitHub Discussions (questions)

**When asking for help:**
- Include Python/OS versions
- Provide minimal reproducible examples
- Share error messages and stack traces

Contributors are recognized in AUTHORS.md and release notes.
