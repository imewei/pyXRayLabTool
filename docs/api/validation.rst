Validation Module
=================

The validation module provides input validation, error handling, and custom exception classes.

.. currentmodule:: xraylabtool.validation

Exception Hierarchy
--------------------

.. automodule:: xraylabtool.exceptions
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Base Exception
~~~~~~~~~~~~~~

.. autoclass:: xraylabtool.exceptions.XRayLabToolError
   :members:
   :show-inheritance:
   :no-index:

   The base exception class for all XRayLabTool-specific errors. All custom exceptions inherit from this class, making it easy to catch any XRayLabTool-related error.

   **Example:**

   .. code-block:: python

      try:
          result = calculate_single_material_properties("InvalidFormula", 1.0, 8000)
      except XRayLabToolError as e:
          print(f"XRayLabTool error: {e}")

Specific Exceptions
~~~~~~~~~~~~~~~~~~~

.. autoclass:: xraylabtool.exceptions.ValidationError
   :members:
   :show-inheritance:
   :no-index:

   Raised when input validation fails.

.. autoclass:: xraylabtool.exceptions.FormulaError
   :members:
   :show-inheritance:
   :no-index:

   Raised when chemical formula parsing or validation fails.

   **Common causes:**
   - Unknown chemical elements
   - Invalid formula syntax
   - Unsupported element combinations

   **Example:**

   .. code-block:: python

      try:
          from xraylabtool.utils import parse_formula
          composition = parse_formula("XYZ123")  # Invalid element
      except FormulaError as e:
          print(f"Formula error: {e}")

.. autoclass:: xraylabtool.exceptions.EnergyError
   :members:
   :show-inheritance:
   :no-index:

   Raised when energy values are invalid or out of supported range.

   **Common causes:**
   - Negative or zero energy values
   - Energy outside supported range (typically 10 eV - 100 keV)
   - Invalid energy array or range specification

   **Example:**

   .. code-block:: python

      try:
          result = calculate_single_material_properties("Si", 2.33, -1000)  # Negative energy
      except EnergyError as e:
          print(f"Energy error: {e}")

.. autoclass:: xraylabtool.exceptions.CalculationError
   :members:
   :show-inheritance:
   :no-index:

   Raised when X-ray property calculations fail due to numerical issues or invalid parameters.

   **Common causes:**
   - Numerical instability in calculations
   - Invalid density values
   - Missing atomic data
   - Convergence failures

Input Validators
----------------

.. automodule:: xraylabtool.validation.validators
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Validation Functions
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: xraylabtool.validation.validators.validate_energy_range
   :no-index:

   Validate X-ray energy values.

   **Parameters:**
   - ``energies`` (float or np.ndarray): Energy value(s) in keV
   - ``min_energy`` (float): Minimum allowed energy in keV (default: 0.1)
   - ``max_energy`` (float): Maximum allowed energy in keV (default: 100.0)

   **Returns:**
   - ``np.ndarray``: Validated energy array

   **Raises:**
   - ``EnergyError``: If energies are invalid

   **Example:**

   .. code-block:: python

      from xraylabtool.validation.validators import validate_energy_range
      import numpy as np

      # Valid energies
      energies = validate_energy_range(8.0)  # Single energy
      energies = validate_energy_range([5.0, 8.0, 10.0])  # Multiple energies
      energies = validate_energy_range(np.array([1.0, 2.0, 3.0]))  # Array

      # Invalid energies will raise EnergyError
      # validate_energy_range(-1.0)  # Negative energy
      # validate_energy_range(200.0)  # Above max energy

.. autofunction:: xraylabtool.validation.validators.validate_chemical_formula
   :no-index:

   Validate and parse a chemical formula.

   **Parameters:**
   - ``formula`` (str): Chemical formula string (e.g., "SiO2", "Ca0.5Sr0.5TiO3")

   **Returns:**
   - ``dict``: Dictionary mapping element symbols to their quantities

   **Raises:**
   - ``FormulaError``: If formula is invalid

   **Example:**

   .. code-block:: python

      from xraylabtool.validation.validators import validate_chemical_formula

      # Valid formulas
      elements = validate_chemical_formula("SiO2")
      print(elements)  # {'Si': 1.0, 'O': 2.0}

      elements = validate_chemical_formula("Al2O3")
      print(elements)  # {'Al': 2.0, 'O': 3.0}

      # Fractional compositions
      elements = validate_chemical_formula("Ca0.5Sr0.5TiO3")
      print(elements)  # {'Ca': 0.5, 'Sr': 0.5, 'Ti': 1.0, 'O': 3.0}

.. autofunction:: xraylabtool.validation.validators.validate_density
   :no-index:

   Validate material density value.

   **Parameters:**
   - ``density`` (float): Density in g/cm³
   - ``min_density`` (float): Minimum allowed density (default: 0.001)
   - ``max_density`` (float): Maximum allowed density (default: 30.0)

   **Returns:**
   - ``float``: Validated density value

   **Raises:**
   - ``ValidationError``: If density is invalid

   **Example:**

   .. code-block:: python

      from xraylabtool.validation.validators import validate_density

      # Valid densities
      density = validate_density(2.33)    # Silicon
      density = validate_density(8.96)    # Copper
      density = validate_density(0.001)   # Gas-phase materials

      # Invalid densities will raise ValidationError
      # validate_density(-1.0)     # Negative density
      # validate_density(0.0)      # Zero density

.. autofunction:: xraylabtool.validation.validators.validate_calculation_parameters
   :no-index:

   Validate all parameters for X-ray calculations.

   **Parameters:**
   - ``formula`` (str): Chemical formula
   - ``energies`` (float or np.ndarray): Energy values in keV
   - ``density`` (float): Material density in g/cm³

   **Returns:**
   - ``tuple``: Validated (formula, energies, density)

   **Raises:**
   - ``ValidationError``: If any parameters are invalid

   **Example:**

   .. code-block:: python

      from xraylabtool.validation.validators import validate_calculation_parameters

      # Validate all parameters at once
      formula, energies, density = validate_calculation_parameters(
          "SiO2", 8.0, 2.20
      )
      print(f"Formula: {formula}, Energy: {energies}, Density: {density}")

Error Context and Suggestions
-----------------------------

XRayLabTool exceptions provide detailed context and suggestions for resolution:

.. code-block:: python

   try:
       result = calculate_single_material_properties("Si123", 2.33, 8000)
   except FormulaError as e:
       print(f"Error: {e}")
       print(f"Suggestion: {e.suggestion}")
       # Output:
       # Error: Invalid formula 'Si123': numbers should follow elements
       # Suggestion: Use format like 'SiO2' or 'Al2O3'

Internal Validation Utilities
-----------------------------

The validation module also includes internal helper functions that are used by the main validation functions but are not part of the public API:

- ``_parse_formula()``: Internal formula parsing logic
- ``_get_valid_element_symbols()``: Returns a set of valid element symbols for validation

Best Practices
--------------

**1. Always Validate Input:**

.. code-block:: python

   from xraylabtool.validation.validators import validate_chemical_formula, validate_energy_range

   def safe_calculation(formula, density, energy):
       validate_chemical_formula(formula)
       validate_energy_range(energy)
       # Proceed with calculation...

**2. Handle Specific Exceptions:**

.. code-block:: python

   from xraylabtool.exceptions import FormulaError, EnergyError

   try:
       result = calculate_single_material_properties(formula, density, energy)
   except FormulaError:
       print("Please check your chemical formula")
   except EnergyError:
       print("Please provide a valid energy value")
   except Exception as e:
       print(f"Unexpected error: {e}")

**3. Use Complete Parameter Validation:**

.. code-block:: python

   from xraylabtool.validation.validators import validate_calculation_parameters

   materials = [
       {"formula": "Si", "density": 2.33},
       {"formula": "InvalidElement", "density": 1.0}
   ]

   # Validate all materials first
   valid_materials = []
   for material in materials:
       try:
           formula, energies, density = validate_calculation_parameters(
               material["formula"], 8.0, material["density"]
           )
           valid_materials.append({
               "formula": formula,
               "density": density,
               "energies": energies
           })
       except (FormulaError, ValidationError) as e:
           print(f"Skipping invalid material: {e}")

   # Process only valid materials
   results = calculate_xray_properties(valid_materials, energy=8000)
