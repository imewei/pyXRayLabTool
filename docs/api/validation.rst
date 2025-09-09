Validation Module
=================

The validation module provides input validation, error handling, and custom exception classes.

.. currentmodule:: xraylabtool.validation

Exception Hierarchy
--------------------

.. automodule:: xraylabtool.validation.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Base Exception
~~~~~~~~~~~~~~

.. autoclass:: xraylabtool.validation.exceptions.XRayLabToolError
   :members:
   :show-inheritance:

   The base exception class for all XRayLabTool-specific errors. All custom exceptions inherit from this class, making it easy to catch any XRayLabTool-related error.

   **Example:**

   .. code-block:: python

      try:
          result = calculate_single_material_properties("InvalidFormula", 1.0, 8000)
      except XRayLabToolError as e:
          print(f"XRayLabTool error: {e}")

Specific Exceptions
~~~~~~~~~~~~~~~~~~~

.. autoclass:: xraylabtool.validation.exceptions.ValidationError
   :members:
   :show-inheritance:

   Raised when input validation fails.

.. autoclass:: xraylabtool.validation.exceptions.FormulaError
   :members:
   :show-inheritance:

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

.. autoclass:: xraylabtool.validation.exceptions.EnergyError
   :members:
   :show-inheritance:

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

.. autoclass:: xraylabtool.validation.exceptions.CalculationError
   :members:
   :show-inheritance:

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

Validation Functions
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: xraylabtool.validation.validators.validate_formula

   Validates chemical formula syntax and element availability.

   **Parameters:**
   - ``formula`` (str): Chemical formula to validate

   **Returns:**
   - ``bool``: True if valid

   **Raises:**
   - ``FormulaError``: If formula is invalid

   **Example:**

   .. code-block:: python

      from xraylabtool.validation.validators import validate_formula
      
      # Valid formulas
      validate_formula("Si")        # True
      validate_formula("SiO2")      # True
      validate_formula("Al2O3")     # True
      
      # Invalid formulas
      validate_formula("XYZ")       # Raises FormulaError
      validate_formula("Si-O2")     # Raises FormulaError (invalid syntax)

.. autofunction:: xraylabtool.validation.validators.validate_energy

   Validates X-ray energy values and ranges.

   **Parameters:**
   - ``energy`` (float or array-like): Energy value(s) in eV

   **Returns:**
   - ``bool``: True if valid

   **Raises:**
   - ``EnergyError``: If energy is invalid

   **Example:**

   .. code-block:: python

      from xraylabtool.validation.validators import validate_energy
      
      # Valid energies
      validate_energy(8000)           # True
      validate_energy([5000, 8000])   # True
      validate_energy(range(1000, 10000, 1000))  # True
      
      # Invalid energies
      validate_energy(-1000)          # Raises EnergyError
      validate_energy(0)              # Raises EnergyError

.. autofunction:: xraylabtool.validation.validators.validate_density

   Validates material density values.

   **Parameters:**
   - ``density`` (float): Density in g/cm³

   **Returns:**
   - ``bool``: True if valid

   **Raises:**
   - ``ValidationError``: If density is invalid

   **Example:**

   .. code-block:: python

      from xraylabtool.validation.validators import validate_density
      
      # Valid densities
      validate_density(2.33)     # Silicon
      validate_density(8.96)     # Copper
      validate_density(0.001)    # Gas-phase materials
      
      # Invalid densities
      validate_density(-1.0)     # Raises ValidationError
      validate_density(0)        # Raises ValidationError

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

Validation Utilities
--------------------

.. autofunction:: xraylabtool.validation.validators.is_valid_element

   Check if a string represents a valid chemical element.

   **Example:**

   .. code-block:: python

      from xraylabtool.validation.validators import is_valid_element
      
      is_valid_element("Si")    # True
      is_valid_element("XYZ")   # False

.. autofunction:: xraylabtool.validation.validators.get_supported_elements

   Get list of all supported chemical elements.

   **Returns:**
   - ``list``: List of element symbols

   **Example:**

   .. code-block:: python

      from xraylabtool.validation.validators import get_supported_elements
      
      elements = get_supported_elements()
      print(f"Total supported elements: {len(elements)}")
      print(f"First 10: {elements[:10]}")

Best Practices
--------------

**1. Always Validate Input:**

.. code-block:: python

   from xraylabtool.validation.validators import validate_formula, validate_energy
   
   def safe_calculation(formula, density, energy):
       validate_formula(formula)
       validate_energy(energy)
       # Proceed with calculation...

**2. Handle Specific Exceptions:**

.. code-block:: python

   from xraylabtool.validation.exceptions import FormulaError, EnergyError
   
   try:
       result = calculate_single_material_properties(formula, density, energy)
   except FormulaError:
       print("Please check your chemical formula")
   except EnergyError:
       print("Please provide a valid energy value")
   except Exception as e:
       print(f"Unexpected error: {e}")

**3. Use Validation Before Batch Processing:**

.. code-block:: python

   materials = [
       {"formula": "Si", "density": 2.33},
       {"formula": "InvalidElement", "density": 1.0}
   ]
   
   # Validate all materials first
   valid_materials = []
   for material in materials:
       try:
           validate_formula(material["formula"])
           valid_materials.append(material)
       except FormulaError as e:
           print(f"Skipping invalid material: {e}")
   
   # Process only valid materials
   results = calculate_xray_properties(valid_materials, energy=8000)