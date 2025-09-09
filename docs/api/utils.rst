Utilities Module
================

The utils module provides utility functions for formula parsing, unit conversions, and mathematical operations.

.. currentmodule:: xraylabtool.utils

Formula Parsing
---------------

.. autofunction:: xraylabtool.utils.parse_formula

   Parse chemical formulas into element composition dictionaries.

   **Parameters:**
   - ``formula`` (str): Chemical formula to parse

   **Returns:**
   - ``dict``: Element symbol to count mapping

   **Raises:**
   - ``FormulaError``: If formula syntax is invalid

   **Supported Formula Types:**

   .. list-table::
      :header-rows: 1
      :widths: 30 40 30

      * - Formula Type
        - Example
        - Parsed Result
      * - Simple elements
        - "Si", "Al", "Cu"
        - {"Si": 1}
      * - Simple compounds
        - "SiO2", "Al2O3"
        - {"Si": 1, "O": 2}
      * - Complex compounds
        - "Ca5(PO4)3F"
        - {"Ca": 5, "P": 3, "O": 12, "F": 1}
      * - Hydrated compounds
        - "CuSO4·5H2O"
        - {"Cu": 1, "S": 1, "O": 9, "H": 10}
      * - Mixed parentheses
        - "Al2(SO4)3·18H2O"
        - {"Al": 2, "S": 3, "O": 30, "H": 36}

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import parse_formula
      
      # Simple compounds
      composition = parse_formula("SiO2")
      print(composition)  # {'Si': 1, 'O': 2}
      
      # Complex compounds
      composition = parse_formula("Ca5(PO4)3F")
      print(composition)  # {'Ca': 5, 'P': 3, 'O': 12, 'F': 1}
      
      # Hydrated compounds
      composition = parse_formula("CuSO4·5H2O")
      print(composition)  # {'Cu': 1, 'S': 1, 'O': 9, 'H': 10}

.. autofunction:: xraylabtool.utils.calculate_molecular_weight

   Calculate molecular weight from chemical formula.

   **Parameters:**
   - ``formula`` (str): Chemical formula

   **Returns:**
   - ``float``: Molecular weight in g/mol

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import calculate_molecular_weight
      
      mw = calculate_molecular_weight("SiO2")
      print(f"SiO2 molecular weight: {mw:.2f} g/mol")
      # Output: SiO2 molecular weight: 60.08 g/mol

.. autofunction:: xraylabtool.utils.normalize_formula

   Normalize chemical formula to standard format.

   **Parameters:**
   - ``formula`` (str): Input formula (may have inconsistent formatting)

   **Returns:**
   - ``str``: Normalized formula

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import normalize_formula
      
      # Normalize various input formats
      print(normalize_formula("si o2"))      # "SiO2"
      print(normalize_formula("AL2O3"))      # "Al2O3"  
      print(normalize_formula("Ca(OH)2"))    # "Ca(OH)2"

Unit Conversions
----------------

.. autofunction:: xraylabtool.utils.energy_to_wavelength

   Convert X-ray photon energy to wavelength.

   **Parameters:**
   - ``energy`` (float or array-like): Energy in eV

   **Returns:**
   - ``float or array``: Wavelength in Angstroms

   **Formula:**

   .. math::

      \\lambda = \\frac{hc}{E} = \\frac{12398.4}{E_{eV}} \\text{ Å}

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import energy_to_wavelength
      import numpy as np
      
      # Single energy
      wavelength = energy_to_wavelength(8000)  # 8 keV
      print(f"8 keV = {wavelength:.3f} Å")
      # Output: 8 keV = 1.550 Å
      
      # Energy array
      energies = np.array([5000, 8000, 10000, 15000])
      wavelengths = energy_to_wavelength(energies)
      for e, w in zip(energies, wavelengths):
          print(f"{e} eV = {w:.3f} Å")

.. autofunction:: xraylabtool.utils.wavelength_to_energy

   Convert X-ray wavelength to photon energy.

   **Parameters:**
   - ``wavelength`` (float or array-like): Wavelength in Angstroms

   **Returns:**
   - ``float or array``: Energy in eV

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import wavelength_to_energy
      
      # Single wavelength
      energy = wavelength_to_energy(1.55)  # Copper K-alpha
      print(f"1.55 Å = {energy:.0f} eV")
      # Output: 1.55 Å = 8000 eV

.. autofunction:: xraylabtool.utils.angle_rad_to_deg

   Convert angles from radians to degrees.

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import angle_rad_to_deg
      import numpy as np
      
      angle_deg = angle_rad_to_deg(np.pi/4)
      print(f"π/4 radians = {angle_deg} degrees")
      # Output: π/4 radians = 45.0 degrees

.. autofunction:: xraylabtool.utils.angle_deg_to_rad

   Convert angles from degrees to radians.

.. autofunction:: xraylabtool.utils.mrad_to_degrees

   Convert milliradians to degrees.

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import mrad_to_degrees
      
      angle_deg = mrad_to_degrees(10.0)  # 10 mrad
      print(f"10 mrad = {angle_deg:.3f}°")
      # Output: 10 mrad = 0.573°

Mathematical Utilities
----------------------

.. autofunction:: xraylabtool.utils.interpolate_linear

   Linear interpolation for atomic data.

   **Parameters:**
   - ``x`` (array): X values (sorted)
   - ``y`` (array): Y values
   - ``xi`` (float or array): Interpolation points

   **Returns:**
   - ``float or array``: Interpolated values

.. autofunction:: xraylabtool.utils.safe_divide

   Division with proper handling of zero divisors.

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import safe_divide
      
      result = safe_divide(10.0, 0.0, default=float('inf'))
      print(result)  # inf

.. autofunction:: xraylabtool.utils.ensure_array

   Convert input to numpy array with proper shape handling.

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import ensure_array
      
      # Single value -> array
      arr = ensure_array(8000)
      print(arr.shape)  # (1,)
      
      # List -> array
      arr = ensure_array([5000, 8000, 10000])
      print(arr.shape)  # (3,)

String Formatting
-----------------

.. autofunction:: xraylabtool.utils.format_scientific

   Format numbers in scientific notation with proper precision.

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import format_scientific
      
      formatted = format_scientific(1.23456789e-6, precision=3)
      print(formatted)  # "1.235e-06"

.. autofunction:: xraylabtool.utils.format_formula_html

   Convert chemical formula to HTML with proper subscripts.

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import format_formula_html
      
      html = format_formula_html("Ca5(PO4)3F")
      print(html)  # "Ca<sub>5</sub>(PO<sub>4</sub>)<sub>3</sub>F"

.. autofunction:: xraylabtool.utils.format_formula_latex

   Convert chemical formula to LaTeX format.

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import format_formula_latex
      
      latex = format_formula_latex("Al2O3")
      print(latex)  # "Al_{2}O_{3}"

Data Processing
---------------

.. autofunction:: xraylabtool.utils.parse_energy_range

   Parse energy range specifications from strings.

   **Supported Formats:**
   - Single values: "8000"
   - Comma-separated: "5000,8000,10000"
   - Ranges: "1000-20000:100" (start-stop:step)
   - Mixed: "5000,8000-12000:1000,15000"

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import parse_energy_range
      
      # Single value
      energies = parse_energy_range("8000")
      print(energies)  # [8000.0]
      
      # Range specification
      energies = parse_energy_range("5000-10000:1000")
      print(energies)  # [5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0]
      
      # Mixed format
      energies = parse_energy_range("5000,8000-10000:1000,15000")
      print(energies)  # [5000.0, 8000.0, 9000.0, 10000.0, 15000.0]

.. autofunction:: xraylabtool.utils.parse_density_with_units

   Parse density values with optional units.

   **Supported Units:**
   - g/cm³ (default)
   - kg/m³
   - g/mL

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import parse_density_with_units
      
      # Default units (g/cm³)
      density = parse_density_with_units("2.33")
      print(density)  # 2.33
      
      # With units
      density = parse_density_with_units("2330 kg/m³")
      print(density)  # 2.33 (converted to g/cm³)

Validation Helpers
------------------

.. autofunction:: xraylabtool.utils.is_valid_element_symbol

   Check if string is a valid chemical element symbol.

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import is_valid_element_symbol
      
      print(is_valid_element_symbol("Si"))   # True
      print(is_valid_element_symbol("XYZ"))  # False

.. autofunction:: xraylabtool.utils.clean_formula_input

   Clean and standardize formula input from various sources.

   **Features:**
   - Remove extra whitespace
   - Standardize element capitalization
   - Handle Unicode characters
   - Remove invalid characters

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import clean_formula_input
      
      # Clean messy input
      clean = clean_formula_input("  si o2  ")
      print(clean)  # "SiO2"
      
      # Handle Unicode
      clean = clean_formula_input("Al₂O₃")
      print(clean)  # "Al2O3"

Performance Utilities
---------------------

.. autofunction:: xraylabtool.utils.chunk_array

   Split large arrays into manageable chunks for processing.

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import chunk_array
      import numpy as np
      
      large_array = np.arange(10000)
      chunks = list(chunk_array(large_array, chunk_size=1000))
      print(f"Split into {len(chunks)} chunks")
      # Output: Split into 10 chunks

.. autofunction:: xraylabtool.utils.estimate_memory_usage

   Estimate memory requirements for calculations.

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import estimate_memory_usage
      
      # Estimate memory for batch calculation
      n_materials = 1000
      n_energies = 100
      memory_mb = estimate_memory_usage(n_materials, n_energies)
      print(f"Estimated memory usage: {memory_mb} MB")

Constants and Reference Data
----------------------------

.. autodata:: xraylabtool.utils.ATOMIC_WEIGHTS

   Dictionary of atomic weights for all supported elements.

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import ATOMIC_WEIGHTS
      
      print(f"Silicon atomic weight: {ATOMIC_WEIGHTS['Si']} g/mol")
      # Output: Silicon atomic weight: 28.0855 g/mol

.. autodata:: xraylabtool.utils.ELEMENT_NAMES

   Dictionary mapping element symbols to full names.

.. autodata:: xraylabtool.utils.COMMON_FORMULAS

   Dictionary of commonly used material formulas with standard densities.

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import COMMON_FORMULAS
      
      for formula, density in COMMON_FORMULAS.items():
          print(f"{formula}: {density} g/cm³")