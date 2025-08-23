Field Name Migration Guide
===========================

XRayLabTool v0.1.5 introduced new descriptive field names for the ``XRayResult`` dataclass to improve code clarity and maintainability. This guide helps users transition from legacy CamelCase names to the new snake_case names.

Overview of Changes
-------------------

The new field names provide better clarity about units and meaning:

.. list-table:: Field Name Changes
   :widths: 40 40 20
   :header-rows: 1

   * - Legacy Name (Deprecated)
     - New Name (Recommended) 
     - Status
   * - ``result.Formula``
     - ``result.formula``
     - ✅ Available
   * - ``result.MW``
     - ``result.molecular_weight_g_mol``
     - ✅ Available
   * - ``result.Number_Of_Electrons``
     - ``result.total_electrons``
     - ✅ Available
   * - ``result.Density``
     - ``result.density_g_cm3``
     - ✅ Available
   * - ``result.Electron_Density``
     - ``result.electron_density_per_ang3``
     - ✅ Available
   * - ``result.Energy``
     - ``result.energy_kev``
     - ✅ Available
   * - ``result.Wavelength``
     - ``result.wavelength_angstrom``
     - ✅ Available
   * - ``result.Dispersion``
     - ``result.dispersion_delta``
     - ✅ Available
   * - ``result.Absorption``
     - ``result.absorption_beta``
     - ✅ Available
   * - ``result.f1``
     - ``result.scattering_factor_f1``
     - ✅ Available
   * - ``result.f2``
     - ``result.scattering_factor_f2``
     - ✅ Available
   * - ``result.Critical_Angle``
     - ``result.critical_angle_degrees``
     - ✅ Available
   * - ``result.Attenuation_Length``
     - ``result.attenuation_length_cm``
     - ✅ Available
   * - ``result.reSLD``
     - ``result.real_sld_per_ang2``
     - ✅ Available
   * - ``result.imSLD``
     - ``result.imaginary_sld_per_ang2``
     - ✅ Available

Backward Compatibility
----------------------

**Legacy field names still work!** They are implemented as deprecated properties that emit ``DeprecationWarning`` messages but return the correct values.

.. code-block:: python

   import xraylabtool as xlt
   
   result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)
   
   # This still works but emits a warning
   print(result.Critical_Angle[0])  # ⚠️ DeprecationWarning
   
   # This is the recommended approach
   print(result.critical_angle_degrees[0])  # ✅ No warning

Migration Strategy
------------------

**Step 1: Identify Usage**

Search your codebase for legacy field names:

.. code-block:: bash

   # Find all occurrences of legacy field names
   grep -r "\.Formula\|\.MW\|\.Critical_Angle\|\.Dispersion" your_project/
   grep -r "\.Absorption\|\.Wavelength\|\.Energy\|\.f1\|\.f2" your_project/

**Step 2: Update Gradually**

Replace legacy names one section at a time:

.. code-block:: python

   # Before (legacy)
   print(f"Critical angle: {result.Critical_Angle[0]:.3f}°")
   print(f"Molecular weight: {result.MW:.2f} g/mol")
   print(f"Dispersion: {result.Dispersion[0]:.2e}")
   
   # After (recommended)
   print(f"Critical angle: {result.critical_angle_degrees[0]:.3f}°")
   print(f"Molecular weight: {result.molecular_weight_g_mol:.2f} g/mol")
   print(f"Dispersion: {result.dispersion_delta[0]:.2e}")

**Step 3: Test**

Ensure your code works correctly with new field names:

.. code-block:: python

   import xraylabtool as xlt
   
   # Test basic functionality
   result = xlt.calculate_single_material_properties("Si", 10.0, 2.33)
   assert hasattr(result, 'critical_angle_degrees')
   assert hasattr(result, 'molecular_weight_g_mol')
   
   print("Migration successful!")

**Step 4: Clean Up**

Remove any deprecation warning suppressions once migration is complete.

Common Migration Patterns
--------------------------

**Single Property Access:**

.. code-block:: python

   # Legacy
   critical_angle = result.Critical_Angle[0]
   molecular_weight = result.MW
   
   # New
   critical_angle = result.critical_angle_degrees[0]
   molecular_weight = result.molecular_weight_g_mol

**Multiple Properties:**

.. code-block:: python

   # Legacy
   properties = {
       'formula': result.Formula,
       'mw': result.MW,
       'critical_angle': result.Critical_Angle[0],
       'dispersion': result.Dispersion[0]
   }
   
   # New
   properties = {
       'formula': result.formula,
       'mw': result.molecular_weight_g_mol,
       'critical_angle': result.critical_angle_degrees[0],
       'dispersion': result.dispersion_delta[0]
   }

**Array Processing:**

.. code-block:: python

   import numpy as np
   
   # Legacy
   energies = result.Energy
   dispersions = result.Dispersion
   angles = result.Critical_Angle
   
   # New
   energies = result.energy_kev
   dispersions = result.dispersion_delta
   angles = result.critical_angle_degrees

**Plotting Code:**

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Legacy
   plt.loglog(result.Energy, result.Dispersion, label='δ')
   plt.loglog(result.Energy, result.Absorption, label='β')
   plt.xlabel('Energy (keV)')
   
   # New
   plt.loglog(result.energy_kev, result.dispersion_delta, label='δ')
   plt.loglog(result.energy_kev, result.absorption_beta, label='β')
   plt.xlabel('Energy (keV)')

Handling Deprecation Warnings
------------------------------

**Temporary Suppression (Not Recommended for Production):**

.. code-block:: python

   import warnings
   
   # Suppress only XRayLabTool deprecation warnings during migration
   with warnings.catch_warnings():
       warnings.filterwarnings("ignore", category=DeprecationWarning, 
                             message=".*deprecated.*")
       # Your legacy code here
       print(f"Result: {result.Critical_Angle[0]}")

**Filtering in Test Suites:**

.. code-block:: python

   import pytest
   
   @pytest.mark.filterwarnings("ignore:.*deprecated.*:DeprecationWarning")
   def test_legacy_field_access():
       # Test that legacy fields still work during migration period
       result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)
       assert result.Critical_Angle[0] > 0

Benefits of New Field Names
----------------------------

**1. Clear Units:**

.. code-block:: python

   # Legacy - units unclear
   result.MW  # g/mol? u? kg/mol?
   result.Critical_Angle  # degrees? radians?
   result.reSLD  # Å⁻²? cm⁻²?
   
   # New - units explicit
   result.molecular_weight_g_mol  # Clearly g/mol
   result.critical_angle_degrees  # Clearly degrees
   result.real_sld_per_ang2  # Clearly Å⁻²

**2. Better Autocompletion:**

New field names provide better IDE autocompletion and documentation.

**3. Self-Documenting Code:**

.. code-block:: python

   # Legacy - requires comments or documentation lookup
   angle = result.Critical_Angle[0]  # Is this in degrees?
   
   # New - self-documenting
   angle = result.critical_angle_degrees[0]  # Obviously degrees

**4. Consistent Naming:**

All field names follow Python naming conventions (snake_case) and include units where applicable.

Field Mapping Reference
-----------------------

For quick reference during migration:

**Material Properties:**

.. code-block:: text

   # Legacy → New
   result.Formula → result.formula
   result.MW → result.molecular_weight_g_mol
   result.Number_Of_Electrons → result.total_electrons
   result.Density → result.density_g_cm3
   result.Electron_Density → result.electron_density_per_ang3

**X-ray Properties (Arrays):**

.. code-block:: text

   # Legacy → New
   result.Energy → result.energy_kev
   result.Wavelength → result.wavelength_angstrom
   result.Dispersion → result.dispersion_delta
   result.Absorption → result.absorption_beta
   result.f1 → result.scattering_factor_f1
   result.f2 → result.scattering_factor_f2

**Derived Quantities (Arrays):**

.. code-block:: text

   # Legacy → New
   result.Critical_Angle → result.critical_angle_degrees
   result.Attenuation_Length → result.attenuation_length_cm
   result.reSLD → result.real_sld_per_ang2
   result.imSLD → result.imaginary_sld_per_ang2

Timeline and Support
--------------------

**Current Status (v0.1.5+):**
   - ✅ New field names available
   - ✅ Legacy field names still work with warnings
   - ✅ All functionality identical

**Future Plans:**
   - Legacy field names will be supported for several more versions
   - Deprecation warnings help identify code that needs updating
   - Eventually legacy names may be removed (with advance notice)

**Recommendation:**
   - Start migration to new field names when convenient
   - No urgent action required - legacy code continues to work
   - New projects should use new field names from the start

Getting Help
------------

If you encounter issues during migration:

1. Check the field mapping table above
2. Review the examples in this documentation
3. Use IDE autocompletion to discover new field names
4. File an issue on GitHub if you need assistance

The migration is designed to be seamless - your existing code will continue to work while you transition to the clearer, more descriptive field names at your own pace.
