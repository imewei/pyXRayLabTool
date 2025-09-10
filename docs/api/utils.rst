Utilities Module
================

The utils module provides utility functions for formula parsing, unit conversions, and mathematical operations.

.. automodule:: xraylabtool.utils
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Chemical Formula Parsing
------------------------

.. autofunction:: xraylabtool.utils.parse_formula

   Parse a chemical formula string into element symbols and their counts.

   **Parameters:**
   - ``formula_str`` (str): Chemical formula string (e.g., "SiO2", "Al2O3", "H0.5He0.5")

   **Returns:**
   - ``tuple``: (element_symbols, element_counts) where element_symbols is a list of element symbols and element_counts is a list of corresponding stoichiometric counts

   **Examples:**

   .. code-block:: python

      from xraylabtool.utils import parse_formula

      # Simple compounds
      symbols, counts = parse_formula("SiO2")
      print(symbols, counts)  # ['Si', 'O'] [1.0, 2.0]

      # Complex compounds
      symbols, counts = parse_formula("Al2O3")
      print(symbols, counts)  # ['Al', 'O'] [2.0, 3.0]

      # Fractional compositions
      symbols, counts = parse_formula("H0.5He0.5")
      print(symbols, counts)  # ['H', 'He'] [0.5, 0.5]

Atomic Data Functions
---------------------

.. autofunction:: xraylabtool.utils.get_atomic_number

   Get atomic number for given element symbol with LRU caching.

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import get_atomic_number

      atomic_num = get_atomic_number('Si')
      print(atomic_num)  # 14

.. autofunction:: xraylabtool.utils.get_atomic_weight

   Get atomic weight for given element symbol with LRU caching.

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import get_atomic_weight

      atomic_weight = get_atomic_weight('Si')
      print(f"Silicon atomic weight: {atomic_weight:.3f} u")

.. autofunction:: xraylabtool.utils.get_atomic_data

   Get comprehensive atomic data for given element symbol with LRU caching.

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import get_atomic_data

      data = get_atomic_data('Si')
      print(f"Symbol: {data['symbol']}")
      print(f"Atomic number: {data['atomic_number']}")
      print(f"Atomic weight: {data['atomic_weight']}")

.. autofunction:: xraylabtool.utils.load_atomic_data

   Backward compatibility alias for get_atomic_data.

Unit Conversions
----------------

.. autofunction:: xraylabtool.utils.energy_to_wavelength

   Convert X-ray energy to wavelength.

   **Parameters:**
   - ``energy`` (float): Energy in keV
   - ``units`` (str): Desired units of wavelength ('angstrom', 'nm', 'm')

   **Returns:**
   - ``float``: Wavelength in specified units

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import energy_to_wavelength

      # Convert 8 keV to wavelength in Angstroms
      wavelength = energy_to_wavelength(8.0)  # 8 keV
      print(f"8 keV = {wavelength:.3f} Å")

.. autofunction:: xraylabtool.utils.wavelength_to_energy

   Convert X-ray wavelength to energy.

   **Parameters:**
   - ``wavelength`` (float): Wavelength value
   - ``units`` (str): Units of wavelength ('angstrom', 'nm', 'm')

   **Returns:**
   - ``float``: Energy in keV

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import wavelength_to_energy

      # Convert 1.55 Å to energy
      energy = wavelength_to_energy(1.55)  # Copper K-alpha
      print(f"1.55 Å = {energy:.1f} keV")

Crystallographic and Diffraction Functions
------------------------------------------

.. autofunction:: xraylabtool.utils.bragg_angle

   Calculate Bragg angle for given d-spacing and wavelength.

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import bragg_angle

      # Calculate Bragg angle for Si(111) reflection at 8 keV
      d_spacing = 3.14  # Angstroms for Si(111)
      wavelength = 1.55  # Angstroms (8 keV)
      angle = bragg_angle(d_spacing, wavelength)
      print(f"Bragg angle: {angle:.2f} degrees")

.. autofunction:: xraylabtool.utils.d_spacing_cubic

   Calculate d-spacing for cubic crystal system.

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import d_spacing_cubic

      # Si(111) d-spacing
      d = d_spacing_cubic(1, 1, 1, 5.43)  # Si lattice parameter
      print(f"Si(111) d-spacing: {d:.3f} Å")

.. autofunction:: xraylabtool.utils.d_spacing_tetragonal

   Calculate d-spacing for tetragonal crystal system.

.. autofunction:: xraylabtool.utils.d_spacing_orthorhombic

   Calculate d-spacing for orthorhombic crystal system.

.. autofunction:: xraylabtool.utils.q_from_angle

   Calculate momentum transfer q from scattering angle.

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import q_from_angle

      q = q_from_angle(28.4, 1.55)  # 2theta, wavelength
      print(f"Momentum transfer: {q:.3f} Å⁻¹")

.. autofunction:: xraylabtool.utils.angle_from_q

   Calculate scattering angle from momentum transfer q.

Data Processing and Analysis
----------------------------

.. autofunction:: xraylabtool.utils.smooth_data

   Apply moving average smoothing to data using optimized NumPy convolution.

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import smooth_data
      import numpy as np

      x = np.linspace(0, 10, 100)
      y = np.sin(x) + 0.1 * np.random.randn(100)  # Noisy data
      smoothed = smooth_data(x, y, window_size=5)

.. autofunction:: xraylabtool.utils.find_peaks

   Find peaks in diffraction data.

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import find_peaks
      import numpy as np

      x = np.linspace(0, 50, 1000)
      y = np.exp(-((x-20)**2)/10) + np.exp(-((x-30)**2)/5)  # Two peaks
      peaks, properties = find_peaks(x, y, prominence=0.1)
      print(f"Found {len(peaks)} peaks at x = {properties['x_values']}")

.. autofunction:: xraylabtool.utils.background_subtraction

   Perform background subtraction on diffraction data.

.. autofunction:: xraylabtool.utils.normalize_intensity

   Normalize intensity data.

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import normalize_intensity
      import numpy as np

      intensity = np.array([100, 200, 150, 300, 250])
      normalized = normalize_intensity(intensity, method="max")
      print(normalized)  # [0.33, 0.67, 0.5, 1.0, 0.83]

Utility Functions
-----------------

.. autofunction:: xraylabtool.utils.progress_bar

   Create a progress bar for iterations.

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import progress_bar

      for i in progress_bar(range(100), desc="Processing"):
          # Your processing code here
          pass

.. autofunction:: xraylabtool.utils.save_processed_data

   Save processed data to file.

   **Example:**

   .. code-block:: python

      from xraylabtool.utils import save_processed_data
      import numpy as np

      x = np.linspace(0, 10, 100)
      y = np.sin(x)
      save_processed_data(x, y, "sine_data.txt", header="# x, sin(x) data")
