Calculation Methods
===================

Detailed explanation of the numerical methods and algorithms used in XRayLabTool.

Compound Analysis
-----------------

Chemical Formula Parsing
~~~~~~~~~~~~~~~~~~~~~~~~~

XRayLabTool uses a robust parser for chemical formulas:

**Grammar Rules:**
1. Element symbols: Capital letter + optional lowercase letter
2. Stoichiometric coefficients: Integer numbers following elements
3. Parentheses: For grouping with multipliers
4. Hydration: Dot notation (·) for water molecules

**Parser Algorithm:**

.. code-block:: text

   FORMULA := TERM (TERM)*
   TERM := ELEMENT COUNT? | '(' FORMULA ')' COUNT?
   ELEMENT := [A-Z][a-z]?
   COUNT := [0-9]+

**Examples:**
- ``SiO2`` → {Si: 1, O: 2}
- ``Ca5(PO4)3F`` → {Ca: 5, P: 3, O: 12, F: 1}
- ``CuSO4·5H2O`` → {Cu: 1, S: 1, O: 9, H: 10}

Error Handling
~~~~~~~~~~~~~~

The parser handles common errors:

**Unknown Elements:**

.. code-block:: python

   try:
       composition = parse_formula("XYZ")
   except FormulaError as e:
       print(f"Error: {e}")
       # Error: Unknown element 'XYZ'
       # Suggestion: Check element symbols (case-sensitive)

**Syntax Errors:**

.. code-block:: python

   try:
       composition = parse_formula("Si-O2")
   except FormulaError as e:
       print(f"Error: {e}")
       # Error: Invalid character '-' in formula
       # Suggestion: Use format like SiO2 or Al2O3

Number Density Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a compound with formula units per unit volume:

.. math::

   n_{total} = \frac{\rho N_A}{M}

Where:
- ρ: Material density (g/cm³)
- N_A: Avogadro's number
- M: Molecular weight (g/mol)

Element-specific number densities:

.. math::

   n_i = n_{total} \times s_i

Where s_i is the stoichiometric coefficient of element i.

Atomic Scattering Factor Interpolation
---------------------------------------

Data Structure
~~~~~~~~~~~~~~

Atomic data is stored as:

.. code-block:: python

   @dataclass
   class AtomicData:
       element: str
       energies: np.ndarray      # Energy grid (eV)
       f1_values: np.ndarray     # Real scattering factors
       f2_values: np.ndarray     # Imaginary scattering factors

Linear Interpolation
~~~~~~~~~~~~~~~~~~~~

For energy E between tabulated points E_i and E_{i+1}:

.. math::

   f_1(E) = f_{1,i} + \frac{E - E_i}{E_{i+1} - E_i}(f_{1,i+1} - f_{1,i})

   f_2(E) = f_{2,i} + \frac{E - E_i}{E_{i+1} - E_i}(f_{2,i+1} - f_{2,i})

Implementation:

.. code-block:: python

   def interpolate_scattering_factors(element, energy):
       data = load_atomic_data(element)
       f1 = np.interp(energy, data.energies, data.f1_values)
       f2 = np.interp(energy, data.energies, data.f2_values)
       return f1, f2

Edge Handling
~~~~~~~~~~~~~

Special care near absorption edges:

**Pre-edge extrapolation:**

.. code-block:: python

   if energy < data.energies[0]:
       # Linear extrapolation from first two points
       slope = (data.f2_values[1] - data.f2_values[0]) / \
               (data.energies[1] - data.energies[0])
       f2 = data.f2_values[0] + slope * (energy - data.energies[0])


**Post-edge extrapolation:**

.. code-block:: python

   if energy > data.energies[-1]:
       # Power law extrapolation: f2 ∝ E^(-3)
       ratio = (energy / data.energies[-1]) ** (-3)
       f2 = data.f2_values[-1] * ratio

Complex Refractive Index Calculation
-------------------------------------

Individual Element Contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each element i in the compound:

.. math::

   \delta_i = \frac{r_e \lambda^2}{2\pi} n_i f_{1,i}

   \beta_i = \frac{r_e \lambda^2}{2\pi} n_i f_{2,i}

Where:
- r_e = 2.818 × 10⁻¹⁵ m (classical electron radius)
- λ: X-ray wavelength (m)
- n_i: Number density of element i (m⁻³)

Total Compound Properties
~~~~~~~~~~~~~~~~~~~~~~~~~

Sum over all elements:

.. math::

   \delta_{total} = \sum_i \delta_i

   \beta_{total} = \sum_i \beta_i

Implementation:

.. code-block:: python

   def calculate_compound_properties(composition, density, wavelength):
       delta_total = 0.0
       beta_total = 0.0

       molecular_weight = sum(ATOMIC_WEIGHTS[elem] * count
                             for elem, count in composition.items())
       number_density = (density * AVOGADRO) / molecular_weight  # molecules/cm³

       for element, count in composition.items():
           f1, f2 = interpolate_scattering_factors(element, energy)
           element_density = number_density * count * 1e6  # Convert to m⁻³

           delta_i = (CLASSICAL_ELECTRON_RADIUS * wavelength**2 *
                     element_density * f1) / (2 * np.pi)
           beta_i = (CLASSICAL_ELECTRON_RADIUS * wavelength**2 *
                    element_density * f2) / (2 * np.pi)

           delta_total += delta_i
           beta_total += beta_i

       return delta_total, beta_total

Derived Quantity Calculations
------------------------------

Critical Angle
~~~~~~~~~~~~~~

From the refractive index decrement:

.. math::

   \theta_c = \sqrt{2\delta}

With unit conversions:

.. code-block:: python

   def calculate_critical_angle(delta):
       theta_rad = np.sqrt(2 * delta)
       theta_deg = theta_rad * 180 / np.pi
       theta_mrad = theta_rad * 1000
       return theta_rad, theta_deg, theta_mrad

Attenuation Coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~

**Linear absorption coefficient:**

.. math::

   \mu = \frac{4\pi\beta}{\lambda}

**Mass absorption coefficient:**

.. math::

   \mu/\rho = \frac{\mu}{\rho}

**Attenuation length:**

.. math::

   l_{att} = \frac{1}{\mu}

Implementation:

.. code-block:: python

   def calculate_attenuation(beta, wavelength, density):
       mu_linear = 4 * np.pi * beta / wavelength  # m⁻¹
       mu_linear_cm = mu_linear / 100  # cm⁻¹
       mu_mass = mu_linear_cm / density  # cm²/g
       attenuation_length = 1 / mu_linear_cm  # cm
       return mu_linear_cm, mu_mass, attenuation_length

Numerical Considerations
------------------------

Precision and Accuracy
~~~~~~~~~~~~~~~~~~~~~~~

**Floating Point Precision:**
- Use 64-bit floats for intermediate calculations
- Guard against underflow for small β values
- Check for overflow in exponential calculations

**Significant Figures:**
- Atomic data typically 3-4 significant figures
- Final results should reflect input precision
- Avoid false precision in output

**Error Propagation:**

.. code-block:: python

   def propagate_uncertainty(f1, f2, df1, df2):
       # δ and β uncertainties from f1, f2 uncertainties
       ddelta = df1 * (r_e * wavelength**2 * number_density) / (2 * np.pi)
       dbeta = df2 * (r_e * wavelength**2 * number_density) / (2 * np.pi)

       # Critical angle uncertainty
       dtheta = ddelta / np.sqrt(2 * delta)
       return ddelta, dbeta, dtheta

Vectorization
~~~~~~~~~~~~~

For efficiency with energy arrays:

.. code-block:: python

   def vectorized_calculation(energies, formula, density):
       """Calculate properties for array of energies."""
       energies = np.asarray(energies)
       results = []

       # Vectorize over energies for each element
       for element, count in composition.items():
           f1_array, f2_array = interpolate_scattering_factors(element, energies)
           # Process entire arrays at once

       return np.array(results)

Boundary Conditions
~~~~~~~~~~~~~~~~~~~

**Energy limits:**

.. code-block:: python

   def validate_energy(energy):
       if np.any(energy <= 0):
           raise EnergyError("Energy must be positive")
       if np.any(energy < MIN_ENERGY):
           warnings.warn(f"Energy below {MIN_ENERGY} eV may be unreliable")
       if np.any(energy > MAX_ENERGY):
           warnings.warn(f"Energy above {MAX_ENERGY} eV requires extrapolation")

**Density validation:**

.. code-block:: python

   def validate_density(density):
       if density <= 0:
           raise ValidationError("Density must be positive")
       if density > MAX_REASONABLE_DENSITY:
           warnings.warn("Very high density - check units (g/cm³)")

Performance Optimizations
--------------------------

Caching Strategies
~~~~~~~~~~~~~~~~~~

**LRU Cache for Interpolation:**

.. code-block:: python

   from functools import lru_cache

   @lru_cache(maxsize=10000)
   def cached_interpolation(element, energy_tuple):
       # Convert tuple back to array for interpolation
       energies = np.array(energy_tuple)
       return interpolate_scattering_factors(element, energies)

**Precomputed Grids:**

.. code-block:: python

   class PrecomputedGrid:
       def __init__(self, energy_min, energy_max, n_points):
           self.energy_grid = np.logspace(
               np.log10(energy_min), np.log10(energy_max), n_points
           )
           self.f1_grid = {}
           self.f2_grid = {}
           self._precompute_common_elements()


Memory Management
~~~~~~~~~~~~~~~~~

**Chunked Processing:**

.. code-block:: python

   def process_large_batch(materials, energies, chunk_size=1000):
       """Process large datasets in chunks to manage memory."""
       n_materials = len(materials)
       results = []

       for i in range(0, n_materials, chunk_size):
           chunk = materials[i:i+chunk_size]
           chunk_results = calculate_batch(chunk, energies)
           results.extend(chunk_results)

           # Optional: garbage collection
           if i % (chunk_size * 10) == 0:
               gc.collect()

       return results

**Sparse Storage:**

.. code-block:: python

   def store_sparse_results(results, threshold=1e-12):
       """Store only non-negligible values to save memory."""
       sparse_results = []
       for result in results:
           if result.beta > threshold:
               sparse_results.append(result)
       return sparse_results


Algorithm Complexity
~~~~~~~~~~~~~~~~~~~~~

**Time Complexity:**
- Single calculation: O(N) where N is number of elements
- Batch processing: O(M×N×E) where M=materials, E=energies
- Interpolation: O(log K) where K is data points per element

**Space Complexity:**
- Atomic data storage: O(K×Z) where Z=number of elements
- Result storage: O(M×E) for batch calculations
- Cache storage: O(C) where C is cache size

Validation and Testing
----------------------

Unit Tests
~~~~~~~~~~

**Atomic Data Consistency:**

.. code-block:: python

   def test_kramers_kronig_consistency():
       """Test that f' and f'' satisfy Kramers-Kronig relations."""
       # Implementation of discrete KK transform test
       pass

   def test_sum_rules():
       """Test Thomas-Reiche-Kuhn sum rule."""
       # ∫ f''(E) dE should equal number of electrons
       pass

**Physical Limits:**

.. code-block:: python

   def test_physical_bounds():
       """Test that results are physically reasonable."""
       result = calculate_properties("Si", 2.33, 8000)
       assert 0 < result.delta < 1e-3  # Reasonable range for δ
       assert 0 < result.beta < result.delta  # Usually β << δ
       assert 0 < result.critical_angle_degrees < 1  # Typical range

Integration Tests
~~~~~~~~~~~~~~~~~

**Cross-validation with literature:**

.. code-block:: python

   def test_literature_values():
       """Compare with published reference values."""
       # Silicon at 8 keV
       result = calculate_properties("Si", 2.33, 8000)
       assert abs(result.critical_angle_degrees - 0.158) < 0.001

**Consistency across energy ranges:**

.. code-block:: python

   def test_energy_continuity():
       """Test smooth behavior across energy ranges."""
       energies = np.linspace(7900, 8100, 201)
       results = calculate_properties_array("Si", 2.33, energies)
       # Check for smooth derivatives, no discontinuities

Error Handling
--------------

Graceful Degradation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def robust_calculation(formula, density, energy):
       try:
           return calculate_properties(formula, density, energy)
       except AtomicDataError:
           # Fall back to approximate methods
           return approximate_calculation(formula, density, energy)
       except Exception as e:
           logger.error(f"Calculation failed: {e}")
           return None

User Feedback
~~~~~~~~~~~~~

.. code-block:: python

   def calculate_with_warnings(formula, density, energy):
       warnings = []

       if energy < 100:
           warnings.append("Low energy: results may be unreliable")
       if density > 20:
           warnings.append("High density: check units")

       result = calculate_properties(formula, density, energy)
       result.warnings = warnings
       return result
