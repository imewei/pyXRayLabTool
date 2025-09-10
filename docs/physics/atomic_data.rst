Atomic Scattering Factor Data
=============================

Understanding the atomic data sources and interpolation methods used in XRayLabTool.

Atomic Scattering Factors
--------------------------

Definition
~~~~~~~~~~

Atomic scattering factors describe how X-rays scatter from atoms:

.. math::

   f = f_0 + f' + if''

Where:

- **f₀**: Thomson scattering (classical, forward scattering)
- **f'**: Dispersion correction (real part)
- **f''**: Absorption (imaginary part)

For X-ray optics calculations, we use:

- **f₁ = f₀ + f'**: Total real part
- **f₂ = f''**: Imaginary part (absorption)

Physical Origin
~~~~~~~~~~~~~~~

**Thomson Scattering (f₀):**
- Classical electron scattering
- Energy-independent
- Equals atomic number Z for forward scattering

**Dispersion Correction (f'):**
- Quantum mechanical correction
- Energy-dependent, especially near absorption edges
- Can be positive or negative

**Absorption (f''):**
- Photoabsorption cross-section
- Always positive
- Shows sharp edges at absorption thresholds

Energy Dependence
-----------------

Away from Absorption Edges
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For energies well away from absorption edges:

.. math::

   f' \approx -\frac{r_e mc^2}{2\pi} \sum_j \frac{\lambda^2 f_{j0}}{(\lambda^2 - \lambda_j^2)}

   f'' \approx \frac{Z^4 \text{const}}{E^3}

Where:
- λⱼ: Absorption edge wavelengths
- fⱼ₀: Oscillator strengths
- The f'' ∝ E⁻³ scaling is approximate

Near Absorption Edges
~~~~~~~~~~~~~~~~~~~~~

Near absorption edges, both f' and f'' show complex structure:

1. **Pre-edge region**: Smooth interpolation
2. **Edge jump**: Sharp discontinuity in f''
3. **Post-edge oscillations**: XANES and EXAFS structure

Data Sources
------------

Henke Tables
~~~~~~~~~~~~

**Coverage:**
- Elements: H (Z=1) to U (Z=92)
- Energy range: 10 eV to 30 keV
- Energy spacing: Variable, denser near edges

**Method:**
- Combines experimental photoabsorption data
- Theoretical calculations for f'
- Kramers-Kronig transformation ensures consistency

**File Format:**
Standard .nff format with columns:
- Energy (eV)
- f₁ (real part)
- f₂ (imaginary part)

CXRO Database
~~~~~~~~~~~~~

**Extended Henke Tables:**
- Updated experimental data
- Extended energy ranges for some elements
- Web interface and downloadable files
- Source: http://henke.lbl.gov/optical_constants/

**Advantages:**
- Regular updates with new measurements
- Quality control and validation
- Widely accepted standard

NIST XCOM
~~~~~~~~~

**Photoabsorption Data:**
- Primary source for absorption coefficients
- Energy range: 1 keV to 100 GeV
- Includes pair production and Compton scattering
- Used to validate and extend other databases

Interpolation Methods
---------------------

Linear Interpolation
~~~~~~~~~~~~~~~~~~~~

XRayLabTool uses linear interpolation between tabulated values:

.. math::

   f(E) = f_1 + \frac{E - E_1}{E_2 - E_1}(f_2 - f_1)

This works well because:
- Data points are closely spaced
- Smooth variation between points
- Computationally efficient

Logarithmic Interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~

For some quantities, logarithmic interpolation may be more accurate:

.. math::

   \ln f(E) = \ln f_1 + \frac{\ln E - \ln E_1}{\ln E_2 - \ln E_1}(\ln f_2 - \ln f_1)

Used when:
- Data spans many orders of magnitude
- Exponential-like behavior expected
- Higher accuracy needed

Spline Interpolation
~~~~~~~~~~~~~~~~~~~~

For critical applications, spline interpolation provides:
- Smooth derivatives
- Better behavior near edges
- Higher computational cost

Edge Handling
-------------

Absorption Edge Structure
~~~~~~~~~~~~~~~~~~~~~~~~~

Absorption edges create discontinuities in f'':

**K-edge (1s electron):**
- Largest jump in f''
- Corresponding feature in f'
- Most prominent for light elements

**L-edges (2s, 2p electrons):**
- Multiple edges (L₁, L₂, L₃)
- Fine structure from chemical environment
- Important for medium-Z elements

**M-edges and higher:**
- Many closely spaced edges
- Complex fine structure
- Important for heavy elements

Pre-edge Features
~~~~~~~~~~~~~~~~~

Near absorption edges:
- **White lines**: Sharp peaks just above edge
- **XANES**: X-ray Absorption Near Edge Structure
- **Pre-edge peaks**: Forbidden transitions

These features contain chemical information but complicate optical calculations.

Kramers-Kronig Relations
~~~~~~~~~~~~~~~~~~~~~~~~

The real and imaginary parts are related by:

.. math::

   f'(E) = \frac{2}{\pi} P \int_0^{\infty} \frac{\omega f''(\omega)}{\omega^2 - E^2} d\omega

Where P denotes the principal value. This ensures physical consistency.

Quality and Accuracy
--------------------

Experimental Uncertainties
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Photoabsorption Measurements:**
- Systematic errors: 2-5% typical
- Statistical errors: 1-2% for good measurements
- Sample contamination affects results
- Temperature and pressure effects

**Transmission Measurements:**
- Sample thickness uncertainty
- Multiple scattering corrections
- Surface oxidation effects
- Grain size and texture effects

Theoretical Limitations
~~~~~~~~~~~~~~~~~~~~~~~

**Isolated Atom Approximation:**
- Ignores chemical bonding effects
- Assumes spherical atoms
- No crystal field effects
- Limited accuracy for light elements

**Relativistic Effects:**
- Important for inner shells of heavy elements
- Affects edge positions and intensities
- Modern calculations include these

Validation Methods
~~~~~~~~~~~~~~~~~~

**Cross-checks between databases:**
- NIST XCOM vs Henke tables
- Independent measurements
- Sum rule tests

**Experimental validation:**
- Reflectometry measurements
- Transmission measurements
- Interferometry techniques

Data Processing in XRayLabTool
------------------------------

Caching Strategy
~~~~~~~~~~~~~~~~

XRayLabTool uses a multi-level caching system:

1. **Preloaded cache**: 92 common elements loaded at startup
2. **LRU cache**: Recently used interpolations cached
3. **Disk cache**: Computed values saved for reuse
4. **Memory management**: Automatic cleanup of old entries

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

**Vectorized operations:**
- NumPy arrays for energy ranges
- Batch interpolation for efficiency
- SIMD operations where available

**Smart interpolation:**
- Adaptive mesh refinement near edges
- Coarse grids away from features
- Error estimation and mesh adaptation

Error Estimation
~~~~~~~~~~~~~~~~

XRayLabTool provides error estimates based on:

1. **Interpolation error**: From data spacing
2. **Experimental uncertainty**: From literature values
3. **Model limitations**: Isolated atom approximation
4. **Numerical precision**: Machine epsilon effects

Usage Guidelines
----------------

Energy Range Selection
~~~~~~~~~~~~~~~~~~~~~~

**Recommended ranges:**
- 100 eV - 30 keV: Henke data most reliable
- 30-100 keV: Extrapolation, larger uncertainties
- Below 100 eV: Strong chemical bonding effects

**Avoiding problematic regions:**
- Very close to absorption edges (±10 eV)
- Regions with sparse data coverage
- Energies requiring large extrapolations

Material Considerations
~~~~~~~~~~~~~~~~~~~~~~~

**Light elements (Z < 10):**
- Large relative bonding effects
- Limited experimental data
- Consider molecular form factors

**Heavy elements (Z > 80):**
- Complex edge structure
- Relativistic effects important
- Multiple absorption edges

**Compounds vs Elements:**
- Additivity assumption generally good
- Chemical shifts usually small
- Exceptions: strongly bonded materials

Future Developments
-------------------

Database Updates
~~~~~~~~~~~~~~~~

- New experimental measurements
- Improved theoretical calculations
- Extended energy ranges
- Better uncertainty estimates

Computational Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Machine learning interpolation
- Quantum mechanical calculations
- Many-body effects
- Temperature-dependent data

Integration Features
~~~~~~~~~~~~~~~~~~~~

- Real-time database updates
- Quality metrics and validation
- User-contributed data
- Community feedback mechanisms

References
----------

**Primary Sources:**
- Henke, B.L., et al. "X-ray interactions: photoabsorption, scattering, transmission, and reflection at E=50-30000 eV, Z=1-92", Atomic Data and Nuclear Data Tables 54, 181-342 (1993)
- NIST XCOM: Photon Cross Sections Database
- CXRO X-ray interactions database

**Theoretical Background:**
- Bethe, H.A. & Salpeter, E.E. "Quantum Mechanics of One- and Two-Electron Atoms"
- Brown, G.S. et al. "X-ray absorption spectroscopy and its applications"
