X-ray Optics Fundamentals
=========================

This section provides the scientific background for X-ray optical property calculations performed by XRayLabTool.

Complex Refractive Index
-------------------------

For X-rays, materials have a complex refractive index:

.. math::

   n = 1 - \delta - i\beta

Where:

- **n**: Complex refractive index
- **δ (delta)**: Real part of refractive index decrement
- **β (beta)**: Imaginary part related to absorption
- **i**: Imaginary unit

Physical Interpretation
~~~~~~~~~~~~~~~~~~~~~~~

**Real Part (δ):**
- Controls phase velocity: :math:`v_{phase} = c/n_{real} = c/(1-\delta)`
- For X-rays, δ > 0, so phase velocity > c (but energy velocity < c)
- Determines critical angle for total external reflection
- Typically :math:`10^{-6}` to :math:`10^{-4}` for hard X-rays

**Imaginary Part (β):**
- Controls absorption and attenuation
- Related to absorption coefficient: :math:`\mu = 4\pi\beta/\lambda`
- Determines penetration depth
- Generally β ≪ δ except near absorption edges

Calculation from Atomic Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a compound with multiple elements:

.. math::

   \delta = \frac{r_e \lambda^2}{2\pi} \sum_i n_i f_1^i

   \beta = \frac{r_e \lambda^2}{2\pi} \sum_i n_i f_2^i

Where:

- :math:`r_e = 2.818 \times 10^{-15}` m (classical electron radius)
- :math:`\lambda`: X-ray wavelength
- :math:`n_i`: Number density of element i
- :math:`f_1^i, f_2^i`: Atomic scattering factors for element i

Critical Angle for Total External Reflection
---------------------------------------------

At grazing incidence, total external reflection occurs when:

.. math::

   \sin(\theta_c) = \sqrt{2\delta}

For small angles (:math:`\theta_c` in radians):

.. math::

   \theta_c \approx \sqrt{2\delta}

Converting to practical units:

.. math::

   \theta_c \text{ (degrees)} = \sqrt{2\delta} \times \frac{180}{\pi}

   \theta_c \text{ (mrad)} = 1000 \times \sqrt{2\delta}

Physical Significance
~~~~~~~~~~~~~~~~~~~~~

- **Total reflection**: For angles θ < θc, X-rays are totally reflected
- **Mirror design**: Critical angle determines useful angular range
- **Energy dependence**: θc ∝ λ², so higher energies have smaller critical angles
- **Material choice**: Higher electron density → larger critical angle

Attenuation and Absorption
---------------------------

Linear Absorption Coefficient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The linear absorption coefficient relates to the imaginary part of the refractive index:

.. math::

   \mu = \frac{4\pi\beta}{\lambda}

With units of cm⁻¹ (or m⁻¹).

Mass Absorption Coefficient
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Often more convenient for comparing materials:

.. math::

   \mu/\rho = \frac{\mu}{\rho}

Where ρ is the material density, giving units of cm²/g.

Attenuation Length
~~~~~~~~~~~~~~~~~~

The 1/e attenuation length (distance for intensity to drop by factor e):

.. math::

   l_{att} = \frac{1}{\mu} = \frac{\lambda}{4\pi\beta}

Beer-Lambert Law
~~~~~~~~~~~~~~~~

Intensity decreases exponentially with thickness:

.. math::

   I(t) = I_0 e^{-\mu t}

Where:
- I₀: Initial intensity
- t: Material thickness
- μ: Linear absorption coefficient

Transmission and Reflection
---------------------------

Fresnel Equations for X-rays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a smooth interface at grazing angle θ:

**Reflectivity:**

.. math::

   R = \left|\frac{n\cos\theta - \sqrt{1 - n^2\sin^2\theta}}{n\cos\theta + \sqrt{1 - n^2\sin^2\theta}}\right|^2

**Transmission:**

.. math::

   T = 1 - R \quad \text{(for non-absorbing case)}

For absorbing materials, both reflection and transmission are reduced, with energy lost to absorption.

Applications in Synchrotron Optics
-----------------------------------

Mirror Design
~~~~~~~~~~~~~

**Substrate Selection:**
- Higher δ → larger critical angle → better reflectivity at higher angles
- Lower β → less absorption → higher throughput
- Thermal properties important for high-power applications

**Coating Optimization:**
- Multilayer coatings can enhance reflectivity
- Periodic structures create artificial Bragg reflections
- Material combinations: W/B₄C, Ni/C, Mo/Si

Beamline Components
~~~~~~~~~~~~~~~~~~~

**Windows and Filters:**
- Balance between transmission and contamination protection
- Optimize thickness: thin enough for transmission, thick enough for strength
- Common materials: Be, diamond, SiN membranes

**Monochromator Crystals:**
- Silicon most common due to perfect crystal structure
- Darwin width determines energy resolution
- Thermal management crucial for stability

Energy Dependence
-----------------

Absorption Edges
~~~~~~~~~~~~~~~~

Near absorption edges, scattering factors show sharp changes:

- **Pre-edge**: Smooth energy dependence
- **Edge jump**: Sharp increase in f₂ (absorption)
- **Post-edge**: EXAFS oscillations in both f₁ and f₂

This creates opportunities and challenges:
- Enhanced contrast near edges
- Monochromator design must account for edge effects
- Material choice depends on X-ray energy range

Scaling Laws
~~~~~~~~~~~~

For energies well away from edges:

.. math::

   f_2 \propto Z^4/E^3

   \delta \propto \lambda^2 \propto E^{-2}

   \beta \propto \lambda^2 \propto E^{-2}

Therefore:
- Critical angle decreases as E⁻¹
- Attenuation length increases as E³
- Higher energies are more penetrating

Practical Considerations
------------------------

Surface Roughness
~~~~~~~~~~~~~~~~~

Real surfaces have roughness that reduces reflectivity:

.. math::

   R_{rough} = R_{smooth} \times e^{-(4\pi\sigma\sin\theta/\lambda)^2}

Where σ is the RMS surface roughness.

Contamination
~~~~~~~~~~~~~

Surface contamination (carbon, oxides) affects optical properties:
- Reduces reflectivity
- Changes effective critical angle
- Time-dependent degradation in some environments

Temperature Effects
~~~~~~~~~~~~~~~~~~~

Thermal expansion changes:
- Lattice spacing (important for crystals)
- Surface figure (thermal distortion)
- Bulk density (usually small effect)

Measurement and Characterization
---------------------------------

Experimental Techniques
~~~~~~~~~~~~~~~~~~~~~~~

**Reflectometry:**
- Measure reflectivity vs angle at fixed energy
- Determine δ and β from curve fitting
- Requires high-quality optical surfaces

**Transmission Measurements:**
- Measure attenuation through known thickness
- Direct determination of absorption coefficient
- Easier for high-Z materials

**Energy Scans:**
- Vary energy at fixed geometry
- Map out absorption edge structure
- Useful for identifying elemental composition

Data Sources
~~~~~~~~~~~~

XRayLabTool uses atomic scattering factor data from:

1. **Henke Tables**: Widely used standard (10 eV - 30 keV)
2. **CXRO Database**: Extended energy range with updates
3. **NIST XCOM**: Photoabsorption cross-sections
4. **Theoretical calculations**: For very light elements or high energies

The data combines experimental measurements with theoretical calculations, with interpolation between tabulated values for smooth energy dependence.

Further Reading
---------------

**Textbooks:**
- Als-Nielsen & McMorrow: "Elements of Modern X-ray Physics"
- Attwood: "Soft X-rays and Extreme Ultraviolet Radiation"
- Willmott: "An Introduction to Synchrotron Radiation"

**Online Resources:**
- CXRO X-ray database: http://henke.lbl.gov/optical_constants/
- NIST XCOM database: https://physics.nist.gov/xcom
- ILL X-ray absorption database: https://www.ill.eu/xop
