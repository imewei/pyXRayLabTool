Kramers-Kronig (kk)
===================

The ``kk`` sub-package computes the real scattering factor f1 from the
tabulated imaginary part f2, optionally spliced with measured near-edge data,
by the piecewise Laurent polynomial method of Watts (2014). Background:
:doc:`../physics/kramers_kronig`; usage: :doc:`../guides/kk_guide`.

Public names are exported lazily from ``xraylabtool.kk`` (``import
xraylabtool`` does not import it). Public functions take energies in keV,
except the low-level functions, which work in eV like the coefficient tables.

Which function implements which equation
----------------------------------------

Equation numbers refer to B. Watts, *Opt. Express* **22**\ (19), 23628 (2014).

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Name (``xraylabtool.kk``)
     - Paper
     - Notes
   * - ``kk_piecewise_poly``
     - Eqs. (7), (10)
     - Closed-form f1, a₋₃ term with the sign of Table 1 (erratum); knot and
       end-point conventions from upstream (D5). Energies in eV.
   * - ``kk_real_spectrum``
     - Eq. (10)
     - Validated, chunked wrapper of ``kk_piecewise_poly`` (eV).
   * - ``kk_f1``
     - Eq. (10)
     - Device-level f1 of a padded ``Spectrum`` (eV); ``jit``/``vmap``/``grad``
       compatible.
   * - ``Spectrum``
     - Eq. (8)
     - Padded piecewise-polynomial f2 table, a JAX pytree (eV).
   * - ``relativistic_correction``
     - Eq. (9)
     - Z* of a formula unit.
   * - ``evaluate_f2``
     - Eq. (8)
     - f2 from the coefficient table, right-continuous at knots (eV).
   * - ``compound_spectrum``
     - Eq. (8), step 1
     - Stoichiometric sum of elemental tables on the union of their knots.
   * - ``load_asf``, ``AsfDatabase``
     - Eq. (8) data
     - Vendored kkcalc ``ASF.json`` tables, 10 eV-500 keV.
   * - ``NearEdgeSpectrum``, ``load_nearedge``
     - step 2
     - Two-column near-edge file reader and validated container.
   * - ``to_asf_scale``
     - step 2
     - Converts NEXAFS/photoabsorption, beta or ASF data to the f2 scale.
   * - ``merge_spectra``
     - step 2
     - Affine endpoint matching of the near-edge data into the table.
   * - ``refine_spectrum``, ``RefineOutcome``
     - not in the paper
     - Upstream ``improve_accuracy``: adaptive midpoint insertion.
   * - ``kk_calculate_real``, ``KKResult``
     - steps 1-6
     - High-level entry point (keV) and its result.
   * - ``warm_up``
     - not in the paper
     - Precompiles the default device programs for given formulas (the GUI
       warms its own Compute call the same way; see :doc:`../guides/kk_guide`).
   * - ``calculate_properties_from_factors``
     - not in the paper
     - ``xraylabtool.calculators.core``: δ, β and derived quantities from
       f1/f2 totals; used by ``KKResult.to_xray_result``.

Package
-------

.. automodule:: xraylabtool.kk
   :no-members:

High-level API
--------------

.. automodule:: xraylabtool.kk.api
   :members:
   :undoc-members:
   :show-inheritance:

Result
------

.. automodule:: xraylabtool.kk.result
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: formula, energy_kev, f1, f2, relativistic_correction, merge_points_kev,
      n_refined_points, backend, nearedge_scaled_kev_f2

Transform kernel
----------------

.. automodule:: xraylabtool.kk.transform
   :members:
   :undoc-members:
   :show-inheritance:

Padded spectrum
---------------

.. autoclass:: xraylabtool.kk.Spectrum
   :members:

Compound spectrum
-----------------

.. automodule:: xraylabtool.kk.spectrum
   :members:
   :undoc-members:
   :show-inheritance:

Near-edge data
--------------

.. automodule:: xraylabtool.kk.nearedge
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: n_segments, nonfinite, n_negative, f2_min, n_inside, spectrum,
      scaled_f2, used_points_ev, status

Refinement
----------

.. automodule:: xraylabtool.kk.refine
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: energy_ev, f1, f2, n_added, exhausted, n_pole_skipped

Database
--------

.. automodule:: xraylabtool.kk.database
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: symbols, mass, knots, coeffs, n_knots

Optical properties from f1 and f2
---------------------------------

.. autofunction:: xraylabtool.calculators.core.calculate_properties_from_factors
   :no-index:

References
----------

- B. Watts, "Calculation of the Kramers-Kronig transform of X-ray spectra by a
  piecewise Laurent polynomial method", *Opt. Express* **22**\ (19),
  23628–23639 (2014), doi:10.1364/OE.22.023628.
- kkcalc (commit f331ca31a654af946f7d550a22cf4e9c11fa8afc), zlib/libpng
  licence, https://github.com/benajamin/kkcalc; see
  :doc:`../guides/kk_guide` for the attribution.
