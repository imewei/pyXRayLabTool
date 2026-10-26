Kramers-Kronig Guide
====================

The ``kk`` tools compute the real part of the atomic scattering factor,
``f1``, as the Kramers-Kronig transform of the imaginary part, ``f2``. The
input is either the tabulated ``f2`` alone or the tabulated ``f2`` with a
measured near-edge spectrum spliced in. This is how you get ``f1`` (and hence
δ) that is consistent with a measured NEXAFS spectrum near an absorption
edge, where the tabulated values are too coarse. The method is described in
:doc:`../physics/kramers_kronig` and every function in :doc:`../api/kk`.

All three interfaces (Python, CLI, GUI) call the same function,
:func:`~xraylabtool.kk.api.kk_calculate_real`.

Python API
----------

Tabulated spectrum only
~~~~~~~~~~~~~~~~~~~~~~~

Energies are in keV, as everywhere in the package:

>>> import numpy as np
>>> from xraylabtool.kk import kk_calculate_real
>>> res = kk_calculate_real("SiO2", energies_kev=[1.0, 8.0])
>>> np.round(res.f1, 4)
array([29.427 , 30.3636])
>>> np.round(res.f2, 4)
array([4.5794, 0.397 ])
>>> round(res.relativistic_correction, 4)
29.9771

Without ``energies_kev`` the result is on the spectrum's own knots, clipped to
0.03-30 keV (an INFO log line reports how many knots were dropped):

>>> grid = kk_calculate_real("SiO2")
>>> grid.energy_kev.size
1116

The result is a :class:`~xraylabtool.kk.result.KKResult` with the fields
``formula``, ``energy_kev``, ``f1``, ``f2``, ``relativistic_correction``,
``merge_points_kev``, ``n_refined_points``, ``backend`` and
``nearedge_scaled_kev_f2``. ``to_xray_result(density)`` turns it into an
ordinary :class:`~xraylabtool.calculators.xray_result.XRayResult` (δ, β,
critical angle, attenuation length, ...) computed from the KK ``f1``. It
accepts only energies in 0.03-30 keV:

>>> xr = res.to_xray_result(2.2)
>>> xr.critical_angle_degrees.shape
(2,)

With a near-edge spectrum
~~~~~~~~~~~~~~~~~~~~~~~~~

Load the measured file and say what kind of values it holds. The tabulated
``f2`` between the two merge points is replaced by the measured data, scaled
and offset so that it meets the tabulated values at both merge points. For the
polystyrene NEXAFS example distributed with upstream kkcalc (energies in eV):

.. code-block:: python

   from xraylabtool.kk import kk_calculate_real, load_nearedge

   spec = load_nearedge("PS_004_-dc.txt", energy_unit="eV")
   res = kk_calculate_real(
       "C8H8",
       nearedge=spec,
       input_type="nexafs",
       merge_points_kev=(0.280, 0.320),  # keV in the Python API
       refine_tolerance=0.05,
   )
   energy_kev, f2_scaled = res.nearedge_scaled_kev_f2  # the data on the f2 scale

The same with a spectrum built in memory (``input_type="asf"`` means the
values are already on the ``f2`` scale):

>>> from xraylabtool.kk import NearEdgeSpectrum
>>> e = np.linspace(520.0, 560.0, 81)  # eV
>>> v = 1.0 + np.tanh((e - 535.0) / 2.0) + 2.0 * np.exp(-((e - 537.0) / 1.5) ** 2)
>>> spec = NearEdgeSpectrum(energy_ev=e, values=v)
>>> res = kk_calculate_real(
...     "SiO2",
...     nearedge=spec,
...     input_type="asf",
...     merge_points_kev=(0.525, 0.555),
...     energies_kev=[0.530, 0.537, 0.545],
... )
>>> np.round(res.f1, 3)
array([16.442, 14.934, 20.393])
>>> res.merge_points_kev
(0.525, 0.555)

``input_type`` is required whenever ``nearedge`` is given (``TypeError``
otherwise). If ``merge_points_kev`` is omitted, the whole overlap of the data
and the tabulated range is used.

Adaptive refinement
~~~~~~~~~~~~~~~~~~~

``refine_tolerance`` (electrons per formula unit, absolute) inserts midpoints
wherever linear interpolation between neighbouring points misses ``f1`` or
``f2`` by more than the tolerance. It never removes points. The output is then
sorted and unique:

>>> r = kk_calculate_real("SiO2", energies_kev=[0.5, 1.0, 2.0], refine_tolerance=0.05)
>>> r.n_refined_points, r.energy_kev.size
(119, 122)

``refine_max_passes`` (default 50) and ``refine_max_points`` (default 100 000)
bound the work. Hitting either limit is logged at WARNING.

Low-level functions
~~~~~~~~~~~~~~~~~~~

The steps can also be run one at a time, in eV:
:func:`~xraylabtool.kk.spectrum.compound_spectrum` (coefficient table),
:func:`~xraylabtool.kk.transform.relativistic_correction`,
:func:`~xraylabtool.kk.transform.kk_real_spectrum` (``f1``),
:func:`~xraylabtool.kk.spectrum.evaluate_f2`,
:func:`~xraylabtool.kk.nearedge.to_asf_scale`,
:func:`~xraylabtool.kk.nearedge.merge_spectra` and
:func:`~xraylabtool.kk.refine.refine_spectrum`.

Command line
------------

``xraylabtool kk`` (also ``xlt kk``) prints ``energy_keV,f1,f2`` as CSV, or
JSON with the same keys. Every flag:

.. code-block:: bash

   # Tabulated spectrum on its own grid within 0.03-30 keV
   xraylabtool kk Si

   # --energies: output energies, always keV (same formats as calc -e)
   xraylabtool kk SiO2 --energies 1.0,8.0
   xraylabtool kk SiO2 --energies 0.1-30:200:log

   # --nearedge + --input-type (required together); file energies in eV by default
   xraylabtool kk C8H8 --nearedge PS_004_-dc.txt --input-type nexafs

   # --merge LOW,HIGH in the file's unit (--energy-unit)
   xraylabtool kk C8H8 --nearedge PS_004_-dc.txt --input-type nexafs --merge 280,320

   # --energy-unit keV: the file's first column AND --merge are keV
   xraylabtool kk C8H8 --nearedge ps_kev.txt --input-type nexafs --energy-unit keV --merge 0.28,0.32

   # --input-type beta with -d/--density (accepted; it cancels, see below)
   xraylabtool kk C8H8 --nearedge ps_beta.txt --input-type beta -d 1.05

   # --refine TOL: insert midpoints; the count goes to stderr
   xraylabtool kk C8H8 --nearedge PS_004_-dc.txt --input-type nexafs --refine 0.05

   # -o/--output and --format (default: from the extension, else csv)
   xraylabtool kk SiO2 --energies 1-10:10 -o sio2_kk.csv
   xraylabtool kk SiO2 --energies 1-10:10 --format json

Example output:

.. code-block:: text

   $ xraylabtool kk SiO2 --energies 1.0,8.0
   energy_keV,f1,f2
   1.0,29.42698660162399,4.579444483959662
   8.0,30.363551083333043,0.3970044362774753

Every usage or data error (missing ``--input-type``, unreadable file, no
overlap with the tabulated range, energies out of range) prints ``Error: ...``
to stderr and exits with code 1. :doc:`cli_reference` lists the flags in the
same format as the other commands.

Units
~~~~~

- ``--energies`` and every output are always **keV**.
- ``--energy-unit`` (default ``eV``) applies only to the near-edge file's first
  column and to ``--merge``.
- A keV file read as eV (energies near 0.28) does not overlap the
  10 eV-500 keV table. The error message states the file's parsed range and
  the required range.
- Merge points that land within 1e-12 relative of an end of the data or of the
  table, which happens after a keV↔eV conversion, are snapped to that end
  instead of being rejected.

GUI
---

The **Kramers-Kronig** tab of ``xraylabtool-gui`` sits next to "Single
Material" and "Multiple Materials".

1. Enter a **Formula** (default ``SiO2``).
2. Optionally pick a **Near-edge file** with *Browse…*, then choose its
   **Input type** (``nexafs``, ``beta``, ``asf``) and **File energy unit**
   (``eV`` or ``keV``). Both are ignored without a file.
3. **Merge points**: ``low, high`` in the file's unit. Leave empty for the
   whole overlap. Merge points without a file are an error.
4. **Refine tolerance**: ``Off`` (0) disables refinement.
5. **Start energy**, **End energy** (0.01-500 keV, default 1-30 keV) and
   **Points** (default 500) define a linear output grid. The CXRO ``f1``
   comparison curve is drawn only for energies inside 0.03-30 keV.
   When merge points are given, the merge window is also sampled with
   **Points** energies, even if it lies outside the start/end range, so a
   narrow near-edge region is never stepped over.
6. Click **Compute**. The calculation runs off the UI thread. An error is
   shown in the status bar, as a toast and in a message box.

The plot shows the measured ``f2`` (near-edge data on the ``f2`` scale), the
spliced ``f2``, the KK ``f1`` and, for comparison, the tabulated CXRO ``f1``.
The table lists energy, ``f1`` and ``f2``. **Export CSV** writes
``energy_keV,f1,f2`` of the last result.

Near-edge file format
---------------------

- Plain text with whitespace-separated columns. The **first** column is the
  photon energy and the **last** column is the value, so extra columns in
  between are ignored.
- Header lines before the first numeric row are skipped, as are comment lines
  starting with ``#``, ``%`` or ``;``. A non-numeric line after the data have
  started is an error that reports its line number. Upstream kkcalc silently
  drops such lines.
- At least two rows. Energies must be positive and strictly increasing, and
  all values finite. Each violation raises
  :class:`~xraylabtool.exceptions.NearEdgeDataError`.
- Input types: ``nexafs`` (also ``photoabsorption``, ``xanes`` in Python) for
  absorption values proportional to the cross-section; ``beta`` for the
  imaginary part of the refractive index; ``asf`` for values already on the
  ``f2`` scale. The input type fixes the energy-dependent conversion (for
  ``nexafs`` the values are multiplied by E). Constant prefactors cancel in
  the affine match at the merge points, so for ``beta`` the density and
  formula mass do not change the result.

Behaviour decisions (D1-D7)
---------------------------

D1, transform
   The fixed five-term kernel of upstream ``KK_PP`` (powers 1 to -3) is
   ported. The general-order variant is not ported, and there is no
   ``orders`` argument.
D2, data
   Upstream's tabulated ``f2`` is used unchanged. For Sc, Mo, Ru, Be, Ti, Si,
   La, Gd and W it differs from the ``.nff`` tables used by ``calc``, so
   ``kk`` and ``calc`` disagree on ``f2`` for those nine elements.
D3, energy range
   The core accepts 10 eV-500 keV (the data range) and never extrapolates.
   The default output grid is clipped to 0.03-30 keV. Only
   ``to_xray_result`` enforces 0.03-30 keV.
D4, merge units
   ``--merge`` uses the same unit as the file (``--energy-unit``). The Python
   API takes ``merge_points_kev`` in keV.
D5, knots and end points
   At a tabulated knot where ``f2`` is continuous the result is the true
   principal value. At a jump in ``f2`` (only He, Ne, Ar, Kr in the data) and
   at the first and last knot, the integral diverges. There the result is
   upstream's finite value, a convention.
D6, negative f2
   If the normalised near-edge data go negative, the values are kept and a
   WARNING reports how many there are and the minimum. They are never
   clipped. Non-finite values are rejected.
D7, refinement grid
   With ``refine_tolerance`` the output is the sorted, unique union of the
   requested energies and the added midpoints. Without it, the output keeps
   the caller's order.

Device pipeline, warm-up and the compile cache
----------------------------------------------

``xraylabtool.kk`` is JAX-only: one ``jax.numpy`` implementation in float64
on the default JAX device (CPU, or the GPU of a CUDA install). ``set_backend``
has no effect on it. :func:`~xraylabtool.kk.api.kk_calculate_real` uploads all
inputs once, assembles and merges the spectrum on the device, fetches one small
status record (the host raises the usual errors from it), evaluates f1 and f2,
runs the refinement one device pass at a time and downloads the result once.
Array shapes are rounded up to buckets, so a second call with other values and
the same buckets compiles nothing.

The first call in a process compiles. Measured on a CPU, ``SiO2`` with
``refine_tolerance=0.01``: about 1.3-1.5 s cold, 0.1-0.16 s warm (see
:doc:`../architecture/005-adr-kk-jax-only` for the table and its caveats).
Two things reduce the cold cost:

- **Persistent compile cache.** The ``xraylabtool kk`` command and the GUI
  launcher enable JAX's on-disk compile cache in ``xraylabtool/jax`` under
  the OS user-cache directory (``$XDG_CACHE_HOME`` or ``~/.cache`` on Linux,
  ``~/Library/Caches`` on macOS, ``%LOCALAPPDATA%`` on Windows). With a filled
  cache the cold refined run takes about 0.6 s. When it turns the cache on it
  also sets ``jax_persistent_cache_min_compile_time_secs`` to 0, because JAX's
  default of 1 s would skip nearly every kk program. It does nothing if you
  already set ``jax_compilation_cache_dir`` (or the threshold), or if
  ``XRAYLABTOOL_NO_JAX_CACHE=1``. The library itself never enables the cache;
  to use it from Python set ``jax_compilation_cache_dir`` yourself.
- **Warm-up.** :func:`~xraylabtool.kk.api.warm_up` compiles the programs a
  default calculation of the given formulas and a refined run touch. The GUI
  calls it on an idle worker after the window is shown; the command line never
  does, so a one-shot call does not pay for formulas it was not asked for.

>>> from xraylabtool.kk import warm_up
>>> warm_up(("Si",), refine_tolerance=None)

Gradients
~~~~~~~~~

:class:`~xraylabtool.kk.Spectrum` (a padded piecewise-polynomial f2 table, a
JAX pytree) and :func:`~xraylabtool.kk.transform.kk_f1` are pure functions
compatible with ``jax.jit``, ``jax.vmap`` and ``jax.grad``. The gradient with
respect to the energies agrees with a central difference:

>>> import jax
>>> import jax.numpy as jnp
>>> from xraylabtool.kk import Spectrum, kk_f1
>>> spec = Spectrum.from_numpy([2.0, 5.0], [[0.0, 1.0, 0.0, 0.0, 0.0]])
>>> def f1_at(e):
...     return kk_f1(e, spec, 0.0)
>>> e, h = jnp.array(10.0), 1e-4
>>> g = jax.grad(f1_at)(e)
>>> fd = (f1_at(e + h) - f1_at(e - h)) / (2 * h)
>>> bool(jnp.isfinite(g)) and abs(float(g - fd)) < 1e-9
True

and so does the gradient with respect to the coefficients (here the
sensitivity of f1(10 eV) to the constant term ``a0``, which equals f1 itself
because f1 is linear in the coefficients; padded rows get gradient 0):

>>> def f1_of_coeffs(c):
...     return kk_f1(e, Spectrum(spec.seg_ev, c, spec.n_segments), 0.0)
>>> gc = jax.grad(f1_of_coeffs)(spec.coeffs)
>>> round(float(gc[0, 1]), 6), float(jnp.abs(gc[1:]).max())
(0.078578, 0.0)

Limits
------

- No ``f2`` contribution below 10 eV or above 500 keV, and no analytic
  high-energy tail. ``f2`` is zero outside the data (see the physics page
  for why 500 keV is the end).
- Upstream's "add background" and "fix distortions" options are not
  implemented.
- Heavy elements below about 20 eV: ``f1`` is good to only about 1e-4
  absolute in float64, because the interval terms cancel. Upstream has the
  same limit.
- Refinement next to the 10 eV and 500 keV ends, where ``f1`` has a
  logarithmic singularity: a piece narrower than 2 % of its energy is left
  unrefined. The count is ``RefineOutcome.n_pole_skipped`` and is logged.
  :class:`~xraylabtool.kk.result.KKResult` does not carry the count. For Au
  the refined points equal those of an emulation of upstream's bisection.
  For Pb and U this rule stops a chain of bisections towards 500 keV that
  upstream runs to its recursion limit (Pb: 461 points added here, 505 in the
  emulation).
- Near-edge data whose values at the two merge points differ by less than
  1e-6 of their size cannot be normalised and are rejected.
- Many negative ``f2`` points after a merge (the D6 WARNING) usually mean a
  wrong ``--energy-unit``. An eV file declared ``keV`` still overlaps the
  table and is merged at 275-390 keV, outside the default output grid, so the
  output looks normal. The polystyrene example read that way gives 70 of 270
  negative points.
- Within a few eV of 10 eV (and of 500 keV), ``f1`` is dominated by the
  truncation singularity of the integral, not by the data. For Au,
  ``f1(10 eV) = -0.561`` (the D5 end-point convention), ``f1(10.0001 eV) =
  -4.645`` and ``f1(10.16 eV) = -0.70``.
- ``-d/--density`` is accepted for ``beta`` input but has no effect on the
  result.
- Without ``--energies`` (``energies_kev``), the output grid is the
  spectrum's knots inside 0.03-30 keV. If a merge leaves no knot there (for
  example, near-edge data spanning the whole 10 eV-500 keV range with only two
  points), this is an ``EnergyError`` (CLI exit code 1) asking for explicit
  energies; it is never an empty result.

Attribution
-----------

The ``kk`` module is a re-implementation of parts of the Kramers-Kronig
Calculator (kkcalc), https://github.com/benajamin/kkcalc, at commit
f331ca31a654af946f7d550a22cf4e9c11fa8afc. Copyright (c) 2013 Benjamin Watts,
Daniel J. Lauk. It is distributed under the zlib/libpng (Zlib) licence. The
files derived from kkcalc are marked "altered from the original". The
vendored ``xraylabtool/data/kkcalc/asf.npz`` holds upstream's ``ASF.json``
values unmodified. The licence text ships as
``xraylabtool/data/kkcalc/LICENSE-kkcalc.txt``, and ``NOTICE`` lists every
derived file. The package as a whole is MIT-licensed and the conda recipe
declares ``MIT AND Zlib``.

Please cite the method as: B. Watts, *Opt. Express* **22**\ (19),
23628–23639 (2014), doi:10.1364/OE.22.023628.

References
----------

- B. Watts, "Calculation of the Kramers-Kronig transform of X-ray spectra by a
  piecewise Laurent polynomial method", *Opt. Express* **22**\ (19),
  23628–23639 (2014), doi:10.1364/OE.22.023628.
- kkcalc (commit f331ca31a654af946f7d550a22cf4e9c11fa8afc), zlib/libpng
  licence, https://github.com/benajamin/kkcalc.
