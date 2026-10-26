"""Piecewise-polynomial Kramers-Kronig transform kernel.

Altered from the original: a broadcasting ``jax.numpy`` re-implementation of
``KK_PP`` and ``calc_relativistic_correction`` from kkcalc/kk.py of the
Kramers-Kronig Calculator (kkcalc, https://github.com/benajamin/kkcalc,
commit f331ca31a654af946f7d550a22cf4e9c11fa8afc), Copyright (c) 2013
Benjamin Watts, Daniel J. Lauk, zlib/libpng licence (full text in
``xraylabtool/data/kkcalc/LICENSE-kkcalc.txt``; see also ``NOTICE``). The
method is described in B. Watts, Opt. Express 22(19), 23628-23639 (2014),
doi:10.1364/OE.22.023628.

Energies are in eV throughout this module (the tabulated coefficients are in
eV); conversion from the keV public API happens in the callers.
"""

# ruff: noqa: RUF002

from __future__ import annotations

from functools import partial
import math
from typing import TYPE_CHECKING, Any, overload

import jax
import jax.numpy as jnp
import numpy as np

from xraylabtool.kk import _device
from xraylabtool.kk._device import Spectrum, ensure_x64
from xraylabtool.kk._shapes import energy_blocks
from xraylabtool.kk.spectrum import _real_float64

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray


def _f64(x: Any) -> jax.Array:
    """``x`` as a float64 JAX array; 64-bit mode is a precondition of the kernel."""
    out = jnp.asarray(x, dtype=jnp.float64)
    if out.dtype != jnp.float64:
        raise TypeError("the Kramers-Kronig kernel needs jax_enable_x64 (float64)")
    return out


@overload
def relativistic_correction(
    atomic_numbers: Sequence[int], counts: Sequence[float]
) -> float: ...
@overload
def relativistic_correction(
    atomic_numbers: jax.Array, counts: Sequence[float] | jax.Array
) -> jax.Array: ...
@overload
def relativistic_correction(
    atomic_numbers: Sequence[int], counts: jax.Array
) -> jax.Array: ...
def relativistic_correction(atomic_numbers: Any, counts: Any) -> float | jax.Array:
    """Relativistic correction Z* added to the Kramers-Kronig transform.

    Computes ``Σ nᵢ (Zᵢ − (Zᵢ/82.5)^2.37)`` over the elements of a formula
    unit, the high-energy limit of f1 used as the integration constant.

    Parameters
    ----------
    atomic_numbers : sequence of int or jax.Array
        Atomic number Z of each element.
    counts : sequence of float or jax.Array
        Number of atoms of each element per formula unit, same length. A
        zero count contributes exactly nothing, so padded element lists work.

    Returns
    -------
    float or jax.Array
        Z* in electrons per formula unit: a ``float`` for sequence input, a
        float64 scalar array (traceable under ``jit``) when either argument is
        a JAX array.

    Raises
    ------
    ValueError
        If the two sequences differ in length.

    References
    ----------
    B. Watts, Opt. Express 22(19), 23628 (2014), Eq. (9).

    Examples
    --------
    >>> from xraylabtool.kk.transform import relativistic_correction
    >>> round(relativistic_correction([14, 8], [1.0, 2.0]), 6)
    29.977129
    """
    if len(atomic_numbers) != len(counts):
        raise ValueError(
            "atomic_numbers and counts must have the same length, got "
            f"{len(atomic_numbers)} and {len(counts)}"
        )
    if isinstance(atomic_numbers, jax.Array) or isinstance(counts, jax.Array):
        z = _f64(atomic_numbers)
        return jnp.sum(_f64(counts) * (z - (z / 82.5) ** 2.37))
    correction = 0.0
    for z, n in zip(atomic_numbers, counts, strict=True):
        correction += (z - (z / 82.5) ** 2.37) * n
    return correction


def kk_piecewise_poly(
    eval_ev: Any, seg_ev: Any, coeffs: Any, rel_corr: Any
) -> jax.Array:
    """Real part f1 of the scattering factor from a piecewise Laurent f2.

    On segment ``j`` (``seg_ev[j] ≤ x ≤ seg_ev[j+1]``) the imaginary part is
    ``f2(x) = a₁x + a₀ + a₋₁/x + a₋₂/x² + a₋₃/x³``. Each segment's principal
    value integral has a closed form; this function sums them and adds Z*.
    Pure ``jax.numpy`` code with no data-dependent Python branching, so it can
    be wrapped in ``jax.jit``; use :func:`kk_f1` for padded, chunked
    evaluation and :func:`kk_real_spectrum` for validated host arrays.

    With ``P_j(E) = a₁E² + a₀E + a₋₁ + a₋₂/E + a₋₃/E²`` (``= E f2(E)`` on
    segment ``j``) and ``Q_j(E)`` the same with the signs of ``a₀`` and
    ``a₋₂`` flipped, segment ``j`` contributes, before division by ``πE``::

        Q_j ln|(x_{j+1}+E)/(x_j+E)| − P_j ln|(x_{j+1}−E)/(x_j−E)|
        − 2a₁E(x_{j+1}−x_j) + (2a₋₂/E) ln|x_{j+1}/x_j|
        + (2a₋₃/E)(1/x_j − 1/x_{j+1})

    The last term has the corrected sign; the typeset Eq. (10) of the
    reference prints it as ``+(2a₋₃/E)(x_{j+1}⁻¹ − x_j⁻¹)``, the opposite of
    its own Table 1. The arithmetic follows upstream
    ``KK_PP`` term by term so results agree with it to rounding.

    Singular evaluation energies (exact float equality with a knot):

    - Interior knot ``E = x_j`` (``0 < j < n``): the two ``ln|x_j − E|``
      terms of the adjacent segments are dropped and
      ``P_j(x_j) ln|(x_{j+1}−x_j)/(x_{j−1}−x_j)|`` is used instead, with
      ``P_j`` from the segment starting at ``x_j``. Where f2 is continuous at
      ``x_j`` this is the principal value, continuous in E. Where f2 jumps,
      the ordinary principal value diverges logarithmically; the finite
      value returned is upstream's convention, not a limit.
    - First or last knot: the boundary segment's ``ln|x − E|`` term is
      dropped with no compensation. This is upstream's convention as well;
      the integral diverges there when f2 is nonzero at the end of the data.

    Every ``log`` sits between two ``where`` calls, so neither the value nor
    the gradient with respect to ``eval_ev`` is NaN at knots.

    Parameters
    ----------
    eval_ev : array, shape (m,)
        Photon energies in eV at which f1 is evaluated; must be positive.
    seg_ev : array, shape (n + 1,)
        Strictly increasing, positive segment edges in eV.
    coeffs : array, shape (n, 5)
        Columns ``a₁, a₀, a₋₁, a₋₂, a₋₃`` per segment (energies in eV).
    rel_corr : float
        Relativistic correction Z* (see :func:`relativistic_correction`).

    Returns
    -------
    jax.Array, shape (m,)
        f1 in electrons per formula unit (float64).

    Notes
    -----
    Raises no exceptions: inputs are not validated (that would need
    data-dependent branching), so non-positive energies or unsorted edges give
    non-finite or meaningless values. :func:`kk_real_spectrum` validates.

    64-bit mode must be on before tracing (otherwise the float64 casts here
    truncate to float32); importing :mod:`xraylabtool.kk` enables it and
    :func:`kk_real_spectrum` re-enables it on every call.

    The per-segment terms can exceed the result by many orders of magnitude
    (heavy elements at low E: terms near 8e12 summing to about 3e3 for Au at
    13 eV), so the result carries an absolute rounding error of a few
    ``eps · max_j |term_j| / (πE)`` (about 1e-4 there, below 1e-7 for light
    elements). Evaluations that differ only in operation order (NumPy vs
    XLA, another chunk size) agree to that level, not to ``rtol=1e-8``.

    References
    ----------
    B. Watts, Opt. Express 22(19), 23628 (2014), Eqs. (7) and (10); Eq. (10)
    with the a₋₃ term as ``+(2a₋₃/E)(x_j⁻¹ − x_{j+1}⁻¹)``, consistent with
    the paper's Table 1 and upstream ``KK_PP``. The knot and endpoint
    conventions are upstream-only.

    Examples
    --------
    >>> import numpy as np
    >>> from xraylabtool.kk.transform import kk_piecewise_poly
    >>> f1 = kk_piecewise_poly(
    ...     np.array([1.0, 10.0]),
    ...     np.array([2.0, 5.0]),
    ...     np.array([[0.0, 1.0, 0.0, 0.0, 0.0]]),
    ...     0.0,
    ... )
    >>> print(np.round(np.asarray(f1), 6))
    [-0.661907  0.078578]
    """
    e = _f64(eval_ev)[:, None]
    seg = _f64(seg_ev)
    c = _f64(coeffs)
    x1 = seg[:-1]
    x2 = seg[1:]
    a1, a0, am1, am2, am3 = c[:, 0], c[:, 1], c[:, 2], c[:, 3], c[:, 4]

    log_ratio = jnp.log(jnp.abs(x2 / x1))
    inv_diff = x2**-1 - x1**-1
    sq_diff = x2**2 - x1**2
    width = x2 - x1
    symb_1 = (
        (a1 * e + a0) * width
        + 0.5 * a1 * sq_diff
        - (am2 / e + am3 * e**-2) * log_ratio
        + am3 / e * inv_diff
    )
    symb_2 = (
        (-a1 * e + a0) * width
        + 0.5 * a1 * sq_diff
        + (am2 / e - am3 * e**-2) * log_ratio
        - am3 / e * inv_diff
    ) + (a1 * e**2 - a0 * e + am1 - am2 * e**-1 + am3 * e**-2) * jnp.log(
        jnp.abs((x2 + e) / (x1 + e))
    )
    p_plus = a1 * e**2 + a0 * e + am1 + am2 * e**-1 + am3 * e**-2

    adjacent = (x2 == e) | (x1 == e)
    num = jnp.where(adjacent, 1.0, x2 - e)
    den = jnp.where(adjacent, 1.0, x1 - e)
    symb_3 = jnp.where(adjacent, 0.0, p_plus * jnp.log(jnp.abs(num / den)))
    symb_b = jnp.sum(symb_2 - symb_1 - symb_3, axis=1)

    # Interior knot x_k = x1[k], k >= 1: right-hand segment k, neighbours
    # x_{k-1} = x1[k-1] and x_{k+1} = x2[k].
    hit = x1[1:] == e
    hit_num = jnp.where(hit, x2[1:] - e, 1.0)
    hit_den = jnp.where(hit, x1[:-1] - e, 1.0)
    singular = jnp.sum(
        jnp.where(hit, p_plus[:, 1:] * jnp.log(jnp.abs(hit_num / hit_den)), 0.0),
        axis=1,
    )
    return (symb_b - singular) / (math.pi * e[:, 0]) + _f64(rel_corr)


@partial(jax.jit, static_argnames="block")
def _kk_f1_blocks(
    padded_energies: jax.Array,
    spectrum: Spectrum,
    rel_corr: Any,
    n_valid: Any,
    *,
    block: int,
) -> jax.Array:
    """Compiled block driver: f1 at bucket-padded energies, zero beyond ``n_valid``.

    ``padded_energies`` has length ``block × n_blocks`` (both static through
    the array shape and ``block``); coefficients of segments at or beyond
    ``spectrum.n_segments`` are masked to exactly zero, so the result does not
    depend on the capacity the spectrum was padded to.
    """
    seg = _f64(spectrum.seg_ev)
    live = jnp.arange(spectrum.coeffs.shape[0]) < spectrum.n_segments
    coeffs = jnp.where(live[:, None], _f64(spectrum.coeffs), 0.0)
    rel = _f64(rel_corr)
    blocks = _f64(padded_energies).reshape(-1, block)
    out = jax.lax.map(lambda e: kk_piecewise_poly(e, seg, coeffs, rel), blocks)
    out = out.reshape(-1)
    return jnp.where(jnp.arange(out.size) < n_valid, out, 0.0)


def kk_f1(
    eval_ev: Any,
    spectrum: Spectrum,
    rel_corr: Any,
    *,
    block: int | None = None,
) -> jax.Array:
    """Real part f1 of a padded :class:`~xraylabtool.kk.Spectrum`, on device.

    Pure and ``jit``/``vmap``/``grad``-compatible: evaluates
    :func:`kk_piecewise_poly` over blocks of energies with ``jax.lax.map``,
    masking coefficients by ``spectrum.n_segments`` so the result is
    independent of the capacity the spectrum is padded to (up to the rounding
    scale documented in :func:`kk_piecewise_poly`). Gradients flow to the real
    coefficients and to the energies; padded coefficient rows get gradient 0.

    No input validation is done (that would need data-dependent branching):
    energies must be positive and finite and the spectrum well formed, as
    built by :meth:`Spectrum.from_numpy`. Inside a user ``jit`` the function
    specialises on ``eval_ev.shape``; empty input is handled by a static
    branch. 64-bit mode must be on (see :func:`kk_piecewise_poly`).

    Parameters
    ----------
    eval_ev : jax.Array, any shape
        Photon energies in eV.
    spectrum : Spectrum
        Padded f2 table.
    rel_corr : float or jax.Array
        Relativistic correction Z* (see :func:`relativistic_correction`).
    block : int, optional
        Static upper bound on energies per block; default
        ``min(2**22 // capacity, m)``, rounded down to a power of two or
        bucket-ladder value. ``m`` is first rounded up to a power of two and
        the block count follows from it (see
        :func:`~xraylabtool.kk._shapes.energy_blocks`), so repeated calls
        share compiled programs and evaluate fewer than ``2m`` energies plus
        at most one block.

    Returns
    -------
    jax.Array of float64, shape ``eval_ev.shape``
        f1 in electrons per formula unit.

    Raises
    ------
    ValueError
        If ``block < 1``.

    Notes
    -----
    A padded spectrum is finite only while its pad knots ``x_n·(1 + k)`` do
    not overflow when squared (edges below about 1e152 eV; see
    :class:`~xraylabtool.kk.Spectrum`), far outside the physical range.

    References
    ----------
    B. Watts, Opt. Express 22(19), 23628 (2014), Eqs. (7) and (10) (see
    :func:`kk_piecewise_poly`). Padding, masking and blocking are this
    package's additions (design spec §2-§4).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from xraylabtool.kk import Spectrum, kk_f1
    >>> spec = Spectrum.from_numpy([2.0, 5.0], [[0.0, 1.0, 0.0, 0.0, 0.0]])
    >>> print(jnp.round(kk_f1(jnp.array([1.0, 10.0]), spec, 0.0), 6))
    [-0.661907  0.078578]
    """
    if block is not None and block < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    e = _f64(eval_ev)
    m = e.size
    if m == 0:
        return jnp.zeros(e.shape, dtype=jnp.float64)
    blk, n_blocks = energy_blocks(block, spectrum.coeffs.shape[0], m)
    padded = jnp.pad(e.reshape(-1), (0, blk * n_blocks - m), mode="edge")
    out: jax.Array = _kk_f1_blocks(padded, spectrum, rel_corr, jnp.int32(m), block=blk)
    return out[:m].reshape(e.shape)


def kk_real_spectrum(
    eval_ev: ArrayLike,
    seg_ev: ArrayLike,
    coeffs: ArrayLike,
    rel_corr: float,
    chunk_size: int | None = None,
) -> NDArray[np.float64]:
    """Validated, chunked evaluation of :func:`kk_piecewise_poly` on host arrays.

    Pads the segment table to a shape bucket and the energies to a whole
    number of blocks (so compiled programs are reused across calls), uploads
    once, evaluates with :func:`kk_f1`'s block driver and downloads once. Each
    energy is computed independently, so chunking, order and duplicates change
    results only at the rounding level described in
    :func:`kk_piecewise_poly`. 64-bit mode is enabled first and inputs are cast
    to float64.

    Parameters
    ----------
    eval_ev : array_like, shape (m,)
        Positive, finite photon energies in eV, in any order, repeats allowed.
    seg_ev : array_like, shape (n + 1,)
        Strictly increasing, positive, finite segment edges in eV, n ≥ 1.
    coeffs : array_like, shape (n, 5)
        Finite coefficients ``a₁, a₀, a₋₁, a₋₂, a₋₃`` per segment.
    rel_corr : float
        Relativistic correction Z* (see :func:`relativistic_correction`).
    chunk_size : int, optional
        Upper bound on energies per block (rounded down to a power of two or
        bucket-ladder value). Default: ``2**22 // capacity``, about 32 MiB per
        float64 temporary.

    Returns
    -------
    numpy.ndarray of float64, shape (m,)
        f1 in electrons per formula unit, in the order of ``eval_ev``.

    Raises
    ------
    ValueError
        If an input is complex, has the wrong shape or is not finite, an
        energy or edge is not positive, the edges are not strictly increasing, or
        ``chunk_size < 1``.

    References
    ----------
    B. Watts, Opt. Express 22(19), 23628 (2014), Eqs. (7) and (10) (see
    :func:`kk_piecewise_poly`). Validation and chunking are this package's
    additions; upstream evaluates all energies in one pass.

    Examples
    --------
    >>> import numpy as np
    >>> from xraylabtool.kk.transform import kk_real_spectrum
    >>> f1 = kk_real_spectrum(
    ...     [1.0, 10.0], [2.0, 5.0], [[0.0, 1.0, 0.0, 0.0, 0.0]], 0.0, chunk_size=1
    ... )
    >>> print(np.round(f1, 6))
    [-0.661907  0.078578]
    """
    e = _real_float64("eval_ev", eval_ev)
    seg, c = _device.validated_arrays(seg_ev, coeffs, e, rel_corr)
    rel = float(rel_corr)
    if chunk_size is not None and chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    ensure_x64()
    m = e.size
    if m == 0:
        return np.empty(0, dtype=np.float64)
    host = _device.host_spectrum(seg, c)
    block, n_blocks = energy_blocks(chunk_size, host.coeffs.shape[0], m)
    # Pad with a real energy (never 0, which would divide by zero).
    padded = np.concatenate([e, np.full(block * n_blocks - m, e[0])])
    dev_spec, dev_e, dev_rel, dev_m = _device.upload(
        (host, padded, np.float64(rel), np.int32(m))
    )
    out = _kk_f1_blocks(dev_e, dev_spec, dev_rel, dev_m, block=block)
    return np.asarray(_device.fetch(out), dtype=np.float64)[:m]
