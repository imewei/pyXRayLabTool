"""Piecewise-polynomial Kramers-Kronig transform kernel.

Altered from the original: a broadcasting NumPy/JAX re-implementation of
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

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from xraylabtool.backend.array_ops import ensure_float64, ops
from xraylabtool.kk.spectrum import _real_float64

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

    from xraylabtool.backend.array_ops import Array

# Default chunking budget: each (chunk, n_segments) float64 temporary holds at
# most this many elements (32 MiB), so the largest tensor stays far below
# 256 MB even with the dozen temporaries NumPy keeps alive at once.
_CHUNK_ELEMENTS = 2**22

_jit_cache: dict[str, Any] = {}


def relativistic_correction(
    atomic_numbers: Sequence[int], counts: Sequence[float]
) -> float:
    """Relativistic correction Z* added to the Kramers-Kronig transform.

    Computes ``Σ nᵢ (Zᵢ − (Zᵢ/82.5)^2.37)`` over the elements of a formula
    unit, the high-energy limit of f1 used as the integration constant.

    Parameters
    ----------
    atomic_numbers : sequence of int
        Atomic number Z of each element.
    counts : sequence of float
        Number of atoms of each element per formula unit, same length.

    Returns
    -------
    float
        Z* in electrons per formula unit.

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
    correction = 0.0
    for z, n in zip(atomic_numbers, counts, strict=True):
        correction += (z - (z / 82.5) ** 2.37) * n
    return correction


def kk_piecewise_poly(
    eval_ev: Array, seg_ev: Array, coeffs: Array, rel_corr: Any
) -> Array:
    """Real part f1 of the scattering factor from a piecewise Laurent f2.

    On segment ``j`` (``seg_ev[j] ≤ x ≤ seg_ev[j+1]``) the imaginary part is
    ``f2(x) = a₁x + a₀ + a₋₁/x + a₋₂/x² + a₋₃/x³``. Each segment's principal
    value integral has a closed form; this function sums them and adds Z*.
    Pure array code with no data-dependent Python branching, so it can be
    wrapped in ``jax.jit``; use :func:`kk_real_spectrum` for chunked, validated
    evaluation.

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
    array, shape (m,)
        f1 in electrons per formula unit, as an array of the active backend
        (float64).

    Notes
    -----
    Raises no exceptions: inputs are not validated (that would need
    data-dependent branching), so non-positive energies or unsorted edges give
    non-finite or meaningless values. :func:`kk_real_spectrum` validates.

    Under JAX, 64-bit mode must be on before tracing (otherwise the float64
    casts here truncate to float32); :func:`kk_real_spectrum` enables it.

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
    f64 = ops.float64
    e = ops.asarray(eval_ev, dtype=f64)[:, None]
    seg = ops.asarray(seg_ev, dtype=f64)
    c = ops.asarray(coeffs, dtype=f64)
    x1 = seg[:-1]
    x2 = seg[1:]
    a1, a0, am1, am2, am3 = c[:, 0], c[:, 1], c[:, 2], c[:, 3], c[:, 4]

    log_ratio = ops.log(ops.abs(x2 / x1))
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
    ) + (a1 * e**2 - a0 * e + am1 - am2 * e**-1 + am3 * e**-2) * ops.log(
        ops.abs((x2 + e) / (x1 + e))
    )
    p_plus = a1 * e**2 + a0 * e + am1 + am2 * e**-1 + am3 * e**-2

    adjacent = (x2 == e) | (x1 == e)
    num = ops.where(adjacent, 1.0, x2 - e)
    den = ops.where(adjacent, 1.0, x1 - e)
    symb_3 = ops.where(adjacent, 0.0, p_plus * ops.log(ops.abs(num / den)))
    symb_b = ops.sum(symb_2 - symb_1 - symb_3, axis=1)

    # Interior knot x_k = x1[k], k >= 1: right-hand segment k, neighbours
    # x_{k-1} = x1[k-1] and x_{k+1} = x2[k].
    hit = x1[1:] == e
    hit_num = ops.where(hit, x2[1:] - e, 1.0)
    hit_den = ops.where(hit, x1[:-1] - e, 1.0)
    singular = ops.sum(
        ops.where(hit, p_plus[:, 1:] * ops.log(ops.abs(hit_num / hit_den)), 0.0),
        axis=1,
    )
    return (symb_b - singular) / (math.pi * e[:, 0]) + rel_corr


def _get_chunked_kernel() -> Any:
    """Return the jitted ``lax.map`` over eval-energy blocks (JAX only)."""
    if "kk_pp" not in _jit_cache:
        import jax  # noqa: PLC0415 (optional dependency, JAX backend only)

        def chunked(blocks: Any, seg_ev: Any, coeffs: Any, rel_corr: Any) -> Any:
            return jax.lax.map(
                lambda block: kk_piecewise_poly(block, seg_ev, coeffs, rel_corr),
                blocks,
            )

        _jit_cache["kk_pp"] = jax.jit(chunked)
    return _jit_cache["kk_pp"]


def kk_real_spectrum(
    eval_ev: ArrayLike,
    seg_ev: ArrayLike,
    coeffs: ArrayLike,
    rel_corr: float,
    chunk_size: int | None = None,
) -> NDArray[np.float64]:
    """Validated, chunked evaluation of :func:`kk_piecewise_poly`.

    Splits ``eval_ev`` into blocks of ``chunk_size`` energies so the
    ``(chunk_size, n)`` temporaries stay bounded. Each energy is computed
    independently, so chunking, order and duplicates change results only at
    the rounding level described in :func:`kk_piecewise_poly` (exactly
    nothing on NumPy for order and duplicates). On the JAX backend the
    blocks run through ``jax.lax.map`` inside one jitted function (64-bit mode
    is enabled first, inputs are cast to float64); on NumPy a Python loop.

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
        Energies per block. Default: ``max(1, 2**22 // n)``, about 32 MiB per
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
    seg = _real_float64("seg_ev", seg_ev)
    c = _real_float64("coeffs", coeffs)
    rel = float(rel_corr)
    if e.ndim != 1:
        raise ValueError(f"eval_ev must be 1-D, got shape {e.shape}")
    if seg.ndim != 1 or seg.size < 2:
        raise ValueError(
            f"seg_ev must be 1-D with at least 2 edges, got shape {seg.shape}"
        )
    if c.shape != (seg.size - 1, 5):
        raise ValueError(f"coeffs must have shape ({seg.size - 1}, 5), got {c.shape}")
    for label, arr in (("eval_ev", e), ("seg_ev", seg), ("coeffs", c)):
        if not np.isfinite(arr).all():
            raise ValueError(f"{label} must be finite")
    if not math.isfinite(rel):
        raise ValueError("rel_corr must be finite")
    if (e <= 0).any() or (seg <= 0).any():
        raise ValueError("eval_ev and seg_ev must be positive")
    if (np.diff(seg) <= 0).any():
        raise ValueError("seg_ev must be strictly increasing")
    if chunk_size is None:
        chunk_size = max(1, _CHUNK_ELEMENTS // c.shape[0])
    elif chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    m = e.size
    if m == 0:
        return np.empty(0, dtype=np.float64)
    chunk = min(chunk_size, m)
    seg_b, c_b, rel_b = ensure_float64(seg), ensure_float64(c), ensure_float64(rel)
    if ops.is_jax():
        n_blocks = -(-m // chunk)
        # Pad with a real energy (never 0, which would divide by zero).
        padded = np.concatenate([e, np.full(n_blocks * chunk - m, e[0])])
        blocks = ensure_float64(padded.reshape(n_blocks, chunk))
        out = _get_chunked_kernel()(blocks, seg_b, c_b, rel_b)
        return np.asarray(out, dtype=np.float64).reshape(-1)[:m]
    parts = [
        np.asarray(kk_piecewise_poly(e[i : i + chunk], seg_b, c_b, rel_b))
        for i in range(0, m, chunk)
    ]
    return np.concatenate(parts).astype(np.float64, copy=False)
