"""Compound imaginary scattering-factor spectrum on a piecewise-polynomial grid.

Altered from the original: a vectorised re-implementation of
``calculate_asf`` and ``coeffs_to_ASF`` from kkcalc/data.py of the
Kramers-Kronig Calculator (kkcalc, https://github.com/benajamin/kkcalc,
commit f331ca31a654af946f7d550a22cf4e9c11fa8afc), Copyright (c) 2013
Benjamin Watts, Daniel J. Lauk, zlib/libpng licence (full text in
``xraylabtool/data/kkcalc/LICENSE-kkcalc.txt``; see also ``NOTICE``). The
method is described in B. Watts, Opt. Express 22(19), 23628-23639 (2014),
doi:10.1364/OE.22.023628.

Energies are in eV throughout this module (the tabulated coefficients are in
eV). The assembly (:func:`compound_spectrum_device`) and the evaluation
(:func:`evaluate_f2_device`) are pure ``jax.numpy`` and stay on the device;
:func:`compound_spectrum` and :func:`evaluate_f2` are the validated NumPy
wrappers around them. The Kramers-Kronig kernel lives in
:mod:`xraylabtool.kk.transform`.
"""

from __future__ import annotations

from functools import partial
import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from xraylabtool.exceptions import FormulaError
from xraylabtool.kk import _device
from xraylabtool.kk._device import Spectrum
from xraylabtool.kk._shapes import bucket, bucket_pow2
from xraylabtool.kk.database import DeviceTables, load_asf

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray


def _real_float64(name: str, values: ArrayLike) -> NDArray[np.float64]:
    """float64 array of ``values``; complex input raises instead of losing its imaginary part."""
    arr = np.asarray(values)
    if arr.dtype.kind == "c":
        raise ValueError(f"{name} must be real numbers, got dtype {arr.dtype}")
    return arr.astype(np.float64, copy=False)


def _atomic_numbers(symbols: Sequence[str], counts: Sequence[float]) -> list[int]:
    """Atomic numbers of ``symbols`` after the formula checks of :func:`compound_spectrum`."""
    if len(symbols) == 0:
        raise FormulaError("No elements given")
    if len(symbols) != len(counts):
        raise FormulaError(
            f"symbols and counts differ in length ({len(symbols)} != {len(counts)})"
        )
    db = load_asf()
    atomic_numbers = []
    for symbol, count in zip(symbols, counts, strict=True):
        if symbol not in db.symbols:
            raise FormulaError(f"Unsupported element {symbol!r} (Z = 1..92 only)")
        try:
            valid = not np.iscomplexobj(count) and math.isfinite(count) and count > 0
        except (TypeError, ValueError):  # not a real number at all (str, None, ...)
            valid = False
        if not valid:
            raise FormulaError(
                f"Element count must be finite and positive, got {count!r} "
                f"for {symbol!r}"
            )
        atomic_numbers.append(db.symbols.index(symbol) + 1)
    return atomic_numbers


def _element_arrays(
    atomic_numbers: Sequence[int], counts: Sequence[float]
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    """Atomic numbers and counts padded to a bucketed length (padded counts are 0)."""
    k = bucket_pow2(len(atomic_numbers))
    z_pad = np.ones(k, dtype=np.int32)
    z_pad[: len(atomic_numbers)] = atomic_numbers
    n_pad = np.zeros(k)
    n_pad[: len(atomic_numbers)] = counts
    return z_pad, n_pad


def compound_spectrum(
    symbols: Sequence[str], counts: Sequence[float]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Piecewise-polynomial f2 of a formula unit on the union of element grids.

    The segment edges are the sorted union of the elemental edges; on each
    union segment the coefficients are the count-weighted sum of the
    polynomial that each element uses there (``Σ nᵢ f2ᵢ``).

    Parameters
    ----------
    symbols : sequence of str
        Element symbols (case-sensitive, ``"H"`` .. ``"U"``).
    counts : sequence of float
        Atoms of each element per formula unit; finite and positive.

    Returns
    -------
    seg_ev : numpy.ndarray, shape (n + 1,)
        Strictly increasing segment edges in eV (10 eV to 500 keV).
    coeffs : numpy.ndarray, shape (n, 5)
        Columns ``a₁, a₀, a₋₁, a₋₂, a₋₃`` of ``E¹ … E⁻³`` (``E`` in eV) for
        the segment ``[seg_ev[i], seg_ev[i + 1])``.

    Raises
    ------
    FormulaError
        If no elements are given, ``symbols`` and ``counts`` differ in length,
        a symbol is not a supported element (Z = 1..92), or a count is not
        finite and positive.

    Notes
    -----
    Elements are accumulated in the order given, so the sums are bit-identical
    to upstream's. All elements share the 10 eV - 500 keV range.

    References
    ----------
    B. Watts, Opt. Express 22(19), 23628 (2014): the f2 spectrum is the
    stoichiometric sum of the tabulated piecewise polynomials entering
    Eq. (8). The union-grid summation is upstream-only
    (``kkcalc.data.calculate_asf``), not an equation of the paper.

    Examples
    --------
    >>> from xraylabtool.kk import compound_spectrum
    >>> seg_ev, coeffs = compound_spectrum(["Si", "O"], [1.0, 2.0])
    >>> seg_ev.shape[0] == coeffs.shape[0] + 1
    True
    >>> coeffs.shape[1]
    5
    >>> float(seg_ev[0]), float(seg_ev[-1])
    (10.0, 500000.0)
    """
    atomic_numbers = _atomic_numbers(symbols, counts)
    z_pad, n_pad = _element_arrays(atomic_numbers, counts)
    db = load_asf()
    _device.ensure_x64()
    tables = db.device_tables()
    z_d, n_d, n_el = _device.upload((z_pad, n_pad, np.int32(len(atomic_numbers))))
    spec: Spectrum = compound_spectrum_device(
        z_d, n_d, n_el, tables, capacity=db.compound_capacity(atomic_numbers)
    )
    return spec.to_numpy()


def _union_knots(flat: jax.Array, size: int) -> tuple[jax.Array, jax.Array]:
    """Sorted distinct finite values of ``flat`` in ``size`` slots, and their count.

    Slots past the count hold ``+inf``. The count is that of the distinct
    finite values of ``flat`` itself, computed independently of the output
    allocation, so an undersized ``size`` truncates the output but shows as a
    count above ``size`` instead of hiding as ``size``. Swappable primitive:
    ``jnp.unique(size=…)`` for the output measured faster than sort plus
    mask-compaction on CPU.
    """
    out = jnp.unique(flat, size=size, fill_value=jnp.inf)
    ordered = jnp.sort(flat)
    distinct = jnp.isfinite(ordered) & jnp.concatenate(
        [jnp.ones(1, dtype=bool), ordered[1:] != ordered[:-1]]
    )
    return out, jnp.sum(distinct, dtype=jnp.int32)


@partial(jax.jit, static_argnames="capacity")
def compound_spectrum_device(
    z: jax.Array,
    counts: jax.Array,
    n_elements: jax.Array,
    tables: DeviceTables,
    *,
    capacity: int,
) -> Spectrum:
    """Piecewise-polynomial f2 of a formula unit, assembled on the device.

    Device counterpart of :func:`compound_spectrum`: the segment edges are the
    sorted union of the elemental edges, and each union segment carries the
    count-weighted sum of the polynomial every element uses there.

    Parameters
    ----------
    z : jax.Array, shape (k,)
        Atomic numbers (1..92) of the element list, padded to a bucketed ``k``.
    counts : jax.Array, shape (k,)
        Atoms per formula unit (float64); padded entries are ignored.
    n_elements : jax.Array
        Number of real entries in ``z`` (int scalar, at least 1).
    tables : DeviceTables
        Packed tables from :meth:`AsfDatabase.device_tables
        <xraylabtool.kk.database.AsfDatabase.device_tables>`.
    capacity : int
        Static segment capacity, at least the true segment count; a smaller
        value silently truncates the union. :meth:`AsfDatabase.compound_capacity
        <xraylabtool.kk.database.AsfDatabase.compound_capacity>` is a safe
        sum bound; the pipeline sizes it to the exact union instead
        (``api._compound_segments``), which is never larger.

    Returns
    -------
    Spectrum
        Padded to ``capacity`` with the pad rule of :class:`Spectrum`;
        ``n_segments`` is the true count.

    Notes
    -----
    The padding of every gathered knot row (and of padded elements) is masked
    to ``+inf`` before the union, so no dummy knot can be mistaken for a real
    one. Elements are accumulated in the given order, so the sums match the
    host reference to rounding and do not depend on ``capacity``.

    References
    ----------
    Design spec §5 (``2026-10-25-kk-jax-first-design.md``); the union-grid
    summation is upstream's (``kkcalc.data.calculate_asf``), not an equation
    of B. Watts, Opt. Express 22(19), 23628 (2014).
    """
    k = z.shape[0]
    rows = z.astype(jnp.int32) - 1
    live = jnp.arange(k) < n_elements
    real = (
        jnp.arange(tables.knots.shape[1])[None, :] < tables.n_knots[rows][:, None]
    ) & live[:, None]
    union, n_knots = _union_knots(
        jnp.where(real, tables.knots[rows], jnp.inf).ravel(), capacity + 1
    )
    pos = jnp.arange(capacity + 1)
    last = union[jnp.maximum(n_knots - 1, 0)]
    seg_ev = jnp.where(pos < n_knots, union, last * (pos - n_knots + 2))
    n_segments = n_knots - 1

    lower = seg_ev[:-1]

    def add_element(
        acc: jax.Array, row: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, None]:
        i, count = row
        # The whole padded row is finite and increasing; the clip handles its tail.
        idx = jnp.clip(
            jnp.searchsorted(tables.knots[i], lower, side="right") - 1,
            0,
            tables.n_knots[i] - 2,
        )
        # The barrier keeps XLA from contracting multiply-add into one FMA, so
        # the sum rounds like upstream's separate multiply and add.
        weighted = jax.lax.optimization_barrier(count * tables.coeffs[i][idx])
        return acc + weighted, None

    coeffs, _ = jax.lax.scan(
        add_element,
        jnp.zeros((capacity, 5)),
        (rows, jnp.where(live, counts, 0.0)),
    )
    coeffs = jnp.where((jnp.arange(capacity) < n_segments)[:, None], coeffs, 0.0)
    return Spectrum(seg_ev, coeffs, n_segments)


def evaluate_f2(
    energy_ev: ArrayLike, seg_ev: ArrayLike, coeffs: ArrayLike
) -> NDArray[np.float64]:
    """Evaluate the piecewise Laurent polynomial f2 at given energies.

    ``f2(E) = a₁E + a₀ + a₋₁/E + a₋₂/E² + a₋₃/E³`` with the coefficients of
    the segment ``[seg_ev[i], seg_ev[i + 1])`` containing ``E``. Segments are
    right-continuous, so an energy on an interior edge uses the segment that
    starts there; the final edge uses the last polynomial, as upstream does.

    Parameters
    ----------
    energy_ev : array_like, shape (m,)
        Photon energies in eV, within ``[seg_ev[0], seg_ev[-1]]``.
    seg_ev : array_like, shape (n + 1,)
        Strictly increasing segment edges in eV.
    coeffs : array_like, shape (n, 5)
        Coefficients ``a₁, a₀, a₋₁, a₋₂, a₋₃`` per segment.

    Returns
    -------
    numpy.ndarray of float64, shape (m,)
        f2 in electrons per formula unit.

    Raises
    ------
    ValueError
        If an input is complex or has the wrong shape, ``energy_ev`` is not
        finite or lies outside the segment range, or ``seg_ev`` is not finite
        and strictly increasing.

    References
    ----------
    B. Watts, Opt. Express 22(19), 23628 (2014), Eq. (8): the piecewise
    Laurent form of f2. The right-continuous edge convention is upstream's
    (``kkcalc.data.coeffs_to_ASF``).

    Examples
    --------
    >>> from xraylabtool.kk import evaluate_f2
    >>> seg = [1.0, 2.0, 4.0]
    >>> c = [[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0, 0.0]]
    >>> evaluate_f2([1.5, 2.0, 4.0], seg, c).tolist()
    [1.0, 1.0, 0.5]
    """
    e = _real_float64("energy_ev", energy_ev)
    seg = _real_float64("seg_ev", seg_ev)
    c = _real_float64("coeffs", coeffs)
    if e.ndim != 1:
        raise ValueError(f"energy_ev must be 1-D, got shape {e.shape}")
    if seg.ndim != 1 or seg.size < 2:
        raise ValueError(
            f"seg_ev must be 1-D with at least 2 edges, got shape {seg.shape}"
        )
    if c.shape != (seg.size - 1, 5):
        raise ValueError(f"coeffs must have shape ({seg.size - 1}, 5), got {c.shape}")
    if not np.isfinite(e).all():
        raise ValueError("energy_ev must be finite")
    if not np.isfinite(seg).all():
        raise ValueError("seg_ev must be finite")
    if (np.diff(seg) <= 0).any():
        raise ValueError("seg_ev must be strictly increasing")
    if e.size and (e.min() < seg[0] or e.max() > seg[-1]):
        raise ValueError(
            f"energy_ev must lie within [{seg[0]}, {seg[-1]}] eV, "
            f"got [{e.min()}, {e.max()}]"
        )
    if e.size == 0:
        return np.empty(0)
    n = c.shape[0]
    cap = bucket(n)
    # Pad edges only need to increase: they are never selected (the index is
    # clamped to the last real segment), and this keeps the wrapper's
    # historical acceptance of non-positive edges.
    seg_p = np.concatenate(
        [seg, seg[-1] + (seg[-1] - seg[0]) * np.arange(1, cap - n + 1)]
    )
    c_p = np.concatenate([c, np.zeros((cap - n, 5))])
    e_p = np.concatenate([e, np.full(bucket_pow2(e.size) - e.size, seg[0])])
    _device.ensure_x64()
    e_d, spec = _device.upload((e_p, Spectrum(seg_p, c_p, np.int32(n))))  # type: ignore[arg-type]
    return np.asarray(
        _device.fetch(evaluate_f2_device(e_d, spec))[: e.size], dtype=np.float64
    )


@jax.jit
def evaluate_f2_device(energy_ev: jax.Array, spectrum: Spectrum) -> jax.Array:
    """Evaluate the piecewise Laurent polynomial f2 on the device.

    Same convention as :func:`evaluate_f2`: right-continuous at interior
    edges, and the segment index is clamped to the last *real* segment
    ``n_segments - 1``, so the final real edge and anything above it use the
    last real polynomial and the padding is never selected. No range check is
    made (the host wrapper validates).

    Parameters
    ----------
    energy_ev : jax.Array
        Photon energies in eV (float64, any shape, including empty).
    spectrum : Spectrum
        Padded table, see :class:`~xraylabtool.kk.Spectrum`.

    Returns
    -------
    jax.Array
        f2 in electrons per formula unit, same shape as ``energy_ev``.

    References
    ----------
    B. Watts, Opt. Express 22(19), 23628 (2014), Eq. (8); design spec §5
    (``2026-10-25-kk-jax-first-design.md``).
    """
    e = energy_ev
    idx = jnp.clip(
        jnp.searchsorted(spectrum.seg_ev, e, side="right") - 1,
        0,
        spectrum.n_segments - 1,
    )
    k = spectrum.coeffs[idx]
    # An FMA of a1*E + a0 rounds once: the exact value, but not upstream's
    # (NumPy's). Near a1*E ~ -a0 the two differ by ~3e-12 relative (Si at
    # 1838.8 eV), beyond the goldens' rtol 1e-12. The barrier keeps the
    # product rounded as upstream's; drop it to get the FMA result.
    linear: jax.Array = jax.lax.optimization_barrier(k[..., 0] * e)
    return linear + k[..., 1] + k[..., 2] / e + k[..., 3] / e**2 + k[..., 4] / e**3
