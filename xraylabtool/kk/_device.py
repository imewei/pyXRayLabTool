"""Device boundary of the Kramers-Kronig pipeline: ``Spectrum`` and the runtime re-exports.

The generic x64 policy, transfer counters and compile-cache helpers live in
:mod:`xraylabtool.backend.runtime`; this module re-exports them for the kk
modules and keeps the x64 switch-on at import. :class:`Spectrum` is the padded
piecewise polynomial f2 table the device kernels consume.

References
----------
Design: ``docs/superpowers/specs/2026-10-25-kk-jax-first-design.md`` §1-§2.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from xraylabtool.backend.runtime import (
    enable_persistent_cache,
    ensure_x64,
    fetch,
    fetch_status,
    reset_transfer_counts,
    transfer_counts,
    upload,
    worker_guard,
)
from xraylabtool.backend.shapes import bucket
from xraylabtool.exceptions import CalculationError

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "Spectrum",
    "check_capacity",
    "enable_persistent_cache",
    "ensure_x64",
    "fetch",
    "fetch_status",
    "host_spectrum",
    "reset_transfer_counts",
    "transfer_counts",
    "upload",
    "validated_arrays",
    "worker_guard",
]

ensure_x64()


def check_capacity(n_segments: int, capacity: int) -> None:
    """Raise if a fetched true segment count exceeds the capacity it was built at.

    ``n_segments`` is the unclamped count of the distinct knots, so a static
    allocation that was too small shows up here instead of silently
    truncating the table.

    Raises
    ------
    CalculationError
        If ``n_segments > capacity`` (an internal sizing error, never user input).

    References
    ----------
    Design spec §3 (``2026-10-25-kk-jax-first-design.md``).

    Examples
    --------
    >>> from xraylabtool.kk._device import check_capacity
    >>> check_capacity(5, 8)
    """
    if n_segments > capacity:
        raise CalculationError(
            f"internal error: the spectrum has {n_segments} segments but its "
            f"capacity {capacity} is smaller, so the table would be truncated"
        )


def validated_arrays(
    seg_ev: ArrayLike,
    coeffs: ArrayLike,
    eval_ev: NDArray[np.float64] | None = None,
    rel_corr: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Host validation of a segment table: float64 ``(seg_ev, coeffs)`` or ``ValueError``.

    With ``eval_ev`` (already float64) and ``rel_corr`` it also checks those, in
    the order and with the messages that ``kk_real_spectrum`` documents:
    conversions, shapes (``eval_ev``, ``seg_ev``, ``coeffs``), finiteness
    (``eval_ev``, ``seg_ev``, ``coeffs``, ``rel_corr``), positivity, ordering.
    """
    # Deferred: spectrum.py imports this module for the device assembly.
    from xraylabtool.kk.spectrum import _real_float64  # noqa: PLC0415

    seg = _real_float64("seg_ev", seg_ev)
    c = _real_float64("coeffs", coeffs)
    rel = None if rel_corr is None else float(rel_corr)
    if eval_ev is not None and eval_ev.ndim != 1:
        raise ValueError(f"eval_ev must be 1-D, got shape {eval_ev.shape}")
    if seg.ndim != 1 or seg.size < 2:
        raise ValueError(
            f"seg_ev must be 1-D with at least 2 edges, got shape {seg.shape}"
        )
    if c.shape != (seg.size - 1, 5):
        raise ValueError(f"coeffs must have shape ({seg.size - 1}, 5), got {c.shape}")
    arrays = [("seg_ev", seg), ("coeffs", c)]
    if eval_ev is not None:
        arrays.insert(0, ("eval_ev", eval_ev))
    for label, arr in arrays:
        if not np.isfinite(arr).all():
            raise ValueError(f"{label} must be finite")
    if rel is not None and not math.isfinite(rel):
        raise ValueError("rel_corr must be finite")
    if eval_ev is None:
        if (seg <= 0).any():
            raise ValueError("seg_ev must be positive")
    elif (eval_ev <= 0).any() or (seg <= 0).any():
        raise ValueError("eval_ev and seg_ev must be positive")
    if (np.diff(seg) <= 0).any():
        raise ValueError("seg_ev must be strictly increasing")
    return seg, c


@dataclass(frozen=True)
class Spectrum:
    """Padded piecewise Laurent polynomial f2 table (a JAX pytree).

    The capacity ``N`` is a shape bucket at least as large as the true number
    of segments ``n``. Knots beyond the last real knot ``x_n`` are
    ``x_n·(1 + k)``, ``k = 1, 2, …`` (strictly increasing and finite) with
    all-zero coefficients, which contribute exactly nothing to the transform.
    ``n_segments`` is a traced int32 leaf, not static metadata, so spectra of
    different true sizes but equal capacity can be stacked and ``vmap``-ed;
    unflattening performs no validation, so it is safe on tracers.

    Parameters
    ----------
    seg_ev : jax.Array, shape (N + 1,)
        Segment edges in eV, padded as described above.
    coeffs : jax.Array, shape (N, 5)
        Columns ``a₁, a₀, a₋₁, a₋₂, a₋₃`` per segment; rows ``≥ n_segments``
        are ignored by the kernels.
    n_segments : jax.Array
        True segment count ``n`` (int32 scalar).

    Notes
    -----
    Pad knots grow as ``x_n·(1 + k)`` and the kernel squares them, so for
    absurd edges (about 1e152 eV and above, far outside the tabulated
    10 eV-500 keV range) the finite pad knots overflow to ``inf`` and the
    result is NaN where the unpadded kernel is still finite. Use
    ``capacity == n`` (no padding) there.

    References
    ----------
    B. Watts, Opt. Express 22(19), 23628 (2014), Eq. (8) (the piecewise
    representation); the padding rule is this package's addition, see the
    design spec §2.

    Examples
    --------
    >>> import numpy as np
    >>> from xraylabtool.kk import Spectrum
    >>> spec = Spectrum.from_numpy([2.0, 5.0], [[0.0, 1.0, 0.0, 0.0, 0.0]], 4)
    >>> spec.coeffs.shape, int(spec.n_segments)
    ((4, 5), 1)
    """

    seg_ev: jax.Array
    coeffs: jax.Array
    n_segments: jax.Array

    @classmethod
    def from_numpy(
        cls, seg_ev: ArrayLike, coeffs: ArrayLike, capacity: int | None = None
    ) -> Spectrum:
        """Validate a host table, pad it to ``capacity`` and upload it.

        Parameters
        ----------
        seg_ev : array_like, shape (n + 1,)
            Strictly increasing, positive, finite edges in eV, ``n ≥ 1``.
        coeffs : array_like, shape (n, 5)
            Finite coefficients per segment.
        capacity : int, optional
            Segment capacity ``N ≥ n``; default ``bucket(n)``.

        Returns
        -------
        Spectrum
            Device-resident, one counted :func:`upload`.

        Raises
        ------
        ValueError
            If the table is malformed (see :func:`validated_arrays`) or
            ``capacity < n``.

        References
        ----------
        Design spec §2-§3 (``2026-10-25-kk-jax-first-design.md``).

        Examples
        --------
        >>> from xraylabtool.kk import Spectrum
        >>> Spectrum.from_numpy([2.0, 5.0], [[0.0, 1.0, 0.0, 0.0, 0.0]]).seg_ev.shape
        (513,)
        """
        ensure_x64()
        return upload(host_spectrum(seg_ev, coeffs, capacity))

    def to_numpy(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Download and slice to the true size.

        Returns
        -------
        seg_ev : numpy.ndarray, shape (n + 1,)
        coeffs : numpy.ndarray, shape (n, 5)

        Raises
        ------
        CalculationError
            If the true segment count exceeds the capacity (the table would
            be truncated).

        References
        ----------
        Design spec §2 (``2026-10-25-kk-jax-first-design.md``).

        Examples
        --------
        >>> from xraylabtool.kk import Spectrum
        >>> seg, c = Spectrum.from_numpy([2.0, 5.0], [[0.0, 1.0, 0.0, 0.0, 0.0]]).to_numpy()
        >>> seg.tolist(), c.shape
        ([2.0, 5.0], (1, 5))
        """
        seg, c, n_seg = fetch((self.seg_ev, self.coeffs, self.n_segments))
        n = int(n_seg)
        check_capacity(n, c.shape[0])
        return (
            np.asarray(seg[: n + 1], dtype=np.float64),
            np.asarray(c[:n], dtype=np.float64),
        )


def host_spectrum(
    seg_ev: ArrayLike, coeffs: ArrayLike, capacity: int | None = None
) -> Spectrum:
    """Validated, padded table with NumPy leaves (no transfer); see ``Spectrum.from_numpy``."""
    seg, c = validated_arrays(seg_ev, coeffs)
    n = c.shape[0]
    cap = bucket(n) if capacity is None else capacity
    if cap < n:
        raise ValueError(f"capacity {cap} is below the segment count {n}")
    pad = cap - n
    seg_p = np.concatenate([seg, seg[-1] * (1.0 + np.arange(1, pad + 1))])
    c_p = np.concatenate([c, np.zeros((pad, 5))])
    return Spectrum(seg_p, c_p, np.int32(n))  # type: ignore[arg-type]


def _flatten(s: Spectrum) -> tuple[tuple[Any, Any, Any], None]:
    return (s.seg_ev, s.coeffs, s.n_segments), None


def _unflatten(_aux: None, children: tuple[Any, Any, Any]) -> Spectrum:
    return Spectrum(*children)


jax.tree_util.register_pytree_node(Spectrum, _flatten, _unflatten)
