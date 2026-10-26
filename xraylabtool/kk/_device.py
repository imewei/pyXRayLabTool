"""Device boundary of the Kramers-Kronig pipeline: x64 policy, transfers, ``Spectrum``.

Everything that crosses the host/device boundary goes through :func:`upload`,
:func:`fetch_status` or :func:`fetch`; each opens a narrow
``jax.transfer_guard("allow")`` scope and bumps a counter, which is what the
transfer-budget tests assert. :class:`Spectrum` is the padded piecewise
polynomial f2 table the device kernels consume.

References
----------
Design: ``docs/superpowers/specs/2026-10-25-kk-jax-first-design.md`` §1-§2.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
import math
import os
import threading
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from xraylabtool.exceptions import CalculationError
from xraylabtool.kk._shapes import bucket

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

_counts: dict[str, int] = {"upload": 0, "fetch_status": 0, "fetch": 0}
_counts_lock = threading.Lock()


def ensure_x64() -> None:
    """Enable 64-bit JAX arrays (a no-op when already on).

    Called at import and at every concrete public entry before any upload or
    trace: with x64 off a float64 buffer would be canonicalised to float32
    inside a later trace.

    References
    ----------
    Design spec §4 (``2026-10-25-kk-jax-first-design.md``).

    Examples
    --------
    >>> from xraylabtool.kk._device import ensure_x64
    >>> ensure_x64()
    """
    jax.config.update("jax_enable_x64", True)


ensure_x64()


def worker_guard() -> AbstractContextManager[Any]:
    """The transfer-guard scope a worker entry opens around a kk call under tests.

    The guard is thread-local, so every worker thread opens its own. It is on
    when ``PYTEST_CURRENT_TEST`` or ``XRAYLABTOOL_TRANSFER_GUARD`` is set
    (the latter for subprocesses a test starts) and a no-op otherwise.

    References
    ----------
    Design spec §1 and §10 (``2026-10-25-kk-jax-first-design.md``).

    Examples
    --------
    >>> from xraylabtool.kk._device import worker_guard
    >>> with worker_guard():
    ...     pass
    """
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get(
        "XRAYLABTOOL_TRANSFER_GUARD"
    ):
        return jax.transfer_guard("disallow_explicit")
    return nullcontext()


def _bump(kind: str) -> None:
    with _counts_lock:
        _counts[kind] += 1


def transfer_counts() -> dict[str, int]:
    """Number of boundary calls so far, by kind (``upload``, ``fetch_status``, ``fetch``)."""
    with _counts_lock:
        return dict(_counts)


def reset_transfer_counts() -> None:
    """Set all boundary counters to zero."""
    with _counts_lock:
        for kind in _counts:
            _counts[kind] = 0


def upload[T](tree: T) -> T:
    """``jax.device_put`` of a whole pytree: one counted host-to-device transfer."""
    _bump("upload")
    with jax.transfer_guard("allow"):
        return jax.device_put(tree)  # type: ignore[no-any-return]


def fetch_status[T](tree: T) -> T:
    """``jax.device_get`` of a small status pytree: one counted transfer."""
    _bump("fetch_status")
    with jax.transfer_guard("allow"):
        return jax.device_get(tree)  # type: ignore[no-any-return]


def fetch[T](tree: T) -> T:
    """``jax.device_get`` of the final output pytree: one counted transfer."""
    _bump("fetch")
    with jax.transfer_guard("allow"):
        return jax.device_get(tree)  # type: ignore[no-any-return]


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
    the order and with the messages of the pre-migration ``kk_real_spectrum``:
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
