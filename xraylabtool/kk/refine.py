"""Adaptive refinement of a sampled scattering-factor spectrum.

Altered from the original: a re-implementation of ``improve_accuracy`` from
kkcalc/kk.py of the Kramers-Kronig Calculator (kkcalc,
https://github.com/benajamin/kkcalc, commit
f331ca31a654af946f7d550a22cf4e9c11fa8afc), Copyright (c) 2013 Benjamin Watts,
Daniel J. Lauk, zlib/libpng licence (full text in
``xraylabtool/data/kkcalc/LICENSE-kkcalc.txt``; see also ``NOTICE``).

Differences from upstream: the grid need not coincide with the segment edges
of the tabulated spectrum (f2 is evaluated at each midpoint from the segment
that contains it), an interval next to a tabulated range end is left alone
where f2 exceeds the tolerance there (upstream skips its first interval
unconditionally, see :func:`refine_spectrum`), the pass and point budgets are
explicit and reported, and an interval that float spacing cannot halve ends
the loop instead of repeating.

Energies are in eV and the tolerance is absolute, in electrons per formula
unit. The passes are host-orchestrated and device-resident: the number of new
points is data dependent, so the host runs one jitted pass per bisection round
on bucketed device buffers and reads back a handful of integer counters per
pass (the continue/stop/rebucket decision and the per-pass log line); the
sorted union is downloaded once at the end (design spec §7).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

# Imported at runtime so ``typing.get_type_hints`` resolves the annotations.
import numpy as np
from numpy.typing import ArrayLike, NDArray

from xraylabtool.exceptions import ValidationError
from xraylabtool.kk import _device
from xraylabtool.kk._device import Spectrum
from xraylabtool.kk._shapes import block_for, bucket_pow2
from xraylabtool.kk.result import _readonly_1d
from xraylabtool.kk.spectrum import _real_float64, evaluate_f2_device
from xraylabtool.kk.transform import _f64, _live_table, kk_piecewise_poly
from xraylabtool.logging_utils import get_logger

logger = get_logger("kk")

# An interval at a singular range end narrower than this fraction of its
# energy is not refined (see refine_spectrum, Notes). The first tabulated
# segment is 1.6 % wide for every element's lower end.
_POLE_REL_WIDTH = 0.02

# Smallest active-interval buffer. The midpoint loop costs in proportion to the
# live intervals, not the capacity (see _midpoint_f1), so a generous floor is
# cheap and every grid of up to this many intervals (all tabulated formulas,
# 505-1230 knots) shares the same compiled passes.
_MIN_ACTIVE = 2048

# Largest ``max_points`` left that still fits a traced int64 argument.
_INT_MAX = 2**62


@dataclass(frozen=True, eq=False)
class RefineOutcome:
    """Result of :func:`refine_spectrum`.

    The dataclass is frozen and compares by identity (``eq=False``): the
    array fields have no element-wise-safe ``==`` or ``hash``.

    Attributes
    ----------
    energy_ev : numpy.ndarray, shape (m + n_added,)
        Strictly increasing energies in eV: the input grid plus the added
        midpoints.
    f1 : numpy.ndarray, shape (m + n_added,)
        Real part at ``energy_ev``, electrons per formula unit.
    f2 : numpy.ndarray, shape (m + n_added,)
        Imaginary part at ``energy_ev``, electrons per formula unit.
    n_added : int
        Number of midpoints added.
    exhausted : bool
        ``True`` if refinement stopped on a limit (``max_passes``,
        ``max_points`` or float spacing) before every interval met the
        tolerance.
    n_pole_skipped : int
        Number of intervals next to a tabulated range end, narrower than 2 %
        of their energy, that were left unrefined because f1 is
        logarithmically singular there (see :func:`refine_spectrum`).

    References
    ----------
    Upstream-only: ``improve_accuracy`` of kkcalc is not part of B. Watts,
    Opt. Express 22(19), 23628 (2014).

    Examples
    --------
    >>> import numpy as np
    >>> from xraylabtool.kk import RefineOutcome
    >>> out = RefineOutcome(np.array([1.0]), np.array([2.0]), np.array([3.0]), 0, False)
    >>> out.n_added, out.exhausted, out.n_pole_skipped
    (0, False, 0)
    """

    energy_ev: NDArray[np.float64]
    f1: NDArray[np.float64]
    f2: NDArray[np.float64]
    n_added: int
    exhausted: bool
    n_pole_skipped: int = 0

    def __post_init__(self) -> None:
        energy = _readonly_1d("energy_ev", self.energy_ev)
        f1 = _readonly_1d("f1", self.f1)
        f2 = _readonly_1d("f2", self.f2)
        if not energy.size == f1.size == f2.size:
            raise ValidationError(
                "energy_ev, f1 and f2 must have the same length",
                "f1",
                (energy.size, f1.size, f2.size),
            )
        if energy.size > 1 and not (np.diff(energy) > 0).all():
            raise ValidationError("energy_ev must be strictly increasing", "energy_ev")
        for name in ("n_added", "n_pole_skipped"):
            if getattr(self, name) < 0:
                raise ValidationError(f"{name} must be >= 0", name, getattr(self, name))
        object.__setattr__(self, "energy_ev", energy)
        object.__setattr__(self, "f1", f1)
        object.__setattr__(self, "f2", f2)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ValidationError(f"{name} must be a positive integer", name, value)
    return int(value)


def _grid_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        if np.iscomplexobj(values):
            raise ValidationError(f"{name} must be real numbers, got complex", name)
        arr = np.array(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be real numbers: {exc}", name) from exc
    if arr.ndim != 1 or arr.size == 0:
        raise ValidationError(f"{name} must be a non-empty 1-D array", name, arr.shape)
    if not np.isfinite(arr).all():
        raise ValidationError(f"{name} must be finite", name)
    return arr


def _checked_inputs(
    energy_ev: ArrayLike,
    f1: ArrayLike,
    f2: ArrayLike,
    seg_ev: ArrayLike,
    tolerance: float,
    max_passes: int,
    max_points: int,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Validated ``(energy, f1, f2, seg)`` float64 arrays of :func:`refine_spectrum`."""
    e = _grid_array(energy_ev, "energy_ev")
    f1_arr = _grid_array(f1, "f1")
    f2_arr = _grid_array(f2, "f2")
    if not (e.size == f1_arr.size == f2_arr.size):
        raise ValidationError(
            "energy_ev, f1 and f2 must have the same length",
            "shapes",
            (e.size, f1_arr.size, f2_arr.size),
        )
    if e[0] <= 0 or (np.diff(e) <= 0).any():
        raise ValidationError(
            "energy_ev must be positive, strictly increasing and unique",
            "energy_ev",
        )
    seg = _real_float64("seg_ev", seg_ev)
    if e[0] < seg[0] or e[-1] > seg[-1]:
        raise ValidationError(
            "energy_ev must lie inside the tabulated range",
            "energy_ev",
            (float(e[0]), float(e[-1])),
        )
    if isinstance(tolerance, bool) or not isinstance(
        tolerance, int | float | np.number
    ):
        raise ValidationError("tolerance must be a number", "tolerance", tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValidationError(
            "tolerance must be finite and positive", "tolerance", tolerance
        )
    _positive_int(max_passes, "max_passes")
    _positive_int(max_points, "max_points")

    return e, f1_arr, f2_arr, seg


class _Knobs(NamedTuple):
    """Scalars of one refinement call, uploaded with its other inputs.

    Keeping them on the device means the pass loop makes no host-to-device
    transfer of its own.
    """

    tolerance: jax.Array  # float64
    max_points: jax.Array  # int64, clipped to 2**62
    m: jax.Array  # int32, valid points of the input grid


class _PassState(NamedTuple):
    """Device state between refinement passes (a pytree).

    ``iv`` holds the active intervals as rows ``lo, hi, f1_lo, f1_hi, f2_lo,
    f2_hi, pole_lo, pole_hi`` (the pole rows are 0.0 or 1.0; a flag marks an
    end that is a tabulated range end where f1 is singular, ``|f2| >
    tolerance``). Its capacity ``B`` is a power of two; columns at or beyond
    ``n_active`` are zero. ``acc`` collects the added points as rows ``energy,
    f1, f2`` in the order they were accepted, ``n_acc`` of them;
    ``points_left`` is what remains of ``max_points``. The sliver counters
    accumulate over all passes.
    """

    iv: jax.Array
    n_active: jax.Array
    acc: jax.Array
    n_acc: jax.Array
    points_left: jax.Array
    n_sliver: jax.Array
    n_wide: jax.Array
    widest: jax.Array
    widest_ev: jax.Array

    @property
    def lo(self) -> jax.Array:
        """Lower ends of the active intervals, ``(B,)``."""
        return self.iv[0]

    @property
    def hi(self) -> jax.Array:
        """Upper ends of the active intervals, ``(B,)``."""
        return self.iv[1]


class _PassStatus(NamedTuple):
    """Integer scalars read back after each pass: all the host loop needs."""

    n_live: jax.Array  # intervals left after the sliver split
    n_unsplit: jax.Array  # of those, midpoint equals an end in floating point
    n_checked: jax.Array  # intervals whose midpoint was evaluated
    n_accepted: jax.Array  # midpoints kept (after the max_points truncation)
    hit_max_points: jax.Array  # the truncation fired
    n_next_active: jax.Array  # intervals handed to the next pass


class _RefineArrays(NamedTuple):
    """Device result of a refinement: sorted union plus the sliver statistics.

    ``energy``, ``f1`` and ``f2`` hold ``m + n_added`` valid entries first;
    anything after them is ``inf`` (energy) or garbage and must be sliced off
    with the host-known count.
    """

    energy: jax.Array
    f1: jax.Array
    f2: jax.Array
    n_sliver: jax.Array
    n_wide: jax.Array
    widest: jax.Array
    widest_ev: jax.Array


@dataclass(frozen=True)
class _RefineInfo:
    """Host-known outcome of the pass loop (no device data)."""

    m: int
    passes: int
    added: int
    reason: str
    accum_start: int = 0  # added-points capacity at the first pass
    accum: int = 0  # ... and at the end (grown on demand)

    @property
    def n_total(self) -> int:
        """Number of valid entries of the sorted union."""
        return self.m + self.added


def _resize(x: jax.Array, n: int) -> jax.Array:
    """First ``n`` entries of a 1-D array, zero-padded if it is shorter."""
    if x.shape[0] >= n:
        return x[:n]
    return jnp.pad(x, (0, n - x.shape[0]))


def _start_state(
    energy: jax.Array,
    f1: jax.Array,
    f2: jax.Array,
    spectrum: Spectrum,
    knobs: _Knobs,
    *,
    active: int,
    accum: int,
) -> _PassState:
    """Initial state: one active interval per neighbouring pair of the grid.

    ``energy``, ``f1`` and ``f2`` have a bucketed length of at least
    ``knobs.m``; their entries beyond it are ignored. ``active`` and ``accum``
    are the static buffer capacities (at least ``m - 1`` and the most points
    that can be added).
    """
    tolerance = knobs.tolerance
    n_int = knobs.m - 1
    live = jnp.arange(active) < n_int

    def head(x: jax.Array) -> jax.Array:
        return jnp.where(live, _resize(x[:-1], active), 0.0)

    def tail(x: jax.Array) -> jax.Array:
        return jnp.where(live, _resize(x[1:], active), 0.0)

    lo, hi = head(energy), tail(energy)
    seg = spectrum.seg_ev
    pole_lo = live & (lo == seg[0]) & (jnp.abs(head(f2)) > tolerance)
    pole_hi = live & (hi == seg[spectrum.n_segments]) & (jnp.abs(tail(f2)) > tolerance)
    iv = jnp.stack(
        [
            lo,
            hi,
            head(f1),
            tail(f1),
            head(f2),
            tail(f2),
            pole_lo.astype(jnp.float64),
            pole_hi.astype(jnp.float64),
        ]
    )
    zero = jnp.int32(0)
    return _PassState(
        iv=iv,
        n_active=n_int,
        acc=jnp.zeros((3, accum)),
        n_acc=zero,
        points_left=knobs.max_points,
        n_sliver=zero,
        n_wide=zero,
        widest=jnp.float64(-1.0),
        widest_ev=jnp.float64(0.0),
    )


def _sliver_split(
    state: _PassState, spectrum: Spectrum
) -> tuple[jax.Array, _PassState]:
    """Mask of intervals that stay active; update the sliver counters.

    A sliver is an interval next to a singular range end narrower than
    ``_POLE_REL_WIDTH`` of its energy. It is dropped before the midpoint test.
    "Wide" means wider than the last *real* tabulated segment at its end,
    never the padding.
    """
    iv = state.iv
    lo, hi = iv[0], iv[1]
    pole_lo, pole_hi = iv[6] > 0, iv[7] > 0
    valid = jnp.arange(iv.shape[1]) < state.n_active
    width = hi - lo
    sliver = (
        valid
        & (pole_lo | pole_hi)
        & (width <= _POLE_REL_WIDTH * jnp.where(pole_lo, lo, hi))
    )
    seg, n = spectrum.seg_ev, spectrum.n_segments
    end_seg = jnp.where(pole_lo, seg[1] - seg[0], seg[n] - seg[n - 1])
    n_sliver = jnp.sum(sliver, dtype=jnp.int32)
    # The widest dropped interval, first one on ties (also across passes).
    w = jnp.where(sliver, width, -1.0)
    j = jnp.argmax(w)
    better = w[j] > state.widest
    return valid & ~sliver, state._replace(
        n_sliver=state.n_sliver + n_sliver,
        n_wide=state.n_wide + jnp.sum(sliver & (width > end_seg), dtype=jnp.int32),
        widest=jnp.where(better, w[j], state.widest),
        widest_ev=jnp.where(better, lo[j], state.widest_ev),
    )


# Energies per trip of the midpoint loop (see _midpoint_f1): whole coarse trips
# for the bulk of the live intervals, fine trips for the remainder.
_COARSE_TRIP = 128
_FINE_TRIP = 16


def _trip_sizes(block: int, cap: int) -> tuple[int, int]:
    """``(coarse, fine)`` energies per loop trip: both at most ``block``.

    ``block`` is the ``chunk_size`` bound; the coarse trip also grows with the
    buffer so a large first pass makes few trips.
    """
    return min(block, max(_COARSE_TRIP, cap // 16)), min(block, _FINE_TRIP)


def _midpoint_f1(
    probe: jax.Array,
    spectrum: Spectrum,
    rel_corr: Any,
    n_active: jax.Array,
    *,
    block: int,
) -> jax.Array:
    """f1 at the first ``n_active`` entries of ``probe`` (whole blocks of them).

    Entries past the last block that holds an active one are left at zero.

    The kernel runs over blocks of energies in two loops with dynamic trip
    counts (coarse blocks for the bulk, fine blocks for the remainder), so a
    pass costs in
    proportion to the live intervals and not to the buffer capacity (a typical
    refinement shrinks from ~10^3 intervals to a handful). Each energy is
    evaluated by :func:`~xraylabtool.kk.transform.kk_piecewise_poly` exactly
    as :func:`~xraylabtool.kk.transform.kk_f1` does. The loop has a dynamic
    trip count, so this is not reverse-differentiable (not needed here).
    """
    cap = probe.shape[0]
    coarse, fine = _trip_sizes(block, cap)
    # Room for the coarse and fine trips past the active prefix: no slice clamps.
    padded = jnp.pad(_f64(probe), (0, coarse + fine), mode="edge")
    seg, coeffs = _live_table(spectrum)
    rel = _f64(rel_corr)

    def trips(
        size: int, start: jax.Array, count: jax.Array, out: jax.Array
    ) -> jax.Array:
        def body(i: jax.Array, out: jax.Array) -> jax.Array:
            at = start + i * size
            chunk = jax.lax.dynamic_slice_in_dim(padded, at, size)
            return jax.lax.dynamic_update_slice_in_dim(
                out, kk_piecewise_poly(chunk, seg, coeffs, rel), at, 0
            )

        return jax.lax.fori_loop(0, count, body, out)  # type: ignore[no-any-return]

    n_coarse = n_active // coarse
    done = n_coarse * coarse
    out = trips(coarse, jnp.int32(0), n_coarse, jnp.zeros(cap + coarse + fine))
    out = trips(fine, done, (n_active - done + fine - 1) // fine, out)
    return out[:cap]


def _pass_body(
    state: _PassState,
    spectrum: Spectrum,
    rel_corr: Any,
    tolerance: Any,
    *,
    block: int,
) -> tuple[_PassState, _PassStatus]:
    """One bisection round on the active intervals of ``state``.

    Drops the pole slivers, evaluates f1 (``_midpoint_f1`` in blocks of ``block``)
    and f2 at the midpoints of the splittable rest, keeps those whose
    interpolation error exceeds ``tolerance`` (truncated to the first
    ``state.points_left`` of them in internal order) and returns the next
    active set, all left halves followed by all right halves, plus the
    status counters. Jitted once per active capacity ``B`` (the shape of
    ``state.iv``) and ``block``.
    """
    live, state = _sliver_split(state, spectrum)
    iv = state.iv
    cap = iv.shape[1]
    lo, hi, f1_lo, f1_hi, f2_lo, f2_hi, pole_lo, pole_hi = iv
    mid = 0.5 * (lo + hi)
    splittable = (mid > lo) & (mid < hi)
    ok = live & splittable
    # Lanes that are not checked evaluate at a harmless real energy.
    probe = jnp.where(ok, mid, spectrum.seg_ev[0])
    f1_mid = _midpoint_f1(probe, spectrum, rel_corr, state.n_active, block=block)
    f2_mid = evaluate_f2_device(probe, spectrum)
    err = (jnp.abs(0.5 * (f1_lo + f1_hi) - f1_mid) > tolerance) | (
        jnp.abs(0.5 * (f2_lo + f2_hi) - f2_mid) > tolerance
    )
    bad_raw = ok & err
    rank = jnp.cumsum(bad_raw, dtype=jnp.int32)  # 1-based among failing intervals
    n_raw = rank[-1]
    bad = bad_raw & (rank <= state.points_left)
    n_bad = jnp.sum(bad, dtype=jnp.int32)

    drop = jnp.int32(2 * cap)  # out-of-range target: scatter with mode="drop"
    to_left = jnp.where(bad, rank - 1, drop)
    to_right = jnp.where(bad, n_bad + rank - 1, drop)
    zero = jnp.zeros_like(mid)
    left = jnp.stack([lo, mid, f1_lo, f1_mid, f2_lo, f2_mid, pole_lo, zero])
    right = jnp.stack([mid, hi, f1_mid, f1_hi, f2_mid, f2_hi, zero, pole_hi])
    new_iv = (
        jnp.zeros((8, 2 * cap))
        .at[:, to_left]
        .set(left, mode="drop")
        .at[:, to_right]
        .set(right, mode="drop")
    )
    to_acc = jnp.where(bad, state.n_acc + rank - 1, jnp.int32(state.acc.shape[1]))
    acc = state.acc.at[:, to_acc].set(jnp.stack([mid, f1_mid, f2_mid]), mode="drop")
    new_state = state._replace(
        iv=new_iv,
        n_active=2 * n_bad,
        acc=acc,
        n_acc=state.n_acc + n_bad,
        points_left=state.points_left - n_bad,
    )
    status = _PassStatus(
        n_live=jnp.sum(live, dtype=jnp.int32),
        n_unsplit=jnp.sum(live & ~splittable, dtype=jnp.int32),
        n_checked=jnp.sum(ok, dtype=jnp.int32),
        n_accepted=n_bad,
        hit_max_points=n_raw > state.points_left,
        n_next_active=2 * n_bad,
    )
    return new_state, status


@partial(jax.jit, static_argnames=("active", "accum"))
def _start(
    energy: jax.Array,
    f1: jax.Array,
    f2: jax.Array,
    spectrum: Spectrum,
    knobs: _Knobs,
    *,
    active: int,
    accum: int,
) -> _PassState:
    """Jitted :func:`_start_state` (used when there is nothing to refine)."""
    return _start_state(energy, f1, f2, spectrum, knobs, active=active, accum=accum)


@partial(jax.jit, static_argnames=("block", "width"), donate_argnums=0)
def _refine_pass(
    state: _PassState,
    spectrum: Spectrum,
    rel_corr: Any,
    tolerance: Any,
    *,
    block: int,
    width: int,
) -> tuple[_PassState, _PassStatus]:
    """Jitted :func:`_pass_body` on the first ``width`` active columns.

    The state is donated: its buffers are reused for the output, so a pass
    copies nothing. ``state.iv`` should have ``2 * width`` columns (the
    children capacity of the previous pass, see :func:`_fit_active`) for the
    donation to be usable. One program per ``(width, block)`` and accumulator
    capacity.
    """
    state = state._replace(iv=state.iv[:, :width])
    return _pass_body(state, spectrum, rel_corr, tolerance, block=block)


@partial(jax.jit, static_argnames=("active", "accum", "block"))
def _first_pass(
    energy: jax.Array,
    f1: jax.Array,
    f2: jax.Array,
    spectrum: Spectrum,
    rel_corr: Any,
    knobs: _Knobs,
    *,
    active: int,
    accum: int,
    block: int,
) -> tuple[_PassState, _PassStatus]:
    """:func:`_start_state` fused with the first :func:`_pass_body` (one program)."""
    state = _start_state(energy, f1, f2, spectrum, knobs, active=active, accum=accum)
    return _pass_body(state, spectrum, rel_corr, knobs.tolerance, block=block)


@partial(jax.jit, static_argnames="out_len")
def _finalize(
    energy: jax.Array,
    f1: jax.Array,
    f2: jax.Array,
    knobs: _Knobs,
    state: _PassState,
    *,
    out_len: int,
) -> _RefineArrays:
    """Sorted union of the input grid and the added points, ``inf``-padded.

    Only the leading ``out_len`` entries of each buffer can be valid
    (``out_len >= m + n_added``), which bounds the sort.
    """
    n_g = min(energy.shape[0], out_len)
    n_a = min(state.acc.shape[1], out_len)
    in_grid = jnp.arange(n_g) < knobs.m
    in_acc = jnp.arange(n_a) < state.n_acc
    all_e = jnp.concatenate(
        [
            jnp.where(in_grid, energy[:n_g], jnp.inf),
            jnp.where(in_acc, state.acc[0, :n_a], jnp.inf),
        ]
    )
    all_f1 = jnp.concatenate([f1[:n_g], state.acc[1, :n_a]])
    all_f2 = jnp.concatenate([f2[:n_g], state.acc[2, :n_a]])
    order = jnp.argsort(all_e, stable=True)[:out_len]
    return _RefineArrays(
        energy=all_e[order],
        f1=all_f1[order],
        f2=all_f2[order],
        n_sliver=state.n_sliver,
        n_wide=state.n_wide,
        widest=state.widest,
        widest_ev=state.widest_ev,
    )


@partial(jax.jit, static_argnames="width")
def _fit_columns(a: jax.Array, width: int) -> jax.Array:
    """``a`` cut or zero-padded to ``width`` columns (jitted: no host constants)."""
    n = a.shape[1]
    return a[:, :width] if n > width else jnp.pad(a, ((0, 0), (0, width - n)))


def _fit_active(state: _PassState, width: int) -> _PassState:
    """``state`` with the active buffer cut or zero-padded to ``width`` columns."""
    if state.iv.shape[1] == width:
        return state
    return state._replace(iv=_fit_columns(state.iv, width))


def _grow_accumulator(state: _PassState, size: int) -> _PassState:
    """``state`` with the added-points buffer zero-padded to ``size`` columns."""
    return state._replace(acc=_fit_columns(state.acc, size))


def _accumulator_need(added: int, n_next: int, max_points: int) -> int:
    """Most points the accumulator can hold after a pass over ``n_next`` intervals.

    A pass adds at most one point per interval and at most what is left of
    ``max_points``.
    """
    return added + min(n_next, max_points - added)


def _refine_device(
    energy: jax.Array,
    f1: jax.Array,
    f2: jax.Array,
    spectrum: Spectrum,
    rel_corr: Any,
    knobs: _Knobs,
    *,
    m: int,
    max_passes: int,
    max_points: int,
    chunk_size: int | None = None,
    min_active: int = _MIN_ACTIVE,
    accum_hint: int | None = None,
) -> tuple[_RefineArrays, _RefineInfo]:
    """Refinement on device-resident inputs, with no upload and no final download.

    The host loop reads one :class:`_PassStatus` per pass (a counted
    :func:`~xraylabtool.kk._device.fetch_status`), logs the same per-pass
    DEBUG line as before and re-buckets the active buffer between passes. It
    makes no host-to-device transfer, so it may run under
    ``jax.transfer_guard("disallow_explicit")``. The returned arrays stay on
    the device; fetch them once and finish with :func:`_outcome`.

    Parameters
    ----------
    energy, f1, f2 : jax.Array, shape (L,)
        The input grid, strictly increasing in its first ``m`` entries
        (``L >= max(m, 2)``; the rest is ignored).
    spectrum : Spectrum
        Padded f2 table.
    rel_corr : float or jax.Array
        Relativistic correction Z*.
    knobs : _Knobs
        Device copies of the tolerance, ``max_points`` and ``m``.
    m, max_passes, max_points, chunk_size
        Host copies of the knobs and the arguments of :func:`refine_spectrum`
        (already validated); the host needs them to size buffers and decide.
    min_active : int, default 2048
        Smallest active-buffer capacity (a power of two).
    accum_hint : int, optional
        Starting capacity of the added-points buffer, default
        ``min(max_points, 4 * (m - 1))``. The buffer is never smaller than
        the first pass needs and grows (to the next power of two, from the
        host-known counts) before any pass that could overflow it, so
        ``max_points`` never sizes an allocation. Mainly a test knob.

    Returns
    -------
    arrays : _RefineArrays
        Device result.
    info : _RefineInfo
        Host-known pass count, points added and exhaustion reason.

    References
    ----------
    Design spec §7 (``2026-10-25-kk-jax-first-design.md``).
    """
    n_next = m - 1
    hint = min(max_points, 4 * n_next) if accum_hint is None else accum_hint
    accum = accum_start = bucket_pow2(
        max(hint, _accumulator_need(0, n_next, max_points))
    )
    capacity = spectrum.coeffs.shape[0]
    state: _PassState | None = None
    added = passes = 0
    reason = ""
    while n_next and passes < max_passes:
        passes += 1
        width = max(min_active, bucket_pow2(n_next))
        block = block_for(chunk_size, capacity, width)
        if state is None:
            state, device_status = _first_pass(
                energy,
                f1,
                f2,
                spectrum,
                rel_corr,
                knobs,
                active=width,
                accum=accum,
                block=block,
            )
        else:
            need = _accumulator_need(added, n_next, max_points)
            if need > accum:
                accum = bucket_pow2(need)
                state = _grow_accumulator(state, accum)
            state, device_status = _refine_pass(
                _fit_active(state, 2 * width),
                spectrum,
                rel_corr,
                knobs.tolerance,
                block=block,
                width=width,
            )
        status = _device.fetch_status(device_status)
        n_next = 0
        if not status.n_live:
            break
        if status.n_unsplit:
            reason = reason or "float spacing"
        if not status.n_checked:
            break
        logger.debug(
            "refine_spectrum: pass %d added %d points (of %d intervals checked)",
            passes,
            status.n_accepted,
            status.n_checked,
        )
        added += int(status.n_accepted)
        if status.hit_max_points:
            reason = f"max_points={max_points}"
            break
        n_next = int(status.n_next_active)
    if not reason and n_next and passes >= max_passes:
        reason = f"max_passes={max_passes}"
    info = _RefineInfo(m, passes, added, reason, accum_start, accum)
    if state is None:  # a single point: nothing was refined
        state = _start(energy, f1, f2, spectrum, knobs, active=min_active, accum=accum)
    arrays = _finalize(energy, f1, f2, knobs, state, out_len=bucket_pow2(info.n_total))
    return arrays, info


def _log_slivers(
    count: int, n_wide: int, widest_energy: float, widest: float, tolerance: float
) -> None:
    """Report the pole slivers left unrefined (INFO; WARNING if any is wide)."""
    if not count:
        return
    log = logger.warning if n_wide else logger.info
    log(
        "refine_spectrum: %d interval(s) next to a tabulated range end left "
        "unrefined (f1 is logarithmically singular at the range end; widest "
        "%.3g eV at %.6g eV)%s",
        count,
        widest,
        widest_energy,
        f"; {n_wide} wider than the tabulated end segment, so the tolerance "
        f"{tolerance:g} is materially unmet there"
        if n_wide
        else "",
    )


def _outcome(
    arrays: _RefineArrays, info: _RefineInfo, tolerance: float
) -> RefineOutcome:
    """Host :class:`RefineOutcome` from a *fetched* device result; logs the totals.

    ``arrays`` holds NumPy arrays (the single final download). Only the
    ``info.n_total`` valid leading entries are kept, so no padding can reach
    the result.
    """
    n = info.n_total
    energy = np.asarray(arrays.energy, dtype=np.float64)[:n]
    n_skipped = int(arrays.n_sliver)
    _log_slivers(
        n_skipped,
        int(arrays.n_wide),
        float(arrays.widest_ev),
        float(arrays.widest),
        tolerance,
    )
    logger.info(
        "refine_spectrum: added %d points in %d passes (tolerance %g)",
        info.added,
        info.passes,
        tolerance,
    )
    if info.reason:
        logger.warning(
            "refine_spectrum exhausted (%s) with %d points added; "
            "the tolerance %g is not met everywhere",
            info.reason,
            info.added,
            tolerance,
        )
    return RefineOutcome(
        energy_ev=energy,
        f1=np.asarray(arrays.f1, dtype=np.float64)[:n],
        f2=np.asarray(arrays.f2, dtype=np.float64)[:n],
        n_added=info.added,
        exhausted=bool(info.reason),
        n_pole_skipped=n_skipped,
    )


def refine_spectrum(
    energy_ev: ArrayLike,
    f1: ArrayLike,
    f2: ArrayLike,
    seg_ev: ArrayLike,
    coeffs: ArrayLike,
    rel_corr: float,
    *,
    tolerance: float,
    max_passes: int = 50,
    max_points: int = 100_000,
    chunk_size: int | None = None,
) -> RefineOutcome:
    """Insert midpoints where linear interpolation of f1 or f2 is inaccurate.

    Each interval between neighbouring energies is bisected. The midpoint's
    f1 (Kramers-Kronig transform) and f2 (tabulated polynomial) are computed
    and compared with the average of the interval's end values; if either
    differs by more than ``tolerance`` the midpoint is kept and both halves
    are checked in the next pass. Intervals that pass are final. Only
    midpoints are tested, so a feature narrower than an interval can still be
    missed. Input points are never dropped or changed.

    Parameters
    ----------
    energy_ev : array_like, shape (m,)
        Strictly increasing, unique, positive energies in eV inside
        ``[seg_ev[0], seg_ev[-1]]``.
    f1 : array_like, shape (m,)
        Real part at ``energy_ev``, electrons per formula unit.
    f2 : array_like, shape (m,)
        Imaginary part at ``energy_ev``, electrons per formula unit.
    seg_ev : array_like, shape (n + 1,)
        Segment edges of the spectrum in eV, as for
        :func:`~xraylabtool.kk.transform.kk_real_spectrum`.
    coeffs : array_like, shape (n, 5)
        Polynomial coefficients per segment.
    rel_corr : float
        Relativistic correction Z* (see
        :func:`~xraylabtool.kk.transform.relativistic_correction`).
    tolerance : float
        Largest accepted interpolation error, absolute, in electrons per
        formula unit; finite and positive.
    max_passes : int, default 50
        Maximum number of bisection passes (an interval can be halved at most
        this many times); at least 1.
    max_points : int, default 100000
        Maximum number of points to add; at least 1. When a pass would
        exceed it, only the first ``max_points - added`` failing intervals of
        that pass, in internal order (not necessarily the lowest energies),
        get their midpoints and refinement stops.
    chunk_size : int, optional
        Upper bound on the energies evaluated per block of the midpoint
        kernel (rounded down to a power of two or bucket-ladder value, never
        up); ``None`` leaves only the memory bound of
        :func:`~xraylabtool.kk._shapes.block_for`. At least 1.

    Returns
    -------
    RefineOutcome
        The sorted union of the input energies and the added midpoints with
        f1 and f2, the number of points added and whether a limit stopped
        refinement. Points added per pass are logged at DEBUG, the total at
        INFO and an exhausted limit at WARNING (logger ``xraylabtool.kk``).

    Raises
    ------
    ValidationError
        If an array is empty, not 1-D, of unequal length or not finite,
        ``energy_ev`` is not strictly increasing, positive and inside the
        tabulated range, ``tolerance`` is not finite and positive, or a limit
        is not a positive integer.
    ValueError
        If ``seg_ev``, ``coeffs`` or ``rel_corr`` are invalid, or
        ``chunk_size < 1``.

    Notes
    -----
    An interval whose midpoint equals one of its ends in floating point
    cannot be split further; it is left as is and reported as exhausted.

    The inputs are uploaded once, the bisection passes run on the device
    (host-orchestrated, one small status read per pass) and the sorted union
    is downloaded once. Midpoint f1 may differ from a separate
    :func:`~xraylabtool.kk.transform.kk_real_spectrum` call at the rounding
    level documented there; a midpoint whose error sits within that rounding
    of ``tolerance`` can therefore be accepted or rejected differently.

    f1 diverges logarithmically at a tabulated range end ``seg_ev[0]`` or
    ``seg_ev[-1]`` where f2 is nonzero, so bisecting towards it does not
    converge once ``|f2|`` there exceeds ``tolerance``. For such an end the
    interval touching it is bisected as usual (a failing midpoint is kept),
    but a piece next to the end that is narrower than 2 % of its energy is
    left unrefined. Pieces dropped this way are counted in ``n_pole_skipped``
    and logged: at INFO when each is no wider than the tabulated segment at
    that end (the expected case), at WARNING otherwise. Upstream instead
    never splits the first interval of its grid, which is the same as this
    for the lower end of every element (the first tabulated segment is 1.6 %
    wide) and leaves the upper end to its recursion limit. On the segment
    grid at tolerance 0.05 this gives the identical point set for Au
    (f2 = 0.17 at 500 keV; the chain converges), and for Pb (f2 = 0.20)
    and U (0.36) 44 fewer points (Pb: 461 against 505): upstream's chain
    towards 500 keV does not converge and adds points until its recursion
    limit, here it stops at the 2 % width.

    References
    ----------
    Upstream-only: this is kkcalc's ``improve_accuracy`` (not in B. Watts,
    Opt. Express 22(19), 23628 (2014)).

    Examples
    --------
    >>> import numpy as np
    >>> from xraylabtool.kk import compound_spectrum, evaluate_f2, refine_spectrum
    >>> from xraylabtool.kk import kk_real_spectrum, relativistic_correction
    >>> seg, coeffs = compound_spectrum(["Si"], [1.0])
    >>> rel = relativistic_correction([14], [1.0])
    >>> e = np.array([100.0, 1000.0, 10000.0])
    >>> out = refine_spectrum(
    ...     e,
    ...     kk_real_spectrum(e, seg, coeffs, rel),
    ...     evaluate_f2(e, seg, coeffs),
    ...     seg,
    ...     coeffs,
    ...     rel,
    ...     tolerance=0.1,
    ... )
    >>> bool(out.n_added > 0 and out.energy_ev.size == 3 + out.n_added)
    True
    """
    e, f1_arr, f2_arr, seg = _checked_inputs(
        energy_ev, f1, f2, seg_ev, tolerance, max_passes, max_points
    )
    seg, coeffs_arr = _device.validated_arrays(seg, coeffs, e, rel_corr)
    if chunk_size is not None and (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int | np.integer)
        or chunk_size < 1
    ):
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")
    _device.ensure_x64()
    length = bucket_pow2(e.size)
    spectrum = _device.host_spectrum(seg, coeffs_arr)
    dev: tuple[Any, ...] = _device.upload(
        (
            np.pad(e, (0, length - e.size)),
            np.pad(f1_arr, (0, length - e.size)),
            np.pad(f2_arr, (0, length - e.size)),
            spectrum,
            np.float64(rel_corr),
            _Knobs(
                np.float64(tolerance),  # type: ignore[arg-type]
                np.int64(min(max_points, _INT_MAX)),  # type: ignore[arg-type]
                np.int32(e.size),  # type: ignore[arg-type]
            ),
        )
    )
    arrays, info = _refine_device(
        *dev,
        m=e.size,
        max_passes=max_passes,
        max_points=max_points,
        chunk_size=chunk_size,
    )
    return _outcome(_device.fetch(arrays), info, float(tolerance))
