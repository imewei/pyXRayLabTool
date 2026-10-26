"""Device programs of the ``kk_calculate_real`` pipeline.

Two jitted programs bracket the host's single status fetch:

* :func:`_assemble` builds the compound spectrum, converts and merges the
  near-edge data, and reduces everything the host must raise or log (error
  flags, endpoint steps, the true segment count, the default-grid size) into
  one small :class:`AssembleStatus`.
* :func:`_evaluate` takes the output energies (the clipped knots of the
  assembled spectrum on the default grid, or the uploaded ones), and returns
  them with f1 and f2.

Both consume and produce device arrays only; the inputs travel in one
:class:`PipelineInputs` pytree uploaded by the caller. Refinement
(:mod:`xraylabtool.kk.refine`) continues from their outputs on the device.

References
----------
Design spec §1 and §8 (``2026-10-25-kk-jax-first-design.md``).
"""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from xraylabtool.kk._device import Spectrum
from xraylabtool.kk.database import DeviceTables
from xraylabtool.kk.nearedge import MergeStatus, merge_device, to_asf_scale_device
from xraylabtool.kk.spectrum import compound_spectrum_device, evaluate_f2_device
from xraylabtool.kk.transform import _kk_f1_blocks

# A leaf of the upload pytree: a NumPy value on the host, a ``jax.Array`` after
# the upload. The pytree is built on the host and consumed on the device.
Leaf = Any


class NearEdgeInputs(NamedTuple):
    """Near-edge arrays of one call (bucket-padded; ``n`` is the real count)."""

    energy_ev: Leaf
    values: Leaf
    n: Leaf
    low_ev: Leaf
    high_ev: Leaf
    density: Leaf | None  # beta input only
    formula_mass: Leaf | None  # beta input only


class PipelineInputs(NamedTuple):
    """Everything a call uploads, as one pytree (a single ``device_put``).

    ``energies_ev`` is the explicit output grid padded to the block shape and
    ``n_grid`` its real length; both are ``None`` for the default grid.
    ``tolerance`` and ``max_points`` are the refinement knobs (``None``
    without refinement).
    """

    z: Leaf
    counts: Leaf
    n_elements: Leaf
    rel_corr: Leaf
    energies_ev: Leaf | None
    n_grid: Leaf | None
    nearedge: NearEdgeInputs | None
    tolerance: Leaf | None
    max_points: Leaf | None


class AssembleStatus(NamedTuple):
    """What the host reads after assembly and merge (one status fetch).

    ``n_segments`` is the unclamped true count of the (merged) spectrum;
    ``n_grid`` the number of output energies (the default-grid size, or the
    uploaded count); ``n_below`` and ``n_above`` count the knots clipped away
    by the default window. ``converted_nonfinite`` flags a near-edge value that
    overflowed in the conversion to the f2 scale, ``merge`` is the
    :class:`~xraylabtool.kk.nearedge.MergeStatus` and ``used_points_ev`` the
    merge points the splice used.
    """

    n_segments: jax.Array
    n_grid: jax.Array
    n_below: jax.Array
    n_above: jax.Array
    converted_nonfinite: jax.Array | None
    merge: MergeStatus | None
    used_points_ev: jax.Array | None


class Assembled(NamedTuple):
    """Device result of :func:`_assemble`; ``status`` is the part to fetch."""

    spectrum: Spectrum
    scaled_f2: jax.Array | None
    n_grid: jax.Array
    status: AssembleStatus


@partial(
    jax.jit,
    static_argnames=("capacity", "merge_capacity", "input_type", "window_ev"),
)
def _assemble(
    tables: DeviceTables,
    inputs: PipelineInputs,
    *,
    capacity: int,
    merge_capacity: int,
    input_type: str | None,
    window_ev: tuple[float, float],
) -> Assembled:
    """Compound spectrum, near-edge merge and status in one program.

    Parameters
    ----------
    tables : DeviceTables
        Packed ASF tables.
    inputs : PipelineInputs
        The uploaded inputs.
    capacity : int
        Segment capacity of the compound spectrum
        (:meth:`AsfDatabase.compound_capacity
        <xraylabtool.kk.database.AsfDatabase.compound_capacity>`).
    merge_capacity : int
        Capacity of the merged spectrum (only used with near-edge data).
    input_type : str or None
        Kind of near-edge values (static); ``None`` without near-edge data.
    window_ev : tuple of float
        Default output window ``(low, high)`` in electron-volts, used for the clipping
        counts.

    Returns
    -------
    Assembled
        Spectrum at the final capacity, the full normalised near-edge curve
        (``None`` without near-edge data), the output count and the status.

    References
    ----------
    Design spec §1, §5 and §6 (``2026-10-25-kk-jax-first-design.md``).
    """
    spec = compound_spectrum_device(
        inputs.z, inputs.counts, inputs.n_elements, tables, capacity=capacity
    )
    ne = inputs.nearedge
    converted_nonfinite = merge_status = used = scaled = None
    if ne is not None:
        converted = to_asf_scale_device(
            ne.energy_ev,
            ne.values,
            str(input_type),
            density=ne.density,
            formula_mass=ne.formula_mass,
        )
        valid = jnp.arange(converted.shape[0]) < ne.n
        converted_nonfinite = ~jnp.all(jnp.isfinite(converted) | ~valid)
        merged = merge_device(
            spec,
            ne.energy_ev,
            converted,
            ne.n,
            ne.low_ev,
            ne.high_ev,
            capacity=merge_capacity,
        )
        spec, scaled, used, merge_status = (
            merged.spectrum,
            merged.scaled_f2,
            merged.used_points_ev,
            merged.status,
        )
    low, high = window_ev
    real = jnp.arange(spec.seg_ev.shape[0]) <= spec.n_segments
    n_below = jnp.sum(real & (spec.seg_ev < low), dtype=jnp.int32)
    n_above = jnp.sum(real & (spec.seg_ev > high), dtype=jnp.int32)
    n_grid = (
        spec.n_segments + 1 - n_below - n_above
        if inputs.n_grid is None
        else inputs.n_grid
    )
    status = AssembleStatus(
        spec.n_segments,
        n_grid,
        n_below,
        n_above,
        converted_nonfinite,
        merge_status,
        used,
    )
    return Assembled(spec, scaled, n_grid, status)


@partial(jax.jit, static_argnames=("block", "n_blocks", "window_ev"))
def _evaluate(
    spectrum: Spectrum,
    energies_ev: jax.Array | None,
    n_grid: jax.Array,
    rel_corr: jax.Array,
    *,
    block: int,
    n_blocks: int,
    window_ev: tuple[float, float],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Output energies, f1 and f2 of a spectrum in one program.

    Parameters
    ----------
    spectrum : Spectrum
        The assembled (and merged) spectrum.
    energies_ev : jax.Array or None
        Uploaded output grid of length ``block * n_blocks`` (any order,
        repeats allowed; entries past ``n_grid`` are ignored), or ``None``
        for the default grid: the knots of ``spectrum`` inside ``window_ev``,
        gathered bit-exactly and padded with the last one.
    n_grid : jax.Array
        Number of output energies (int32, at least 1).
    rel_corr : jax.Array
        Relativistic correction Z*.
    block, n_blocks : int
        Static block shape of the kernel (:func:`~xraylabtool.kk._shapes.energy_blocks`).
    window_ev : tuple of float
        Default window ``(low, high)`` in electron-volts.

    Returns
    -------
    energy_ev, f1, f2 : jax.Array, shape (block * n_blocks,)
        The grid (padded with an in-range energy) and the two parts at it;
        f1 is exactly 0 past ``n_grid``.

    References
    ----------
    Design spec §4 and §8 (``2026-10-25-kk-jax-first-design.md``).
    """
    length = block * n_blocks
    if energies_ev is None:
        start = jnp.searchsorted(spectrum.seg_ev, window_ev[0], side="left")
        offset = jnp.minimum(jnp.arange(length), jnp.maximum(n_grid, 1) - 1)
        energies_ev = spectrum.seg_ev[start + offset]
    elif energies_ev.shape[0] != length:
        raise ValueError(
            f"energies_ev has length {energies_ev.shape[0]}, expected {length}"
        )
    f1 = _kk_f1_blocks(energies_ev, spectrum, rel_corr, n_grid, block=block)
    return energies_ev, f1, evaluate_f2_device(energies_ev, spectrum)
