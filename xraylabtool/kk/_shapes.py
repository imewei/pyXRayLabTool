"""Static-shape buckets for the Kramers-Kronig device kernels.

The ladder, block sizing and element budgets are generic and live in
:mod:`xraylabtool.backend.shapes`; this module re-exports them and keeps
:func:`energy_blocks`, the kk-specific energy tiling.

References
----------
Design: ``docs/superpowers/specs/2026-10-25-kk-jax-first-design.md`` §3.
"""

from __future__ import annotations

from xraylabtool.backend.shapes import (
    ACCELERATOR_EVAL_ELEMENTS,
    EVAL_ELEMENTS,
    _on_cpu,
    block_for,
    bucket,
    bucket_pow2,
)

__all__ = [
    "ACCELERATOR_EVAL_ELEMENTS",
    "EVAL_ELEMENTS",
    "block_for",
    "bucket",
    "bucket_pow2",
    "energy_blocks",
]


def energy_blocks(chunk_size: int | None, capacity: int, m: int) -> tuple[int, int]:
    """Static ``(block, n_blocks)`` for evaluating ``m`` energies.

    The rule depends on the default JAX platform, because a loop trip is cheap
    on a CPU and costs a kernel launch on an accelerator. The kernel evaluates
    only the blocks that hold a real energy, so padding beyond them costs a
    predicate test per block.

    **CPU.** ``m`` is rounded up to a power of two (``m_b``, no minimum); the
    block is :func:`block_for` ``(chunk_size, capacity, m_b)`` with the
    cache-sized budget :data:`EVAL_ELEMENTS`, and ``n_blocks = ⌈m_b / block⌉``.
    The work done is below ``m + block`` energies.

    **Accelerator.** ``m`` is rounded up to the fine :func:`bucket` ladder
    (``padded``, at most 25 % above ``m``). The largest block is
    :func:`block_for` ``(chunk_size, capacity, padded)`` with the budget
    :data:`ACCELERATOR_EVAL_ELEMENTS`; ``n_blocks = ⌈padded / largest⌉`` and the
    blocks are made equal, ``block = ⌈padded / n_blocks⌉`` rounded up to a
    multiple of 8 (and kept at most ``largest``). One block when ``padded``
    fits the budget, and never a second, mostly empty block just above it.
    The padded length is below ``padded + 8 n_blocks`` and the work done below
    ``m + block`` energies.

    On both platforms the number of distinct pairs grows only with ``log₂ m``
    (times the distinct capacities and chunk sizes in use) and ``block`` never
    exceeds ``chunk_size``.

    Parameters
    ----------
    chunk_size : int or None
        Requested maximum block size.
    capacity : int
        Segment capacity of the spectrum.
    m : int
        Number of energies, at least 1.

    Returns
    -------
    block, n_blocks : int
        Static block size and block count.

    References
    ----------
    Design spec §3-§4 (``2026-10-25-kk-jax-first-design.md``).

    Examples
    --------
    >>> from xraylabtool.kk._shapes import energy_blocks
    >>> energy_blocks(7, 1248, 1000)[0]
    4
    """
    if _on_cpu():
        m_b = bucket_pow2(m, minimum=1)
        block = block_for(chunk_size, capacity, m_b, EVAL_ELEMENTS)
        return block, -(-m_b // block)
    padded = bucket(m)
    largest = block_for(chunk_size, capacity, padded, ACCELERATOR_EVAL_ELEMENTS)
    n_blocks = -(-padded // largest)
    equal = -(-padded // n_blocks)
    even = -(-equal // 8) * 8  # equal blocks, a multiple of 8
    return min(largest, even), n_blocks
