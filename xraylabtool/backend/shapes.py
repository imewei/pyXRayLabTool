"""Static-shape buckets for the JAX device programs.

Compiled JAX programs specialise on array shapes, so host wrappers round every
true size up to one of a few bucket sizes and carry the true count alongside.
The ladder is a deterministic recurrence (nothing is hard-coded): ``b₀`` is the
minimum and ``b_{k+1} = 8·⌊1.25·b_k / 8⌋``, so every bucket is a multiple of 8
and, above the minimum, padding wastes at most 25 %. The core device programs
pad the energy axis with :func:`bucket_energy` instead: powers of two up to 4096
(a compile costs far more than the padded work on a small axis), the same
ladder above.

Importing this module has no side effect.

References
----------
Design: ``docs/superpowers/specs/2026-10-25-kk-jax-first-design.md`` §3.
"""

from __future__ import annotations

import jax

__all__ = [
    "ACCELERATOR_EVAL_ELEMENTS",
    "CHUNK_ELEMENTS",
    "EVAL_ELEMENTS",
    "block_for",
    "bucket",
    "bucket_energy",
    "bucket_pow2",
]

# Each (block, n_segments) float64 temporary holds at most this many elements
# (32 MiB), so the largest tensor stays far below 256 MB even with the dozen
# temporaries the kernel keeps alive at once.
CHUNK_ELEMENTS = 2**22

# CPU energy evaluation works on blocks of at most this many (block, n_segments)
# elements (2 MiB per float64 temporary): small enough to stay cache-resident,
# which measured fastest, and fine enough that the energies evaluated beyond the
# real count are below one block.
EVAL_ELEMENTS = 2**18

# On an accelerator a loop trip costs a kernel launch, so energies run as one
# block when they fit this budget (per float64 temporary) and in few large
# blocks otherwise.
ACCELERATOR_EVAL_ELEMENTS = 2**22

_MIN_BUCKET_FLOOR = 64

# Above this the energy axis is compute-bound, so padding is worth bounding.
_ENERGY_POW2_LIMIT = 4096


def _next_bucket(b: int) -> int:
    """Next ladder value: ``8·⌊1.25·b / 8⌋`` in integer arithmetic."""
    return 8 * (5 * b // 32)


def bucket(n: int, minimum: int = 512) -> int:
    """Smallest ladder value that is at least ``n``.

    Parameters
    ----------
    n : int
        True size (non-negative).
    minimum : int, default 512
        First rung of the ladder; a multiple of 8 and at least 64 (the
        recurrence stalls at 24 and below; 64 is a conservative floor).

    Returns
    -------
    int
        ``b ≥ n``, a multiple of 8; ``b ≤ 1.25·n`` whenever ``n > minimum``.

    Raises
    ------
    ValueError
        If ``minimum`` is not a multiple of 8 or is below 64, or ``n < 0``.

    References
    ----------
    Design spec §3 (``2026-10-25-kk-jax-first-design.md``).

    Examples
    --------
    >>> from xraylabtool.backend.shapes import bucket
    >>> [bucket(n) for n in (1, 512, 513, 1001)]
    [512, 512, 640, 1248]
    """
    if minimum % 8 or minimum < _MIN_BUCKET_FLOOR:
        raise ValueError(
            f"minimum must be a multiple of 8 and at least {_MIN_BUCKET_FLOOR}, "
            f"got {minimum}"
        )
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    b = minimum
    while b < n:
        b = _next_bucket(b)
    return b


def bucket_energy(n: int) -> int:
    """Energy-axis bucket of the core programs: coarse when small, fine when large.

    A new bucket is a new compile (hundreds of milliseconds), while padding a
    short axis costs microseconds, so up to 4096 the rungs are powers of two
    (at most 7 compiles for the whole range); above, the :func:`bucket` ladder
    keeps the padding below 25 %.

    Parameters
    ----------
    n : int
        True number of energies (non-negative).

    Returns
    -------
    int
        ``b ≥ max(n, 64)``, a power of two up to 4096 and a ladder rung above.

    Raises
    ------
    ValueError
        If ``n < 0``.

    Examples
    --------
    >>> from xraylabtool.backend.shapes import bucket_energy
    >>> [bucket_energy(n) for n in (1, 64, 65, 500, 4096, 4097)]
    [64, 64, 128, 512, 4096, 5120]
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n <= _ENERGY_POW2_LIMIT:
        return bucket_pow2(n, minimum=_MIN_BUCKET_FLOOR)
    return bucket(n, minimum=_ENERGY_POW2_LIMIT)


def bucket_pow2(n: int, minimum: int = 8) -> int:
    """Smallest power of two that is at least ``max(n, minimum)``.

    Parameters
    ----------
    n : int
        True size.
    minimum : int, default 8
        Lower bound on the result.

    Returns
    -------
    int
        A power of two.

    References
    ----------
    Design spec §3 (``2026-10-25-kk-jax-first-design.md``).

    Examples
    --------
    >>> from xraylabtool.backend.shapes import bucket_pow2
    >>> [bucket_pow2(n) for n in (1, 8, 9, 100)]
    [8, 8, 16, 128]
    """
    return 1 << (max(int(n), minimum, 1) - 1).bit_length()


def block_for(
    chunk_size: int | None,
    capacity: int,
    m: int,
    elements: int = CHUNK_ELEMENTS,
) -> int:
    """Energies per block: a ladder or power-of-two value never above the request.

    The upper bound is ``min(chunk_size, elements // capacity, m)`` (at least 1);
    the result is the largest power of two or :func:`bucket` ladder value that
    does not exceed it, so a requested ``chunk_size`` is never exceeded.

    Parameters
    ----------
    chunk_size : int or None
        Requested maximum block size; ``None`` leaves only the memory bound.
    capacity : int
        Segment capacity of the spectrum (columns of each temporary).
    m : int
        Number of energies to evaluate.
    elements : int, default ``CHUNK_ELEMENTS``
        Element budget of one ``(block, capacity)`` temporary.

    Returns
    -------
    int
        Block size, at least 1.

    References
    ----------
    Design spec §3-§4 (``2026-10-25-kk-jax-first-design.md``).

    Examples
    --------
    >>> from xraylabtool.backend.shapes import block_for
    >>> block_for(7, 1228, 1000), block_for(None, 1228, 1)
    (4, 1)
    """
    limit = min(elements // max(capacity, 1), m)
    if chunk_size is not None:
        limit = min(limit, chunk_size)
    limit = max(int(limit), 1)
    best = 1 << (limit.bit_length() - 1)
    rung = 512
    while rung <= limit:
        best = max(best, rung)
        rung = _next_bucket(rung)
    return best


def _on_cpu() -> bool:
    """True when the default JAX backend is the host CPU (a static, host-side choice)."""
    return bool(jax.default_backend() == "cpu")
