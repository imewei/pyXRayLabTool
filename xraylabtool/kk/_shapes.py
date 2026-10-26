"""Static-shape buckets for the Kramers-Kronig device kernels.

Compiled JAX programs specialise on array shapes, so host wrappers round every
true size up to one of a few bucket sizes and carry the true count alongside.
The ladder is a deterministic recurrence (nothing is hard-coded): ``b₀`` is the
minimum and ``b_{k+1} = 8·⌊1.25·b_k / 8⌋``, so every bucket is a multiple of 8
and, above the minimum, padding wastes at most 25 %.

References
----------
Design: ``docs/superpowers/specs/2026-10-25-kk-jax-first-design.md`` §3.
"""

from __future__ import annotations

# Each (block, n_segments) float64 temporary holds at most this many elements
# (32 MiB), so the largest tensor stays far below 256 MB even with the dozen
# temporaries the kernel keeps alive at once.
CHUNK_ELEMENTS = 2**22

# Energy evaluation works on blocks of at most this many (block, n_segments)
# elements (2 MiB per float64 temporary): small enough to stay cache-resident,
# which measured fastest on CPU, and fine enough that the energies evaluated
# beyond the real count are below one block.
EVAL_ELEMENTS = 2**18

_MIN_BUCKET_FLOOR = 64


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
        recurrence stalls below that).

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
    >>> from xraylabtool.kk._shapes import bucket
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
    >>> from xraylabtool.kk._shapes import bucket_pow2
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
    >>> from xraylabtool.kk._shapes import block_for
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


def energy_blocks(chunk_size: int | None, capacity: int, m: int) -> tuple[int, int]:
    """Static ``(block, n_blocks)`` for evaluating ``m`` energies.

    ``m`` is first rounded up to a power of two (``m_b``, no minimum); the
    block is then :func:`block_for` ``(chunk_size, capacity, m_b)`` with the
    cache-sized budget :data:`EVAL_ELEMENTS`, and ``n_blocks = ⌈m_b / block⌉``.
    The padded length ``block * n_blocks`` is below ``2m`` plus at most one
    block and the number of distinct pairs grows only with ``log₂ m`` (times
    the distinct capacities and chunk sizes in use). The kernel evaluates only
    the ``⌈m / block⌉`` blocks that hold a real energy, so the work done is
    below ``m + block`` energies whatever the padded length.

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
    >>> energy_blocks(None, 1248, 1000), energy_blocks(7, 1248, 1000)
    ((128, 8), (4, 256))
    """
    m_b = bucket_pow2(m, minimum=1)
    block = block_for(chunk_size, capacity, m_b, EVAL_ELEMENTS)
    return block, -(-m_b // block)
