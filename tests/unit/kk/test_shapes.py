"""Shape buckets of the Kramers-Kronig device kernels (design spec §3)."""

from __future__ import annotations

import pytest

from xraylabtool.kk import _shapes
from xraylabtool.kk._shapes import (
    ACCELERATOR_EVAL_ELEMENTS,
    EVAL_ELEMENTS,
    block_for,
    bucket,
    bucket_pow2,
    energy_blocks,
)


@pytest.mark.unit
def test_bucket_ladder_head() -> None:
    assert bucket(1) == 512
    assert bucket(512) == 512
    assert bucket(513) == 640
    assert [bucket(n) for n in (640, 641, 800, 801, 1000, 1001)] == [
        640,
        800,
        800,
        1000,
        1000,
        1248,
    ]


@pytest.mark.unit
def test_bucket_is_multiple_of_8_monotone_and_bounded() -> None:
    previous = 0
    for n in range(1, 20001):
        b = bucket(n)
        assert b % 8 == 0
        assert b >= n
        assert b >= previous
        if n > 512:
            assert b <= 1.25 * n
        previous = b


@pytest.mark.unit
def test_few_distinct_buckets_cover_realistic_sizes() -> None:
    assert len({bucket(n) for n in range(512, 5001)}) < 14


@pytest.mark.unit
@pytest.mark.parametrize("minimum", [0, 8, 56, 63, 100, 513])
def test_bucket_rejects_bad_minimum(minimum: int) -> None:
    with pytest.raises(ValueError, match="minimum"):
        bucket(10, minimum=minimum)


@pytest.mark.unit
def test_bucket_small_minimum_still_advances() -> None:
    assert [bucket(n, minimum=64) for n in (1, 64, 65, 81, 101)] == [
        64,
        64,
        80,
        96,
        120,
    ]


@pytest.mark.unit
def test_bucket_pow2() -> None:
    assert bucket_pow2(9) == 16
    assert bucket_pow2(1) == 8
    assert bucket_pow2(8) == 8
    assert bucket_pow2(0) == 8
    assert bucket_pow2(5, minimum=2) == 8
    assert bucket_pow2(1025) == 2048


@pytest.mark.unit
def test_block_for_examples() -> None:
    assert block_for(7, 1228, 1000) == 4
    assert block_for(None, 1228, 1) == 1
    assert block_for(1, 1228, 1000) == 1
    # memory bound: 2**22 // 1228 = 3415, ladder value 3024
    assert block_for(None, 1228, 10**6) == 3024


@pytest.mark.unit
def test_block_for_never_exceeds_the_request() -> None:
    for chunk in (None, 1, 2, 3, 7, 100, 511, 512, 513, 5000):
        for capacity in (512, 1228, 10**7):
            for m in (1, 2, 9, 600, 100_000):
                b = block_for(chunk, capacity, m)
                assert 1 <= b <= m
                assert b <= max(1, 2**22 // capacity)
                if chunk is not None:
                    assert b <= chunk


@pytest.fixture(params=["cpu", "accelerator"])
def platform(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run the block rule as on a CPU and as on an accelerator (a host-side switch)."""
    monkeypatch.setattr(_shapes, "_on_cpu", lambda: request.param == "cpu")
    return str(request.param)


def _budget(platform: str) -> int:
    return EVAL_ELEMENTS if platform == "cpu" else ACCELERATOR_EVAL_ELEMENTS


@pytest.mark.unit
@pytest.mark.parametrize("m", [1, 7, 100, 1000, 1116, 3000, 10000, 100000])
@pytest.mark.parametrize("chunk", [None, 7, 512])
def test_energy_blocks_padded_length_is_bounded(
    platform: str, m: int, chunk: int | None
) -> None:
    block, n_blocks = energy_blocks(chunk, 1248, m)
    padded = block * n_blocks
    assert padded >= m
    # CPU: power-of-two m_b < 2m; accelerator: fine ladder <= max(512, 1.25 m)
    bound = 2 * m if platform == "cpu" else max(512, 1.25 * m)
    assert padded < bound + block
    assert chunk is None or block <= chunk
    assert block * 1248 <= max(_budget(platform), 1248)
    if chunk == 7 and m >= 7:
        assert block == 4


@pytest.mark.unit
@pytest.mark.parametrize("capacity", [512, 1248, 3000])
@pytest.mark.parametrize("chunk", [None, 7, 512])
def test_energy_blocks_live_blocks_fit_and_cpu_padding_is_bounded(
    platform: str, capacity: int, chunk: int | None
) -> None:
    """Live blocks fit; on CPU the padded length is under 2 m plus one block (m_b < 2 m)."""
    for m in range(1, 20001, 3):
        block, n_blocks = energy_blocks(chunk, capacity, m)
        live = -(-m // block)
        assert live <= n_blocks
        if platform == "cpu":  # the accelerator rule pads to few shapes by design
            assert n_blocks * block < 2 * m + block
        assert block * capacity <= max(_budget(platform), capacity)


@pytest.mark.unit
def test_energy_blocks_cpu_examples(platform: str) -> None:
    if platform != "cpu":
        pytest.skip("CPU rule")
    assert energy_blocks(None, 1248, 1116) == (128, 16)
    assert energy_blocks(7, 1248, 1000) == (4, 256)


@pytest.mark.unit
def test_energy_blocks_accelerator_runs_one_fine_block_when_it_fits(
    platform: str,
) -> None:
    if platform == "cpu":
        pytest.skip("accelerator rule")
    assert energy_blocks(None, 1248, 1000) == (1000, 1)
    assert energy_blocks(None, 1248, 3000) == (3024, 1)
    # beyond the 2**22-element budget: blocks of the budget size, 34 live of 36
    assert energy_blocks(None, 1248, 100_000) == (3024, 36)
    # a requested chunk_size is an upper bound on the block
    assert energy_blocks(100, 1248, 1000)[0] <= 100


@pytest.mark.unit
def test_energy_blocks_have_few_distinct_shapes(platform: str) -> None:
    pairs = {energy_blocks(None, 1248, m) for m in range(1, 20001)}
    capped = {energy_blocks(7, 1248, m) for m in range(1, 20001)}
    if platform == "cpu":
        assert len(pairs) == 16  # one per power of two m_b = 1 .. 32768
        assert len(capped) == 16
    else:
        assert len(pairs) == 15  # one per fine-ladder value
        assert len(capped) == 18


def test_shape_helpers_accept_numpy_integers() -> None:
    import numpy as np

    assert bucket_pow2(np.int64(100)) == bucket_pow2(100)
    assert block_for(np.int64(7), np.int64(1228), np.int64(1000)) == block_for(
        7, 1228, 1000
    )
