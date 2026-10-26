"""Shape buckets of the Kramers-Kronig device kernels (design spec §3)."""

from __future__ import annotations

import pytest

from xraylabtool.kk._shapes import block_for, bucket, bucket_pow2


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
