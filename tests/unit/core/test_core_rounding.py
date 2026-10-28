"""Unit test of ``tests.fixtures.core_rounding.assert_ulp_close``."""

from __future__ import annotations

import numpy as np
import pytest

from tests.fixtures.core_rounding import assert_ulp_close


def _step(x: float, n: int) -> float:
    for _ in range(abs(n)):
        x = float(np.nextafter(x, np.inf if n > 0 else -np.inf))
    return x


@pytest.mark.parametrize("x", [1.0, 3.7e-9, -2.5e12, 6.02e23])
def test_within_and_beyond_the_bound(x):
    assert_ulp_close(_step(x, 4), x, ulps=4)
    assert_ulp_close(_step(x, -4), x, ulps=4)
    with pytest.raises(AssertionError, match="ulp"):
        assert_ulp_close(_step(x, 5), x, ulps=4)
    assert_ulp_close(_step(x, 5), x, ulps=5)


def test_arrays_report_the_worst_entry():
    expected = np.array([1.0, 2.0, 3.0])
    actual = expected.copy()
    actual[1] = _step(2.0, 9)
    with pytest.raises(AssertionError, match=r"9\.0 ulp"):
        assert_ulp_close(actual, expected)


def test_nan_must_match_positions():
    assert_ulp_close([1.0, np.nan], [1.0, np.nan])
    with pytest.raises(AssertionError, match="NaN"):
        assert_ulp_close([1.0, np.nan], [np.nan, 1.0])
    with pytest.raises(AssertionError, match="NaN"):
        assert_ulp_close([1.0, 2.0], [1.0, np.nan])


def test_infinities_must_be_equal_and_never_close_to_finite():
    assert_ulp_close([np.inf, -np.inf, 1.0], [np.inf, -np.inf, 1.0])
    with pytest.raises(AssertionError, match="infinite"):
        assert_ulp_close([np.inf], [-np.inf])
    with pytest.raises(AssertionError, match="infinite"):
        assert_ulp_close([np.inf], [1.7e308])


def test_zero_expected_requires_zero():
    assert_ulp_close([0.0], [0.0])
    assert_ulp_close([-0.0], [0.0])
    with pytest.raises(AssertionError, match="ulp"):
        assert_ulp_close([1e-300], [0.0])


def test_shape_mismatch_and_empty():
    with pytest.raises(AssertionError, match="shape"):
        assert_ulp_close(np.zeros(3), np.zeros(4))
    assert_ulp_close(np.array([]), np.array([]))


def test_golden_zero_floor_applies_only_to_flagged_arrays():
    from tests.characterization.core.test_core_golden import compare_arrays

    expected = np.array([-8.9e-16, 2.0])
    got = np.array([5.3e-15, 2.0])
    assert compare_arrays(got, expected)[0] is not None
    assert compare_arrays(got, expected, zero_floor=True)[0] is None
    assert compare_arrays(np.array([1e-6, 2.0]), expected, zero_floor=True)[0]
    assert compare_arrays(np.array([0.0, 2.0 + 1e-6]), expected, zero_floor=True)[0]
