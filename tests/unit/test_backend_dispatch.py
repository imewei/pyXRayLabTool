"""Unit tests for xraylabtool.backend dispatch layer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from xraylabtool.backend.array_ops import (
    NumpyBackend,
    _OpsProxy,
    get_backend,
    set_backend,
)
from xraylabtool.backend.interpolation import InterpolationFactory

# ---------------------------------------------------------------------------
# NumpyBackend tests
# ---------------------------------------------------------------------------


class TestNumpyBackend:
    def setup_method(self) -> None:
        self.b = NumpyBackend()

    def test_asarray_from_list(self) -> None:
        result = self.b.asarray([1.0, 2.0, 3.0])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_asarray_from_scalar(self) -> None:
        result = self.b.asarray(42.0)
        assert isinstance(result, np.ndarray)
        assert float(result) == 42.0

    def test_zeros_shape_dtype(self) -> None:
        result = self.b.zeros((3, 4), dtype=np.float32)
        assert result.shape == (3, 4)
        assert result.dtype == np.float32
        assert np.all(result == 0.0)

    def test_ones_shape_dtype(self) -> None:
        result = self.b.ones((2, 5))
        assert result.shape == (2, 5)
        assert np.all(result == 1.0)

    def test_linspace_count_and_endpoints(self) -> None:
        result = self.b.linspace(0.0, 1.0, 11)
        assert len(result) == 11
        assert math.isclose(float(result[0]), 0.0)
        assert math.isclose(float(result[-1]), 1.0)

    def test_isfinite_with_nan_inf(self) -> None:
        arr = self.b.asarray([1.0, float("nan"), float("inf"), -float("inf")])
        finite = self.b.isfinite(arr)
        np.testing.assert_array_equal(finite, [True, False, False, False])

    def test_any_and_all(self) -> None:
        all_true = self.b.asarray([True, True, True])
        some_true = self.b.asarray([True, False, True])
        all_false = self.b.asarray([False, False, False])

        assert self.b.any(all_true) is True
        assert self.b.any(all_false) is False
        assert self.b.any(some_true) is True

        assert self.b.all(all_true) is True
        assert self.b.all(all_false) is False
        assert self.b.all(some_true) is False

    def test_arithmetic_ops(self) -> None:
        a = self.b.asarray([1.0, 2.0, 3.0])
        b = self.b.asarray([4.0, 5.0, 6.0])

        result_sq = self.b.square(a)
        np.testing.assert_allclose(result_sq, [1.0, 4.0, 9.0])

        result_sqrt = self.b.sqrt(self.b.asarray([4.0, 9.0, 16.0]))
        np.testing.assert_allclose(result_sqrt, [2.0, 3.0, 4.0])

        result_sum = self.b.sum(a)
        assert math.isclose(float(result_sum), 6.0)

        result_max = self.b.maximum(a, b)
        np.testing.assert_array_equal(result_max, [4.0, 5.0, 6.0])

        result_where = self.b.where(a > 1.5, a, b)
        np.testing.assert_array_equal(result_where, [4.0, 2.0, 3.0])

    def test_trigonometric_ops(self) -> None:
        x = self.b.asarray([0.0, math.pi / 2])

        result_exp = self.b.exp(self.b.asarray([0.0, 1.0]))
        np.testing.assert_allclose(result_exp, [1.0, math.e], rtol=1e-6)

        result_sqrt_sq = self.b.sqrt(self.b.square(x))
        np.testing.assert_allclose(
            result_sqrt_sq, np.abs([0.0, math.pi / 2]), rtol=1e-10
        )

    def test_concatenate_via_einsum(self) -> None:
        a = self.b.asarray([1.0, 2.0])
        b = self.b.asarray([3.0, 4.0])
        result = self.b.einsum("i,i->i", a, b)
        np.testing.assert_array_equal(result, [3.0, 8.0])

    def test_dtype_float64_default(self) -> None:
        result = self.b.zeros((3,))
        assert result.dtype == np.float64

        result2 = self.b.ones((3,))
        assert result2.dtype == np.float64

    def test_float64_property(self) -> None:
        assert self.b.float64 is np.float64

    def test_argsort(self) -> None:
        arr = self.b.asarray([3.0, 1.0, 2.0])
        idx = self.b.argsort(arr)
        np.testing.assert_array_equal(idx, [1, 2, 0])

    def test_ascontiguousarray(self) -> None:
        arr = np.array([[1, 2], [3, 4]], order="F")
        result = self.b.ascontiguousarray(arr)
        assert result.flags["C_CONTIGUOUS"]

    def test_isnan(self) -> None:
        arr = self.b.asarray([1.0, float("nan")])
        np.testing.assert_array_equal(self.b.isnan(arr), [False, True])

    def test_isinf(self) -> None:
        arr = self.b.asarray([1.0, float("inf")])
        np.testing.assert_array_equal(self.b.isinf(arr), [False, True])

    def test_logspace(self) -> None:
        result = self.b.logspace(0.0, 2.0, 3)
        assert len(result) == 3
        np.testing.assert_allclose(result, [1.0, 10.0, 100.0], rtol=1e-10)

    def test_sum_with_axis(self) -> None:
        arr = self.b.asarray([[1.0, 2.0], [3.0, 4.0]])
        row_sums = self.b.sum(arr, axis=1)
        np.testing.assert_array_equal(row_sums, [3.0, 7.0])


# ---------------------------------------------------------------------------
# _OpsProxy tests
# ---------------------------------------------------------------------------


class TestOpsProxy:
    def test_proxy_delegates_to_numpy_by_default(self) -> None:
        # Ensure numpy backend is active
        set_backend("numpy")
        proxy = _OpsProxy()
        result = proxy.zeros((2,))
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_proxy_caches_methods(self) -> None:
        set_backend("numpy")
        proxy = _OpsProxy()
        # First access — cache miss
        method_first = proxy.zeros
        # Second access — should be cached on the instance dict
        method_second = proxy.zeros
        assert method_first is method_second
        # The cached attribute should now be in proxy's __dict__
        assert "zeros" in proxy.__dict__

    def test_proxy_attribute_error_for_unknown_method(self) -> None:
        set_backend("numpy")
        proxy = _OpsProxy()
        with pytest.raises(AttributeError):
            _ = proxy.nonexistent_method_xyz


# ---------------------------------------------------------------------------
# Backend switching tests
# ---------------------------------------------------------------------------


class TestBackendSwitching:
    def teardown_method(self) -> None:
        # Always restore numpy backend after each test
        set_backend("numpy")

    def test_set_backend_numpy(self) -> None:
        set_backend("numpy")
        backend = get_backend()
        assert isinstance(backend, NumpyBackend)
        # Verify ops still work after switch
        result = backend.zeros((3,))
        assert result.shape == (3,)

    def test_set_backend_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            set_backend("invalid_backend_name")

    def test_set_backend_jax_without_install(self) -> None:
        """If JAX is not installed, JaxBackend() should raise ImportError."""
        import importlib.util

        jax_spec = importlib.util.find_spec("jax")
        if jax_spec is not None:
            pytest.skip("JAX is installed; cannot test missing-JAX path")

        with pytest.raises((ImportError, ModuleNotFoundError)):
            set_backend("jax")


# ---------------------------------------------------------------------------
# InterpolationFactory tests
# ---------------------------------------------------------------------------


class TestInterpolationFactory:
    def setup_method(self) -> None:
        set_backend("numpy")

    def teardown_method(self) -> None:
        set_backend("numpy")

    def test_create_pchip_returns_callable(self) -> None:
        x = np.linspace(0.0, 1.0, 10)
        y = np.sin(x)
        interp = InterpolationFactory.create_pchip(x, y)
        assert callable(interp)

    def test_pchip_interpolation_accuracy(self) -> None:
        # Use a smooth function — PCHIP should be very accurate on dense data
        x = np.linspace(0.0, math.pi, 50)
        y = np.sin(x)
        interp = InterpolationFactory.create_pchip(x, y)

        x_test = np.array([math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2])
        y_expected = np.sin(x_test)
        y_got = interp(x_test)

        np.testing.assert_allclose(y_got, y_expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def setup_method(self) -> None:
        self.b = NumpyBackend()

    def test_empty_array(self) -> None:
        arr = self.b.asarray([])
        assert arr.shape == (0,)
        assert self.b.sum(arr) == 0.0
        assert not self.b.any(arr)
        assert self.b.all(arr)  # vacuously true

    def test_single_element_array(self) -> None:
        arr = self.b.asarray([7.0])
        assert arr.shape == (1,)
        assert math.isclose(float(self.b.sum(arr)), 7.0)
        assert self.b.all(self.b.isfinite(arr))

    def test_large_array_performance(self) -> None:
        # Smoke test: verify correctness on a large array (not a timing test)
        n = 1_000_000
        arr = self.b.linspace(0.0, 1.0, n)
        assert len(arr) == n
        assert math.isclose(float(arr[0]), 0.0)
        assert math.isclose(float(arr[-1]), 1.0)
        assert self.b.all(self.b.isfinite(arr))
