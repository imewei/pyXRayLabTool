"""JaxPchipInterpolator must reproduce scipy's PchipInterpolator."""

from __future__ import annotations

from collections.abc import Iterator
import warnings

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

pytest.importorskip("jax")

from xraylabtool.backend.interpolation import JaxPchipInterpolator


def _scipy(x: np.ndarray, y: np.ndarray) -> PchipInterpolator:
    return PchipInterpolator(x, y, extrapolate=False)


def _query(x: np.ndarray) -> np.ndarray:
    # Nodes themselves plus off-node points, including the exact end points.
    return np.unique(np.concatenate([x, np.linspace(x[0], x[-1], 997)]))


@pytest.mark.unit
class TestMatchesScipy:
    @pytest.mark.parametrize(
        "y",
        [
            np.array(
                [0.0, 0.0, 1.0, 1.0, 1.0, 3.0, 2.0, 2.0, 5.0]
            ),  # flats, sign flips
            np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]),  # monotone
            np.array([3.0, -1.0, 4.0, -1.0, 5.0, -9.0, 2.0, 6.0, -5.0]),  # oscillating
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),  # step at the end
        ],
    )
    def test_nonuniform_grid(self, y: np.ndarray) -> None:
        x = np.array([0.0, 0.2, 0.5, 1.1, 1.2, 2.5, 4.0, 4.1, 9.0])
        q = _query(x)
        np.testing.assert_allclose(
            np.asarray(JaxPchipInterpolator(x, y)(q)),
            _scipy(x, y)(q),
            rtol=1e-12,
            atol=1e-13,
        )

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_short_tables(self, n: int) -> None:
        x = np.linspace(1.0, 3.0, n) ** 2
        y = np.array([2.0, 5.0, 1.0, 4.0])[:n]
        q = _query(x)
        np.testing.assert_allclose(
            np.asarray(JaxPchipInterpolator(x, y)(q)), _scipy(x, y)(q), rtol=1e-12
        )

    @pytest.mark.parametrize("element", ["Au", "Si", "O"])
    def test_cxro_tables(self, element: str) -> None:
        from xraylabtool.calculators.scattering_data import load_scattering_factor_data

        data = load_scattering_factor_data(element)
        x = np.asarray(data["E"].values, dtype=np.float64)
        q = _query(x)
        for column in ("f1", "f2"):
            y = np.asarray(data[column].values, dtype=np.float64)
            np.testing.assert_allclose(
                np.asarray(JaxPchipInterpolator(x, y)(q)),
                _scipy(x, y)(q),
                rtol=1e-11,
                atol=1e-12,
                err_msg=f"{element}.{column}",
            )


@pytest.mark.unit
class TestShapePreservation:
    def test_step_data_does_not_overshoot(self) -> None:
        x = np.arange(5.0)
        y = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        values = np.asarray(JaxPchipInterpolator(x, y)(np.linspace(0.0, 4.0, 401)))
        assert values.min() >= 0.0
        assert values.max() <= 1.0

    def test_interpolates_nodes_exactly(self) -> None:
        x = np.array([1.0, 2.0, 4.0, 7.0])
        y = np.array([3.0, -2.0, 5.0, 0.5])
        np.testing.assert_allclose(np.asarray(JaxPchipInterpolator(x, y)(x)), y)


@pytest.mark.unit
class TestQueryHandling:
    x = np.array([1.0, 2.0, 4.0, 7.0])
    y = np.array([3.0, -2.0, 5.0, 0.5])

    def test_outside_table_is_nan(self) -> None:
        values = np.asarray(JaxPchipInterpolator(self.x, self.y)([0.5, 7.5, np.nan]))
        assert np.isnan(values).all()

    @pytest.mark.parametrize("shape", [(), (3,), (2, 3)])
    def test_preserves_query_shape(self, shape: tuple[int, ...]) -> None:
        q = np.full(shape, 3.0)
        out = np.asarray(JaxPchipInterpolator(self.x, self.y)(q))
        assert out.shape == shape
        np.testing.assert_allclose(out, _scipy(self.x, self.y)(q), rtol=1e-12)

    def test_usable_inside_jit(self) -> None:
        import jax

        interp = JaxPchipInterpolator(self.x, self.y)
        out = jax.jit(lambda q: interp(q) * 2.0)(np.array([1.5, 3.0]))
        np.testing.assert_allclose(
            np.asarray(out), 2.0 * _scipy(self.x, self.y)(np.array([1.5, 3.0]))
        )


@pytest.fixture
def restore_x64() -> Iterator[None]:
    import jax

    original = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", original)


@pytest.mark.unit
@pytest.mark.usefixtures("restore_x64")
class TestPrecision:
    x = np.array([1.0, 1.0 + 1e-8, 1.0 + 2e-8])
    y = np.array([0.0, 1.0, 2.0])
    q = np.array([1.0 + 0.5e-8, 1.0 + 1.5e-8, 1.0 + 2e-8])

    def test_cached_interpolator_after_x64_is_disabled(self) -> None:
        import jax

        interp = JaxPchipInterpolator(self.x, self.y)
        interp(self.q).block_until_ready()
        jax.config.update("jax_enable_x64", False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = interp(self.q)

        assert out.dtype == np.float64
        np.testing.assert_allclose(out, _scipy(self.x, self.y)(self.q), rtol=1e-12)
        assert caught == []
        assert not jax.config.jax_enable_x64

    @pytest.mark.parametrize("construct_inside", [False, True])
    def test_disabled_thread_local_context(self, construct_inside: bool) -> None:
        import jax

        interp = None if construct_inside else JaxPchipInterpolator(self.x, self.y)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with jax.enable_x64(False):
                if interp is None:
                    interp = JaxPchipInterpolator(self.x, self.y)
                out = interp(self.q)
                assert not jax.config.jax_enable_x64

        assert interp.x.dtype == np.float64
        assert interp.y.dtype == np.float64
        assert interp.d.dtype == np.float64
        assert out.dtype == np.float64
        np.testing.assert_allclose(out, _scipy(self.x, self.y)(self.q), rtol=1e-12)
        assert caught == []


@pytest.mark.unit
class TestValidation:
    @pytest.mark.parametrize(
        ("x", "y"),
        [
            (np.array([1.0 + 1j, 2.0 + 1j]), np.array([1.0, 2.0])),
            (np.array([1.0, 2.0]), np.array([1.0 + 1j, 2.0 + 1j])),
            (np.array([1.0 + 0j, 2.0 + 0j]), np.array([1.0, 2.0])),
            (np.array([1.0, 2.0]), np.array([complex(1.0, np.inf), 2.0])),
        ],
    )
    def test_rejects_complex_tables(self, x: np.ndarray, y: np.ndarray) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="real"):
                JaxPchipInterpolator(x, y)

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            ([1.0], [1.0]),  # too short
            ([1.0, 2.0, 3.0], [1.0, 2.0]),  # length mismatch
            ([1.0, 1.0, 2.0], [1.0, 2.0, 3.0]),  # duplicate x
            ([3.0, 2.0, 1.0], [1.0, 2.0, 3.0]),  # decreasing x
            ([1.0, np.nan, 3.0], [1.0, 2.0, 3.0]),  # non-finite x
            ([1.0, 2.0, 3.0], [1.0, np.inf, 3.0]),  # non-finite y
            ([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]),  # not 1-D
        ],
    )
    def test_rejects_invalid_tables(self, x: object, y: object) -> None:
        with pytest.raises(ValueError):
            JaxPchipInterpolator(x, y)
