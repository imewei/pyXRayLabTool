"""Backend primitives used by the Kramers-Kronig kernel: log, abs, ensure_float64."""

from __future__ import annotations

import numpy as np
import pytest

from xraylabtool.backend.array_ops import (
    JaxBackend,
    ensure_float64,
    get_backend,
    ops,
    set_backend,
)


@pytest.fixture(params=["numpy", "jax"])
def backend_name(request: pytest.FixtureRequest):
    """Run the test under each backend, restoring the original afterwards."""
    if request.param == "jax":
        pytest.importorskip("jax")
    original = "jax" if isinstance(get_backend(), JaxBackend) else "numpy"
    set_backend(request.param)
    yield request.param
    set_backend(original)


@pytest.mark.unit
def test_log_abs_parity_numpy_jax(backend_name: str) -> None:
    mixed = np.array([-3.5, -1e-12, 0.0, 2.25, 1e9])
    positive = np.array([1e-300, 1e-12, 0.5, 1.0, 7.0, 1e300])

    got_abs = np.asarray(ops.abs(mixed))
    got_log = np.asarray(ops.log(positive))

    assert np.isfinite(got_abs).all()
    assert np.isfinite(got_log).all()
    np.testing.assert_allclose(got_abs, np.abs(mixed), rtol=1e-15)
    np.testing.assert_allclose(got_log, np.log(positive), rtol=1e-15)


@pytest.mark.unit
def test_ensure_float64_casts_float32(backend_name: str) -> None:
    out = ensure_float64(np.float32([1, 2]))
    assert out.dtype == np.float64
    np.testing.assert_array_equal(np.asarray(out), [1.0, 2.0])


@pytest.mark.unit
def test_ensure_float64_enables_x64_under_jax() -> None:
    jax = pytest.importorskip("jax")
    original = "jax" if isinstance(get_backend(), JaxBackend) else "numpy"
    set_backend("jax")
    try:
        jax.config.update("jax_enable_x64", False)
        out = ensure_float64(np.float32([1, 2]))
        assert jax.config.jax_enable_x64
        assert out.dtype == np.float64
    finally:
        set_backend(original)
