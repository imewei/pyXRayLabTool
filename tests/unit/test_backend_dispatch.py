"""Tests for the ``xraylabtool.backend`` stubs and the in-tree PCHIP interpolator."""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from xraylabtool import backend
from xraylabtool.backend.interpolation import JaxPchipInterpolator

STUB_MESSAGE = "the NumPy backend was removed; see ADR-006"

_IMPORT_PROBE = """
import sys

import xraylabtool.backend

leaked = [
    name
    for name in (
        "jax",
        "xraylabtool.backend.array_ops",
        "xraylabtool.backend.runtime",
        "xraylabtool.backend.shapes",
        "xraylabtool.backend.interpolation",
    )
    if name in sys.modules
]
assert not leaked, leaked
print("OK")
"""


class TestBackendStubs:
    def test_set_backend_jax_is_accepted_and_returns_none(self) -> None:
        assert backend.set_backend("jax") is None

    def test_get_backend_is_jax(self) -> None:
        assert backend.get_backend() == "jax"

    @pytest.mark.parametrize("name", ["numpy", "NumPy", "invalid_backend_name", ""])
    def test_set_backend_rejects_every_other_name(self, name: str) -> None:
        with pytest.raises(ValueError, match=STUB_MESSAGE):
            backend.set_backend(name)

    def test_a_rejected_name_leaves_the_backend_unchanged(self) -> None:
        with pytest.raises(ValueError, match=STUB_MESSAGE):
            backend.set_backend("numpy")
        assert backend.get_backend() == "jax"

    @pytest.mark.parametrize("removed", ["ops", "InterpolationFactory"])
    def test_removed_names_are_gone(self, removed: str) -> None:
        assert not hasattr(backend, removed)

    def test_importing_the_backend_package_imports_no_jax(self) -> None:
        done = subprocess.run(
            [sys.executable, "-c", _IMPORT_PROBE],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert done.returncode == 0, done.stderr[-2000:]
        assert done.stdout.strip() == "OK"


class TestJaxPchipInterpolator:
    def test_is_callable(self) -> None:
        x = np.linspace(0.0, 1.0, 10)
        assert callable(JaxPchipInterpolator(x, np.sin(x)))

    def test_accuracy_on_a_smooth_function(self) -> None:
        # PCHIP should be very accurate on dense data
        x = np.linspace(0.0, math.pi, 50)
        interp = JaxPchipInterpolator(x, np.sin(x))

        x_test = np.array([math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2])
        np.testing.assert_allclose(
            np.asarray(interp(x_test)), np.sin(x_test), atol=1e-3
        )

    def test_outside_the_table_is_nan(self) -> None:
        x = np.linspace(1.0, 2.0, 5)
        values = np.asarray(JaxPchipInterpolator(x, x**2)(np.array([0.5, 1.5, 2.5])))
        assert np.isnan(values[0])
        assert np.isfinite(values[1])
        assert np.isnan(values[2])
