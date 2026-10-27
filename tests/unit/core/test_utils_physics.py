"""The ``utils`` physics and signal helpers on the device (spec section 10, plan 4.1).

The Task 0 goldens (``tests/characterization/core``) pin the values; this file
pins what they cannot: the legacy NumPy formulas as an inline oracle over
lengths on both sides of the shape buckets (padding never changes a result),
the compile and transfer budgets, scalar-in/scalar-out behaviour and the
import discipline.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from tests.fixtures.kk_compiles import record_compiles
from xraylabtool import utils
from xraylabtool.backend import runtime

PROGRAMS = {
    "smooth_data",
    "linear_background",
    "polynomial_background",
    "normalize_intensity",
    "wavelength_to_energy",
    "energy_to_wavelength",
    "bragg_angle",
    "q_from_angle",
}
LENGTHS = (1, 2, 3, 17, 63, 64, 65, 200)


def _mine(names: list[str]) -> list[str]:
    return [n for n in names if n in PROGRAMS]


def _signal(n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(n)
    x = np.linspace(10.0, 80.0, n)
    return x, 5.0 + 0.002 * (x - 45.0) ** 2 + np.exp(
        -(((x - 40.0) / 3.0) ** 2)
    ) + 0.05 * rng.random(n)


def _smooth_reference(y: np.ndarray, window: int) -> np.ndarray:
    if window >= y.size:
        return np.full(y.size, np.mean(y))
    padded = np.pad(y, window // 2, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: y.size]


@pytest.fixture(autouse=True)
def _x64() -> None:
    runtime.ensure_x64()


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("window", [1, 2, 5, 10, 100])
def test_smooth_data_matches_the_numpy_formula_at_every_length(n, window):
    _, y = _signal(n)
    np.testing.assert_allclose(
        utils.smooth_data(None, y, window), _smooth_reference(y, window), rtol=1e-12
    )


@pytest.mark.parametrize("n", LENGTHS)
def test_normalize_intensity_matches_the_numpy_formulas(n):
    _, y = _signal(n)
    np.testing.assert_allclose(
        utils.normalize_intensity(y, "max"), y / np.max(y), rtol=1e-12
    )
    if n > 1:
        np.testing.assert_allclose(
            utils.normalize_intensity(y, "area"), y / np.trapezoid(y), rtol=1e-12
        )
        np.testing.assert_allclose(
            utils.normalize_intensity(y, "standard"),
            (y - np.mean(y)) / np.std(y),
            rtol=1e-12,
            atol=1e-12,  # points at the mean cancel; the sums differ by an ulp
        )


@pytest.mark.parametrize("n", LENGTHS)
def test_background_subtraction_matches_the_numpy_formulas(n):
    x, y = _signal(n)
    np.testing.assert_allclose(
        utils.background_subtraction(x, y, "linear"),
        y - np.linspace(y[0], y[-1], n),
        rtol=1e-12,
        atol=1e-12,
    )
    if n >= 3:
        low = int(np.argmin(y))
        fit = np.polyfit([x[0], x[low], x[-1]], [y[0], y[low], y[-1]], 2)
        np.testing.assert_allclose(
            utils.background_subtraction(x, y, "polynomial"),
            y - np.polyval(fit, x),
            rtol=1e-12,
            atol=1e-12,
        )


def test_series_inside_one_bucket_share_a_compiled_program():
    utils.smooth_data(None, _signal(40)[1], 5)  # warm
    utils.normalize_intensity(_signal(40)[1], "area")
    with record_compiles() as names:
        for n in (1, 7, 40, 64):
            utils.smooth_data(None, _signal(n)[1], 5)
            utils.normalize_intensity(_signal(n)[1], "area")
    assert _mine(names) == [], names


def test_a_new_window_is_a_new_program_and_a_repeat_is_not():
    y = _signal(50)[1]
    with record_compiles() as names:
        utils.smooth_data(None, y, 7)
    assert _mine(names) == ["smooth_data"], names
    with record_compiles() as names:
        utils.smooth_data(None, y, 7)
    assert _mine(names) == [], names


def test_a_scalar_conversion_is_one_upload_and_one_fetch():
    utils.energy_to_wavelength(8.0)  # warm
    runtime.reset_transfer_counts()
    out = utils.energy_to_wavelength(8.0)
    assert runtime.transfer_counts() == {"upload": 1, "fetch_status": 0, "fetch": 1}
    assert type(out) is float


def test_a_signal_helper_is_one_upload_and_one_fetch():
    y = _signal(50)[1]
    utils.normalize_intensity(y)  # warm
    runtime.reset_transfer_counts()
    utils.normalize_intensity(y)
    assert runtime.transfer_counts() == {"upload": 1, "fetch_status": 0, "fetch": 1}


@pytest.mark.parametrize(
    "call",
    [
        lambda: utils.wavelength_to_energy(1.5406),
        lambda: utils.energy_to_wavelength(np.float64(8.0), "nm"),
        lambda: utils.bragg_angle(3.1356, 1.5406, 2),
        lambda: utils.d_spacing_cubic(1, 1, 1, 5.431),
        lambda: utils.q_from_angle(30.0, 1.5406),
        lambda: utils.angle_from_q(2.0, 1.5406),
    ],
)
def test_scalar_in_gives_a_python_float(call):
    assert type(call()) is float


def test_arrays_larger_than_one_element_still_fail_in_float():
    with pytest.raises(TypeError):
        utils.energy_to_wavelength(np.array([1.0, 2.0]))


def test_signal_helpers_require_one_dimensional_data():
    with pytest.raises(ValueError, match="one-dimensional"):
        utils.normalize_intensity(np.ones((3, 3)))


def test_importing_utils_does_not_import_jax():
    code = (
        "import sys, numpy\n"
        "from xraylabtool import utils\n"
        "assert utils.parse_formula('SiO2')\n"
        "assert 'jax' not in sys.modules and 'xraylabtool._physics' not in sys.modules\n"
        "utils.energy_to_wavelength(8.0)\n"
        "assert 'jax' in sys.modules\n"
        "print('OK')\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "XRAYLABTOOL_NO_JAX_CACHE": "1"},
        timeout=180,
        check=False,
    )
    assert done.stdout.strip().endswith("OK"), done.stderr[-1500:]
