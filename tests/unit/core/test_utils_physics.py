"""The ``utils`` physics and signal helpers on the device (spec section 10, plan 4.1).

The Task 0 goldens (``tests/characterization/core``) pin the values; this file
pins what they cannot: the legacy NumPy formulas as an inline oracle over
lengths on both sides of the shape buckets, agreement of the same data at two
pad widths (masked reductions: equal up to summation order, not bitwise), the
compile and transfer budgets, scalar-in/scalar-out behaviour and the import
discipline.
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys

import jax
import numpy as np
import pytest

from tests.fixtures.kk_compiles import record_compiles
from xraylabtool import utils
from xraylabtool.backend import runtime
from xraylabtool.calculators import host

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


@pytest.mark.parametrize(
    ("warm", "lengths"), [(40, (1, 7, 40, 64)), (70, (65, 72, 80))]
)
def test_series_inside_one_bucket_share_a_compiled_program(warm, lengths):
    utils.smooth_data(None, _signal(warm)[1], 5)  # warm
    utils.normalize_intensity(_signal(warm)[1], "area")
    with record_compiles() as names:
        for n in lengths:
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


def test_one_dimensional_only_helpers_reject_other_ranks():
    grid = np.ones((3, 3))
    with pytest.raises(ValueError, match="one-dimensional"):
        utils.normalize_intensity(grid, "area")
    with pytest.raises(ValueError, match="one-dimensional"):
        utils.smooth_data(None, grid)
    with pytest.raises(ValueError, match="one-dimensional"):
        utils.background_subtraction(None, np.ones((3, 3, 3)))


def test_normalize_max_and_standard_are_global_over_any_shape():
    grid = np.arange(12.0).reshape(3, 4) + 1.0
    np.testing.assert_allclose(utils.normalize_intensity(grid), grid / grid.max())
    np.testing.assert_allclose(
        utils.normalize_intensity(grid, "standard"),
        (grid - grid.mean()) / grid.std(),
        rtol=1e-12,
    )
    scalar = utils.normalize_intensity(np.array(2.0))
    assert type(scalar) is np.float64
    assert scalar == 1.0


def test_linear_background_is_column_wise_on_a_2d_array():
    x = np.linspace(0.0, 3.0, 4)
    columns = np.c_[np.arange(4.0), np.arange(4.0) ** 2]
    got = utils.background_subtraction(x, columns)
    assert got.shape == (4, 2)
    for k in range(2):
        np.testing.assert_allclose(
            got[:, k], utils.background_subtraction(x, columns[:, k])
        )


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


def _two_widths(name, series, n, extra=(), **static):
    """The ``_physics`` program on the first ``n`` samples at two pad widths."""
    from xraylabtool import _physics
    from xraylabtool.backend.shapes import bucket

    program = jax.jit(functools.partial(getattr(_physics, name), **static))
    narrow = bucket(n, minimum=64)
    outs = []
    for width in (narrow, bucket(narrow + 1, minimum=64)):
        padded = [np.concatenate([s[:n], np.full(width - n, s[n - 1])]) for s in series]
        got = host.execute(lambda dev: program(*dev), (*padded, np.int32(n), *extra))
        outs.append(got[:n])
    return outs


@pytest.mark.parametrize("n", [20, 50, 150])
@pytest.mark.parametrize(
    ("name", "static"),
    [
        ("smooth_data", {"window_size": 5}),
        ("smooth_data", {"window_size": 500}),
        ("normalize_intensity", {"method": "max"}),
        ("normalize_intensity", {"method": "area"}),
        ("normalize_intensity", {"method": "standard"}),
        ("linear_background", {}),
    ],
)
def test_the_same_data_at_two_pad_widths_agrees_up_to_summation_order(n, name, static):
    """Masked sums reduce a different number of zeros, so XLA may sum in another order.

    The two widths therefore agree to 1e-10 relative, with an absolute floor of
    1e-12 of the result's scale for points that sit on the mean (``standard``
    subtracts two nearly equal numbers there); they are not bit-identical.
    """
    _, y = _signal(n)
    a, b = _two_widths(name, [y], n, **static)
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12 * np.max(np.abs(a)))


def test_the_polynomial_background_at_two_pad_widths_agrees():
    n = 50
    x, y = _signal(n)
    low = int(np.argmin(y))
    fit = np.polyfit([x[0], x[low], x[-1]], [y[0], y[low], y[-1]], 2)
    a, b = _two_widths("polynomial_background", [x, y], n, extra=(fit,))
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12 * np.max(np.abs(a)))
