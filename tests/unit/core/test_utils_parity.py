"""Public-API parity of the ``utils`` helpers with the pre-JAX NumPy implementation.

``LEGACY`` holds, as literals, what ``xraylabtool/utils.py`` at commit 9b18d70
(the NumPy implementation, since deleted) returned or raised for each case; it
was frozen by running that module once, so this file does not import it. Each
case must reproduce the legacy outcome (values at ``rtol=1e-12``, exception type
and message exact) unless it is listed in ``DEPARTURES``, which freezes the new
outcome of every intentional change (see the CHANGELOG "Changed" entry):

* float32 input is computed in float64 (legacy computed in float32);
* ``smooth_data`` accepts one-dimensional data only (``ValueError``);
* ``normalize_intensity(..., "area")`` on other than 1-D data and mismatched
  polynomial ``x``/``y`` lengths raise ``ValueError`` (legacy ``TypeError`` or a
  NumPy broadcast error / ``IndexError``);
* ``energy_to_wavelength([10.0])`` still raises ``TypeError``, with NumPy's
  conversion message instead of the sequence-multiplication one.
"""

from __future__ import annotations

from math import inf, nan
import warnings

import numpy as np
import pytest

from xraylabtool import utils

y = np.array([3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.0, 3.5])
x = np.linspace(0, 7, 8)
CASES = [
    ("energy_to_wavelength", (10.0,), {}),
    ("energy_to_wavelength", (np.float32(10.0),), {}),
    ("energy_to_wavelength", (10,), {"units": "nm"}),
    ("energy_to_wavelength", (np.array(10.0),), {}),
    ("energy_to_wavelength", (np.array([10.0]),), {}),
    ("energy_to_wavelength", ([10.0],), {}),
    ("energy_to_wavelength", (np.array([8.0, 10.0]),), {}),
    ("energy_to_wavelength", (np.inf,), {}),
    ("energy_to_wavelength", (np.nan,), {}),
    ("energy_to_wavelength", (-1.0,), {}),
    ("energy_to_wavelength", (10.0,), {"units": "pm"}),
    ("energy_to_wavelength", ("10",), {}),
    ("wavelength_to_energy", (1.54,), {}),
    ("wavelength_to_energy", (1.54,), {"units": "m"}),
    ("wavelength_to_energy", (0.0,), {}),
    ("wavelength_to_energy", (1.54,), {"units": "x"}),
    ("wavelength_to_energy", (np.array([1.0, 2.0]),), {}),
    ("wavelength_to_energy", (np.inf,), {}),
    ("bragg_angle", (3.0, 1.54), {}),
    ("bragg_angle", (3.0, 1.54, 2), {}),
    ("bragg_angle", (0.5, 1.54), {}),
    ("bragg_angle", (-3.0, 1.54), {}),
    ("bragg_angle", (3.0, np.nan), {}),
    ("bragg_angle", (np.array([3.0, 4.0]), 1.54), {}),
    ("d_spacing_cubic", (1, 1, 1, 5.43), {}),
    ("d_spacing_cubic", (0, 0, 0, 5.43), {}),
    ("d_spacing_cubic", (1, 1, 1, -5.43), {}),
    ("d_spacing_cubic", (1.5, 1, 1, 5.43), {}),
    ("d_spacing_tetragonal", (1, 0, 1, 4.0, 6.0), {}),
    ("d_spacing_orthorhombic", (1, 1, 1, 4.0, 5.0, 6.0), {}),
    ("q_from_angle", (30.0, 1.54), {}),
    ("q_from_angle", (30.0, 0.0), {}),
    ("q_from_angle", (np.nan, 1.54), {}),
    ("q_from_angle", (np.array([10.0, 20.0]), 1.54), {}),
    ("q_from_angle", (np.array([10.0]), 1.54), {}),
    ("angle_from_q", (2.0, 1.54), {}),
    ("angle_from_q", (100.0, 1.54), {}),
    ("angle_from_q", (np.nan, 1.54), {}),
    ("smooth_data", (x, y), {}),
    ("smooth_data", (x, y), {"window_size": 4}),
    ("smooth_data", (x, y), {"window_size": 8}),
    ("smooth_data", (x, y.astype(np.float32)), {"window_size": 3}),
    ("smooth_data", (x, list(y)), {"window_size": 3}),
    ("smooth_data", (x, np.array([])), {}),
    ("smooth_data", (x, np.array(2.0)), {}),
    ("smooth_data", (x, np.vstack([y, y])), {"window_size": 3}),
    ("smooth_data", (x, np.vstack([y, y])), {"window_size": 30}),
    ("smooth_data", (x, y), {"window_size": 0}),
    ("smooth_data", (x, np.r_[y[:3], np.nan, y[4:]]), {"window_size": 3}),
    ("background_subtraction", (x, y), {}),
    ("background_subtraction", (x, y), {"method": "polynomial"}),
    ("background_subtraction", (x, y), {"method": "bad"}),
    ("background_subtraction", (x, np.array([])), {}),
    ("background_subtraction", (x, np.array([5.0])), {}),
    ("background_subtraction", (x, np.vstack([y, y]).T), {}),
    ("background_subtraction", (x[:5], y), {"method": "polynomial"}),
    ("background_subtraction", (np.r_[x, 8.0], y), {"method": "polynomial"}),
    ("background_subtraction", (x, y.astype(np.float32)), {}),
    ("normalize_intensity", (y,), {}),
    ("normalize_intensity", (y,), {"method": "area"}),
    ("normalize_intensity", (y,), {"method": "standard"}),
    ("normalize_intensity", (y.astype(np.float32),), {}),
    ("normalize_intensity", (list(y),), {}),
    ("normalize_intensity", ([1, 2, 3],), {}),
    ("normalize_intensity", (np.array([]),), {}),
    ("normalize_intensity", (np.array([]),), {"method": "area"}),
    ("normalize_intensity", (np.array(2.0),), {}),
    ("normalize_intensity", (np.vstack([y, y]),), {}),
    ("normalize_intensity", (np.vstack([y, y]),), {"method": "area"}),
    ("normalize_intensity", (np.vstack([y, y]),), {"method": "standard"}),
    ("normalize_intensity", (np.r_[y, np.nan],), {}),
    ("normalize_intensity", (np.zeros(4),), {"method": "standard"}),
    ("normalize_intensity", (y,), {"method": "bad"}),
]


LEGACY = {
    0: ("ok", "float", 1.2398417166827829),
    1: ("ok", "float", 1.2398416996002197),
    2: ("ok", "float", 0.12398417166827828),
    3: ("ok", "float", 1.2398417166827829),
    4: (
        "err",
        "TypeError",
        "only 0-dimensional arrays can be converted to Python scalars",
    ),
    5: ("err", "TypeError", "can't multiply sequence by non-int of type 'float'"),
    6: (
        "err",
        "TypeError",
        "only 0-dimensional arrays can be converted to Python scalars",
    ),
    7: ("ok", "float", 0.0),
    8: ("err", "ValueError", "Energy must be positive"),
    9: ("err", "ValueError", "Energy must be positive"),
    10: ("err", "ValueError", "Units must be 'angstrom', 'nm', or 'm'"),
    11: (
        "err",
        "UFuncTypeError",
        "ufunc 'greater' did not contain a loop with signature matching types (<class "
        "'numpy.dtypes",
    ),
    12: ("ok", "float", 8.050920238199888),
    13: ("ok", "float", 8.050920238199889e-10),
    14: ("err", "ValueError", "Wavelength must be positive"),
    15: ("err", "ValueError", "Units must be 'angstrom', 'nm', or 'm'"),
    16: (
        "err",
        "TypeError",
        "only 0-dimensional arrays can be converted to Python scalars",
    ),
    17: ("ok", "float", 0.0),
    18: ("ok", "float", 14.87236545017045),
    19: ("ok", "float", 30.88611759812826),
    20: ("err", "ValueError", "No diffraction possible for given parameters"),
    21: ("err", "ValueError", "d-spacing must be positive"),
    22: ("ok", "float", nan),
    23: (
        "err",
        "ValueError",
        "The truth value of an array with more than one element is ambiguous. Use "
        "a.any() or a.all(",
    ),
    24: ("ok", "float", 3.135011961699668),
    25: ("err", "ValueError", "Miller indices (0, 0, 0) have no defined d-spacing"),
    26: ("err", "ValueError", "Lattice parameters must be positive"),
    27: ("ok", "float", 2.633936887894576),
    28: ("ok", "float", 3.328201177351375),
    29: ("ok", "float", 2.770542579237661),
    30: ("ok", "float", 2.1119584693589726),
    31: ("ok", "float", inf),
    32: ("ok", "float", nan),
    33: (
        "err",
        "TypeError",
        "only 0-dimensional arrays can be converted to Python scalars",
    ),
    34: (
        "err",
        "TypeError",
        "only 0-dimensional arrays can be converted to Python scalars",
    ),
    35: ("ok", "float", 28.375322970792194),
    36: ("err", "ValueError", "Invalid q value for given wavelength"),
    37: ("ok", "float", nan),
    38: (
        "ok",
        "ndarray",
        "float64",
        (8,),
        [
            2.8000000000000007,
            2.5,
            3.7,
            3.62,
            4.42,
            4.32,
            4.720000000000001,
            3.6200000000000006,
        ],
    ),
    39: (
        "ok",
        "ndarray",
        "float64",
        (8,),
        [2.5, 2.75, 2.375, 3.875, 4.275, 4.525, 5.025, 3.65],
    ),
    40: ("ok", "ndarray", "float64", (8,), [3.7, 3.7, 3.7, 3.7, 3.7, 3.7, 3.7, 3.7]),
    41: (
        "ok",
        "ndarray",
        "float64",
        (8,),
        [
            2.3333333333333335,
            2.6666666666666665,
            2.1666666666666665,
            4.833333333333333,
            4.3666666348775225,
            5.5333333015441895,
            3.699999968210856,
            3.9999999999999996,
        ],
    ),
    42: (
        "ok",
        "ndarray",
        "float64",
        (8,),
        [
            2.3333333333333335,
            2.6666666666666665,
            2.1666666666666665,
            4.833333333333333,
            4.366666666666667,
            5.533333333333333,
            3.6999999999999997,
            3.9999999999999996,
        ],
    ),
    43: ("ok", "ndarray", "float64", (0,), []),
    44: ("err", "TypeError", "len() of unsized object"),
    45: ("ok", "ndarray", "float64", (2,), [3.7, 3.7]),
    46: ("ok", "ndarray", "float64", (2,), [3.7, 3.7]),
    47: ("err", "ValueError", "Window size must be at least 1"),
    48: (
        "ok",
        "ndarray",
        "float64",
        (8,),
        [
            2.3333333333333335,
            2.6666666666666665,
            nan,
            nan,
            nan,
            5.533333333333333,
            3.6999999999999997,
            3.9999999999999996,
        ],
    ),
    49: (
        "ok",
        "ndarray",
        "float64",
        (8,),
        [
            0.0,
            -2.0714285714285716,
            0.8571428571428572,
            -1.7142857142857144,
            5.714285714285714,
            -0.7571428571428571,
            1.5714285714285716,
            0.0,
        ],
    ),
    50: (
        "ok",
        "ndarray",
        "float64",
        (8,),
        [
            -2.220446049250313e-15,
            0.0,
            4.309523809523812,
            2.428571428571431,
            9.85714285714286,
            2.6952380952380977,
            3.642857142857143,
            -4.440892098500626e-16,
        ],
    ),
    51: ("err", "ValueError", "Method must be 'linear' or 'polynomial'"),
    52: ("err", "IndexError", "index 0 is out of bounds for axis 0 with size 0"),
    53: ("ok", "ndarray", "float64", (1,), [0.0]),
    54: (
        "ok",
        "ndarray",
        "float64",
        (8, 2),
        [
            [0.0, 0.0],
            [-2.0714285714285716, -2.0714285714285716],
            [0.8571428571428572, 0.8571428571428572],
            [-1.7142857142857144, -1.7142857142857144],
            [5.714285714285714, 5.714285714285714],
            [-0.7571428571428571, -0.7571428571428571],
            [1.5714285714285716, 1.5714285714285716],
            [0.0, 0.0],
        ],
    ),
    55: (
        "err",
        "ValueError",
        "operands could not be broadcast together with shapes (8,) (5,) ",
    ),
    56: (
        "err",
        "ValueError",
        "operands could not be broadcast together with shapes (8,) (9,) ",
    ),
    57: (
        "ok",
        "ndarray",
        "float32",
        (8,),
        [
            0.0,
            -2.0714285373687744,
            0.8571429252624512,
            -1.7142858505249023,
            5.714285850524902,
            -0.7571430206298828,
            1.5714285373687744,
            0.0,
        ],
    ),
    58: (
        "ok",
        "ndarray",
        "float64",
        (8,),
        [
            0.3333333333333333,
            0.1111111111111111,
            0.4444444444444444,
            0.16666666666666666,
            1.0,
            0.2888888888888889,
            0.5555555555555556,
            0.3888888888888889,
        ],
    ),
    59: (
        "ok",
        "ndarray",
        "float64",
        (8,),
        [
            0.11385199240986717,
            0.03795066413662239,
            0.15180265654648956,
            0.056925996204933584,
            0.3415559772296015,
            0.09867172675521821,
            0.18975332068311193,
            0.13282732447817835,
        ],
    ),
    60: (
        "ok",
        "ndarray",
        "float64",
        (8,),
        [
            -0.29936680624647266,
            -1.1547005383792515,
            0.12830005981991674,
            -0.9408671053460568,
            2.266634390151864,
            -0.4704335526730284,
            0.5559669258863061,
            -0.08553337321327796,
        ],
    ),
    61: (
        "ok",
        "ndarray",
        "float32",
        (8,),
        [
            0.3333333432674408,
            0.1111111119389534,
            0.4444444477558136,
            0.1666666716337204,
            1.0,
            0.2888888716697693,
            0.5555555820465088,
            0.3888888955116272,
        ],
    ),
    62: (
        "ok",
        "ndarray",
        "float64",
        (8,),
        [
            0.3333333333333333,
            0.1111111111111111,
            0.4444444444444444,
            0.16666666666666666,
            1.0,
            0.2888888888888889,
            0.5555555555555556,
            0.3888888888888889,
        ],
    ),
    63: (
        "ok",
        "ndarray",
        "float64",
        (3,),
        [0.3333333333333333, 0.6666666666666666, 1.0],
    ),
    64: (
        "err",
        "ValueError",
        "zero-size array to reduction operation maximum which has no identity",
    ),
    65: ("ok", "ndarray", "float64", (0,), []),
    66: ("ok", "float64", 1.0),
    67: (
        "ok",
        "ndarray",
        "float64",
        (2, 8),
        [
            [
                0.3333333333333333,
                0.1111111111111111,
                0.4444444444444444,
                0.16666666666666666,
                1.0,
                0.2888888888888889,
                0.5555555555555556,
                0.3888888888888889,
            ],
            [
                0.3333333333333333,
                0.1111111111111111,
                0.4444444444444444,
                0.16666666666666666,
                1.0,
                0.2888888888888889,
                0.5555555555555556,
                0.3888888888888889,
            ],
        ],
    ),
    68: (
        "err",
        "TypeError",
        "only 0-dimensional arrays can be converted to Python scalars",
    ),
    69: (
        "ok",
        "ndarray",
        "float64",
        (2, 8),
        [
            [
                -0.29936680624647266,
                -1.1547005383792515,
                0.12830005981991674,
                -0.9408671053460568,
                2.266634390151864,
                -0.4704335526730284,
                0.5559669258863061,
                -0.08553337321327796,
            ],
            [
                -0.29936680624647266,
                -1.1547005383792515,
                0.12830005981991674,
                -0.9408671053460568,
                2.266634390151864,
                -0.4704335526730284,
                0.5559669258863061,
                -0.08553337321327796,
            ],
        ],
    ),
    70: (
        "ok",
        "ndarray",
        "float64",
        (9,),
        [nan, nan, nan, nan, nan, nan, nan, nan, nan],
    ),
    71: ("ok", "ndarray", "float64", (4,), [nan, nan, nan, nan]),
    72: ("err", "ValueError", "Method must be 'max', 'area', or 'standard'"),
}

DEPARTURES = {
    1: ("ok", "float", 1.2398417166827829),
    5: (
        "err",
        "TypeError",
        "only 0-dimensional arrays can be converted to Python scalars",
    ),
    44: ("err", "ValueError", "Signal data must be one-dimensional"),
    45: ("err", "ValueError", "Signal data must be one-dimensional"),
    46: ("err", "ValueError", "Signal data must be one-dimensional"),
    55: ("err", "ValueError", "x and y must have the same length"),
    56: ("err", "ValueError", "x and y must have the same length"),
    57: (
        "ok",
        "ndarray",
        "float64",
        (8,),
        [
            0.0,
            -2.0714285714285716,
            0.8571428571428572,
            -1.7142857142857144,
            5.714285714285714,
            -0.7571429525102888,
            1.5714285714285716,
            0.0,
        ],
    ),
    61: (
        "ok",
        "ndarray",
        "float64",
        (8,),
        [
            0.3333333333333333,
            0.1111111111111111,
            0.4444444444444444,
            0.16666666666666666,
            1.0,
            0.2888888782925076,
            0.5555555555555556,
            0.38888888888888884,
        ],
    ),
    68: ("err", "ValueError", "Signal data must be one-dimensional"),
}


def _outcome(name, args, kw):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # divide/invalid/size-1 float() noise
            r = getattr(utils, name)(*args, **kw)
    except Exception as exc:
        return ("err", type(exc).__name__, str(exc)[:90])
    if isinstance(r, np.ndarray):
        return ("ok", "ndarray", r.dtype.name, r.shape, r.tolist())
    return ("ok", type(r).__name__, float(r))


def _same(got, want):
    if got[0] != want[0] or got[1] != want[1]:
        return False
    if got[0] == "err":
        return got[2] == want[2]
    if got[1] == "ndarray":
        return got[2:4] == want[2:4] and np.allclose(
            got[4], want[4], rtol=1e-12, atol=0.0, equal_nan=True
        )
    return bool(np.isclose(got[2], want[2], rtol=1e-12, atol=0.0, equal_nan=True))


@pytest.mark.parametrize("index", range(len(CASES)))
def test_matches_the_legacy_outcome_or_the_documented_departure(index):
    name, args, kw = CASES[index]
    got = _outcome(name, args, kw)
    want = DEPARTURES.get(index, LEGACY[index])
    assert _same(got, want), (name, kw, got, want)


def test_every_departure_is_a_real_difference():
    """A departure entry must differ from legacy, or it is stale."""
    for index, new in DEPARTURES.items():
        assert not _same(new, LEGACY[index]), index
