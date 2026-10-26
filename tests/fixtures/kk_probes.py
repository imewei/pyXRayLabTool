"""Shared probe energies for the pre-migration baseline and every later KK test.

One function builds the evaluation set so the generator
(``scripts/gen_kk_prejax_baseline.py``) and the tests that compare against its
``.npz`` use identical energies by construction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

SEED = 2026
N_LOG_UNIFORM = 200
LOW_EV, HIGH_EV = 10.0, 500_000.0


def probe_energies(seg_ev: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluation energies in eV for a spectrum with segment edges ``seg_ev``.

    In order: 200 log-uniform energies in 10 eV-500 keV (``default_rng(2026)``);
    the first knot; the first three segment midpoints; the last real knot;
    three interior knots at the quarter, half and three-quarter index
    (continuous f2 for every baseline compound); a quarter-way point of two
    segments (at one third and two thirds of the index range); and three
    above-range energies ``x_n·(1 + k)``, k = 1, 2, 3, the pad-knot positions of
    the padding rule (``x_n`` is the last real knot).

    Parameters
    ----------
    seg_ev : array, shape (n + 1,)
        Strictly increasing segment edges in eV, ``n >= 4``.

    Returns
    -------
    numpy.ndarray of float64, shape (213,)
        Positive energies in eV, not sorted.
    """
    seg = np.asarray(seg_ev, dtype=np.float64)
    n = seg.size - 1
    rng = np.random.default_rng(SEED)
    log_uniform = np.exp(rng.uniform(np.log(LOW_EV), np.log(HIGH_EV), N_LOG_UNIFORM))
    midpoints = (seg[:3] + seg[1:4]) / 2
    interior = seg[[n // 4, n // 2, 3 * n // 4]]
    quarter = np.array(
        [seg[j] + (seg[j + 1] - seg[j]) / 4 for j in (n // 3, 2 * n // 3)]
    )
    above = seg[-1] * (1.0 + np.arange(1, 4))
    return np.concatenate(
        [log_uniform, seg[:1], midpoints, seg[-1:], interior, quarter, above]
    )
