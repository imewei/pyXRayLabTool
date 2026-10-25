"""Henke atomic scattering factor (ASF) database vendored from kkcalc.

The tables are piecewise-polynomial fits of the imaginary scattering factor
``f2(E) = a1*E + a0 + a-1/E + a-2/E**2 + a-3/E**3`` per element (``E`` in eV)
over 10 eV - 500 keV, taken unaltered from upstream kkcalc ``ASF.json`` and
repacked into ``data/kkcalc/asf.npz`` by ``scripts/build_kk_data.py``.

References
----------
B. Watts, Opt. Express 22(19), 23628-23639 (2014), doi:10.1364/OE.22.023628
(upstream-only: data loading has no counterpart equation).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_ASF_PATH = Path(__file__).resolve().parents[1] / "data" / "kkcalc" / "asf.npz"


@dataclass(frozen=True)
class AsfDatabase:
    """Per-element piecewise-polynomial ``f2`` tables (energies in eV).

    Attributes
    ----------
    symbols : tuple of str
        Element symbols for Z = 1..92.
    mass : ndarray, shape (92,)
        Atomic masses (g/mol) for Z = 1..92.
    """

    symbols: tuple[str, ...]
    mass: NDArray[np.float64]
    _energy: NDArray[np.float64]
    _energy_offsets: NDArray[np.int64]
    _coeffs: NDArray[np.float64]
    _coeff_offsets: NDArray[np.int64]

    def energy_ev(self, z: int) -> NDArray[np.float64]:
        """Segment knot energies of element ``z`` (1-based).

        Parameters
        ----------
        z : int
            Atomic number, 1..92.

        Returns
        -------
        ndarray, shape (n + 1,)
            Knot energies in eV, strictly increasing from 10 eV to 500 keV.

        Raises
        ------
        ValueError
            If ``z`` is outside 1..92.
        """
        _check_z(z)
        return self._energy[self._energy_offsets[z - 1] : self._energy_offsets[z]]

    def im_coeffs(self, z: int) -> NDArray[np.float64]:
        """Polynomial coefficients of ``f2`` for element ``z`` (1-based).

        Parameters
        ----------
        z : int
            Atomic number, 1..92.

        Returns
        -------
        ndarray, shape (n, 5)
            Columns ``a1, a0, a-1, a-2, a-3`` of ``E**1 ... E**-3`` for the
            segment ``[energy_ev(z)[i], energy_ev(z)[i + 1])``.

        Raises
        ------
        ValueError
            If ``z`` is outside 1..92.
        """
        _check_z(z)
        return self._coeffs[self._coeff_offsets[z - 1] : self._coeff_offsets[z]]


def _check_z(z: int) -> None:
    if not 1 <= z <= 92:
        raise ValueError(f"atomic number must be in 1..92, got {z}")


@cache
def load_asf() -> AsfDatabase:
    """Load the vendored ASF database (cached after the first call).

    Returns
    -------
    AsfDatabase
        Tables for Z = 1..92.

    Examples
    --------
    >>> db = load_asf()
    >>> db.symbols[13]
    'Si'
    >>> float(db.energy_ev(14)[0])
    10.0
    """
    with np.load(_ASF_PATH, allow_pickle=False) as npz:
        return AsfDatabase(
            symbols=tuple(str(s) for s in npz["symbol"]),
            mass=npz["mass"],
            _energy=npz["E"],
            _energy_offsets=npz["E_offsets"],
            _coeffs=npz["Im"],
            _coeff_offsets=npz["Im_offsets"],
        )
