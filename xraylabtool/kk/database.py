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

    References
    ----------
    Upstream-only: data handling of kkcalc, not an equation of B. Watts,
    Opt. Express 22(19), 23628 (2014); the table form is its Eq. (8).

    Examples
    --------
    >>> from xraylabtool.kk import load_asf
    >>> db = load_asf()
    >>> db.symbols[0], db.mass.shape
    ('H', (92,))
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
            If ``z`` is not an integer in 1..92.

        References
        ----------
        Upstream-only (kkcalc ``ASF.json`` layout); the knots are the
        ``x_j`` of B. Watts, Opt. Express 22(19), 23628 (2014), Eq. (10).

        Examples
        --------
        >>> from xraylabtool.kk import load_asf
        >>> knots = load_asf().energy_ev(14)
        >>> float(knots[0]), float(knots[-1])
        (10.0, 500000.0)
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
            If ``z`` is not an integer in 1..92.

        References
        ----------
        Upstream-only (kkcalc ``ASF.json`` layout); the columns are the
        coefficients of B. Watts, Opt. Express 22(19), 23628 (2014), Eq. (8).

        Examples
        --------
        >>> from xraylabtool.kk import load_asf
        >>> db = load_asf()
        >>> db.im_coeffs(14).shape == (db.energy_ev(14).size - 1, 5)
        True
        """
        _check_z(z)
        return self._coeffs[self._coeff_offsets[z - 1] : self._coeff_offsets[z]]


def _check_z(z: int) -> None:
    if isinstance(z, bool) or not isinstance(z, int | np.integer) or not 1 <= z <= 92:
        raise ValueError(f"atomic number must be an integer in 1..92, got {z}")


@cache
def load_asf() -> AsfDatabase:
    """Load the vendored ASF database (cached after the first call).

    Returns
    -------
    AsfDatabase
        Tables for Z = 1..92.

    References
    ----------
    Upstream-only: kkcalc's data loading, not an equation of B. Watts,
    Opt. Express 22(19), 23628 (2014).

    Examples
    --------
    >>> from xraylabtool.kk import load_asf
    >>> db = load_asf()
    >>> db.symbols[13]
    'Si'
    >>> float(db.energy_ev(14)[0])
    10.0
    """
    with np.load(_ASF_PATH, allow_pickle=False) as npz:
        arrays = {
            key: npz[key] for key in ("mass", "E", "E_offsets", "Im", "Im_offsets")
        }
        symbols = tuple(str(s) for s in npz["symbol"])
    # The cached tables back every later result: accessor slices are views,
    # so a caller's in-place write must raise instead of corrupting them.
    for array in arrays.values():
        array.setflags(write=False)
    return AsfDatabase(
        symbols=symbols,
        mass=arrays["mass"],
        _energy=arrays["E"],
        _energy_offsets=arrays["E_offsets"],
        _coeffs=arrays["Im"],
        _coeff_offsets=arrays["Im_offsets"],
    )
