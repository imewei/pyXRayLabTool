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

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
import threading
from typing import Any

import jax
import numpy as np
from numpy.typing import NDArray

from xraylabtool.kk import _device
from xraylabtool.kk._shapes import bucket

_ASF_PATH = Path(__file__).resolve().parents[1] / "data" / "kkcalc" / "asf.npz"


@dataclass(frozen=True)
class DeviceTables:
    """Packed per-element ASF tables resident on the device (a JAX pytree).

    Rows are indexed by ``Z - 1``. Knots beyond an element's last real knot
    ``x`` are ``x·(1 + k)``, ``k = 1, 2, …`` (finite and strictly increasing,
    so a ``searchsorted`` over a whole row is well defined) and the matching
    coefficient rows are exactly zero.

    Attributes
    ----------
    knots : jax.Array, shape (92, Kmax)
        Segment edges in eV, padded as described above.
    coeffs : jax.Array, shape (92, Kmax - 1, 5)
        ``a₁, a₀, a₋₁, a₋₂, a₋₃`` per segment, zero past the true length.
    n_knots : jax.Array, shape (92,)
        True knot count of each element (int32).

    References
    ----------
    Design spec §1 and §5 (``2026-10-25-kk-jax-first-design.md``); the
    tabulated data are upstream kkcalc's, B. Watts, Opt. Express 22(19),
    23628 (2014), Eq. (8).
    """

    knots: jax.Array
    coeffs: jax.Array
    n_knots: jax.Array


def _flatten_tables(t: DeviceTables) -> tuple[tuple[Any, Any, Any], None]:
    return (t.knots, t.coeffs, t.n_knots), None


def _unflatten_tables(_aux: None, children: tuple[Any, Any, Any]) -> DeviceTables:
    return DeviceTables(*children)


jax.tree_util.register_pytree_node(DeviceTables, _flatten_tables, _unflatten_tables)

# Built lazily, once per (database, default device); the lock makes concurrent
# first calls single-flight.
_tables_cache: dict[tuple[int, Any], DeviceTables] = {}
_tables_lock = threading.Lock()


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

    def compound_capacity(self, z: Sequence[int]) -> int:
        """Conservative segment capacity for a compound of the elements ``z``.

        The union of the elemental knot sets cannot hold more than
        ``Σ (nᵢ - 1)`` segments, so this bucketed sum is an upper bound on the
        true segment count and overflow is excluded by construction.

        Parameters
        ----------
        z : sequence of int
            Atomic numbers (1..92); repeats are allowed.

        Returns
        -------
        int
            ``bucket(Σ (n_knots - 1))``.

        Raises
        ------
        ValueError
            If an atomic number is outside 1..92.

        References
        ----------
        Design spec §3 (``2026-10-25-kk-jax-first-design.md``).

        Examples
        --------
        >>> from xraylabtool.kk import load_asf
        >>> load_asf().compound_capacity([14, 8]) >= 512
        True
        """
        total = 0
        for zi in z:
            _check_z(zi)
            total += int(self._energy_offsets[zi] - self._energy_offsets[zi - 1]) - 1
        return bucket(total)

    def device_tables(self) -> DeviceTables:
        """Packed ASF tables on the default device, built once and cached.

        The host packing is I/O preparation; the arrays are uploaded in one
        :func:`~xraylabtool.kk._device.upload`. Construction is single-flight
        under a lock and the cache is keyed on the default device.

        Returns
        -------
        DeviceTables
            Knots, coefficients and true knot counts of Z = 1..92.

        References
        ----------
        Design spec §1 and §5 (``2026-10-25-kk-jax-first-design.md``).

        Examples
        --------
        >>> from xraylabtool.kk import load_asf
        >>> load_asf().device_tables().knots.shape[0]
        92
        """
        _device.ensure_x64()
        device = jax.config.jax_default_device or jax.devices()[0]
        key = (id(self), device)
        with _tables_lock:
            tables = _tables_cache.get(key)
            if tables is None:
                tables = _device.upload(self._pack_host())
                _tables_cache[key] = tables
            return tables

    def _pack_host(self) -> DeviceTables:
        """NumPy-leaved :class:`DeviceTables` following the padding rule."""
        sizes = np.diff(self._energy_offsets)
        kmax = int(sizes.max())
        knots = np.empty((92, kmax))
        coeffs = np.zeros((92, kmax - 1, 5))
        for z in range(1, 93):
            row = self.energy_ev(z)
            n = row.size
            knots[z - 1, :n] = row
            knots[z - 1, n:] = row[-1] * (1.0 + np.arange(1, kmax - n + 1))
            coeffs[z - 1, : n - 1] = self.im_coeffs(z)
        return DeviceTables(knots, coeffs, sizes.astype(np.int32))  # type: ignore[arg-type]


def _check_z(z: int) -> None:
    if isinstance(z, bool) or not isinstance(z, int | np.integer) or not 1 <= z <= 92:
        raise ValueError(f"atomic number must be an integer in 1..92, got {z}")


_load_lock = threading.Lock()


def load_asf() -> AsfDatabase:
    """Load the vendored ASF database (cached after the first call).

    Single-flight: concurrent first callers share one database object (the
    device-table cache is keyed on its identity, so two objects would upload
    the tables twice).

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
    with _load_lock:
        return _load_asf_cached()


@cache
def _load_asf_cached() -> AsfDatabase:
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
