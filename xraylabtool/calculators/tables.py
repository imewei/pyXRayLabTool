"""Resident, device-side scattering-factor tables.

``xraylabtool/data/scattering_tables.npz`` (built by
``scripts/gen_scattering_tables.py`` from the ``.nff`` files) is read once,
uploaded once and cached for the process. The tables are a program *argument*,
never a closed-over constant.

This module imports ``jax`` at module level, so callers import it inside
functions; importing it has no side effect (no ``ensure_x64()``).

References
----------
Design: ``docs/superpowers/specs/2026-10-26-package-jax-first-design.md`` §2.
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path
import threading
from typing import Any

import jax
import numpy as np

from xraylabtool.backend.runtime import upload
from xraylabtool.exceptions import UnknownElementError

__all__ = ["ScatteringTables", "clear_tables", "element_index", "load_tables"]

_NPZ = Path(__file__).resolve().parent.parent / "data" / "scattering_tables.npz"


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=["x", "f1", "f2", "d1", "d2", "n", "coef"],
    meta_fields=["symbols"],
)
@dataclasses.dataclass(frozen=True, eq=False)
class ScatteringTables:
    """Padded PCHIP tables for the 92 elements, in atomic-number order.

    Attributes
    ----------
    x, f1, f2, d1, d2 : Array, float64, shape ``[Z, N]``
        Energy knots (eV), f1, f2 and their PCHIP node slopes. Past ``n[z]`` a
        row holds strictly increasing dummy knots, the last value and zero
        slopes, so a padded row evaluates like the unpadded one on
        ``[x[0], x[n-1]]``.
    n : Array, int32, shape ``[Z]``
        Real knots per row.
    coef : Array, float64, shape ``[Z, N-1, 9]``
        Per interval ``i`` of each row: ``x[i]`` and the power-basis cubic
        ``c0 + c1 t + c2 t² + c3 t³`` (``t = e - x[i]``) of f1 (entries 1-4) and
        f2 (entries 5-8), derived from the knots, values and slopes above so one
        gather fetches everything an energy needs. The device programs read only
        ``x``, ``coef`` and ``n``; ``f1``, ``f2``, ``d1`` and ``d2`` stay as the
        source of ``coef`` and for the ``TableInterpolator`` shims, so editing
        them without rebuilding ``coef`` changes nothing on the device.
    symbols : tuple of str
        Element symbols (host metadata, not a pytree leaf).
    """

    x: Any
    f1: Any
    f2: Any
    d1: Any
    d2: Any
    n: Any
    coef: Any
    symbols: tuple[str, ...]


def _coefficients(
    x: np.ndarray, f1: np.ndarray, f2: np.ndarray, d1: np.ndarray, d2: np.ndarray
) -> np.ndarray:
    """Per-interval layout of :attr:`ScatteringTables.coef` (plain float64 arithmetic)."""
    h = x[:, 1:] - x[:, :-1]

    def cubic(y: np.ndarray, d: np.ndarray) -> np.ndarray:
        delta = (y[:, 1:] - y[:, :-1]) / h
        c2 = (3.0 * delta - 2.0 * d[:, :-1] - d[:, 1:]) / h
        c3 = (d[:, :-1] + d[:, 1:] - 2.0 * delta) / (h * h)
        return np.stack([y[:, :-1], d[:, :-1], c2, c3], axis=-1)

    return np.concatenate([x[:, :-1, None], cubic(f1, d1), cubic(f2, d2)], axis=-1)


_lock = threading.Lock()
_tables: ScatteringTables | None = None


@functools.cache
def _symbol_index() -> dict[str, int]:
    with np.load(_NPZ) as f:
        return {str(s): i for i, s in enumerate(f["symbols"])}


def element_index(symbol: str) -> int:
    """Row of ``symbol`` in the tables (atomic number minus one).

    Raises
    ------
    UnknownElementError
        If ``symbol`` is not one of the 92 tabulated elements.

    Examples
    --------
    >>> from xraylabtool.calculators.tables import element_index
    >>> element_index("Si")
    13
    """
    try:
        return _symbol_index()[symbol]
    except (KeyError, TypeError):
        raise UnknownElementError(str(symbol)) from None


def load_tables() -> ScatteringTables:
    """Return the resident tables, reading and uploading them on first call.

    Thread-safe: concurrent first calls perform a single ``runtime.upload``.

    References
    ----------
    Design spec §2.

    Examples
    --------
    >>> from xraylabtool.calculators.tables import load_tables
    >>> load_tables().x.shape
    (92, 449)
    """
    global _tables  # noqa: PLW0603
    with _lock:
        if _tables is None:
            with np.load(_NPZ) as f:
                host = ScatteringTables(
                    x=f["x"],
                    f1=f["f1"],
                    f2=f["f2"],
                    d1=f["d1"],
                    d2=f["d2"],
                    n=f["n"],
                    coef=_coefficients(f["x"], f["f1"], f["f2"], f["d1"], f["d2"]),
                    symbols=tuple(str(s) for s in f["symbols"]),
                )
            # Without a local x64 scope device_put would canonicalise to float32.
            with jax.enable_x64(True):
                _tables = upload(host)
        return _tables


def clear_tables() -> None:
    """Drop the resident tables; the next :func:`load_tables` reloads them."""
    global _tables  # noqa: PLW0603
    with _lock:
        _tables = None
