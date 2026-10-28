#!/usr/bin/env python
"""Build ``xraylabtool/data/scattering_tables.npz`` from the CXRO ``.nff`` files.

Task 1.2 of the JAX-first plan (design spec section 2). The ``.nff`` files stay the
source of truth; the npz is the resident, padded form the device programs
consume::

    uv run python scripts/gen_scattering_tables.py            # (re)write the npz
    uv run python scripts/gen_scattering_tables.py --check    # exit 1 on drift

Contents of the npz (``Z`` = 92 elements in atomic-number order, ``N`` = the
longest table, derived)
-----------------------------------------------------------------------------
``x, f1, f2, d1, d2``  float64 ``[Z, N]``: energy (eV), f1, f2 and their PCHIP
                       node slopes from ``backend.interpolation._pchip_slopes``
``n``                  int32 ``[Z]``: real knots per row
``symbols``            ``<U2`` ``[Z]``: element symbols

Padding past ``n[z]``: strictly increasing dummy knots ``x_n * (1 + k)``
(``k = 1, 2, ...``), f values equal to the last value, slopes 0, so a padded
row evaluates like the unpadded one on ``[x[0], x[n-1]]``.

The data directory holds 93 files (``C.nff`` and ``c.nff``); files are grouped
by ``stem.capitalize()`` as ``scattering_data._AVAILABLE_ELEMENTS`` does, and the
build fails if two files of one group differ.
"""

# ruff: noqa: PLC0415

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "xraylabtool" / "data" / "AtomicScatteringFactor"
DEFAULT_OUT = REPO / "xraylabtool" / "data" / "scattering_tables.npz"
N_ELEMENTS = 92


def _unique_files() -> dict[str, Path]:
    """One file per capitalised stem; fail if duplicates (``C``/``c``) differ."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(DATA_DIR.glob("*.nff")):
        groups[path.stem.capitalize()].append(path)
    for symbol, paths in groups.items():
        first = paths[0].read_bytes()
        for other in paths[1:]:
            if other.read_bytes() != first:
                raise SystemExit(f"{symbol}: {paths[0].name} and {other.name} differ")
    return {symbol: paths[0] for symbol, paths in groups.items()}


def _symbols_by_atomic_number(available: set[str]) -> list[str]:
    from xraylabtool.data_handling.atomic_cache import _ATOMIC_DATA_PRELOADED

    ordered = sorted(
        _ATOMIC_DATA_PRELOADED, key=lambda s: _ATOMIC_DATA_PRELOADED[s]["atomic_number"]
    )
    numbers = [_ATOMIC_DATA_PRELOADED[s]["atomic_number"] for s in ordered]
    if numbers != list(range(1, N_ELEMENTS + 1)) or set(ordered) != available:
        raise SystemExit("data files do not match the 92 elements Z = 1..92")
    return ordered


def _row(symbol: str) -> tuple[np.ndarray, ...]:
    import jax
    import jax.numpy as jnp

    from xraylabtool.backend.interpolation import _pchip_slopes
    from xraylabtool.calculators.scattering_data import load_scattering_factor_data

    data = load_scattering_factor_data(symbol)
    x, f1, f2 = (
        np.asarray(data[c].values, dtype=np.float64) for c in ("E", "f1", "f2")
    )
    if x.shape[0] < 2 or not (np.diff(x) > 0).all() or x[0] <= 0:
        raise SystemExit(f"{symbol}: energies must be positive and strictly increasing")
    if not (np.isfinite(f1).all() and np.isfinite(f2).all()):
        raise SystemExit(f"{symbol}: non-finite scattering factors")
    # CPU compute so the npz is identical whichever host builds it.
    with jax.enable_x64(True), jax.default_device(jax.devices("cpu")[0]):
        xj = jnp.asarray(x)
        d1 = np.asarray(_pchip_slopes(xj, jnp.asarray(f1)))
        d2 = np.asarray(_pchip_slopes(xj, jnp.asarray(f2)))
    return x, f1, f2, d1, d2


def build_tables() -> dict[str, np.ndarray]:
    """Build every array of the npz in memory (deterministic)."""
    files = _unique_files()
    symbols = _symbols_by_atomic_number(set(files))
    rows = [_row(s) for s in symbols]
    n = np.array([r[0].shape[0] for r in rows], dtype=np.int32)
    width = int(n.max())
    out = {k: np.zeros((N_ELEMENTS, width)) for k in ("x", "f1", "f2", "d1", "d2")}
    for z, (x, f1, f2, d1, d2) in enumerate(rows):
        m = x.shape[0]
        pad = np.arange(1, width - m + 1)
        out["x"][z] = np.concatenate([x, x[-1] * (1.0 + pad)])
        out["f1"][z] = np.concatenate([f1, np.full(width - m, f1[-1])])
        out["f2"][z] = np.concatenate([f2, np.full(width - m, f2[-1])])
        out["d1"][z, :m] = d1
        out["d2"][z, :m] = d2
    return {**out, "n": n, "symbols": np.array(symbols, dtype="<U2")}


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    """Exact for integers and text; a few ulps for floats (XLA rounds per platform)."""
    if a.dtype.kind == "f":
        return bool(np.allclose(a, b, rtol=1e-13, atol=0.0, equal_nan=True))
    return bool(np.array_equal(a, b))


def _drift(built: dict[str, np.ndarray], shipped: dict[str, np.ndarray]) -> list[str]:
    bad = [k for k in built if k not in shipped]
    bad += [k for k in shipped if k not in built]
    for k, value in built.items():
        if k in shipped and not (
            value.shape == shipped[k].shape
            and value.dtype == shipped[k].dtype
            and _same(value, shipped[k])
        ):
            bad.append(k)
    return sorted(set(bad))


def main(argv: list[str] | None = None) -> int:
    """Write the npz, or with ``--check`` compare against it (exit 1 on drift)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 on drift")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    built = build_tables()
    if args.check:
        if not args.out.exists():
            print(f"missing: {args.out}", file=sys.stderr)
            return 1
        with np.load(args.out) as shipped:
            bad = _drift(built, {k: shipped[k] for k in shipped.files})
        if bad:
            print(
                f"scattering tables drifted from the .nff files: {bad}", file=sys.stderr
            )
            return 1
        print(f"{args.out.name}: up to date")
        return 0
    np.savez(args.out, **built)
    print(f"wrote {args.out} ({built['x'].shape[0]} x {built['x'].shape[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
