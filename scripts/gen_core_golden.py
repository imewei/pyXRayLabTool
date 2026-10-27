#!/usr/bin/env python
"""Freeze the NumPy-path results of the core calculators as golden files.

Task 0 of the JAX-first plan. The goldens are the oracle every later stage is
compared against, so they are generated from the NumPy backend (selected
explicitly below; on a GPU host the default would otherwise be JAX) and are
never regenerated to make a test pass::

    uv run python scripts/gen_core_golden.py [--out DIR]

Output (default ``tests/characterization/core``)
------------------------------------------------
``golden.npz``       every input array (``inputs|<name>``) and every result
                     array (``<family>|<case>|<field>``)
``golden_text.json`` per family: python return types, exception type and
                     message of every invalid-input case, result keys, warning
                     text and other non-array facts
``golden_meta.json`` provenance: git commit, library versions, date, the active
                     backend, per-family array counts

The cases live in ``tests/characterization/core/cases.py`` and are shared with
``test_core_golden.py``.
"""

# ruff: noqa: PLC0415

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "tests" / "characterization" / "core"
sys.path.insert(0, str(REPO))

from tests.characterization.core import cases  # noqa: E402
from xraylabtool.backend.array_ops import get_backend, set_backend  # noqa: E402


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],  # noqa: S607
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _versions() -> dict[str, str]:
    import mendeleev
    import scipy

    out = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "mendeleev": getattr(mendeleev, "__version__", "unknown"),
    }
    try:
        import jax

        out["jax"] = jax.__version__
    except ImportError:
        out["jax"] = "not installed"
    return out


def main() -> int:
    """Generate the golden files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    set_backend("numpy")
    backend = type(get_backend()).__name__
    if backend != "NumpyBackend":
        raise SystemExit(f"expected the NumPy backend, got {backend}")

    inputs = cases.build_inputs()
    arrays: dict[str, np.ndarray] = {
        f"inputs{cases.SEP}{name}": value for name, value in inputs.items()
    }
    text: dict[str, object] = {}
    counts: dict[str, int] = {}
    for name in cases.FAMILIES:
        family_arrays, family_text = cases.compute_family(name, inputs)
        overlap = arrays.keys() & family_arrays.keys()
        if overlap:
            raise SystemExit(f"duplicate golden keys in {name}: {sorted(overlap)[:3]}")
        arrays.update(family_arrays)
        text[name] = family_text
        counts[name] = len(family_arrays)
        print(f"{name:24s} {len(family_arrays):6d} arrays")

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "golden.npz", **arrays)
    (args.out / "golden_text.json").write_text(
        json.dumps(text, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    )
    meta = {
        "backend": backend,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", "xraylabtool")),
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "versions": _versions(),
        "seed": cases.SEED,
        "rtol": cases.RTOL,
        "zero_atol": cases.ZERO_ATOL,
        "family_array_counts": counts,
        "total_arrays": len(arrays),
        "total_bytes": int(sum(a.nbytes for a in arrays.values())),
    }
    (args.out / "golden_meta.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True) + "\n"
    )
    size_mb = (args.out / "golden.npz").stat().st_size / 1e6
    print(
        f"{len(arrays)} arrays in {len(counts)} families; golden.npz {size_mb:.2f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
