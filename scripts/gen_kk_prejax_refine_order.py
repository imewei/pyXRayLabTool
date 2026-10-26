#!/usr/bin/env python
"""Freeze the internal interval order of the host ``refine_spectrum`` loop.

Run once on the unmigrated (host NumPy) ``xraylabtool/kk/refine.py`` and commit
the result; ``tests/unit/kk/test_device_refine.py`` compares the device
implementation's next-active order against it::

    uv run python scripts/gen_kk_prejax_refine_order.py

Provenance: generated at commit ``1dbcb76`` (branch ``feat/kk-jax-first``, the
host ``refine.py`` before Task 4 of the migration) with the jax and numpy
versions stored in the ``.npz`` (``git_commit``, ``jax_version``,
``numpy_version``). Do not regenerate after the migration: the script spies on
the host helper ``refine._drop_slivers``, which no longer exists, so it only
runs against that commit.

Output ``tests/unit/kk/data/prejax_refine_order.npz``
-----------------------------------------------------
Two cases on Si, absolute tolerance 0.05, grid ``geomspace(100, 20000, 12)``:
``free`` (no limit reached) and ``capped`` (``max_points=10``, which truncates
a pass). Per case ``<case>``:

``<case>_n_passes``       number of passes that evaluated midpoints
``<case>_lo_<k>``, ``<case>_hi_<k>``
                          the active intervals of pass ``k`` (1-based) in
                          internal order, after the sliver split, i.e. the
                          intervals whose midpoints are checked
``<case>_energy_ev``      final sorted grid; ``<case>_n_added``, ``<case>_exhausted``
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from xraylabtool.kk import (  # noqa: E402
    compound_spectrum,
    evaluate_f2,
    kk_real_spectrum,
    refine,
)
from xraylabtool.kk.transform import relativistic_correction  # noqa: E402

OUT = REPO / "tests" / "unit" / "kk" / "data" / "prejax_refine_order.npz"
GRID = np.geomspace(100.0, 20000.0, 12)
TOLERANCE = 0.05
CASES = {"free": {}, "capped": {"max_points": 10}}


def _run(case: str, kwargs: dict[str, int]) -> dict[str, object]:
    seg, coeffs = compound_spectrum(["Si"], [1.0])
    rel = relativistic_correction([14], [1.0])
    f1 = kk_real_spectrum(GRID, seg, coeffs, rel)
    f2 = evaluate_f2(GRID, seg, coeffs)
    seen: list[tuple[np.ndarray, np.ndarray]] = []
    real = refine._drop_slivers

    def spy(iv, seg_ev):
        out = real(iv, seg_ev)
        seen.append((out[0][0].copy(), out[0][1].copy()))
        return out

    refine._drop_slivers = spy
    try:
        res = refine.refine_spectrum(
            GRID, f1, f2, seg, coeffs, rel, tolerance=TOLERANCE, **kwargs
        )
    finally:
        refine._drop_slivers = real
    data: dict[str, object] = {
        f"{case}_n_passes": len(seen),
        f"{case}_energy_ev": res.energy_ev,
        f"{case}_n_added": res.n_added,
        f"{case}_exhausted": res.exhausted,
    }
    for k, (lo, hi) in enumerate(seen, start=1):
        data[f"{case}_lo_{k}"] = lo
        data[f"{case}_hi_{k}"] = hi
    return data


def main() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    data: dict[str, object] = {
        "git_commit": commit,
        "jax_version": version("jax"),
        "numpy_version": version("numpy"),
    }
    for case, kwargs in CASES.items():
        data.update(_run(case, kwargs))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **data)  # type: ignore[arg-type]
    print(f"wrote {OUT} ({commit[:7]})")


if __name__ == "__main__":
    main()
