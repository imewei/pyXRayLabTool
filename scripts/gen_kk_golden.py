#!/usr/bin/env python
"""Generate golden oracle files for the ``xraylabtool.kk`` tests from upstream kkcalc.

Upstream (https://github.com/benajamin/kkcalc, Zlib licence) is cloned at a
pinned commit into a temporary directory and imported *by this script only*;
the package never imports it. Run with warnings off, because upstream emits an
import-time ``SyntaxWarning`` and unclosed-file ``ResourceWarning``s::

    uv run python -W ignore scripts/gen_kk_golden.py [--upstream DIR] [--out DIR]

``--upstream`` reuses an existing checkout (its HEAD must equal the pinned SHA).

NumPy-compat patches applied to upstream: none (unmodified ``KK_PP``,
``kk_calculate_real``, ``improve_accuracy`` and ``data.calculate_asf`` run on
NumPy 2.x with finite output for every case below).

Output (``tests/characterization/kk/golden``)
----------------------------------------------
Transform cases ``synthetic.npz``, ``si.npz``, ``au.npz``, ``sio2.npz``, ``ar.npz``, ``ne.npz``,
``c8h8_nearedge.npz``:

``eval_ev``      evaluation energies (eV), sorted, include interior knots with
                 continuous f2, knots at an f2 jump, and the first/last knot
``eval_kind``    int per eval point: 0 generic, 1 interior knot (f2 continuous),
                 2 interior knot at an f2 jump, 3 first or last knot
``seg_ev``       segment knot energies (eV), shape (n + 1,)
``coeffs``       segment polynomial coefficients, shape (n, 5)
``rel_corr``     relativistic correction (float)
``f1_kk_pp``     upstream ``KK_PP(eval_ev, seg_ev, coeffs, rel_corr)``
and, for formula cases (not ``synthetic``), ``energy_ev, f1, f2`` from
upstream ``kk_calculate_real`` on its full, unclipped integration grid.
``c8h8_nearedge.npz`` also holds ``merge_points``, ``pre_seg_ev`` and
``pre_coeffs`` (the pure-ASF grid before the near-edge merge).

Refinement cases ``refine_si.npz``, ``refine_sio2.npz``:
``input_ev`` (grid handed to ``improve_accuracy``), ``f1_input``, ``seg_ev``,
``coeffs``, ``rel_corr``, ``tolerance`` (absolute), ``recursion``, and the
refined ``energy_ev, f1, f2``.

Every file also carries ``upstream_commit`` and ``patches`` strings.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np

UPSTREAM_URL = "https://github.com/benajamin/kkcalc.git"
UPSTREAM_COMMIT = "f331ca31a654af946f7d550a22cf4e9c11fa8afc"
PATCHES = "none"

REPO = Path(__file__).resolve().parents[1]
CHAR = REPO / "tests" / "characterization" / "kk"
NEAREDGE = CHAR / "nearedge_example.txt"

# Ar and Ne carry real f2 jumps (ionisation thresholds); Si, Au, SiO2 are continuous.
FORMULA_CASES = {"si": "Si", "au": "Au", "sio2": "SiO2", "ar": "Ar", "ne": "Ne"}
POLYSTYRENE = "C8H8"
MERGE_POINTS = [280, 320]  # as in upstream examples/example.py
REFINE_CASES = {"refine_si": "Si", "refine_sio2": "SiO2"}
REFINE_TOLERANCE = 0.05  # absolute, same units as f1/f2 (upstream's own __main__ value)
REFINE_RECURSION = 50
SEED = 20241024


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args],  # noqa: S607
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _checkout(upstream: Path | None, tmp: Path) -> Path:
    if upstream is None:
        upstream = tmp / "kkcalc"
        _git("clone", "--quiet", UPSTREAM_URL, str(upstream))
        _git("checkout", "--quiet", UPSTREAM_COMMIT, cwd=upstream)
    head = _git("rev-parse", "HEAD", cwd=upstream)
    if head != UPSTREAM_COMMIT:
        raise SystemExit(f"upstream HEAD {head} != pinned {UPSTREAM_COMMIT}")
    return upstream


def _eval_points(
    seg_ev: np.ndarray, coeffs: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Generic points plus knots: continuous interior, f2-jump, and both ends."""
    n = len(coeffs)
    # f2 on the left and right of each interior knot; the last segment is never
    # evaluated to the right of seg_ev[-1].
    inner = seg_ev[1:-1]
    left = _poly(coeffs[:-1], inner)
    right = _poly(coeffs[1:], inner)
    jump = np.abs(left - right) > 1e-9 * np.maximum(np.abs(left), np.abs(right))
    cont_idx = np.flatnonzero(~jump)[:: max(1, int((~jump).sum()) // 12)]
    jump_idx = np.flatnonzero(jump)[:6]
    generic = np.exp(rng.uniform(np.log(seg_ev[0]), np.log(seg_ev[-1]), 60))
    energies = np.concatenate(
        [generic, inner[cont_idx], inner[jump_idx], seg_ev[[0, -1]]]
    )
    kinds = np.concatenate(
        [
            np.zeros(len(generic), int),
            np.ones(len(cont_idx), int),
            np.full(len(jump_idx), 2),
            np.full(2, 3),
        ]
    )
    order = np.argsort(energies, kind="stable")
    energies, kinds = energies[order], kinds[order]
    if n < 2 or len(np.unique(energies)) != len(energies):
        raise SystemExit("evaluation points are not unique")
    return energies, kinds


def _poly(c: np.ndarray, e: np.ndarray) -> np.ndarray:
    return c[:, 0] * e + c[:, 1] + c[:, 2] / e + c[:, 3] / e**2 + c[:, 4] / e**3


def _finite(**arrays: np.ndarray) -> None:
    for name, arr in arrays.items():
        if not np.isfinite(arr).all():
            raise SystemExit(f"upstream produced non-finite values in {name!r}")


def _save(path: Path, **arrays: object) -> None:
    np.savez_compressed(
        path,
        **{k: np.asarray(v) for k, v in arrays.items()},
        upstream_commit=np.array(UPSTREAM_COMMIT),
        patches=np.array(PATCHES),
    )
    print(f"wrote {path.name}")


def _transform_case(kk, seg_ev, coeffs, rel_corr, rng) -> dict[str, np.ndarray]:
    eval_ev, eval_kind = _eval_points(seg_ev, coeffs, rng)
    f1 = kk.KK_PP(eval_ev, seg_ev, coeffs, rel_corr)
    _finite(f1_kk_pp=f1)
    return {
        "eval_ev": eval_ev,
        "eval_kind": eval_kind,
        "seg_ev": seg_ev,
        "coeffs": coeffs,
        "rel_corr": np.float64(rel_corr),
        "f1_kk_pp": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--upstream", type=Path, help="existing upstream checkout at the pinned SHA"
    )
    parser.add_argument("--out", type=Path, default=CHAR / "golden")
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    args.out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        up = _checkout(args.upstream, Path(tmp))
        if (up / "examples" / "PS_004_-dc.txt").read_bytes() != NEAREDGE.read_bytes():
            raise SystemExit(
                "nearedge_example.txt differs from upstream examples/PS_004_-dc.txt"
            )
        sys.path.insert(0, str(up))
        sys.dont_write_bytecode = True
        import kkcalc  # noqa: PLC0415  (upstream, oracle only)
        from kkcalc import data, kk  # noqa: PLC0415

        print(
            f"upstream kkcalc {kkcalc.__version__} @ {UPSTREAM_COMMIT[:10]}, numpy {np.__version__}"
        )

        def formula(name: str) -> tuple[np.ndarray, np.ndarray, float]:
            stoich = data.ParseChemicalFormula(name)
            seg_ev, coeffs = data.calculate_asf(stoich)
            return seg_ev, coeffs, kk.calc_relativistic_correction(stoich)

        # Synthetic: hand-sized fixed arrays, with an f2 jump at 120 and 800 eV.
        rng = np.random.default_rng(SEED)
        seg_ev = np.array([10.0, 20.0, 50.0, 120.0, 300.0, 800.0, 2000.0])
        coeffs = np.column_stack(
            [
                rng.normal(0, 1e-3, 6),
                rng.normal(2, 0.5, 6),
                rng.normal(0, 50, 6),
                rng.normal(0, 2e3, 6),
                rng.normal(0, 1e4, 6),
            ]
        )
        _save(
            args.out / "synthetic.npz", **_transform_case(kk, seg_ev, coeffs, 14.0, rng)
        )

        for key, name in FORMULA_CASES.items():
            rng = np.random.default_rng(SEED)
            seg_ev, coeffs, rel = formula(name)
            out = kk.kk_calculate_real(None, name)  # full grid, unclipped
            _finite(output=out)
            _save(
                args.out / f"{key}.npz",
                **_transform_case(kk, seg_ev, coeffs, rel, rng),
                formula=name,
                energy_ev=out[:, 0],
                f1=out[:, 1],
                f2=out[:, 2],
            )

        # Polystyrene with the upstream near-edge example.
        rng = np.random.default_rng(SEED)
        stoich = data.ParseChemicalFormula(POLYSTYRENE)
        rel = kk.calc_relativistic_correction(stoich)
        pre_ev, pre_coeffs = data.calculate_asf(stoich)
        near = data.convert_data(
            data.load_data(str(NEAREDGE), None), FromType="NEXAFS", ToType="asf"
        )
        seg_ev, coeffs = data.merge_spectra(
            near, pre_ev, pre_coeffs, merge_points=MERGE_POINTS
        )
        out = kk.kk_calculate_real(
            str(NEAREDGE),
            POLYSTYRENE,
            load_options=None,
            input_data_type="NEXAFS",
            merge_points=MERGE_POINTS,
            add_background=False,
            fix_distortions=False,
            curve_tolerance=None,
        )
        _finite(output=out)
        if not (np.array_equal(out[:, 0], seg_ev)):
            raise SystemExit("kk_calculate_real grid differs from merge_spectra grid")
        _save(
            args.out / "c8h8_nearedge.npz",
            **_transform_case(kk, seg_ev, coeffs, rel, rng),
            formula=POLYSTYRENE,
            merge_points=np.array(MERGE_POINTS, dtype=float),
            pre_seg_ev=pre_ev,
            pre_coeffs=pre_coeffs,
            energy_ev=out[:, 0],
            f1=out[:, 1],
            f2=out[:, 2],
        )

        # Adaptive refinement (improve_accuracy via curve_tolerance).
        for key, name in REFINE_CASES.items():
            seg_ev, coeffs, rel = formula(name)
            f1_in = kk.KK_PP(seg_ev, seg_ev, coeffs, rel)
            refined = kk.kk_calculate_real(
                None,
                name,
                curve_tolerance=REFINE_TOLERANCE,
                curve_recursion=REFINE_RECURSION,
            )
            _finite(f1_input=f1_in, refined=refined)
            _save(
                args.out / f"{key}.npz",
                formula=name,
                input_ev=seg_ev,
                f1_input=f1_in,
                seg_ev=seg_ev,
                coeffs=coeffs,
                rel_corr=np.float64(rel),
                tolerance=np.float64(REFINE_TOLERANCE),
                recursion=np.int64(REFINE_RECURSION),
                energy_ev=refined[:, 0],
                f1=refined[:, 1],
                f2=refined[:, 2],
            )
    shutil.rmtree(up / "kkcalc" / "__pycache__", ignore_errors=True)


if __name__ == "__main__":
    main()
