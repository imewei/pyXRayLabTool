#!/usr/bin/env python
"""Freeze the pre-migration ``xraylabtool.kk`` outputs for the JAX-first migration.

Run once on the unmigrated code (NumPy kernel, forced below) and commit the
result; later tasks compare the JAX implementation against it::

    uv run python scripts/gen_kk_prejax_baseline.py

Provenance: generated at commit ``1bf56d904500e2aa128b32f0f082e83f91579799``
(``Merge pull request #32``, branch ``feat/kk-jax-first`` before any kernel
change) with jax 0.12.0.1 and numpy 2.5.4 installed, the NumPy backend. The commit
and versions are also stored in the ``.npz`` (``git_commit``, ``jax_version``,
``numpy_version``). Do not regenerate after the migration: the file is the
reference for what the old code returned.

Output ``tests/unit/kk/data/prejax_baseline.npz``
-------------------------------------------------
Per case in ``H, Mo, U, Au, SiO2, Fe2O3, Ar``:

``<case>_seg``, ``<case>_coeffs``  ``compound_spectrum`` output (the spectra that
                                   vanish when assembly moves to the device)
``<case>_rel_corr``                relativistic correction Z*
``<case>_energy_ev``               ``tests.fixtures.kk_probes.probe_energies``
``<case>_f1``                      ``kk_real_spectrum`` at those energies. The three
                                   above-range energies ``x_n·(1+k)`` are evaluated on
                                   the unpadded spectrum: the reference a padded
                                   evaluation must reproduce.

``merge_*`` (polystyrene C8H8 + ``tests/characterization/kk/nearedge_example.txt``,
NEXAFS, merge points 280-320 eV): ``merge_seg``, ``merge_coeffs`` from
``merge_spectra``; ``merge_energy_ev``, ``merge_f1`` at its probe energies;
``merge_api_energy_kev``, ``merge_api_f1``, ``merge_api_f2`` from
``kk_calculate_real`` on its default grid.

``refine_*`` (Si, absolute tolerance 0.05): ``refine_input_ev`` (the knot grid),
``refine_input_f1``, ``refine_energy_ev``, ``refine_f1``, ``refine_f2``,
``refine_n_added`` from ``refine_spectrum``; ``refine_api_energy_kev``,
``refine_api_f1``, ``refine_api_f2``, ``refine_api_n_refined`` from
``kk_calculate_real("Si", refine_tolerance=0.05)``.
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests.fixtures.kk_probes import probe_energies  # noqa: E402
from xraylabtool.backend.array_ops import set_backend  # noqa: E402
from xraylabtool.kk import (  # noqa: E402
    compound_spectrum,
    evaluate_f2,
    kk_calculate_real,
    kk_real_spectrum,
    load_asf,
    load_nearedge,
    merge_spectra,
    refine_spectrum,
    relativistic_correction,
    to_asf_scale,
)
from xraylabtool.utils import parse_formula  # noqa: E402

PRE_MIGRATION_COMMIT = "1bf56d904500e2aa128b32f0f082e83f91579799"
CASES = ("H", "Mo", "U", "Au", "SiO2", "Fe2O3", "Ar")
NEAREDGE = REPO / "tests" / "characterization" / "kk" / "nearedge_example.txt"
MERGE_EV = (280.0, 320.0)
REFINE_TOLERANCE = 0.05
OUT = REPO / "tests" / "unit" / "kk" / "data" / "prejax_baseline.npz"


def _rel_corr(symbols: list[str], counts: list[float]) -> float:
    db = load_asf()
    return relativistic_correction([db.symbols.index(s) + 1 for s in symbols], counts)


def _finite(name: str, arr: np.ndarray) -> np.ndarray:
    if not np.isfinite(arr).all():
        raise SystemExit(f"non-finite values in {name}")
    return arr


def main() -> None:
    set_backend("numpy")
    out: dict[str, object] = {}

    for case in CASES:
        symbols, counts = parse_formula(case)
        seg, coeffs = compound_spectrum(symbols, counts)
        rel = _rel_corr(symbols, counts)
        energy = probe_energies(seg)
        out[f"{case}_seg"] = seg
        out[f"{case}_coeffs"] = coeffs
        out[f"{case}_rel_corr"] = np.float64(rel)
        out[f"{case}_energy_ev"] = energy
        out[f"{case}_f1"] = _finite(case, kk_real_spectrum(energy, seg, coeffs, rel))

    symbols, counts = parse_formula("C8H8")
    near = to_asf_scale(load_nearedge(NEAREDGE), "nexafs")
    pre_seg, pre_coeffs = compound_spectrum(symbols, counts)
    seg, coeffs = merge_spectra(near, pre_seg, pre_coeffs, merge_points_ev=MERGE_EV)
    rel = _rel_corr(symbols, counts)
    energy = probe_energies(seg)
    api = kk_calculate_real(
        "C8H8",
        nearedge=load_nearedge(NEAREDGE),
        input_type="nexafs",
        merge_points_kev=(MERGE_EV[0] / 1000, MERGE_EV[1] / 1000),
    )
    out |= {
        "merge_seg": seg,
        "merge_coeffs": coeffs,
        "merge_energy_ev": energy,
        "merge_f1": _finite("merge_f1", kk_real_spectrum(energy, seg, coeffs, rel)),
        "merge_api_energy_kev": api.energy_kev,
        "merge_api_f1": _finite("merge_api_f1", api.f1),
        "merge_api_f2": _finite("merge_api_f2", api.f2),
    }

    symbols, counts = parse_formula("Si")
    seg, coeffs = compound_spectrum(symbols, counts)
    rel = _rel_corr(symbols, counts)
    f1_in = kk_real_spectrum(seg, seg, coeffs, rel)
    outcome = refine_spectrum(
        seg,
        f1_in,
        evaluate_f2(seg, seg, coeffs),
        seg,
        coeffs,
        rel,
        tolerance=REFINE_TOLERANCE,
    )
    api = kk_calculate_real("Si", refine_tolerance=REFINE_TOLERANCE)
    out |= {
        "refine_input_ev": seg,
        "refine_input_f1": _finite("refine_input_f1", f1_in),
        "refine_energy_ev": outcome.energy_ev,
        "refine_f1": _finite("refine_f1", outcome.f1),
        "refine_f2": _finite("refine_f2", outcome.f2),
        "refine_n_added": np.int64(outcome.n_added),
        "refine_api_energy_kev": api.energy_kev,
        "refine_api_f1": _finite("refine_api_f1", api.f1),
        "refine_api_f2": _finite("refine_api_f2", api.f2),
        "refine_api_n_refined": np.int64(api.n_refined_points),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        **{k: np.asarray(v) for k, v in out.items()},
        git_commit=np.array(PRE_MIGRATION_COMMIT),
        jax_version=np.array(version("jax")),
        numpy_version=np.array(np.__version__),
    )
    print(f"wrote {OUT.relative_to(REPO)} ({len(out)} arrays)")


if __name__ == "__main__":
    main()
