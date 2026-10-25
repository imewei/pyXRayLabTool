"""Generate ``truncation.png`` for ``docs/physics/kramers_kronig.rst``.

Plots ``f1_trunc - f1_full`` for C, Fe and Pb, where ``f1_trunc`` integrates
the tabulated f2 only up to 30 keV and ``f1_full`` up to 500 keV, the same
computation as ``test_30kev_truncation_matches_paper_trend`` in
``tests/characterization/kk/test_end_to_end_golden.py``.

Run from the repository root::

    uv run python docs/_static/kk/make_truncation_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from xraylabtool.kk import (
    compound_spectrum,
    kk_real_spectrum,
    load_asf,
    relativistic_correction,
)

CUT_EV = 30000.0
OUT = Path(__file__).with_name("truncation.png")


def truncation_deficit(symbol: str, energy_ev: np.ndarray) -> np.ndarray:
    """``f1_full - f1_trunc`` (electrons/atom) at ``energy_ev`` (eV)."""
    z = load_asf().symbols.index(symbol) + 1
    seg, coeffs = compound_spectrum([symbol], [1.0])
    rel_corr = relativistic_correction([z], [1.0])
    full = kk_real_spectrum(energy_ev, seg, coeffs, rel_corr)
    seg_t = np.append(seg[seg < CUT_EV], CUT_EV)
    trunc = kk_real_spectrum(energy_ev, seg_t, coeffs[: seg_t.size - 1], rel_corr)
    return full - trunc


def main() -> None:
    energy_ev = np.geomspace(30.0, 29500.0, 400)
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=110)
    for symbol, color in (("C", "#1b9e77"), ("Fe", "#d95f02"), ("Pb", "#7570b3")):
        deficit = -truncation_deficit(symbol, energy_ev)
        assert np.isfinite(deficit).all() and (deficit > 0).all()
        ax.loglog(energy_ev / 1000.0, deficit, color=color, lw=1.4, label=symbol)
    ax.axvline(CUT_EV / 1000.0, color="0.4", ls="--", lw=1.0)
    ax.text(27.0, 1e-2, "30 keV cut", ha="right", va="center", fontsize=8)
    ax.set_xlabel("Photon energy (keV)")
    ax.set_ylabel(r"$f_1^{\rm trunc} - f_1^{\rm full}$ (e/atom)")
    ax.set_title("f1 lost by integrating the tabulated f2 only to 30 keV", fontsize=10)
    ax.grid(True, which="major", lw=0.4, alpha=0.5)
    ax.legend(frameon=False, fontsize=9, loc="center left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
