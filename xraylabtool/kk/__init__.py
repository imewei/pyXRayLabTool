"""Kramers-Kronig transform of X-ray scattering factors (kkcalc port).

Public names are imported lazily on first access, so ``import xraylabtool.kk``
stays cheap until a function is used.
"""

from __future__ import annotations

import importlib
from typing import Any

# public name -> submodule that defines it
_EXPORTS = {
    "kk_calculate_real": "api",
    "KKResult": "result",
    "compound_spectrum": "spectrum",
    "evaluate_f2": "spectrum",
    "relativistic_correction": "transform",
    "kk_piecewise_poly": "transform",
    "kk_real_spectrum": "transform",
    "load_asf": "database",
    "AsfDatabase": "database",
    "NearEdgeSpectrum": "nearedge",
    "load_nearedge": "nearedge",
    "to_asf_scale": "nearedge",
    "merge_spectra": "nearedge",
}

__all__ = [
    "AsfDatabase",
    "KKResult",
    "NearEdgeSpectrum",
    "compound_spectrum",
    "evaluate_f2",
    "kk_calculate_real",
    "kk_piecewise_poly",
    "kk_real_spectrum",
    "load_asf",
    "load_nearedge",
    "merge_spectra",
    "relativistic_correction",
    "to_asf_scale",
]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
