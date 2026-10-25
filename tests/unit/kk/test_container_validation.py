"""Invariants of the frozen result containers ``KKResult`` and ``RefineOutcome``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from xraylabtool.exceptions import ValidationError
from xraylabtool.kk import KKResult, RefineOutcome


def _result(**overrides: Any) -> KKResult:
    fields: dict[str, Any] = {
        "formula": "Si",
        "energy_kev": np.array([1.0, 2.0, 3.0]),
        "f1": np.array([10.0, 11.0, 12.0]),
        "f2": np.array([1.0, 0.5, 0.2]),
        "relativistic_correction": 0.1,
        "merge_points_kev": None,
        "n_refined_points": 0,
        "backend": "numpy",
    }
    fields.update(overrides)
    return KKResult(**fields)


def _outcome(**overrides: Any) -> RefineOutcome:
    fields: dict[str, Any] = {
        "energy_ev": np.array([1.0, 2.0, 3.0]),
        "f1": np.array([1.0, 2.0, 3.0]),
        "f2": np.array([0.1, 0.2, 0.3]),
        "n_added": 1,
        "exhausted": False,
    }
    fields.update(overrides)
    return RefineOutcome(**fields)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"f1": np.array([1.0])}, "same length"),
        ({"f2": np.array([1.0, np.nan, 2.0])}, "finite"),
        ({"energy_kev": np.ones((3, 1))}, "1-D"),
        ({"n_refined_points": -1}, "n_refined_points"),
        ({"backend": "cuda"}, "backend"),
        ({"merge_points_kev": (0.32, 0.28)}, "low < high"),
        ({"merge_points_kev": (0.28, np.inf)}, "low < high"),
        (
            {"nearedge_scaled_kev_f2": (np.array([1.0, 2.0]), np.array([1.0]))},
            "same length",
        ),
    ],
)
def test_kk_result_rejects_inconsistent_state(
    overrides: dict[str, Any], needle: str
) -> None:
    with pytest.raises(ValidationError, match=needle):
        _result(**overrides)


@pytest.mark.unit
def test_kk_result_arrays_are_read_only_copies() -> None:
    f1 = np.array([10.0, 11.0, 12.0])
    res = _result(f1=f1, nearedge_scaled_kev_f2=(np.array([1.0]), np.array([2.0])))
    f1[0] = -1.0  # the caller's array no longer aliases the result
    assert res.f1[0] == 10.0
    for arr in (res.energy_kev, res.f1, res.f2, *res.nearedge_scaled_kev_f2):  # type: ignore[misc]
        assert not arr.flags.writeable


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"f1": np.array([1.0])}, "same length"),
        ({"energy_ev": np.array([1.0, 1.0, 3.0])}, "strictly increasing"),
        ({"f1": np.array([1.0, np.inf, 3.0])}, "finite"),
        ({"n_added": -1}, "n_added"),
        ({"n_pole_skipped": -2}, "n_pole_skipped"),
    ],
)
def test_refine_outcome_rejects_inconsistent_state(
    overrides: dict[str, Any], needle: str
) -> None:
    with pytest.raises(ValidationError, match=needle):
        _outcome(**overrides)


@pytest.mark.unit
def test_refine_outcome_arrays_are_read_only_copies() -> None:
    f2 = np.array([0.1, 0.2, 0.3])
    out = _outcome(f2=f2)
    f2[0] = 9.0
    assert out.f2[0] == 0.1
    assert not out.energy_ev.flags.writeable
    assert not out.f1.flags.writeable
    assert not out.f2.flags.writeable
