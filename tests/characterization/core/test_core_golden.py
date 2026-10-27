"""Compare the core calculators against the frozen NumPy-path goldens (Task 0).

``golden.npz`` / ``golden_text.json`` were generated once from the NumPy
backend by ``scripts/gen_core_golden.py`` and are the oracle of the JAX-first
migration. Each family of ``cases.py`` has its own test, so a later stage can
name the failing family.

Rules (never loosened; see the spec, section Verification):

* values: ``rtol = 1e-12``; ``atol`` only where the golden is exactly 0
  (``cases.ZERO_ATOL``); NaN stays NaN in the same places and +-inf equals
  +-inf; shape and dtype must match exactly;
* python return types, exception types and messages, result keys and their
  order, warning text: exact equality.

The package is JAX-only, so the tests compare the device programs on whatever
JAX platform is active (CPU, or the GPU of a CUDA host). Set
``XLT_EXPECT_BACKEND=cpu`` (or ``gpu``) to assert which platform ran.
"""

from __future__ import annotations

from functools import cache
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.characterization.core import cases

GOLDEN_DIR = Path(__file__).parent
SEP = cases.SEP
MAX_REPORTED = 20


@cache
def _npz() -> Any:
    return np.load(GOLDEN_DIR / "golden.npz", allow_pickle=False)


@cache
def _text() -> dict[str, Any]:
    return json.loads((GOLDEN_DIR / "golden_text.json").read_text())  # type: ignore[no-any-return]


@cache
def _meta() -> dict[str, Any]:
    return json.loads((GOLDEN_DIR / "golden_meta.json").read_text())  # type: ignore[no-any-return]


@cache
def _inputs() -> dict[str, np.ndarray]:
    npz = _npz()
    prefix = f"inputs{SEP}"
    return {k[len(prefix) :]: npz[k] for k in npz.files if k.startswith(prefix)}


def _backend_label() -> str:
    import jax

    return f"jax/{jax.default_backend()}"


def _structure_problem(got: np.ndarray, expected: np.ndarray) -> str | None:
    """Shape, dtype, NaN placement, infinities and non-float arrays (exact checks)."""
    if got.shape != expected.shape:
        return f"shape {got.shape} != {expected.shape}"
    if got.dtype != expected.dtype:
        return f"dtype {got.dtype} != {expected.dtype}"
    if expected.dtype.kind not in "fc":
        return None if np.array_equal(got, expected) else "values differ"
    if np.any(np.isnan(got) != np.isnan(expected)):
        return "NaN positions differ"
    nonfinite = ~np.isfinite(expected)
    if not np.array_equal(got[nonfinite], expected[nonfinite], equal_nan=True):
        return "inf values differ"
    return None


def compare_arrays(actual: Any, expected: np.ndarray) -> tuple[str | None, float]:
    """Return (problem or None, largest relative error) for one golden array."""
    got = np.asarray(actual)
    problem = _structure_problem(got, expected)
    if problem or expected.dtype.kind not in "fc":
        return problem, 0.0 if problem is None else np.inf
    mask = np.isfinite(expected)
    got, expected = got[mask], expected[mask]
    if got.size == 0:
        return None, 0.0
    diff = np.abs(got - expected)
    scale = np.abs(expected)
    zero = scale == 0.0
    tol = np.where(zero, cases.ZERO_ATOL, cases.RTOL * scale)
    worst = float(np.max(np.where(zero, diff, diff / np.where(zero, 1.0, scale))))
    if np.any(diff > tol):
        i = int(np.argmax(diff - tol))
        return (
            f"max rel err {worst:.3e} (golden {expected[i]!r}, got {got[i]!r})",
            worst,
        )
    return None, worst


def _compare_text(label: str, actual: Any, expected: Any, problems: list[str]) -> None:
    if actual != expected:
        problems.append(f"{label}: expected {expected!r}, got {actual!r}")


def check_family(name: str) -> tuple[list[str], float]:
    """Recompute family ``name`` and return (problems, worst relative error)."""
    arrays, text = cases.compute_family(name, _inputs())
    npz, gold_text = _npz(), _text()[name]
    prefix = f"{name}{SEP}"
    gold_keys = {k for k in npz.files if k.startswith(prefix)}
    problems: list[str] = []
    worst = 0.0

    problems += [f"missing result {k}" for k in sorted(gold_keys - arrays.keys())]
    problems += [f"unexpected result {k}" for k in sorted(arrays.keys() - gold_keys)]
    for key in sorted(gold_keys & arrays.keys()):
        problem, rel = compare_arrays(arrays[key], npz[key])
        worst = max(worst, rel if np.isfinite(rel) else 0.0)
        if problem:
            problems.append(f"{key}: {problem}")

    for section in ("types", "errors", "facts"):
        got, want = text[section], gold_text[section]
        for key in sorted(want.keys() | got.keys()):
            if key not in got:
                problems.append(f"{section} {key}: missing, golden {want[key]!r}")
            elif key not in want:
                problems.append(f"{section} {key}: unexpected {got[key]!r}")
            else:
                _compare_text(f"{section} {key}", got[key], want[key], problems)
    return problems, worst


def test_active_backend_label(record_property: Any) -> None:
    """Print which JAX platform the comparisons below ran on (CPU vs CUDA hosts)."""
    label = _backend_label()
    record_property("backend", label)
    print(f"\ncore golden comparison backend: {label}")
    expected = os.environ.get("XLT_EXPECT_BACKEND")
    if expected:
        assert expected.lower() in label.lower()


def test_golden_files_are_numpy_path_provenance() -> None:
    meta = _meta()
    assert meta["backend"] == "NumpyBackend"
    assert meta["rtol"] == cases.RTOL


def test_inputs_match_builder() -> None:
    """The case builder still reproduces the stored inputs.

    Integer and string inputs match exactly; float inputs are built with libm
    (``sin``, ``exp``), which differs in the last ulps between platforms, so they
    match to a few ulps. The goldens are always computed from the stored inputs.
    """
    built = cases.build_inputs()
    stored = _inputs()
    assert built.keys() == stored.keys()
    for name, value in built.items():
        if np.issubdtype(value.dtype, np.floating):
            np.testing.assert_allclose(
                value, stored[name], rtol=1e-10, atol=0.0, equal_nan=True, err_msg=name
            )
        else:
            assert np.array_equal(value, stored[name], equal_nan=True), name


def test_golden_covers_every_family() -> None:
    meta_counts = _meta()["family_array_counts"]
    assert set(meta_counts) == set(cases.FAMILIES)
    assert set(_text()) == set(cases.FAMILIES)
    npz = _npz()
    for name, count in meta_counts.items():
        assert sum(k.startswith(f"{name}{SEP}") for k in npz.files) == count, name


@pytest.mark.parametrize("family", list(cases.FAMILIES))
def test_golden_family(family: str, record_property: Any) -> None:
    problems, worst = check_family(family)
    record_property("worst_relative_error", worst)
    print(f"\n{family}: worst relative error {worst:.3e}")
    if problems:
        shown = "\n  ".join(problems[:MAX_REPORTED])
        more = (
            f"\n  ... and {len(problems) - MAX_REPORTED} more"
            if len(problems) > MAX_REPORTED
            else ""
        )
        pytest.fail(f"{family}: {len(problems)} mismatches\n  {shown}{more}")
