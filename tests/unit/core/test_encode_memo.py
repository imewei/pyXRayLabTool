"""The per-formula encoding memo of the core path (spec risk 1).

``core._encode_material`` is memoised so a large batch parses and encodes each
distinct formula once. Failures must keep raising, the memo must follow
``clear_scattering_factor_cache`` (the element cache is refilled by the next call)
and the density screen must treat plain numbers like arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

from xraylabtool.calculators import core
from xraylabtool.calculators.cache import (
    clear_scattering_factor_cache,
    get_cached_elements,
)
from xraylabtool.exceptions import FormulaError


@pytest.mark.unit
def test_a_formula_is_encoded_once_and_the_arrays_are_read_only() -> None:
    first = core._encode_material("SiO2")
    assert core._encode_material("SiO2") is first
    z, counts, _mw, _ne = first
    assert not z.flags.writeable
    assert not counts.flags.writeable


@pytest.mark.unit
def test_a_failing_formula_raises_every_time() -> None:
    for _ in range(2):
        with pytest.raises(Exception, match="Xx"):
            core._encode_material("Xx2O")


@pytest.mark.unit
def test_clearing_the_scattering_cache_clears_the_memo_and_refills_the_element_cache() -> (
    None
):
    before = core._encode_material("SiO2")
    clear_scattering_factor_cache()
    assert get_cached_elements() == []
    after = core._encode_material("SiO2")
    assert after is not before
    assert {"Si", "O"} <= set(get_cached_elements())
    np.testing.assert_array_equal(after[0], before[0])


@pytest.mark.unit
@pytest.mark.parametrize("density", [2.2, 2, np.float64(2.2), np.array([2.2])])
def test_density_screen_accepts_positive_numbers_of_every_kind(density: object) -> None:
    core._validate_formula_and_density("SiO2", density)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize(
    "density", [0.0, -1.0, 0, np.float64(-2.0), np.array([1.0, 0.0])]
)
def test_density_screen_rejects_non_positive_numbers_of_every_kind(
    density: object,
) -> None:
    with pytest.raises(ValueError, match="Mass density must be positive"):
        core._validate_formula_and_density("SiO2", density)  # type: ignore[arg-type]


@pytest.mark.unit
def test_empty_formula_still_raises_formula_error() -> None:
    with pytest.raises(FormulaError):
        core._validate_formula_and_density("", 2.2)
