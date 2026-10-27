"""``encode_formula`` and the cached formula parser (spec section 1, plan step 1.5)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.characterization.core import cases
from tests.characterization.core.test_core_golden import _text
from xraylabtool.calculators import optics
from xraylabtool.calculators.core import (
    _calculate_molecular_properties,
    get_bulk_atomic_data,
)
from xraylabtool.exceptions import FormulaError, UnknownElementError
from xraylabtool.utils import parse_formula

GOLDEN_FORMULAS = [f for f, _ in cases.FORMULAS] + list(cases.SYMBOLS)


@pytest.mark.parametrize("formula", GOLDEN_FORMULAS)
def test_matches_molecular_properties_bit_for_bit(formula):
    symbols, counts = parse_formula(formula)
    mw, ne = _calculate_molecular_properties(
        symbols, counts, get_bulk_atomic_data(tuple(symbols))
    )
    z, c, mw_e, ne_e = optics.encode_formula(formula)
    assert type(mw_e) is float
    assert type(ne_e) is float
    assert mw_e == mw  # exact equality, not approx
    assert ne_e == ne
    assert z.dtype == np.int32
    assert c.dtype == np.float64
    assert z.shape == c.shape == (len(symbols),)
    assert c.tolist() == counts
    bulk = get_bulk_atomic_data(tuple(symbols))
    assert z.tolist() == [bulk[s]["atomic_number"] - 1 for s in symbols]


def test_repeated_elements_are_aggregated_in_first_seen_order():
    z, c, _, ne = optics.encode_formula("H2OH2")
    assert z.tolist() == [0, 7]
    assert c.tolist() == [4.0, 1.0]
    assert ne == 4.0 * 1 + 1.0 * 8


_PARSE_ERRORS = [
    "formula_unknown_element",
    "formula_unbalanced_open",
    "formula_unbalanced_close",
    "formula_lowercase",
    "formula_digits_only",
    "formula_symbol",
]
_BAD_FORMULA = {
    "formula_unknown_element": "Xx2",
    "formula_unbalanced_open": "Si(O2",
    "formula_unbalanced_close": "SiO2)",
    "formula_lowercase": "sio2",
    "formula_digits_only": "123",
    "formula_symbol": "Si-O",
}


@pytest.mark.parametrize("label", _PARSE_ERRORS)
def test_errors_match_the_task0_catalogue(label):
    want = _text()["single_errors"]["errors"][label]
    with pytest.raises(Exception) as info:
        optics.encode_formula(_BAD_FORMULA[label])
    assert type(info.value).__name__ == want["type"]
    assert type(info.value).__module__ == want["module"]
    assert str(info.value) == want["message"]


def test_empty_formula_raises_formula_error():
    for bad in ("", "   "):
        with pytest.raises(FormulaError):
            optics.encode_formula(bad)


def test_element_without_table_raises_unknown_element():
    # Pu has atomic data but no scattering table; today this surfaces later as
    # FileNotFoundError, which no golden pins, so the clean error is kept.
    with pytest.raises(UnknownElementError):
        optics.encode_formula("PuO2")


def test_parse_formula_is_cached_but_public_result_is_fresh_lists():
    from xraylabtool.utils import _parse_formula_cached

    _parse_formula_cached.cache_clear()
    first = parse_formula("Ca5(PO4)3OH")
    assert first == (["Ca", "P", "O", "H"], [5.0, 3.0, 13.0, 1.0])
    assert isinstance(first[0], list)
    assert isinstance(first[1], list)
    first[0].append("junk")
    first[1].clear()
    second = parse_formula("Ca5(PO4)3OH")
    assert second == (["Ca", "P", "O", "H"], [5.0, 3.0, 13.0, 1.0])
    assert _parse_formula_cached.cache_info().hits >= 1
    symbols, counts = _parse_formula_cached("Ca5(PO4)3OH")
    assert isinstance(symbols, tuple)
    assert isinstance(counts, tuple)


def test_parse_errors_are_not_cached_and_repeat():
    for _ in range(2):
        with pytest.raises(FormulaError, match="Unmatched opening"):
            parse_formula("Si(O2")
