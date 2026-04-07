"""
Tests for the canonical chemical formula parser and its delegating wrappers.

Covers basic formulas, decimal stoichiometry, parentheses (nested),
error handling, and consistency between the three call sites.
"""

from __future__ import annotations

import pytest

from xraylabtool.exceptions import FormulaError
from xraylabtool.utils import parse_formula

# ---------------------------------------------------------------------------
# Canonical parser: basic formulas
# ---------------------------------------------------------------------------


class TestParseFormulaBasic:
    """Test basic formula parsing."""

    def test_sio2(self):
        syms, cnts = parse_formula("SiO2")
        assert syms == ["Si", "O"]
        assert cnts == [1.0, 2.0]

    def test_al2o3(self):
        syms, cnts = parse_formula("Al2O3")
        assert syms == ["Al", "O"]
        assert cnts == [2.0, 3.0]

    def test_h2o(self):
        syms, cnts = parse_formula("H2O")
        assert syms == ["H", "O"]
        assert cnts == [2.0, 1.0]

    def test_nacl(self):
        syms, cnts = parse_formula("NaCl")
        assert syms == ["Na", "Cl"]
        assert cnts == [1.0, 1.0]

    def test_caco3(self):
        syms, cnts = parse_formula("CaCO3")
        assert syms == ["Ca", "C", "O"]
        assert cnts == [1.0, 1.0, 3.0]

    def test_single_element(self):
        syms, cnts = parse_formula("Si")
        assert syms == ["Si"]
        assert cnts == [1.0]

    def test_single_letter_element(self):
        syms, cnts = parse_formula("H")
        assert syms == ["H"]
        assert cnts == [1.0]

    def test_two_letter_element(self):
        syms, cnts = parse_formula("He")
        assert syms == ["He"]
        assert cnts == [1.0]

    def test_element_with_explicit_count_1(self):
        syms, cnts = parse_formula("O1")
        assert syms == ["O"]
        assert cnts == [1.0]

    def test_large_counts(self):
        syms, cnts = parse_formula("C100H200")
        assert syms == ["C", "H"]
        assert cnts == [100.0, 200.0]

    def test_co_vs_Co(self):
        """CO (carbon monoxide) vs Co (cobalt)."""
        syms, cnts = parse_formula("CO")
        assert syms == ["C", "O"]
        assert cnts == [1.0, 1.0]

        syms, cnts = parse_formula("Co")
        assert syms == ["Co"]
        assert cnts == [1.0]

    def test_duplicate_elements_aggregated(self):
        """Repeated elements in flat formulas are aggregated."""
        syms, cnts = parse_formula("H2OH3")
        assert "H" in syms
        idx = syms.index("H")
        assert cnts[idx] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Decimal stoichiometry
# ---------------------------------------------------------------------------


class TestParseFormulaDecimal:
    def test_h05_he05(self):
        syms, cnts = parse_formula("H0.5He0.5")
        assert syms == ["H", "He"]
        assert cnts == pytest.approx([0.5, 0.5])

    def test_ca05_sr05_tio3(self):
        syms, cnts = parse_formula("Ca0.5Sr0.5TiO3")
        assert syms == ["Ca", "Sr", "Ti", "O"]
        assert cnts == pytest.approx([0.5, 0.5, 1.0, 3.0])

    def test_mixed_fractional_and_integer(self):
        syms, cnts = parse_formula("H2O0.5")
        assert syms == ["H", "O"]
        assert cnts == pytest.approx([2.0, 0.5])

    def test_precise_decimals(self):
        _syms, cnts = parse_formula("H0.123He0.876")
        assert cnts[0] == pytest.approx(0.123, abs=1e-10)
        assert cnts[1] == pytest.approx(0.876, abs=1e-10)

    def test_very_small_decimal(self):
        _syms, cnts = parse_formula("H0.001")
        assert cnts == [0.001]

    def test_integer_zero(self):
        _syms, cnts = parse_formula("H0")
        assert cnts == [0.0]


# ---------------------------------------------------------------------------
# Parentheses (including nested)
# ---------------------------------------------------------------------------


class TestParseFormulaParentheses:
    def test_ca5_po4_3oh(self):
        syms, cnts = parse_formula("Ca5(PO4)3OH")
        result = dict(zip(syms, cnts))
        assert result == pytest.approx({"Ca": 5.0, "P": 3.0, "O": 13.0, "H": 1.0})

    def test_ca3_po4_2(self):
        syms, cnts = parse_formula("Ca3(PO4)2")
        result = dict(zip(syms, cnts))
        assert result == pytest.approx({"Ca": 3.0, "P": 2.0, "O": 8.0})

    def test_mg_oh_2(self):
        syms, cnts = parse_formula("Mg(OH)2")
        result = dict(zip(syms, cnts))
        assert result == pytest.approx({"Mg": 1.0, "O": 2.0, "H": 2.0})

    def test_nested_ca10_po4_6_oh_2(self):
        syms, cnts = parse_formula("Ca10(PO4)6(OH)2")
        result = dict(zip(syms, cnts))
        assert result == pytest.approx({"Ca": 10.0, "P": 6.0, "O": 26.0, "H": 2.0})

    def test_paren_no_multiplier(self):
        syms, cnts = parse_formula("(OH)")
        result = dict(zip(syms, cnts))
        assert result == pytest.approx({"O": 1.0, "H": 1.0})


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestParseFormulaErrors:
    def test_empty_string(self):
        with pytest.raises(FormulaError):
            parse_formula("")

    def test_whitespace_only(self):
        with pytest.raises(FormulaError):
            parse_formula("   ")

    def test_unmatched_open_paren(self):
        with pytest.raises(FormulaError):
            parse_formula("Ca(OH")

    def test_unmatched_close_paren(self):
        with pytest.raises(FormulaError):
            parse_formula("CaOH)")

    def test_only_numbers(self):
        with pytest.raises(FormulaError):
            parse_formula("123")

    def test_lowercase_only(self):
        with pytest.raises(FormulaError):
            parse_formula("xyz")


# ---------------------------------------------------------------------------
# Consistency: delegating wrappers agree with canonical parser
# ---------------------------------------------------------------------------


_CONSISTENCY_FORMULAS = [
    "SiO2",
    "Al2O3",
    "H2O",
    "NaCl",
    "Ca5(PO4)3OH",
    "Ca3(PO4)2",
    "Mg(OH)2",
    "Ca10(PO4)6(OH)2",
]


class TestDelegatingWrappers:
    @pytest.fixture(params=_CONSISTENCY_FORMULAS)
    def formula(self, request):
        return request.param

    def test_validators_parse_formula_matches(self, formula):
        from xraylabtool.validation.validators import _parse_formula

        syms, cnts = parse_formula(formula)
        expected = dict(zip(syms, cnts))
        result = _parse_formula(formula)
        assert result == pytest.approx(expected)

    def test_compound_analysis_parse_matches(self, formula):
        from xraylabtool.data_handling.compound_analysis import parse_chemical_formula

        syms, cnts = parse_formula(formula)
        expected = {s: round(c) for s, c in zip(syms, cnts)}
        result = parse_chemical_formula(formula)
        assert result == expected


# ---------------------------------------------------------------------------
# Regex compatibility with Julia implementation
# ---------------------------------------------------------------------------


class TestRegexCompatibility:
    """Verify number-format handling matches the Julia regex."""

    @pytest.mark.parametrize(
        "formula, expected_symbols, expected_counts",
        [
            ("H1", ["H"], [1.0]),
            ("H10", ["H"], [10.0]),
            ("H0.5", ["H"], [0.5]),
            ("H0.123", ["H"], [0.123]),
            ("H10.5", ["H"], [10.5]),
            ("HeO2.5", ["He", "O"], [1.0, 2.5]),
            ("CaC12H22O11", ["Ca", "C", "H", "O"], [1.0, 12.0, 22.0, 11.0]),
        ],
    )
    def test_number_formats(self, formula, expected_symbols, expected_counts):
        syms, cnts = parse_formula(formula)
        assert syms == expected_symbols, f"Failed for {formula}"
        assert cnts == expected_counts, f"Failed for {formula}"
