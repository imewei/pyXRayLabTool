"""
Tests for chemical formula parsing functionality.

This test module mirrors the comprehensive test suite from test/formula_parsing.jl,
ensuring identical behavior between the Julia and Python implementations.
"""

import os
import sys
import pytest

try:
    from xraylabtool.utils import parse_formula
except ImportError:
    # Add parent directory to path to import xraylabtool
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from xraylabtool.utils import parse_formula


class TestFormulaParsingBasic:
    """Test basic functionality mirroring Julia tests."""

    def test_single_atoms(self):
        """Test single atom parsing (Julia: Single atoms test set)."""
        # Test carbon
        symbols, counts = parse_formula("C")
        assert symbols == ["C"]
        assert counts == [1.0]

        # Test uranium
        symbols, counts = parse_formula("U")
        assert symbols == ["U"]
        assert counts == [1.0]

    def test_mixed_case_correctness(self):
        """Test mixed case element handling (Julia: Mixed case correctness test set)."""
        # Test CO (Carbon Monoxide) - two separate elements
        symbols, counts = parse_formula("CO")
        assert symbols == ["C", "O"]
        assert counts == [1.0, 1.0]

        # Test Co (Cobalt) - single element
        symbols, counts = parse_formula("Co")
        assert symbols == ["Co"]
        assert counts == [1.0]


class TestFormulaParsingFractional:
    """Test fractional stoichiometry parsing."""

    def test_fractional_stoichiometry(self):
        """Test fractional stoichiometry (Julia: Fractional stoichiometry test set)."""
        # Test equal fractions
        symbols, counts = parse_formula("H0.5He0.5")
        assert symbols == ["H", "He"]
        assert counts == [0.5, 0.5]

        # Test mixed fractional and integer counts
        symbols, counts = parse_formula("H2O0.5")
        assert symbols == ["H", "O"]
        assert counts == [2.0, 0.5]

        # Test decimal with no leading integer
        symbols, counts = parse_formula("C.25")
        assert symbols == ["C"]
        assert counts == [0.25]


class TestFormulaParsingLong:
    """Test long formulas with many elements."""

    def test_long_formulas(self):
        """Test formula with ≥10 elements (Julia: Long formulas test set)."""
        # Test formula with ≥10 elements
        symbols, counts = parse_formula("HHeLiBeBCHNOFNeNaMg")
        expected_symbols = [
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "H",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
        ]
        expected_counts = [1.0 for _ in range(13)]

        assert symbols == expected_symbols
        assert counts == expected_counts
        assert len(symbols) >= 10  # Verify it's truly a long formula

        # Test with numeric subscripts on long formula
        symbols, counts = parse_formula("H2He3Li4Be5B6C7N8O9F10Ne11")
        expected_symbols = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]
        expected_counts = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]

        assert symbols == expected_symbols
        assert counts == expected_counts
        assert len(symbols) >= 10


class TestFormulaParsingComplex:
    """Test complex formulas and validation."""

    def test_complex_formulas_and_validation(self):
        """Test common chemical compounds (Julia: Complex formulas and validation test set)."""
        # Test common chemical compounds
        symbols, counts = parse_formula("SiO2")
        assert symbols == ["Si", "O"]
        assert counts == [1.0, 2.0]

        symbols, counts = parse_formula("Al2O3")
        assert symbols == ["Al", "O"]
        assert counts == [2.0, 3.0]

        symbols, counts = parse_formula("CaCO3")
        assert symbols == ["Ca", "C", "O"]
        assert counts == [1.0, 1.0, 3.0]

        # Test formula with large numbers
        symbols, counts = parse_formula("C100H200")
        assert symbols == ["C", "H"]
        assert counts == [100.0, 200.0]

        # Test formula with very precise decimals
        symbols, counts = parse_formula("H0.123He0.876")
        assert symbols == ["H", "He"]
        # Use pytest.approx for floating point comparison
        assert counts[0] == pytest.approx(0.123, abs=1e-10)
        assert counts[1] == pytest.approx(0.876, abs=1e-10)


class TestFormulaParsingInvalid:
    """Test invalid inputs and error handling."""

    def test_invalid_inputs(self):
        """Test invalid inputs (Julia: Invalid inputs test set)."""
        # Test empty string
        with pytest.raises(ValueError, match="Invalid chemical formula"):
            parse_formula("")

        # Test string with only numbers
        with pytest.raises(ValueError, match="Invalid chemical formula"):
            parse_formula("123")

        # Test lowercase-only string
        with pytest.raises(ValueError, match="Invalid chemical formula"):
            parse_formula("xyz")

        # Additional invalid inputs
        with pytest.raises(ValueError, match="Invalid chemical formula"):
            parse_formula("abc")


class TestFormulaParsingEdgeCases:
    """Test edge cases and specific regex behavior."""

    def test_edge_cases(self):
        """Test various edge cases to ensure robust parsing."""
        # Test single letter elements
        symbols, counts = parse_formula("H")
        assert symbols == ["H"]
        assert counts == [1.0]

        # Test two-letter elements
        symbols, counts = parse_formula("He")
        assert symbols == ["He"]
        assert counts == [1.0]

        # Test three-letter elements (if any exist in periodic table)
        # Note: Most elements are 1-2 letters, but regex should handle more

        # Test very small decimal numbers
        symbols, counts = parse_formula("H0.001")
        assert symbols == ["H"]
        assert counts == [0.001]

        # Test decimal with many digits
        symbols, counts = parse_formula("C1.234567")
        assert symbols == ["C"]
        assert counts == [1.234567]

        # Test integer zero (should be valid though chemically meaningless)
        symbols, counts = parse_formula("H0")
        assert symbols == ["H"]
        assert counts == [0.0]


class TestFormulaParsingRegexCompatibility:
    """Test specific cases that verify regex compatibility with Julia version."""

    def test_regex_identical_behavior(self):
        """Test cases that specifically verify the regex behaves identically to Julia version."""
        # The Julia regex: r"([A-Z][a-z]*)(\\d*\\.\\d*|\\d*)"
        # Should match exactly the same patterns

        # Test various number formats
        test_cases = [
            ("H1", ["H"], [1.0]),
            ("H10", ["H"], [10.0]),
            ("H0.5", ["H"], [0.5]),
            ("H.5", ["H"], [0.5]),
            ("H0.123", ["H"], [0.123]),
            ("H10.5", ["H"], [10.5]),
            ("HeO2.5", ["He", "O"], [1.0, 2.5]),
            ("CaC12H22O11", ["Ca", "C", "H", "O"], [1.0, 12.0, 22.0, 11.0]),
        ]

        for formula, expected_symbols, expected_counts in test_cases:
            symbols, counts = parse_formula(formula)
            assert symbols == expected_symbols, f"Failed for formula: {formula}"
            assert counts == expected_counts, f"Failed for formula: {formula}"

    def test_duplicate_elements(self):
        """Test formulas with duplicate elements (which should be preserved)."""
        # Note: The Julia tests show "HHeLiBeBCHNOFNeNaMg" has two H's
        # This should be preserved as separate entries
        symbols, counts = parse_formula("HHeLiBeBCHNOFNeNaMg")

        # Count occurrences of H
        h_count = symbols.count("H")
        assert h_count == 2  # Should have two separate H entries

        # Test another case with duplicates
        symbols, counts = parse_formula("H2HeH3")
        assert symbols == ["H", "He", "H"]
        assert counts == [2.0, 1.0, 3.0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
