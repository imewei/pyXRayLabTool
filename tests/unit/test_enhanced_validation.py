"""
Tests for enhanced input validation framework with intelligent error detection.

This module tests the new validation system that provides better error handling,
fuzzy string matching for corrections, and context-aware error messages for
improved user experience in the CLI.
"""

from unittest.mock import patch

import pytest

# We'll import these as we implement them
try:
    from xraylabtool.exceptions import FormulaError, ValidationError
    from xraylabtool.validation.enhanced_validator import (
        EnhancedValidator,
        ValidationResult,
        ValidationSuggestion,
    )
    from xraylabtool.validation.fuzzy_matcher import FormulaFuzzyMatcher
except ImportError:
    # These modules don't exist yet - we'll create them
    pytest.skip(
        "Enhanced validation modules not yet implemented", allow_module_level=True
    )


class TestValidationResult:
    """Test the ValidationResult data structure."""

    def test_validation_result_success(self):
        """Test successful validation result."""
        result = ValidationResult(
            is_valid=True,
            original_input="SiO2",
            validated_input="SiO2",
            error_type=None,
            error_message=None,
            suggestions=[],
        )

        assert result.is_valid is True
        assert result.original_input == "SiO2"
        assert result.validated_input == "SiO2"
        assert result.error_type is None
        assert result.error_message is None
        assert result.suggestions == []

    def test_validation_result_failure_with_suggestions(self):
        """Test failed validation result with suggestions."""
        suggestions = [
            ValidationSuggestion(
                suggestion="SiO2",
                confidence=0.85,
                reason="Likely meant to remove extra 'O'",
            )
        ]

        result = ValidationResult(
            is_valid=False,
            original_input="SiO2O",
            validated_input=None,
            error_type="InvalidFormula",
            error_message="Invalid chemical formula detected",
            suggestions=suggestions,
        )

        assert result.is_valid is False
        assert result.original_input == "SiO2O"
        assert result.validated_input is None
        assert result.error_type == "InvalidFormula"
        assert result.error_message == "Invalid chemical formula detected"
        assert len(result.suggestions) == 1
        assert result.suggestions[0].suggestion == "SiO2"
        assert result.suggestions[0].confidence == 0.85


class TestValidationSuggestion:
    """Test the ValidationSuggestion data structure."""

    def test_validation_suggestion_creation(self):
        """Test creation of validation suggestions."""
        suggestion = ValidationSuggestion(
            suggestion="Al2O3", confidence=0.92, reason="Common typo: 'Al203' → 'Al2O3'"
        )

        assert suggestion.suggestion == "Al2O3"
        assert suggestion.confidence == 0.92
        assert suggestion.reason == "Common typo: 'Al203' → 'Al2O3'"

    def test_validation_suggestion_comparison(self):
        """Test that suggestions can be compared by confidence."""
        suggestion1 = ValidationSuggestion("SiO2", 0.8, "reason1")
        suggestion2 = ValidationSuggestion("Si2O", 0.9, "reason2")

        # Should be sortable by confidence
        suggestions = [suggestion1, suggestion2]
        sorted_suggestions = sorted(
            suggestions, key=lambda x: x.confidence, reverse=True
        )

        assert sorted_suggestions[0].suggestion == "Si2O"
        assert sorted_suggestions[1].suggestion == "SiO2"


class TestFormulaFuzzyMatcher:
    """Test the fuzzy string matching for chemical formulas."""

    def test_fuzzy_matcher_initialization(self):
        """Test fuzzy matcher initializes with known formulas."""
        matcher = FormulaFuzzyMatcher()

        # Should have common chemical formulas in its database
        assert len(matcher.known_formulas) > 0
        assert "SiO2" in matcher.known_formulas
        assert "Al2O3" in matcher.known_formulas

    def test_exact_match(self):
        """Test exact matches return high confidence."""
        matcher = FormulaFuzzyMatcher()
        suggestions = matcher.get_suggestions("SiO2")

        # Exact match should return the same formula with high confidence
        assert len(suggestions) >= 1
        assert suggestions[0].suggestion == "SiO2"
        assert suggestions[0].confidence >= 0.95

    def test_common_typos(self):
        """Test common chemical formula typos are detected."""
        matcher = FormulaFuzzyMatcher()

        test_cases = [
            ("SiO2O", "SiO2"),  # Extra element
            ("Al203", "Al2O3"),  # Missing subscript formatting
            ("CaCo3", "CaCO3"),  # Wrong capitalization
            ("H2SO4", "H2SO4"),  # Should match exactly
        ]

        for typo, expected in test_cases:
            suggestions = matcher.get_suggestions(typo)
            assert len(suggestions) > 0

            # The first suggestion should be the expected correction
            if typo != expected:  # Not an exact match
                assert suggestions[0].suggestion == expected
                assert suggestions[0].confidence > 0.5

    def test_element_name_corrections(self):
        """Test corrections for element name typos."""
        matcher = FormulaFuzzyMatcher()

        # Test single element corrections
        test_cases = [
            ("Silicion", "Si"),
            ("Aluminium", "Al"),
            ("Oxigen", "O"),
        ]

        for typo, expected in test_cases:
            suggestions = matcher.get_suggestions(typo)
            assert len(suggestions) > 0
            # Should suggest the correct element symbol
            assert any(s.suggestion == expected for s in suggestions)

    def test_no_suggestions_for_gibberish(self):
        """Test that completely invalid input returns no good suggestions."""
        matcher = FormulaFuzzyMatcher()
        suggestions = matcher.get_suggestions("xyz123abc")

        # Should either return no suggestions or very low confidence ones
        assert len(suggestions) == 0 or all(s.confidence < 0.3 for s in suggestions)


class TestEnhancedValidator:
    """Test the main enhanced validation framework."""

    def test_validator_initialization(self):
        """Test validator initializes with fuzzy matcher."""
        validator = EnhancedValidator()

        assert validator.fuzzy_matcher is not None
        assert isinstance(validator.fuzzy_matcher, FormulaFuzzyMatcher)

    def test_valid_formula_validation(self):
        """Test validation of valid chemical formulas."""
        validator = EnhancedValidator()

        valid_formulas = ["SiO2", "Al2O3", "CaCO3", "H2SO4", "NaCl"]

        for formula in valid_formulas:
            result = validator.validate_formula(formula)
            assert result.is_valid is True
            assert result.validated_input == formula
            assert result.error_type is None

    def test_invalid_formula_with_suggestions(self):
        """Test validation of invalid formulas provides suggestions."""
        validator = EnhancedValidator()

        invalid_cases = [
            ("sio2", "SiO2"),  # Case issue
            ("invalid_xyz", None),  # Completely invalid
            ("", None),  # Empty string
            ("123abc", None),  # Invalid format
        ]

        for invalid_formula, expected_suggestion in invalid_cases:
            result = validator.validate_formula(invalid_formula)
            assert result.is_valid is False
            assert result.original_input == invalid_formula
            assert result.error_type is not None

            if expected_suggestion:
                assert len(result.suggestions) > 0
                assert result.suggestions[0].suggestion == expected_suggestion

    def test_energy_validation(self):
        """Test validation of energy values."""
        validator = EnhancedValidator()

        # Valid energy values
        valid_energies = [1.0, 10.0, 25.0, 50.0]
        for energy in valid_energies:
            result = validator.validate_energy(energy)
            assert result.is_valid is True
            assert result.validated_input == energy

        # Invalid energy values
        invalid_energies = [-1.0, 0.0, 1000.0]  # Negative, zero, too high
        for energy in invalid_energies:
            result = validator.validate_energy(energy)
            assert result.is_valid is False
            assert len(result.suggestions) > 0  # Should suggest valid ranges

    def test_density_validation(self):
        """Test validation of density values."""
        validator = EnhancedValidator()

        # Valid density values
        valid_densities = [0.1, 2.33, 10.0, 22.4]  # Range from light to heavy materials
        for density in valid_densities:
            result = validator.validate_density(density)
            assert result.is_valid is True
            assert result.validated_input == density

        # Invalid density values
        invalid_densities = [-1.0, 0.0, 100.0]  # Negative, zero, unrealistically high
        for density in invalid_densities:
            result = validator.validate_density(density)
            assert result.is_valid is False
            assert len(result.suggestions) > 0

    def test_context_aware_error_messages(self):
        """Test that error messages are context-aware."""
        validator = EnhancedValidator()

        # Test formula error in calc command context
        result = validator.validate_formula("invalid_xyz", command_context="calc")
        assert result.is_valid is False
        assert (
            "calc" in result.error_message
            or "calculation" in result.error_message.lower()
        )

        # Test energy error in batch command context
        result = validator.validate_energy(-5.0, command_context="batch")
        assert result.is_valid is False
        assert (
            "batch" in result.error_message or "batch" in result.error_message.lower()
        )

    def test_validation_with_debug_info(self):
        """Test validation includes debug information when requested."""
        validator = EnhancedValidator()

        result = validator.validate_formula("invalid_xyz", debug=True)
        assert result.is_valid is False
        assert hasattr(result, "debug_info")
        assert result.debug_info is not None
        assert "validation_steps" in result.debug_info

    def test_batch_validation(self):
        """Test batch validation of multiple inputs."""
        validator = EnhancedValidator()

        inputs = ["SiO2", "sio2", "CaCO3", "invalid_xyz"]
        results = validator.validate_batch_formulas(inputs)

        assert len(results) == 4
        assert results[0].is_valid is True  # SiO2
        assert results[1].is_valid is False  # sio2 (should suggest SiO2)
        assert results[2].is_valid is True  # CaCO3
        assert results[3].is_valid is False  # invalid_xyz

        # Check that invalid formulas have suggestions
        assert len(results[1].suggestions) > 0
        assert results[1].suggestions[0].suggestion == "SiO2"


class TestValidationIntegration:
    """Test integration between validation components."""

    def test_validator_fuzzy_matcher_integration(self):
        """Test that validator properly uses fuzzy matcher."""
        validator = EnhancedValidator()

        with patch.object(
            validator.fuzzy_matcher, "get_suggestions"
        ) as mock_suggestions:
            mock_suggestions.return_value = [
                ValidationSuggestion("SiO2", 0.9, "Close match")
            ]

            result = validator.validate_formula("invalid_xyz")

            # Verify fuzzy matcher was called
            mock_suggestions.assert_called_once_with("invalid_xyz")
            assert result.is_valid is False
            assert len(result.suggestions) == 1
            assert result.suggestions[0].suggestion == "SiO2"

    def test_error_message_formatting(self):
        """Test that error messages are properly formatted."""
        validator = EnhancedValidator()

        result = validator.validate_formula("invalid123")
        assert result.is_valid is False
        assert result.error_message is not None
        assert len(result.error_message) > 0

        # Error message should be informative
        assert (
            "invalid" in result.error_message.lower()
            or "formula" in result.error_message.lower()
        )

    def test_performance_with_large_inputs(self):
        """Test validation performance with large inputs."""
        validator = EnhancedValidator()

        # Test with many formulas
        large_input = ["SiO2", "Al2O3", "CaCO3"] * 100

        import time

        start_time = time.time()
        results = validator.validate_batch_formulas(large_input)
        end_time = time.time()

        # Should complete reasonably quickly (less than 1 second)
        assert end_time - start_time < 1.0
        assert len(results) == 300
        assert all(r.is_valid for r in results)  # All should be valid


class TestValidationErrorHandling:
    """Test error handling in validation framework."""

    def test_validation_with_none_input(self):
        """Test validation handles None input gracefully."""
        validator = EnhancedValidator()

        result = validator.validate_formula(None)
        assert result.is_valid is False
        assert result.error_type == "NullInput"
        assert "None" in result.error_message or "null" in result.error_message.lower()

    def test_validation_with_non_string_input(self):
        """Test validation handles non-string input."""
        validator = EnhancedValidator()

        invalid_inputs = [123, [], {}, object()]

        for invalid_input in invalid_inputs:
            result = validator.validate_formula(invalid_input)
            assert result.is_valid is False
            assert result.error_type == "InvalidType"

    def test_fuzzy_matcher_exception_handling(self):
        """Test that fuzzy matcher exceptions are handled gracefully."""
        validator = EnhancedValidator()

        with patch.object(
            validator.fuzzy_matcher, "get_suggestions"
        ) as mock_suggestions:
            mock_suggestions.side_effect = Exception("Test exception")

            # Should not raise exception, should return error result
            result = validator.validate_formula("invalid_xyz")
            assert result.is_valid is False
            # Should have handled the fuzzy matcher exception gracefully
            assert len(result.suggestions) == 0  # No suggestions due to exception


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
