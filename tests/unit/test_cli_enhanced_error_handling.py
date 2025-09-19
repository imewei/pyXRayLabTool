"""
Tests for CLI enhanced error handling integration.

This module tests how the enhanced validation framework integrates with
the CLI commands, including error message formatting, debug output,
and user-friendly suggestions in command-line context.
"""

from io import StringIO
import sys
from unittest.mock import MagicMock, patch

import pytest

# Import CLI modules (some may not exist yet)
try:
    from xraylabtool.exceptions import FormulaError, ValidationError
    from xraylabtool.interfaces.cli_enhanced import (
        CLIErrorHandler,
        enhance_parser_with_validation,
        format_cli_error_message,
    )
    from xraylabtool.validation.enhanced_validator import (
        EnhancedValidator,
        ValidationResult,
        ValidationSuggestion,
    )
except ImportError:
    # These modules don't exist yet - we'll create them
    pytest.skip(
        "CLI enhanced error handling modules not yet implemented",
        allow_module_level=True,
    )


class TestCLIErrorHandler:
    """Test the CLI error handler integration."""

    def test_cli_error_handler_initialization(self):
        """Test CLI error handler initializes properly."""
        handler = CLIErrorHandler()

        assert handler.validator is not None
        assert isinstance(handler.validator, EnhancedValidator)
        assert handler.debug_mode is False

    def test_cli_error_handler_debug_mode(self):
        """Test CLI error handler debug mode."""
        handler = CLIErrorHandler(debug=True)
        assert handler.debug_mode is True

    def test_formula_validation_in_cli_context(self):
        """Test formula validation with CLI-specific error formatting."""
        handler = CLIErrorHandler()

        # Mock validation result
        validation_result = ValidationResult(
            is_valid=False,
            original_input="SiO2O",
            validated_input=None,
            error_type="InvalidFormula",
            error_message="Invalid chemical formula",
            suggestions=[
                ValidationSuggestion("SiO2", 0.9, "Likely meant to remove extra 'O'")
            ],
        )

        with patch.object(
            handler.validator, "validate_formula", return_value=validation_result
        ):
            error_msg = handler.handle_formula_error("SiO2O", command="calc")

            assert "SiO2O" in error_msg
            assert "SiO2" in error_msg  # Suggestion should be included
            assert "calc" in error_msg.lower()  # Command context
            assert "Did you mean" in error_msg or "suggest" in error_msg.lower()

    def test_energy_validation_in_cli_context(self):
        """Test energy validation with CLI-specific error formatting."""
        handler = CLIErrorHandler()

        validation_result = ValidationResult(
            is_valid=False,
            original_input=-5.0,
            validated_input=None,
            error_type="InvalidEnergy",
            error_message="Energy must be positive",
            suggestions=[ValidationSuggestion("5.0", 0.8, "Remove negative sign")],
        )

        with patch.object(
            handler.validator, "validate_energy", return_value=validation_result
        ):
            error_msg = handler.handle_energy_error(-5.0, command="batch")

            assert "-5.0" in error_msg
            assert "5.0" in error_msg  # Suggestion
            assert "positive" in error_msg.lower()
            assert "batch" in error_msg.lower()

    def test_debug_output_formatting(self):
        """Test debug output is properly formatted for CLI."""
        handler = CLIErrorHandler(debug=True)

        validation_result = ValidationResult(
            is_valid=False,
            original_input="invalid",
            validated_input=None,
            error_type="InvalidFormula",
            error_message="Invalid formula",
            suggestions=[],
            debug_info={
                "validation_steps": ["step1", "step2"],
                "parser_details": {"regex_match": False},
            },
        )

        with patch.object(
            handler.validator, "validate_formula", return_value=validation_result
        ):
            error_msg = handler.handle_formula_error("invalid", command="calc")

            # In debug mode, should include debug information
            assert "DEBUG:" in error_msg or "debug" in error_msg.lower()
            assert "validation_steps" in error_msg
            assert "step1" in error_msg

    def test_no_debug_output_in_normal_mode(self):
        """Test debug output is not included in normal mode."""
        handler = CLIErrorHandler(debug=False)

        validation_result = ValidationResult(
            is_valid=False,
            original_input="invalid",
            validated_input=None,
            error_type="InvalidFormula",
            error_message="Invalid formula",
            suggestions=[],
            debug_info={"details": "sensitive_info"},
        )

        with patch.object(
            handler.validator, "validate_formula", return_value=validation_result
        ):
            error_msg = handler.handle_formula_error("invalid", command="calc")

            # Should not include debug information
            assert "DEBUG:" not in error_msg
            assert "sensitive_info" not in error_msg


class TestCLIErrorMessageFormatting:
    """Test CLI error message formatting functions."""

    def test_format_simple_error_message(self):
        """Test formatting of simple error message."""
        result = ValidationResult(
            is_valid=False,
            original_input="invalid",
            validated_input=None,
            error_type="InvalidFormula",
            error_message="Invalid chemical formula",
            suggestions=[],
        )

        formatted = format_cli_error_message(result, command="calc")

        assert "Error:" in formatted or "error" in formatted.lower()
        assert "Invalid chemical formula" in formatted
        assert "invalid" in formatted

    def test_format_error_message_with_suggestions(self):
        """Test formatting of error message with suggestions."""
        result = ValidationResult(
            is_valid=False,
            original_input="SiO2O",
            validated_input=None,
            error_type="InvalidFormula",
            error_message="Invalid chemical formula",
            suggestions=[
                ValidationSuggestion("SiO2", 0.9, "Remove extra oxygen"),
                ValidationSuggestion("Si2O", 0.7, "Alternative interpretation"),
            ],
        )

        formatted = format_cli_error_message(result, command="calc")

        assert "SiO2O" in formatted
        assert "SiO2" in formatted  # Top suggestion
        assert "Did you mean" in formatted or "suggestion" in formatted.lower()
        # Should prioritize higher confidence suggestion
        assert formatted.index("SiO2") < formatted.index("Si2O")

    def test_format_error_message_with_command_context(self):
        """Test error messages include command-specific context."""
        result = ValidationResult(
            is_valid=False,
            original_input="invalid",
            validated_input=None,
            error_type="InvalidFormula",
            error_message="Invalid chemical formula",
            suggestions=[],
        )

        # Test different command contexts
        commands = ["calc", "batch", "formula"]
        for command in commands:
            formatted = format_cli_error_message(result, command=command)
            assert command in formatted.lower()

    def test_format_error_with_help_hint(self):
        """Test error messages include helpful hints."""
        result = ValidationResult(
            is_valid=False,
            original_input="",
            validated_input=None,
            error_type="EmptyInput",
            error_message="Empty formula provided",
            suggestions=[],
        )

        formatted = format_cli_error_message(result, command="calc")

        # Should include help hint
        assert "help" in formatted.lower() or "--help" in formatted
        assert "calc" in formatted


class TestCLIParserEnhancement:
    """Test enhanced argument parser integration."""

    def test_enhance_parser_adds_debug_flag(self):
        """Test that parser enhancement adds debug flag."""
        import argparse

        parser = argparse.ArgumentParser()

        enhanced_parser = enhance_parser_with_validation(parser)

        # Should have debug flag
        args = enhanced_parser.parse_args(["--debug"])
        assert args.debug is True

        args = enhanced_parser.parse_args([])
        assert args.debug is False

    def test_enhanced_parser_formula_validation(self):
        """Test enhanced parser validates formula arguments."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("formula")

        enhanced_parser = enhance_parser_with_validation(parser)

        # Mock the validation
        with patch(
            "xraylabtool.interfaces.cli_enhanced.CLIErrorHandler"
        ) as mock_handler:
            mock_handler_instance = MagicMock()
            mock_handler.return_value = mock_handler_instance
            mock_handler_instance.handle_formula_error.return_value = (
                "Error: Invalid formula"
            )

            # This should trigger validation
            with pytest.raises(SystemExit):  # argparse exits on error
                enhanced_parser.parse_args(["invalid_formula"])

    def test_enhanced_parser_preserves_original_functionality(self):
        """Test enhanced parser preserves original argument parsing."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--energy", type=float)
        parser.add_argument("formula")

        enhanced_parser = enhance_parser_with_validation(parser)

        # Should still parse normal arguments correctly
        args = enhanced_parser.parse_args(["SiO2", "--energy", "10.0"])
        assert args.formula == "SiO2"
        assert args.energy == 10.0


class TestCLIIntegrationScenarios:
    """Test complete CLI integration scenarios."""

    def test_calc_command_with_invalid_formula(self):
        """Test calc command handles invalid formula gracefully."""
        handler = CLIErrorHandler()

        # Simulate calc command with invalid formula
        with patch("sys.stderr", new=StringIO()) as fake_stderr:
            with patch("sys.exit"):
                error_msg = handler.handle_formula_error("SiO2O", command="calc")
                print(error_msg, file=sys.stderr)

                stderr_output = fake_stderr.getvalue()
                assert "SiO2O" in stderr_output
                assert "calc" in stderr_output.lower()

    def test_batch_command_with_invalid_energy_range(self):
        """Test batch command handles invalid energy range."""
        handler = CLIErrorHandler()

        error_msg = handler.handle_energy_error(-10.0, command="batch")

        assert "batch" in error_msg.lower()
        assert "-10.0" in error_msg
        assert "positive" in error_msg.lower() or "valid" in error_msg.lower()

    def test_formula_command_with_validation_feedback(self):
        """Test formula command provides detailed validation feedback."""
        handler = CLIErrorHandler(debug=True)

        validation_result = ValidationResult(
            is_valid=False,
            original_input="Al203",
            validated_input=None,
            error_type="InvalidFormula",
            error_message="Invalid subscript format",
            suggestions=[
                ValidationSuggestion("Al2O3", 0.95, "Correct subscript format")
            ],
            debug_info={
                "parse_attempts": ["Al203", "Al2O3"],
                "regex_details": {"subscript_missing": True},
            },
        )

        with patch.object(
            handler.validator, "validate_formula", return_value=validation_result
        ):
            error_msg = handler.handle_formula_error("Al203", command="formula")

            assert "Al203" in error_msg
            assert "Al2O3" in error_msg
            assert "subscript" in error_msg.lower()
            # Debug mode should show additional details
            assert "parse_attempts" in error_msg or "DEBUG" in error_msg

    def test_error_recovery_suggestions(self):
        """Test error recovery provides actionable suggestions."""
        handler = CLIErrorHandler()

        validation_result = ValidationResult(
            is_valid=False,
            original_input="SiO2O",
            validated_input=None,
            error_type="InvalidFormula",
            error_message="Extra element detected",
            suggestions=[
                ValidationSuggestion("SiO2", 0.9, "Remove extra 'O'"),
                ValidationSuggestion("Si2O3", 0.6, "Alternative interpretation"),
            ],
        )

        with patch.object(
            handler.validator, "validate_formula", return_value=validation_result
        ):
            error_msg = handler.handle_formula_error("SiO2O", command="calc")

            # Should include recovery suggestions
            assert "try again" in error_msg.lower() or "use" in error_msg.lower()
            assert "SiO2" in error_msg
            # Should provide clear next steps
            assert (
                "xraylabtool calc SiO2" in error_msg or "formula" in error_msg.lower()
            )


class TestCLIErrorHandlingPerformance:
    """Test performance aspects of CLI error handling."""

    def test_error_handling_performance(self):
        """Test error handling completes quickly."""
        handler = CLIErrorHandler()

        import time

        start_time = time.time()

        # Generate multiple validation errors
        for i in range(100):
            error_msg = handler.handle_formula_error(f"invalid{i}", command="calc")
            assert len(error_msg) > 0

        end_time = time.time()

        # Should complete quickly (less than 1 second for 100 errors)
        assert end_time - start_time < 1.0

    def test_memory_usage_during_error_handling(self):
        """Test error handling doesn't consume excessive memory."""
        handler = CLIErrorHandler()

        import gc
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Generate many errors
        for i in range(1000):
            handler.handle_formula_error(f"invalid{i}", command="calc")

        gc.collect()  # Force garbage collection
        final_memory = process.memory_info().rss

        # Memory usage should not increase dramatically (< 50MB increase)
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        assert memory_increase < 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
