"""Batch processing partial-failure tests (item 4.3).

Verifies that calculate_batch_properties returns valid results for good formulas
even when other formulas in the same batch are invalid.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from xraylabtool.data_handling.batch_processing import (
    BatchConfig,
    calculate_batch_properties,
    process_single_calculation,
)

# A minimal energy array that is definitely in range [0.03, 30] keV
_ENERGIES = np.array([8.0, 10.0, 12.0], dtype=np.float64)

# Formulas known to be valid
_VALID_FORMULAS = ["SiO2", "Al2O3"]
_VALID_DENSITIES = [2.2, 3.95]

# Formulas known to be invalid (non-existent elements / garbage)
_INVALID_FORMULAS = ["Xx999", "@@##invalid"]
_INVALID_DENSITIES = [1.0, 1.0]


@pytest.fixture
def no_progress_config() -> BatchConfig:
    """BatchConfig with progress bar disabled to keep test output clean."""
    return BatchConfig(enable_progress=False, max_workers=1, chunk_size=10)


# ---------------------------------------------------------------------------
# process_single_calculation
# ---------------------------------------------------------------------------


class TestProcessSingleCalculation:
    def test_valid_formula_returns_result(self) -> None:
        formula, result = process_single_calculation("SiO2", _ENERGIES, 2.2)
        assert formula == "SiO2"
        assert result is not None

    def test_invalid_formula_returns_none(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            formula, result = process_single_calculation("Xx999invalid", _ENERGIES, 1.0)
        assert formula == "Xx999invalid"
        assert result is None

    def test_invalid_formula_emits_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            process_single_calculation("@@invalid@@", _ENERGIES, 1.0)
        assert len(caught) >= 1
        messages = [str(w.message) for w in caught]
        assert any("@@invalid@@" in m for m in messages)


# ---------------------------------------------------------------------------
# calculate_batch_properties — mixed valid / invalid
# ---------------------------------------------------------------------------


class TestBatchPartialFailure:
    def test_valid_formulas_return_results(
        self, no_progress_config: BatchConfig
    ) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = calculate_batch_properties(
                _VALID_FORMULAS, _ENERGIES, _VALID_DENSITIES, config=no_progress_config
            )

        # At least one key corresponds to a valid result
        valid_results = {k: v for k, v in results.items() if v is not None}
        assert len(valid_results) >= 1

    def test_invalid_formulas_yield_none(self, no_progress_config: BatchConfig) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = calculate_batch_properties(
                _INVALID_FORMULAS,
                _ENERGIES,
                _INVALID_DENSITIES,
                config=no_progress_config,
            )

        # All results for invalid formulas should be None
        for value in results.values():
            assert value is None

    def test_mixed_batch_valid_survive(self, no_progress_config: BatchConfig) -> None:
        """Valid formulas yield XRayResult even when invalid ones co-exist."""
        formulas = _VALID_FORMULAS + _INVALID_FORMULAS
        densities = _VALID_DENSITIES + _INVALID_DENSITIES

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = calculate_batch_properties(
                formulas, _ENERGIES, densities, config=no_progress_config
            )

        valid_results = {k: v for k, v in results.items() if v is not None}
        assert len(valid_results) >= len(_VALID_FORMULAS)

    def test_mixed_batch_invalid_are_none(
        self, no_progress_config: BatchConfig
    ) -> None:
        formulas = _VALID_FORMULAS + _INVALID_FORMULAS
        densities = _VALID_DENSITIES + _INVALID_DENSITIES

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = calculate_batch_properties(
                formulas, _ENERGIES, densities, config=no_progress_config
            )

        # Keys for failed formulas map to None
        for invalid_formula in _INVALID_FORMULAS:
            if invalid_formula in results:
                assert results[invalid_formula] is None

    def test_mixed_batch_emits_warnings_for_failures(
        self, no_progress_config: BatchConfig
    ) -> None:
        formulas = ["SiO2", *_INVALID_FORMULAS]
        densities = [2.2, *_INVALID_DENSITIES]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calculate_batch_properties(
                formulas, _ENERGIES, densities, config=no_progress_config
            )

        messages = [str(w.message) for w in caught]
        # At least one warning for the invalid formulas
        assert any(any(inv in m for m in messages) for inv in _INVALID_FORMULAS)

    def test_result_count_matches_input_count(
        self, no_progress_config: BatchConfig
    ) -> None:
        """The result dict has an entry for every formula submitted."""
        formulas = _VALID_FORMULAS + _INVALID_FORMULAS
        densities = _VALID_DENSITIES + _INVALID_DENSITIES

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = calculate_batch_properties(
                formulas, _ENERGIES, densities, config=no_progress_config
            )

        assert len(results) == len(formulas)

    def test_valid_result_has_correct_energy_shape(
        self, no_progress_config: BatchConfig
    ) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = calculate_batch_properties(
                ["SiO2"], _ENERGIES, [2.2], config=no_progress_config
            )

        valid_results = {k: v for k, v in results.items() if v is not None}
        assert len(valid_results) == 1
        result = next(iter(valid_results.values()))
        assert result is not None
        assert len(result.energy_kev) == len(_ENERGIES)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestBatchInputValidation:
    def test_empty_formula_list_raises(self, no_progress_config: BatchConfig) -> None:
        with pytest.raises(ValueError, match="empty"):
            calculate_batch_properties([], _ENERGIES, [], config=no_progress_config)

    def test_mismatched_lengths_raise(self, no_progress_config: BatchConfig) -> None:
        with pytest.raises(ValueError):
            calculate_batch_properties(
                ["SiO2", "Al2O3"], _ENERGIES, [2.2], config=no_progress_config
            )

    def test_invalid_energy_range_raises(self, no_progress_config: BatchConfig) -> None:
        bad_energies = np.array([100.0])  # out of [0.03, 30] keV range
        with pytest.raises(ValueError):
            calculate_batch_properties(
                ["SiO2"], bad_energies, [2.2], config=no_progress_config
            )

    def test_negative_energy_raises(self, no_progress_config: BatchConfig) -> None:
        bad_energies = np.array([-1.0])
        with pytest.raises(ValueError):
            calculate_batch_properties(
                ["SiO2"], bad_energies, [2.2], config=no_progress_config
            )
