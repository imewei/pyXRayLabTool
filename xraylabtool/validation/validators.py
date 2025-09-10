"""
Validation functions for XRayLabTool.

This module contains functions for validating input parameters,
chemical formulas, energy ranges, and other data used in calculations.
"""

import re

import numpy as np

from xraylabtool.exceptions import EnergyError, FormulaError, ValidationError


def validate_energy_range(
    energies: float | np.ndarray,
    min_energy: float = 0.1,
    max_energy: float = 100.0,
) -> np.ndarray:
    """
    Validate X-ray energy values.

    Args:
        energies: Energy value(s) in keV
        min_energy: Minimum allowed energy in keV
        max_energy: Maximum allowed energy in keV

    Returns:
        Validated energy array

    Raises:
        EnergyError: If energies are invalid
    """
    energy_array = np.asarray(energies)

    # Check for NaN or infinite values
    if np.any(~np.isfinite(energy_array)):
        raise EnergyError(
            "Energy values must be finite", valid_range=f"{min_energy}-{max_energy} keV"
        )

    # Check for positive values
    if np.any(energy_array <= 0):
        raise EnergyError(
            "Energy values must be positive",
            valid_range=f"{min_energy}-{max_energy} keV",
        )

    # Check energy range
    if np.any(energy_array < min_energy):
        raise EnergyError(
            "Energy below minimum allowed value",
            energy=float(energy_array.min()),
            valid_range=f"{min_energy}-{max_energy} keV",
        )

    if np.any(energy_array > max_energy):
        raise EnergyError(
            "Energy above maximum allowed value",
            energy=float(energy_array.max()),
            valid_range=f"{min_energy}-{max_energy} keV",
        )

    return energy_array


def validate_chemical_formula(formula: str) -> dict[str, float]:
    """
    Validate and parse a chemical formula.

    Args:
        formula: Chemical formula string (e.g., "SiO2", "Ca0.5Sr0.5TiO3")

    Returns:
        Dictionary mapping element symbols to their quantities

    Raises:
        FormulaError: If formula is invalid
    """
    if not formula or not isinstance(formula, str):
        raise FormulaError("Formula must be a non-empty string", formula)

    # Remove whitespace and validate basic format
    formula = formula.strip()
    if not formula:
        raise FormulaError("Formula cannot be empty", formula)

    # Check for valid characters (letters, numbers, dots, parentheses)
    if not re.match(r"^[A-Za-z0-9().]+$", formula):
        raise FormulaError(
            "Formula contains invalid characters. Use only element symbols, "
            "numbers, dots (for fractional amounts), and parentheses",
            formula,
        )

    # Parse the formula
    try:
        elements = _parse_formula(formula)

        # Validate that we found at least one element
        if not elements:
            raise FormulaError("No elements found in formula", formula)

        # Validate element symbols (basic check)
        valid_elements = _get_valid_element_symbols()
        for element in elements:
            if element not in valid_elements:
                raise FormulaError(f"Unknown element symbol: {element}", formula)

        return elements

    except Exception as e:
        if isinstance(e, FormulaError):
            raise
        raise FormulaError(f"Error parsing formula: {str(e)}", formula) from e


def validate_density(
    density: float, min_density: float = 0.001, max_density: float = 30.0
) -> float:
    """
    Validate material density value.

    Args:
        density: Density in g/cm³
        min_density: Minimum allowed density
        max_density: Maximum allowed density

    Returns:
        Validated density value

    Raises:
        ValidationError: If density is invalid
    """
    if not isinstance(density, int | float | np.number):
        raise ValidationError(
            "Density must be a numeric value", parameter="density", value=density
        )

    if not np.isfinite(density):
        raise ValidationError(
            "Density must be finite", parameter="density", value=density
        )

    if density <= 0:
        raise ValidationError(
            "Density must be positive", parameter="density", value=density
        )

    if density < min_density:
        raise ValidationError(
            f"Density below minimum allowed value ({min_density} g/cm³)",
            parameter="density",
            value=density,
        )

    if density > max_density:
        raise ValidationError(
            f"Density above maximum allowed value ({max_density} g/cm³)",
            parameter="density",
            value=density,
        )

    return float(density)


def validate_calculation_parameters(
    formula: str, energies: float | np.ndarray, density: float
) -> tuple[str, np.ndarray, float]:
    """
    Validate all parameters for X-ray calculations.

    Args:
        formula: Chemical formula
        energies: Energy values in keV
        density: Material density in g/cm³

    Returns:
        Tuple of validated (formula, energies, density)

    Raises:
        ValidationError: If any parameters are invalid
    """
    # Validate formula
    validate_chemical_formula(formula)

    # Validate energies
    validated_energies = validate_energy_range(energies)

    # Validate density
    validated_density = validate_density(density)

    return formula, validated_energies, validated_density


def _parse_formula(formula: str) -> dict[str, float]:
    """Parse chemical formula into elements and quantities."""
    # This is a simplified parser - in production, would use more robust parsing
    elements: dict[str, float] = {}

    # Simple regex to match element symbols and their quantities
    pattern = r"([A-Z][a-z]?)(\d*\.?\d*)"
    matches = re.findall(pattern, formula)

    for element, count_str in matches:
        count = float(count_str) if count_str else 1.0
        elements[element] = elements.get(element, 0.0) + count

    return elements


def _get_valid_element_symbols() -> set[str]:
    """Get set of valid element symbols."""
    # Simplified list - in production, would use complete periodic table
    return {
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
    }
