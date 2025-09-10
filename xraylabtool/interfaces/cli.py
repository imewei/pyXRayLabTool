#!/usr/bin/env python3
"""
Command Line Interface for XRayLabTool.

This module provides a comprehensive CLI for calculating X-ray optical properties
of materials, including single material calculations, batch processing, utility
functions for X-ray analysis, and shell completion installation.

Available Commands:
    calc                Calculate X-ray properties for a single material
    batch               Process multiple materials from CSV file
    convert             Convert between energy and wavelength units
    formula             Parse and analyze chemical formulas
    atomic              Look up atomic data for elements
    bragg               Calculate Bragg angles for diffraction
    list                List available data and information
    install-completion  Install shell completion for xraylabtool

The CLI supports various output formats (table, CSV, JSON), field filtering,
precision control, and comprehensive shell completion for enhanced usability.
"""

import argparse
import json
from pathlib import Path
import sys
from textwrap import dedent
from typing import Any

import numpy as np
import pandas as pd

# Import the main XRayLabTool functionality
from xraylabtool import __version__
from xraylabtool.calculators.core import (
    XRayResult,
    calculate_single_material_properties,
)
from xraylabtool.utils import (
    bragg_angle,
    energy_to_wavelength,
    get_atomic_number,
    get_atomic_weight,
    parse_formula,
    wavelength_to_energy,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="xraylabtool",
        description="X-ray optical properties calculator for materials science",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """
        Examples:
          # Calculate properties for SiO2 at 10 keV
          xraylabtool calc SiO2 -e 10.0 -d 2.2

          # Energy sweep for silicon
          xraylabtool calc Si -e 5.0,10.0,15.0,20.0 -d 2.33 -o silicon_sweep.csv

          # Batch calculation from CSV file
          xraylabtool batch materials.csv -o results.csv

          # Convert energy to wavelength
          xraylabtool convert energy 10.0 --to wavelength

          # Parse chemical formula
          xraylabtool formula SiO2 --verbose

          # Install shell completion
          xraylabtool install-completion

        For more detailed help on specific commands, use:
          xraylabtool <command> --help
        """
        ),
    )

    parser.add_argument(
        "--version", action="version", version=f"XRayLabTool {__version__}"
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    # Add completion installation flags
    completion_group = parser.add_argument_group("completion installation")
    completion_group.add_argument(
        "--install-completion",
        nargs="?",
        const="auto",
        choices=["auto", "bash", "zsh", "fish", "powershell"],
        metavar="SHELL",
        help="Install shell completion for specified shell "
        "(auto-detects if not specified)",
    )
    completion_group.add_argument(
        "--test",
        action="store_true",
        help="Test completion installation (use with --install-completion)",
    )
    completion_group.add_argument(
        "--system",
        action="store_true",
        help="Install system-wide completion (use with --install-completion)",
    )
    completion_group.add_argument(
        "--uninstall",
        action="store_true",
        help="Uninstall completion (use with --install-completion)",
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", metavar="COMMAND"
    )

    # Add subcommands
    add_calc_command(subparsers)
    add_batch_command(subparsers)
    add_convert_command(subparsers)
    add_formula_command(subparsers)
    add_atomic_command(subparsers)
    add_bragg_command(subparsers)
    add_list_command(subparsers)
    add_install_completion_command(subparsers)
    add_uninstall_completion_command(subparsers)

    return parser


def add_calc_command(subparsers: Any) -> None:
    """Add the 'calc' subcommand for single material calculations."""
    parser = subparsers.add_parser(
        "calc",
        help="Calculate X-ray properties for a single material",
        description=(
            "Calculate X-ray optical properties for a single material composition"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """
        Examples:
          # Single energy calculation
          xraylabtool calc SiO2 -e 10.0 -d 2.2

          # Multiple energies (comma-separated)
          xraylabtool calc Si -e 5.0,10.0,15.0,20.0 -d 2.33

          # Energy range with linear spacing
          xraylabtool calc Al2O3 -e 5-15:11 -d 3.95

          # Energy range with log spacing
          xraylabtool calc C -e 1-30:100:log -d 3.52

          # Save results to file
          xraylabtool calc SiO2 -e 8.0,10.0,12.0 -d 2.2 -o results.csv

          # JSON output format
          xraylabtool calc Si -e 10.0 -d 2.33 -o results.json --format json
        """
        ),
    )

    parser.add_argument("formula", help="Chemical formula (e.g., SiO2, Al2O3, Fe2O3)")

    parser.add_argument(
        "-e",
        "--energy",
        required=True,
        help=dedent(
            """
        X-ray energy in keV. Formats:
        - Single value: 10.0
        - Comma-separated: 5.0,10.0,15.0
        - Range with count: 5-15:11 (11 points from 5 to 15 keV)
        - Log range: 1-30:100:log (100 log-spaced points)
        """
        ).strip(),
    )

    parser.add_argument(
        "-d", "--density", type=float, required=True, help="Material density in g/cm³"
    )

    parser.add_argument(
        "-o", "--output", help="Output filename (CSV or JSON based on extension)"
    )

    parser.add_argument(
        "--format",
        choices=["table", "csv", "json"],
        default="table",
        help="Output format (default: table)",
    )

    parser.add_argument(
        "--fields", help="Comma-separated list of fields to output (default: all)"
    )

    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Number of decimal places for output (default: 6)",
    )


def add_batch_command(subparsers: Any) -> None:
    """Add the 'batch' subcommand for processing multiple materials."""
    parser = subparsers.add_parser(
        "batch",
        help="Process multiple materials from CSV file",
        description="Calculate X-ray properties for multiple materials from CSV input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """
        Input CSV format:
        The input CSV file should have columns: formula, density, energy

        Example CSV content:
        formula,density,energy
        SiO2,2.2,10.0
        Al2O3,3.95,"5.0,10.0,15.0"
        Si,2.33,8.0

        Examples:
          # Process materials from CSV
          xraylabtool batch materials.csv -o results.csv

          # Specific output format
          xraylabtool batch materials.csv -o results.json --format json

          # Parallel processing with 4 workers
          xraylabtool batch materials.csv -o results.csv --workers 4
        """
        ),
    )

    parser.add_argument("input_file", help="Input CSV file with materials data")

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output filename (CSV or JSON based on extension)",
    )

    parser.add_argument(
        "--format",
        choices=["csv", "json"],
        help="Output format (auto-detected from extension if not specified)",
    )

    parser.add_argument(
        "--workers", type=int, help="Number of parallel workers (default: auto)"
    )

    parser.add_argument(
        "--fields", help="Comma-separated list of fields to include in output"
    )


def add_convert_command(subparsers: Any) -> None:
    """Add the 'convert' subcommand for unit conversions."""
    parser = subparsers.add_parser(
        "convert",
        help="Convert between energy and wavelength units",
        description="Convert between X-ray energy and wavelength units",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """
        Examples:
          # Convert energy to wavelength
          xraylabtool convert energy 10.0 --to wavelength

          # Convert wavelength to energy
          xraylabtool convert wavelength 1.24 --to energy

          # Multiple values
          xraylabtool convert energy 5.0,10.0,15.0 --to wavelength

          # Save to file
          xraylabtool convert energy 5.0,10.0,15.0 --to wavelength -o conversions.csv
        """
        ),
    )

    parser.add_argument(
        "from_unit", choices=["energy", "wavelength"], help="Input unit type"
    )

    parser.add_argument(
        "values", help="Value(s) to convert (comma-separated for multiple)"
    )

    parser.add_argument(
        "--to",
        dest="to_unit",
        choices=["energy", "wavelength"],
        required=True,
        help="Output unit type",
    )

    parser.add_argument("-o", "--output", help="Output filename (CSV format)")


def add_formula_command(subparsers: Any) -> None:
    """Add the 'formula' subcommand for formula parsing."""
    parser = subparsers.add_parser(
        "formula",
        help="Parse and analyze chemical formulas",
        description="Parse chemical formulas and show elemental composition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """
        Examples:
          # Parse a simple formula
          xraylabtool formula SiO2

          # Detailed information
          xraylabtool formula Al2O3 --verbose

          # Multiple formulas
          xraylabtool formula SiO2,Al2O3,Fe2O3

          # Save results to file
          xraylabtool formula SiO2,Al2O3 -o formulas.json
        """
        ),
    )

    parser.add_argument(
        "formulas", help="Chemical formula(s) (comma-separated for multiple)"
    )

    parser.add_argument("-o", "--output", help="Output filename (JSON format)")


def add_atomic_command(subparsers: Any) -> None:
    """Add the 'atomic' subcommand for atomic data lookup."""
    parser = subparsers.add_parser(
        "atomic",
        help="Look up atomic data for elements",
        description="Look up atomic numbers, weights, and other properties",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """
        Examples:
          # Single element
          xraylabtool atomic Si

          # Multiple elements
          xraylabtool atomic H,C,N,O,Si

          # Save to file
          xraylabtool atomic Si,Al,Fe -o atomic_data.csv
        """
        ),
    )

    parser.add_argument(
        "elements", help="Element symbol(s) (comma-separated for multiple)"
    )

    parser.add_argument(
        "-o", "--output", help="Output filename (CSV or JSON based on extension)"
    )


def add_bragg_command(subparsers: Any) -> None:
    """Add the 'bragg' subcommand for Bragg angle calculations."""
    parser = subparsers.add_parser(
        "bragg",
        help="Calculate Bragg angles for diffraction",
        description="Calculate Bragg diffraction angles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """
        Examples:
          # Single calculation
          xraylabtool bragg -d 3.14 -w 1.54 --order 1

          # Multiple d-spacings
          xraylabtool bragg -d 3.14,2.45,1.92 -w 1.54

          # Energy instead of wavelength
          xraylabtool bragg -d 3.14 -e 8.0
        """
        ),
    )

    parser.add_argument(
        "-d",
        "--dspacing",
        required=True,
        help="d-spacing in Angstroms (comma-separated for multiple)",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-w", "--wavelength", help="X-ray wavelength in Angstroms")
    group.add_argument("-e", "--energy", help="X-ray energy in keV")

    parser.add_argument(
        "--order", type=int, default=1, help="Diffraction order (default: 1)"
    )

    parser.add_argument("-o", "--output", help="Output filename (CSV format)")


def add_list_command(subparsers: Any) -> None:
    """Add the 'list' subcommand for listing available data."""
    parser = subparsers.add_parser(
        "list",
        help="List available data and information",
        description="List available elements, constants, or other information",
    )

    parser.add_argument(
        "type",
        choices=["constants", "fields", "examples"],
        help="Type of information to list",
    )


def add_install_completion_command(subparsers: Any) -> None:
    """Add the 'install-completion' subcommand for shell completion setup."""
    parser = subparsers.add_parser(
        "install-completion",
        help="Install shell completion for xraylabtool",
        description="Install shell completion for xraylabtool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """
        Examples:
          # Install completion for current shell (auto-detected)
          xraylabtool install-completion

          # Install for specific shell
          xraylabtool install-completion bash
          xraylabtool install-completion zsh
          xraylabtool install-completion fish

          # Install completion system-wide (requires sudo)
          xraylabtool install-completion --system

          # Test if completion is working
          xraylabtool install-completion --test

          # Uninstall completion
          xraylabtool install-completion --uninstall
        """
        ),
    )

    # Positional argument for shell type
    parser.add_argument(
        "shell",
        nargs="?",
        choices=["bash", "zsh", "fish", "powershell"],
        default=None,
        help="Shell type to install completion for (auto-detected if not specified)",
    )

    parser.add_argument(
        "--user",
        action="store_true",
        default=True,
        help="Install for current user only (default)",
    )

    parser.add_argument(
        "--system",
        action="store_true",
        help="Install system-wide (requires sudo privileges)",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Test if completion is working",
    )

    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Uninstall existing completion",
    )


def add_uninstall_completion_command(subparsers: Any) -> None:
    """Add the 'uninstall-completion' subcommand for shell completion removal."""
    parser = subparsers.add_parser(
        "uninstall-completion",
        help="Uninstall shell completion for xraylabtool",
        description="Remove shell completion functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """
        Examples:
          # Uninstall completion for current shell (auto-detected)
          xraylabtool uninstall-completion

          # Uninstall for specific shell
          xraylabtool uninstall-completion bash
          xraylabtool uninstall-completion zsh
          xraylabtool uninstall-completion fish

          # Uninstall system-wide completion (requires sudo)
          xraylabtool uninstall-completion --system

          # Clean up active session
          xraylabtool uninstall-completion --cleanup
        """
        ),
    )

    parser.add_argument(
        "shell_type",
        nargs="?",
        choices=["bash", "zsh", "fish", "powershell"],
        help="Shell type to remove completion from (auto-detected if not specified)",
    )

    parser.add_argument(
        "--user",
        action="store_true",
        default=True,
        help="Remove from current user only (default)",
    )

    parser.add_argument(
        "--system",
        action="store_true",
        help="Remove system-wide completion (requires sudo privileges)",
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up active shell session",
    )


def parse_energy_string(energy_str: str) -> np.ndarray:
    """Parse energy string into numpy array."""
    if "," in energy_str:
        # Comma-separated values
        return np.array([float(x.strip()) for x in energy_str.split(",")])
    elif "-" in energy_str and ":" in energy_str:
        # Range format: start-end:count or start-end:count:spacing
        parts = energy_str.split(":")
        range_part = parts[0]
        count = int(parts[1])
        spacing = parts[2] if len(parts) > 2 else "linear"

        start, end = map(float, range_part.split("-"))

        if spacing.lower() == "log":
            return np.logspace(np.log10(start), np.log10(end), count)
        else:
            return np.linspace(start, end, count)
    else:
        # Single value
        return np.array([float(energy_str)])


def _get_default_fields() -> tuple[list[str], list[str]]:
    """Get default scalar and array fields."""
    array_fields = [
        "energy_kev",
        "wavelength_angstrom",
        "dispersion_delta",
        "absorption_beta",
        "scattering_factor_f1",
        "scattering_factor_f2",
        "critical_angle_degrees",
        "attenuation_length_cm",
        "real_sld_per_ang2",
        "imaginary_sld_per_ang2",
    ]
    scalar_fields = [
        "formula",
        "molecular_weight_g_mol",
        "total_electrons",
        "density_g_cm3",
        "electron_density_per_ang3",
    ]
    return scalar_fields, array_fields


def _format_as_json(result: XRayResult, fields: list[str]) -> str:
    """Format result as JSON."""
    data = {}
    for field in fields:
        value = getattr(result, field)
        if isinstance(value, np.ndarray):
            data[field] = value.tolist()
        else:
            data[field] = value
    return json.dumps(data, indent=2)


def _format_as_csv(result: XRayResult, fields: list[str], precision: int) -> str:
    """Format result as CSV."""
    data_rows = []
    n_energies = len(result.energy_kev)

    for i in range(n_energies):
        row = {}
        for field in fields:
            value = getattr(result, field)
            if isinstance(value, np.ndarray):
                row[field] = round(value[i], precision)
            else:
                row[field] = value
        data_rows.append(row)

    if data_rows:
        df = pd.DataFrame(data_rows)
        csv_output: str = df.to_csv(index=False)
        return csv_output
    return ""


def _format_material_properties(result: XRayResult, precision: int) -> list[str]:
    """Format material properties section."""
    return [
        "Material Properties:",
        f"  Formula: {result.formula}",
        f"  Molecular Weight: {result.molecular_weight_g_mol: .{precision}f} g/mol",
        f"  Total Electrons: {result.total_electrons: .{precision}f}",
        f"  Density: {result.density_g_cm3: .{precision}f} g/cm³",
        (
            f"  Electron Density: {result.electron_density_per_ang3: .{precision}e} "
            "electrons/Å³"
        ),
        "",
    ]


def _format_single_energy(result: XRayResult, precision: int) -> list[str]:
    """Format single energy point properties."""
    return [
        "X-ray Properties:",
        f"  Energy: {result.energy_kev[0]:.{precision}f} keV",
        f"  Wavelength: {result.wavelength_angstrom[0]:.{precision}f} Å",
        f"  Dispersion (δ): {result.dispersion_delta[0]:.{precision}e}",
        f"  Absorption (β): {result.absorption_beta[0]:.{precision}e}",
        f"  Scattering f1: {result.scattering_factor_f1[0]:.{precision}f}",
        f"  Scattering f2: {result.scattering_factor_f2[0]:.{precision}f}",
        f"  Critical Angle: {result.critical_angle_degrees[0]:.{precision}f}°",
        f"  Attenuation Length: {result.attenuation_length_cm[0]:.{precision}f} cm",
        f"  Real SLD: {result.real_sld_per_ang2[0]:.{precision}e} Å⁻²",
        f"  Imaginary SLD: {result.imaginary_sld_per_ang2[0]:.{precision}e} Å⁻²",
    ]


def _format_multiple_energies(result: XRayResult, precision: int) -> list[str]:
    """Format multiple energy points as table."""
    output_lines = ["X-ray Properties (tabular):"]

    df_data = {
        "Energy (keV)": result.energy_kev,
        "λ (Å)": result.wavelength_angstrom,
        "δ": result.dispersion_delta,
        "β": result.absorption_beta,
        "f1": result.scattering_factor_f1,
        "f2": result.scattering_factor_f2,
        "θc (°)": result.critical_angle_degrees,
        "μ (cm)": result.attenuation_length_cm,
    }

    df = pd.DataFrame(df_data)
    pd.set_option("display.float_format", f"{{: .{precision}g}}".format)
    table_str = df.to_string(index=False)
    output_lines.append(table_str)

    return output_lines


def _format_scalar_field(field: str, value: Any, precision: int) -> str:
    """Format a single scalar field."""
    from collections.abc import Callable

    def default_formatter(v: Any, p: int) -> str:  # noqa: ARG001
        return ""

    formatters: dict[str, Callable[[Any, int], str]] = {
        "formula": lambda v, _: f"  Formula: {v}",
        "molecular_weight_g_mol": lambda v, p: f"  Molecular Weight: {v: .{p}f} g/mol",
        "total_electrons": lambda v, p: f"  Total Electrons: {v: .{p}f}",
        "density_g_cm3": lambda v, p: f"  Density: {v: .{p}f} g/cm³",
        "electron_density_per_ang3": (
            lambda v, p: f"  Electron Density: {v: .{p}e} electrons/Å³"
        ),  # noqa: E501
    }
    formatter = formatters.get(field, default_formatter)
    return formatter(value, precision)


def _format_array_field_single(field: str, value: float, precision: int) -> str:
    """Format a single array field for single energy point."""
    formatters = {
        "energy_kev": (f"  Energy: {{: .{precision}f}} keV", "f"),
        "wavelength_angstrom": (f"  Wavelength: {{: .{precision}f}} Å", "f"),
        "dispersion_delta": (f"  Dispersion (δ): {{: .{precision}e}}", "e"),
        "absorption_beta": (f"  Absorption (β): {{: .{precision}e}}", "e"),
        "scattering_factor_f1": (f"  Scattering f1: {{: .{precision}f}}", "f"),
        "scattering_factor_f2": (f"  Scattering f2: {{: .{precision}f}}", "f"),
        "critical_angle_degrees": (f"  Critical Angle: {{: .{precision}f}}°", "f"),
        "attenuation_length_cm": (f"  Attenuation Length: {{: .{precision}f}} cm", "f"),
        "real_sld_per_ang2": (f"  Real SLD: {{: .{precision}e}} Å⁻²", "e"),
        "imaginary_sld_per_ang2": (f"  Imaginary SLD: {{: .{precision}e}} Å⁻²", "e"),
    }

    if field in formatters:
        template, _ = formatters[field]
        return template.format(value)
    return ""


def _get_field_labels() -> dict[str, str]:
    """Get mapping of field names to display labels."""
    return {
        "energy_kev": "Energy (keV)",
        "wavelength_angstrom": "λ (Å)",
        "dispersion_delta": "δ",
        "absorption_beta": "β",
        "scattering_factor_f1": "f1",
        "scattering_factor_f2": "f2",
        "critical_angle_degrees": "θc (°)",
        "attenuation_length_cm": "μ (cm)",
        "real_sld_per_ang2": "Real SLD",
        "imaginary_sld_per_ang2": "Imag SLD",
    }


def _format_scalar_fields_section(
    result: XRayResult, fields_to_show: list[str], precision: int
) -> list[str]:
    """Format scalar fields section."""
    if not fields_to_show:
        return []

    output_lines = ["Material Properties:"]
    for field in fields_to_show:
        value = getattr(result, field)
        line = _format_scalar_field(field, value, precision)
        if line:
            output_lines.append(line)
    output_lines.append("")
    return output_lines


def _format_single_energy_section(
    result: XRayResult, fields_to_show: list[str], precision: int
) -> list[str]:
    """Format single energy point array fields."""
    if not fields_to_show:
        return []

    output_lines = ["X-ray Properties:"]
    for field in fields_to_show:
        value = getattr(result, field)[0]
        line = _format_array_field_single(field, value, precision)
        if line:
            output_lines.append(line)
    return output_lines


def _format_multiple_energy_section(
    result: XRayResult, fields_to_show: list[str], precision: int
) -> list[str]:
    """Format multiple energy points as tabular data."""
    if not fields_to_show:
        return []

    output_lines = ["X-ray Properties (tabular):"]
    field_labels = _get_field_labels()
    df_data = {}

    for field in fields_to_show:
        label = field_labels.get(field, field)
        df_data[label] = getattr(result, field)

    if df_data:
        df = pd.DataFrame(df_data)
        format_str = f"{{: .{precision}g}}"
        pd.set_option("display.float_format", format_str.format)
        table_str = df.to_string(index=False)
        output_lines.append(table_str)

    return output_lines


def _format_filtered_table(
    result: XRayResult, fields: list[str], precision: int
) -> str:
    """Format table with only specified fields."""
    # Separate scalar and array fields
    scalar_fields, array_fields = _get_default_fields()
    scalar_fields_to_show = [f for f in fields if f in scalar_fields]
    array_fields_to_show = [f for f in fields if f in array_fields]

    output_lines = []

    # Add scalar fields section
    output_lines.extend(
        _format_scalar_fields_section(result, scalar_fields_to_show, precision)
    )

    # Add array fields section
    if array_fields_to_show:
        if len(result.energy_kev) == 1:
            output_lines.extend(
                _format_single_energy_section(result, array_fields_to_show, precision)
            )
        else:
            output_lines.extend(
                _format_multiple_energy_section(result, array_fields_to_show, precision)
            )

    return "\n".join(output_lines)


def format_xray_result(
    result: XRayResult,
    format_type: str,
    precision: int = 6,
    fields: list[str] | None = None,
) -> str:
    """Format XRayResult for output."""
    if fields is None:
        scalar_fields, array_fields = _get_default_fields()
        fields = scalar_fields + array_fields

    if format_type == "json":
        return _format_as_json(result, fields)
    elif format_type == "csv":
        return _format_as_csv(result, fields, precision)
    else:  # table format
        # For table format with custom fields, use a filtered output
        if fields != _get_default_fields()[0] + _get_default_fields()[1]:
            return _format_filtered_table(result, fields, precision)

        # Default table format (all fields)
        output_lines = _format_material_properties(result, precision)

        if len(result.energy_kev) == 1:
            output_lines.extend(_format_single_energy(result, precision))
        else:
            output_lines.extend(_format_multiple_energies(result, precision))

        return "\n".join(output_lines)


def _validate_calc_inputs(args: Any, energies: np.ndarray) -> bool:
    """Validate calc command inputs."""
    if args.density <= 0:
        print("Error: Density must be positive", file=sys.stderr)
        return False

    if np.any(energies <= 0):
        print("Error: All energies must be positive", file=sys.stderr)
        return False

    if np.any(energies < 0.03) or np.any(energies > 30):
        print("Warning: Energy values outside typical X-ray range (0.03-30 keV)")

    return True


def _print_calc_verbose_info(args: Any, energies: np.ndarray) -> None:
    """Print verbose calculation information."""
    print(f"Calculating X-ray properties for {args.formula}...")
    print(
        f"Energy range: {energies.min(): .3f} - {energies.max(): .3f} keV "
        f"({len(energies)} points)"
    )
    print(f"Density: {args.density} g/cm³")
    print()


def _determine_output_format(args: Any) -> str:
    """Determine output format based on args and file extension."""
    output_format: str = args.format

    if args.output:
        output_path = Path(args.output)
        if not output_format or output_format == "table":
            if output_path.suffix.lower() == ".json":
                output_format = "json"
            elif output_path.suffix.lower() == ".csv":
                output_format = "csv"

    return output_format


def _save_or_print_output(formatted_output: str, args: Any) -> None:
    """Save output to file or print to stdout."""
    if args.output:
        Path(args.output).write_text(formatted_output)
        if args.verbose:
            print(f"Results saved to {args.output}")
    else:
        print(formatted_output)


def cmd_calc(args: Any) -> int:
    """Handle the 'calc' command."""
    try:
        energies = parse_energy_string(args.energy)

        if not _validate_calc_inputs(args, energies):
            return 1

        if args.verbose:
            _print_calc_verbose_info(args, energies)

        result = calculate_single_material_properties(
            args.formula, energies, args.density
        )

        fields = None
        if args.fields:
            fields = [field.strip() for field in args.fields.split(",")]

        output_format = _determine_output_format(args)
        formatted_output = format_xray_result(
            result, output_format, args.precision, fields
        )

        _save_or_print_output(formatted_output, args)
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _validate_batch_input(args: Any) -> pd.DataFrame | None:
    """Validate batch input file and columns."""
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file {args.input_file} not found", file=sys.stderr)
        return None

    df_input = pd.read_csv(input_path)

    required_columns = ["formula", "density", "energy"]
    missing_columns = [col for col in required_columns if col not in df_input.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}", file=sys.stderr)
        return None

    return df_input


def _parse_batch_data(
    df_input: pd.DataFrame,
) -> tuple[list[str] | None, list[float] | None, list[list[float]] | None]:
    """Parse batch data from DataFrame."""
    formulas = []
    densities = []
    energy_sets = []

    for _, row in df_input.iterrows():
        formulas.append(row["formula"])
        densities.append(float(row["density"]))

        energy_str = str(row["energy"])
        try:
            if "," in energy_str:
                energies = [float(x.strip()) for x in energy_str.split(",")]
            else:
                energies = [float(energy_str)]
            energy_sets.append(energies)
        except ValueError:
            print(
                f"Error: Invalid energy format for {row['formula']}: {energy_str}",
                file=sys.stderr,
            )
            return None, None, None

    return formulas, densities, energy_sets


def _convert_result_to_dict(result: XRayResult, energy_index: int) -> dict[str, Any]:
    """Convert XRayResult to dictionary for specific energy point."""
    return {
        "formula": result.formula,
        "density_g_cm3": result.density_g_cm3,
        "energy_kev": result.energy_kev[energy_index],
        "wavelength_angstrom": result.wavelength_angstrom[energy_index],
        "molecular_weight_g_mol": result.molecular_weight_g_mol,
        "total_electrons": result.total_electrons,
        "electron_density_per_ang3": result.electron_density_per_ang3,
        "dispersion_delta": result.dispersion_delta[energy_index],
        "absorption_beta": result.absorption_beta[energy_index],
        "scattering_factor_f1": result.scattering_factor_f1[energy_index],
        "scattering_factor_f2": result.scattering_factor_f2[energy_index],
        "critical_angle_degrees": result.critical_angle_degrees[energy_index],
        "attenuation_length_cm": result.attenuation_length_cm[energy_index],
        "real_sld_per_ang2": result.real_sld_per_ang2[energy_index],
        "imaginary_sld_per_ang2": result.imaginary_sld_per_ang2[energy_index],
    }


def _process_batch_materials(
    formulas: list[str],
    densities: list[float],
    energy_sets: list[list[float]],
    args: Any,
) -> list[dict[str, Any]]:
    """Process all materials and return results."""
    results = []

    if args.verbose:
        print(f"Processing {len(formulas)} materials...")

    for i, (formula, density, energies) in enumerate(
        zip(formulas, densities, energy_sets, strict=False)
    ):
        try:
            if args.verbose:
                print(f"  {i + 1}/{len(formulas)}: {formula}")

            result = calculate_single_material_properties(formula, energies, density)

            for j, _energy in enumerate(energies):
                result_dict = _convert_result_to_dict(result, j)
                results.append(result_dict)

        except Exception as e:
            print(f"Warning: Failed to process {formula}: {e}")
            continue

    return results


def _save_batch_results(results: list[dict[str, Any]], args: Any) -> None:
    """Save batch results to output file."""
    if args.fields:
        field_list = [field.strip() for field in args.fields.split(",")]
        results = [
            {k: v for k, v in result.items() if k in field_list} for result in results
        ]

    output_format = args.format
    output_path = Path(args.output)
    if not output_format:
        output_format = "json" if output_path.suffix.lower() == ".json" else "csv"

    if output_format == "json":
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
    else:
        df_output = pd.DataFrame(results)
        df_output.to_csv(args.output, index=False)

    if args.verbose:
        print(f"Results saved to {args.output}")
        print(
            f"Processed {len(results)} data points from "
            f"{len({r['formula'] for r in results})} unique materials"
        )


def cmd_batch(args: Any) -> int:
    """Handle the 'batch' command."""
    try:
        df_input = _validate_batch_input(args)
        if df_input is None:
            return 1

        parsed_data = _parse_batch_data(df_input)
        if parsed_data[0] is None:
            return 1

        formulas, densities, energy_sets = parsed_data
        assert (
            formulas is not None and densities is not None and energy_sets is not None
        )
        results = _process_batch_materials(formulas, densities, energy_sets, args)

        if not results:
            print("Error: No materials were successfully processed", file=sys.stderr)
            return 1

        _save_batch_results(results, args)
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_convert(args: Any) -> int:
    """Handle the 'convert' command."""
    try:
        # Parse values
        values = [float(x.strip()) for x in args.values.split(",")]

        # Perform conversion
        if args.from_unit == "energy" and args.to_unit == "wavelength":
            converted = [energy_to_wavelength(v) for v in values]
            unit_label = "Å"
        elif args.from_unit == "wavelength" and args.to_unit == "energy":
            converted = [wavelength_to_energy(v) for v in values]
            unit_label = "keV"
        else:
            print(
                f"Error: Cannot convert from {args.from_unit} to {args.to_unit}",
                file=sys.stderr,
            )
            return 1

        # Format output
        if args.output:
            # Save to CSV
            df = pd.DataFrame(
                {
                    f"{args.from_unit}": values,
                    f"{args.to_unit} ({unit_label})": converted,
                }
            )
            df.to_csv(args.output, index=False)
            print(f"Conversion results saved to {args.output}")
        else:
            # Print to console
            print(f"{args.from_unit.title()} to {args.to_unit.title()} Conversion:")
            print("-" * 40)
            for original, converted_val in zip(values, converted, strict=False):
                print(f"{original: >10.4f} → {converted_val: >10.4f} {unit_label}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _get_atomic_data(elements: list[str]) -> list[dict[str, Any]]:
    """Get atomic data for list of elements."""
    atomic_data = []
    for element in elements:
        try:
            atomic_data.append(
                {
                    "element": element,
                    "atomic_number": get_atomic_number(element),
                    "atomic_weight": get_atomic_weight(element),
                }
            )
        except Exception as e:
            print(f"Warning: Could not get atomic data for {element}: {e}")
    return atomic_data


def _process_formula(formula: str, verbose: bool) -> dict[str, Any]:
    """Process a single formula and return info."""
    elements, counts = parse_formula(formula)

    formula_info = {
        "formula": formula,
        "elements": elements,
        "counts": counts,
        "element_count": len(elements),
        "total_atoms": sum(counts),
    }

    if verbose:
        formula_info["atomic_data"] = _get_atomic_data(elements)

    return formula_info


def _output_formula_results(results: list[dict[str, Any]], args: Any) -> None:
    """Output formula results to file or console."""
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Formula analysis saved to {args.output}")
    else:
        _print_formula_results(results, args.verbose)


def _print_formula_results(results: list[dict[str, Any]], verbose: bool) -> None:
    """Print formula results to console."""
    for result in results:
        print(f"Formula: {result['formula']}")
        print(f"Elements: {', '.join(result['elements'])}")
        print(f"Counts: {', '.join(map(str, result['counts']))}")
        print(f"Total atoms: {result['total_atoms']}")

        if verbose and "atomic_data" in result:
            print("Atomic data:")
            for atom_data in result["atomic_data"]:
                print(
                    f"  {atom_data['element']: >2}: "
                    f"Z={atom_data['atomic_number']: >3}, "
                    f"MW={atom_data['atomic_weight']: >8.3f}"
                )
        print()


def cmd_formula(args: Any) -> int:
    """Handle the 'formula' command."""
    try:
        formulas = [f.strip() for f in args.formulas.split(",")]
        results = []

        for formula in formulas:
            try:
                formula_info = _process_formula(formula, args.verbose)
                results.append(formula_info)
            except Exception as e:
                print(f"Error parsing formula {formula}: {e}", file=sys.stderr)
                continue

        _output_formula_results(results, args)
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_atomic(args: Any) -> int:
    """Handle the 'atomic' command."""
    try:
        elements = [e.strip() for e in args.elements.split(",")]
        results = []

        for element in elements:
            try:
                atomic_number = get_atomic_number(element)
                atomic_weight = get_atomic_weight(element)

                element_data = {
                    "element": element,
                    "atomic_number": atomic_number,
                    "atomic_weight": atomic_weight,
                }
                results.append(element_data)

            except Exception as e:
                print(f"Error getting atomic data for {element}: {e}", file=sys.stderr)
                continue

        if not results:
            print("No valid elements found", file=sys.stderr)
            return 1

        # Output results
        if args.output:
            output_path = Path(args.output)
            if output_path.suffix.lower() == ".json":
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
            else:  # CSV
                df = pd.DataFrame(results)
                df.to_csv(args.output, index=False)
            print(f"Atomic data saved to {args.output}")
        else:
            print("Atomic Data:")
            print("-" * 30)
            print(f"{'Element': >8} {'Z': >3} {'MW (u)': >10}")
            print("-" * 30)
            for data in results:
                print(
                    f"{data['element']: >8} {data['atomic_number']: >3} "
                    f"{data['atomic_weight']: >10.3f}"
                )

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_bragg(args: Any) -> int:
    """Handle the 'bragg' command."""
    try:
        # Parse d-spacings
        d_spacings = [float(x.strip()) for x in args.dspacing.split(",")]

        # Determine wavelength
        if args.wavelength:
            wavelength = float(args.wavelength)
        else:  # args.energy
            energy = float(args.energy)
            wavelength = energy_to_wavelength(energy)

        # Calculate Bragg angles
        results = []
        for d_spacing in d_spacings:
            try:
                angle = bragg_angle(d_spacing, wavelength, args.order)
                results.append(
                    {
                        "d_spacing_angstrom": d_spacing,
                        "wavelength_angstrom": wavelength,
                        "order": args.order,
                        "bragg_angle_degrees": angle,
                        "two_theta_degrees": 2 * angle,
                    }
                )
            except Exception as e:
                print(
                    f"Warning: Could not calculate Bragg angle for d={d_spacing}: {e}"
                )
                continue

        if not results:
            print("No valid Bragg angles calculated", file=sys.stderr)
            return 1

        # Output results
        if args.output:
            df = pd.DataFrame(results)
            df.to_csv(args.output, index=False)
            print(f"Bragg angle results saved to {args.output}")
        else:
            print("Bragg Angle Calculations:")
            print("-" * 50)
            print(f"{'d (Å)': >8} {'θ (°)': >8} {'2θ (°)': >8}")
            print("-" * 50)
            for result in results:
                print(
                    f"{result['d_spacing_angstrom']: >8.3f} "
                    f"{result['bragg_angle_degrees']: >8.3f} "
                    f"{result['two_theta_degrees']: >8.3f}"
                )

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_list(args: Any) -> int:
    """Handle the 'list' command."""
    if args.type == "constants":
        print("Physical Constants:")
        print("=" * 40)
        from xraylabtool import constants

        const_names = [
            "THOMPSON",
            "SPEED_OF_LIGHT",
            "PLANCK",
            "ELEMENT_CHARGE",
            "AVOGADRO",
            "ENERGY_TO_WAVELENGTH_FACTOR",
            "PI",
            "TWO_PI",
        ]
        for name in const_names:
            if hasattr(constants, name):
                value = getattr(constants, name)
                print(f"{name: <25}: {value}")

    elif args.type == "fields":
        print("Available XRayResult Fields (new snake_case names):")
        print("=" * 60)
        field_descriptions = [
            ("formula", "Chemical formula string"),
            ("molecular_weight_g_mol", "Molecular weight (g/mol)"),
            ("total_electrons", "Total electrons per molecule"),
            ("density_g_cm3", "Mass density (g/cm³)"),
            ("electron_density_per_ang3", "Electron density (electrons/Å³)"),
            ("energy_kev", "X-ray energies (keV)"),
            ("wavelength_angstrom", "X-ray wavelengths (Å)"),
            ("dispersion_delta", "Dispersion coefficient δ"),
            ("absorption_beta", "Absorption coefficient β"),
            ("scattering_factor_f1", "Real atomic scattering factor"),
            ("scattering_factor_f2", "Imaginary atomic scattering factor"),
            ("critical_angle_degrees", "Critical angles (degrees)"),
            ("attenuation_length_cm", "Attenuation lengths (cm)"),
            ("real_sld_per_ang2", "Real SLD (Å⁻²)"),
            ("imaginary_sld_per_ang2", "Imaginary SLD (Å⁻²)"),
        ]

        for field, description in field_descriptions:
            print(f"{field: <25}: {description}")

    elif args.type == "examples":
        print("CLI Usage Examples:")
        print("=" * 40)
        examples = [
            ("Single material calculation", "xraylabtool calc SiO2 -e 10.0 -d 2.2"),
            ("Multiple energies", "xraylabtool calc Si -e 5.0,10.0,15.0 -d 2.33"),
            ("Energy range", "xraylabtool calc Al2O3 -e 5-15:11 -d 3.95"),
            ("Save to CSV", "xraylabtool calc SiO2 -e 10.0 -d 2.2 -o results.csv"),
            ("Batch processing", "xraylabtool batch materials.csv -o results.csv"),
            ("Unit conversion", "xraylabtool convert energy 10.0 --to wavelength"),
            ("Formula parsing", "xraylabtool formula SiO2 --verbose"),
            ("Bragg angles", "xraylabtool bragg -d 3.14 -e 8.0"),
            ("Install completion", "xraylabtool install-completion"),
        ]

        for description, command in examples:
            print(f"\n{description}:")
            print(f"  {command}")

    return 0


def cmd_install_completion(args: Any) -> int:
    """Handle the 'install-completion' command."""
    from .completion import install_completion_main

    return install_completion_main(args)


def cmd_uninstall_completion(args: Any) -> int:
    """Handle the 'uninstall-completion' command."""
    from ..completion_installer import uninstall_completion_main

    return uninstall_completion_main(args)


def main() -> int:
    """Execute the main CLI application."""
    parser = create_parser()

    try:
        args = parser.parse_args()
    except SystemExit as e:
        # Handle argparse sys.exit calls gracefully in tests
        if e.code == 0:  # --help or --version
            raise  # Re-raise for normal help/version behavior
        else:
            # Invalid arguments - return error code instead of exiting
            return 1

    # Handle --install-completion flag before checking for subcommands
    if hasattr(args, "install_completion") and args.install_completion is not None:
        from .completion import install_completion_main

        # Create a mock args object that matches the install-completion
        # subcommand format
        class MockArgs:
            def __init__(
                self,
                shell_type: str | None,
                test: bool = False,
                system: bool = False,
                uninstall: bool = False,
            ) -> None:
                self.shell = shell_type if shell_type != "auto" else None
                self.system = system
                # user installation is default unless system is specified
                self.user = not system
                self.uninstall = uninstall
                self.test = test

        mock_args = MockArgs(
            args.install_completion,
            test=getattr(args, "test", False),
            system=getattr(args, "system", False),
            uninstall=getattr(args, "uninstall", False),
        )
        return install_completion_main(mock_args)

    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return 1

    # Route to appropriate command handler
    command_handlers = {
        "calc": cmd_calc,
        "batch": cmd_batch,
        "convert": cmd_convert,
        "formula": cmd_formula,
        "atomic": cmd_atomic,
        "bragg": cmd_bragg,
        "list": cmd_list,
        "install-completion": cmd_install_completion,
        "uninstall-completion": cmd_uninstall_completion,
    }

    handler = command_handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
