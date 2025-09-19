#!/usr/bin/env python3
"""
Command Line Interface for XRayLabTool.

This module provides a comprehensive CLI for calculating X-ray optical properties
of materials, including single material calculations, batch processing, utility
functions for X-ray analysis, and shell completion installation.

Available Commands:
    calc                Calculate X-ray properties for a single material
    batch               Process multiple materials from CSV file
    compare             Compare X-ray properties between multiple materials
    analyze             Advanced analysis of single material properties
    export              Export data with advanced formatting and visualization
    convert             Convert between energy and wavelength units
    formula             Parse and analyze chemical formulas
    atomic              Look up atomic data for elements
    bragg               Calculate Bragg angles for diffraction
    list                List available data and information
    install-completion  Install shell completion for xraylabtool
    uninstall-completion Remove shell completion for xraylabtool

The CLI supports various output formats (table, CSV, JSON), field filtering,
precision control, and comprehensive shell completion for enhanced usability.
"""

import argparse
import json
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any

# Essential imports only - heavy modules imported lazily in functions
# pandas import moved to function level to reduce startup time
from xraylabtool import __version__

# These basic utilities are lightweight and used frequently
from xraylabtool.utils import (
    bragg_angle,
    energy_to_wavelength,
    get_atomic_number,
    get_atomic_weight,
    parse_formula,
    wavelength_to_energy,
)

# Heavy imports moved to lazy loading:
# - numpy, pandas: imported when needed for data processing
# - analysis modules: imported in cmd_analyze, cmd_compare functions

# Import monitoring and performance classes used in batch processing
try:
    from xraylabtool.data_handling.batch_processing import AdaptiveChunkSizer
    from xraylabtool.data_handling.memory_profiler import MemoryMonitor
    from xraylabtool.optimization.regression_detector import PerformanceMetrics
    from xraylabtool.progress import create_batch_progress_tracker
except ImportError:
    # Fallback implementations for missing modules
    class MemoryMonitor:
        def __init__(self):
            pass

    class PerformanceMetrics:
        def __init__(self):
            pass

    class AdaptiveChunkSizer:
        def __init__(self):
            pass

    def create_batch_progress_tracker(**kwargs):
        from contextlib import nullcontext

        return nullcontext()


# - export modules: imported in cmd_export function
# - progress modules: imported in cmd_batch function
# - validation modules: imported when needed


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

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for detailed error information",
    )

    # Add completion installation flags
    completion_group = parser.add_argument_group("completion installation")
    completion_group.add_argument(
        "--install-completion",
        nargs="?",
        const="auto",
        choices=["auto", "bash", "zsh", "fish", "powershell"],
        metavar="SHELL",
        help=(
            "Install shell completion for specified shell "
            "(auto-detects if not specified)"
        ),
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
    add_compare_command(subparsers)
    add_analyze_command(subparsers)
    add_export_command(subparsers)
    add_convert_command(subparsers)
    add_formula_command(subparsers)
    add_atomic_command(subparsers)
    add_bragg_command(subparsers)
    add_list_command(subparsers)
    add_completion_command(subparsers)
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

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for detailed error information",
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

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for detailed error information",
    )

    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bar during batch processing",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar (overrides --progress)",
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

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for detailed error information",
    )


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


def add_completion_command(subparsers: Any) -> None:
    """Add the 'completion' subcommand for the new completion system."""
    parser = subparsers.add_parser(
        "completion",
        help="Manage virtual environment-centric shell completion",
        description=(
            "Manage shell completion that activates/deactivates with virtual"
            " environments"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """
        Examples:
          # Install completion in current virtual environment
          xraylabtool completion install

          # Install for specific shell
          xraylabtool completion install --shell zsh

          # List all environments with completion status
          xraylabtool completion list

          # Show completion status for current environment
          xraylabtool completion status

          # Uninstall from current environment
          xraylabtool completion uninstall

          # Uninstall from all environments
          xraylabtool completion uninstall --all

          # Show system information
          xraylabtool completion info

        The new completion system:
          • Installs per virtual environment (no system-wide changes)
          • Automatically activates/deactivates with environment
          • Supports venv, conda, Poetry, Pipenv environments
          • Provides native completion for multiple shells
        """
        ),
    )

    # Forward all arguments to the completion subcommand handler
    parser.set_defaults(completion_subcommand=True)


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


def add_compare_command(subparsers: Any) -> None:
    """Add the 'compare' subcommand for material comparison."""
    parser = subparsers.add_parser(
        "compare",
        help="Compare X-ray properties between multiple materials",
        description=(
            "Compare X-ray optical properties across multiple materials with"
            " side-by-side analysis"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """
        Examples:
          # Compare two materials at single energy
          xraylabtool compare SiO2,2.2 Al2O3,3.95 -e 10.0

          # Compare materials across energy range
          xraylabtool compare Si,2.33 Ge,5.32 -e 5-15:11

          # Compare specific properties
          xraylabtool compare SiO2,2.2 Si3N4,3.2 -e 8.0,10.0,12.0 --properties dispersion_delta,absorption_beta

          # Save comparison to file
          xraylabtool compare SiO2,2.2 Al2O3,3.95 -e 10.0 -o comparison.csv

          # Generate detailed report
          xraylabtool compare Si,2.33 GaAs,5.32 -e 10.0 --report --output comparison_report.txt
        """
        ),
    )

    parser.add_argument(
        "materials",
        nargs="+",
        help="Materials in format 'formula,density' (e.g., SiO2,2.2 Al2O3,3.95)",
    )

    parser.add_argument(
        "-e",
        "--energy",
        required=True,
        help="X-ray energy in keV (single value, comma-separated, or range format)",
    )

    parser.add_argument(
        "--properties",
        help=(
            "Comma-separated list of properties to compare (default: all standard"
            " properties)"
        ),
    )

    parser.add_argument("-o", "--output", help="Output filename for comparison results")

    parser.add_argument(
        "--format",
        choices=["table", "csv", "json"],
        default="table",
        help="Output format (default: table)",
    )

    parser.add_argument(
        "--report", action="store_true", help="Generate detailed comparison report"
    )

    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Number of decimal places for output (default: 6)",
    )


def add_analyze_command(subparsers: Any) -> None:
    """Add the 'analyze' subcommand for advanced single material analysis."""
    parser = subparsers.add_parser(
        "analyze",
        help="Advanced analysis of single material properties",
        description=(
            "Perform advanced analysis including edge detection, energy optimization,"
            " and property analysis"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """
        Examples:
          # Detect absorption edges
          xraylabtool analyze SiO2 -d 2.2 --edges -e 1-20:1000

          # Optimize energy for maximum critical angle
          xraylabtool analyze Si -d 2.33 --optimize critical_angle_degrees -e 5-15:100

          # Find energy for specific transmission
          xraylabtool analyze Al -d 2.7 --transmission 0.5 --thickness 10.0 -e 1-30:200

          # Comprehensive analysis with all features
          xraylabtool analyze SiO2 -d 2.2 --edges --optimize dispersion_delta -e 1-25:500 -o analysis.json

          # Statistical analysis of energy scan
          xraylabtool analyze GaAs -d 5.32 --stats -e 5-15:100 --output stats_report.txt
        """
        ),
    )

    parser.add_argument("formula", help="Chemical formula (e.g., SiO2, Al2O3, GaAs)")

    parser.add_argument(
        "-d", "--density", type=float, required=True, help="Material density in g/cm³"
    )

    parser.add_argument(
        "-e",
        "--energy",
        required=True,
        help="Energy range for analysis in keV (range format recommended: 1-30:1000)",
    )

    parser.add_argument(
        "--edges",
        action="store_true",
        help="Detect absorption edges using f2 derivative analysis",
    )

    parser.add_argument(
        "--optimize",
        help=(
            "Optimize energy for specific property (e.g., critical_angle_degrees,"
            " dispersion_delta)"
        ),
    )

    parser.add_argument(
        "--transmission",
        type=float,
        help="Find energy for target transmission (0-1, requires --thickness)",
    )

    parser.add_argument(
        "--thickness",
        type=float,
        help="Material thickness in micrometers (for transmission analysis)",
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Perform statistical analysis of properties across energy range",
    )

    parser.add_argument("-o", "--output", help="Output filename for analysis results")

    parser.add_argument(
        "--format",
        choices=["table", "csv", "json", "report"],
        default="table",
        help="Output format (default: table)",
    )

    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Number of decimal places for output (default: 6)",
    )


def add_export_command(subparsers: Any) -> None:
    """Add the 'export' subcommand for advanced data export and visualization."""
    parser = subparsers.add_parser(
        "export",
        help="Export data with advanced formatting and visualization",
        description=(
            "Export X-ray calculation results with professional formatting, plots, and"
            " reports"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """
        Examples:
          # Export calculation results to Excel with charts
          xraylabtool calc SiO2 -e 5-15:11 -d 2.2 | xraylabtool export --format excel --template material_characterization -o report.xlsx

          # Create HTML report with interactive plots
          xraylabtool export data.csv --format html --template research_report -o report.html --plots

          # Export to multiple formats using template
          xraylabtool export results.json --template publication_data -o publication_data --formats csv,excel,pdf

          # Convert between data formats
          xraylabtool export data.xlsx --format json -o converted_data.json

          # Generate plots only
          xraylabtool export data.csv --plots-only --plot-types line,scatter -o plots/
        """
        ),
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        help="Input data file (if not provided, reads from stdin)",
    )

    parser.add_argument(
        "-o", "--output", required=True, help="Output file or directory path"
    )

    parser.add_argument(
        "--format",
        choices=["csv", "json", "excel", "html", "pdf", "hdf5", "xml"],
        help="Output format (auto-detected from output path if not specified)",
    )

    parser.add_argument(
        "--formats",
        help="Comma-separated list of output formats for multi-format export",
    )

    parser.add_argument(
        "--template",
        choices=[
            "material_characterization",
            "beamline_optimization",
            "publication_data",
            "quality_control",
            "research_report",
            "comparison_study",
            "energy_scan_analysis",
            "absorption_edge_study",
        ],
        help="Export template for predefined workflows",
    )

    parser.add_argument("--plots", action="store_true", help="Include plots in export")

    parser.add_argument(
        "--plots-only", action="store_true", help="Generate plots only (no data export)"
    )

    parser.add_argument(
        "--plot-types",
        help=(
            "Comma-separated list of plot types (line, scatter, comparison,"
            " energy_scan, etc.)"
        ),
    )

    parser.add_argument(
        "--excel-template",
        choices=["basic", "comparison", "analysis", "report", "publication"],
        default="basic",
        help="Excel template style (default: basic)",
    )

    parser.add_argument(
        "--html-template",
        choices=["scientific", "executive", "technical", "presentation"],
        default="scientific",
        help="HTML report template (default: scientific)",
    )

    parser.add_argument(
        "--include-metadata",
        action="store_true",
        default=True,
        help="Include metadata in export (default: true)",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate interactive plots (requires plotly)",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Plot DPI for high-quality output (default: 300)",
    )

    parser.add_argument("--style", help="Plot style (matplotlib style name)")

    parser.add_argument("--title", help="Custom title for plots and reports")

    parser.add_argument("--author", help="Author name for reports")

    parser.add_argument("--institution", help="Institution name for reports")


def parse_energy_string(energy_str: str):
    """Parse energy string and return numpy array."""
    import numpy as np

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


def _format_as_json(result, fields: list[str]) -> str:
    """Format result as JSON."""
    import numpy as np

    data = {}
    for field in fields:
        value = getattr(result, field)
        if isinstance(value, np.ndarray):
            data[field] = value.tolist()
        else:
            data[field] = value
    return json.dumps(data, indent=2)


def _format_as_csv(result, fields: list[str], precision: int) -> str:
    """Format result as CSV."""
    import csv
    import io

    import numpy as np

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
        # Use CSV module instead of pandas
        output = io.StringIO()
        fieldnames = fields
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_rows)
        return output.getvalue()
    return ""


def _format_material_properties(result, precision: int) -> list[str]:
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


def _format_single_energy(result, precision: int) -> list[str]:
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


def _format_multiple_energies(result, precision: int) -> list[str]:
    """Format multiple energy points as table."""
    import numpy as np

    output_lines = ["X-ray Properties (tabular):"]

    # Create table without pandas
    headers = ["Energy (keV)", "λ (Å)", "δ", "β", "f1", "f2", "θc (°)", "μ (cm)"]
    data_arrays = [
        result.energy_kev,
        result.wavelength_angstrom,
        result.dispersion_delta,
        result.absorption_beta,
        result.scattering_factor_f1,
        result.scattering_factor_f2,
        result.critical_angle_degrees,
        result.attenuation_length_cm,
    ]

    # Calculate column widths
    col_widths = [max(len(header), 12) for header in headers]

    # Format header
    header_line = "  ".join(
        header.ljust(width) for header, width in zip(headers, col_widths, strict=False)
    )
    output_lines.append(header_line)

    # Format data rows
    n_energies = len(result.energy_kev)
    for i in range(n_energies):
        row_values = []
        for data_array in data_arrays:
            if isinstance(data_array, np.ndarray):
                value = data_array[i]
            else:
                value = data_array
            row_values.append(f"{value:.{precision}g}")

        row_line = "  ".join(
            val.ljust(width) for val, width in zip(row_values, col_widths, strict=False)
        )
        output_lines.append(row_line)

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
        "electron_density_per_ang3": lambda v, p: (
            f"  Electron Density: {v: .{p}e} electrons/Å³"
        ),
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
    result, fields_to_show: list[str], precision: int
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
    result, fields_to_show: list[str], precision: int
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
    result, fields_to_show: list[str], precision: int
) -> list[str]:
    """Format multiple energy points as tabular data."""
    import numpy as np

    if not fields_to_show:
        return []

    output_lines = ["X-ray Properties (tabular):"]
    field_labels = _get_field_labels()

    # Collect headers and data arrays
    headers = []
    data_arrays = []
    for field in fields_to_show:
        label = field_labels.get(field, field)
        headers.append(label)
        data_arrays.append(getattr(result, field))

    if headers:
        # Calculate column widths
        col_widths = [max(len(header), 12) for header in headers]

        # Format header
        header_line = "  ".join(
            header.ljust(width)
            for header, width in zip(headers, col_widths, strict=False)
        )
        output_lines.append(header_line)

        # Format data rows
        n_rows = len(data_arrays[0]) if data_arrays else 0
        for i in range(n_rows):
            row_values = []
            for data_array in data_arrays:
                if isinstance(data_array, np.ndarray):
                    value = data_array[i]
                else:
                    value = data_array
                row_values.append(f"{value:.{precision}g}")

            row_line = "  ".join(
                val.ljust(width)
                for val, width in zip(row_values, col_widths, strict=False)
            )
            output_lines.append(row_line)

    return output_lines


def _format_filtered_table(result, fields: list[str], precision: int) -> str:
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
    result,  # XRayResult - type hint removed for lazy loading
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


def _validate_calc_inputs(args: Any, energies) -> bool:
    """Validate calculation inputs."""
    import numpy as np

    if args.density <= 0:
        print("Error: Density must be positive", file=sys.stderr)
        return False

    if np.any(energies <= 0):
        print("Error: All energies must be positive", file=sys.stderr)
        return False

    if np.any(energies < 0.03) or np.any(energies > 30):
        print("Warning: Energy values outside typical X-ray range (0.03-30 keV)")

    return True


def _print_calc_verbose_info(args: Any, energies) -> None:
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
        # Lazy imports for this command
        from xraylabtool.calculators.core import calculate_single_material_properties
        from xraylabtool.validation import validate_chemical_formula, validate_density

        # Basic validation
        try:
            validate_chemical_formula(args.formula)
        except Exception as e:
            print(
                f"Error: Invalid chemical formula '{args.formula}': {e}",
                file=sys.stderr,
            )
            return 1

        try:
            validate_density(args.density)
        except Exception as e:
            print(f"Error: Invalid density '{args.density}': {e}", file=sys.stderr)
            return 1

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
        debug_mode = getattr(args, "debug", False)
        if debug_mode:
            import traceback

            print("🔍 Debug: Full traceback:", file=sys.stderr)
            traceback.print_exc()
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _validate_batch_input(args: Any):
    """Validate batch input file and return data."""
    import csv

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file {args.input_file} not found", file=sys.stderr)
        return None

    try:
        # Read CSV using standard library
        with open(input_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data_rows = list(reader)

        if not data_rows:
            print("Error: Input file is empty", file=sys.stderr)
            return None

        # Check for required columns
        required_columns = ["formula", "density", "energy"]
        actual_columns = set(data_rows[0].keys()) if data_rows else set()
        missing_columns = [col for col in required_columns if col not in actual_columns]
        if missing_columns:
            print(
                f"Error: Missing required columns: {missing_columns}", file=sys.stderr
            )
            return None

        return data_rows
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        return None


def _parse_batch_data(
    data_input,  # list of dict
) -> tuple[list[str] | None, list[float] | None, list[list[float]] | None]:
    """Parse batch data from list of dictionaries."""
    formulas = []
    densities = []
    energy_sets = []

    for row in data_input:
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


def _convert_result_to_dict(result, energy_index: int) -> dict[str, Any]:
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
    """Process all materials and return results with progress tracking."""
    # Import required calculation function
    from xraylabtool.calculators.core import calculate_single_material_properties

    results = []

    # Initialize progress tracking and performance monitoring
    enable_progress = getattr(args, "progress", False) and not getattr(
        args, "no_progress", False
    )
    # Auto-enable progress for large batches unless explicitly disabled
    if len(formulas) > 10 and not getattr(args, "no_progress", False):
        enable_progress = True

    # Initialize monitoring
    memory_monitor = MemoryMonitor()
    performance_metrics = PerformanceMetrics()
    chunk_sizer = AdaptiveChunkSizer()

    if args.verbose:
        print(f"Processing {len(formulas)} materials...")
        if enable_progress:
            print("Progress tracking enabled")

    # Create progress tracker
    with create_batch_progress_tracker(
        total_items=len(formulas),
        desc="Processing materials",
        verbose=args.verbose,
        disable_progress=not enable_progress,
    ) as progress:
        for i, (formula, density, energies) in enumerate(
            zip(formulas, densities, energy_sets, strict=False)
        ):
            try:
                # Update memory monitoring
                memory_monitor.update()

                # Time the operation for performance metrics
                with performance_metrics.time_operation():
                    if args.verbose and not enable_progress:
                        print(f"  {i + 1}/{len(formulas)}: {formula}")

                    result = calculate_single_material_properties(
                        formula, energies, density
                    )

                    for j, _energy in enumerate(energies):
                        result_dict = _convert_result_to_dict(result, j)
                        results.append(result_dict)

                # Record the operation
                performance_metrics.record_operations(len(energies))

            except Exception as e:
                if not enable_progress:  # Only print if progress bar isn't showing
                    print(f"Warning: Failed to process {formula}: {e}")
                continue

            finally:
                # Update progress
                progress.update(1)

    # Show performance summary if verbose
    if args.verbose:
        print("\n" + "=" * 50)
        performance_metrics.print_summary(verbose=True)
        memory_monitor.print_summary()

        # Show chunk sizing recommendation for future runs
        recommended_chunk = chunk_sizer.calculate_chunk_size(len(formulas))
        if len(formulas) > recommended_chunk:
            print(
                "💡 For optimal memory usage, consider processing in chunks of"
                f" {recommended_chunk}"
            )

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
        # Write CSV without pandas
        import csv

        if results:
            with open(args.output, "w", newline="", encoding="utf-8") as f:
                fieldnames = results[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

    if args.verbose:
        print(f"Results saved to {args.output}")
        print(
            f"Processed {len(results)} data points from "
            f"{len({r['formula'] for r in results})} unique materials"
        )


def cmd_batch(args: Any) -> int:
    """Handle the 'batch' command."""
    try:
        # Lazy imports for batch processing
        from xraylabtool.validation.enhanced_validator import EnhancedValidator
        from xraylabtool.validation.error_recovery import ErrorRecoveryManager

        df_input = _validate_batch_input(args)
        if df_input is None:
            return 1

        # Initialize enhanced error handling for batch processing
        debug_mode = getattr(args, "debug", False)
        validator = EnhancedValidator(debug=debug_mode)
        recovery_manager = ErrorRecoveryManager(
            validator, interactive=False
        )  # Non-interactive for batch

        # Validate all formulas in the batch
        formulas = df_input["formula"].tolist()
        validation_results = validator.validate_batch_formulas(
            formulas, command_context="batch"
        )

        # Try to recover from validation errors
        recovered_formulas = recovery_manager.recover_batch_errors(
            validation_results, "batch processing", fail_fast=False
        )

        # Update the dataframe with recovered formulas
        for i, (original_formula, recovered_formula) in enumerate(
            zip(formulas, recovered_formulas, strict=False)
        ):
            if recovered_formula and recovered_formula != original_formula:
                df_input.loc[i, "formula"] = recovered_formula
                if args.verbose:
                    print(
                        f"✅ Auto-corrected formula {i + 1}: '{original_formula}' →"
                        f" '{recovered_formula}'"
                    )
            elif not recovered_formula:
                if args.verbose:
                    print(
                        f"⚠️  Could not process formula {i + 1}: '{original_formula}' -"
                        " skipping"
                    )

        # Generate batch improvement suggestions
        batch_suggestions = recovery_manager.suggest_batch_improvements(
            validation_results
        )
        if batch_suggestions["status"] == "errors_found":
            if args.verbose or debug_mode:
                print("\n📊 Batch Processing Summary:")
                print(f"   Total items: {batch_suggestions['summary']['total_items']}")
                print(
                    f"   Success rate: {batch_suggestions['summary']['success_rate']}"
                )
                if batch_suggestions["recommendations"]:
                    print("   Recommendations:")
                    for rec in batch_suggestions["recommendations"]:
                        print(f"   • {rec}")
                print()

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

        # Show recovery statistics if in verbose or debug mode
        if args.verbose or debug_mode:
            recovery_stats = recovery_manager.get_recovery_stats()
            if recovery_stats["total_errors"] > 0:
                print("\n📈 Error Recovery Statistics:")
                print(f"   Total errors encountered: {recovery_stats['total_errors']}")
                print(f"   Auto-recovery rate: {recovery_stats['auto_recovery_rate']}")
                print(
                    "   Overall recovery rate:"
                    f" {recovery_stats['overall_recovery_rate']}"
                )

        return 0

    except Exception as e:
        debug_mode = getattr(args, "debug", False)
        if debug_mode:
            import traceback

            print("🔍 Debug: Full traceback:", file=sys.stderr)
            traceback.print_exc()
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
            import csv

            with open(args.output, "w", newline="", encoding="utf-8") as f:
                fieldnames = [f"{args.from_unit}", f"{args.to_unit} ({unit_label})"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for val, conv in zip(values, converted, strict=False):
                    writer.writerow({fieldnames[0]: val, fieldnames[1]: conv})
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
                # Basic validation
                from xraylabtool.validation import validate_chemical_formula

                validate_chemical_formula(formula)

                formula_info = _process_formula(formula, args.verbose)
                results.append(formula_info)

            except Exception as e:
                print(f"Error parsing formula {formula}: {e}", file=sys.stderr)
                if len(formulas) == 1:
                    return 1
                continue

        if not results:
            print("No valid formulas were processed", file=sys.stderr)
            return 1

        _output_formula_results(results, args)
        return 0

    except Exception as e:
        debug_mode = getattr(args, "debug", False)
        if debug_mode:
            import traceback

            print("🔍 Debug: Full traceback:", file=sys.stderr)
            traceback.print_exc()
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
                import csv

                if results:
                    with open(args.output, "w", newline="", encoding="utf-8") as f:
                        fieldnames = results[0].keys()
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(results)
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
            import csv

            if results:
                with open(args.output, "w", newline="", encoding="utf-8") as f:
                    fieldnames = results[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results)
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
    from xraylabtool.interfaces.completion import install_completion_main

    return install_completion_main(args)


def cmd_uninstall_completion(args: Any) -> int:
    """Handle the 'uninstall-completion' command."""
    from xraylabtool.interfaces.completion import uninstall_completion_main

    return uninstall_completion_main(args)


def cmd_completion(args: Any) -> int:
    """Handle the 'completion' command for the new completion system."""
    # Extract the remaining arguments after 'completion'
    import sys

    from xraylabtool.interfaces.completion_v2.cli import completion_main

    remaining_args = sys.argv[2:]  # Skip 'xraylabtool' and 'completion'

    return completion_main(remaining_args)


def cmd_compare(args: Any) -> int:
    """Handle the 'compare' command for material comparison."""
    try:
        # Lazy imports for comparison functionality
        from xraylabtool.analysis import MaterialComparator

        # Parse materials input
        materials = []
        formulas = []
        densities = []

        for material_str in args.materials:
            try:
                parts = material_str.split(",")
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid material format: {material_str}. Expected"
                        " 'formula,density'"
                    )

                formula = parts[0].strip()
                density = float(parts[1].strip())

                formulas.append(formula)
                densities.append(density)
                materials.append((formula, density))

            except ValueError as e:
                print(f"Error parsing material '{material_str}': {e}", file=sys.stderr)
                return 1

        if len(materials) < 2:
            print(
                "Error: At least two materials required for comparison", file=sys.stderr
            )
            return 1

        # Parse energies
        try:
            energies = parse_energy_string(args.energy).tolist()
        except Exception as e:
            print(f"Error parsing energy range: {e}", file=sys.stderr)
            return 1

        # Parse properties
        properties = None
        if args.properties:
            properties = [prop.strip() for prop in args.properties.split(",")]

        # Perform comparison
        comparator = MaterialComparator()

        try:
            result = comparator.compare_materials(
                formulas=formulas,
                densities=densities,
                energies=energies,
                properties=properties,
            )
        except Exception as e:
            print(f"Error during comparison: {e}", file=sys.stderr)
            return 1

        # Generate output
        if args.report or args.format == "report":
            report = comparator.generate_comparison_report(result)

            if args.output:
                with open(args.output, "w") as f:
                    f.write(report)
                print(f"Comparison report saved to {args.output}")
            else:
                print(report)

        else:
            # Create comparison table
            table = comparator.create_comparison_table(result)

            if args.output:
                output_path = Path(args.output)
                if output_path.suffix.lower() == ".json":
                    # Convert to JSON format
                    output_data = {
                        "materials": result.materials,
                        "energies": result.energies,
                        "properties": result.properties,
                        "data": result.data,
                        "summary_stats": result.summary_stats,
                        "recommendations": result.recommendations,
                    }
                    with open(args.output, "w") as f:
                        json.dump(output_data, f, indent=2)
                else:  # CSV
                    table.to_csv(args.output, index=False)
                print(f"Comparison results saved to {args.output}")
            # Print table to console
            elif args.format == "json":
                output_data = {
                    "materials": result.materials,
                    "energies": result.energies,
                    "properties": result.properties,
                    "data": result.data,
                    "summary_stats": result.summary_stats,
                    "recommendations": result.recommendations,
                }
                print(json.dumps(output_data, indent=2))
            elif args.format == "csv":
                print(table.to_csv(index=False))
            else:  # table
                print(table.to_string(index=False))

        return 0

    except Exception as e:
        print(f"Comparison failed: {e}", file=sys.stderr)
        if hasattr(args, "debug") and args.debug:
            import traceback

            traceback.print_exc()
        return 1


def cmd_analyze(args: Any) -> int:
    """Handle the 'analyze' command for advanced material analysis."""
    try:
        # Lazy imports for analysis functionality
        from xraylabtool.analysis import (
            AbsorptionEdgeDetector,
            EnergyOptimizer,
            StatisticalAnalyzer,
        )

        # Parse energies
        try:
            energies = parse_energy_string(args.energy)
        except Exception as e:
            print(f"Error parsing energy range: {e}", file=sys.stderr)
            return 1

        if len(energies) < 10:
            print(
                "Warning: Analysis works best with dense energy sampling (>=100"
                " points)",
                file=sys.stderr,
            )

        results = {}

        # Absorption edge detection
        if args.edges:
            print("Detecting absorption edges...")
            detector = AbsorptionEdgeDetector()

            try:
                edges = detector.detect_edges(
                    formula=args.formula,
                    density=args.density,
                    energy_range=(float(energies.min()), float(energies.max())),
                    energy_points=len(energies),
                )

                results["edges"] = []
                for edge in edges:
                    edge_data = {
                        "energy_kev": edge.energy_kev,
                        "element": edge.element,
                        "edge_type": edge.edge_type,
                        "strength": edge.strength,
                        "confidence": edge.confidence,
                    }
                    if edge.theoretical_energy:
                        edge_data["theoretical_energy"] = edge.theoretical_energy
                    results["edges"].append(edge_data)

                if args.format == "report":
                    print(detector.generate_edge_report(edges))
                else:
                    print(f"Found {len(edges)} absorption edges")

            except Exception as e:
                print(f"Edge detection failed: {e}", file=sys.stderr)

        # Energy optimization
        if args.optimize:
            print(f"Optimizing energy for {args.optimize}...")
            optimizer = EnergyOptimizer()

            try:
                opt_result = optimizer.optimize_for_property(
                    formula=args.formula,
                    density=args.density,
                    property_name=args.optimize,
                    energy_range=(float(energies.min()), float(energies.max())),
                    optimization_type="maximize",
                )

                results["optimization"] = {
                    "optimal_energy": opt_result.optimal_energy,
                    "optimal_value": opt_result.optimal_value,
                    "property": opt_result.property_name,
                    "type": opt_result.optimization_type,
                }

                if args.format == "report":
                    print(optimizer.generate_optimization_report(opt_result))
                else:
                    print(
                        f"Optimal energy for {args.optimize}:"
                        f" {opt_result.optimal_energy:.3f} keV"
                    )
                    print(f"Optimal value: {opt_result.optimal_value:.6g}")

            except Exception as e:
                print(f"Energy optimization failed: {e}", file=sys.stderr)

        # Transmission analysis
        if args.transmission is not None:
            if args.thickness is None:
                print(
                    "Error: --thickness required for transmission analysis",
                    file=sys.stderr,
                )
                return 1

            print(
                f"Finding energy for {args.transmission * 100}% transmission through"
                f" {args.thickness} µm..."
            )
            optimizer = EnergyOptimizer()

            try:
                trans_result = optimizer.find_optimal_transmission_energy(
                    formula=args.formula,
                    density=args.density,
                    thickness_um=args.thickness,
                    target_transmission=args.transmission,
                    energy_range=(float(energies.min()), float(energies.max())),
                )

                results["transmission"] = {
                    "target_transmission": args.transmission,
                    "thickness_um": args.thickness,
                    "optimal_energy": trans_result.optimal_energy,
                    "actual_transmission": trans_result.optimal_value,
                }

                print(f"Optimal energy: {trans_result.optimal_energy:.3f} keV")
                print(f"Actual transmission: {trans_result.optimal_value:.1%}")

            except Exception as e:
                print(f"Transmission analysis failed: {e}", file=sys.stderr)

        # Statistical analysis
        if args.stats:
            print("Performing statistical analysis...")
            optimizer = EnergyOptimizer()

            try:
                # Scan properties across energy range
                property_names = [
                    "dispersion_delta",
                    "absorption_beta",
                    "critical_angle_degrees",
                    "attenuation_length_cm",
                    "scattering_factor_f1",
                    "scattering_factor_f2",
                ]

                scan_data = optimizer.scan_energy_range(
                    formula=args.formula,
                    density=args.density,
                    energy_range=(float(energies.min()), float(energies.max())),
                    property_names=property_names,
                    num_points=len(energies),
                )

                # Perform statistical analysis
                analyzer = StatisticalAnalyzer()
                stats_summary = analyzer.analyze_batch_results(
                    scan_data, material_column="energies", exclude_columns=["energies"]
                )

                results["statistics"] = {
                    "total_points": stats_summary.total_materials,
                    "properties_analyzed": stats_summary.properties_analyzed,
                    "summary_stats": stats_summary.summary_statistics,
                }

                if args.format == "report":
                    print(analyzer.create_summary_report(stats_summary))
                else:
                    print(
                        "Statistical analysis completed for"
                        f" {len(property_names)} properties"
                    )

            except Exception as e:
                print(f"Statistical analysis failed: {e}", file=sys.stderr)

        # Save results if output specified
        if args.output and results:
            output_path = Path(args.output)

            if args.format == "json" or output_path.suffix.lower() == ".json":
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
            else:
                # Generate comprehensive text report
                with open(args.output, "w") as f:
                    f.write("ADVANCED ANALYSIS REPORT\n")
                    f.write(f"{'=' * 50}\n\n")
                    f.write(f"Material: {args.formula}\n")
                    f.write(f"Density: {args.density} g/cm³\n")
                    f.write(
                        f"Energy Range: {energies.min():.1f} -"
                        f" {energies.max():.1f} keV\n\n"
                    )

                    for analysis_type, data in results.items():
                        f.write(f"{analysis_type.upper()} RESULTS\n")
                        f.write(f"{'-' * 30}\n")
                        f.write(f"{json.dumps(data, indent=2)}\n\n")

            print(f"Analysis results saved to {args.output}")

        return 0

    except Exception as e:
        print(f"Analysis failed: {e}", file=sys.stderr)
        if hasattr(args, "debug") and args.debug:
            import traceback

            traceback.print_exc()
        return 1


def cmd_export(args: Any) -> int:
    """Handle the 'export' command for advanced data export and visualization."""
    try:
        # Lazy imports for export functionality - these are the heaviest modules
        from xraylabtool.export import (
            DataExporter,
            ExcelConfiguration,
            ExcelExporter,
            ExcelTemplate,
            ExportFormat,
            ExportTemplateManager,
            HTMLReporter,
            PlotConfiguration,
            PlotGenerator,
            PlotType,
            ReportTemplate,
            TemplateType,
        )

        # Determine input source
        if args.input_file:
            input_path = Path(args.input_file)
            if not input_path.exists():
                print(
                    f"Error: Input file '{args.input_file}' not found", file=sys.stderr
                )
                return 1

            # Read input data based on file extension
            try:
                if input_path.suffix.lower() == ".csv":
                    import csv

                    with open(input_path, newline="", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        data = list(reader)
                elif input_path.suffix.lower() == ".json":
                    with open(input_path) as f:
                        data = json.load(f)
                elif input_path.suffix.lower() in [".xlsx", ".xls"]:
                    print(
                        "Error: Excel files (.xlsx/.xls) not supported without pandas. "
                        "Please convert to CSV format.",
                        file=sys.stderr,
                    )
                    return 1
                else:
                    print(
                        f"Error: Unsupported input format '{input_path.suffix}'",
                        file=sys.stderr,
                    )
                    return 1
            except Exception as e:
                print(f"Error reading input file: {e}", file=sys.stderr)
                return 1
        else:
            # Read from stdin
            try:
                stdin_content = sys.stdin.read().strip()
                if not stdin_content:
                    print("Error: No input data provided", file=sys.stderr)
                    return 1

                # Try to parse as JSON first
                try:
                    data = json.loads(stdin_content)
                except json.JSONDecodeError:
                    # Try to parse as CSV
                    import csv
                    from io import StringIO

                    reader = csv.DictReader(StringIO(stdin_content))
                    data = list(reader)
            except Exception as e:
                print(f"Error reading input data: {e}", file=sys.stderr)
                return 1

        # Handle template-based export
        if args.template:
            try:
                template_manager = ExportTemplateManager()
                template_type = TemplateType(args.template)
                template = template_manager.get_template(template_type)

                # Apply template and export
                if args.output:
                    output_base = Path(args.output).stem
                    output_dir = Path(args.output).parent
                else:
                    output_base = "export_output"
                    output_dir = Path.cwd()

                results = template_manager.apply_template(
                    data=data,
                    template=template,
                    output_base_path=output_dir / output_base,
                    metadata=getattr(args, "metadata", None),
                )

                print("Template export completed:")
                for format_name, file_path in results.items():
                    print(f"  {format_name}: {file_path}")

                return 0

            except ValueError:
                print(f"Error: Invalid template '{args.template}'", file=sys.stderr)
                print(
                    f"Available templates: {', '.join([t.value for t in TemplateType])}"
                )
                return 1
            except Exception as e:
                print(f"Template export failed: {e}", file=sys.stderr)
                return 1

        # Handle plots-only export
        if args.plots_only:
            try:
                plot_generator = PlotGenerator()

                # Determine plot types
                if args.plot_types:
                    plot_types = [
                        PlotType(t.strip()) for t in args.plot_types.split(",")
                    ]
                else:
                    plot_types = [PlotType.LINE, PlotType.SCATTER]

                # Generate plots
                output_dir = Path(args.output) if args.output else Path.cwd() / "plots"
                output_dir.mkdir(exist_ok=True)

                for plot_type in plot_types:
                    config = PlotConfiguration(
                        plot_type=plot_type, interactive=args.interactive_plots
                    )

                    plot_path = plot_generator.generate_plot(
                        data=data,
                        config=config,
                        output_path=(
                            output_dir / f"{plot_type.value}_plot.html"
                            if args.interactive_plots
                            else output_dir / f"{plot_type.value}_plot.png"
                        ),
                    )
                    print(f"Generated plot: {plot_path}")

                return 0

            except Exception as e:
                print(f"Plot generation failed: {e}", file=sys.stderr)
                return 1

        # Handle standard format export
        if args.format:
            try:
                # Single format export
                if args.format == "excel":
                    exporter = ExcelExporter()
                    template = (
                        ExcelTemplate(args.excel_template)
                        if args.excel_template
                        else ExcelTemplate.SCIENTIFIC
                    )
                    config = ExcelConfiguration(
                        template=template,
                        include_charts=args.excel_charts,
                        auto_formatting=True,
                    )

                    output_path = args.output or "export_output.xlsx"
                    result_path = exporter.export_to_excel(
                        data=data,
                        output_path=output_path,
                        config=config,
                        metadata=getattr(args, "metadata", None),
                    )
                    print(f"Excel export completed: {result_path}")

                elif args.format == "html":
                    reporter = HTMLReporter()
                    template = (
                        ReportTemplate(args.html_template)
                        if args.html_template
                        else ReportTemplate.SCIENTIFIC
                    )

                    output_path = args.output or "report.html"
                    result_path = reporter.generate_report(
                        data=data,
                        output_path=output_path,
                        template=template,
                        metadata=getattr(args, "metadata", None),
                        include_interactive_plots=args.interactive_plots,
                    )
                    print(f"HTML report generated: {result_path}")

                else:
                    # Use DataExporter for other formats
                    exporter = DataExporter()
                    export_format = ExportFormat(args.format)

                    output_path = args.output or f"export_output.{args.format}"
                    result = exporter.export_data(
                        data=data,
                        output_path=output_path,
                        format=export_format,
                        metadata=getattr(args, "metadata", None),
                        compression=args.compression,
                    )

                    if result.success:
                        print(f"Export completed: {result.output_path}")
                        if result.metadata:
                            print(
                                "Records exported:"
                                f" {result.metadata.get('record_count', 'N/A')}"
                            )
                    else:
                        print(f"Export failed: {result.error_message}", file=sys.stderr)
                        return 1

                return 0

            except ValueError:
                print(f"Error: Invalid format '{args.format}'", file=sys.stderr)
                print(
                    f"Available formats: {', '.join([f.value for f in ExportFormat])}"
                )
                return 1
            except Exception as e:
                print(f"Export failed: {e}", file=sys.stderr)
                return 1

        # Handle multiple formats export
        if args.formats:
            try:
                format_list = [f.strip() for f in args.formats.split(",")]
                output_base = Path(args.output).stem if args.output else "export_output"
                output_dir = Path(args.output).parent if args.output else Path.cwd()

                exporter = DataExporter()

                for format_name in format_list:
                    try:
                        export_format = ExportFormat(format_name)
                        output_path = output_dir / f"{output_base}.{format_name}"

                        result = exporter.export_data(
                            data=data,
                            output_path=output_path,
                            format=export_format,
                            metadata=getattr(args, "metadata", None),
                        )

                        if result.success:
                            print(f"Exported {format_name}: {result.output_path}")
                        else:
                            print(
                                f"Failed to export {format_name}:"
                                f" {result.error_message}",
                                file=sys.stderr,
                            )

                    except ValueError:
                        print(
                            f"Warning: Skipping unsupported format '{format_name}'",
                            file=sys.stderr,
                        )

                return 0

            except Exception as e:
                print(f"Multi-format export failed: {e}", file=sys.stderr)
                return 1

        # Generate plots if requested
        if args.plots:
            try:
                plot_generator = PlotGenerator()
                plot_types = (
                    [PlotType(t.strip()) for t in args.plot_types.split(",")]
                    if args.plot_types
                    else [PlotType.LINE]
                )

                for plot_type in plot_types:
                    config = PlotConfiguration(
                        plot_type=plot_type, interactive=args.interactive_plots
                    )

                    plot_output = (
                        Path(args.output).parent
                        / f"{Path(args.output).stem}_{plot_type.value}_plot.html"
                        if args.output
                        else f"{plot_type.value}_plot.html"
                    )

                    plot_path = plot_generator.generate_plot(
                        data=data, config=config, output_path=plot_output
                    )
                    print(f"Generated plot: {plot_path}")

            except Exception as e:
                print(f"Plot generation failed: {e}", file=sys.stderr)

        # Default: show available options if no specific action was taken
        if not any(
            [args.template, args.format, args.formats, args.plots_only, args.plots]
        ):
            print("No export action specified. Use one of:")
            print("  --format FORMAT    Export to specific format")
            print("  --formats LIST     Export to multiple formats")
            print("  --template NAME    Use predefined template")
            print("  --plots-only       Generate plots only")
            print("  --plots            Include plots with export")
            return 1

        return 0

    except Exception as e:
        print(f"Export command failed: {e}", file=sys.stderr)
        if hasattr(args, "debug") and args.debug:
            import traceback

            traceback.print_exc()
        return 1


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

    # Show debug mode status if enabled
    if getattr(args, "debug", False):
        print(
            "🔍 Debug mode enabled - detailed error information will be shown",
            file=sys.stderr,
        )

    # Handle --install-completion flag before checking for subcommands
    if hasattr(args, "install_completion") and args.install_completion is not None:
        from xraylabtool.interfaces.completion import install_completion_main

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
        "compare": cmd_compare,
        "analyze": cmd_analyze,
        "export": cmd_export,
        "convert": cmd_convert,
        "formula": cmd_formula,
        "atomic": cmd_atomic,
        "bragg": cmd_bragg,
        "list": cmd_list,
        "completion": cmd_completion,
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
