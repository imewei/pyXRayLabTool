"""Argument parser construction for the xraylabtool CLI."""

import argparse
from textwrap import dedent
from typing import Any

from xraylabtool import __version__


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="xraylabtool",
        description="X-ray optical properties calculator for materials science",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""
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
        """),
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
        epilog=dedent("""
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
        """),
    )

    parser.add_argument("formula", help="Chemical formula (e.g., SiO2, Al2O3, Fe2O3)")

    parser.add_argument(
        "-e",
        "--energy",
        required=True,
        help=dedent("""
        X-ray energy in keV. Formats:
        - Single value: 10.0
        - Comma-separated: 5.0,10.0,15.0
        - Range with count: 5-15:11 (11 points from 5 to 15 keV)
        - Log range: 1-30:100:log (100 log-spaced points)
        """).strip(),
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
        epilog=dedent("""
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
        """),
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
        epilog=dedent("""
        Examples:
          # Convert energy to wavelength
          xraylabtool convert energy 10.0 --to wavelength

          # Convert wavelength to energy
          xraylabtool convert wavelength 1.24 --to energy

          # Multiple values
          xraylabtool convert energy 5.0,10.0,15.0 --to wavelength

          # Save to file
          xraylabtool convert energy 5.0,10.0,15.0 --to wavelength -o conversions.csv
        """),
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
        epilog=dedent("""
        Examples:
          # Parse a simple formula
          xraylabtool formula SiO2

          # Detailed information
          xraylabtool formula Al2O3 --verbose

          # Multiple formulas
          xraylabtool formula SiO2,Al2O3,Fe2O3

          # Save results to file
          xraylabtool formula SiO2,Al2O3 -o formulas.json
        """),
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
        epilog=dedent("""
        Examples:
          # Single element
          xraylabtool atomic Si

          # Multiple elements
          xraylabtool atomic H,C,N,O,Si

          # Save to file
          xraylabtool atomic Si,Al,Fe -o atomic_data.csv
        """),
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
        epilog=dedent("""
        Examples:
          # Single calculation
          xraylabtool bragg -d 3.14 -w 1.54 --order 1

          # Multiple d-spacings
          xraylabtool bragg -d 3.14,2.45,1.92 -w 1.54

          # Energy instead of wavelength
          xraylabtool bragg -d 3.14 -e 8.0
        """),
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
        epilog=dedent("""
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
        """),
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
        epilog=dedent("""
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
        """),
    )

    # Create subparsers for completion actions
    completion_subparsers = parser.add_subparsers(
        dest="completion_action", help="Available completion actions", metavar="ACTION"
    )

    # Install subcommand
    install_parser = completion_subparsers.add_parser(
        "install",
        help="Install completion in virtual environment",
    )
    install_parser.add_argument(
        "--shell",
        "-s",
        choices=["bash", "zsh", "fish", "powershell"],
        help="Shell type (auto-detected if not specified)",
    )
    install_parser.add_argument(
        "--env",
        "-e",
        help="Target environment name (current environment if not specified)",
    )
    install_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force reinstallation if already installed",
    )

    # Uninstall subcommand
    uninstall_parser = completion_subparsers.add_parser(
        "uninstall",
        help="Remove completion from environment(s)",
    )
    uninstall_parser.add_argument(
        "--env",
        "-e",
        help="Target environment name (current environment if not specified)",
    )
    uninstall_parser.add_argument(
        "--all",
        action="store_true",
        help="Remove from all environments",
    )

    # List subcommand
    completion_subparsers.add_parser(
        "list",
        help="List environments with completion status",
    )

    # Status subcommand
    completion_subparsers.add_parser(
        "status",
        help="Show completion status for current environment",
    )

    # Info subcommand
    completion_subparsers.add_parser(
        "info",
        help="Show information about the completion system",
    )


def add_uninstall_completion_command(subparsers: Any) -> None:
    """Add the 'uninstall-completion' subcommand for shell completion removal."""
    parser = subparsers.add_parser(
        "uninstall-completion",
        help="Uninstall shell completion for xraylabtool",
        description="Remove shell completion functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""
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
        """),
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
        epilog=dedent("""
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
        """),
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
