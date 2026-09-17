"""Implementation of the 'calc' command."""

from pathlib import Path
import sys
from typing import Any

from xraylabtool.interfaces.cli.formatting import (
    _get_default_fields,
    format_xray_result,
)
from xraylabtool.interfaces.cli.utils import parse_energy_string


def _validate_calc_inputs(args: Any, energies: Any) -> bool:
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


def _print_calc_verbose_info(args: Any, energies: Any) -> None:
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
            # Validate field names up front so every output format behaves the
            # same: previously an unknown field was silently dropped in table
            # format (exit 0) but raised AttributeError for csv/json (exit 1).
            scalar_fields, array_fields = _get_default_fields()
            known = set(scalar_fields) | set(array_fields)
            unknown = [f for f in fields if f not in known]
            if unknown:
                print(
                    f"Error: Unknown field(s): {', '.join(unknown)}",
                    file=sys.stderr,
                )
                return 1

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
