"""Implementation of the 'compare' command."""

import json
from pathlib import Path
import sys
from typing import Any

from xraylabtool.interfaces.cli.utils import parse_energy_string


def _parse_materials(
    material_strs: list[str],
) -> tuple[list[str], list[float]] | None:
    """Parse 'formula,density' strings. Prints an error and returns None on failure."""
    formulas = []
    densities = []

    for material_str in material_strs:
        try:
            parts = material_str.split(",")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid material format: {material_str}. Expected"
                    " 'formula,density'"
                )

            formulas.append(parts[0].strip())
            densities.append(float(parts[1].strip()))

        except ValueError as e:
            print(f"Error parsing material '{material_str}': {e}", file=sys.stderr)
            return None

    if len(formulas) < 2:
        print("Error: At least two materials required for comparison", file=sys.stderr)
        return None

    return formulas, densities


def _result_to_dict(result: Any) -> dict[str, Any]:
    """Convert a ComparisonResult to a JSON-serializable dict."""
    return {
        "materials": result.materials,
        "energies": result.energies,
        "properties": result.properties,
        "data": result.data,
        "summary_stats": result.summary_stats,
        "recommendations": result.recommendations,
    }


def _write_report(comparator: Any, result: Any, args: Any) -> None:
    """Handle --report / --format report output."""
    report = comparator.generate_comparison_report(result)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Comparison report saved to {args.output}")
    else:
        print(report)


def _write_table(comparator: Any, result: Any, args: Any) -> None:
    """Handle table/csv/json output (to a file or stdout)."""
    table = comparator.create_comparison_table(result)

    if args.output:
        output_path = Path(args.output)
        if output_path.suffix.lower() == ".json":
            with open(args.output, "w") as f:
                json.dump(_result_to_dict(result), f, indent=2)
        else:
            table.to_csv(args.output, index=False)
        print(f"Comparison results saved to {args.output}")
    elif args.format == "json":
        print(json.dumps(_result_to_dict(result), indent=2))
    elif args.format == "csv":
        print(table.to_csv(index=False))
    else:
        print(table.to_string(index=False))


def cmd_compare(args: Any) -> int:
    """Handle the 'compare' command for material comparison."""
    try:
        # Lazy imports for comparison functionality
        from xraylabtool.analysis import MaterialComparator

        parsed_materials = _parse_materials(args.materials)
        if parsed_materials is None:
            return 1
        formulas, densities = parsed_materials

        try:
            energies = parse_energy_string(args.energy).tolist()
        except Exception as e:
            print(f"Error parsing energy range: {e}", file=sys.stderr)
            return 1

        properties = None
        if args.properties:
            properties = [prop.strip() for prop in args.properties.split(",")]

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

        if args.report or args.format == "report":
            _write_report(comparator, result, args)
        else:
            _write_table(comparator, result, args)

        return 0

    except Exception as e:
        print(f"Comparison failed: {e}", file=sys.stderr)
        if hasattr(args, "debug") and args.debug:
            import traceback

            traceback.print_exc()
        return 1
