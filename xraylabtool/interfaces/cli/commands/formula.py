"""Implementation of the 'formula' command."""

import json
import sys
from typing import Any

from xraylabtool.utils import get_atomic_number, get_atomic_weight, parse_formula


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
