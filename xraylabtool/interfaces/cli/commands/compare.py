"""Implementation of the 'compare' command."""

import json
from pathlib import Path
import sys
from typing import Any

from xraylabtool.interfaces.cli.utils import parse_energy_string


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
