"""Implementation of the 'atomic' command."""

import json
from pathlib import Path
import sys
from typing import Any

from xraylabtool.utils import get_atomic_number, get_atomic_weight


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
