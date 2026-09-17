"""Implementation of the 'convert' command."""

import sys
from typing import Any

from xraylabtool.utils import energy_to_wavelength, wavelength_to_energy


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
