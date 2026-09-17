"""Implementation of the 'bragg' command."""

import sys
from typing import Any

from xraylabtool.utils import bragg_angle, energy_to_wavelength


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
