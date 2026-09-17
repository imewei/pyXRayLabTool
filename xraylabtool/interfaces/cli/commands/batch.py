"""Implementation of the 'batch' command."""

import json
from pathlib import Path
import sys
from typing import Any

from xraylabtool.interfaces.cli.utils import create_batch_progress_tracker


def _validate_batch_input(args: Any) -> list[Any] | None:
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
    data_input: list[Any],
) -> tuple[list[str] | None, list[float] | None, list[list[float]] | None]:
    """Parse batch data from list of dictionaries."""
    formulas = []
    densities = []
    energy_sets = []

    for row in data_input:
        # Parse the whole row inside the guard so a single malformed cell skips
        # only that row (with a warning) instead of aborting the entire batch
        # and discarding every valid material. Keeps the three lists aligned.
        try:
            formula = row["formula"]
            density = float(row["density"])
            energy_str = str(row["energy"])
            if "," in energy_str:
                energies = [float(x.strip()) for x in energy_str.split(",")]
            else:
                energies = [float(energy_str)]
        except (ValueError, KeyError) as exc:
            print(
                f"Warning: Skipping {row.get('formula', '?')}: invalid row ({exc})",
                file=sys.stderr,
            )
            continue

        formulas.append(formula)
        densities.append(density)
        energy_sets.append(energies)

    return formulas, densities, energy_sets


def _convert_result_to_dict(result: Any, energy_index: int) -> dict[str, Any]:
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
                if args.verbose and not enable_progress:
                    print(f"  {i + 1}/{len(formulas)}: {formula}")

                result = calculate_single_material_properties(
                    formula, energies, density
                )

                for j, _energy in enumerate(energies):
                    result_dict = _convert_result_to_dict(result, j)
                    results.append(result_dict)

            except Exception as e:
                if not enable_progress:  # Only print if progress bar isn't showing
                    print(f"Warning: Failed to process {formula}: {e}")
                continue

            finally:
                # Update progress
                progress.update(1)
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
        debug_mode = getattr(args, "debug", False)
        if debug_mode:
            import traceback

            print("🔍 Debug: Full traceback:", file=sys.stderr)
            traceback.print_exc()
        print(f"Error: {e}", file=sys.stderr)
        return 1
