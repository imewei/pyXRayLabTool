"""Implementation of the 'kk' command."""

import csv
import io
import json
import math
from pathlib import Path
import sys
from typing import Any

from xraylabtool.constants import EV_TO_KEV
from xraylabtool.interfaces.cli.utils import parse_energy_string


def _fail(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return 1


def _parse_merge(spec: str, energy_unit: str) -> tuple[float, float]:
    """Parse ``LOW,HIGH`` in ``energy_unit`` and return it in keV."""
    parts = spec.split(",")
    try:
        low, high = (float(p) for p in parts)
    except ValueError:
        raise ValueError(
            f"--merge must be two comma-separated numbers LOW,HIGH, got {spec!r}"
        ) from None
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError(f"--merge values must be finite, got {spec!r}")
    scale = EV_TO_KEV if energy_unit == "eV" else 1.0
    return low * scale, high * scale


def _parse_energies(spec: str) -> Any:
    """Parse ``--energies`` (keV) and reject non-finite or non-positive values."""
    import numpy as np

    try:
        energies = parse_energy_string(spec)
    except ValueError as exc:
        raise ValueError(f"Invalid --energies {spec!r}: {exc}") from exc
    if not np.isfinite(energies).all() or (energies <= 0).any():
        raise ValueError(f"--energies must be finite and positive (keV), got {spec!r}")
    return energies


def _format_output(result: Any, output_format: str) -> str:
    """Render ``energy_keV, f1, f2`` as CSV or JSON."""
    columns = {
        "energy_keV": result.energy_kev.tolist(),
        "f1": result.f1.tolist(),
        "f2": result.f2.tolist(),
    }
    if output_format == "json":
        return json.dumps(columns, indent=2)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(zip(*columns.values(), strict=True))
    return buffer.getvalue().rstrip("\n")


def cmd_kk(args: Any) -> int:
    """Handle the 'kk' command."""
    if args.nearedge and not args.input_type:
        return _fail("--input-type is required with --nearedge")
    if not args.nearedge and (args.input_type or args.merge):
        return _fail("--input-type and --merge require --nearedge")
    if args.density is not None and not (
        math.isfinite(args.density) and args.density > 0
    ):
        return _fail(f"--density must be a finite positive number, got {args.density}")

    output_format = args.format
    if output_format is None:
        suffix = Path(args.output).suffix.lower() if args.output else ""
        output_format = "json" if suffix == ".json" else "csv"

    # Lazy imports for this command
    from xraylabtool.exceptions import XRayLabToolError
    from xraylabtool.kk import kk_calculate_real, load_nearedge
    from xraylabtool.kk._device import enable_persistent_cache

    # The CLI never calls warm_up (a one-shot call must not compile formulas it
    # was not asked for); it only lets repeated invocations reuse compiled code.
    enable_persistent_cache()
    try:
        energies = _parse_energies(args.energies) if args.energies else None
        merge = _parse_merge(args.merge, args.energy_unit) if args.merge else None
        nearedge = (
            load_nearedge(args.nearedge, energy_unit=args.energy_unit)
            if args.nearedge
            else None
        )
        result = kk_calculate_real(
            args.formula,
            energies_kev=energies,
            nearedge=nearedge,
            input_type=args.input_type,
            merge_points_kev=merge,
            refine_tolerance=args.refine,
        )
        if args.refine is not None:
            print(f"Refinement added {result.n_refined_points} points", file=sys.stderr)
            if result.refine_exhausted:
                print(
                    "Warning: refinement stopped on a limit before every "
                    f"interval met the tolerance ({args.refine:g})",
                    file=sys.stderr,
                )

        text = _format_output(result, output_format)
        if args.output:
            Path(args.output).write_text(text + "\n")
        else:
            print(text)
        return 0
    except (XRayLabToolError, TypeError, ValueError, OSError) as exc:
        return _fail(str(exc))
