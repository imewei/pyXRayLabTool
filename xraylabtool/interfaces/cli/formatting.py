"""Result formatting (table/CSV/JSON) for the 'calc' command output."""

import json
from typing import Any

import numpy as np


def _get_default_fields() -> tuple[list[str], list[str]]:
    """Get default scalar and array fields."""
    array_fields = [
        "energy_kev",
        "wavelength_angstrom",
        "dispersion_delta",
        "absorption_beta",
        "scattering_factor_f1",
        "scattering_factor_f2",
        "critical_angle_degrees",
        "attenuation_length_cm",
        "real_sld_per_ang2",
        "imaginary_sld_per_ang2",
    ]
    scalar_fields = [
        "formula",
        "molecular_weight_g_mol",
        "total_electrons",
        "density_g_cm3",
        "electron_density_per_ang3",
    ]
    return scalar_fields, array_fields


def _format_as_json(result: Any, fields: list[str]) -> str:
    """Format result as JSON."""
    data = {}
    for field in fields:
        value = getattr(result, field)
        if isinstance(value, np.ndarray):
            data[field] = value.tolist()
        else:
            data[field] = value
    return json.dumps(data, indent=2)


def _format_as_csv(result: Any, fields: list[str], precision: int) -> str:
    """Format result as CSV."""
    import csv
    import io

    n_energies = len(result.energy_kev)

    # Vectorized approach: separate array and scalar fields for efficiency
    array_fields = [f for f in fields if isinstance(getattr(result, f), np.ndarray)]
    scalar_fields = [
        f for f in fields if not isinstance(getattr(result, f), np.ndarray)
    ]

    # Vectorize array operations
    data_arrays = {
        field: np.round(getattr(result, field), precision) for field in array_fields
    }
    scalar_data = {field: getattr(result, field) for field in scalar_fields}

    # Create rows efficiently using vectorized data
    data_rows = [
        {
            **scalar_data,
            **{field: float(data_arrays[field][i]) for field in array_fields},
        }
        for i in range(n_energies)
    ]

    if data_rows:
        # Use CSV module instead of pandas
        output = io.StringIO()
        fieldnames = fields
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_rows)
        return output.getvalue()
    return ""


def _format_material_properties(result: Any, precision: int) -> list[str]:
    """Format material properties section."""
    return [
        "Material Properties:",
        f"  Formula: {result.formula}",
        f"  Molecular Weight: {result.molecular_weight_g_mol: .{precision}f} g/mol",
        f"  Total Electrons: {result.total_electrons: .{precision}f}",
        f"  Density: {result.density_g_cm3: .{precision}f} g/cm³",
        (
            f"  Electron Density: {result.electron_density_per_ang3: .{precision}e} "
            "electrons/Å³"
        ),
        "",
    ]


def _format_single_energy(result: Any, precision: int) -> list[str]:
    """Format single energy point properties."""
    return [
        "X-ray Properties:",
        f"  Energy: {result.energy_kev[0]:.{precision}f} keV",
        f"  Wavelength: {result.wavelength_angstrom[0]:.{precision}f} Å",
        f"  Dispersion (δ): {result.dispersion_delta[0]:.{precision}e}",
        f"  Absorption (β): {result.absorption_beta[0]:.{precision}e}",
        f"  Scattering f1: {result.scattering_factor_f1[0]:.{precision}f}",
        f"  Scattering f2: {result.scattering_factor_f2[0]:.{precision}f}",
        f"  Critical Angle: {result.critical_angle_degrees[0]:.{precision}f}°",
        f"  Attenuation Length: {result.attenuation_length_cm[0]:.{precision}f} cm",
        f"  Real SLD: {result.real_sld_per_ang2[0]:.{precision}e} Å⁻²",
        f"  Imaginary SLD: {result.imaginary_sld_per_ang2[0]:.{precision}e} Å⁻²",
    ]


def _format_multiple_energies(result: Any, precision: int) -> list[str]:
    """Format multiple energy points as table."""
    output_lines = ["X-ray Properties (tabular):"]

    # Create table without pandas
    headers = ["Energy (keV)", "λ (Å)", "δ", "β", "f1", "f2", "θc (°)", "μ (cm)"]
    data_arrays = [
        result.energy_kev,
        result.wavelength_angstrom,
        result.dispersion_delta,
        result.absorption_beta,
        result.scattering_factor_f1,
        result.scattering_factor_f2,
        result.critical_angle_degrees,
        result.attenuation_length_cm,
    ]

    # Calculate column widths
    col_widths = [max(len(header), 12) for header in headers]

    # Format header
    header_line = "  ".join(
        header.ljust(width) for header, width in zip(headers, col_widths, strict=False)
    )
    output_lines.append(header_line)

    # Format data rows
    n_energies = len(result.energy_kev)
    for i in range(n_energies):
        row_values = []
        for data_array in data_arrays:
            value = data_array[i] if isinstance(data_array, np.ndarray) else data_array
            row_values.append(f"{value:.{precision}g}")

        row_line = "  ".join(
            val.ljust(width) for val, width in zip(row_values, col_widths, strict=False)
        )
        output_lines.append(row_line)

    return output_lines


def _format_scalar_field(field: str, value: Any, precision: int) -> str:
    """Format a single scalar field."""
    from collections.abc import Callable

    def default_formatter(v: Any, p: int) -> str:
        return ""

    formatters: dict[str, Callable[[Any, int], str]] = {
        "formula": lambda v, _: f"  Formula: {v}",
        "molecular_weight_g_mol": lambda v, p: f"  Molecular Weight: {v: .{p}f} g/mol",
        "total_electrons": lambda v, p: f"  Total Electrons: {v: .{p}f}",
        "density_g_cm3": lambda v, p: f"  Density: {v: .{p}f} g/cm³",
        "electron_density_per_ang3": lambda v, p: (
            f"  Electron Density: {v: .{p}e} electrons/Å³"
        ),
    }
    formatter = formatters.get(field, default_formatter)
    return formatter(value, precision)


def _format_array_field_single(field: str, value: float, precision: int) -> str:
    """Format a single array field for single energy point."""
    formatters: dict[str, tuple[str, str, str]] = {
        "energy_kev": ("  Energy:", "f", " keV"),
        "wavelength_angstrom": ("  Wavelength:", "f", " Å"),
        "dispersion_delta": ("  Dispersion (δ):", "e", ""),
        "absorption_beta": ("  Absorption (β):", "e", ""),
        "scattering_factor_f1": ("  Scattering f1:", "f", ""),
        "scattering_factor_f2": ("  Scattering f2:", "f", ""),
        "critical_angle_degrees": ("  Critical Angle:", "f", "°"),
        "attenuation_length_cm": ("  Attenuation Length:", "f", " cm"),
        "real_sld_per_ang2": ("  Real SLD:", "e", " Å⁻²"),
        "imaginary_sld_per_ang2": ("  Imaginary SLD:", "e", " Å⁻²"),
    }

    if field in formatters:
        label, fmt, suffix = formatters[field]
        return f"{label} {value: .{precision}{fmt}}{suffix}"
    return ""


def _get_field_labels() -> dict[str, str]:
    """Get mapping of field names to display labels."""
    return {
        "energy_kev": "Energy (keV)",
        "wavelength_angstrom": "λ (Å)",
        "dispersion_delta": "δ",
        "absorption_beta": "β",
        "scattering_factor_f1": "f1",
        "scattering_factor_f2": "f2",
        "critical_angle_degrees": "θc (°)",
        "attenuation_length_cm": "μ (cm)",
        "real_sld_per_ang2": "Real SLD",
        "imaginary_sld_per_ang2": "Imag SLD",
    }


def _format_scalar_fields_section(
    result: Any, fields_to_show: list[str], precision: int
) -> list[str]:
    """Format scalar fields section."""
    if not fields_to_show:
        return []

    output_lines = ["Material Properties:"]
    for field in fields_to_show:
        value = getattr(result, field)
        line = _format_scalar_field(field, value, precision)
        if line:
            output_lines.append(line)
    output_lines.append("")
    return output_lines


def _format_single_energy_section(
    result: Any, fields_to_show: list[str], precision: int
) -> list[str]:
    """Format single energy point array fields."""
    if not fields_to_show:
        return []

    output_lines = ["X-ray Properties:"]
    for field in fields_to_show:
        value = getattr(result, field)[0]
        line = _format_array_field_single(field, value, precision)
        if line:
            output_lines.append(line)
    return output_lines


def _format_multiple_energy_section(
    result: Any, fields_to_show: list[str], precision: int
) -> list[str]:
    """Format multiple energy points as tabular data."""
    if not fields_to_show:
        return []

    output_lines = ["X-ray Properties (tabular):"]
    field_labels = _get_field_labels()

    # Collect headers and data arrays
    headers = []
    data_arrays = []
    for field in fields_to_show:
        label = field_labels.get(field, field)
        headers.append(label)
        data_arrays.append(getattr(result, field))

    if headers:
        # Calculate column widths
        col_widths = [max(len(header), 12) for header in headers]

        # Format header
        header_line = "  ".join(
            header.ljust(width)
            for header, width in zip(headers, col_widths, strict=False)
        )
        output_lines.append(header_line)

        # Format data rows
        n_rows = len(data_arrays[0]) if data_arrays else 0
        for i in range(n_rows):
            row_values = []
            for data_array in data_arrays:
                if isinstance(data_array, np.ndarray):
                    value = data_array[i]
                else:
                    value = data_array
                row_values.append(f"{value:.{precision}g}")

            row_line = "  ".join(
                val.ljust(width)
                for val, width in zip(row_values, col_widths, strict=False)
            )
            output_lines.append(row_line)

    return output_lines


def _format_filtered_table(result: Any, fields: list[str], precision: int) -> str:
    """Format table with only specified fields."""
    # Separate scalar and array fields
    scalar_fields, array_fields = _get_default_fields()
    scalar_fields_to_show = [f for f in fields if f in scalar_fields]
    array_fields_to_show = [f for f in fields if f in array_fields]

    output_lines = []

    # Add scalar fields section
    output_lines.extend(
        _format_scalar_fields_section(result, scalar_fields_to_show, precision)
    )

    # Add array fields section
    if array_fields_to_show:
        if len(result.energy_kev) == 1:
            output_lines.extend(
                _format_single_energy_section(result, array_fields_to_show, precision)
            )
        else:
            output_lines.extend(
                _format_multiple_energy_section(result, array_fields_to_show, precision)
            )

    return "\n".join(output_lines)


def format_xray_result(
    result: Any,  # XRayResult - type hint removed for lazy loading
    format_type: str,
    precision: int = 6,
    fields: list[str] | None = None,
) -> str:
    """Format XRayResult for output."""
    if fields is None:
        scalar_fields, array_fields = _get_default_fields()
        fields = scalar_fields + array_fields

    if format_type == "json":
        return _format_as_json(result, fields)
    elif format_type == "csv":
        return _format_as_csv(result, fields, precision)
    else:  # table format
        # For table format with custom fields, use a filtered output
        if fields != _get_default_fields()[0] + _get_default_fields()[1]:
            return _format_filtered_table(result, fields, precision)

        # Default table format (all fields)
        output_lines = _format_material_properties(result, precision)

        if len(result.energy_kev) == 1:
            output_lines.extend(_format_single_energy(result, precision))
        else:
            output_lines.extend(_format_multiple_energies(result, precision))

        return "\n".join(output_lines)
