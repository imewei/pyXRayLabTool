"""Material comparison functionality for X-ray properties analysis."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from xraylabtool.calculators.core import calculate_xray_properties


@dataclass
class ComparisonResult:
    """Result container for material comparisons."""

    materials: list[str]
    energies: list[float]
    properties: list[str]
    data: dict[str, Any]
    summary_stats: dict[str, dict[str, float]]
    recommendations: list[str]


class MaterialComparator:
    """Compare X-ray properties between multiple materials."""

    def __init__(self):  # type: ignore[no-untyped-def]
        self.default_properties = [
            "critical_angle_degrees",
            "attenuation_length_cm",
            "dispersion_delta",
            "absorption_beta",
        ]

    def compare_materials(
        self,
        formulas: list[str],
        densities: list[float],
        energies: list[float],
        properties: list[str] | None = None,
    ) -> ComparisonResult:
        """
        Compare X-ray properties across multiple materials.

        Args:
            formulas: List of chemical formulas
            densities: List of material densities in g/cm³
            energies: List of X-ray energies in keV
            properties: Properties to compare (uses defaults if None)

        Returns:
            ComparisonResult with comparison data
        """
        self._validate_comparison_inputs(formulas, densities, energies)

        if properties is None:
            properties = self.default_properties.copy()

        material_data = self._calculate_material_data(formulas, densities, energies)
        comparison_data = self._extract_comparison_data(
            material_data, properties, energies
        )
        summary_stats = self._calculate_summary_stats(comparison_data, properties)
        recommendations = self._generate_recommendations(
            formulas, comparison_data, summary_stats, energies
        )

        return ComparisonResult(
            materials=[
                f"{f} ({d} g/cm³)" for f, d in zip(formulas, densities, strict=False)
            ],
            energies=energies,
            properties=properties,
            data=comparison_data,
            summary_stats=summary_stats,
            recommendations=recommendations,
        )

    @staticmethod
    def _validate_comparison_inputs(
        formulas: list[str], densities: list[float], energies: list[float]
    ) -> None:
        """Validate compare_materials inputs."""
        if len(formulas) != len(densities):
            raise ValueError("Number of formulas must match number of densities")

        if len(formulas) < 2:
            raise ValueError("At least two materials required for comparison")

        if not energies:
            raise ValueError("At least one energy value required")

    @staticmethod
    def _calculate_material_data(
        formulas: list[str], densities: list[float], energies: list[float]
    ) -> dict[str, Any]:
        """Calculate X-ray properties for each material, keyed by "formula (density)"."""
        material_data = {}
        for formula, density in zip(formulas, densities, strict=False):
            try:
                result_dict = calculate_xray_properties(
                    formulas=[formula], energies=energies, densities=[density]
                )
                xray_result = result_dict[formula]
                material_key = f"{formula} ({density} g/cm³)"
                material_data[material_key] = xray_result
            except Exception as e:
                raise ValueError(f"Failed to calculate properties for {formula}: {e}")
        return material_data

    @staticmethod
    def _extract_comparison_data(
        material_data: dict[str, Any], properties: list[str], energies: list[float]
    ) -> dict[str, Any]:
        """Extract requested properties from each material's XRayResult."""
        comparison_data: dict[str, Any] = {}
        for prop in properties:
            comparison_data[prop] = {}
            for material_key, xray_result in material_data.items():
                if hasattr(xray_result, prop):
                    values = getattr(xray_result, prop)
                    if isinstance(values, np.ndarray):
                        comparison_data[prop][material_key] = values.tolist()
                    else:
                        comparison_data[prop][material_key] = [values] * len(energies)
        return comparison_data

    @staticmethod
    def _calculate_summary_stats(
        comparison_data: dict[str, Any], properties: list[str]
    ) -> dict[str, dict[str, float]]:
        """Calculate mean/std/min/max/range across all materials, per property."""
        summary_stats = {}
        for prop in properties:
            if prop not in comparison_data:
                continue
            all_values = []
            for material_values in comparison_data[prop].values():
                all_values.extend(material_values)

            if all_values:
                summary_stats[prop] = {
                    "mean": float(np.mean(all_values)),
                    "std": float(np.std(all_values)),
                    "min": float(np.min(all_values)),
                    "max": float(np.max(all_values)),
                    "range": float(np.max(all_values) - np.min(all_values)),
                }
        return summary_stats

    def create_comparison_table(self, result: ComparisonResult) -> pd.DataFrame:
        """
        Create a pandas DataFrame from comparison results.

        Args:
            result: ComparisonResult object

        Returns:
            DataFrame with comparison data
        """
        rows = []

        for i, energy in enumerate(result.energies):
            for material in result.materials:
                row: dict[str, Any] = {"Material": material, "Energy_keV": energy}

                for prop in result.properties:
                    row[prop] = self._comparison_table_cell(result, prop, material, i)

                rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def _comparison_table_cell(
        result: ComparisonResult, prop: str, material: str, energy_index: int
    ) -> float | None:
        """Resolve one (property, material, energy) cell to a plain float or None."""
        if prop not in result.data or material not in result.data[prop]:
            return None

        values = result.data[prop][material]
        if not len(values):
            return None
        val = values[energy_index] if energy_index < len(values) else values[0]

        try:
            # Coerce length-1 numpy arrays to plain float
            if hasattr(val, "__len__") and not isinstance(val, (str, bytes)):
                if len(val) == 1:
                    val = val[0]
            if val is None:
                return None
            try:
                return float(val)
            except Exception:
                # numpy scalar fallback
                return float(np.asarray(val).squeeze())
        except Exception:
            return None

    def generate_comparison_report(self, result: ComparisonResult) -> str:
        """
        Generate a detailed text report from comparison results.

        Args:
            result: ComparisonResult object

        Returns:
            Formatted text report
        """
        lines = ["X-RAY PROPERTIES COMPARISON REPORT", "=" * 50, ""]
        lines.extend(self._report_materials_section(result))
        lines.extend(self._report_energy_section(result))
        lines.extend(self._report_properties_section(result))
        lines.extend(self._report_stats_section(result))
        lines.extend(self._report_rankings_section(result))
        lines.extend(self._report_recommendations_section(result))
        lines.append("Report generated by XRayLabTool")

        return "\n".join(lines)

    @staticmethod
    def _report_materials_section(result: ComparisonResult) -> list[str]:
        lines = ["MATERIALS COMPARED:"]
        for i, material in enumerate(result.materials, 1):
            lines.append(f"  {i}. {material}")
        lines.append("")
        return lines

    @staticmethod
    def _report_energy_section(result: ComparisonResult) -> list[str]:
        if len(result.energies) == 1:
            lines = [f"ENERGY: {result.energies[0]:.3f} keV"]
        else:
            lines = [
                f"ENERGY RANGE: {min(result.energies):.3f} - "
                f"{max(result.energies):.3f} keV",
                f"  ({len(result.energies)} energy points)",
            ]
        lines.append("")
        return lines

    @staticmethod
    def _report_properties_section(result: ComparisonResult) -> list[str]:
        lines = ["PROPERTIES ANALYZED:"]
        for prop in result.properties:
            lines.append(f"  • {prop.replace('_', ' ').title()}")
        lines.append("")
        return lines

    @staticmethod
    def _report_stats_section(result: ComparisonResult) -> list[str]:
        if not result.summary_stats:
            return []
        lines = ["SUMMARY STATISTICS:", "-" * 30]
        for prop, stats in result.summary_stats.items():
            lines.append(f"\n{prop.replace('_', ' ').title()}:")
            lines.append(f"  Mean: {stats['mean']:.6g}")
            lines.append(f"  Std:  {stats['std']:.6g}")
            lines.append(f"  Min:  {stats['min']:.6g}")
            lines.append(f"  Max:  {stats['max']:.6g}")
            lines.append(f"  Range: {stats['range']:.6g}")
        lines.append("")
        return lines

    @staticmethod
    def _report_rankings_section(result: ComparisonResult) -> list[str]:
        # Rankings only make sense at a single energy point.
        if len(result.energies) != 1:
            return []

        lines = ["MATERIAL RANKINGS:", "-" * 20]
        for prop in result.properties:
            if prop not in result.data:
                continue
            prop_data = result.data[prop]
            sorted_materials = sorted(
                prop_data.items(),
                key=lambda x: x[1][0] if x[1] else 0,
                reverse=True,
            )

            lines.append(f"\n{prop.replace('_', ' ').title()} (highest to lowest):")
            for i, (material, values) in enumerate(sorted_materials, 1):
                value = values[0] if values else 0
                lines.append(f"  {i}. {material}: {value:.6g}")
        lines.append("")
        return lines

    @staticmethod
    def _report_recommendations_section(result: ComparisonResult) -> list[str]:
        if not result.recommendations:
            return []
        lines = ["RECOMMENDATIONS:", "-" * 15]
        for i, rec in enumerate(result.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")
        return lines

    def _generate_recommendations(
        self,
        _formulas: list[str],
        data: dict[str, Any],
        stats: dict[str, dict[str, float]],
        energies: list[float],
    ) -> list[str]:
        """Generate analysis recommendations based on comparison results."""
        recommendations = []

        # Check for significant differences
        for prop, prop_stats in stats.items():
            if (
                prop_stats["std"] / prop_stats["mean"] > 0.5
            ):  # High coefficient of variation
                recommendations.append(
                    f"Large variation in {prop.replace('_', ' ')} across materials - "
                    "consider this for material selection"
                )

        # Energy-specific recommendations
        if len(energies) > 1:
            recommendations.append(
                "Multiple energies analyzed - check energy-dependent behavior for optimal selection"
            )

        # Critical angle recommendations
        if "critical_angle_degrees" in data:
            recommendations.append(
                "For grazing incidence applications, materials with larger critical angles "
                "provide better penetration"
            )

        # Attenuation recommendations
        if "attenuation_length_cm" in data:
            recommendations.append(
                "For transmission applications, materials with longer attenuation lengths "
                "are preferred"
            )

        if not recommendations:
            recommendations.append(
                "All materials show similar X-ray properties at the given energies"
            )

        return recommendations
