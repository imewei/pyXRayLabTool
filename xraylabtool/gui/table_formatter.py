"""Shared table-cell formatting for X-ray result rows."""

from __future__ import annotations

import math
from typing import Any


class TableFormatter:
    """Formats X-ray calculation results into display-ready cell strings."""

    @staticmethod
    def format_single_row(result: Any, index: int) -> list[str]:
        """Format one energy point of a single-material result.

        Returns a list of strings matching the single-material table columns:
        Energy, Wavelength, delta, beta, Critical Angle, mrad, Atten Length,
        mu, f1, f2, Re(SLD), Im(SLD).
        """
        e = result.energy_kev[index]
        wl = result.wavelength_angstrom[index]
        delta = result.dispersion_delta[index]
        beta = result.absorption_beta[index]
        crit = result.critical_angle_degrees[index]
        atten = result.attenuation_length_cm[index]
        resld = result.real_sld_per_ang2[index]
        imsld = result.imaginary_sld_per_ang2[index]

        mu = 1.0 / atten if atten != 0 else 0.0
        mrad = crit * math.pi / 180.0 * 1000.0

        return [
            f"{e:.4f}",
            f"{wl:.5f}",
            f"{delta:.3e}",
            f"{beta:.3e}",
            f"{crit:.4f}",
            f"{mrad:.3f}",
            f"{atten:.4e}",
            f"{mu:.4e}",
            f"{result.scattering_factor_f1[index]:.3f}",
            f"{result.scattering_factor_f2[index]:.3f}",
            f"{resld:.3e}",
            f"{imsld:.3e}",
        ]

    @staticmethod
    def format_multi_row(formula: str, result: Any, index: int) -> list[str]:
        """Format one energy point of a multi-material result.

        Returns a list of strings matching the multi-material table columns:
        Formula, Density, Energy, Wavelength, delta, beta, Critical Angle,
        mrad, Atten Length, mu, f1, f2, Re(SLD), Im(SLD).
        """
        e = result.energy_kev[index]
        wl = result.wavelength_angstrom[index]
        delta = result.dispersion_delta[index]
        beta = result.absorption_beta[index]
        crit = result.critical_angle_degrees[index]
        atten = result.attenuation_length_cm[index]
        resld = result.real_sld_per_ang2[index]
        imsld = result.imaginary_sld_per_ang2[index]
        density = getattr(result, "density_g_cm3", 0.0)

        mu = 1.0 / atten if atten != 0 else 0.0
        mrad = crit * math.pi / 180.0 * 1000.0

        return [
            str(formula),
            f"{density:.4f}",
            f"{e:.4f}",
            f"{wl:.5f}",
            f"{delta:.3e}",
            f"{beta:.3e}",
            f"{crit:.4f}",
            f"{mrad:.3f}",
            f"{atten:.4e}",
            f"{mu:.4e}",
            f"{result.scattering_factor_f1[index]:.3f}",
            f"{result.scattering_factor_f2[index]:.3f}",
            f"{resld:.3e}",
            f"{imsld:.3e}",
        ]

    @staticmethod
    def format_summary(result: Any) -> list[str]:
        """Format the single-material summary row.

        Returns: [formula, molecular_weight, density, electron_density, total_electrons].
        """
        return [
            result.formula,
            f"{result.molecular_weight_g_mol:.4f}",
            f"{result.density_g_cm3:.4f}",
            f"{result.electron_density_per_ang3:.4f}",
            f"{result.total_electrons:.2f}",
        ]
