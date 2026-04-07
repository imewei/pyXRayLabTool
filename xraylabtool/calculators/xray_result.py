"""XRayResult dataclass for X-ray optical property calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
import warnings

import numpy as np

if TYPE_CHECKING:
    from xraylabtool.typing_extensions import (
        EnergyArray,
        OpticalConstantArray,
        WavelengthArray,
    )


@dataclass
class XRayResult:
    """
    Dataclass containing complete X-ray optical property calculations for a material.

    This comprehensive data structure holds all computed X-ray properties including
    fundamental scattering factors, optical constants, and derived quantities like
    critical angles and attenuation lengths. All fields use descriptive snake_case
    names with clear units for maximum clarity.

    The dataclass is optimized for scientific workflows, supporting both single-energy
    calculations and energy-dependent analysis. All array fields are automatically
    converted to numpy arrays for efficient numerical operations.

    **Legacy Compatibility:**
    Deprecated CamelCase property aliases are available for backward compatibility
    but emit DeprecationWarning when accessed. Use the new snake_case field names
    for all new code.

    Attributes:
        Material Properties:

        formula (str): Chemical formula string exactly as provided
        molecular_weight_g_mol (float): Molecular weight in g/mol
        total_electrons (float): Total electrons per molecule (sum over all atoms)
        density_g_cm3 (float): Mass density in g/cm³
        electron_density_per_ang3 (float): Electron density in electrons/Å³

        X-ray Energy and Wavelength:

        energy_kev (np.ndarray): X-ray photon energies in keV
        wavelength_angstrom (np.ndarray): Corresponding X-ray wavelengths in Å

        Fundamental X-ray Properties:

        dispersion_delta (np.ndarray): Dispersion coefficient δ (real part of
                                      refractive index decrement: n = 1 - δ - iβ)
        absorption_beta (np.ndarray): Absorption coefficient β (imaginary part of
                                     refractive index decrement)
        scattering_factor_f1 (np.ndarray): Real part of atomic scattering factor
        scattering_factor_f2 (np.ndarray): Imaginary part of atomic scattering factor

        Derived Optical Properties:

        critical_angle_degrees (np.ndarray): Critical angles for total external
                                            reflection in degrees (θc = √(2δ))
        attenuation_length_cm (np.ndarray): 1/e penetration depths in cm
        real_sld_per_ang2 (np.ndarray): Real part of scattering length density in Å⁻²
        imaginary_sld_per_ang2 (np.ndarray): Imaginary part of scattering length
                                            density in Å⁻²

    Physical Relationships:

    - Refractive Index: n = 1 - δ - iβ where δ and β are wavelength-dependent
    - Critical Angle: θc = √(2δ) for grazing incidence geometry
    - Attenuation Length: μ^-1 = (4πβ/λ)^-1 for exponential decay
    - Dispersion/Absorption: Related to f1, f2 via classical electron radius

    Examples:
        Basic Property Access:

        >>> import xraylabtool as xlt
        >>> result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)
        >>> print(f"Material: {result.formula}")
        Material: SiO2
        >>> print(f"MW: {result.molecular_weight_g_mol:.2f} g/mol")
        MW: 60.08 g/mol
        >>> print(result.critical_angle_degrees[0] > 0.1)  # Reasonable critical angle
        True

        Array Properties for Energy Scans:

        >>> import numpy as np
        >>> energies = np.linspace(8, 12, 5)
        >>> result = xlt.calculate_single_material_properties("Si", energies, 2.33)
        >>> print(f"Energies: {result.energy_kev}")
        Energies: [ 8.  9. 10. 11. 12.]
        >>> print(len(result.wavelength_angstrom))
        5

        Optical Constants Analysis:

        >>> print(result.dispersion_delta.min() > 0)  # δ should be positive
        True
        >>> print(result.absorption_beta.min() >= 0)  # β should be non-negative
        True

        Derived Quantities:

        >>> print(len(result.critical_angle_degrees))
        5
        >>> print(len(result.attenuation_length_cm))
        5

    Note:
        All numpy arrays have the same length as the input energy array. For scalar
        energy inputs, arrays will have length 1. Use standard numpy operations
        for analysis (e.g., np.min(), np.max(), np.argmin(), indexing).

    See Also:
        calculate_single_material_properties : Primary function returning this class
        calculate_xray_properties : Function returning Dict[str, XRayResult]
    """

    # Material properties with enhanced type annotations
    formula: str  # Chemical formula string
    molecular_weight_g_mol: float  # Molecular weight (g/mol)
    total_electrons: float  # Total electrons per molecule
    density_g_cm3: float  # Mass density (g/cm³)
    electron_density_per_ang3: float  # Electron density (electrons/Å³)

    # X-ray energy and wavelength arrays (performance-optimized dtypes)
    energy_kev: EnergyArray = field()  # X-ray energies in keV
    wavelength_angstrom: WavelengthArray = field()  # X-ray wavelengths in Å

    # Fundamental optical constants (performance-critical arrays)
    dispersion_delta: OpticalConstantArray = field()  # Dispersion coefficient δ
    absorption_beta: OpticalConstantArray = field()  # Absorption coefficient β

    # Atomic scattering factors (complex arrays for scientific accuracy)
    scattering_factor_f1: OpticalConstantArray = (
        field()
    )  # Real part of scattering factor
    scattering_factor_f2: OpticalConstantArray = (
        field()
    )  # Imaginary part of scattering factor

    # Derived optical properties (performance-optimized arrays)
    critical_angle_degrees: OpticalConstantArray = field()  # Critical angles (degrees)
    attenuation_length_cm: OpticalConstantArray = field()  # Attenuation lengths (cm)
    real_sld_per_ang2: OpticalConstantArray = field()  # Real SLD (Å⁻²)
    imaginary_sld_per_ang2: OpticalConstantArray = field()  # Imaginary SLD (Å⁻²)

    def __post_init__(self) -> None:
        """Post-initialization to handle any setup after object creation."""
        # Only convert if not already a numpy array (e.g. when constructed from
        # raw Python lists or scalars, not from the internal calculation path
        # which already produces float64 contiguous arrays).
        if not isinstance(self.energy_kev, np.ndarray):
            self.energy_kev = np.asarray(self.energy_kev, dtype=np.float64)
        if not isinstance(self.wavelength_angstrom, np.ndarray):
            self.wavelength_angstrom = np.asarray(
                self.wavelength_angstrom, dtype=np.float64
            )
        if not isinstance(self.dispersion_delta, np.ndarray):
            self.dispersion_delta = np.asarray(self.dispersion_delta, dtype=np.float64)
        if not isinstance(self.absorption_beta, np.ndarray):
            self.absorption_beta = np.asarray(self.absorption_beta, dtype=np.float64)
        if not isinstance(self.scattering_factor_f1, np.ndarray):
            self.scattering_factor_f1 = np.asarray(
                self.scattering_factor_f1, dtype=np.float64
            )
        if not isinstance(self.scattering_factor_f2, np.ndarray):
            self.scattering_factor_f2 = np.asarray(
                self.scattering_factor_f2, dtype=np.float64
            )
        if not isinstance(self.critical_angle_degrees, np.ndarray):
            self.critical_angle_degrees = np.asarray(
                self.critical_angle_degrees, dtype=np.float64
            )
        if not isinstance(self.attenuation_length_cm, np.ndarray):
            self.attenuation_length_cm = np.asarray(
                self.attenuation_length_cm, dtype=np.float64
            )
        if not isinstance(self.real_sld_per_ang2, np.ndarray):
            self.real_sld_per_ang2 = np.asarray(
                self.real_sld_per_ang2, dtype=np.float64
            )
        if not isinstance(self.imaginary_sld_per_ang2, np.ndarray):
            self.imaginary_sld_per_ang2 = np.asarray(
                self.imaginary_sld_per_ang2, dtype=np.float64
            )

    # Convenience properties used in docs/notebooks
    @property
    def energy_ev(self):  # type: ignore[no-untyped-def]
        return self.energy_kev * 1000.0

    @property
    def delta(self):  # type: ignore[no-untyped-def]
        return self.dispersion_delta

    @property
    def beta(self):  # type: ignore[no-untyped-def]
        return self.absorption_beta

    @property
    def critical_angle_mrad(self):  # type: ignore[no-untyped-def]
        return self.critical_angle_degrees * np.pi / 180.0 * 1000.0

    @property
    def linear_absorption_coefficient(self):  # type: ignore[no-untyped-def]
        # μ = 1 / attenuation length
        arr = np.where(
            self.attenuation_length_cm != 0, 1.0 / self.attenuation_length_cm, 0.0
        )
        return np.asarray(arr)

    # Legacy property aliases (deprecated) - emit warnings when accessed
    @property
    def Formula(self) -> str:
        """Deprecated: Use 'formula' instead."""
        warnings.warn(
            "Formula is deprecated, use 'formula' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.formula

    @property
    def MW(self) -> float:
        """Deprecated: Use 'molecular_weight_g_mol' instead."""
        warnings.warn(
            "MW is deprecated, use 'molecular_weight_g_mol' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.molecular_weight_g_mol

    @property
    def Number_Of_Electrons(self) -> float:
        """Deprecated: Use 'total_electrons' instead."""
        warnings.warn(
            "Number_Of_Electrons is deprecated, use 'total_electrons' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.total_electrons

    @property
    def Density(self) -> float:
        """Deprecated: Use 'density_g_cm3' instead."""
        warnings.warn(
            "Density is deprecated, use 'density_g_cm3' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.density_g_cm3

    @property
    def Electron_Density(self) -> float:
        """Deprecated: Use 'electron_density_per_ang3' instead."""
        warnings.warn(
            "Electron_Density is deprecated, use 'electron_density_per_ang3' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.electron_density_per_ang3

    @property
    def Energy(self) -> np.ndarray:
        """Deprecated: Use 'energy_kev' instead."""
        warnings.warn(
            "Energy is deprecated, use 'energy_kev' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.energy_kev

    @property
    def Wavelength(self) -> np.ndarray:
        """Deprecated: Use 'wavelength_angstrom' instead."""
        warnings.warn(
            "Wavelength is deprecated, use 'wavelength_angstrom' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.wavelength_angstrom

    @property
    def Dispersion(self) -> np.ndarray:
        """Deprecated: Use 'dispersion_delta' instead."""
        warnings.warn(
            "Dispersion is deprecated, use 'dispersion_delta' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.dispersion_delta

    @property
    def Absorption(self) -> np.ndarray:
        """Deprecated: Use 'absorption_beta' instead."""
        warnings.warn(
            "Absorption is deprecated, use 'absorption_beta' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.absorption_beta

    @property
    def f1(self) -> np.ndarray:
        """Deprecated: Use 'scattering_factor_f1' instead."""
        warnings.warn(
            "f1 is deprecated, use 'scattering_factor_f1' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.scattering_factor_f1

    @property
    def f2(self) -> np.ndarray:
        """Deprecated: Use 'scattering_factor_f2' instead."""
        warnings.warn(
            "f2 is deprecated, use 'scattering_factor_f2' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.scattering_factor_f2

    @property
    def Critical_Angle(self) -> np.ndarray:
        """Deprecated: Use 'critical_angle_degrees' instead."""
        warnings.warn(
            "Critical_Angle is deprecated, use 'critical_angle_degrees' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.critical_angle_degrees

    @property
    def Attenuation_Length(self) -> np.ndarray:
        """Deprecated: Use 'attenuation_length_cm' instead."""
        warnings.warn(
            "Attenuation_Length is deprecated, use 'attenuation_length_cm' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.attenuation_length_cm

    @property
    def reSLD(self) -> np.ndarray:
        """Deprecated: Use 'real_sld_per_ang2' instead."""
        warnings.warn(
            "reSLD is deprecated, use 'real_sld_per_ang2' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.real_sld_per_ang2

    @property
    def imSLD(self) -> np.ndarray:
        """Deprecated: Use 'imaginary_sld_per_ang2' instead."""
        warnings.warn(
            "imSLD is deprecated, use 'imaginary_sld_per_ang2' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.imaginary_sld_per_ang2

    @classmethod
    def from_legacy(
        cls,
        formula: str | None = None,
        mw: float | None = None,
        number_of_electrons: float | None = None,
        density: float | None = None,
        electron_density: float | None = None,
        energy: np.ndarray | None = None,
        wavelength: np.ndarray | None = None,
        dispersion: np.ndarray | None = None,
        absorption: np.ndarray | None = None,
        f1: np.ndarray | None = None,
        f2: np.ndarray | None = None,
        critical_angle: np.ndarray | None = None,
        attenuation_length: np.ndarray | None = None,
        re_sld: np.ndarray | None = None,
        im_sld: np.ndarray | None = None,
        **kwargs: Any,
    ) -> XRayResult:
        """Create XRayResult from legacy field names (for internal use)."""
        return cls(
            formula=formula or kwargs.get("formula", ""),
            molecular_weight_g_mol=mw or kwargs.get("molecular_weight_g_mol", 0.0),
            total_electrons=number_of_electrons or kwargs.get("total_electrons", 0.0),
            density_g_cm3=density or kwargs.get("density_g_cm3", 0.0),
            electron_density_per_ang3=(
                electron_density or kwargs.get("electron_density_per_ang3", 0.0)
            ),
            energy_kev=(
                energy if energy is not None else kwargs.get("energy_kev", np.array([]))
            ),
            wavelength_angstrom=(
                wavelength
                if wavelength is not None
                else kwargs.get("wavelength_angstrom", np.array([]))
            ),
            dispersion_delta=(
                dispersion
                if dispersion is not None
                else kwargs.get("dispersion_delta", np.array([]))
            ),
            absorption_beta=(
                absorption
                if absorption is not None
                else kwargs.get("absorption_beta", np.array([]))
            ),
            scattering_factor_f1=(
                f1
                if f1 is not None
                else kwargs.get("scattering_factor_f1", np.array([]))
            ),
            scattering_factor_f2=(
                f2
                if f2 is not None
                else kwargs.get("scattering_factor_f2", np.array([]))
            ),
            critical_angle_degrees=(
                critical_angle
                if critical_angle is not None
                else kwargs.get("critical_angle_degrees", np.array([]))
            ),
            attenuation_length_cm=(
                attenuation_length
                if attenuation_length is not None
                else kwargs.get("attenuation_length_cm", np.array([]))
            ),
            real_sld_per_ang2=(
                re_sld
                if re_sld is not None
                else kwargs.get("real_sld_per_ang2", np.array([]))
            ),
            imaginary_sld_per_ang2=(
                im_sld
                if im_sld is not None
                else kwargs.get("imaginary_sld_per_ang2", np.array([]))
            ),
        )
