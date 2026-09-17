"""Implementation of the 'list' command."""

from typing import Any


def cmd_list(args: Any) -> int:
    """Handle the 'list' command."""
    if args.type == "constants":
        print("Physical Constants:")
        print("=" * 40)
        from xraylabtool import constants

        const_names = [
            "THOMPSON",
            "SPEED_OF_LIGHT",
            "PLANCK",
            "ELEMENT_CHARGE",
            "AVOGADRO",
            "ENERGY_TO_WAVELENGTH_FACTOR",
            "PI",
            "TWO_PI",
        ]
        for name in const_names:
            if hasattr(constants, name):
                value = getattr(constants, name)
                print(f"{name: <25}: {value}")

    elif args.type == "fields":
        print("Available XRayResult Fields (new snake_case names):")
        print("=" * 60)
        field_descriptions = [
            ("formula", "Chemical formula string"),
            ("molecular_weight_g_mol", "Molecular weight (g/mol)"),
            ("total_electrons", "Total electrons per molecule"),
            ("density_g_cm3", "Mass density (g/cm³)"),
            ("electron_density_per_ang3", "Electron density (electrons/Å³)"),
            ("energy_kev", "X-ray energies (keV)"),
            ("wavelength_angstrom", "X-ray wavelengths (Å)"),
            ("dispersion_delta", "Dispersion coefficient δ"),
            ("absorption_beta", "Absorption coefficient β"),
            ("scattering_factor_f1", "Real atomic scattering factor"),
            ("scattering_factor_f2", "Imaginary atomic scattering factor"),
            ("critical_angle_degrees", "Critical angles (degrees)"),
            ("attenuation_length_cm", "Attenuation lengths (cm)"),
            ("real_sld_per_ang2", "Real SLD (Å⁻²)"),
            ("imaginary_sld_per_ang2", "Imaginary SLD (Å⁻²)"),
        ]

        for field, description in field_descriptions:
            print(f"{field: <25}: {description}")

    elif args.type == "examples":
        print("CLI Usage Examples:")
        print("=" * 40)
        examples = [
            ("Single material calculation", "xraylabtool calc SiO2 -e 10.0 -d 2.2"),
            ("Multiple energies", "xraylabtool calc Si -e 5.0,10.0,15.0 -d 2.33"),
            ("Energy range", "xraylabtool calc Al2O3 -e 5-15:11 -d 3.95"),
            ("Save to CSV", "xraylabtool calc SiO2 -e 10.0 -d 2.2 -o results.csv"),
            ("Batch processing", "xraylabtool batch materials.csv -o results.csv"),
            ("Unit conversion", "xraylabtool convert energy 10.0 --to wavelength"),
            ("Formula parsing", "xraylabtool formula SiO2 --verbose"),
            ("Bragg angles", "xraylabtool bragg -d 3.14 -e 8.0"),
            ("Install completion", "xraylabtool install-completion"),
        ]

        for description, command in examples:
            print(f"\n{description}:")
            print(f"  {command}")

    return 0
