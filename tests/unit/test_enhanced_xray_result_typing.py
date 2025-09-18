"""
Tests for enhanced XRayResult dataclass typing and type safety.

This module validates that the XRayResult dataclass correctly handles
type annotations, provides proper type validation, and maintains
performance characteristics with enhanced type safety.
"""

from typing import TYPE_CHECKING
import warnings

import numpy as np
import pytest

if TYPE_CHECKING:
    pass

from tests.fixtures.test_base import BaseXRayLabToolTest
from xraylabtool.calculators.core import XRayResult
from xraylabtool.typing_extensions import (
    ensure_complex128_array,
    ensure_float64_array,
    validate_energy_array,
)


class TestXRayResultTyping(BaseXRayLabToolTest):
    """Test suite for XRayResult dataclass typing enhancements."""

    def test_xray_result_field_types(self):
        """Test that XRayResult fields have correct type annotations."""

        # Get type hints for XRayResult
        hints = XRayResult.__annotations__

        # Test scalar field types (with __future__ annotations, all become strings)
        assert hints["formula"] == "str"
        assert hints["molecular_weight_g_mol"] == "float"
        assert hints["total_electrons"] == "float"
        assert hints["density_g_cm3"] == "float"
        assert hints["electron_density_per_ang3"] == "float"

        # Test array field types (these are TYPE_CHECKING imports, so no quotes in annotations)
        assert hints["energy_kev"] == "EnergyArray"
        assert hints["wavelength_angstrom"] == "WavelengthArray"
        assert hints["dispersion_delta"] == "OpticalConstantArray"
        assert hints["absorption_beta"] == "OpticalConstantArray"
        assert hints["scattering_factor_f1"] == "OpticalConstantArray"
        assert hints["scattering_factor_f2"] == "OpticalConstantArray"
        assert hints["critical_angle_degrees"] == "OpticalConstantArray"
        assert hints["attenuation_length_cm"] == "OpticalConstantArray"
        assert hints["real_sld_per_ang2"] == "OpticalConstantArray"
        assert hints["imaginary_sld_per_ang2"] == "OpticalConstantArray"

    def test_xray_result_creation_with_correct_types(self):
        """Test XRayResult creation with properly typed inputs."""
        # Create test data with correct types
        formula = "SiO2"
        molecular_weight = 60.08
        total_electrons = 30.0
        density = 2.2
        electron_density = 0.066

        # Create properly typed arrays
        energies = np.array([10.0], dtype=np.float64)
        wavelengths = np.array([1.24], dtype=np.float64)
        dispersion = np.array([1.74e-6], dtype=np.float64)
        absorption = np.array([8.85e-8], dtype=np.float64)
        f1_values = np.array([14.0], dtype=np.float64)
        f2_values = np.array([0.1], dtype=np.float64)
        critical_angles = np.array([0.174], dtype=np.float64)
        attenuation_lengths = np.array([15.2], dtype=np.float64)
        real_sld = np.array([2.1e-5], dtype=np.float64)
        imag_sld = np.array([1.0e-7], dtype=np.float64)

        # Create XRayResult instance
        result = XRayResult(
            formula=formula,
            molecular_weight_g_mol=molecular_weight,
            total_electrons=total_electrons,
            density_g_cm3=density,
            electron_density_per_ang3=electron_density,
            energy_kev=energies,
            wavelength_angstrom=wavelengths,
            dispersion_delta=dispersion,
            absorption_beta=absorption,
            scattering_factor_f1=f1_values,
            scattering_factor_f2=f2_values,
            critical_angle_degrees=critical_angles,
            attenuation_length_cm=attenuation_lengths,
            real_sld_per_ang2=real_sld,
            imaginary_sld_per_ang2=imag_sld,
        )

        # Verify types are preserved
        assert isinstance(result.formula, str)
        assert isinstance(result.molecular_weight_g_mol, float)
        assert isinstance(result.total_electrons, float)
        assert isinstance(result.density_g_cm3, float)
        assert isinstance(result.electron_density_per_ang3, float)

        # Verify array types and dtypes
        assert isinstance(result.energy_kev, np.ndarray)
        assert result.energy_kev.dtype == np.float64
        assert isinstance(result.wavelength_angstrom, np.ndarray)
        assert result.wavelength_angstrom.dtype == np.float64
        assert isinstance(result.dispersion_delta, np.ndarray)
        assert result.dispersion_delta.dtype == np.float64

    def test_xray_result_array_conversion(self):
        """Test that XRayResult properly converts input arrays to numpy arrays."""
        # Test with Python lists (should be converted to numpy arrays)
        result = XRayResult(
            formula="Si",
            molecular_weight_g_mol=28.09,
            total_electrons=14.0,
            density_g_cm3=2.33,
            electron_density_per_ang3=0.1,
            energy_kev=[8.0, 10.0, 12.0],  # Python list
            wavelength_angstrom=[1.55, 1.24, 1.03],  # Python list
            dispersion_delta=[2.1e-6, 1.7e-6, 1.4e-6],
            absorption_beta=[1.2e-7, 8.8e-8, 6.5e-8],
            scattering_factor_f1=[13.8, 13.9, 14.0],
            scattering_factor_f2=[0.08, 0.10, 0.12],
            critical_angle_degrees=[0.191, 0.174, 0.158],
            attenuation_length_cm=[18.5, 15.2, 12.8],
            real_sld_per_ang2=[2.3e-5, 2.1e-5, 1.9e-5],
            imaginary_sld_per_ang2=[1.5e-7, 1.0e-7, 8.0e-8],
        )

        # Verify all arrays are numpy arrays
        assert isinstance(result.energy_kev, np.ndarray)
        assert isinstance(result.wavelength_angstrom, np.ndarray)
        assert isinstance(result.dispersion_delta, np.ndarray)
        assert isinstance(result.absorption_beta, np.ndarray)
        assert isinstance(result.scattering_factor_f1, np.ndarray)
        assert isinstance(result.scattering_factor_f2, np.ndarray)
        assert isinstance(result.critical_angle_degrees, np.ndarray)
        assert isinstance(result.attenuation_length_cm, np.ndarray)
        assert isinstance(result.real_sld_per_ang2, np.ndarray)
        assert isinstance(result.imaginary_sld_per_ang2, np.ndarray)

        # Verify array lengths are consistent
        assert len(result.energy_kev) == 3
        assert len(result.wavelength_angstrom) == 3
        assert len(result.critical_angle_degrees) == 3

    def test_xray_result_array_dtype_consistency(self):
        """Test that XRayResult arrays maintain consistent dtypes for performance."""
        # Create result with mixed input types
        result = XRayResult(
            formula="Al2O3",
            molecular_weight_g_mol=101.96,
            total_electrons=50.0,
            density_g_cm3=3.95,
            electron_density_per_ang3=0.12,
            energy_kev=np.array([10], dtype=np.float32),  # float32 input
            wavelength_angstrom=np.array([1.24], dtype=np.float64),  # float64 input
            dispersion_delta=[1.5e-6],  # Python list
            absorption_beta=np.array([7.2e-8]),  # default numpy dtype
            scattering_factor_f1=np.array([42.0]),
            scattering_factor_f2=np.array([0.15]),
            critical_angle_degrees=np.array([0.162]),
            attenuation_length_cm=np.array([14.8]),
            real_sld_per_ang2=np.array([1.8e-5]),
            imaginary_sld_per_ang2=np.array([9.5e-8]),
        )

        # All arrays should be converted to appropriate dtypes
        # (The current implementation uses np.asarray which preserves reasonable dtypes)
        assert result.energy_kev.dtype in [np.float32, np.float64]
        assert result.wavelength_angstrom.dtype == np.float64
        assert result.dispersion_delta.dtype in [np.float32, np.float64]

    def test_xray_result_legacy_property_warnings(self):
        """Test that legacy property access emits deprecation warnings."""
        # Create a simple XRayResult
        result = XRayResult(
            formula="C",
            molecular_weight_g_mol=12.01,
            total_electrons=6.0,
            density_g_cm3=3.52,
            electron_density_per_ang3=0.18,
            energy_kev=np.array([10.0]),
            wavelength_angstrom=np.array([1.24]),
            dispersion_delta=np.array([8.9e-7]),
            absorption_beta=np.array([4.1e-8]),
            scattering_factor_f1=np.array([5.8]),
            scattering_factor_f2=np.array([0.04]),
            critical_angle_degrees=np.array([0.125]),
            attenuation_length_cm=np.array([21.5]),
            real_sld_per_ang2=np.array([3.2e-5]),
            imaginary_sld_per_ang2=np.array([5.8e-8]),
        )

        # Test that legacy properties emit warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Access legacy property
            formula = result.Formula
            assert formula == "C"

            # Check that warning was emitted
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    @pytest.mark.performance
    def test_xray_result_memory_efficiency(self):
        """Test that XRayResult maintains memory efficiency with type annotations."""

        # Create results with different array sizes
        small_result = self._create_test_result(array_size=10)
        large_result = self._create_test_result(array_size=1000)

        # Memory usage should scale reasonably with array size
        # Use total memory of all arrays instead of object size
        def get_total_array_memory(result):
            return sum(
                getattr(result, field).nbytes
                for field in [
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
            )

        small_memory = get_total_array_memory(small_result)
        large_memory = get_total_array_memory(large_result)

        # Large result should use proportionally more memory
        # (1000 elements vs 10 elements = 100x scaling)
        memory_ratio = large_memory / small_memory
        assert 50 < memory_ratio < 150  # Allow some variance in scaling

        # Verify arrays are using appropriate dtypes
        assert small_result.energy_kev.dtype in [np.float32, np.float64]
        assert large_result.energy_kev.dtype in [np.float32, np.float64]

    def test_typing_extension_validation_helpers(self):
        """Test that typing extension validation helpers work correctly."""
        # Test energy array validation
        valid_energies = np.array([5.0, 10.0, 15.0], dtype=np.float64)
        invalid_energies_type = [5.0, 10.0, 15.0]  # Python list
        invalid_energies_range = np.array(
            [0.01, 50.0], dtype=np.float64
        )  # Out of range

        assert validate_energy_array(valid_energies) is True
        assert validate_energy_array(invalid_energies_type) is False
        assert validate_energy_array(invalid_energies_range) is False

        # Test array conversion helpers
        input_list = [1.0, 2.0, 3.0]
        converted_float = ensure_float64_array(input_list)
        assert isinstance(converted_float, np.ndarray)
        assert converted_float.dtype == np.float64

        converted_complex = ensure_complex128_array(input_list)
        assert isinstance(converted_complex, np.ndarray)
        assert converted_complex.dtype == np.complex128

    def test_xray_result_type_guards(self):
        """Test type guard functionality for XRayResult arrays."""
        from xraylabtool.typing_extensions import (
            is_complex_array,
            is_energy_array,
            is_real_array,
        )

        # Create test arrays
        energy_array = np.array([8.0, 10.0, 12.0], dtype=np.float64)
        complex_array = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
        real_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        invalid_array = ["not", "an", "array"]

        # Test type guards
        assert is_energy_array(energy_array) is True
        assert is_energy_array(invalid_array) is False

        assert is_real_array(real_array) is True
        assert is_real_array(complex_array) is False
        assert is_real_array(invalid_array) is False

        assert is_complex_array(complex_array) is True
        assert is_complex_array(real_array) is False
        assert is_complex_array(invalid_array) is False

    def test_xray_result_performance_with_types(self):
        """Test that type annotations don't impact XRayResult performance."""
        import time

        # Time creation of XRayResult instances
        n_iterations = 100
        array_size = 50

        start_time = time.time()
        for _ in range(n_iterations):
            result = self._create_test_result(array_size)
            # Access properties to ensure they're computed
            _ = result.formula
            _ = result.energy_kev.mean()
            _ = result.critical_angle_degrees.max()

        total_time = time.time() - start_time
        time_per_creation = total_time / n_iterations

        # Should be fast (< 1ms per creation for reasonable array sizes)
        assert time_per_creation < 0.001, (
            f"XRayResult creation too slow: {time_per_creation:.4f}s"
        )

    def _create_test_result(self, array_size: int = 10) -> XRayResult:
        """Helper method to create test XRayResult instances."""
        energies = np.linspace(5.0, 15.0, array_size)

        return XRayResult(
            formula="TestMaterial",
            molecular_weight_g_mol=50.0,
            total_electrons=25.0,
            density_g_cm3=2.5,
            electron_density_per_ang3=0.1,
            energy_kev=energies,
            wavelength_angstrom=1.24 / energies,
            dispersion_delta=np.full(array_size, 1.5e-6),
            absorption_beta=np.full(array_size, 8.0e-8),
            scattering_factor_f1=np.full(array_size, 20.0),
            scattering_factor_f2=np.full(array_size, 0.1),
            critical_angle_degrees=np.full(array_size, 0.15),
            attenuation_length_cm=np.full(array_size, 12.0),
            real_sld_per_ang2=np.full(array_size, 2.0e-5),
            imaginary_sld_per_ang2=np.full(array_size, 1.0e-7),
        )


class TestXRayResultProtocolCompliance:
    """Test that XRayResult can work with protocol-based interfaces."""

    def test_xray_result_as_calculation_result(self):
        """Test that XRayResult can be used in protocol-based contexts."""
        # This test validates that XRayResult works with duck typing
        # and protocol-based interfaces as defined in typing_extensions

        # Create a test result
        result = XRayResult(
            formula="ProtocolTest",
            molecular_weight_g_mol=75.0,
            total_electrons=37.0,
            density_g_cm3=3.0,
            electron_density_per_ang3=0.08,
            energy_kev=np.array([10.0]),
            wavelength_angstrom=np.array([1.24]),
            dispersion_delta=np.array([1.2e-6]),
            absorption_beta=np.array([6.5e-8]),
            scattering_factor_f1=np.array([30.0]),
            scattering_factor_f2=np.array([0.08]),
            critical_angle_degrees=np.array([0.145]),
            attenuation_length_cm=np.array([16.2]),
            real_sld_per_ang2=np.array([1.5e-5]),
            imaginary_sld_per_ang2=np.array([7.8e-8]),
        )

        # Test that result has required attributes for protocol compliance
        assert hasattr(result, "formula")
        assert hasattr(result, "energy_kev")
        assert hasattr(result, "dispersion_delta")
        assert hasattr(result, "absorption_beta")

        # Test that arrays are accessible and properly typed
        assert isinstance(result.energy_kev, np.ndarray)
        assert isinstance(result.dispersion_delta, np.ndarray)
        assert isinstance(result.absorption_beta, np.ndarray)

        # Test that protocol-expected operations work
        assert len(result.energy_kev) == len(result.dispersion_delta)
        assert len(result.energy_kev) == len(result.absorption_beta)

    def test_xray_result_array_interface(self):
        """Test that XRayResult arrays support standard array operations."""
        result = XRayResult(
            formula="ArrayTest",
            molecular_weight_g_mol=100.0,
            total_electrons=50.0,
            density_g_cm3=4.0,
            electron_density_per_ang3=0.12,
            energy_kev=np.array([8.0, 10.0, 12.0]),
            wavelength_angstrom=np.array([1.55, 1.24, 1.03]),
            dispersion_delta=np.array([2.1e-6, 1.7e-6, 1.4e-6]),
            absorption_beta=np.array([1.2e-7, 8.8e-8, 6.5e-8]),
            scattering_factor_f1=np.array([45.0, 46.0, 47.0]),
            scattering_factor_f2=np.array([0.12, 0.10, 0.08]),
            critical_angle_degrees=np.array([0.191, 0.174, 0.158]),
            attenuation_length_cm=np.array([18.5, 15.2, 12.8]),
            real_sld_per_ang2=np.array([2.3e-5, 2.1e-5, 1.9e-5]),
            imaginary_sld_per_ang2=np.array([1.5e-7, 1.0e-7, 8.0e-8]),
        )

        # Test standard array operations
        assert result.energy_kev.min() == 8.0
        assert result.energy_kev.max() == 12.0
        assert result.energy_kev.mean() == 10.0

        # Test indexing
        assert result.energy_kev[0] == 8.0
        assert result.wavelength_angstrom[1] == 1.24
        assert result.critical_angle_degrees[2] == 0.158

        # Test array slicing
        assert len(result.energy_kev[1:]) == 2
        assert len(result.dispersion_delta[:2]) == 2

        # Test boolean indexing would work
        mask = result.energy_kev > 9.0
        assert len(result.energy_kev[mask]) == 2
