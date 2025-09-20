"""
Enhanced type definitions for XRayLabTool.

This module provides performance-optimized type aliases and definitions
specifically designed for scientific computing applications. It includes
specialized NumPy array types, protocol definitions, and type helpers
that enable both type safety and high-performance vectorized operations.

Performance Note:
- All type definitions use TYPE_CHECKING to avoid runtime overhead
- NumPy array types specify dtypes for optimal memory layout
- Protocol definitions enable duck typing without inheritance overhead
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Any, Protocol, TypeVar

    import numpy as np
    from numpy.typing import NDArray

    # =============================================================================
    # NumPy Array Types for Scientific Computing Performance
    # =============================================================================
    # Primary numerical types for X-ray calculations
    EnergyArray = NDArray[np.float64]  # Energy values in keV
    WavelengthArray = NDArray[np.float64]  # Wavelengths in Angstrom
    DensityArray = NDArray[np.float64]  # Density values in g/cm³
    AngleArray = NDArray[np.float64]  # Angles in degrees

    # Complex arrays for scattering factors
    ComplexArray = NDArray[np.complex128]  # Complex scattering factors (f1 + if2)
    ScatteringFactorArray = ComplexArray  # Alias for clarity

    # Specialized arrays for derived quantities
    OpticalConstantArray = NDArray[np.float64]  # Dispersion (δ) and absorption (β)
    AttenuationArray = NDArray[np.float64]  # Attenuation lengths in cm
    CriticalAngleArray = NDArray[np.float64]  # Critical angles in degrees
    SLDArray = NDArray[np.float64]  # Scattering length density in Å⁻²

    # Generic numerical arrays
    RealArray = NDArray[np.floating[Any]]  # Any real floating-point array
    IntArray = NDArray[np.integer[Any]]  # Any integer array
    BoolArray = NDArray[np.bool_]  # Boolean arrays for masks

    # Flexible array types for input validation
    FloatLike = float | np.floating[Any] | EnergyArray
    ArrayLike = Sequence[float] | NDArray[Any]

    # =============================================================================
    # Protocol Definitions for Performance-Critical Interfaces
    # =============================================================================

    class AtomicDataProvider(Protocol):
        """Protocol for atomic data sources with performance guarantees."""

        def get_scattering_factors(
            self, element: str, energies: EnergyArray
        ) -> ComplexArray:
            """
            Get atomic scattering factors for element at given energies.

            Parameters
            ----------
            element : str
                Chemical element symbol (e.g., 'Si', 'O')
            energies : EnergyArray
                X-ray energies in keV

            Returns
            -------
            ComplexArray
                Complex scattering factors (f1 + if2)
            """
            ...

        def is_element_cached(self, element: str) -> bool:
            """Check if element data is cached for fast access."""
            ...

    class InterpolatorProtocol(Protocol):
        """Protocol for high-performance interpolation functions."""

        def __call__(self, x: EnergyArray) -> RealArray:
            """Interpolate values at given x coordinates."""
            ...

        @property
        def extrapolation_mode(self) -> str:
            """Get extrapolation behavior ('linear', 'constant', 'error')."""
            ...

    class CalculationEngine(Protocol):
        """Protocol for X-ray calculation engines."""

        def calculate_optical_constants(
            self,
            formula: str,
            energies: EnergyArray,
            density: float,
        ) -> tuple[OpticalConstantArray, OpticalConstantArray]:
            """
            Calculate dispersion and absorption coefficients.

            Returns
            -------
            tuple[OpticalConstantArray, OpticalConstantArray]
                Dispersion (δ) and absorption (β) coefficients
            """
            ...

        def calculate_derived_quantities(
            self,
            dispersion: OpticalConstantArray,
            absorption: OpticalConstantArray,
            energies: EnergyArray,
        ) -> dict[str, RealArray]:
            """Calculate derived quantities from optical constants."""
            ...

    # =============================================================================
    # Type Variables for Generic Functions
    # =============================================================================

    T = TypeVar("T", bound=np.floating[Any])
    ArrayT = TypeVar("ArrayT", bound=NDArray[Any])
    NumericT = TypeVar("NumericT", int, float, complex)

    # =============================================================================
    # Function Type Signatures for Performance-Critical Operations
    # =============================================================================

    # Vectorized calculation function signatures
    VectorizedCalculator = Callable[[EnergyArray, ...], RealArray]
    ElementProcessor = Callable[[str, EnergyArray], ComplexArray]
    FormulaParser = Callable[[str], Mapping[str, int]]

    # Cache function signatures
    CacheValidator = Callable[[str], bool]
    CacheLoader = Callable[[str], Any]

    # =============================================================================
    # Data Structure Types
    # =============================================================================

    # Chemical formula representation
    ElementComposition = Mapping[str, int]  # Element -> count mapping

    # Calculation results structure
    CalculationResults = Mapping[str, str | float | RealArray]

    # Configuration and settings
    CalculationConfig = Mapping[str, Any]
    CacheConfig = Mapping[str, str | int | bool]

    # =============================================================================
    # Performance Validation Types
    # =============================================================================

    class PerformanceMetrics(Protocol):
        """Protocol for performance measurement and validation."""

        @property
        def calculations_per_second(self) -> float:
            """Get current calculation rate in calculations/second."""
            ...

        @property
        def memory_usage_mb(self) -> float:
            """Get current memory usage in MB."""
            ...

        def validate_performance_target(self, target_cps: float = 150_000.0) -> bool:
            """Validate that performance meets target calculations/second."""
            ...

else:
    # Runtime fallbacks for when TYPE_CHECKING is False
    # These ensure the module can be imported without errors during runtime

    # Simple runtime aliases that don't affect performance
    import numpy as np
    from numpy.typing import NDArray

    # Basic array types available at runtime
    EnergyArray = NDArray[np.float64]
    WavelengthArray = NDArray[np.float64]
    ComplexArray = NDArray[np.complex128]
    RealArray = NDArray[np.floating]

    # Runtime type aliases for basic functionality
    FloatLike = float | np.floating | NDArray
    ArrayLike = list | NDArray

    # Protocol placeholders (will be ignored at runtime)
    class AtomicDataProvider:
        pass

    class InterpolatorProtocol:
        pass

    class CalculationEngine:
        pass

    class PerformanceMetrics:
        pass


# =============================================================================
# Runtime Helper Functions for Type Validation
# =============================================================================


def validate_energy_array(energies: Any) -> bool:
    """
    Validate that input is a proper energy array.

    Parameters
    ----------
    energies : Any
        Input to validate

    Returns
    -------
    bool
        True if valid energy array
    """
    import numpy as np

    if not isinstance(energies, np.ndarray):
        return False

    if energies.dtype not in [np.float32, np.float64]:
        return False

    if energies.ndim != 1:
        return False

    # Check for valid energy range (0.03-30 keV)
    return not (np.any(energies < 0.03) or np.any(energies > 30.0))


def validate_complex_array(array: Any) -> bool:
    """
    Validate that input is a proper complex scattering factor array.

    Parameters
    ----------
    array : Any
        Input to validate

    Returns
    -------
    bool
        True if valid complex array
    """
    import numpy as np

    if not isinstance(array, np.ndarray):
        return False

    return array.dtype in [np.complex64, np.complex128]


def ensure_float64_array(array: ArrayLike) -> EnergyArray:
    """
    Ensure input is converted to float64 array for performance.

    Parameters
    ----------
    array : ArrayLike
        Input array or sequence

    Returns
    -------
    EnergyArray
        Converted float64 array
    """
    import numpy as np

    return np.asarray(array, dtype=np.float64)


def ensure_complex128_array(array: ArrayLike) -> ComplexArray:
    """
    Ensure input is converted to complex128 array for performance.

    Parameters
    ----------
    array : ArrayLike
        Input array or sequence

    Returns
    -------
    ComplexArray
        Converted complex128 array
    """
    import numpy as np

    return np.asarray(array, dtype=np.complex128)


# =============================================================================
# Type Guards for Runtime Type Checking
# =============================================================================


def is_energy_array(obj: Any) -> bool:
    """Type guard for energy arrays."""
    return validate_energy_array(obj)


def is_complex_array(obj: Any) -> bool:
    """Type guard for complex arrays."""
    return validate_complex_array(obj)


def is_real_array(obj: Any) -> bool:
    """Type guard for real arrays."""
    import numpy as np

    return isinstance(obj, np.ndarray) and np.issubdtype(obj.dtype, np.floating)


# =============================================================================
# Performance Optimization Helpers
# =============================================================================


def optimize_array_memory_layout(array: NDArray) -> NDArray:
    """
    Optimize array memory layout for performance.

    Parameters
    ----------
    array : NDArray
        Input array

    Returns
    -------
    NDArray
        Memory-optimized array
    """
    import numpy as np

    # Ensure C-contiguous layout for best performance
    if not array.flags.c_contiguous:
        return np.ascontiguousarray(array)
    return array


def get_optimal_chunk_size(array_length: int, memory_limit_mb: float = 100.0) -> int:
    """
    Calculate optimal chunk size for batch processing.

    Parameters
    ----------
    array_length : int
        Total array length
    memory_limit_mb : float
        Memory limit in MB

    Returns
    -------
    int
        Optimal chunk size
    """
    # Estimate memory per element (8 bytes for float64)
    bytes_per_element = 8
    memory_limit_bytes = memory_limit_mb * 1024 * 1024

    max_chunk_size = int(memory_limit_bytes / bytes_per_element)
    return min(array_length, max_chunk_size)
