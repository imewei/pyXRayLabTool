"""Scattering factor data loading and element path management."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Pre-computed element file paths for faster access
_AVAILABLE_ELEMENTS: dict[str, Path] = {}


def _initialize_element_paths() -> None:
    """
    Pre-compute all available element file paths at module load time.
    This optimization eliminates repeated file system checks.
    """

    base_paths = [
        Path.cwd() / "src" / "AtomicScatteringFactor",
        Path(__file__).parent.parent.parent
        / "src"
        / "AtomicScatteringFactor",  # For old structure compatibility
        Path(__file__).parent.parent
        / "data"
        / "AtomicScatteringFactor",  # New structure
    ]

    for base_path in base_paths:
        if base_path.exists():
            for nff_file in base_path.glob("*.nff"):
                element = nff_file.stem.capitalize()
                if element not in _AVAILABLE_ELEMENTS:
                    _AVAILABLE_ELEMENTS[element] = nff_file


def load_scattering_factor_data(element: str) -> Any:
    """
    Load f1/f2 scattering factor data for a specific element from .nff files.

    This function reads .nff files using CSV parsing and caches the results
    in a module-level dictionary keyed by element symbol. Returns a pandas-compatible
    data structure for accessing columns E, f1, f2.

    Args:
        element: Element symbol (e.g., 'H', 'C', 'N', 'O', 'Si', 'Ge')

    Returns:
        ScatteringData object with pandas-like interface containing columns: E (energy), f1, f2

    Raises:
        FileNotFoundError: If the .nff file for the element is not found
        ValueError: If the element symbol is invalid, empty, or file format is invalid

    Examples:
        >>> from xraylabtool.calculators.scattering_data import load_scattering_factor_data
        >>> data = load_scattering_factor_data('Si')
        >>> print(data.columns)
        ['E', 'f1', 'f2']
        >>> print(len(data) > 100)  # Verify we have enough data points
        True
    """
    from xraylabtool.calculators.cache import _scattering_factor_cache

    # Validate input
    if not element or not isinstance(element, str):
        raise ValueError(f"Element symbol must be a non-empty string, got: {element!r}")

    # Normalize element symbol (capitalize first letter, lowercase rest)
    element = element.capitalize()

    # Check if already cached
    if element in _scattering_factor_cache:
        return _scattering_factor_cache[element]

    # Use pre-computed element paths for faster access
    if element not in _AVAILABLE_ELEMENTS:
        raise FileNotFoundError(
            f"Scattering factor data file not found for element '{element}'. "
            f"Available elements: {sorted(_AVAILABLE_ELEMENTS.keys())}"
        )

    file_path = _AVAILABLE_ELEMENTS[element]

    try:
        # Read and validate header line
        with open(file_path) as _f:
            header_line = _f.readline().strip()

        header = [col.strip() for col in header_line.split(",")]
        expected_columns = {"E", "f1", "f2"}
        actual_columns = set(header)

        if not expected_columns.issubset(actual_columns):
            missing_cols = expected_columns - actual_columns
            raise ValueError(
                f"Invalid .nff file format for element '{element}'. "
                f"Missing required columns: {missing_cols}. "
                f"Found columns: {list(actual_columns)}"
            )

        # Get column indices for correct ordering
        e_idx = header.index("E")
        f1_idx = header.index("f1")
        f2_idx = header.index("f2")

        # Load entire file at C-level via np.loadtxt — 3-8x faster than csv.reader loop
        raw = np.loadtxt(file_path, delimiter=",", skiprows=1, dtype=np.float64)

        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        if len(raw) == 0:
            raise ValueError(
                "Empty scattering factor data file for element "
                f"'{element}': {file_path}"
            )

        # Re-order columns to canonical [E, f1, f2] if needed
        data_array = raw[:, [e_idx, f1_idx, f2_idx]]

        scattering_data = ScatteringData(data_array, ["E", "f1", "f2"])

        # Cache the data
        _scattering_factor_cache[element] = scattering_data

        return scattering_data

    except (OSError, ValueError) as e:
        raise ValueError(
            "Error parsing scattering factor data file for element "
            f"'{element}': {file_path}. "
            f"Expected CSV format with columns: E,f1,f2. Error: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            "Unexpected error loading scattering factor data for element "
            f"'{element}' from {file_path}: {e}"
        ) from e


class ScatteringData:
    """Pandas-like interface for scattering factor data arrays."""

    def __init__(self, data_array: np.ndarray, column_names: list[str]) -> None:
        self.data = data_array
        self.columns = column_names
        self._column_indices = {name: i for i, name in enumerate(column_names)}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, column: str) -> Any:
        idx = self._column_indices[column]

        # Return object with .values attribute for compatibility
        class ColumnProxy:
            def __init__(self, data: np.ndarray) -> None:
                self.values = data

        return ColumnProxy(self.data[:, idx])


class AtomicScatteringFactor:
    """
    Class for handling atomic scattering factors.

    This class loads and manages atomic scattering factor data
    from NFF files using the module-level cache.
    """

    def __init__(self) -> None:
        # Maintain backward compatibility with existing tests
        self.data: dict[str, Any] = {}
        self.data_path = (
            Path(__file__).parent.parent / "data" / "AtomicScatteringFactor"
        )

        # Create data directory if it doesn't exist (for test compatibility)
        self.data_path.mkdir(parents=True, exist_ok=True)

    def load_element_data(self, element: str) -> Any:
        """
        Load scattering factor data for a specific element.

        Args:
            element: Element symbol (e.g., 'H', 'C', 'N', 'O', 'Si', 'Ge')

        Returns:
            DataFrame containing scattering factor data with columns: E, f1, f2

        Raises:
            FileNotFoundError: If the .nff file for the element is not found
            ValueError: If the element symbol is invalid
        """
        return load_scattering_factor_data(element)

    def get_scattering_factor(self, _element: str, q_values: np.ndarray) -> np.ndarray:
        """
        Calculate scattering factors for given q values.

        Args:
            element: Element symbol
            q_values: Array of momentum transfer values

        Returns:
            Array of scattering factor values
        """
        # Placeholder implementation
        return np.ones_like(q_values)


class CrystalStructure:
    """
    Class for representing and manipulating crystal structures.
    """

    def __init__(
        self, lattice_parameters: tuple[float, float, float, float, float, float]
    ):
        """
        Initialize crystal structure.

        Args:
            lattice_parameters: (a, b, c, alpha, beta, gamma) in Angstroms and degrees
        """
        self.a, self.b, self.c, self.alpha, self.beta, self.gamma = lattice_parameters
        self.atoms: list[dict[str, Any]] = []

    def add_atom(
        self, element: str, position: tuple[float, float, float], occupancy: float = 1.0
    ) -> None:
        """
        Add an atom to the crystal structure.

        Args:
            element: Element symbol
            position: Fractional coordinates (x, y, z)
            occupancy: Site occupancy factor
        """
        self.atoms.append(
            {"element": element, "position": position, "occupancy": occupancy}
        )

    def calculate_structure_factor(self, _hkl: tuple[int, int, int]) -> complex:
        """
        Calculate structure factor for given Miller indices.

        Args:
            hkl: Miller indices (h, k, l)

        Returns:
            Complex structure factor
        """
        # Placeholder implementation
        return complex(1.0, 0.0)


def load_data_file(filename: str) -> Any:
    """
    Load data from various file formats commonly used in X-ray analysis.

    Args:
        filename: Path to the data file

    Returns:
        DataFrame containing the loaded data
    """
    file_path = Path(filename)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {filename}")

    # Lazy import pandas only when needed
    import pandas as pd

    # Determine file format and load accordingly
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    elif file_path.suffix.lower() in [".txt", ".dat"]:
        return pd.read_csv(file_path, delim_whitespace=True)  # type: ignore[call-overload]
    else:
        # Try to load as generic text file
        return pd.read_csv(file_path, delim_whitespace=True)  # type: ignore[call-overload]


# Initialize element paths at module import time for performance
_initialize_element_paths()
