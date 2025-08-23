"""
High-performance batch processing module for X-ray calculations.

This module provides optimized batch processing capabilities with memory management,
parallel execution, and progress tracking for large-scale X-ray property calculations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterator, Any
import concurrent.futures
from dataclasses import dataclass
import gc
import psutil
import os
import warnings

from .core import calculate_single_material_properties, XRayResult


@dataclass
class BatchConfig:
    """
    Configuration for batch processing operations.

    Args:
        max_workers: Maximum number of parallel workers (default: auto-detect)
        chunk_size: Number of calculations per chunk (default: 100)
        memory_limit_gb: Memory limit in GB before forcing garbage collection
        enable_progress: Whether to show progress bars
        cache_results: Whether to cache intermediate results
    """

    max_workers: Optional[int] = None
    chunk_size: int = 100
    memory_limit_gb: float = 4.0
    enable_progress: bool = True
    cache_results: bool = False

    def __post_init__(self) -> None:
        if self.max_workers is None:
            # Auto-detect optimal worker count based on system capabilities
            cpu_count = os.cpu_count() or 1
            # Use 75% of available CPUs, but cap at 8 for memory efficiency
            self.max_workers = min(max(1, int(cpu_count * 0.75)), 8)


class MemoryMonitor:
    """
    Memory usage monitor for batch operations.
    """

    def __init__(self, limit_gb: float = 4.0):
        self.limit_bytes = limit_gb * 1024 * 1024 * 1024
        self.process = psutil.Process()

    def check_memory(self) -> bool:
        """
        Check if memory usage is below limit.

        Returns:
            True if within limits, False if exceeded
        """
        try:
            memory_info = self.process.memory_info()
            return bool(memory_info.rss < self.limit_bytes)
        except Exception:
            return True  # If we can't check, assume it's fine

    def get_memory_usage_mb(self) -> float:
        """
        Get current memory usage in MB.

        Returns:
            Memory usage in megabytes
        """
        try:
            memory_info = self.process.memory_info()
            return float(memory_info.rss / (1024 * 1024))
        except Exception:
            return 0.0

    def force_gc(self) -> None:
        """
        Force garbage collection to free memory.
        """
        gc.collect()


def chunk_iterator(
    data: List[Tuple[Any, ...]], chunk_size: int
) -> Iterator[List[Tuple[Any, ...]]]:
    """
    Yield successive chunks of data.

    Args:
        data: List of data tuples to chunk
        chunk_size: Size of each chunk

    Yields:
        Lists of data tuples of specified chunk size
    """
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def process_single_calculation(
    formula: str, energies: np.ndarray, density: float
) -> Tuple[str, Optional[XRayResult]]:
    """
    Process a single X-ray calculation.

    Args:
        formula: Chemical formula
        energies: Energy array
        density: Material density

    Returns:
        Tuple of (formula, XRayResult)
    """
    try:
        result = calculate_single_material_properties(formula, energies, density)
        return (formula, result)
    except Exception as e:
        warnings.warn(f"Failed to process formula '{formula}': {e}")
        return (formula, None)


def process_batch_chunk(
    chunk: List[Tuple[str, np.ndarray, float]], config: BatchConfig
) -> List[Tuple[str, Optional[XRayResult]]]:
    """
    Process a chunk of calculations in parallel.

    Args:
        chunk: List of (formula, energies, density) tuples
        config: Batch processing configuration

    Returns:
        List of (formula, result) tuples
    """
    results = []
    memory_monitor = MemoryMonitor(config.memory_limit_gb)

    # Use ThreadPoolExecutor for I/O bound operations (file loading)
    # ProcessPoolExecutor would be better for CPU-bound, but has pickle
    # overhead
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config.max_workers
    ) as executor:
        # Submit all calculations in the chunk
        future_to_formula = {
            executor.submit(
                process_single_calculation, formula, energies, density
            ): formula
            for formula, energies, density in chunk
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_formula):
            formula = future_to_formula[future]
            try:
                # 5 minute timeout per calculation
                result = future.result(timeout=300)
                results.append(result)

                # Memory management
                if not memory_monitor.check_memory():
                    memory_monitor.force_gc()

            except concurrent.futures.TimeoutError:
                warnings.warn(f"Timeout processing formula '{formula}'")
                results.append((formula, None))
            except Exception as e:
                warnings.warn(f"Error processing formula '{formula}': {e}")
                results.append((formula, None))

    return results


def _validate_batch_inputs(formulas: List[str], densities: List[float]) -> None:
    """Validate batch processing inputs."""
    if len(formulas) != len(densities):
        raise ValueError("Number of formulas must match number of densities")
    if not formulas:
        raise ValueError("Formula list cannot be empty")


def _prepare_energies_array(
    energies: Union[float, List[float], np.ndarray],
) -> np.ndarray:
    """Convert energies to numpy array and validate."""
    if np.isscalar(energies):
        # Handle scalar values including complex numbers
        if isinstance(energies, complex):
            energies_array = np.array([float(energies.real)], dtype=np.float64)
        else:
            energies_array = np.array([float(energies)], dtype=np.float64)
    else:
        energies_array = np.array(energies, dtype=np.float64)

    if np.any(energies_array <= 0):
        raise ValueError("All energies must be positive")
    if np.any(energies_array < 0.03) or np.any(energies_array > 30):
        raise ValueError("Energy values must be in range 0.03-30 keV")

    return energies_array


def _initialize_progress_bar(config: BatchConfig, total: int) -> Any:
    """Initialize progress bar if enabled."""
    if not config.enable_progress:
        return None

    try:
        from tqdm import tqdm

        return tqdm(total=total, desc="Processing materials")
    except ImportError:
        config.enable_progress = False
        warnings.warn("tqdm not available, progress tracking disabled")
        return None


def _process_chunks(
    calculation_data: List[Tuple[str, np.ndarray, float]],
    config: BatchConfig,
    progress_bar: Any,
) -> Dict[str, Optional[XRayResult]]:
    """Process data chunks and collect results."""
    all_results = {}
    memory_monitor = MemoryMonitor(config.memory_limit_gb)

    for chunk in chunk_iterator(calculation_data, config.chunk_size):
        chunk_results = process_batch_chunk(chunk, config)

        for formula, result in chunk_results:
            all_results[formula] = result

        if progress_bar is not None:
            progress_bar.update(len(chunk))

        if not memory_monitor.check_memory():
            memory_monitor.force_gc()

    return all_results


def calculate_batch_properties(
    formulas: List[str],
    energies: Union[float, List[float], np.ndarray],
    densities: List[float],
    config: Optional[BatchConfig] = None,
) -> Dict[str, Optional[XRayResult]]:
    """
    Calculate X-ray properties for multiple materials with optimized batch processing.

    This function processes large batches of calculations efficiently using chunking,
    parallel processing, and memory management.

    Args:
        formulas: List of chemical formulas
        energies: Energy values (shared across all materials)
        densities: List of material densities
        config: Batch processing configuration (optional)

    Returns:
        Dictionary mapping formulas to XRayResult objects

    Raises:
        ValueError: If input validation fails

    Examples:
        >>> formulas = ["SiO2", "Al2O3", "Fe2O3"] * 100  # 300 materials
        >>> energies = np.linspace(5, 15, 101)  # 101 energy points
        >>> densities = [2.2, 3.95, 5.24] * 100
        >>> results = calculate_batch_properties(formulas, energies, densities)
        >>> print(f"Processed {len(results)} materials")
    """
    if config is None:
        config = BatchConfig()

    _validate_batch_inputs(formulas, densities)
    energies_array = _prepare_energies_array(energies)

    calculation_data = [
        (formula, energies_array, density)
        for formula, density in zip(formulas, densities)
    ]

    progress_bar = _initialize_progress_bar(config, len(formulas))

    try:
        return dict(_process_chunks(calculation_data, config, progress_bar))
    finally:
        if progress_bar is not None:
            progress_bar.close()


def _prepare_result_data(valid_results: Dict[str, XRayResult]) -> List[Dict[str, Any]]:
    """Prepare result data for export."""
    data_rows = []

    for formula, result in valid_results.items():
        base_data = {
            "formula": result.formula,
            "molecular_weight_g_mol": result.molecular_weight_g_mol,
            "total_electrons": result.total_electrons,
            "density_g_cm3": result.density_g_cm3,
            "electron_density_per_ang3": result.electron_density_per_ang3,
        }

        for i in range(len(result.energy_kev)):
            row_data = base_data.copy()
            row_data.update(_get_energy_point_data(result, i))
            data_rows.append(row_data)

    return data_rows


def _get_energy_point_data(result: XRayResult, index: int) -> Dict[str, Any]:
    """Get data for a specific energy point."""
    return {
        "energy_kev": result.energy_kev[index],
        "wavelength_angstrom": result.wavelength_angstrom[index],
        "dispersion_delta": result.dispersion_delta[index],
        "absorption_beta": result.absorption_beta[index],
        "scattering_factor_f1": result.scattering_factor_f1[index],
        "scattering_factor_f2": result.scattering_factor_f2[index],
        "critical_angle_degrees": result.critical_angle_degrees[index],
        "attenuation_length_cm": result.attenuation_length_cm[index],
        "real_sld_per_ang2": result.real_sld_per_ang2[index],
        "imaginary_sld_per_ang2": result.imaginary_sld_per_ang2[index],
    }


def _filter_dataframe_fields(
    df: pd.DataFrame, fields: Optional[List[str]]
) -> pd.DataFrame:
    """Filter DataFrame columns based on requested fields."""
    if fields is None:
        return df

    available_fields = set(df.columns)
    requested_fields = set(fields)
    missing_fields = requested_fields - available_fields

    if missing_fields:
        warnings.warn(f"Requested fields not found: {missing_fields}")

    valid_fields = [f for f in fields if f in available_fields]
    return df[valid_fields] if valid_fields else df


def _save_dataframe(df: pd.DataFrame, output_path: Path, format: str) -> None:
    """Save DataFrame to file in specified format."""
    format_lower = format.lower()

    if format_lower == "csv":
        df.to_csv(output_path, index=False)
    elif format_lower == "json":
        df.to_json(output_path, orient="records", indent=2)
    elif format_lower == "parquet":
        try:
            df.to_parquet(output_path, index=False)
        except ImportError:
            raise ValueError("Parquet format requires pyarrow or fastparquet")
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_batch_results(
    results: Dict[str, Optional[XRayResult]],
    output_file: Union[str, Path],
    format: str = "csv",
    fields: Optional[List[str]] = None,
) -> None:
    """
    Save batch calculation results to file.

    Args:
        results: Dictionary of calculation results
        output_file: Output file path
        format: Output format ('csv', 'json', 'parquet')
        fields: List of fields to include (default: all)

    Raises:
        ValueError: If format is not supported
        IOError: If file cannot be written
    """
    output_path = Path(output_file)

    valid_results = {
        formula: result for formula, result in results.items() if result is not None
    }

    if not valid_results:
        raise ValueError("No valid results to save")

    data_rows = _prepare_result_data(valid_results)
    df = pd.DataFrame(data_rows)
    df = _filter_dataframe_fields(df, fields)
    _save_dataframe(df, output_path, format)


def load_batch_input(
    input_file: Union[str, Path],
    formula_column: str = "formula",
    density_column: str = "density",
    energy_column: Optional[str] = None,
) -> Tuple[List[str], List[float], Optional[List[np.ndarray]]]:
    """
    Load batch input data from file.

    Args:
        input_file: Input file path
        formula_column: Name of formula column
        density_column: Name of density column
        energy_column: Name of energy column (optional, for per-material energies)

    Returns:
        Tuple of (formulas, densities, energies)
        where energies is either None or a list of numpy arrays

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If required columns are missing
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Load data based on file extension
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(input_path)
    elif input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        # Try CSV as default
        df = pd.read_csv(input_path)

    # Validate required columns
    required_columns = [formula_column, density_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Extract data
    formulas = df[formula_column].astype(str).tolist()
    densities = df[density_column].astype(float).tolist()

    # Extract energies if specified
    energies = None
    if energy_column and energy_column in df.columns:
        energy_data = df[energy_column].tolist()
        # Handle different energy formats
        if isinstance(energy_data[0], str):
            # Parse comma-separated values
            energies = []
            for energy_str in energy_data:
                energy_list = [float(e.strip()) for e in energy_str.split(",")]
                energies.append(np.array(energy_list))
        else:
            energies = [np.array([float(e)]) for e in energy_data]

    return formulas, densities, energies
