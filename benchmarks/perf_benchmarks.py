"""
Performance benchmarks for xraylabtool.

Measures before/after impact of optimizations on the main calculation pipeline.
Run with: uv run python benchmarks/perf_benchmarks.py
"""

from __future__ import annotations

import gc
import statistics
import subprocess
import sys
import textwrap
import time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ns_to_ms(ns: float) -> float:
    return ns / 1_000_000


def _run_benchmark(
    fn,
    iterations: int = 100,
    warmup: int = 3,
    label: str = "",
) -> dict[str, float]:
    """Run *fn* repeatedly and return timing statistics in milliseconds."""
    # Warmup
    for _ in range(warmup):
        fn()

    gc.collect()
    gc.disable()
    try:
        timings_ns: list[int] = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            fn()
            t1 = time.perf_counter_ns()
            timings_ns.append(t1 - t0)
    finally:
        gc.enable()

    timings_ms = [_ns_to_ms(t) for t in timings_ns]
    return {
        "label": label,
        "iterations": iterations,
        "mean_ms": statistics.mean(timings_ms),
        "std_ms": statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
        "median_ms": statistics.median(timings_ms),
    }


def _print_table(rows: list[dict[str, float]]) -> None:
    """Print results as a formatted table."""
    header = f"{'Benchmark':<50} {'Iters':>6} {'Mean ms':>10} {'Std ms':>10} {'Min ms':>10} {'Max ms':>10} {'Median ms':>10}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['label']:<50} {int(r['iterations']):>6} "
            f"{r['mean_ms']:>10.3f} {r['std_ms']:>10.3f} "
            f"{r['min_ms']:>10.3f} {r['max_ms']:>10.3f} "
            f"{r['median_ms']:>10.3f}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# 1. Cold-start benchmark (runs in a subprocess)
# ---------------------------------------------------------------------------

def bench_cold_start() -> dict[str, float]:
    """Measure import + first calculation in a fresh process."""
    script = textwrap.dedent("""\
        import time, sys
        t0 = time.perf_counter_ns()
        from xraylabtool.calculators.core import calculate_single_material_properties
        import numpy as np
        energies = np.linspace(1.0, 30.0, 500)
        _ = calculate_single_material_properties("SiO2", energies, 2.2)
        t1 = time.perf_counter_ns()
        print(t1 - t0)
    """)
    timings_ns: list[int] = []
    n_runs = 5  # Fewer runs because subprocess is heavy
    for _ in range(n_runs):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  Cold start subprocess error: {result.stderr.strip()}")
            continue
        timings_ns.append(int(result.stdout.strip()))

    if not timings_ns:
        return {
            "label": "Cold start (import + first calc, 500 pts)",
            "iterations": 0,
            "mean_ms": 0.0,
            "std_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "median_ms": 0.0,
        }

    timings_ms = [_ns_to_ms(t) for t in timings_ns]
    return {
        "label": "Cold start (import + first calc, 500 pts)",
        "iterations": len(timings_ms),
        "mean_ms": statistics.mean(timings_ms),
        "std_ms": statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
        "median_ms": statistics.median(timings_ms),
    }


# ---------------------------------------------------------------------------
# 2. Warm single-material benchmarks
# ---------------------------------------------------------------------------

def bench_single_material() -> list[dict[str, float]]:
    """Benchmark calculate_single_material_properties with warm cache."""
    import numpy as np

    from xraylabtool.calculators.core import calculate_single_material_properties

    results = []

    for n_pts in (500, 5000):
        energies = np.linspace(1.0, 30.0, n_pts)

        # Warm the cache
        calculate_single_material_properties("SiO2", energies, 2.2)

        r = _run_benchmark(
            lambda e=energies: calculate_single_material_properties("SiO2", e, 2.2),
            iterations=100,
            warmup=5,
            label=f"Single material SiO2 ({n_pts} pts)",
        )
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# 3. Multi-material batch benchmarks
# ---------------------------------------------------------------------------

def bench_multi_material() -> list[dict[str, float]]:
    """Benchmark calculate_xray_properties for multiple materials."""
    import numpy as np

    from xraylabtool.calculators.core import calculate_xray_properties

    energies = np.linspace(1.0, 30.0, 500)

    materials_5 = ["SiO2", "Al2O3", "Fe2O3", "TiO2", "CaCO3"]
    densities_5 = [2.2, 3.95, 5.24, 4.23, 2.71]

    materials_20 = [
        "SiO2", "Al2O3", "Fe2O3", "TiO2", "CaCO3",
        "MgO", "ZnO", "CuO", "NiO", "CoO",
        "Cr2O3", "MnO2", "V2O5", "ZrO2", "HfO2",
        "Nb2O5", "Ta2O5", "WO3", "MoO3", "GeO2",
    ]
    densities_20 = [
        2.2, 3.95, 5.24, 4.23, 2.71,
        3.58, 5.61, 6.31, 6.67, 6.44,
        5.22, 5.03, 3.36, 5.68, 9.68,
        4.60, 8.20, 7.16, 4.69, 4.23,
    ]

    results = []

    # Warm cache for all materials
    calculate_xray_properties(materials_20, energies, densities_20)

    r5 = _run_benchmark(
        lambda: calculate_xray_properties(materials_5, energies, densities_5),
        iterations=50,
        warmup=3,
        label="Multi-material batch (5 materials, 500 pts)",
    )
    results.append(r5)

    r20 = _run_benchmark(
        lambda: calculate_xray_properties(materials_20, energies, densities_20),
        iterations=20,
        warmup=2,
        label="Multi-material batch (20 materials, 500 pts)",
    )
    results.append(r20)

    return results


# ---------------------------------------------------------------------------
# 4. Element data loading benchmarks
# ---------------------------------------------------------------------------

def bench_element_loading() -> list[dict[str, float]]:
    """Benchmark loading a single element's .nff data (cold and warm)."""
    from xraylabtool.calculators.core import (
        _scattering_factor_cache,
        load_scattering_factor_data,
    )

    results = []

    # --- Warm load (data already in cache) ---
    load_scattering_factor_data("Si")  # Ensure cached
    r_warm = _run_benchmark(
        lambda: load_scattering_factor_data("Si"),
        iterations=1000,
        warmup=10,
        label="Element data load - warm (Si)",
    )
    results.append(r_warm)

    # --- Cold load (clear cache first each iteration) ---
    timings_ns: list[int] = []
    n_cold = 50
    for _ in range(n_cold):
        _scattering_factor_cache.pop("Si", None)
        gc.collect()
        t0 = time.perf_counter_ns()
        load_scattering_factor_data("Si")
        t1 = time.perf_counter_ns()
        timings_ns.append(t1 - t0)

    timings_ms = [_ns_to_ms(t) for t in timings_ns]
    results.append({
        "label": "Element data load - cold (Si)",
        "iterations": n_cold,
        "mean_ms": statistics.mean(timings_ms),
        "std_ms": statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
        "median_ms": statistics.median(timings_ms),
    })

    return results


# ---------------------------------------------------------------------------
# 5. XRayResult construction benchmark
# ---------------------------------------------------------------------------

def bench_xray_result_construction() -> dict[str, float]:
    """Benchmark constructing XRayResult objects."""
    import numpy as np

    from xraylabtool.calculators.core import XRayResult

    n_pts = 500
    dummy_arrays = {
        "energy_kev": np.linspace(1.0, 30.0, n_pts),
        "wavelength_angstrom": np.linspace(0.4, 12.4, n_pts),
        "dispersion_delta": np.random.rand(n_pts) * 1e-5,
        "absorption_beta": np.random.rand(n_pts) * 1e-7,
        "scattering_factor_f1": np.random.rand(n_pts) * 14.0,
        "scattering_factor_f2": np.random.rand(n_pts) * 0.5,
        "critical_angle_degrees": np.random.rand(n_pts) * 0.3,
        "attenuation_length_cm": np.random.rand(n_pts) * 0.01,
        "real_sld_per_ang2": np.random.rand(n_pts) * 1e-5,
        "imaginary_sld_per_ang2": np.random.rand(n_pts) * 1e-7,
    }

    def create_1000():
        for _ in range(1000):
            XRayResult(
                formula="SiO2",
                molecular_weight_g_mol=60.08,
                total_electrons=30.0,
                density_g_cm3=2.2,
                electron_density_per_ang3=0.66,
                **dummy_arrays,
            )

    return _run_benchmark(
        create_1000,
        iterations=20,
        warmup=2,
        label="XRayResult construction (1000 objects, 500 pts each)",
    )


# ---------------------------------------------------------------------------
# 6. Interpolator creation benchmark
# ---------------------------------------------------------------------------

def bench_interpolator_creation() -> list[dict[str, float]]:
    """Benchmark creating scattering factor interpolators."""
    from xraylabtool.calculators.core import (
        _interpolator_cache,
        create_scattering_factor_interpolators,
    )

    results = []

    # Warm
    create_scattering_factor_interpolators("Si")
    r_warm = _run_benchmark(
        lambda: create_scattering_factor_interpolators("Si"),
        iterations=1000,
        warmup=10,
        label="Interpolator creation - warm (Si)",
    )
    results.append(r_warm)

    # Cold (clear LRU + dict cache each time)
    timings_ns: list[int] = []
    n_cold = 50
    for _ in range(n_cold):
        create_scattering_factor_interpolators.cache_clear()
        _interpolator_cache.pop("Si", None)
        t0 = time.perf_counter_ns()
        create_scattering_factor_interpolators("Si")
        t1 = time.perf_counter_ns()
        timings_ns.append(t1 - t0)

    timings_ms = [_ns_to_ms(t) for t in timings_ns]
    results.append({
        "label": "Interpolator creation - cold (Si)",
        "iterations": n_cold,
        "mean_ms": statistics.mean(timings_ms),
        "std_ms": statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
        "median_ms": statistics.median(timings_ms),
    })

    return results


# ---------------------------------------------------------------------------
# 7. Atomic data lookup benchmark
# ---------------------------------------------------------------------------

def bench_atomic_data_lookup() -> dict[str, float]:
    """Benchmark get_atomic_data_fast for preloaded elements."""
    from xraylabtool.data_handling.atomic_cache import get_atomic_data_fast

    # Warm
    get_atomic_data_fast("Si")

    def lookup_100():
        for _ in range(100):
            get_atomic_data_fast("Si")
            get_atomic_data_fast("O")
            get_atomic_data_fast("Al")

    return _run_benchmark(
        lookup_100,
        iterations=500,
        warmup=10,
        label="Atomic data lookup (300 lookups: Si, O, Al)",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 120)
    print("  xraylabtool Performance Benchmarks")
    print(f"  Python {sys.version.split()[0]} | {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 120)
    print()

    all_results: list[dict[str, float]] = []

    # 1. Cold start
    print("[1/7] Cold start benchmark (subprocess, may take a moment)...")
    all_results.append(bench_cold_start())

    # 2. Single material
    print("[2/7] Single material benchmarks...")
    all_results.extend(bench_single_material())

    # 3. Multi-material batch
    print("[3/7] Multi-material batch benchmarks...")
    all_results.extend(bench_multi_material())

    # 4. Element data loading
    print("[4/7] Element data loading benchmarks...")
    all_results.extend(bench_element_loading())

    # 5. XRayResult construction
    print("[5/7] XRayResult construction benchmark...")
    all_results.append(bench_xray_result_construction())

    # 6. Interpolator creation
    print("[6/7] Interpolator creation benchmarks...")
    all_results.extend(bench_interpolator_creation())

    # 7. Atomic data lookup
    print("[7/7] Atomic data lookup benchmark...")
    all_results.append(bench_atomic_data_lookup())

    print()
    _print_table(all_results)
    print()


if __name__ == "__main__":
    main()
