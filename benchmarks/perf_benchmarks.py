"""
Performance benchmarks for xraylabtool, runnable against any release.

The same rows run on the current tree and on a released tag, so the JAX-first
implementation can be compared with the NumPy-era baseline:

    python benchmarks/perf_benchmarks.py                     # current tree
    python benchmarks/perf_benchmarks.py --json out.json     # ... and save rows
    python benchmarks/perf_benchmarks.py --vs v0.4.8         # current vs a tag
    python benchmarks/perf_benchmarks.py --vs v0.4.8 --rounds 3 \
        --markdown benchmarks/RESULTS.md --json benchmarks/results_v0.4.8_vs_current.json

``--baseline-path DIR`` adds DIR to the baseline's ``PYTHONPATH`` only (v0.4.8
declared ``interpax`` and uses it on a CUDA host: ``uv pip install --target DIR
--no-deps interpax equinox jaxtyping lineax wadler_lindig jaxkd``).
``--vs REF`` extracts REF with ``git archive`` into a temporary directory, runs
this script once per side per round (alternating sides, one fresh process each,
``PYTHONPATH`` pointing at the tree) and reports the best median of the rounds
per row. Set ``JAX_PLATFORMS=cpu`` to pin both sides to the CPU. Rows that only
make sense on one side are skipped there. Run on a quiet host: the load average is
recorded in the output.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time

REPO = Path(__file__).resolve().parent.parent

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


_COLD_FIRST_CALC = textwrap.dedent("""\
    import time, sys
    t0 = time.perf_counter_ns()
    from xraylabtool.calculators.core import calculate_single_material_properties
    import numpy as np
    energies = np.linspace(1.0, 30.0, 500)
    _ = calculate_single_material_properties("SiO2", energies, 2.2)
    t1 = time.perf_counter_ns()
    print(t1 - t0)
""")
_COLD_IMPORT = textwrap.dedent("""\
    import time
    t0 = time.perf_counter_ns()
    import xraylabtool
    t1 = time.perf_counter_ns()
    print(t1 - t0)
""")


def bench_cold_start(
    script: str = _COLD_FIRST_CALC,
    label: str = "Cold start (import + first calc, 500 pts)",
) -> dict[str, float]:
    """Measure a script in fresh processes (it prints elapsed nanoseconds)."""
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
            "label": label,
            "iterations": 0,
            "mean_ms": 0.0,
            "std_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "median_ms": 0.0,
        }

    timings_ms = [_ns_to_ms(t) for t in timings_ns]
    return {
        "label": label,
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
        "SiO2",
        "Al2O3",
        "Fe2O3",
        "TiO2",
        "CaCO3",
        "MgO",
        "ZnO",
        "CuO",
        "NiO",
        "CoO",
        "Cr2O3",
        "MnO2",
        "V2O5",
        "ZrO2",
        "HfO2",
        "Nb2O5",
        "Ta2O5",
        "WO3",
        "MoO3",
        "GeO2",
    ]
    densities_20 = [
        2.2,
        3.95,
        5.24,
        4.23,
        2.71,
        3.58,
        5.61,
        6.31,
        6.67,
        6.44,
        5.22,
        5.03,
        3.36,
        5.68,
        9.68,
        4.60,
        8.20,
        7.16,
        4.69,
        4.23,
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


def bench_large_batch() -> dict[str, float]:
    """Benchmark calculate_xray_properties on 200 materials (mixed element counts)."""
    import numpy as np

    from xraylabtool.calculators.core import calculate_xray_properties

    base = ["SiO2", "Al2O3", "Fe2O3", "TiO2", "CaCO3", "MgO", "ZnO", "CuO", "H2O"]
    base += ["Ca5(PO4)3OH", "C6H12O6", "Si", "Au", "Pt", "LiFePO4", "BaTiO3"]
    formulas = [base[i % len(base)] for i in range(200)]
    densities = [2.0 + 0.01 * (i % 50) for i in range(200)]
    energies = np.linspace(1.0, 30.0, 100)
    calculate_xray_properties(formulas, energies, densities)  # compile / fill caches
    return _run_benchmark(
        lambda: calculate_xray_properties(formulas, energies, densities),
        iterations=10,
        warmup=1,
        label="Large batch (200 materials, 100 pts)",
    )


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
    results.append(
        {
            "label": "Element data load - cold (Si)",
            "iterations": n_cold,
            "mean_ms": statistics.mean(timings_ms),
            "std_ms": statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0,
            "min_ms": min(timings_ms),
            "max_ms": max(timings_ms),
            "median_ms": statistics.median(timings_ms),
        }
    )

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
    results.append(
        {
            "label": "Interpolator creation - cold (Si)",
            "iterations": n_cold,
            "mean_ms": statistics.mean(timings_ms),
            "std_ms": statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0,
            "min_ms": min(timings_ms),
            "max_ms": max(timings_ms),
            "median_ms": statistics.median(timings_ms),
        }
    )

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


def _environment(label: str) -> dict[str, object]:
    import importlib.metadata as md

    import xraylabtool

    try:
        import jax

        devices = ", ".join(str(d) for d in jax.devices())
        jax_version = jax.__version__
    except Exception:  # shortcut: baseline trees may not use JAX at all
        devices, jax_version = "n/a", "n/a"
    try:
        load = os.getloadavg()[0]
    except (OSError, AttributeError):
        load = float("nan")
    return {
        "label": label,
        "xraylabtool": getattr(xraylabtool, "__version__", "?"),
        "tree": str(Path(xraylabtool.__file__).resolve().parent.parent),
        "python": sys.version.split()[0],
        "numpy": md.version("numpy"),
        "jax": jax_version,
        "devices": devices,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "load_avg_1m": round(load, 2),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def run_all(label: str = "current") -> dict[str, object]:
    """Run every row in this process (cold-start rows spawn fresh ones)."""
    steps = [
        ("Cold start", lambda: [bench_cold_start()]),
        (
            "Import",
            lambda: [
                bench_cold_start(_COLD_IMPORT, "Import xraylabtool (fresh process)")
            ],
        ),
        ("Single material", bench_single_material),
        ("Multi-material batch", bench_multi_material),
        ("Large batch", lambda: [bench_large_batch()]),
        ("Element data loading", bench_element_loading),
        ("XRayResult construction", lambda: [bench_xray_result_construction()]),
        ("Interpolator creation", bench_interpolator_creation),
        ("Atomic data lookup", lambda: [bench_atomic_data_lookup()]),
    ]
    rows: list[dict[str, float]] = []
    for n, (name, fn) in enumerate(steps, 1):
        print(f"[{n}/{len(steps)}] {name}...", file=sys.stderr, flush=True)
        try:
            rows.extend(fn())
        except (
            Exception
        ) as exc:  # shortcut: a row missing on one side is reported, not fatal
            print(f"  skipped ({type(exc).__name__}: {exc})", file=sys.stderr)
    return {"environment": _environment(label), "rows": rows}


def _extract_ref(ref: str, dest: Path) -> Path:
    """``git archive REF`` into ``dest`` and return the tree root."""
    archive = subprocess.run(
        ["git", "archive", ref], cwd=REPO, capture_output=True, check=True
    )
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)
    return dest


def _run_side(tree: Path, label: str, extra: str = "") -> dict[str, object]:
    """Run this script in a fresh process with ``tree`` first on ``PYTHONPATH``."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        out = Path(handle.name)
    path = os.pathsep.join(p for p in (extra, str(tree)) if p)
    env = {**os.environ, "PYTHONPATH": path, "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run(
        [
            sys.executable,
            str(REPO / "benchmarks" / "perf_benchmarks.py"),
            "--json",
            str(out),
            "--label",
            label,
        ],
        env=env,
        cwd=tempfile.gettempdir(),
        check=True,
        stdout=subprocess.DEVNULL,
    )
    data = json.loads(out.read_text())
    out.unlink()
    return data


def _best(runs: list[dict[str, object]]) -> dict[str, object]:
    """Per row, the round with the lowest median (best-of-N on a noisy host)."""
    best: dict[str, dict[str, float]] = {}
    order: list[str] = []
    for run in runs:
        for row in run["rows"]:  # type: ignore[attr-defined]
            key = row["label"]
            if key not in best:
                order.append(key)
            if key not in best or row["median_ms"] < best[key]["median_ms"]:
                best[key] = row
    return {"environment": runs[0]["environment"], "rows": [best[k] for k in order]}


def compare(base: dict[str, object], new: dict[str, object]) -> list[dict[str, object]]:
    """Join the two sides on the row label; speedup is base median / new median."""
    base_rows = {r["label"]: r for r in base["rows"]}  # type: ignore[attr-defined]
    new_rows = {r["label"]: r for r in new["rows"]}  # type: ignore[attr-defined]
    out = []
    for label in [*base_rows, *[k for k in new_rows if k not in base_rows]]:
        b, n = base_rows.get(label), new_rows.get(label)
        speedup = (
            b["median_ms"] / n["median_ms"]
            if b and n and n["median_ms"] > 0 and b["median_ms"] > 0
            else None
        )
        out.append(
            {
                "label": label,
                "base_median_ms": b["median_ms"] if b else None,
                "new_median_ms": n["median_ms"] if n else None,
                "speedup": speedup,
            }
        )
    return out


def _fmt(value: float | None, spec: str = ".4g") -> str:
    return "n/a" if value is None else format(value, spec)


def comparison_markdown(
    base: dict[str, object],
    new: dict[str, object],
    rows: list[dict[str, object]],
    command: str,
) -> str:
    be, ne = base["environment"], new["environment"]
    lines = [
        f"# Benchmarks: {be['label']} vs {ne['label']}",  # type: ignore[index]
        "",
        "Medians in milliseconds; speedup = baseline / current (>1 means the current "
        f"tree is faster). Best median over alternating rounds, one fresh "
        "process per side per round. Regenerate with "
        f"`{command}`.",
        "",
        "| Benchmark | Baseline ms | Current ms | Speedup |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        sp = r["speedup"]
        lines.append(
            f"| {r['label']} | {_fmt(r['base_median_ms'])} | "  # type: ignore[arg-type]
            f"{_fmt(r['new_median_ms'])} | "  # type: ignore[arg-type]
            f"{'n/a' if sp is None else format(sp, '.2f') + 'x'} |"
        )
    lines += ["", "## Environment", "", "| | Baseline | Current |", "|---|---|---|"]
    for key in (
        "xraylabtool",
        "python",
        "numpy",
        "jax",
        "devices",
        "platform",
        "cpu_count",
        "load_avg_1m",
        "date",
    ):
        lines.append(f"| {key} | {be[key]} | {ne[key]} |")  # type: ignore[index]
    lines += [
        "",
        "Notes: `Element data load`, `Interpolator creation` and `Atomic data lookup` "
        "exercise compatibility APIs that the JAX-first core keeps as thin readers of "
        "the resident tables (ADR-006), so those rows compare an API, not an "
        "algorithm. Cold-start rows include JAX import and the first compile on the "
        "current side. Host-load figures above matter: re-run on a quiet host before "
        "quoting absolute numbers.",
        "",
    ]
    return "\n".join(lines)


def _print_comparison(rows: list[dict[str, object]]) -> None:
    print(f"{'Benchmark':<55} {'Base ms':>10} {'New ms':>10} {'Speedup':>9}")
    print("-" * 87)
    for r in rows:
        sp = r["speedup"]
        print(
            f"{r['label']:<55} {_fmt(r['base_median_ms']):>10} "  # type: ignore[arg-type]
            f"{_fmt(r['new_median_ms']):>10} "  # type: ignore[arg-type]
            f"{'n/a' if sp is None else format(sp, '.2f') + 'x':>9}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--json", type=Path, help="write the rows and environment")
    parser.add_argument("--label", default="current", help="name of this side")
    parser.add_argument(
        "--vs", metavar="REF", help="compare the current tree to a git ref"
    )
    parser.add_argument(
        "--rounds", type=int, default=1, help="rounds per side with --vs"
    )
    parser.add_argument(
        "--baseline-path",
        default="",
        help="extra PYTHONPATH for the baseline side only (v0.4.8 on a CUDA host "
        "imports interpax, which the current tree no longer needs)",
    )
    parser.add_argument("--markdown", type=Path, help="write the --vs comparison here")
    args = parser.parse_args()

    if args.vs:
        with tempfile.TemporaryDirectory() as tmp:
            tree = _extract_ref(args.vs, Path(tmp))
            base_runs, new_runs = [], []
            for n in range(args.rounds):
                print(f"round {n + 1}/{args.rounds}: {args.vs}", file=sys.stderr)
                base_runs.append(_run_side(tree, args.vs, args.baseline_path))
                print(f"round {n + 1}/{args.rounds}: current", file=sys.stderr)
                new_runs.append(_run_side(REPO, "current"))
        base, new = _best(base_runs), _best(new_runs)
        rows = compare(base, new)
        _print_comparison(rows)
        if args.markdown:
            command = " ".join(
                ["python", "benchmarks/perf_benchmarks.py", *sys.argv[1:]]
            )
            if "JAX_PLATFORMS" in os.environ:
                command = f"JAX_PLATFORMS={os.environ['JAX_PLATFORMS']} {command}"
            args.markdown.write_text(comparison_markdown(base, new, rows, command))
        if args.json:
            args.json.write_text(
                json.dumps(
                    {"baseline": base, "current": new, "comparison": rows}, indent=2
                )
            )
        return

    result = run_all(args.label)
    env = result["environment"]
    print("=" * 120)
    print(f"  xraylabtool Performance Benchmarks ({env['label']})")  # type: ignore[index]
    print(
        f"  xraylabtool {env['xraylabtool']} | Python {env['python']} | {env['date']}"
    )  # type: ignore[index]
    print("=" * 120)
    _print_table(result["rows"])  # type: ignore[arg-type]
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
