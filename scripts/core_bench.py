#!/usr/bin/env python
"""Standalone measurements for the JAX-first acceptance gates (P1-P10, P12, P13).

Not collected by pytest and asserts nothing: it prints tables that are copied
into the spec ("Task 0 baselines") and re-run as the gate after each stage.

    uv run python scripts/core_bench.py                 # CPU rows + CLI cold start
    uv run python scripts/core_bench.py --device gpu    # CUDA rows (run outside the sandbox)
    uv run python scripts/core_bench.py --skip-cli

Method: every number is the median of ``--runs`` (default 5) fresh interpreter
processes. "first" is the first call in the process (includes table loading and
compilation); "warm" is the median per-call time after that first call. Run it
on an idle machine; do not run the test suite at the same time. Every worker and
cold CLI run uses an explicitly uncached environment (JAX cache variables removed,
``XRAYLABTOOL_NO_JAX_CACHE=1``, a fresh ``XDG_CACHE_HOME``); the "P9 warm persistent
cache" rows instead populate one dedicated cache directory and then reuse it. A
command that exits nonzero aborts the run instead of being timed.

Backends: ``numpy`` calls ``set_backend("numpy")`` (skipped once the NumPy
backend is removed), ``jax`` calls ``set_backend("jax")``; after the migration
both rows collapse into the single JAX-first path. CPU rows set
``JAX_PLATFORMS=cpu``; GPU rows leave it unset. The CLI rows are measured with
``nvidia-smi`` hidden from ``PATH`` ("cpu-host") and with the inherited
``PATH`` ("gpu-host").
"""

# ruff: noqa: PLC0415

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]

POOL = (  # 2 to 4 elements, mean 2.4; the spec's earlier 17.9 ms NumPy figure came from an ad-hoc scratch pool
    "SiO2",
    "Al2O3",
    "CaCO3",
    "Fe2O3",
    "H2O",
    "NaCl",
    "MgO",
    "TiO2",
    "Ca(OH)2",
    "CuSO4(H2O)5",
)
CASES = {
    "single200": ("SiO2 x 200 E", 200),
    "single20k": ("SiO2 x 20000 E", 40),
    "batch200": ("200 formulas x 200 E", 15),
    "batch2000": ("2000 formulas x 200 E", 4),
}
CLI_P9 = {
    "calc": ["calc", "SiO2", "-e", "10", "-d", "2.2"],
    "convert": ["convert", "energy", "10.0", "--to", "wavelength"],
    "bragg": ["bragg", "-d", "3.14", "-w", "1.54"],
}
CLI_P10 = {
    "--version": ["--version"],
    "--help": ["--help"],
    "formula": ["formula", "SiO2"],
    "atomic": ["atomic", "Si"],
    "list": ["list", "constants"],
    "completion": ["completion", "list"],
}


def worker(case: str, backend: str) -> None:
    """Run one case in this fresh process and print ``RESULT={json}``."""
    import numpy as np

    if backend != "default":
        from xraylabtool.backend import set_backend

        try:
            set_backend(backend)
        except ValueError as exc:
            print("RESULT=" + json.dumps({"skipped": str(exc)}))
            return

    from xraylabtool import (
        calculate_single_material_properties,
        calculate_xray_properties,
    )

    compile_events: list[float] = []
    # Registered unconditionally: the package no longer imports jax before the first call.
    import jax.monitoring

    def _on_duration(event: str, duration: float, **_: object) -> None:
        if "backend_compile" in event:
            compile_events.append(duration)

    jax.monitoring.register_event_duration_secs_listener(_on_duration)

    if case == "compiles":
        print("RESULT=" + json.dumps(_compile_counts(compile_events)))
        return

    energies = np.linspace(1.0, 29.0, 200)
    if case == "single200":

        def call() -> object:
            return calculate_single_material_properties("SiO2", energies, 2.2)

    elif case == "single20k":
        energies = np.linspace(0.05, 29.95, 20000)

        def call() -> object:
            return calculate_single_material_properties("SiO2", energies, 2.2)

    else:
        n = {"batch200": 200, "batch2000": 2000}[case]
        formulas = [POOL[i % len(POOL)] for i in range(n)]
        densities = [2.0 + 0.001 * i for i in range(n)]

        def call() -> object:
            return calculate_xray_properties(formulas, energies, densities)

    reps = CASES[case][1]
    t0 = time.perf_counter()
    call()
    first = time.perf_counter() - t0
    first_compiles = len(compile_events)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        call()
        times.append(time.perf_counter() - t0)
    out = {
        "first_ms": first * 1e3,
        "warm_ms": statistics.median(times) * 1e3,
        "first_compiles": first_compiles,
        "warm_compiles": len(compile_events) - first_compiles,
    }
    print("RESULT=" + json.dumps(out))


def _compile_counts(events: list[float]) -> dict[str, int]:
    """P12 / P13: compiles for a repeated call and for a new energy count."""
    import numpy as np

    from xraylabtool import calculate_single_material_properties as calc

    calc("SiO2", np.linspace(1.0, 29.0, 200), 2.2)
    base = len(events)
    calc("SiO2", np.linspace(1.0, 29.0, 200), 2.3)
    same = len(events) - base
    base = len(events)
    calc("SiO2", np.linspace(1.0, 29.0, 201), 2.2)
    new_count = len(events) - base
    base = len(events)
    # 216 is the top of the 200/201 bucket (bucket(m, 64)); 250 would cross into the next one.
    calc("SiO2", np.linspace(1.0, 29.0, 216), 2.2)
    new_count_2 = len(events) - base
    base = len(events)
    calc("Si", np.linspace(1.0, 29.0, 200), 2.33)
    new_material = len(events) - base
    return {
        "p12_same_shape": same,
        "p13_energy_200_to_201": new_count,
        "p13_energy_201_to_216": new_count_2,
        "new_material_same_k": new_material,
    }


_CACHE_VARS = (
    "JAX_COMPILATION_CACHE_DIR",
    "JAX_ENABLE_COMPILATION_CACHE",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES",
)


def cold_env(base: dict[str, str]) -> tuple[dict[str, str], str]:
    """Return (env, cache_dir): no persistent compile cache, fresh user-cache directory."""
    env = {k: v for k, v in base.items() if k not in _CACHE_VARS}
    cache_dir = tempfile.mkdtemp(prefix="core_bench_cache_")
    env["XRAYLABTOOL_NO_JAX_CACHE"] = "1"
    env["XDG_CACHE_HOME"] = cache_dir
    return env, cache_dir


def timed_run(argv: list[str], env: dict[str, str]) -> float:
    """Run ``argv`` and return the wall time in seconds; a nonzero exit is an error."""
    t0 = time.perf_counter()
    done = subprocess.run(
        argv, capture_output=True, text=True, env=env, cwd=REPO, check=False
    )
    elapsed = time.perf_counter() - t0
    if done.returncode != 0:
        raise RuntimeError(
            f"{argv} exited with {done.returncode}:\n{done.stderr[-1500:]}"
        )
    return elapsed


def spawn_case(
    case: str, backend: str, device: str, runs: int
) -> dict[str, float] | None:
    """Median over fresh processes; ``None`` if the backend is not available."""
    base = dict(os.environ)
    if device == "cpu":
        base["JAX_PLATFORMS"] = "cpu"
    else:
        base.pop("JAX_PLATFORMS", None)
    results = []
    for _ in range(runs):
        env, cache_dir = cold_env(base)  # P8: first call compiles, no persistent cache
        done = subprocess.run(
            [sys.executable, __file__, "--worker", case, "--backend", backend],
            capture_output=True,
            text=True,
            env=env,
            cwd=REPO,
            check=False,
        )
        shutil.rmtree(cache_dir, ignore_errors=True)
        lines = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT=")]
        if done.returncode != 0 or not lines:
            raise RuntimeError(
                f"{case}/{backend}/{device} failed:\n{done.stderr[-1500:]}"
            )
        result = json.loads(lines[-1].removeprefix("RESULT="))
        if "skipped" in result:
            return None
        results.append(result)
    return {k: statistics.median(r[k] for r in results) for k in results[0]}


def _fmt(row: dict[str, float] | None, key: str) -> str:
    return "n/a" if row is None else f"{row[key]:.3f}"


def run_core_rows(device: str, runs: int) -> None:
    """Print the first-call / warm table for the backends relevant to ``device``."""
    backends = ["numpy", "jax"] if device == "cpu" else ["jax"]
    print(
        f"\n== core calculators, device={device}, median of {runs} fresh processes (ms) =="
    )
    header = "backend".ljust(8) + "".join(f"{CASES[c][0]:>26}" for c in CASES)
    print(header)
    rows: dict[str, dict[str, dict[str, float] | None]] = {}
    for backend in backends:
        rows[backend] = {c: spawn_case(c, backend, device, runs) for c in CASES}
        print(
            "first".ljust(8)
            + "".join(f"{_fmt(rows[backend][c], 'first_ms'):>26}" for c in CASES)
            + f"   [{backend}]"
        )
        print(
            "warm".ljust(8)
            + "".join(f"{_fmt(rows[backend][c], 'warm_ms'):>26}" for c in CASES)
            + f"   [{backend}]"
        )
        if rows[backend]["single200"] is not None:
            fc = {
                c: rows[backend][c]["first_compiles"] for c in CASES if rows[backend][c]
            }  # type: ignore[index]
            print(f"         compiles in first call: {fc}")
    if "jax" in backends:
        counts = spawn_compile_counts(device, runs)
        print(f"P12/P13 compile counts (jax, {device}): {counts}")


def spawn_compile_counts(device: str, runs: int) -> dict[str, float] | None:
    return spawn_case("compiles", "jax", device, runs)


def _importtime(env: dict[str, str], runs: int) -> tuple[float, bool]:
    """Median cumulative ``import xraylabtool`` time (ms) over ``runs`` fresh processes."""
    totals = []
    saw_jax = False
    for _ in range(runs):
        done = subprocess.run(
            [sys.executable, "-X", "importtime", "-c", "import xraylabtool"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if done.returncode != 0:
            raise RuntimeError(
                f"import xraylabtool failed ({done.returncode}):\n{done.stderr[-1500:]}"
            )
        total = None
        for line in done.stderr.splitlines():
            m = re.match(r"import time:\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\S.*)$", line)
            if not m:
                continue
            if m.group(3).strip() == "xraylabtool":
                total = float(m.group(2)) / 1e3
            saw_jax = saw_jax or m.group(3).strip() == "jax"
        if total is None:
            raise RuntimeError("no `xraylabtool` line in the -X importtime output")
        totals.append(total)
    return statistics.median(totals), saw_jax


def _median_cli(exe: Path, argv: list[str], base: dict[str, str], runs: int) -> float:
    """Cold CLI time: no persistent compile cache, a fresh user-cache directory per run."""
    times = []
    for _ in range(runs):
        env, cache_dir = cold_env(base)
        try:
            times.append(timed_run([str(exe), *argv], env))
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)
    return statistics.median(times)


def _median_cli_warm_cache(
    exe: Path, argv: list[str], base: dict[str, str], runs: int
) -> float:
    """P9 warm mode: populate a dedicated cache directory once, then time reuse of it."""
    cache_dir = tempfile.mkdtemp(prefix="core_bench_warm_")
    env = {k: v for k, v in base.items() if k not in _CACHE_VARS}
    env.pop("XRAYLABTOOL_NO_JAX_CACHE", None)
    env["XDG_CACHE_HOME"] = cache_dir
    try:
        timed_run(
            [str(exe), *argv], env
        )  # populates the cache (if the command uses it)
        return statistics.median(timed_run([str(exe), *argv], env) for _ in range(runs))
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def run_cli_rows(runs: int) -> None:
    """P9 / P10: CLI cold start on a cpu-host (no nvidia-smi on PATH) and a gpu-host."""
    exe = Path(sys.executable).parent / "xraylabtool"
    hosts: dict[str, dict[str, str]] = {}
    empty = tempfile.mkdtemp(prefix="core_bench_path_")
    hosts["cpu-host"] = {**os.environ, "PATH": empty}
    if shutil.which("nvidia-smi"):
        hosts["gpu-host"] = dict(os.environ)
    print(f"\n== CLI cold start, median of {runs} runs (s); exe={exe} ==")
    try:
        for label, base in hosts.items():
            imp, jax_seen = _importtime(base, runs)
            print(
                f"[{label}] `import xraylabtool` (-X importtime cumulative, median of "
                f"{runs}): {imp:.0f} ms, jax imported: {jax_seen}"
            )
            for group, commands in (("P9", CLI_P9), ("P10", CLI_P10)):
                cells = [
                    f"{name}={_median_cli(exe, argv, base, runs):.3f}"
                    for name, argv in commands.items()
                ]
                print(f"[{label}] {group} cold: " + "  ".join(cells))
            cells = [
                f"{name}={_median_cli_warm_cache(exe, argv, base, runs):.3f}"
                for name, argv in CLI_P9.items()
            ]
            print(f"[{label}] P9 warm persistent cache: " + "  ".join(cells))
    finally:
        shutil.rmtree(empty, ignore_errors=True)


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--skip-cli", action="store_true")
    parser.add_argument("--worker")
    parser.add_argument("--backend", default="default")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.backend)
        return 0
    run_core_rows(args.device, args.runs)
    if not args.skip_cli and args.device == "cpu":
        run_cli_rows(args.runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
