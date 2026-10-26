#!/usr/bin/env python
"""Standalone benchmark of the ``xraylabtool.kk`` JAX pipeline (not collected by pytest).

Every cold measurement runs in a fresh subprocess with ``JAX_PLATFORMS`` set
before ``jax`` is imported::

    uv run python scripts/kk_bench.py [--platform cpu] [--repeats 15]

Measures, for SiO2 (1201 segments):

* the kernel at m = 100 / 1 000 / 10 000 / 100 000 log-uniform energies: the
  host wrapper ``kk_real_spectrum`` (upload, kernel, download; the figure the
  acceptance table compares with the removed NumPy path) and the
  device-resident ``kk_f1`` waited on with ``block_until_ready``; first call
  including compilation, then the min and median of ``--repeats`` warm calls;
* the first ``kk_calculate_real("SiO2", energies_kev=<1 000 energies>)`` after
  import (cold, compile included);
* ``kk_calculate_real("SiO2", refine_tolerance=0.01)`` cold and warm (min and
  median of ``--repeats``).

Compilation is recorded for every live run with the version-tested recorder of
``tests/fixtures/kk_compiles.py``: the number of top-level numerical compile
EVENTS (a function compiled at two shapes counts twice; JAX's trivial
primitive compiles are excluded). The JAX pipeline is the only
implementation, so there is a single backend. The gates of the design spec's
acceptance table (CPU rows) are printed with a pass/fail mark; a missed gate
is reported, never adjusted.

Historical table of the removed NumPy implementation (commit 1bf56d9, jax
0.12.0.1, numpy 2.5.4, CPU only: 20 cores, no GPU in this sandbox; ms unless
noted), kept as the baseline the new numbers are compared with::

    kernel (first call / warm median, ms)
      m = 100       18.7 / 17.7
      m = 1 000     139.9 / 157.3
      m = 10 000    1910 / 1892
      m = 100 000   21994 / 14746
    kk_calculate_real("SiO2", refine_tolerance=0.01):
      cold 0.29 s, warm 0.273 s, 1383 points (267 refined)

and of the first JAX pipeline (one jit per chunk shape; kernel host wrapper):

      m = 100       128.8 / 2.8
      m = 1 000     142.2 / 20.7
      m = 10 000    259.1 / 150.9
      m = 100 000   1215 / 1309
      refined run: cold 2.29 s, warm 0.067 s, 14 compile events

Run-to-run spread on a shared machine is up to 30 %; treat the numbers as
indicative and read the min next to the median.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SIZES = (100, 1_000, 10_000, 100_000)
REFINE_TOLERANCE = 0.01
FIRST_CALL_ENERGIES = 1_000
SEED = 2026
MARKER = "KKBENCH_JSON:"

# CPU rows of the acceptance table in the design spec.
GATES_MS = {
    "kernel m=1000 (ms)": 35.0,
    "kernel m=10000 (ms)": 250.0,
    "first call, 1000 energies (ms)": 1000.0,
    "refined cold (ms)": 1700.0,
    "refined warm (ms)": 100.0,
}
PROGRAM_GATE = 6


def _wait(value: object) -> None:
    block = getattr(value, "block_until_ready", None)
    if block is not None:
        block()


def _timed_ms(fn: object, repeats: int) -> tuple[float, float]:
    """``(min, median)`` of ``repeats`` warm calls of ``fn`` in ms."""
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        _wait(fn())  # type: ignore[operator]
        times.append(1e3 * (time.perf_counter() - t))
    return min(times), statistics.median(times)


def _recorder():  # type: ignore[no-untyped-def]
    sys.path.insert(0, str(ROOT))
    from tests.fixtures import kk_compiles  # noqa: PLC0415

    return kk_compiles


def _kernel_worker(repeats: int) -> dict[str, object]:
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    t0 = time.perf_counter()
    jax.config.update("jax_enable_x64", True)
    jax.numpy.zeros(1).block_until_ready()
    jax_init_ms = 1e3 * (time.perf_counter() - t0)
    from xraylabtool.kk import (  # noqa: PLC0415
        Spectrum,
        compound_spectrum,
        kk_f1,
        kk_real_spectrum,
        load_asf,
        relativistic_correction,
    )
    from xraylabtool.utils import parse_formula  # noqa: PLC0415

    symbols, counts = parse_formula("SiO2")
    seg, coeffs = compound_spectrum(symbols, counts)
    db = load_asf()
    rel = relativistic_correction([db.symbols.index(s) + 1 for s in symbols], counts)
    spectrum = Spectrum.from_numpy(seg, coeffs)
    rng = np.random.default_rng(SEED)
    rows = []
    for m in SIZES:
        energy = np.exp(rng.uniform(np.log(10.0), np.log(5e5), m))
        device_energy = jax.device_put(energy)
        t = time.perf_counter()
        kk_real_spectrum(energy, seg, coeffs, rel)
        first = 1e3 * (time.perf_counter() - t)
        host_min, host_med = _timed_ms(
            lambda e=energy: kk_real_spectrum(e, seg, coeffs, rel), repeats
        )
        _wait(kk_f1(device_energy, spectrum, rel))
        dev_min, dev_med = _timed_ms(
            lambda e=device_energy: kk_f1(e, spectrum, rel), repeats
        )
        rows.append(
            {
                "m": m,
                "first_ms": first,
                "host_min_ms": host_min,
                "host_median_ms": host_med,
                "device_min_ms": dev_min,
                "device_median_ms": dev_med,
            }
        )
    return {"n_segments": int(seg.size - 1), "jax_init_ms": jax_init_ms, "rows": rows}


def _call_worker(kind: str, repeats: int) -> dict[str, object]:
    """Cold (recorded) and warm ``kk_calculate_real`` runs in a fresh interpreter."""
    import numpy as np  # noqa: PLC0415

    kk_compiles = _recorder()
    from xraylabtool.kk import kk_calculate_real  # noqa: PLC0415

    if kind == "first":
        energies = np.geomspace(0.05, 20.0, FIRST_CALL_ENERGIES)

        def call() -> object:
            return kk_calculate_real("SiO2", energies_kev=energies)

    else:

        def call() -> object:
            return kk_calculate_real("SiO2", refine_tolerance=REFINE_TOLERANCE)

    with kk_compiles.record_compiles() as events:
        t = time.perf_counter()
        result = call()
        cold = 1e3 * (time.perf_counter() - t)
    warm_min, warm_med = _timed_ms(call, repeats)
    return {
        "cold_ms": cold,
        "warm_min_ms": warm_min,
        "warm_median_ms": warm_med,
        "n_points": int(result.energy_kev.size),  # type: ignore[attr-defined]
        "n_refined": int(result.n_refined_points),  # type: ignore[attr-defined]
        "cold_compile_events": len(kk_compiles.numerical_programs(events)),
        "cold_kk_programs": sorted(kk_compiles.kk_programs(events)),
        "cold_other_events": sorted(
            set(kk_compiles.numerical_programs(events))
            - set(kk_compiles.kk_programs(events))
        ),
    }


def _run_worker(kind: str, platform: str, repeats: int) -> dict[str, object]:
    env = {
        **os.environ,
        "JAX_PLATFORMS": platform,
        "PYTHONWARNINGS": "ignore",
        "PYTHONPATH": str(ROOT),
    }
    proc = subprocess.run(
        [sys.executable, __file__, "--worker", kind, "--repeats", str(repeats)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(MARKER):
            data: dict[str, object] = json.loads(line[len(MARKER) :])
            return data
    raise SystemExit(f"{kind} worker failed:\n{proc.stderr[-2000:]}")


def _gate(name: str, value: float, limit: float) -> str:
    return f"  GATE {'PASS' if value <= limit else 'MISSED'}  {name}: {value:.1f} <= {limit:g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--platform", default="cpu", help="JAX_PLATFORMS value")
    parser.add_argument("--repeats", type=int, default=15, help="warm calls per row")
    parser.add_argument("--worker", choices=["kernel", "first", "refine"])
    args = parser.parse_args()

    if args.worker:
        if args.worker == "kernel":
            out = _kernel_worker(args.repeats)
        else:
            out = _call_worker(args.worker, args.repeats)
        print(MARKER + json.dumps(out))
        return

    print(
        f"platform={args.platform}  repo={ROOT}  warm repeats={args.repeats}  "
        f"load average={os.getloadavg()[0]:.1f}"
    )
    kern = _run_worker("kernel", args.platform, args.repeats)
    print(f"\nkernel, SiO2 ({kern['n_segments']} segments)")
    print(f"  import jax + x64 + device init: {kern['jax_init_ms']:.0f} ms")
    print(
        "  m        first call   host wrapper min / median   "
        "device kk_f1 min / median   (ms)"
    )
    gates = []
    for row in kern["rows"]:  # type: ignore[attr-defined]
        print(
            f"  {row['m']:<8d} {row['first_ms']:>10.1f}   "
            f"{row['host_min_ms']:>10.1f} / {row['host_median_ms']:<10.1f}   "
            f"{row['device_min_ms']:>10.1f} / {row['device_median_ms']:<10.1f}"
        )
        for m in (1_000, 10_000):
            if row["m"] == m:
                gates.append(
                    _gate(
                        f"kernel m={m} (ms)",
                        row["host_median_ms"],
                        GATES_MS[f"kernel m={m} (ms)"],
                    )
                )

    first = _run_worker("first", args.platform, args.repeats)
    print(
        f"\nfirst kk_calculate_real('SiO2', {FIRST_CALL_ENERGIES} energies) after import: "
        f"cold {first['cold_ms']:.0f} ms, warm min {first['warm_min_ms']:.1f} / "
        f"median {first['warm_median_ms']:.1f} ms; "
        f"compile events {first['cold_compile_events']} {first['cold_kk_programs']}"
    )
    gates.append(
        _gate(
            "first call, 1000 energies (ms)",
            first["cold_ms"],  # type: ignore[arg-type]
            GATES_MS["first call, 1000 energies (ms)"],
        )
    )

    ref = _run_worker("refine", args.platform, args.repeats)
    print(
        f"\nkk_calculate_real('SiO2', refine_tolerance={REFINE_TOLERANCE}): "
        f"cold {ref['cold_ms']:.0f} ms, warm min {ref['warm_min_ms']:.1f} / "
        f"median {ref['warm_median_ms']:.1f} ms, "
        f"{ref['n_points']} points ({ref['n_refined']} refined)"
    )
    print(
        f"  cold top-level numerical compile events {ref['cold_compile_events']}: "
        f"{ref['cold_kk_programs']}; other events {ref['cold_other_events']}"
    )
    gates.append(
        _gate("refined cold (ms)", ref["cold_ms"], GATES_MS["refined cold (ms)"])
    )  # type: ignore[arg-type]
    gates.append(
        _gate("refined warm (ms)", ref["warm_median_ms"], GATES_MS["refined warm (ms)"])  # type: ignore[arg-type]
    )
    events = int(ref["cold_compile_events"])  # type: ignore[call-overload]
    gates.append(
        f"  GATE {'PASS' if events <= PROGRAM_GATE else 'MISSED'}  "
        f"top-level programs in the refined run: {events} <= {PROGRAM_GATE}"
    )
    print("\nCPU acceptance gates (warm rows use the median):")
    print("\n".join(gates))


if __name__ == "__main__":
    main()
