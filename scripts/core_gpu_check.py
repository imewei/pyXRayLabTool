#!/usr/bin/env python
"""Manual GPU gate for the JAX-first core (standalone, not collected by pytest).

Run on the CUDA machine (outside the sandbox, on a quiet host)::

    uv run python scripts/core_gpu_check.py [--repeats 5] [--skip-bench]

It repeats on the default JAX device the checks the CPU suite makes and the GPU
rows of ``scripts/core_bench.py``, with the thresholds of the design spec's
acceptance gates (``docs/superpowers/specs/2026-10-26-package-jax-first-design.md``),
prints one table with a verdict line per gate and exits 1 if any row misses:

* goldens (``tests/characterization/core``, rtol 1e-12, never loosened);
* P11: transfer budget with ``jax.transfer_guard("disallow_explicit")`` live,
  plus the ``runtime`` counters (<= 1 upload and <= 1 fetch per device program
  call: per public call for a single material, per block for a batch);
* padding invariance (bit-exact, ``tests/unit/core/test_optics.py``);
* P5 to P7: warm GPU timings, median of ``--repeats`` fresh processes
  (``core_bench`` workers, no persistent compile cache);
* P12 and P13: compiles after warm-up and for a new energy count inside a
  bucket (``core_bench`` ``compiles`` worker, the JAX compile-event listener).

The timings run first, in fresh processes, before this process touches the device:
a resident parent CUDA context (it preallocates most of the GPU memory and keeps
the driver and clocks up) makes the child timings about 1.3 ms faster for P6 and
shortens the first call by about 200 ms, so a parent that holds the GPU is not a
faithful model of a fresh user process.

P8 to P10 are CPU-host or CLI gates measured by ``scripts/core_bench.py``. The
script refuses to run when JAX has no CUDA device. Record the load average
printed at the start and the end with the numbers.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Same environment as tests/conftest.py: no persistent compile cache, bounded memory.
os.environ.setdefault("XRAYLABTOOL_NO_JAX_CACHE", "1")

import jax  # noqa: E402
import numpy as np  # noqa: E402

# Gate limits (ms) of the spec's acceptance table: (core_bench case, limit).
GPU_GATES_MS = {
    "P5": ("single200", 1.5, "SiO2 x 200 E"),
    "P6": ("batch200", 5.0, "200 formulas x 200 E"),
    "P7": ("single20k", 2.0, "SiO2 x 20000 E"),
}

# Program calls (F-blocks) the batch cases must take: the 200 formulas share one K
# bucket; the mixed-K list spans three (1-2, 3-4 and 5-8 elements).
EXPECTED_BLOCKS = {"batch 200 formulas": 1, "batch mixed K": 3}

Row = tuple[str, bool, str]
rows: list[Row] = []


def record(name: str, ok: bool, detail: str) -> None:
    rows.append((name, ok, detail))
    print(f"  {'ok  ' if ok else 'MISS'} {name}: {detail}", flush=True)


def guarded(name: str, fn: Callable[[], None]) -> None:
    """Run one check family; an exception is a miss, not a crash."""
    print(f"\n{name}", flush=True)
    try:
        fn()
    except Exception:
        record(name, False, traceback.format_exc(limit=3).strip().splitlines()[-1])
        traceback.print_exc()


# --- goldens ---------------------------------------------------------------------


def check_goldens() -> None:
    from tests.characterization.core import cases  # noqa: PLC0415
    from tests.characterization.core.test_core_golden import (  # noqa: PLC0415
        check_family,
    )

    for family in cases.FAMILIES:
        problems, worst = check_family(family)
        record(
            f"golden {family}",
            not problems,
            f"worst rel err {worst:.2e} (limit {cases.RTOL:.0e})"
            + (
                f"; {len(problems)} mismatches, first: {problems[0]}"
                if problems
                else ""
            ),
        )


# --- P11: transfer budget with the guard live ---------------------------------------


def _same(a: object, b: object) -> bool:
    if isinstance(a, dict):  # batch: formula -> result
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)  # type: ignore[attr-defined]
    fields = ("dispersion_delta", "absorption_beta", "critical_angle_degrees")
    return all(
        np.array_equal(getattr(a, f), getattr(b, f), equal_nan=True) for f in fields
    )


def check_transfers() -> None:
    from xraylabtool.backend import runtime  # noqa: PLC0415
    from xraylabtool.calculators import core, host  # noqa: PLC0415

    energies = np.linspace(1.0, 29.0, 200)
    ones = np.ones(energies.size)
    mixed = ["Si", "SiO2", "CaCO3", "Ca5(PO4)3F", "KAl3Si3O10(OH)2", "H2O"]
    formulas = [f"Si{'O2' if i % 2 else 'C'}" for i in range(200)]
    blocks: list[int] = []
    real = host.run_optical

    def counting(*args: object, **kw: object) -> object:
        blocks.append(1)
        return real(*args, **kw)  # type: ignore[arg-type]

    cases: dict[str, Callable[[], object]] = {
        "single SiO2": lambda: core.calculate_single_material_properties(
            "SiO2", energies, 2.2
        ),
        "from factors": lambda: core.calculate_properties_from_factors(
            "SiO2", energies, 2.2, ones, ones
        ),
        "batch 200 formulas": lambda: core.calculate_xray_properties(
            formulas, energies, [1.0 + i / 100 for i in range(200)]
        ),
        "batch mixed K": lambda: core.calculate_xray_properties(
            mixed, energies, [2.0] * len(mixed)
        ),
    }
    host.run_optical = counting  # type: ignore[assignment]
    try:
        for name, call in cases.items():
            expected = call()  # warm: compiles and uploads the tables unguarded
            try:
                with jax.transfer_guard("disallow_explicit"):
                    guarded_result = call()
                guard = "guard live"
                same = _same(guarded_result, expected)
                if not same:
                    guard = "guarded result differs"
            except Exception as exc:
                same, guard = False, f"guard raised {type(exc).__name__}: {exc}"
            blocks.clear()
            runtime.reset_transfer_counts()
            call()
            counts = runtime.transfer_counts()
            n = EXPECTED_BLOCKS.get(name, 1)
            want = {"upload": n, "fetch_status": 0, "fetch": n}
            seen = len(blocks) if name in EXPECTED_BLOCKS else n
            record(
                f"P11 transfers {name}",
                same and counts == want and seen == n,
                f"{guard}; counts {counts} (want {want}); blocks {seen} (want {n})",
            )
    finally:
        host.run_optical = real  # type: ignore[assignment]


# --- padding invariance -------------------------------------------------------------


def check_padding() -> None:
    from tests.unit.core import test_optics  # noqa: PLC0415
    from xraylabtool.backend import runtime  # noqa: PLC0415

    runtime.ensure_x64()
    for name in (
        "test_padding_invariance_is_bit_exact",
        "test_tail_program_padding_invariance_is_bit_exact",
        "test_each_material_is_independent_of_its_batch_mates",
        "test_padded_rows_are_finite_and_ok",
    ):
        try:
            getattr(test_optics, name)()
            record(f"padding {name.removeprefix('test_')}", True, "passes")
        except AssertionError as exc:
            record(
                f"padding {name.removeprefix('test_')}",
                False,
                (str(exc).splitlines() or ["assertion failed"])[0],
            )


# --- P5-P7, P12, P13: fresh-process measurements -------------------------------------


def check_bench(repeats: int) -> None:
    import core_bench  # noqa: PLC0415

    for gate, (case, limit, label) in GPU_GATES_MS.items():
        runs = core_bench.spawn_runs(case, "default", "gpu", repeats)
        assert runs is not None
        warm = [r["warm_ms"] for r in runs]
        first = [r["first_ms"] for r in runs]
        med = core_bench.median_of(runs)["warm_ms"]
        print(
            f"  {label}: warm per process {[round(w, 3) for w in warm]} ms, "
            f"first call {[round(f, 1) for f in first]} ms"
        )
        record(
            f"{gate} {label} (warm, GPU)",
            med <= limit,
            f"median {med:.3f} (min {min(warm):.3f}, max {max(warm):.3f}) <= {limit:g} ms",
        )
    runs = core_bench.spawn_runs("compiles", "default", "gpu", repeats)
    assert runs is not None
    worst = {k: max(r[k] for r in runs) for k in runs[0]}
    record(
        "P12 compiles after warm-up, same buckets",
        worst["p12_same_shape"] == 0,
        f"max over {repeats} processes: {worst['p12_same_shape']:g} (limit 0)",
    )
    record(
        "P13 compiles for a new energy count in a bucket",
        worst["p13_energy_200_to_201"] == 0 and worst["p13_energy_201_to_216"] == 0,
        f"200->201: {worst['p13_energy_200_to_201']:g}, "
        f"201->216: {worst['p13_energy_201_to_216']:g} (limit 0)",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--repeats", type=int, default=5, help="fresh processes per timing"
    )
    parser.add_argument(
        "--skip-bench", action="store_true", help="skip the subprocess measurements"
    )
    args = parser.parse_args()

    # The CUDA probe and the fresh-process timings run before this process touches
    # the device: a resident parent context (it preallocates most of the GPU
    # memory) changes the timings of the child processes.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import jax; d = jax.devices()[0]; print(d.platform, d)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    platform, _, device = probe.stdout.strip().partition(" ")
    if (
        probe.returncode != 0
        or platform not in ("gpu", "cuda")
        or "cuda" not in device.lower()
    ):
        print(
            "core_gpu_check.py needs a CUDA device; JAX's default device is "
            f"{device or 'unavailable'} (platform {platform!r}). Run it on the CUDA "
            "machine, outside the sandbox, without JAX_PLATFORMS=cpu.\n"
            + probe.stderr[-500:],
            file=sys.stderr,
        )
        return 2

    import core_bench  # noqa: PLC0415

    print(f"jax {jax.__version__}  device {device}  platform {platform}")
    print(f"machine before: {core_bench.machine_state()}")
    if not args.skip_bench:
        guarded("P5-P7, P12, P13 (fresh processes)", lambda: check_bench(args.repeats))
    guarded("goldens (rtol 1e-12)", check_goldens)
    guarded("P11 transfer budget (guard live)", check_transfers)
    guarded("padding invariance", check_padding)
    print(f"machine after: {core_bench.machine_state()}")

    misses = [r for r in rows if not r[1]]
    print(f"\n{'gate / check':<52} result")
    for name, ok, detail in rows:
        print(f"{name:<52} {'ok' if ok else 'MISS'}  {detail}")
    print(f"\n{len(rows) - len(misses)} ok, {len(misses)} missed")
    return 1 if misses else 0


if __name__ == "__main__":
    sys.exit(main())
