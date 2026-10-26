#!/usr/bin/env python
"""Manual GPU gate for ``xraylabtool.kk`` (standalone, not collected by pytest).

Run on the CUDA machine (outside the sandbox)::

    uv run python scripts/kk_gpu_check.py [--repeats 15] [--skip-bench]

It repeats on the default JAX device the checks the CPU test suite makes, with
the thresholds of the test modules and of the design spec's acceptance table
(``docs/superpowers/specs/2026-10-25-kk-jax-first-design.md``), prints one
table and exits 1 if any row misses:

* independent quadrature oracle (``tests/unit/kk/test_oracle.py``);
* additivity of compounds (``tests/unit/kk/test_additivity.py``);
* transfer budget with ``jax.transfer_guard("disallow_explicit")`` live on the
  device, plus the boundary-helper counters;
* padding invariance (``tests/unit/kk/test_device_kernel.py``);
* f2 at the Si K-edge knot 1838.8 eV (and the floats beside it) against an
  independent NumPy evaluation at rtol 1e-15;
* benchmarks in fresh subprocesses (``scripts/kk_bench.py`` workers) against
  the acceptance gates of the detected platform.

It also reports, without a pass/fail threshold, the extra sort cost of the
second sort in ``spectrum._union_knots`` (``jnp.unique(size=...)`` plus the
sort that counts the distinct knots), run once per assembly and once per
near-edge merge.

Keep the output for the PR description, not the repository.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import contextlib
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

from tests.fixtures.kk_rounding import (  # noqa: E402
    ROUNDING_K,
    assert_kk_close,
    kk_rounding_scale,
)
from tests.unit.kk import test_additivity as additivity  # noqa: E402
from tests.unit.kk import test_device_kernel as kernel_tests  # noqa: E402
from tests.unit.kk import test_oracle as oracle  # noqa: E402
from xraylabtool.kk import (  # noqa: E402
    Spectrum,
    _device,
    compound_spectrum,
    kk_calculate_real,
    kk_real_spectrum,
    load_asf,
    load_nearedge,
)
from xraylabtool.kk._shapes import bucket  # noqa: E402
from xraylabtool.kk.spectrum import _union_knots  # noqa: E402
from xraylabtool.utils import parse_formula  # noqa: E402

EDGE = ROOT / "tests" / "characterization" / "kk" / "nearedge_example.txt"
SI_EDGE_EV = 1838.8
SI_EDGE_RTOL = 1e-15

# Acceptance gates (ms) of the design spec for a CUDA device; the CPU gates are
# kk_bench.GATES_MS.
GPU_GATES_MS = {
    "kernel m=10000 (ms)": 20.0,
    "refined cold (ms)": 2500.0,
}
PROGRAM_GATE = 6

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


def _spectrum(formula: str) -> tuple[np.ndarray, np.ndarray, float]:
    symbols, counts = parse_formula(formula)
    seg, coeffs = compound_spectrum(symbols, counts)
    db = load_asf()
    rel = _rel(symbols, counts, db)
    return seg, coeffs, rel


def _rel(symbols: list[str], counts: list[float], db: object) -> float:
    from xraylabtool.kk import relativistic_correction  # noqa: PLC0415

    return relativistic_correction(
        [db.symbols.index(s) + 1 for s in symbols],  # type: ignore[attr-defined]
        counts,
    )


# --- oracle ----------------------------------------------------------------


def check_oracle() -> None:
    from tests.fixtures.kk_oracle import f1_reference  # noqa: PLC0415

    for formula in oracle.FORMULAS:
        seg, coeffs, rel = _spectrum(formula)
        energy = oracle.oracle_energies(seg, coeffs)
        ref = np.array([f1_reference(e, seg, coeffs, rel) for e in energy])
        f1 = kk_real_spectrum(energy, seg, coeffs, rel)
        err = np.abs(f1 - ref)
        bound = np.maximum(
            3.0 * oracle.MEASURED_MAX[formula],
            ROUNDING_K * kk_rounding_scale(energy, seg, coeffs),
        )
        worst = int(np.argmax(err - bound))
        ok = bool(
            np.isfinite(f1).all()
            and err[worst] <= bound[worst]
            and np.median(err) < oracle.MEDIAN_LIMIT
        )
        record(
            f"oracle {formula}",
            ok,
            f"max err {err.max():.2e}, median {np.median(err):.1e}, "
            f"worst margin {err[worst] - bound[worst]:.1e}",
        )


# --- additivity --------------------------------------------------------------


def check_additivity() -> None:
    for formula in additivity.FORMULAS:
        symbols, counts = parse_formula(formula)
        seg, coeffs = compound_spectrum(symbols, counts)
        energy_kev = additivity.additivity_energies_kev(seg)
        energy_ev = energy_kev * 1e3
        whole = kk_calculate_real(formula, energies_kev=energy_kev)
        f1_sum = np.zeros_like(energy_kev)
        f2_sum = np.zeros_like(energy_kev)
        allowance = kk_rounding_scale(energy_ev, seg, coeffs)
        for symbol, count in zip(symbols, counts, strict=True):
            part = kk_calculate_real(symbol, energies_kev=energy_kev)
            f1_sum += count * part.f1
            f2_sum += count * part.f2
            seg_i, coeffs_i = compound_spectrum([symbol], [1.0])
            allowance += count * kk_rounding_scale(energy_ev, seg_i, coeffs_i)
        bound = np.maximum(
            3.0 * additivity.F1_MEASURED_MAX[formula], ROUNDING_K * allowance
        )
        err = np.abs(whole.f1 - f1_sum)
        f2_rel = np.max(np.abs(whole.f2 - f2_sum) / np.abs(f2_sum))
        ok = bool(
            np.isfinite(whole.f1).all()
            and (err <= bound).all()
            and f2_rel <= additivity.F2_RTOL
        )
        record(
            f"additivity {formula}",
            ok,
            f"f1 max {err.max():.2e}, f2 max rel {f2_rel:.1e} "
            f"(limit {additivity.F2_RTOL:.0e})",
        )


# --- transfer budget -----------------------------------------------------------


def _passes_of(fn: Callable[[], object]) -> int:
    """Refinement passes of the call, from its INFO summary."""
    import logging  # noqa: PLC0415

    class Grab(logging.Handler):
        found = 0

        def emit(self, record: logging.LogRecord) -> None:
            msg = record.getMessage()
            if " passes" in msg and " in " in msg:
                with contextlib.suppress(ValueError):
                    self.found = int(msg.split(" in ")[1].split(" passes")[0])

    logger = logging.getLogger("xraylabtool")
    handler = Grab(logging.INFO)
    old = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        fn()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old)
    return handler.found


def check_transfers() -> None:
    cases = {
        "explicit": lambda: kk_calculate_real(
            "SiO2", energies_kev=np.geomspace(0.05, 20.0, 64)
        ),
        "default": lambda: kk_calculate_real("SiO2"),
        "refined": lambda: kk_calculate_real("SiO2", refine_tolerance=0.05),
        "nearedge": lambda: kk_calculate_real(
            "C",
            nearedge=load_nearedge(str(EDGE)),
            input_type="nexafs",
            merge_points_kev=(0.28, 0.32),
        ),
    }
    for name, call in cases.items():
        expected = call()  # warm: compiles and uploads the ASF tables unguarded
        try:
            with jax.transfer_guard("disallow_explicit"):
                guarded_result = call()
            same = bool(np.array_equal(guarded_result.f1, expected.f1))
            guard = "guard live" if same else "guarded result differs"
        except Exception as exc:
            same, guard = False, f"guard raised {type(exc).__name__}: {exc}"
        _device.reset_transfer_counts()
        passes = _passes_of(call)
        counts = _device.transfer_counts()
        want = {"upload": 1, "fetch_status": 1 + passes, "fetch": 1}
        record(
            f"transfers {name}",
            same and counts == want,
            f"{guard}; counts {counts} (want {want})",
        )


# --- padding invariance --------------------------------------------------------


def check_padding() -> None:
    for name in ("Au", "SiO2"):
        case = kernel_tests._case(name)
        seg, coeffs = case["seg"], case["coeffs"]
        n = coeffs.shape[0]
        rng = np.random.default_rng(11)
        extra = np.exp(rng.uniform(np.log(10.0), np.log(5e5), 10))
        energy = np.concatenate([case["energy"], extra])
        base = kernel_tests._eval(case, n, energy)
        try:
            for capacity in (bucket(n), bucket(n + 1), bucket(bucket(n + 1) + 1)):
                out = kernel_tests._eval(case, capacity, energy)
                assert_kk_close(out, base, energy, seg, coeffs, rtol=1e-8)
            record(f"padding {name}", True, "capacities n, +1, +2 buckets agree")
        except AssertionError as exc:
            record(f"padding {name}", False, str(exc).splitlines()[0])
    rng = np.random.default_rng(5)
    seg = np.array([2.0, 3.5, 5.0, 8.0, 13.0, 21.0])
    coeffs = rng.uniform(0.5, 2.0, size=(5, 5)) * rng.choice([-1.0, 1.0], size=(5, 5))
    energy = np.concatenate([[1.0, 2.7, 4.2, 6.1, 10.0, 20.0, 50.0, 300.0], seg])
    case = {"seg": seg, "coeffs": coeffs, "rel": 1.5}
    base = kernel_tests._eval(case, 5, energy)
    worst = max(
        float(np.max(np.abs(kernel_tests._eval(case, c, energy) / base - 1.0)))
        for c in (8, 64, 640)
    )
    record("padding synthetic", worst <= 1e-12, f"max rel {worst:.1e} (limit 1e-12)")


# --- f2 at the Si edge --------------------------------------------------------------


def _f2_numpy(energy: np.ndarray, seg: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Independent NumPy evaluation: right-continuous segments, last one closed."""
    i = np.clip(np.searchsorted(seg, energy, side="right") - 1, 0, coeffs.shape[0] - 1)
    c = coeffs[i]
    return (
        c[:, 0] * energy
        + c[:, 1]
        + c[:, 2] / energy
        + c[:, 3] / energy**2
        + c[:, 4] / energy**3
    )


def check_f2_si_edge() -> None:
    seg, coeffs, _ = _spectrum("Si")
    energy = np.array(
        [
            np.nextafter(SI_EDGE_EV, 0.0),
            SI_EDGE_EV,
            np.nextafter(SI_EDGE_EV, np.inf),
        ]
    )
    assert SI_EDGE_EV in seg, "the Si table has no knot at 1838.8 eV"
    ref = _f2_numpy(energy, seg, coeffs)
    dev = _evaluate_f2_device(energy, seg, coeffs)
    rel = np.abs(dev - ref) / np.abs(ref)
    record(
        "f2 at Si edge 1838.8 eV",
        bool((rel <= SI_EDGE_RTOL).all()),
        f"max rel {rel.max():.2e} (limit {SI_EDGE_RTOL:.0e}) at "
        f"E = nextafter below / on / above the knot",
    )


def _evaluate_f2_device(
    energy: np.ndarray, seg: np.ndarray, coeffs: np.ndarray
) -> np.ndarray:
    from xraylabtool.kk.spectrum import evaluate_f2_device  # noqa: PLC0415

    out = evaluate_f2_device(_device.upload(energy), Spectrum.from_numpy(seg, coeffs))
    return np.asarray(_device.fetch(out))


# --- cost of the second sort in _union_knots -----------------------------------


def _median_ms(fn: Callable[[], object], repeats: int) -> float:
    for _ in range(3):
        jax.block_until_ready(fn())
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(1e3 * (time.perf_counter() - t))
    return statistics.median(times)


def report_union_sort_cost(repeats: int) -> None:
    db = load_asf()
    kmax = max(db.energy_ev(z).size for z in range(1, 93))

    def padded(z: int) -> np.ndarray:
        knots = db.energy_ev(z)
        return np.concatenate([knots, np.full(kmax - knots.size, np.inf)])

    assembly = np.concatenate([padded(14), padded(8)])  # Si, O
    near = np.sort(np.loadtxt(EDGE, comments="%")[:, 0])
    merge = np.concatenate([db.energy_ev(14), db.energy_ev(8), near])
    print("\nextra sort cost of the second sort in _union_knots (no threshold)")
    for label, flat in (("assembly SiO2", assembly), ("merge SiO2 + near-edge", merge)):
        size = bucket(int(np.unique(flat[np.isfinite(flat)]).size) + 2)
        x = jax.device_put(flat)
        only_unique = jax.jit(
            lambda f, s=size: jnp.unique(f, size=s, fill_value=jnp.inf)
        )
        both = jax.jit(lambda f, s=size: _union_knots(f, s))
        a = _median_ms(lambda f=only_unique, v=x: f(v), repeats)
        b = _median_ms(lambda f=both, v=x: f(v), repeats)
        print(
            f"  {label}: {flat.size} values -> {size} slots; "
            f"unique alone {a:.3f} ms, with the counting sort {b:.3f} ms, "
            f"extra {b - a:+.3f} ms"
        )


# --- benchmarks ----------------------------------------------------------------------


def check_bench(repeats: int) -> None:
    import kk_bench  # noqa: PLC0415

    platform = os.environ.get("JAX_PLATFORMS") or jax.default_backend()
    on_cpu = platform == "cpu"
    gates = kk_bench.GATES_MS if on_cpu else GPU_GATES_MS
    kern = kk_bench._run_worker("kernel", platform, repeats)
    for row in kern["rows"]:  # type: ignore[attr-defined]
        print(
            f"  kernel m={row['m']}: host wrapper {row['host_min_ms']:.1f} / "
            f"{row['host_median_ms']:.1f} ms (min / median), device kk_f1 "
            f"{row['device_min_ms']:.1f} / {row['device_median_ms']:.1f} ms"
        )
        key = f"kernel m={row['m']} (ms)"
        if key in gates:
            record(
                key,
                row["host_median_ms"] <= gates[key],
                f"{row['host_median_ms']:.1f} <= {gates[key]:g}",
            )
    first = kk_bench._run_worker("first", platform, repeats)
    if "first call, 1000 energies (ms)" in gates:
        limit = gates["first call, 1000 energies (ms)"]
        record(
            "first call, 1000 energies (ms)",
            first["cold_ms"] <= limit,
            f"{first['cold_ms']:.0f} <= {limit:g}",
        )  # type: ignore[operator]
    ref = kk_bench._run_worker("refine", platform, repeats)
    record(
        "refined cold (ms)",
        ref["cold_ms"] <= gates["refined cold (ms)"],  # type: ignore[operator]
        f"{ref['cold_ms']:.0f} <= {gates['refined cold (ms)']:g}",
    )
    if "refined warm (ms)" in gates:
        limit = gates["refined warm (ms)"]
        record(
            "refined warm (ms)",
            ref["warm_median_ms"] <= limit,
            f"median {ref['warm_median_ms']:.1f} (min {ref['warm_min_ms']:.1f}) <= {limit:g}",
        )  # type: ignore[operator]
    events = int(ref["cold_compile_events"])  # type: ignore[call-overload]
    record(
        "top-level programs, refined run",
        events <= PROGRAM_GATE,
        f"{events} <= {PROGRAM_GATE}: {ref['cold_kk_programs']}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=15, help="warm calls per timing")
    parser.add_argument(
        "--skip-bench", action="store_true", help="skip the subprocess benchmarks"
    )
    args = parser.parse_args()

    device = jax.devices()[0]
    print(
        f"jax {jax.__version__}  device {device}  platform {device.platform}  x64 {jax.config.jax_enable_x64}"
    )
    guarded("oracle", check_oracle)
    guarded("additivity", check_additivity)
    guarded("transfer budget (guard live)", check_transfers)
    guarded("padding invariance", check_padding)
    guarded("f2 at the Si edge", check_f2_si_edge)
    guarded("second sort in _union_knots", lambda: report_union_sort_cost(args.repeats))
    if not args.skip_bench:
        guarded("benchmarks (fresh subprocesses)", lambda: check_bench(args.repeats))

    misses = [r for r in rows if not r[1]]
    print(f"\n{'check':<40} result")
    for name, ok, detail in rows:
        print(f"{name:<40} {'ok' if ok else 'MISS'}  {detail}")
    print(f"\n{len(rows) - len(misses)} ok, {len(misses)} missed")
    return 1 if misses else 0


if __name__ == "__main__":
    sys.exit(main())
