#!/usr/bin/env python
"""Standalone benchmark of the ``xraylabtool.kk`` transform (not collected by pytest).

Every cold measurement runs in a fresh subprocess with ``JAX_PLATFORMS`` set
before ``jax`` is imported::

    uv run python scripts/kk_bench.py [--platform cpu] [--backends numpy,jax]

Measures, for SiO2 (1201 segments): the kernel ``kk_real_spectrum`` at
m = 100 / 1 000 / 10 000 / 100 000 log-uniform energies (first call including
compilation, then the warm median of five; the JAX result is waited on with
``block_until_ready`` when the call returns a device array, otherwise the
public function has already downloaded it), and
``kk_calculate_real("SiO2", refine_tolerance=0.01)`` cold and warm (median of
three). On a JAX backend the compile events of the cold refined run are
recorded from the ``jax._src.interpreters.pxla`` logger (``jax.log_compiles``)
and counted; this is replaced by the compile recorder once it exists.

``--backends`` selects ``set_backend`` values; if the package has no
``set_backend`` (after the JAX-first migration) the single default backend is
measured and labelled ``default``.

Pre-migration table (commit 1bf56d9, jax 0.12.0.1, numpy 2.5.4, CPU only: 20
cores, no GPU in this sandbox; ms unless noted)::

    numpy backend, kernel (first call / warm median, ms)
      m = 100       18.7 / 17.7
      m = 1 000     139.9 / 157.3
      m = 10 000    1910 / 1892
      m = 100 000   21994 / 14746
    numpy backend, kk_calculate_real("SiO2", refine_tolerance=0.01):
      cold 0.29 s, warm 0.273 s, 1383 points (267 refined)

    jax backend (CPU), import jax + x64 + device init: 405 ms
      m = 100       128.8 / 2.8
      m = 1 000     142.2 / 20.7
      m = 10 000    259.1 / 150.9
      m = 100 000   1215 / 1309
    jax backend, kk_calculate_real("SiO2", refine_tolerance=0.01):
      cold 2.29 s (1.58 s in an earlier run), warm 0.067 s, 1383 points
      cold compile events 14: jit(chunked) x13 (one per distinct array
      shape), jit(convert_element_type) x1

Run-to-run spread on this shared machine is up to 30 % (NumPy m = 10 000:
1.36-1.91 s in two runs); treat the numbers as indicative.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

SIZES = (100, 1_000, 10_000, 100_000)
KERNEL_REPEATS = 5
REFINE_REPEATS = 3
REFINE_TOLERANCE = 0.01
SEED = 2026
MARKER = "KKBENCH_JSON:"


def _select_backend(name: str) -> str:
    """Apply ``set_backend(name)`` if the package still has it; return the label."""
    if name == "default":
        return name
    from xraylabtool.backend import array_ops  # noqa: PLC0415

    if not hasattr(array_ops, "set_backend"):
        return "default"
    array_ops.set_backend(name)
    return name


def _wait(value: object) -> None:
    block = getattr(value, "block_until_ready", None)
    if block is not None:
        block()


def _kernel_worker(backend: str) -> dict[str, object]:
    t0 = time.perf_counter()
    import numpy as np  # noqa: PLC0415

    jax_import_ms = None
    if backend in ("jax", "default"):
        try:
            import jax  # noqa: PLC0415

            jax.config.update("jax_enable_x64", True)
            jax.numpy.zeros(1).block_until_ready()
            jax_import_ms = 1e3 * (time.perf_counter() - t0)
        except ImportError:
            pass
    label = _select_backend(backend)
    from xraylabtool.kk import (  # noqa: PLC0415
        compound_spectrum,
        kk_real_spectrum,
        load_asf,
        relativistic_correction,
    )
    from xraylabtool.utils import parse_formula  # noqa: PLC0415

    symbols, counts = parse_formula("SiO2")
    seg, coeffs = compound_spectrum(symbols, counts)
    db = load_asf()
    rel = relativistic_correction([db.symbols.index(s) + 1 for s in symbols], counts)
    rng = np.random.default_rng(SEED)
    rows = []
    for m in SIZES:
        energy = np.exp(rng.uniform(np.log(10.0), np.log(5e5), m))
        t = time.perf_counter()
        _wait(kk_real_spectrum(energy, seg, coeffs, rel))
        first = 1e3 * (time.perf_counter() - t)
        warm = []
        for _ in range(KERNEL_REPEATS):
            t = time.perf_counter()
            _wait(kk_real_spectrum(energy, seg, coeffs, rel))
            warm.append(1e3 * (time.perf_counter() - t))
        rows.append({"m": m, "first_ms": first, "warm_ms": statistics.median(warm)})
    return {
        "backend": label,
        "n_segments": int(seg.size - 1),
        "jax_import_ms": jax_import_ms,
        "rows": rows,
    }


class _CompileRecorder(logging.Handler):
    """Collect ``Compiling <name> ...`` records from ``jax._src.interpreters.pxla``."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.names: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith("Compiling "):
            self.names.append(message.split()[1])


def _refine_worker(backend: str) -> dict[str, object]:
    label = _select_backend(backend)
    recorder = _CompileRecorder()
    ctx = None
    try:
        if backend == "numpy":
            raise ImportError  # keep jax out of the NumPy measurement
        import jax  # noqa: PLC0415

        pxla = logging.getLogger("jax._src.interpreters.pxla")
        pxla.addHandler(recorder)
        pxla.setLevel(logging.DEBUG)
        ctx = jax.log_compiles(True)
    except ImportError:
        pass
    from xraylabtool.kk import kk_calculate_real  # noqa: PLC0415

    def call() -> object:
        return kk_calculate_real("SiO2", refine_tolerance=REFINE_TOLERANCE)

    if ctx is not None:
        ctx.__enter__()
    t = time.perf_counter()
    result = call()
    cold = time.perf_counter() - t
    if ctx is not None:
        ctx.__exit__(None, None, None)
    n_compiles = len(recorder.names)
    warm = []
    for _ in range(REFINE_REPEATS):
        t = time.perf_counter()
        call()
        warm.append(time.perf_counter() - t)
    return {
        "backend": label,
        "cold_s": cold,
        "warm_s": statistics.median(warm),
        "n_points": int(result.energy_kev.size),  # type: ignore[attr-defined]
        "n_refined": int(result.n_refined_points),  # type: ignore[attr-defined]
        "cold_compile_events": n_compiles,
        "cold_distinct_programs": len(set(recorder.names)),
        "program_names": [
            f"{k} x{v}" for k, v in sorted(Counter(recorder.names).items())
        ],
    }


def _run_worker(kind: str, backend: str, platform: str) -> dict[str, object]:
    env = {**os.environ, "JAX_PLATFORMS": platform, "PYTHONWARNINGS": "ignore"}
    proc = subprocess.run(
        [sys.executable, __file__, "--worker", kind, "--backend", backend],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(MARKER):
            data: dict[str, object] = json.loads(line[len(MARKER) :])
            return data
    raise SystemExit(f"{kind}/{backend} worker failed:\n{proc.stderr[-2000:]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--platform", default="cpu", help="JAX_PLATFORMS value")
    parser.add_argument("--backends", default="numpy,jax")
    parser.add_argument("--worker", choices=["kernel", "refine"])
    parser.add_argument("--backend", default="default")
    args = parser.parse_args()

    if args.worker:
        work = _kernel_worker if args.worker == "kernel" else _refine_worker
        print(MARKER + json.dumps(work(args.backend)))
        return

    print(f"platform={args.platform}  repo={Path(__file__).resolve().parents[1]}")
    for backend in args.backends.split(","):
        kern = _run_worker("kernel", backend, args.platform)
        label = kern["backend"]
        print(f"\n[{label}] kernel, SiO2 ({kern['n_segments']} segments)")
        if kern["jax_import_ms"] is not None:
            print(f"  import jax + x64 + device init: {kern['jax_import_ms']:.0f} ms")
        print("  m        first call (ms)   warm median (ms)")
        for row in kern["rows"]:  # type: ignore[attr-defined]
            print(
                f"  {row['m']:<8d} {row['first_ms']:>14.1f}   {row['warm_ms']:>14.1f}"
            )
        ref = _run_worker("refine", backend, args.platform)
        print(
            f"[{label}] kk_calculate_real('SiO2', refine_tolerance={REFINE_TOLERANCE}): "
            f"cold {ref['cold_s']:.2f} s, warm {ref['warm_s']:.3f} s, "
            f"{ref['n_points']} points ({ref['n_refined']} refined)"
        )
        print(
            f"  cold compile events {ref['cold_compile_events']}, "
            f"distinct programs {ref['cold_distinct_programs']}: "
            f"{', '.join(ref['program_names'])}"  # type: ignore[arg-type]
        )


if __name__ == "__main__":
    main()
