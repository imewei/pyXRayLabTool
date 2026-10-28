# Benchmarks

`perf_benchmarks.py` times the public pipeline (cold start, warm single material,
batches, compatibility APIs) on the current tree and, with `--vs REF`, on a released
tag extracted with `git archive`, then prints and saves the comparison. Options and
the exact commands are in the script docstring.

| File | Content |
|---|---|
| `RESULTS_cpu.md`, `results_cpu.json` | v0.4.8 vs v0.5.0 (the JAX-first core), both pinned to the CPU (`JAX_PLATFORMS=cpu`; v0.4.8 runs its NumPy path). Mean of 10 rounds, run on 2026-10-28. |
| `RESULTS_gpu.md`, `results_gpu.json` | the same on a CUDA host with default devices (RTX 4090 Laptop, v0.4.8 on its JAX backend). Mean of 10 rounds, run on 2026-10-28. |

Each table is the mean of the per-round medians over 10 alternating rounds (one fresh
process per side per round) with their standard deviation (`--stat mean --rounds 10`;
the default `--stat best` keeps the best round instead).

Reading them:

- **CPU (v0.4.8 to v0.5.0):** the new core wins on import (about 36x), large batches
  (200 materials, about 2.3x) and `Interpolator creation - cold`, and 20 materials are
  at parity (1.00x). It is **slower for warm small calls**: single material 500 points
  0.105 ms to 0.277 ms (5000 points 0.30 ms to 0.62 ms), 5 materials 0.55 ms to 0.99 ms,
  because each public call pays one device upload, one program launch and one fetch.
  Cold start (import plus first call) is about 10% slower because the first call
  compiles. Against the first JAX-first build (single material 0.51 ms, 5 materials
  1.40 ms), the single-call latency work cut warm small calls by about half.
- `XRayResult construction` is 1.10 ms to 1.67 ms per 1000 results (0.5 us each, CPU):
  the constructor now normalizes float32 and integer arrays to float64 and rejects
  complex values, which costs one dtype check per field.
- **GPU (v0.4.8 on its JAX backend to v0.5.0):** every calculation row is faster:
  single material about 5.7x (3.8 ms to 0.66 ms), 5 materials about 14x, 20 materials
  about 36x, 200 materials about 150x; import about 50x. Cold start is about 1.6x
  faster. The GPU run was noisy (load average rose from about 4.7 to 27 during it, and
  the round-to-round spread is 30 to 50% on several rows), so quote the ratios rather
  than the absolute milliseconds.
- `Element data load`, `Interpolator creation` and `Atomic data lookup` compare
  compatibility APIs, not algorithms; the large `Interpolator creation - cold`
  ratio only reflects that the new `TableInterpolator` is a thin reader.
- v0.4.8 on a CUDA host imports `interpax`, which the current tree no longer
  needs. The GPU run installed it into a throwaway directory
  (`uv pip install --target DIR --no-deps interpax equinox jaxtyping lineax
  wadler_lindig jaxkd`) and passed it with `--baseline-path DIR`.
- The recorded runs were on a busy laptop (load average 3 to 4.5 on 20 cores for the CPU
  run). For absolute numbers use a quiet host; `scripts/core_bench.py` and
  `scripts/core_gpu_check.py` hold the acceptance gates (P1 to P13) with their own
  quiet-host checks.
