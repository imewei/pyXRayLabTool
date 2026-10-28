# Benchmarks

`perf_benchmarks.py` times the public pipeline (cold start, warm single material,
batches, compatibility APIs) on the current tree and, with `--vs REF`, on a released
tag extracted with `git archive`, then prints and saves the comparison. Options and
the exact commands are in the script docstring.

| File | Content |
|---|---|
| `RESULTS_cpu.md`, `results_cpu.json` | v0.4.8 vs the JAX-first core, both pinned to the CPU (`JAX_PLATFORMS=cpu`; v0.4.8 runs its NumPy path) |
| `RESULTS_gpu.md`, `results_gpu.json` | the same on a CUDA host with default devices (RTX 4090 Laptop, v0.4.8 on its JAX backend) |

Reading them:

- **CPU:** the new core wins on import (about 34x), large batches (about 2.3x) and
  `Interpolator creation - cold`, but is **slower for warm small calls**
  (single material 0.1 ms to 0.5 ms; 5 materials 0.5 ms to 1.4 ms): each public call
  now pays one device upload, one program launch and one fetch. Cold start (import
  plus first call) is about 15% slower because the first call compiles.
- **GPU:** every calculation row is faster (single material about 5x, 20 materials
  about 35x, 200 materials about 140x).
- `Element data load`, `Interpolator creation` and `Atomic data lookup` compare
  compatibility APIs, not algorithms; the large `Interpolator creation - cold`
  ratio only reflects that the new `TableInterpolator` is a thin reader.
- v0.4.8 on a CUDA host imports `interpax`, which the current tree no longer
  needs. The GPU run installed it into a throwaway directory
  (`uv pip install --target DIR --no-deps interpax equinox jaxtyping lineax
  wadler_lindig jaxkd`) and passed it with `--baseline-path DIR`.
- The recorded runs were on a busy laptop (load average 2 to 4 on 20 cores), best of
  three rounds. For absolute numbers use a quiet host; `scripts/core_bench.py` and
  `scripts/core_gpu_check.py` hold the acceptance gates (P1 to P13) with their own
  quiet-host checks.
