# Benchmarks: v0.4.8 vs current

Medians in milliseconds; speedup = baseline / current (>1 means the current tree is faster). Best median over alternating rounds, one fresh process per side per round. Regenerate with `python benchmarks/perf_benchmarks.py --vs v0.4.8 --rounds 3 --baseline-path <dir with interpax> --markdown benchmarks/RESULTS_gpu.md --json benchmarks/results_gpu.json`.

| Benchmark | Baseline ms | Current ms | Speedup |
|---|---:|---:|---:|
| Cold start (import + first calc, 500 pts) | 2277 | 1041 | 2.19x |
| Import xraylabtool (fresh process) | 763.9 | 15.64 | 48.84x |
| Single material SiO2 (500 pts) | 3 | 0.6059 | 4.95x |
| Single material SiO2 (5000 pts) | 3.112 | 0.6838 | 4.55x |
| Multi-material batch (5 materials, 500 pts) | 15.04 | 1.279 | 11.77x |
| Multi-material batch (20 materials, 500 pts) | 62.09 | 1.757 | 35.34x |
| Large batch (200 materials, 100 pts) | 618.7 | 4.321 | 143.21x |
| Element data load - warm (Si) | 0.000303 | 0.000295 | 1.03x |
| Element data load - cold (Si) | 0.2907 | 0.2987 | 0.97x |
| XRayResult construction (1000 objects, 500 pts each) | 1.115 | 1.089 | 1.02x |
| Interpolator creation - warm (Si) | 0.000103 | 9.2e-05 | 1.12x |
| Interpolator creation - cold (Si) | 1.84 | 0.000908 | 2026.44x |
| Atomic data lookup (300 lookups: Si, O, Al) | 0.02379 | 0.02316 | 1.03x |

## Environment

| | Baseline | Current |
|---|---|---|
| xraylabtool | 0.4.8 | 0.4.8 |
| python | 3.14.0 | 3.14.0 |
| numpy | 2.5.4 | 2.5.4 |
| jax | 0.12.0.1 | 0.12.0.1 |
| devices | cuda:0 | cuda:0 |
| platform | Linux-6.8.0-147-generic-x86_64-with-glibc2.39 | Linux-6.8.0-147-generic-x86_64-with-glibc2.39 |
| cpu_count | 20 | 20 |
| load_avg_1m | 2.06 | 1.9 |
| date | 2026-10-28 00:48:46 | 2026-10-28 00:48:59 |

Notes: `Element data load`, `Interpolator creation` and `Atomic data lookup` exercise compatibility APIs that the JAX-first core keeps as thin readers of the resident tables (ADR-006), so those rows compare an API, not an algorithm. Cold-start rows include JAX import and the first compile on the current side. Host-load figures above matter: re-run on a quiet host before quoting absolute numbers.
