# Benchmarks: v0.4.8 vs current

Milliseconds; speedup = baseline / current (>1 means the current tree is faster). Mean of the per-round medians (± is their standard deviation across rounds) over alternating rounds, one fresh process per side per round. Regenerate with `python benchmarks/perf_benchmarks.py --vs v0.4.8 --rounds 10 --stat mean --baseline-path /tmp/claude-1000/-home-wei-Documents-GitHub-pyXRayLabTool/c715663b-52ff-407a-832d-544122790440/scratchpad/baseline_deps --markdown benchmarks/RESULTS_gpu.md --json benchmarks/results_gpu.json`.

| Benchmark | Baseline ms | ± | Current ms | ± | Speedup |
|---|---:|---:|---:|---:|---:|
| Cold start (import + first calc, 500 pts) | 2668 | 4.9e+02 | 1630 | 5.9e+02 | 1.64x |
| Import xraylabtool (fresh process) | 941.3 | 2.3e+02 | 19.01 | 6.1 | 49.52x |
| Single material SiO2 (500 pts) | 3.79 | 0.98 | 0.6631 | 0.32 | 5.72x |
| Single material SiO2 (5000 pts) | 4.363 | 2 | 0.7665 | 0.27 | 5.69x |
| Multi-material batch (5 materials, 500 pts) | 21.98 | 9.5 | 1.517 | 0.89 | 14.50x |
| Multi-material batch (20 materials, 500 pts) | 85.89 | 37 | 2.362 | 0.96 | 36.36x |
| Large batch (200 materials, 100 pts) | 928.3 | 4.6e+02 | 6.156 | 2.4 | 150.79x |
| Element data load - warm (Si) | 0.0003665 | 9.4e-05 | 0.000333 | 7.4e-05 | 1.10x |
| Element data load - cold (Si) | 0.3535 | 0.084 | 0.3372 | 0.09 | 1.05x |
| XRayResult construction (1000 objects, 500 pts each) | 1.3 | 0.39 | 1.82 | 0.5 | 0.71x |
| Interpolator creation - warm (Si) | 0.000123 | 1.7e-05 | 0.0001087 | 1.6e-05 | 1.13x |
| Interpolator creation - cold (Si) | 2.797 | 1.6 | 0.001073 | 0.00029 | 2607.20x |
| Atomic data lookup (300 lookups: Si, O, Al) | 0.02761 | 0.0069 | 0.02734 | 0.0081 | 1.01x |

## Environment

| | Baseline | Current |
|---|---|---|
| xraylabtool | 0.4.8 | 0.5.0 |
| python | 3.14.0 | 3.14.0 |
| numpy | 2.5.4 | 2.5.4 |
| jax | 0.12.0.1 | 0.12.0.1 |
| devices | cuda:0 | cuda:0 |
| platform | Linux-6.8.0-147-generic-x86_64-with-glibc2.39 | Linux-6.8.0-147-generic-x86_64-with-glibc2.39 |
| cpu_count | 20 | 20 |
| load_avg_1m | 4.69 | 4.28 |
| date | 2026-10-28 13:22:26 | 2026-10-28 13:22:40 |

Notes: `Element data load`, `Interpolator creation` and `Atomic data lookup` exercise compatibility APIs that the JAX-first core keeps as thin readers of the resident tables (ADR-006), so those rows compare an API, not an algorithm. Cold-start rows include JAX import and the first compile on the current side. Host-load figures above matter: re-run on a quiet host before quoting absolute numbers.
