# Benchmarks: v0.4.8 vs current

Medians in milliseconds; speedup = baseline / current (>1 means the current tree is faster). Best median over alternating rounds, one fresh process per side per round. Regenerate with `JAX_PLATFORMS=cpu python benchmarks/perf_benchmarks.py --vs v0.4.8 --rounds 3 --markdown benchmarks/RESULTS_cpu.md --json benchmarks/results_cpu.json`.

| Benchmark | Baseline ms | Current ms | Speedup |
|---|---:|---:|---:|
| Cold start (import + first calc, 500 pts) | 542.4 | 641.9 | 0.85x |
| Import xraylabtool (fresh process) | 523.4 | 15.19 | 34.46x |
| Single material SiO2 (500 pts) | 0.09887 | 0.5148 | 0.19x |
| Single material SiO2 (5000 pts) | 0.2987 | 1.133 | 0.26x |
| Multi-material batch (5 materials, 500 pts) | 0.5449 | 1.398 | 0.39x |
| Multi-material batch (20 materials, 500 pts) | 2.081 | 2.49 | 0.84x |
| Large batch (200 materials, 100 pts) | 16.15 | 7.072 | 2.28x |
| Element data load - warm (Si) | 0.000285 | 0.000286 | 1.00x |
| Element data load - cold (Si) | 0.2704 | 0.2914 | 0.93x |
| XRayResult construction (1000 objects, 500 pts each) | 1.092 | 1.091 | 1.00x |
| Interpolator creation - warm (Si) | 9.6e-05 | 9.1e-05 | 1.05x |
| Interpolator creation - cold (Si) | 0.2637 | 0.000893 | 295.30x |
| Atomic data lookup (300 lookups: Si, O, Al) | 0.0232 | 0.02313 | 1.00x |

## Environment

| | Baseline | Current |
|---|---|---|
| xraylabtool | 0.4.8 | 0.4.8 |
| python | 3.14.0 | 3.14.0 |
| numpy | 2.5.4 | 2.5.4 |
| jax | 0.12.0.1 | 0.12.0.1 |
| devices | cpu:0 | cpu:0 |
| platform | Linux-6.8.0-147-generic-x86_64-with-glibc2.39 | Linux-6.8.0-147-generic-x86_64-with-glibc2.39 |
| cpu_count | 20 | 20 |
| load_avg_1m | 2.03 | 2.59 |
| date | 2026-10-28 00:44:28 | 2026-10-28 00:44:37 |

Notes: `Element data load`, `Interpolator creation` and `Atomic data lookup` exercise compatibility APIs that the JAX-first core keeps as thin readers of the resident tables (ADR-006), so those rows compare an API, not an algorithm. Cold-start rows include JAX import and the first compile on the current side. Host-load figures above matter: re-run on a quiet host before quoting absolute numbers.
