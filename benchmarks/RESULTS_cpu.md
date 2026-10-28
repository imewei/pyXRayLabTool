# Benchmarks: v0.4.8 vs current

Medians in milliseconds; speedup = baseline / current (>1 means the current tree is faster). Best median over alternating rounds, one fresh process per side per round. Regenerate with `JAX_PLATFORMS=cpu python benchmarks/perf_benchmarks.py --vs v0.4.8 --rounds 3 --markdown benchmarks/RESULTS_cpu.md --json benchmarks/results_cpu.json`.

| Benchmark | Baseline ms | Current ms | Speedup |
|---|---:|---:|---:|
| Cold start (import + first calc, 500 pts) | 589.5 | 666.7 | 0.88x |
| Import xraylabtool (fresh process) | 564.5 | 15.84 | 35.65x |
| Single material SiO2 (500 pts) | 0.103 | 0.2644 | 0.39x |
| Single material SiO2 (5000 pts) | 0.3028 | 0.5834 | 0.52x |
| Multi-material batch (5 materials, 500 pts) | 0.5498 | 0.9299 | 0.59x |
| Multi-material batch (20 materials, 500 pts) | 2.104 | 1.979 | 1.06x |
| Large batch (200 materials, 100 pts) | 16.52 | 6.508 | 2.54x |
| Element data load - warm (Si) | 0.00031 | 0.000307 | 1.01x |
| Element data load - cold (Si) | 0.2803 | 0.3002 | 0.93x |
| XRayResult construction (1000 objects, 500 pts each) | 1.102 | 1.605 | 0.69x |
| Interpolator creation - warm (Si) | 9.8e-05 | 9.6e-05 | 1.02x |
| Interpolator creation - cold (Si) | 0.2675 | 0.0009515 | 281.12x |
| Atomic data lookup (300 lookups: Si, O, Al) | 0.02393 | 0.02362 | 1.01x |

## Environment

| | Baseline | Current |
|---|---|---|
| xraylabtool | 0.4.8 | 0.5.0 |
| python | 3.14.0 | 3.14.0 |
| numpy | 2.5.4 | 2.5.4 |
| jax | 0.12.0.1 | 0.12.0.1 |
| devices | cpu:0 | cpu:0 |
| platform | Linux-6.8.0-147-generic-x86_64-with-glibc2.39 | Linux-6.8.0-147-generic-x86_64-with-glibc2.39 |
| cpu_count | 20 | 20 |
| load_avg_1m | 3.29 | 3.27 |
| date | 2026-10-28 13:02:01 | 2026-10-28 13:02:09 |

Notes: `Element data load`, `Interpolator creation` and `Atomic data lookup` exercise compatibility APIs that the JAX-first core keeps as thin readers of the resident tables (ADR-006), so those rows compare an API, not an algorithm. Cold-start rows include JAX import and the first compile on the current side. Host-load figures above matter: re-run on a quiet host before quoting absolute numbers.
