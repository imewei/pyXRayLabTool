# Benchmarks: v0.4.8 vs current

Milliseconds; speedup = baseline / current (>1 means the current tree is faster). Mean of the per-round medians (± is their standard deviation across rounds) over alternating rounds, one fresh process per side per round. Regenerate with `JAX_PLATFORMS=cpu python benchmarks/perf_benchmarks.py --vs v0.4.8 --rounds 10 --stat mean --markdown benchmarks/RESULTS_cpu.md --json benchmarks/results_cpu.json`.

| Benchmark | Baseline ms | ± | Current ms | ± | Speedup |
|---|---:|---:|---:|---:|---:|
| Cold start (import + first calc, 500 pts) | 604.5 | 15 | 670.8 | 9.2 | 0.90x |
| Import xraylabtool (fresh process) | 585.8 | 13 | 16.11 | 0.29 | 36.36x |
| Single material SiO2 (500 pts) | 0.1053 | 0.0044 | 0.277 | 0.014 | 0.38x |
| Single material SiO2 (5000 pts) | 0.3021 | 0.0027 | 0.6152 | 0.043 | 0.49x |
| Multi-material batch (5 materials, 500 pts) | 0.5531 | 0.0042 | 0.9935 | 0.068 | 0.56x |
| Multi-material batch (20 materials, 500 pts) | 2.114 | 0.018 | 2.108 | 0.12 | 1.00x |
| Large batch (200 materials, 100 pts) | 16.56 | 0.13 | 7.22 | 0.29 | 2.29x |
| Element data load - warm (Si) | 0.0003152 | 1.5e-05 | 0.0003108 | 2e-05 | 1.01x |
| Element data load - cold (Si) | 0.2872 | 0.011 | 0.3161 | 0.018 | 0.91x |
| XRayResult construction (1000 objects, 500 pts each) | 1.1 | 0.0097 | 1.666 | 0.028 | 0.66x |
| Interpolator creation - warm (Si) | 0.0001025 | 3.8e-06 | 0.0001021 | 4.5e-06 | 1.00x |
| Interpolator creation - cold (Si) | 0.2677 | 0.0028 | 0.0009981 | 6.9e-05 | 268.22x |
| Atomic data lookup (300 lookups: Si, O, Al) | 0.02425 | 0.00064 | 0.02437 | 0.00066 | 1.00x |

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
| load_avg_1m | 3.14 | 4.37 |
| date | 2026-10-28 13:18:38 | 2026-10-28 13:18:47 |

Notes: `Element data load`, `Interpolator creation` and `Atomic data lookup` exercise compatibility APIs that the JAX-first core keeps as thin readers of the resident tables (ADR-006), so those rows compare an API, not an algorithm. Cold-start rows include JAX import and the first compile on the current side. Host-load figures above matter: re-run on a quiet host before quoting absolute numbers.
