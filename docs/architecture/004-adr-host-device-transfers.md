# ADR-004: Host-Device Transfer Minimization Strategy

**Status:** ACCEPTED
**Date:** 2026-04-06
**Deciders:** Architecture Team
**Supersedes:** None

---

## Context

### The Transfer Problem

JAX arrays live on a "device" (CPU via XLA, GPU, or TPU). NumPy arrays live on the "host" (regular Python memory). Every conversion between JAX and NumPy arrays triggers a data transfer:

```python
jax_array = jnp.array(numpy_array)    # Host -> Device transfer
numpy_array = np.array(jax_array)      # Device -> Host transfer
numpy_array = jax_array.to_py()        # Device -> Host transfer (explicit)
```

On CPU-only systems (the primary deployment for pyXRayLabTool), the "device" is the same physical memory, so the transfer cost is a memcpy (~1us for small arrays, ~100us for large ones). On GPU, transfers go over PCIe and cost 10-100x more.

### Where Transfers Happen in the Current Architecture

Analyzing the data flow through the codebase:

```
[LOAD: numpy] -> [CACHE: numpy dict] -> [INTERPOLATE: scipy on numpy]
  -> [COMPUTE: numpy broadcast/einsum] -> [RESULT: XRayResult(numpy)]
  -> [PLOT: matplotlib takes numpy] / [EXPORT: pandas takes numpy]
```

With a naive JAX migration, transfers would occur at every boundary:

```
[LOAD: numpy] -> HOST-DEVICE -> [CACHE: jax?] -> [INTERPOLATE: interpax on jax]
  -> [COMPUTE: jax jit] -> DEVICE-HOST -> [RESULT: XRayResult(numpy)]
  -> [PLOT: pyqtgraph takes numpy] / [EXPORT: pandas takes numpy]
```

**Critical transfer points identified:**

1. **Interpolator input:** Energy values (numpy) -> interpolator (scipy/interpax)
2. **Interpolator output:** f1/f2 values -> scattering factor calculation
3. **Scattering factors -> derived quantities:** Should be zero-copy within JAX
4. **Computation result -> XRayResult:** JAX arrays -> numpy for storage
5. **XRayResult -> plotting:** numpy arrays -> PyQtGraph (which accepts numpy)
6. **XRayResult -> export:** numpy arrays -> pandas DataFrame

### Array Sizes in This Workload

| Array | Typical Size | Bytes (float64) |
|-------|-------------|-----------------|
| Energy sweep | 50-1000 points | 400B - 8KB |
| Wavelength | same as energy | 400B - 8KB |
| f1/f2 per element | same as energy | 400B - 8KB |
| Dispersion/Absorption | same as energy | 400B - 8KB |
| f1_matrix (n_elements x n_energies) | 2-10 x 50-1000 | 800B - 80KB |
| Scattering factor table (loaded from .nff) | ~500 rows x 3 cols | ~12KB |

These are small arrays. The transfer overhead matters not because of data size, but because of **frequency** -- a batch of 100 materials means 100 x N transfers.

## Decision

**Minimize host-device transfers by keeping data as JAX arrays through the entire computation pipeline, converting to NumPy only at consumption boundaries (GUI, export, CLI output).**

### Transfer Policy

| Boundary | Direction | Policy |
|----------|-----------|--------|
| File loading (.nff data) | Disk -> Host (numpy) | **Keep as numpy.** File I/O is inherently host-side. |
| Scattering data cache | Host -> Device | **Convert once at cache time.** `jnp.array(numpy_data)` when caching interpolator inputs. |
| Interpolator creation | Host -> Device | **Convert interpolator coefficients to JAX at creation time.** interpax stores coefficients as JAX arrays. |
| Energy input from user | Host -> Device | **Single `jnp.asarray()` at the entry point** of `calculate_single_material_properties()`. |
| Computation pipeline | Device -> Device | **Zero transfers.** All ops (`calculate_scattering_factors`, `calculate_derived_quantities`) stay in JAX. |
| XRayResult construction | Device -> Host | **Single bulk transfer.** Convert all result arrays to numpy in `XRayResult.__post_init__()`. |
| GUI plotting | Host (numpy) -> Qt | **Zero additional transfers.** PyQtGraph accepts numpy arrays directly. |
| CLI output | Host (numpy) -> stdout | **Zero additional transfers.** `print()` calls `.item()` or `str()`. |
| Batch export | Host (numpy) -> pandas | **Zero additional transfers.** pandas accepts numpy arrays. |

### Implementation Pattern

```python
# calculators/core.py (migrated)
def calculate_single_material_properties(formula, energy_keV, density):
    from xraylabtool.backend import ops

    # === SINGLE HOST -> DEVICE TRANSFER ===
    energy_kev = ops.asarray(energy_kev, dtype=ops.float64)

    # ... parse formula (pure Python, no arrays) ...

    # === ALL COMPUTATION ON DEVICE ===
    wavelength = ENERGY_TO_WAVELENGTH_FACTOR / energy_kev  # JAX scalar / JAX array
    energy_ev = energy_kev * 1000.0

    # Interpolation (interpax, device-side)
    f1_values = f1_interp(energy_ev)  # JAX array in, JAX array out

    # Scattering factors (JIT-compiled, all device-side)
    dispersion, absorption, f1_total, f2_total = _calculate_scattering_factors_jit(
        energy_ev, wavelength, mass_density, molecular_weight, element_data
    )

    # Derived quantities (JIT-compiled, all device-side)
    electron_density, critical_angle, attenuation_length, re_sld, im_sld = (
        _calculate_derived_quantities_jit(wavelength, dispersion, absorption, ...)
    )

    # === SINGLE DEVICE -> HOST TRANSFER ===
    return XRayResult(
        formula=formula_str,
        molecular_weight_g_mol=molecular_weight,
        # ... scalar fields ...
        energy_kev=np.asarray(energy_kev),          # JAX -> numpy
        wavelength_angstrom=np.asarray(wavelength * METER_TO_ANGSTROM),
        dispersion_delta=np.asarray(dispersion),
        absorption_beta=np.asarray(absorption),
        # ... etc ...
    )
```

### Batch Processing Optimization

For multi-material calculations, the transfer savings compound:

```python
# BEFORE (naive): N materials x 2 transfers each = 2N transfers
for formula, density in materials:
    energy_jax = jnp.asarray(energy_np)        # Transfer 1
    result = compute(energy_jax, ...)           # Device computation
    result_np = np.asarray(result)              # Transfer 2

# AFTER (optimized): 1 transfer in + 1 transfer out = 2 transfers total
energy_jax = jnp.asarray(energy_np)            # Transfer 1 (shared across materials)
all_results_jax = jax.vmap(compute)(energy_jax, material_params)  # Single vmap call
all_results_np = jax.tree.map(np.asarray, all_results_jax)        # Transfer 2 (bulk)
```

## Consequences

### Positive
- **Minimal overhead:** Only 2 transfers per single-material calculation (input + output), regardless of computation complexity.
- **Batch efficiency:** `vmap`-based batch processing shares the input transfer, reducing from 2N to 2 transfers for N materials.
- **JIT effectiveness:** Keeping data on-device allows XLA to fuse the entire computation pipeline into a single kernel, maximizing JIT benefits.
- **Future GPU readiness:** If GPU support is added later, the transfer minimization strategy prevents PCIe bottlenecks.

### Negative
- **Conversion at XRayResult:** The `XRayResult` dataclass stores numpy arrays, requiring a bulk device-to-host conversion at result creation. This is intentional -- `XRayResult` is the public API boundary, and downstream consumers (GUI, export, CLI) all expect numpy.
- **Cache storage:** Scattering factor data is cached as JAX arrays, using device memory. For 92 elements x ~12KB each, this is ~1MB -- negligible.
- **Cannot mix backends in computation:** Once data enters the JAX pipeline, all intermediate operations must use JAX ops. No falling back to scipy mid-computation.

### Monitoring

The existing `MemoryMonitor` in `batch_processing.py` will be extended to track:
- Number of host-device transfers per calculation
- Total bytes transferred per calculation
- Transfer time as percentage of total computation time

This data feeds into the `bottleneck_analyzer.py` reporting.

---

## Appendix: Transfer Cost Reference (CPU Backend)

| Array Size | Transfer Time (est.) | Notes |
|-----------|---------------------|-------|
| 100 floats (800B) | ~1us | Negligible |
| 1000 floats (8KB) | ~5us | Negligible |
| 10000 floats (80KB) | ~50us | Noticeable in tight loops |
| 100000 floats (800KB) | ~500us | Should avoid in hot path |

For a typical calculation with 200-point energy sweep and 3 elements:
- Input transfer: 200 floats = ~1us
- Output transfer: 200 x 10 fields = 2000 floats = ~10us
- Total transfer overhead: ~11us out of ~1ms total computation = ~1%

This confirms that the 2-transfer strategy keeps overhead well below 5% of computation time.
