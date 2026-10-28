# ADR-004: Host-Device Transfer Minimization Strategy

**Status:** ACCEPTED
**Date:** 2026-04-06
**Deciders:** Architecture Team
**Supersedes:** None
**Updated by:** ADR-006 (the shipped flow below replaces the `ops`-based sketch; the policy is unchanged)

---

## Context

### The Transfer Problem

JAX arrays live on a "device" (CPU via XLA, GPU, or TPU). NumPy arrays live on the "host" (regular Python memory). Every conversion between JAX and NumPy arrays triggers a data transfer:

```python
jax_array = jnp.array(numpy_array)  # Host -> Device transfer
numpy_array = np.array(jax_array)  # Device -> Host transfer
numpy_array = jax_array.to_py()  # Device -> Host transfer (explicit)
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
[LOAD: numpy] -> HOST-DEVICE -> [CACHE: jax?] -> [INTERPOLATE: JaxPchipInterpolator on jax]
  -> [COMPUTE: jax jit] -> DEVICE-HOST -> [RESULT: XRayResult(numpy)]
  -> [PLOT: pyqtgraph takes numpy] / [EXPORT: pandas takes numpy]
```

**Critical transfer points identified:**

1. **Interpolator input:** Energy values (numpy) -> interpolator (scipy / JaxPchipInterpolator)
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
| Scattering data cache | Host -> Device | **Upload once.** `load_tables()` reads `scattering_tables.npz` and uploads the stacked tables once; they are a program argument. |
| Interpolator creation | Host -> Device | **No per-element interpolators.** The PCHIP nodes and slopes of all 92 elements are rows of the resident tables. |
| Energy input from user | Host -> Device | **One `runtime.upload`** of the encoded formula, density and energies at the entry point of `calculate_single_material_properties()`. |
| Computation pipeline | Device -> Device | **Zero transfers.** One compiled program (`optics.optical_properties`) covers interpolation, scattering factors and derived quantities. |
| XRayResult construction | Device -> Host | **One `runtime.fetch`** of the whole result tree; the host slices off padding and builds `XRayResult`. |
| GUI plotting | Host (numpy) -> Qt | **Zero additional transfers.** PyQtGraph accepts numpy arrays directly. |
| CLI output | Host (numpy) -> stdout | **Zero additional transfers.** `print()` calls `.item()` or `str()`. |
| Batch export | Host (numpy) -> pandas | **Zero additional transfers.** pandas accepts numpy arrays. |

### Implementation Pattern

The shipped flow (ADR-006). Only `runtime.upload`, `runtime.fetch_status` and
`runtime.fetch` cross the boundary; each opens a narrow
`jax.transfer_guard("allow")` scope and bumps a counter, and tests run the
device region under `jax.transfer_guard("disallow_explicit")` and assert the
counters (at most 1 upload and 1 fetch per device program call).

```
host:   validate formula, energies, density; encode z_idx, counts, mw, n_electrons
  |-> runtime.upload  ((z_idx, counts, mw, rho, n_electrons, energies_keV), one call)
  |-> device: one compiled program
  |     interpolate f1/f2 from the resident tables -> delta, beta
  |     -> critical angle, attenuation length, SLD, electron density, ok flag
  \-> runtime.fetch   ({"out": (F, 9, M), "electron_density": (F,), "ok": (F,)}, one call)
host:   slice off padding, raise on a failed status, build XRayResult (NumPy)
```

```python
# calculators/host.py (sketch of run_optical)
def run_optical(batch, energies_kev):
    tables = device_tables()  # resident, uploaded once
    device_batch = runtime.upload((*batch, energies_kev))  # SINGLE HOST -> DEVICE
    out = _optical_program(tables, device_batch)  # ALL COMPUTATION ON DEVICE
    return runtime.fetch(out)  # SINGLE DEVICE -> HOST
```

### Batch Processing Optimization

For multi-material calculations, the transfer savings compound:

```python
# BEFORE (naive): N materials x 2 transfers each = 2N transfers
for formula, density in materials:
    energy_jax = jnp.asarray(energy_np)  # Transfer 1
    result = compute(energy_jax, ...)  # Device computation
    result_np = np.asarray(result)  # Transfer 2

# AFTER (shipped): one upload and one fetch per F-block of materials
device_batch = runtime.upload(
    (z_idx, counts, mw, rho, n_electrons, energies)
)  # Transfer 1
out = _optical_program(tables, device_batch)  # vmap over materials inside the program
host_out = runtime.fetch(out)  # Transfer 2 (bulk)
```

## Consequences

### Positive
- **Minimal overhead:** Only 2 transfers per single-material calculation (input + output), regardless of computation complexity.
- **Batch efficiency:** `vmap`-based batch processing shares the input transfer, reducing from 2N to 2 transfers for N materials.
- **JIT effectiveness:** Keeping data on-device allows XLA to fuse the entire computation pipeline into a single kernel, maximizing JIT benefits.
- **Future GPU readiness:** If GPU support is added later, the transfer minimization strategy prevents PCIe bottlenecks.

### Negative
- **Conversion at XRayResult:** The `XRayResult` dataclass stores numpy arrays, requiring a bulk device-to-host conversion at result creation. This is intentional -- `XRayResult` is the public API boundary, and downstream consumers (GUI, export, CLI) all expect numpy.
- **Cache storage:** The scattering tables are resident device arrays: 92 elements padded to 449 knots, five arrays, 1.7 MB in float64 -- negligible.
- **Cannot mix backends in computation:** Once data enters the JAX pipeline, all intermediate operations must use JAX ops. No falling back to scipy mid-computation.

### Monitoring

`xraylabtool.backend.runtime` counts uploads and fetches (`transfer_counts`,
`thread_transfer_counts`, `reset_transfer_counts`); the transfer-budget tests
assert them. `MemoryMonitor` in `batch_processing.py` still checks host result
memory only.

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
