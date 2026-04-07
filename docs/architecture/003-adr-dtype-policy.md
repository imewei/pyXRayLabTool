# ADR-003: float64 Preservation Policy

**Status:** ACCEPTED
**Date:** 2026-04-06
**Deciders:** Architecture Team
**Supersedes:** None

---

## Context

### The float64 Requirement in X-ray Calculations

pyXRayLabTool calculates X-ray optical properties that span many orders of magnitude:

| Quantity | Typical Range | Precision Needed |
|----------|--------------|------------------|
| Dispersion delta | 1e-5 to 1e-8 | Relative 1e-6 |
| Absorption beta | 1e-6 to 1e-10 | Relative 1e-6 |
| Attenuation length (cm) | 1e-4 to 1e+2 | Relative 1e-4 |
| Critical angle (degrees) | 0.01 to 1.0 | Absolute 1e-4 |
| SLD (Ang^-2) | 1e-5 to 1e-6 | Relative 1e-6 |
| Energy (eV) | 30 to 30000 | Absolute 0.1 |
| Wavelength (m) | 4e-11 to 4e-8 | Relative 1e-8 |

The computation involves products of physical constants spanning 30+ orders of magnitude:

```python
# From constants.py
THOMPSON = 2.8179403227e-15    # meters
AVOGADRO = 6.02214199e23       # mol^-1
SCATTERING_FACTOR = THOMPSON * AVOGADRO * 1e6 / (2 * pi)  # ~2.7e14

# From core.py calculate_scattering_factors():
common_factor = SCATTERING_FACTOR * mass_density / molecular_weight
wave_sq = wavelength ** 2    # ~(1e-10)^2 = 1e-20
dispersion = wave_sq * common_factor * f1_total
# = 1e-20 * 2.7e14 * density/MW * f1 -> ~1e-6
```

The intermediate product `wave_sq * common_factor` is ~1e-6, and the final dispersion is also ~1e-6. With float32 (7 decimal digits of precision), the relative error in dispersion would be ~1e-1, which is catastrophically wrong. **float64 is non-negotiable for correctness.**

### JAX float32 Default

JAX defaults to float32 for all operations. This is a deliberate design choice for ML workloads where float32 is sufficient and 2x faster. However, JAX provides a configuration flag:

```python
jax.config.update("jax_enable_x64", True)  # Enable float64 globally
```

When enabled, `jnp.array([1.0])` creates a float64 array, matching NumPy behavior.

### Current Codebase float64 Usage

The codebase is consistently float64:

```python
# core.py -- explicit float64 everywhere
np.zeros(n_energies, dtype=np.float64, order="C")
np.empty((n_elements, n_energies), dtype=np.float64, order="C")
np.asarray(f1_values, dtype=np.float64)

# vectorized_core.py
np.empty((n_elements, n_energies), dtype=np.float64, order="C")

# batch_processing.py
np.array([float(energies)], dtype=np.float64)
```

All array types in `typing_extensions.py` are defined as `NDArray[np.float64]`.

## Decision

**Enforce float64 globally via JAX configuration at application startup. Treat float32 computation as a correctness bug.**

Implementation:
1. Set `jax.config.update("jax_enable_x64", True)` in `xraylabtool/__init__.py` before any JAX imports.
2. Add a runtime assertion in `backend/jax_impl.py` that verifies x64 mode is enabled.
3. All explicit `dtype=np.float64` specifications in the codebase are preserved (they become `dtype=jnp.float64`).
4. The `validate_energy_array()` function in `typing_extensions.py` already checks `dtype in [np.float32, np.float64]` -- tighten this to float64-only.
5. Add a CI check that runs the test suite with `JAX_ENABLE_X64=0` and expects failures (proving that float64 is correctly enforced).

## Consequences

### Positive
- **Correctness guarantee:** No possibility of silent float32 precision loss in X-ray calculations.
- **Behavioral parity:** `jnp.array([1.0]).dtype == jnp.float64`, identical to NumPy.
- **Explicit contract:** The dtype policy is documented and enforced, not implicit.

### Negative
- **Performance cost:** float64 operations are ~2x slower than float32 on GPU and ~1.3x slower on CPU. For this workload (X-ray property calculations, not ML training), this is an acceptable tradeoff -- correctness is Priority 1 per CLAUDE.md.
- **Memory cost:** float64 arrays use 2x the memory of float32. For typical energy sweeps (50-1000 points, 1-10 materials), this means kilobytes, not a concern.
- **JAX ecosystem friction:** Some JAX libraries default to float32. Any third-party JAX code must be validated for float64 compatibility.

### Enforcement Strategy

```python
# xraylabtool/__init__.py (FIRST LINES, before any other imports)
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

# Verify enforcement at import time
def _verify_float64():
    """Verify that JAX float64 mode is enabled."""
    try:
        import jax
        import jax.numpy as jnp
        if not jax.config.jax_enable_x64:
            raise RuntimeError(
                "JAX float64 mode is required for X-ray calculations. "
                "Set JAX_ENABLE_X64=1 or call "
                "jax.config.update('jax_enable_x64', True)"
            )
        # Verify actual behavior
        test_val = jnp.array(1.0)
        if test_val.dtype != jnp.float64:
            raise RuntimeError("JAX float64 mode enabled but arrays are float32")
    except ImportError:
        pass  # JAX not installed, numpy backend handles this

_verify_float64()
```

---

## Appendix: Numerical Precision Requirements by Function

| Function | Min Precision Required | float32 OK? | float64 OK? |
|----------|----------------------|-------------|-------------|
| `energy_to_wavelength` | 1e-8 relative | NO (6e-7 range) | YES |
| `calculate_scattering_factors` | 1e-12 relative | NO (1e-7 intermediate products) | YES |
| `calculate_derived_quantities` | 1e-8 relative | NO (sqrt of 1e-8 values) | YES |
| `calculate_critical_angle` | 1e-4 absolute | MARGINAL | YES |
| `calculate_attenuation_length` | 1e-4 relative | NO (division by 1e-10) | YES |
| `parse_formula` | Exact (integer) | N/A | N/A |
| `PchipInterpolator` | 1e-6 relative | NO (cubic polynomial coefficients) | YES |

**Conclusion:** No function in the calculation pipeline can safely use float32.
