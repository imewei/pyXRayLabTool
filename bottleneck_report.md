# Bottleneck Root Cause Analysis Report

**Date:** 2026-04-06
**Scope:** xraylabtool computational hot paths
**Analyst:** bottleneck-hunter agent

---

## Executive Summary

Three high-impact bottlenecks identified in the main calculation pipeline. The primary hot path flows through `calculate_single_material_properties` -> `_calculate_single_material_xray_properties` -> `calculate_scattering_factors` + `calculate_derived_quantities`. All bottlenecks are in this critical path.

---

## Bottleneck #1: Redundant Array Copies in XRayResult Construction (HIGH IMPACT)

**Location:** `calculators/core.py:1574-1606` (in `calculate_single_material_properties`)

**Root Cause:** Double array conversion. The function calls `np.ascontiguousarray(..., dtype=np.float64)` on every array field when constructing the XRayResult. Then `XRayResult.__post_init__` (line 184-193) calls `np.asarray()` on every field *again*. This means each of the 10 array fields is copied/converted twice per calculation.

**Current Pattern:**
```python
# In calculate_single_material_properties (line 1580-1606):
return XRayResult(
    energy_kev=np.ascontiguousarray(properties["energy"], dtype=np.float64),
    wavelength_angstrom=np.ascontiguousarray(properties["wavelength"], dtype=np.float64),
    # ... 8 more fields, all np.ascontiguousarray
)

# Then in XRayResult.__post_init__ (line 184-193):
self.energy_kev = np.asarray(self.energy_kev)
self.wavelength_angstrom = np.asarray(self.wavelength_angstrom)
# ... 8 more fields, all np.asarray
```

**Why It's Slow:** Each `np.ascontiguousarray` with explicit dtype forces a copy even when the data is already float64 and contiguous (which it usually is, since `ops.zeros` already creates float64 arrays). Then `__post_init__` does another pass. For a 1000-energy-point calculation, this is 20 unnecessary array operations (10 fields x 2 passes).

**Proposed Fix:** 
1. Remove explicit `dtype=np.float64` from `np.ascontiguousarray` calls when the source arrays are already float64.
2. Add a guard in `__post_init__` to skip conversion if the array is already a numpy float64 array.
3. Or better: remove the `np.ascontiguousarray` wrapping entirely from `calculate_single_material_properties` since `__post_init__` already handles conversion.

**Expected Improvement:** 10-20% reduction in per-calculation overhead for large energy arrays. Eliminates ~20 unnecessary array allocations per call.

**Risk:** Low. The arrays are already float64 from the calculation pipeline. Only affects construction path.

---

## Bottleneck #2: MappingProxyType Allocation on Every Atomic Data Lookup (HIGH IMPACT)

**Location:** `data_handling/atomic_cache.py:143-148` (in `get_atomic_data_fast`)

**Root Cause:** Every call to `get_atomic_data_fast()` creates a new `types.MappingProxyType` wrapper around the cached dict, even for preloaded elements. This is an allocation on every lookup in the hot path.

**Current Pattern:**
```python
def get_atomic_data_fast(element: str) -> types.MappingProxyType[str, float]:
    element_key = element.capitalize()
    if element_key in _ATOMIC_DATA_PRELOADED:
        return types.MappingProxyType(_ATOMIC_DATA_PRELOADED[element_key])  # NEW OBJECT EVERY CALL
    if element_key in _RUNTIME_CACHE:
        return types.MappingProxyType(_RUNTIME_CACHE[element_key])  # NEW OBJECT EVERY CALL
```

**Why It's Slow:** `MappingProxyType` is constructed fresh on every call. In the hot path, `get_bulk_atomic_data_fast` calls `get_atomic_data_fast` once per element, and this is called for every material calculation. While each individual allocation is small (~100ns), it adds up in batch processing (e.g., 1000 materials x 3 elements = 3000 proxy allocations).

**Proposed Fix:** Pre-create the `MappingProxyType` wrappers at module load time for the preloaded data:

```python
# At module level, after _ATOMIC_DATA_PRELOADED:
_ATOMIC_DATA_PROXIED = {
    k: types.MappingProxyType(v) for k, v in _ATOMIC_DATA_PRELOADED.items()
}

def get_atomic_data_fast(element: str) -> types.MappingProxyType[str, float]:
    element_key = element.capitalize()
    if element_key in _ATOMIC_DATA_PROXIED:
        return _ATOMIC_DATA_PROXIED[element_key]  # NO ALLOCATION
    ...
```

**Expected Improvement:** Eliminates ~3000 object allocations per 1000-material batch. Modest per-call improvement (~100ns savings per element) but significant in batch mode.

**Risk:** Very low. MappingProxyType is read-only; pre-creating it is safe.

---

## Bottleneck #3: Deprecated Property Access Triggering Warnings in calculate_multiple_xray_properties (CRITICAL BUG)

**Location:** `calculators/core.py:1389-1406` (in `calculate_multiple_xray_properties`)

**Root Cause:** The function accesses every result field through the deprecated CamelCase property aliases (`result.Formula`, `result.MW`, `result.f1`, etc.), each of which emits a `DeprecationWarning` via `warnings.warn()`. This means **every call produces 14 deprecation warnings** per material.

**Current Pattern:**
```python
result_dict = {
    "formula": result.Formula,           # triggers DeprecationWarning
    "molecular_weight": result.MW,        # triggers DeprecationWarning
    "number_of_electrons": result.Number_Of_Electrons,  # triggers DeprecationWarning
    "mass_density": result.Density,       # triggers DeprecationWarning
    # ... 10 more deprecated property accesses
}
```

**Why It's Slow:** `warnings.warn()` is expensive -- it captures the call stack, checks filters, and potentially writes to stderr. 14 warnings per material, times N materials, creates substantial overhead. The `warnings` module also holds the GIL during stack introspection.

**Proposed Fix:** Use the new snake_case field names directly:
```python
result_dict = {
    "formula": result.formula,
    "molecular_weight": result.molecular_weight_g_mol,
    "number_of_electrons": result.total_electrons,
    "mass_density": result.density_g_cm3,
    "electron_density": result.electron_density_per_ang3,
    "energy": result.energy_kev,
    "wavelength": result.wavelength_angstrom,
    "dispersion": result.dispersion_delta,
    "absorption": result.absorption_beta,
    "f1_total": result.scattering_factor_f1,
    "f2_total": result.scattering_factor_f2,
    "critical_angle": result.critical_angle_degrees,
    "attenuation_length": result.attenuation_length_cm,
    "re_sld": result.real_sld_per_ang2,
    "im_sld": result.imaginary_sld_per_ang2,
}
```

**Expected Improvement:** Eliminates 14 `warnings.warn()` calls per material. For batch processing of 100 materials, this removes 1400 stack-capture operations. Likely 5-15% speedup for `calculate_multiple_xray_properties`.

**Risk:** None. This is a pure bugfix -- the function should have been updated when the field names were modernized.

---

## Bottleneck #4: CSV Row-by-Row Parsing in load_scattering_factor_data (MEDIUM IMPACT)

**Location:** `calculators/core.py:554-591` (in `load_scattering_factor_data`)

**Root Cause:** The function reads .nff files using Python's `csv.reader` and builds a list of lists row by row, then converts to numpy. This is O(n) in Python with per-row float() conversions.

**Current Pattern:**
```python
reader = csv.reader(file)
header = next(reader)
data_rows = []
for row in reader:
    if len(row) >= max(e_idx, f1_idx, f2_idx) + 1:
        data_rows.append([float(row[e_idx]), float(row[f1_idx]), float(row[f2_idx])])
data_array = np.array(data_rows, dtype=np.float64)
```

**Why It's Slow:** Python-level loop with per-element float conversion. `np.loadtxt` or `np.genfromtxt` would be faster for well-formatted numeric CSV files, and the data is already cached so this only affects cold start.

**Proposed Fix:**
```python
data_array = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=(e_idx, f1_idx, f2_idx), dtype=np.float64)
```

**Expected Improvement:** 2-5x faster file loading on cold start. Since results are cached, this only affects the first load per element.

**Risk:** Low. `np.loadtxt` is well-tested for CSV. Need to verify .nff file format compatibility.

---

## Bottleneck #5: Unnecessary Energy Sorting in calculate_xray_properties (LOW-MEDIUM IMPACT)

**Location:** `calculators/core.py:1877-1878` (in `calculate_xray_properties`)

**Root Cause:** The function always sorts energies and later reverses the sort, even when energies are already sorted (which is the common case for `np.linspace` inputs).

**Current Pattern:**
```python
sort_indices = np.argsort(energies_array)
sorted_energies = energies_array[sort_indices]
# ... later in _create_process_formula_function:
if not np.array_equal(sort_indices, np.arange(len(sort_indices))):
    reverse_indices = np.argsort(sort_indices)
    result = _restore_energy_order(result, reverse_indices)
```

**Why It's Slow:** `np.argsort` + array indexing + `np.array_equal` check + potential `np.argsort` reversal + 10-field re-indexing in `_restore_energy_order`. For already-sorted arrays (common case), this is pure waste.

**Proposed Fix:** Check if sorted first:
```python
is_sorted = np.all(energies_array[:-1] <= energies_array[1:])
if is_sorted:
    sorted_energies = energies_array
    sort_indices = None
else:
    sort_indices = np.argsort(energies_array)
    sorted_energies = energies_array[sort_indices]
```

**Expected Improvement:** Eliminates sorting overhead for the common case. Saves ~2 argsort + 10 array indexing operations per material.

**Risk:** Very low. Simple optimization of the common path.

---

## Bottleneck #6: ThreadPoolExecutor for CPU-Bound Work in batch_processing.py (MEDIUM IMPACT)

**Location:** `data_handling/batch_processing.py:173-176` (in `process_batch_chunk`)

**Root Cause:** Uses `ThreadPoolExecutor` for X-ray calculations which are CPU-bound (numpy operations). The GIL prevents true parallelism for CPU-bound Python code.

**Current Pattern:**
```python
with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
    # Submit CPU-bound calculations
```

The code comments acknowledge this: "ThreadPoolExecutor for I/O bound operations (file loading) / ProcessPoolExecutor would be better for CPU-bound, but has pickle overhead"

**Why It's Slow:** ThreadPoolExecutor provides zero parallelism for numpy operations that hold the GIL. The thread creation/management overhead is pure waste for CPU-bound work.

**Proposed Fix:** For batch processing, process sequentially in the main thread (eliminating thread overhead) or use ProcessPoolExecutor with careful serialization. Since the calculations are already fast individually (cached data + vectorized numpy), sequential processing with cache warming would outperform threaded processing.

**Expected Improvement:** 5-10% improvement in batch processing by eliminating thread overhead.

**Risk:** Medium. Need to verify that no I/O operations within the calculation release the GIL.

---

## Summary: Top 3 Bottlenecks by Impact

| Rank | Bottleneck | Location | Impact | Fix Effort |
|------|-----------|----------|--------|------------|
| 1 | Deprecated property warnings in multi-material calc | core.py:1389-1406 | Critical (14 warnings/material) | Trivial (rename fields) |
| 2 | Double array conversion in XRayResult | core.py:1574-1606 + 184-193 | High (20 copies/calc) | Low (remove redundant pass) |
| 3 | MappingProxyType allocation per lookup | atomic_cache.py:143-148 | High (in batch) | Low (pre-create proxies) |
