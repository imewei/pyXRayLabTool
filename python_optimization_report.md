# Python-Level Optimization Report — pyXRayLabTool

**Analyst:** python-optimizer agent  
**Date:** 2026-04-06  
**Scope:** Python-level performance; files analyzed: `calculators/core.py`, `data_handling/atomic_cache.py`, `data_handling/batch_processing.py`, `data_handling/compound_analysis.py`, `utils.py`, `constants.py`

---

## Executive Summary

The codebase is reasonably well-structured with existing caching layers. The main performance bottlenecks are: (1) repeated `np.ascontiguousarray` calls on arrays already contiguous from prior ops, (2) row-by-row CSV parsing via Python's `csv` module instead of `np.loadtxt`, (3) redundant `np.asarray()` wrapping in the hot path of `calculate_derived_quantities`, and (4) deprecated property aliases in `calculate_multiple_xray_properties` triggering DeprecationWarning machinery on every call. Secondary issues include unnecessary `lru_cache(maxsize=1)` wrappers around constants, double-lookup in module-level `__getattr__`, and `find_similar_compounds` doing O(n²) repeated `get_elements_for_compound` calls.

---

## Findings

### P1 — HIGH IMPACT

---

#### OPT-1: Replace `csv.reader` loop with `np.loadtxt` in `.nff` file parsing

**File:** `xraylabtool/calculators/core.py:549–613`  
**Current pattern:**
```python
import csv
with open(file_path) as file:
    reader = csv.reader(file)
    header = next(reader)
    ...
    data_rows = []
    for row in reader:
        if len(row) >= ...:
            data_rows.append([float(row[e_idx]), float(row[f1_idx]), float(row[f2_idx])])
data_array = np.array(data_rows, dtype=np.float64)
```
**Problem:** Pure-Python `csv.reader` loop + Python-level `float()` conversion + intermediate `data_rows` list + final `np.array()` call. For a typical .nff file (~500 rows), this creates ~1500 Python float objects and a list of lists before building the array.

**Proposed pattern:**
```python
data_array = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=(0, 1, 2), dtype=np.float64)
```
`np.loadtxt` (or preferably `np.genfromtxt` for robustness) reads directly into a contiguous C array without intermediate Python objects. For files with arbitrary column order, read the header separately first.

**Expected speedup:** 3–8× on first load per element (cold cache). Warm cache is unaffected.  
**Implementation effort:** Low (10–15 lines changed in one function).  
**Priority:** High. Every cold-start element load goes through this path.

---

#### OPT-2: Eliminate redundant `np.ascontiguousarray` calls in `calculate_single_material_properties`

**File:** `xraylabtool/calculators/core.py:1580–1605`  
**Current pattern:**
```python
energy_kev=np.ascontiguousarray(properties["energy"], dtype=np.float64),
wavelength_angstrom=np.ascontiguousarray(properties["wavelength"], dtype=np.float64),
# ... 8 more identical calls
```
Plus `__post_init__` at lines 184–193 calls `np.asarray()` on every field again:
```python
self.energy_kev = np.asarray(self.energy_kev)
```
**Problem:** Each result construction performs 10 `np.ascontiguousarray` calls and then 10 more `np.asarray` calls in `__post_init__`. For arrays already produced as float64 C-contiguous numpy arrays (which they are, since `ops` produces them that way), both calls are no-ops that still pay Python dispatch overhead and a C-level contiguity check.

**Proposed pattern:**  
- Remove the `np.asarray()` calls from `__post_init__` — they are redundant since construction already passes numpy arrays.  
- In `calculate_single_material_properties`, verify upstream that arrays are already float64 contiguous (they are) and drop the `np.ascontiguousarray` wrappers, or consolidate to a single `np.asarray(..., dtype=np.float64)` only where the type is actually uncertain (the `energy` field from backward-compat eV path at line 1556).

**Expected speedup:** ~5–15% on the final result-construction step per call; measurable for tight loops computing many materials.  
**Implementation effort:** Low (delete or simplify ~20 lines).  
**Priority:** High for batch workloads.

---

#### OPT-3: Eliminate deprecated-property access in `calculate_multiple_xray_properties`

**File:** `xraylabtool/calculators/core.py:1390–1406`  
**Current pattern:**
```python
result_dict = {
    "formula": result.Formula,        # triggers DeprecationWarning
    "molecular_weight": result.MW,     # triggers DeprecationWarning
    "number_of_electrons": result.Number_Of_Electrons,  # triggers DeprecationWarning
    ...
}
```
**Problem:** Each of the 14 dictionary fields accesses a deprecated property alias, which calls `warnings.warn()` with `stacklevel=2`. Even when warnings are filtered, `warnings.warn` still checks the registry and does stack inspection. With N materials × 14 properties, this adds ~14N unnecessary `warnings.warn` calls.

**Proposed pattern:**
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
**Expected speedup:** Eliminates 14 `warnings.warn` calls per material; for 100-material batches, removes 1400 redundant Python calls.  
**Implementation effort:** Trivial (field name substitution).  
**Priority:** High — simple fix with zero risk.

---

### P2 — MEDIUM IMPACT

---

#### OPT-4: Remove redundant scalar-extraction boilerplate in `calculate_derived_quantities`

**File:** `xraylabtool/calculators/core.py:1044–1063`  
**Current pattern:**
```python
density_val = (
    np.asarray(mass_density).item()
    if np.asarray(mass_density).size == 1
    else mass_density
)
# same pattern × 3 for mol_weight_val, electrons_val
```
**Problem:** Creates 2 temporary numpy scalars per scalar input (one for `.size` check, one for `.item()`). Since `mass_density`, `molecular_weight`, and `number_of_electrons` are always plain Python `float` at this call site (checked at `core.py:1292–1293`), the entire guard is unnecessary.

**Proposed pattern:**
```python
electron_density = float(
    1e6 * mass_density / molecular_weight * AVOGADRO * number_of_electrons / 1e30
)
```
**Expected speedup:** Eliminates 6 temporary array allocations per call.  
**Implementation effort:** Low.  
**Priority:** Medium.

---

#### OPT-5: Cache-miss path in `get_atomic_data_fast` still calls `get_atomic_number` + `get_atomic_weight` separately (2 Mendeleev queries instead of 1)

**File:** `xraylabtool/data_handling/atomic_cache.py:151–161`  
**Current pattern:**
```python
atomic_data = {
    "atomic_number": get_atomic_number(element),    # opens mendeleev, queries DB
    "atomic_weight": get_atomic_weight(element),    # opens mendeleev, queries DB again
}
```
**Problem:** Two separate `mendeleev.element()` calls for a cache miss. The mendeleev call opens a SQLite DB each time.

**Proposed pattern:**
```python
from mendeleev import element as _get_element
elem = _get_element(element)
atomic_data = {
    "atomic_number": int(elem.atomic_number),
    "atomic_weight": float(elem.atomic_weight),
}
```
**Expected speedup:** Halves Mendeleev DB queries on cache miss. Negligible after warm-up (preloaded table covers all 92 elements).  
**Implementation effort:** Low.  
**Priority:** Medium (only affects unknown elements beyond the 92 preloaded).

---

#### OPT-6: `find_similar_compounds` calls `get_elements_for_compound` O(n) times inside a loop, each potentially parsing a formula

**File:** `xraylabtool/data_handling/compound_analysis.py:295–313`  
**Current pattern:**
```python
for compound in COMMON_COMPOUNDS:
    compound_elements = set(get_elements_for_compound(compound))
    ...
similar_compounds.sort(
    key=lambda x: len(set(get_elements_for_compound(x)) & target_elements),
    ...
)
```
**Problem:** `get_elements_for_compound` is called once per compound in the loop (checked against `COMMON_COMPOUNDS` dict which short-circuits to `.keys()`), then called again in the sort key for each similar compound. No caching. With `COMMON_COMPOUNDS` having ~45 entries, this executes ~45 + len(similar_compounds) dict lookups and `set()` constructions every call.

**Proposed pattern:** Pre-compute `_COMPOUND_ELEMENTS: dict[str, frozenset[str]]` at module level:
```python
_COMPOUND_ELEMENTS: dict[str, frozenset[str]] = {
    formula: frozenset(elements.keys())
    for formula, elements in COMMON_COMPOUNDS.items()
}
```
Then use `_COMPOUND_ELEMENTS[compound]` in the loop. Eliminates all runtime parsing for known compounds.

**Expected speedup:** 2–5× for `find_similar_compounds` calls; minor absolute impact since this only runs during cache warming.  
**Implementation effort:** Low.  
**Priority:** Medium.

---

### P3 — LOW IMPACT / STARTUP

---

#### OPT-7: `utils.py` wraps already-constant values in `lru_cache(maxsize=1)` functions

**File:** `xraylabtool/utils.py:32–74`  
**Current pattern:**
```python
@lru_cache(maxsize=1)
def get_planck_constant() -> float:
    return float(PLANCK)

@lru_cache(maxsize=1)
def get_speed_of_light() -> float:
    return float(_SPEED_OF_LIGHT)
```
**Problem:** These are pure constant accessors that already access module-level `Final[float]` constants. The `lru_cache` machinery adds a dict lookup and lock check on every call, providing no benefit since `float(PLANCK)` is O(1) and deterministic.

**Proposed pattern:** Remove the functions entirely and import `PLANCK`, `SPEED_OF_LIGHT`, etc. directly from `constants.py` wherever they are needed, or use the already-computed `ENERGY_TO_WAVELENGTH_FACTOR`. The `_NumpyProxy` class (lines 41–50) is similarly over-engineered — numpy is already a direct dependency; lazy import is irrelevant at module level.

**Expected speedup:** Minimal wall-clock, but reduces import-time object allocation and removes indirection noise from profiler traces.  
**Implementation effort:** Low (but touches public API, so needs care if these are exported).  
**Priority:** Low.

---

#### OPT-8: `_constants_cache` dict in `utils.py` is redundant with the `lru_cache` on the accessor functions

**File:** `xraylabtool/utils.py:78–99`  
**Current pattern:**
```python
_constants_cache = {}
def __getattr__(name: str):
    if name == "PLANCK_CONSTANT":
        if name not in _constants_cache:
            _constants_cache[name] = get_planck_constant()
        return _constants_cache[name]
```
**Problem:** Double indirection — `__getattr__` checks a dict, then calls an `lru_cache` function. Two cache lookups for one constant.

**Proposed pattern:** Remove `_constants_cache` and the `__getattr__` hook; export the constants directly.  
**Expected speedup:** Negligible individually.  
**Priority:** Low (cleanup).

---

#### OPT-9: `_NumpyProxy` dynamic attribute dispatch in `utils.py`

**File:** `xraylabtool/utils.py:41–50`  
**Current pattern:**
```python
class _NumpyProxy:
    def __getattr__(self, name: str) -> Any:
        np = _get_numpy()
        return getattr(np, name)
np = _NumpyProxy()
```
**Problem:** Every numpy call in `utils.py` (e.g. `np.arcsin`, `np.sqrt`) goes through `_NumpyProxy.__getattr__` → `_get_numpy()` → `lru_cache` check → `getattr(real_np, name)`. This is 2–3× slower than a direct `real_np.arcsin` call.

**Proposed pattern:** Since numpy is a hard dependency and already imported in `core.py`, simply `import numpy as np` at the top of `utils.py`. Lazy import is not justified when numpy is used on every call to any util function.

**Expected speedup:** 2–3× on individual numpy attribute lookups in utils. Absolute magnitude is small since utils functions are not on the hot path.  
**Implementation effort:** Low.  
**Priority:** Low.

---

## Compiled Extension Assessment

| Candidate | Justification | Verdict |
|-----------|---------------|---------|
| `load_scattering_factor_data` (csv loop) | Pure-Python loop with float conversion | Unnecessary — `np.loadtxt` is C-level and sufficient (OPT-1) |
| `parse_formula` (regex) | Python `re.findall` — already fast; called infrequently | Not worth Cython/Rust |
| `calculate_scattering_factors` | Already vectorized via numpy/JAX matrix ops | Not worth Cython; JAX JIT is the right path |
| `calculate_derived_quantities` | Vectorized numpy ops | JAX JIT already applicable |
| `parse_chemical_formula` in compound_analysis | Python string manipulation with re | Not worth compiled extension; call frequency is low |

**Conclusion:** No compiled extension (Cython, mypyc, PyO3) is justified at this time. The dominant hot paths are already in numpy/JAX C-level operations. The remaining pure-Python overhead (CSV parsing, warning machinery, proxy dispatch) can be resolved with standard Python refactoring as described above.

---

## Priority Ranking

| Rank | ID | Description | Effort | Impact |
|------|----|-------------|--------|--------|
| 1 | OPT-1 | Replace `csv.reader` loop with `np.loadtxt` in `.nff` parser | Low | High (3–8× cold load) |
| 2 | OPT-3 | Use canonical field names in `calculate_multiple_xray_properties` | Trivial | High (eliminates 14N warn calls per batch) |
| 3 | OPT-2 | Remove redundant `np.ascontiguousarray`/`np.asarray` on result construction | Low | High for batch (5–15% result-construction overhead) |
| 4 | OPT-4 | Remove scalar-extraction boilerplate in `calculate_derived_quantities` | Low | Medium |
| 5 | OPT-6 | Pre-compute `_COMPOUND_ELEMENTS` in compound_analysis | Low | Medium (only during cache warm) |
| 6 | OPT-5 | Single Mendeleev query on cache miss | Low | Low (only for elements beyond the 92 preloaded) |
| 7 | OPT-7/8/9 | Remove `_NumpyProxy`, `lru_cache` wrappers, `_constants_cache` in utils | Low | Low (startup/trace cleanup) |

---

## Top 3 Highest-Impact Recommendations (for team-lead)

1. **OPT-1 — `np.loadtxt` for .nff parsing** (`core.py:549`): Replaces a ~600-line Python CSV loop + list-of-lists construction with a single C-level array read. 3–8× speedup on cold element loads; every first-time calculation of any compound touches this path.

2. **OPT-3 — Remove deprecated-property access in batch path** (`core.py:1390`): One-line field name substitution eliminates 14 `warnings.warn` calls per material in `calculate_multiple_xray_properties`. Zero risk change; measurable for any batch of more than a few dozen materials.

3. **OPT-2 — Remove double-wrapping on `XRayResult` construction** (`core.py:1574–1606` + `__post_init__:184`): Eliminates 20 redundant `np.ascontiguousarray`/`np.asarray` calls per result object. These arrays are already float64 contiguous numpy arrays — the wrapping is pure overhead.
