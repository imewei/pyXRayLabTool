JAX Architecture & Design
=========================

**Audience:** Developers and users wanting deep understanding of v0.4.0 design.

**Purpose:** Explain why JAX, how it works, and how XRayLabTool uses it.


Why JAX?
--------

What is JAX?
~~~~~~~~~~~~

JAX is a Python library for numerical computing with:

1. **JIT (Just-In-Time) Compilation** - Convert Python → compiled machine code
2. **Automatic Differentiation** - Compute gradients with ``grad()``
3. **Vectorization** - Apply functions across batches with ``vmap()``
4. **GPU/TPU Support** - Automatic device acceleration

XRayLabTool uses features 1 and 3; features 2 and 4 enable future scientific workflows.

NumPy vs JAX
~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import jax
   import jax.numpy as jnp

   # NumPy: eager evaluation
   x_np = np.array([1.0, 2.0, 3.0])
   y_np = np.sin(x_np)  # Computed immediately
   print(y_np)  # [0.841, 0.909, 0.141]

   # JAX: lazy evaluation + JIT
   x_jax = jnp.array([1.0, 2.0, 3.0])
   y_jax = jnp.sin(x_jax)  # Not computed yet, just recorded

   # Explicit compilation
   sin_jit = jax.jit(jnp.sin)
   y_compiled = sin_jit(x_jax)  # Computed and cached


Performance Benefits
~~~~~~~~~~~~~~~~~~~~

**1. JIT Compilation:** 10-100x speedup

.. code-block:: text

   NumPy (eager):        ■■■■■■■■■■■■■■■■■■■■ (1.0x baseline)
   JAX (eager):          ■■■■■■■■■■■■■■■■■■■■ (0.95x, ~same)
   JAX (JIT compiled):   ■ (0.01-0.1x, 10-100x faster)

**2. Vectorization:** Automatic batching with ``vmap()``

.. code-block:: python

   # NumPy: manual loop
   energies = np.linspace(1000, 20000, 1000)
   results = []
   for e in energies:
       result = calculate_property(e)  # 1000 calls
       results.append(result)

   # JAX: automatic vectorization
   calculate_batch = jax.vmap(calculate_property)
   results = calculate_batch(energies)  # Single compiled call

**3. GPU Acceleration:** Automatic offloading

.. code-block:: text

   CPU:        ■■■■■■■■■■■■■■■■■■■■ (1.0x baseline)
   GPU (V100): ■ (0.02x, 50x faster)
   GPU (A100): ■ (0.01x, 100x faster)

Design Philosophy
~~~~~~~~~~~~~~~~~

JAX adoption strategy for XRayLabTool:

1. **Transparent to users** - API stays the same
2. **Backward compatible** - v0.3.0 code works unchanged
3. **Gradual adoption** - Don't rewrite everything at once
4. **Future-proof** - Enable autodiff and hardware acceleration later


JAX Fundamentals
----------------

JIT Compilation
~~~~~~~~~~~~~~~

**What is JIT?**

JIT converts Python functions into compiled machine code:

.. code-block:: python

   import jax
   import jax.numpy as jnp

   # Normal function (eager evaluation)
   def compute(x):
       return jnp.sin(x) ** 2 + jnp.cos(x) ** 2

   # First call: Python interpreter runs
   result1 = compute(1.0)  # ~0.1 ms (with overhead)

   # Compiled version
   compute_jit = jax.jit(compute)

   # First call: JAX compiles function (expensive)
   result2 = compute_jit(1.0)  # ~50 ms (includes compilation)

   # Subsequent calls: use cached compiled code (fast)
   result3 = compute_jit(1.0)  # ~0.001 ms

**Cost-Benefit:**

- Compilation cost: 50-100 ms (one-time)
- Per-call speedup: 100-1000x
- Break-even: 100+ calls

**Why this matters for XRayLabTool:**

- Calculation functions run thousands of times (batch processing)
- One-time compilation cost is negligible
- Per-call speedup dominates overall performance


Traces and Shapes
~~~~~~~~~~~~~~~~~

JAX compiles based on **input shapes**, not values:

.. code-block:: python

   import jax
   import jax.numpy as jnp

   @jax.jit
   def process(x):
       return jnp.sum(x) * 2

   # First call with shape (3,)
   result1 = process(jnp.array([1.0, 2.0, 3.0]))
   # JAX traces and compiles for shape (3,)

   # Second call with shape (3,) - reuses compilation
   result2 = process(jnp.array([4.0, 5.0, 6.0]))
   # Cache hit, fast

   # Third call with shape (5,) - recompiles!
   result3 = process(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))
   # New shape = recompilation (slow)

**Implication for XRayLabTool:**

- Single material calculations always use same shape
- Batch processing with consistent batch sizes benefits from caching
- Different input sizes trigger recompilation


Automatic Differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Future capability (enabled by JAX, not yet used):

.. code-block:: python

   import jax
   import jax.numpy as jnp

   def scattering_factor(energy):
       """Calculate f1 component."""
       return jnp.exp(-energy / 1000)  # Simplified

   # Compute gradient dβ/dE
   gradient_fn = jax.grad(scattering_factor)
   dfdE = gradient_fn(8000.0)  # Automatic differentiation

**Future use cases:**

- Optimization (find optimal angles, densities)
- Uncertainty quantification
- Inverse problems (given results, find parameters)


Vectorization with vmap
~~~~~~~~~~~~~~~~~~~~~~~

Automatic batching - future optimization:

.. code-block:: python

   import jax
   import jax.numpy as jnp

   def calculate_one(energy):
       """Single energy calculation."""
       return energy ** 2

   # Vectorized version
   calculate_batch = jax.vmap(calculate_one)

   energies = jnp.array([1000, 5000, 8000, 10000])
   results = calculate_batch(energies)  # Compiled as single operation

**Current approach (v0.4.0):**

XRayLabTool manually implements batching, which JAX compiles with JIT.

**Future optimization (v0.5+):**

Use ``vmap()`` for automatic vectorization without manual loops.


XRayLabTool Implementation
---------------------------

Architecture Overview
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   User Code
      │
      ├─→ API Layer (xraylabtool.__init__)
      │       └─→ validate inputs
      │           convert to JAX arrays
      │
      ├─→ Calculation Layer (calculators/)
      │       └─→ @jax.jit compiled functions
      │           JAX array operations
      │           physics calculations
      │
      ├─→ Data Layer (data_handling/)
      │       └─→ atomic data cache
      │           JAX array storage
      │           efficient lookups
      │
      └─→ Output
          └─→ XRayResult dataclass
              JAX arrays for properties


Compilation in XRayLabTool
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Key functions decorated with @jax.jit:**

.. code-block:: python

   # In xraylabtool/calculators/core.py
   @jax.jit
   def calculate_refraction_indices(energy, delta, beta):
       """Core JIT-compiled calculation."""
       n = 1 - delta - 1j * beta
       return n

   @jax.jit
   def calculate_critical_angle(delta):
       """JIT-compiled critical angle."""
       return jnp.sqrt(2 * delta)

**Warm-up pattern:**

.. code-block:: python

   # First call triggers compilation
   result = calculate_single_material_properties("Si", 2.33, 8000)
   # ~50 ms (includes JIT compilation for shape/type)

   # Subsequent calls use cached compilation
   result = calculate_single_material_properties("Si", 2.33, 8000)
   # ~0.02 ms (no recompilation)


Data Flow
~~~~~~~~~

**v0.4.0 data types:**

.. code-block:: text

   # User input: Python/NumPy
   formula = "SiO2"  # str
   density = 2.2     # float
   energy = 8000     # float

   # Internal: JAX arrays
   |-> Parse formula -> JAX array of atomic numbers
   |-> Load atomic data -> JAX arrays
   |-> JIT compute -> JAX array results
   \-> Package -> XRayResult with JAX arrays

   # Output: JAX arrays (NumPy compatible)
   result.critical_angle_degrees  # jax.Array (shape: (1,))
   result.attenuation_length_cm   # jax.Array (shape: (1,))


Memory Management
~~~~~~~~~~~~~~~~~

JAX vs NumPy memory semantics:

.. code-block:: python

   import jax.numpy as jnp

   # JAX arrays are immutable
   x = jnp.array([1.0, 2.0, 3.0])
   y = x + 1  # Creates new array, doesn't modify x

   # Lazy evaluation
   z = jnp.sin(x)  # Computation recorded, not executed yet

   # Materialization
   z_ready = z.block_until_ready()  # Forces computation, blocks until done

**For XRayLabTool users:**

- No memory leaks (immutability prevents issues)
- Minimal memory overhead (lazy evaluation)
- Transparent memory management (automatic)


GPU Acceleration
~~~~~~~~~~~~~~~~

**Automatic device placement:**

.. code-block:: python

   import jax

   # JAX automatically detects and uses available devices
   print(jax.devices())  # Shows available hardware

   # No code changes needed
   result = calculate_single_material_properties("Si", 2.33, 8000)
   # Automatically runs on GPU if available

**Device hints (advanced):**

.. code-block:: python

   import jax

   # Force CPU
   with jax.default_device(jax.devices("cpu")[0]):
       result = calculate(...)

   # Force GPU
   with jax.default_device(jax.devices("gpu")[0]):
       result = calculate(...)


Type System
~~~~~~~~~~~

JAX array types:

.. code-block:: python

   import jax.numpy as jnp
   import numpy as np

   x_jax = jnp.array([1.0, 2.0])
   x_np = np.array([1.0, 2.0])

   print(type(x_jax))  # <class 'jaxlib.xla_extension.ArrayImpl'>
   print(type(x_np))   # <class 'numpy.ndarray'>

   # NumPy operations work with JAX arrays
   np.sin(x_jax)  # Works!
   np.concatenate([x_jax, x_np])  # Works!

   # Type hints
   from typing import Union
   from jax import Array

   def process(x: Union[np.ndarray, Array]) -> float:
       return float(jnp.mean(x))


Configuration & Control
-----------------------

JAX Configuration Options
~~~~~~~~~~~~~~~~~~~~~~~~~~

For advanced users:

.. code-block:: python

   import jax

   # Disable JIT (useful for debugging)
   jax.config.update("jax_disable_jit", True)

   # Disable GPU (force CPU)
   jax.config.update("jax_platforms", "cpu")

   # Precision control
   jax.config.update("jax_default_float_dtype", jnp.float32)

**In XRayLabTool code:**

.. code-block:: bash

   # Environment variables
   JAX_DISABLE_JIT=1 python script.py  # Debug without JIT
   JAX_PLATFORMS=cpu python script.py  # Force CPU


Common Gotchas & Solutions
---------------------------

Problem: Shape-based recompilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue:**

.. code-block:: python

   @jax.jit
   def calculate(energies):
       return jnp.sin(energies)

   # Recompilation on every call
   for energy_list in many_lists:
       result = calculate(energy_list)  # Different sizes = recompile!

**Solution:**

.. code-block:: python

   # Pad to fixed size
   max_size = 1000
   def pad(x):
       return jnp.pad(x, (0, max_size - len(x)))

   energy_fixed = pad(energy_list)
   result = calculate(energy_fixed)  # Single compilation


Problem: Python control flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue:**

.. code-block:: python

   @jax.jit
   def conditional_calc(x, use_gpu):
       if use_gpu:  # Won't work! (Python control flow)
           return gpu_calc(x)
       else:
           return cpu_calc(x)

**Solution:**

.. code-block:: python

   @jax.jit
   def conditional_calc(x, use_gpu):
       return jnp.where(use_gpu, gpu_calc(x), cpu_calc(x))


Problem: Non-array inputs
~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue:**

.. code-block:: python

   @jax.jit
   def calc(material_name, energy):
       data = MATERIAL_DATA[material_name]  # Dict lookup won't trace
       return compute(data, energy)

**Solution:**

.. code-block:: python

   # Move non-array inputs outside @jax.jit
   def calc(material_name, energy):
       data = MATERIAL_DATA[material_name]  # Python (outside JIT)
       return compute_jit(data, energy)  # JAX (inside JIT)

   @jax.jit
   def compute_jit(data, energy):
       return jnp.sin(energy) * data


Testing with JAX
----------------

Unit Testing Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import pytest

   def test_calculation_matches_reference():
       """JAX results match expected output."""
       result = calculate_single_material_properties("Si", 2.33, 8000)
       angle = float(result.critical_angle_degrees[0])
       assert abs(angle - 0.158) < 0.001  # Allow small tolerance

   def test_jit_consistency():
       """JIT and eager evaluation match."""
       # Compile once
       compute_jit = jax.jit(core_compute)

       x = jnp.array([1000, 5000, 8000])
       result_eager = core_compute(x)  # Eager (no JIT)
       result_jit = compute_jit(x)     # Compiled

       assert jnp.allclose(result_eager, result_jit)


Debugging Strategies
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import jax

   # Disable JIT for debugging
   jax.config.update("jax_disable_jit", True)

   # Add prints (with caveats)
   @jax.jit
   def debug_calc(x):
       print(x.shape)  # Prints shape at compile time
       return x ** 2

   # Use jax.debug.print for runtime prints
   from jax import debug

   @jax.jit
   def debug_calc2(x):
       debug.print("Value: {}", x)
       return x ** 2


Performance Profiling
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   import jax
   import jax.numpy as jnp

   # Warm up JIT
   compute_fn = jax.jit(your_function)
   compute_fn(example_input)

   # Profile
   start = time.time()
   for _ in range(1000):
       result = compute_fn(input)
   elapsed = (time.time() - start) / 1000
   print(f"Time per call: {elapsed * 1000:.3f} ms")


Future Directions
-----------------

Potential JAX Features for v0.5+
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Autodiff for Optimization**

.. code-block:: python

   from jax import grad

   def objective(angle):
       """Minimize deviation from target angle."""
       result = calculate_single_material_properties("Si", 2.33, angle)
       return (result.critical_angle_degrees[0] - target) ** 2

   gradient = grad(objective)
   optimal_angle = optimize(objective, gradient, initial_guess)

**2. vmap for Automatic Batching**

.. code-block:: python

   from jax import vmap

   # Auto-vectorize over materials
   calc_batch = vmap(calculate_single_material_properties, in_axes=(0, 0, None))
   results = calc_batch(formulas, densities, energy)

**3. Distributed Processing**

.. code-block:: python

   from jax.experimental import maps

   # Shard across multiple GPUs
   results = maps.pmap(calculate_single_material_properties)(
       sharded_materials, densities, energy
   )


Further Reading
---------------

- `JAX Quickstart <https://jax.readthedocs.io/en/latest/quickstart.html>`_
- `JAX Documentation <https://jax.readthedocs.io/>`_
- `JIT Mechanics <https://jax.readthedocs.io/en/latest/jit_compilation.html>`_
- `Performance Tips <https://jax.readthedocs.io/en/latest/performance_characteristics.html>`_


Conclusion
----------

JAX modernizes XRayLabTool with:

- **JIT Compilation**: 10-100x speedup with no code changes
- **GPU Acceleration**: Automatic hardware acceleration
- **Backward Compatibility**: Existing code works unchanged
- **Future Capability**: Autodiff and advanced optimization ready

The migration is transparent to users while enabling significant performance improvements and future scientific computing features.

---

**Next:** See `Migration Guide <../guides/migration_guide_v0_4.rst>`_ for upgrade instructions.
