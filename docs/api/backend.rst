Backend
=======

The backend package holds the JAX runtime helpers, the shape buckets that bound
compile counts, and the in-tree PCHIP interpolator. XRayLabTool runs on JAX only
(ADR-006); ``set_backend("jax")`` is a no-op, ``set_backend("numpy")`` raises
``ValueError``, and ``get_backend()`` returns ``"jax"``.

.. automodule:: xraylabtool.backend
   :members:
   :undoc-members:
   :show-inheritance:

Runtime
-------

.. automodule:: xraylabtool.backend.runtime
   :members:
   :undoc-members:
   :show-inheritance:

Shapes
------

.. automodule:: xraylabtool.backend.shapes
   :members:
   :undoc-members:
   :show-inheritance:

Interpolation
-------------

.. automodule:: xraylabtool.backend.interpolation
   :members:
   :undoc-members:
   :show-inheritance:
