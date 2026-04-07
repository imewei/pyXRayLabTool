Rollback: v0.4.0 → v0.3.0
==========================

If v0.4.0 causes issues, revert to the NumPy-based v0.3.0.


Quick Rollback
--------------

.. code-block:: bash

   # 1. Uninstall v0.4.0 and JAX dependencies
   pip uninstall -y xraylabtool jax jaxlib pyqtgraph

   # 2. Install v0.3.0
   pip install xraylabtool==0.3.0

   # 3. Verify
   python -c "import xraylabtool; print(xraylabtool.__version__)"
   python -c "import xraylabtool as xrt; \
       r = xrt.calculate_single_material_properties('Si', 2.33, 8000); \
       print(f'Critical angle: {r.critical_angle_degrees}')"

All data and CSV files are compatible between versions. No data migration
is needed in either direction.


When to Roll Back
-----------------

- Calculation results differ from v0.3.0 by more than floating-point
  tolerance (~1e-12 relative)
- GUI fails to launch after PyQtGraph installation
- JAX import errors that ``pip install --upgrade xraylabtool`` does not fix


Pre-Upgrade Backup (Optional)
------------------------------

.. code-block:: bash

   pip freeze > requirements_v0_3_0.txt


Re-Upgrading After a Fix
-------------------------

After the issue is resolved in a patch release:

.. code-block:: bash

   pip install --upgrade xraylabtool>=0.4.1

Verify with the same checks above before resuming use.

---

**See also:** `Migration Guide <../guides/migration_guide_v0_4.rst>`_ for the full
upgrade procedure.
