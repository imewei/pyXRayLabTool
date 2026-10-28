"""``xraylabtool.backend`` resolves lazily after a bare ``import xraylabtool``.

The package init no longer imports ``backend`` as a side effect (it used to, via
the float64 configuration), so ``backend`` is a lazy submodule.
"""

from __future__ import annotations

import subprocess
import sys

_PROBE = """
import sys
import xraylabtool

assert "jax" not in sys.modules, "bare import leaked jax"
assert callable(xraylabtool.backend.set_backend)
assert callable(xraylabtool.backend.get_backend)
print("OK")
"""


def test_backend_lazy_attribute_resolves_after_bare_import() -> None:
    done = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == "OK"


_OPTICS_PROBE = """
import sys
import xraylabtool.calculators as calculators

assert "jax" not in sys.modules, "importing the calculators leaked jax"
from xraylabtool.calculators import optics

for name in ("OpticalProps", "ScatteringTables", "encode_formula", "load_tables",
             "optical_properties"):
    assert getattr(calculators, name) is getattr(optics, name), name
assert "jax" in sys.modules
try:
    calculators.no_such_name
except AttributeError:
    print("OK")
"""


def test_calculators_reexport_the_optics_api_lazily() -> None:
    done = subprocess.run(
        [sys.executable, "-c", _OPTICS_PROBE],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == "OK"
