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
