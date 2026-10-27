"""``backend.runtime`` / ``backend.shapes``: generic helpers promoted out of ``kk``.

The two modules must have no import-time side effect (no ``ensure_x64()``, no
compile-cache enabling, no device query), ``backend/__init__`` must import
neither, and the transfer counters live in one module that ``kk._device``
merely re-exports (spec section 3, plan step 1.1).
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_PROBE = """
import sys

import jax

import xraylabtool.backend  # package init only (it may import jax and touch x64)

assert "xraylabtool.backend.runtime" not in sys.modules, "backend/__init__ imports runtime"
assert "xraylabtool.backend.shapes" not in sys.modules, "backend/__init__ imports shapes"

jax.config.update("jax_enable_x64", False)  # make the x64 check non-vacuous
from jax._src import xla_bridge

before = (
    jax.config.jax_enable_x64,
    jax.config.jax_compilation_cache_dir,
    xla_bridge.backends_are_initialized(),
)

import xraylabtool.backend.runtime  # noqa: F401
import xraylabtool.backend.shapes  # noqa: F401

after = (
    jax.config.jax_enable_x64,
    jax.config.jax_compilation_cache_dir,
    xla_bridge.backends_are_initialized(),
)
assert after == before, (before, after)
assert after[0] is False, "importing runtime/shapes enabled x64"
assert not after[1], "importing runtime/shapes enabled the compile cache"
print("OK")
"""


@pytest.mark.unit
def test_runtime_and_shapes_import_has_no_side_effect() -> None:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("JAX_ENABLE_X64", "JAX_COMPILATION_CACHE_DIR")
    }
    done = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().endswith("OK")


@pytest.mark.unit
def test_kk_device_and_shapes_re_export_the_promoted_names() -> None:
    from xraylabtool.backend import runtime, shapes
    from xraylabtool.kk import _device, _shapes

    for name in (
        "ensure_x64",
        "enable_persistent_cache",
        "worker_guard",
        "transfer_counts",
        "reset_transfer_counts",
        "upload",
        "fetch_status",
        "fetch",
    ):
        assert getattr(_device, name) is getattr(runtime, name), name
        assert name in _device.__all__
    for name in ("bucket", "bucket_pow2", "block_for"):
        assert getattr(_shapes, name) is getattr(shapes, name), name
        assert name in _shapes.__all__


@pytest.mark.unit
def test_transfer_counters_live_in_runtime_only() -> None:
    import numpy as np

    from xraylabtool.backend import runtime
    from xraylabtool.kk import _device

    assert not hasattr(_device, "_counts")
    runtime.reset_transfer_counts()
    tree = runtime.upload({"a": np.arange(3.0)})
    runtime.fetch(tree)
    assert _device.transfer_counts() == {"upload": 1, "fetch_status": 0, "fetch": 1}
    runtime.reset_transfer_counts()
