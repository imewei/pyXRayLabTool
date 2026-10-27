"""``import xraylabtool`` and the cheap CLI paths must not import JAX (spec section 10).

Each case runs in a fresh interpreter and inspects ``sys.modules`` afterwards.

* ``test_no_jax_without_gpu_tools``: ``PATH`` holds no ``nvidia-smi``, which is
  how a CI runner looks. Must pass today.
* ``test_no_jax_with_gpu_tools``: the inherited ``PATH``. On a GPU host the
  package init imports ``backend.array_ops``, whose backend auto-select imports
  JAX when ``nvidia-smi`` exists, so every subcase is a strict ``xfail`` there.
  Task 2.0 removes the leak and its ``xfail`` marks (one per subcase, so a
  subcase that still leaks after 2.0 keeps only its own mark).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

_PROBE = """
import json, sys

mode = sys.argv[1]
if mode == "import":
    import xraylabtool
else:
    from xraylabtool.interfaces.cli import main

    sys.argv = ["xraylabtool", "--" + mode]
    try:
        main()
    except SystemExit:
        pass
leaked = sorted(
    m for m in sys.modules
    if m in ("jax", "xraylabtool.kk") or m.startswith(("jax.", "xraylabtool.kk."))
)
print("LEAKED=" + json.dumps(leaked))
"""

SUBCASES = ("import", "help", "version")
GPU_HOST = shutil.which("nvidia-smi") is not None
_LEAK_REASON = (
    "package init imports backend.array_ops, whose auto-select imports JAX when "
    "nvidia-smi exists; removed in Task 2.0"
)


def _leaked_modules(mode: str, env: dict[str, str]) -> list[str]:
    done = subprocess.run(
        [sys.executable, "-c", _PROBE, mode],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    marker = [ln for ln in done.stdout.splitlines() if ln.startswith("LEAKED=")]
    assert marker, done.stdout[-2000:]
    return json.loads(marker[-1].removeprefix("LEAKED="))  # type: ignore[no-any-return]


@pytest.mark.parametrize("mode", SUBCASES)
def test_no_jax_without_gpu_tools(mode: str, tmp_path: Path) -> None:
    env = {**os.environ, "PATH": str(tmp_path)}  # empty directory: no nvidia-smi
    assert _leaked_modules(mode, env) == []


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(
            mode,
            marks=pytest.mark.xfail(
                condition=GPU_HOST, reason=_LEAK_REASON, strict=True
            ),
        )
        for mode in SUBCASES
    ],
)
def test_no_jax_with_gpu_tools(mode: str) -> None:
    assert _leaked_modules(mode, dict(os.environ)) == []
