"""Per-command JAX discipline of the CLI (spec section 10, plan step 3.5a).

Each case runs ``interfaces.cli.main`` in a fresh interpreter.

* ``formula``, ``atomic``, ``list`` and ``completion`` must finish successfully
  without importing JAX or ``xraylabtool.kk``.
* ``calc``, ``batch``, ``compare`` and ``kk`` must run successfully and enable the
  persistent compile cache in the OS user-cache directory (``XDG_CACHE_HOME``
  points at a temporary directory, so the real cache is never touched), unless
  ``XRAYLABTOOL_NO_JAX_CACHE=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

_PROBE = """
import json, sys

from xraylabtool.interfaces.cli import main

sys.argv = ["xraylabtool", *sys.argv[1:]]
rc = main()
leaked = sorted(
    m for m in sys.modules
    if m in ("jax", "xraylabtool.kk") or m.startswith(("jax.", "xraylabtool.kk."))
)
cache_dir = None
if "jax" in sys.modules:
    import jax

    cache_dir = jax.config.jax_compilation_cache_dir
print("PROBE=" + json.dumps({"rc": rc, "leaked": leaked, "cache_dir": cache_dir}))
"""

JAX_FREE = {
    "formula": ["formula", "SiO2"],
    "atomic": ["atomic", "Si"],
    "list": ["list", "constants"],
    "completion": ["completion", "info"],
}


def _run_cli(argv: list[str], cache_home: Path, *, opt_out: bool) -> dict[str, object]:
    env = {**os.environ, "XDG_CACHE_HOME": str(cache_home)}
    if opt_out:
        env["XRAYLABTOOL_NO_JAX_CACHE"] = "1"
    else:
        env.pop("XRAYLABTOOL_NO_JAX_CACHE", None)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    done = subprocess.run(
        [sys.executable, "-c", _PROBE, *argv],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    marker = [ln for ln in done.stdout.splitlines() if ln.startswith("PROBE=")]
    assert marker, (done.stdout[-1500:], done.stderr[-1500:])
    probe: dict[str, object] = json.loads(marker[-1].removeprefix("PROBE="))
    assert probe["rc"] == 0, (done.stdout[-1500:], done.stderr[-1500:])
    return probe


@pytest.mark.parametrize("command", sorted(JAX_FREE))
def test_jax_free_commands_run_without_importing_jax(
    command: str, tmp_path: Path
) -> None:
    probe = _run_cli(JAX_FREE[command], tmp_path, opt_out=False)
    assert probe["leaked"] == []
    assert not (tmp_path / "xraylabtool").exists()


def _calc_argv(tmp_path: Path) -> list[str]:
    return ["calc", "SiO2", "-e", "8.0", "-d", "2.2"]


def _batch_argv(tmp_path: Path) -> list[str]:
    source = tmp_path / "materials.csv"
    source.write_text("formula,density,energy\nSiO2,2.2,8.0\nAl2O3,3.95,8.0\n")
    return ["batch", str(source), "-o", str(tmp_path / "out.csv")]


def _compare_argv(tmp_path: Path) -> list[str]:
    return ["compare", "SiO2,2.2", "Al2O3,3.95", "-e", "8.0"]


def _kk_argv(tmp_path: Path) -> list[str]:
    return ["kk", "Si", "--energies", "5,6,7"]


DEVICE_COMMANDS = {
    "calc": _calc_argv,
    "batch": _batch_argv,
    "compare": _compare_argv,
    "kk": _kk_argv,
}


@pytest.mark.parametrize("command", sorted(DEVICE_COMMANDS))
def test_device_commands_enable_the_persistent_cache(
    command: str, tmp_path: Path
) -> None:
    cache_home = tmp_path / "cache_home"
    argv = DEVICE_COMMANDS[command](tmp_path)
    probe = _run_cli(argv, cache_home, opt_out=False)
    expected = cache_home / "xraylabtool" / "jax"
    assert probe["cache_dir"] == str(expected)
    assert any(expected.iterdir())  # a program was compiled and persisted


@pytest.mark.parametrize("command", sorted(DEVICE_COMMANDS))
def test_device_commands_honour_the_cache_opt_out(command: str, tmp_path: Path) -> None:
    cache_home = tmp_path / "cache_home"
    argv = DEVICE_COMMANDS[command](tmp_path)
    probe = _run_cli(argv, cache_home, opt_out=True)
    assert probe["cache_dir"] in (None, "")
    assert not cache_home.exists()
