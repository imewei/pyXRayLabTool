"""The library enables the persistent compile cache only when asked to."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

_CODE = textwrap.dedent(
    """
    import jax
    import numpy as np
    import xraylabtool as xlt
    xlt.calculate_single_material_properties("SiO2", np.array([8.0]), 2.2)
    print(repr(jax.config.jax_compilation_cache_dir))
    """
)


def _run(tmp_path, **env_extra: str) -> str:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("XRAYLABTOOL_", "JAX_COMPILATION_CACHE"))
    }
    env.update(
        XDG_CACHE_HOME=str(tmp_path), LOCALAPPDATA=str(tmp_path), JAX_PLATFORMS="cpu"
    )
    env.update(env_extra)
    done = subprocess.run(
        [sys.executable, "-c", _CODE],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    return done.stdout.strip().splitlines()[-1]


@pytest.mark.unit
def test_a_core_call_leaves_the_cache_alone_by_default(tmp_path):
    assert _run(tmp_path) in ("None", "''")


@pytest.mark.unit
def test_the_opt_in_env_var_enables_the_cache(tmp_path):
    out = _run(tmp_path, XRAYLABTOOL_JAX_CACHE="1")
    assert str(tmp_path) in out
    assert "jax" in out.rsplit(os.sep, 1)[-1]
    assert any((tmp_path / "xraylabtool" / "jax").iterdir())  # programs were written


@pytest.mark.unit
def test_a_relative_xdg_cache_home_is_ignored(tmp_path):
    out = _run(
        tmp_path,
        XRAYLABTOOL_JAX_CACHE="1",
        XDG_CACHE_HOME="relative/cache",
        HOME=str(tmp_path),  # the fallback must land under the temp home
    )
    expected = tmp_path / ".cache" / "xraylabtool" / "jax"
    assert out.strip("'\"") == str(expected)


@pytest.mark.unit
def test_the_opt_out_wins_over_the_opt_in(tmp_path):
    out = _run(tmp_path, XRAYLABTOOL_JAX_CACHE="1", XRAYLABTOOL_NO_JAX_CACHE="1")
    assert out in ("None", "''")


@pytest.mark.unit
@pytest.mark.parametrize("value", ["", "0", "true"])
def test_only_the_value_1_opts_in(tmp_path, value):
    assert _run(tmp_path, XRAYLABTOOL_JAX_CACHE=value) in ("None", "''")


@pytest.mark.unit
def test_a_user_preset_cache_directory_is_respected(tmp_path):
    mine = tmp_path / "mine"
    out = _run(tmp_path, XRAYLABTOOL_JAX_CACHE="1", JAX_COMPILATION_CACHE_DIR=str(mine))
    assert str(mine) in out


@pytest.mark.unit
def test_a_failed_cache_setup_is_retried_not_hidden(monkeypatch):
    from xraylabtool.backend import runtime

    calls = []

    def failing() -> None:
        calls.append(1)
        raise OSError("cache directory unavailable")

    monkeypatch.setenv("XRAYLABTOOL_JAX_CACHE", "1")
    monkeypatch.setattr(runtime, "_cache_opt_in_checked", False)
    monkeypatch.setattr(runtime, "enable_persistent_cache", failing)
    for _ in range(2):
        with pytest.raises(OSError, match="unavailable"):
            runtime.ensure_x64()
    assert len(calls) == 2
