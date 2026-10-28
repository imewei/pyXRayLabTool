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
    env = {k: v for k, v in os.environ.items() if not k.startswith("XRAYLABTOOL_")}
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
    assert out.endswith("jax'") or out.endswith('jax"')


@pytest.mark.unit
def test_the_opt_out_wins_over_the_opt_in(tmp_path):
    out = _run(tmp_path, XRAYLABTOOL_JAX_CACHE="1", XRAYLABTOOL_NO_JAX_CACHE="1")
    assert out in ("None", "''")
