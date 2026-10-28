"""``warm_up`` and the persistent compile-cache helper (design spec §10)."""

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import jax
import pytest

from tests.unit.kk.test_pipeline_device import _subprocess
from xraylabtool import kk
from xraylabtool.backend import runtime

_CONFIGS = ("jax_compilation_cache_dir", "jax_persistent_cache_min_compile_time_secs")


@pytest.fixture
def cache_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Cache opt-out cleared; the two JAX options restored afterwards."""
    saved = {name: getattr(jax.config, name) for name in _CONFIGS}
    monkeypatch.delenv("XRAYLABTOOL_NO_JAX_CACHE", raising=False)
    monkeypatch.delenv("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", raising=False)
    jax.config.update("jax_compilation_cache_dir", None)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    yield
    for name, value in saved.items():
        jax.config.update(name, value)


_WARM_SCRIPT = """
    import json
    from tests.fixtures.kk_compiles import kk_programs, record_compiles
    from xraylabtool.kk import kk_calculate_real, warm_up
    if {warm}:
        warm_up(("SiO2",))
    with record_compiles() as events:
        kk_calculate_real("SiO2")
        kk_calculate_real("SiO2", refine_tolerance=0.01)
    print(json.dumps(kk_programs(events)))
"""


@pytest.mark.unit
def test_warm_up_then_no_compile() -> None:
    """Fresh interpreters: after ``warm_up`` the default runs compile nothing."""
    assert _subprocess(_WARM_SCRIPT.format(warm=False))  # control: cold run compiles
    assert _subprocess(_WARM_SCRIPT.format(warm=True)) == []


@pytest.mark.unit
def test_warm_up_exported_lazily() -> None:
    assert "warm_up" in kk.__all__
    code = textwrap.dedent(
        """
        import sys, xraylabtool
        assert "xraylabtool.kk" not in sys.modules
        xraylabtool.kk.warm_up
        assert "xraylabtool.kk" in sys.modules
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


@pytest.mark.unit
def test_cache_helper_enables_and_sets_threshold(
    cache_config: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "linux")
    assert runtime.enable_persistent_cache() is True
    assert jax.config.jax_compilation_cache_dir == str(tmp_path / "xraylabtool" / "jax")
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.0
    assert runtime.enable_persistent_cache() is False  # already configured


@pytest.mark.unit
def test_cache_helper_respects_opt_out_and_existing_config(
    cache_config: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XRAYLABTOOL_NO_JAX_CACHE", "1")
    assert runtime.enable_persistent_cache() is False
    assert jax.config.jax_compilation_cache_dir in (None, "")

    monkeypatch.delenv("XRAYLABTOOL_NO_JAX_CACHE")
    jax.config.update("jax_compilation_cache_dir", "/user/chose/this")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 3.0)
    assert runtime.enable_persistent_cache() is False
    assert jax.config.jax_compilation_cache_dir == "/user/chose/this"
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 3.0


@pytest.mark.unit
def test_cache_helper_keeps_user_threshold(
    cache_config: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "linux")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 5.0)
    assert runtime.enable_persistent_cache() is True
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 5.0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("platform", "env", "expected"),
    [
        ("linux", {"XDG_CACHE_HOME": "{root}/xdg"}, "/xdg/xraylabtool/jax"),
        ("darwin", {}, "Library/Caches/xraylabtool/jax"),
        ("win32", {"LOCALAPPDATA": "/local"}, "/local/xraylabtool/jax"),
    ],
)
def test_cache_dir_is_os_appropriate(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    env: dict[str, str],
    expected: str,
    tmp_path: Path,
) -> None:
    for name in ("XDG_CACHE_HOME", "LOCALAPPDATA"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        # XDG_CACHE_HOME must be absolute on the host OS ("/xdg" has no drive on Windows)
        monkeypatch.setenv(name, value.format(root=tmp_path.as_posix()))
    assert runtime._user_cache_dir(platform).as_posix().endswith(expected)


@pytest.mark.unit
def test_library_import_does_not_enable_cache() -> None:
    code = textwrap.dedent(
        """
        import jax
        from xraylabtool.kk import kk_calculate_real
        kk_calculate_real("Si", energies_kev=[1.0])
        assert not jax.config.jax_compilation_cache_dir
        """
    )
    # conftest sets the opt-out for the suite; the library must not need it.
    env = {k: v for k, v in os.environ.items() if k != "XRAYLABTOOL_NO_JAX_CACHE"}
    subprocess.run([sys.executable, "-c", code], check=True, timeout=300, env=env)


@pytest.mark.unit
def test_cli_does_not_call_warm_up(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import patch

    from xraylabtool.interfaces.cli import main

    with (
        patch("sys.argv", ["xraylabtool", "kk", "Si", "--energies", "1.0"]),
        patch.object(kk, "warm_up", side_effect=AssertionError("CLI warmed up")),
    ):
        assert main() == 0
