"""Packaging checks for the Kramers-Kronig data and licence files (slow)."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA_FILES = ("data/kkcalc/asf.npz", "data/kkcalc/LICENSE-kkcalc.txt")

CHECK_SCRIPT = """
import numpy as np
import xraylabtool
from xraylabtool.kk import kk_calculate_real

res = kk_calculate_real("Si")
print(xraylabtool.__file__)
print(bool(np.isfinite(res.f1).all() and np.isfinite(res.f2).all()))
# Si at 10 keV is far from any edge: f1 ~ Z = 14 (14.2 tabulated), f2 ~ 0.3.
at_10 = kk_calculate_real("Si", energies_kev=[10.0])
print(float(at_10.f1[0]), float(at_10.f2[0]))
"""


def _run(
    cmd: list[str], cwd: Path, timeout: int = 900
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )
    assert proc.returncode == 0, f"{cmd}\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
    return proc


@pytest.mark.slow
@pytest.mark.integration
def test_built_wheel_and_sdist_contain_kk_data_and_licence(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to build and install the artefacts")

    dist = tmp_path / "dist"
    _run([uv, "build", "--out-dir", str(dist)], ROOT)
    (wheel,) = dist.glob("*.whl")
    (sdist,) = dist.glob("*.tar.gz")

    with zipfile.ZipFile(wheel) as zf:
        wheel_names = set(zf.namelist())
    for rel in DATA_FILES:
        assert f"xraylabtool/{rel}" in wheel_names
    # setuptools places `license-files` (LICENSE, NOTICE) under dist-info/licenses.
    assert any(n.endswith(".dist-info/licenses/NOTICE") for n in wheel_names)

    with tarfile.open(sdist) as tf:
        sdist_names = set(tf.getnames())
    prefix = sdist.name.removesuffix(".tar.gz")
    for rel in (*(f"xraylabtool/{r}" for r in DATA_FILES), "NOTICE"):
        assert f"{prefix}/{rel}" in sdist_names

    # Install the wheel alone (no dependency resolution, so this works offline)
    # and borrow third-party packages from the running environment. The repo
    # root is not exposed, so the wheel is what gets imported.
    venv = tmp_path / "venv"
    _run([uv, "venv", "--python", sys.executable, str(venv)], tmp_path)
    python = venv / "bin" / "python"
    if not python.exists():  # Windows layout
        python = venv / "Scripts" / "python.exe"
    _run(
        [uv, "pip", "install", "--no-deps", "--python", str(python), str(wheel)],
        tmp_path,
    )
    site = Path(
        _run(
            [
                str(python),
                "-I",
                "-c",
                "import sysconfig;print(sysconfig.get_path('purelib'))",
            ],
            tmp_path,
        ).stdout.strip()
    )
    (site / "_host_deps.pth").write_text(sysconfig.get_path("purelib") + "\n")

    out = _run([str(python), "-I", "-c", CHECK_SCRIPT], tmp_path).stdout.splitlines()
    assert Path(out[0]).is_relative_to(venv)
    assert out[1] == "True"
    f1, f2 = map(float, out[2].split())
    assert 13.5 < f1 < 15.0
    assert 0.0 < f2 < 1.0
