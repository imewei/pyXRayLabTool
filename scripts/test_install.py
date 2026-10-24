#!/usr/bin/env python3
"""Smoke-test installing xraylabtool into a fresh virtual environment.

Usage:
    python scripts/test_install.py wheel [PATH]     # newest dist/*.whl by default
    python scripts/test_install.py testpypi [--version X.Y.Z]
    python scripts/test_install.py pypi [--version X.Y.Z]

After installing, runs one SiO2 calculation and prints the critical angle, so a
broken wheel (missing data files, bad entry points) fails here rather than for users.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

SMOKE = (
    "import xraylabtool as x; "
    "r = x.calculate_single_material_properties('SiO2', 10.0, 2.2); "
    "print(f'ok: xraylabtool {x.__version__}, "
    "SiO2 critical angle {r.critical_angle_degrees[0]:.3f} deg')"
)


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("source", choices=["wheel", "testpypi", "pypi"])
    parser.add_argument("path", nargs="?", help="wheel file (source=wheel only)")
    parser.add_argument("--version", help="release to install from an index")
    args = parser.parse_args()

    if args.source == "wheel":
        wheels = sorted(Path("dist").glob("*.whl"), key=lambda p: p.stat().st_mtime)
        wheel = Path(args.path) if args.path else (wheels[-1] if wheels else None)
        if wheel is None or not wheel.exists():
            print("error: no wheel found; run `make build` first", file=sys.stderr)
            return 1
        install = [str(wheel.resolve())]
    else:
        spec = f"xraylabtool=={args.version}" if args.version else "xraylabtool"
        index = (
            [
                "--index-url",
                "https://test.pypi.org/simple/",
                "--extra-index-url",
                "https://pypi.org/simple/",
            ]
            if args.source == "testpypi"
            else []
        )
        install = [*index, spec]

    with tempfile.TemporaryDirectory() as tmp:
        venv = Path(tmp) / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        python = str(_venv_python(venv))
        subprocess.run([python, "-m", "pip", "install", *install], check=True)
        result = subprocess.run(
            [python, "-c", SMOKE], capture_output=True, text=True, check=True
        )
        print(result.stdout.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
