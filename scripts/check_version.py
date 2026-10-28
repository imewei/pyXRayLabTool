#!/usr/bin/env python3
"""Check that every place that records the package version agrees.

The version lives in several files that must be bumped together. This reads
them as text (it does not import the package, so it is safe in pre-commit and
before an install) and exits non-zero on any mismatch.

Usage:
    python scripts/check_version.py              # sources agree with each other
    python scripts/check_version.py --expect 0.5.0   # ... and equal this release
    python scripts/check_version.py --installed  # ... and the installed metadata
"""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import re
import sys
import tomllib

ROOT = Path(__file__).resolve().parent.parent


def _search(path: str, pattern: str) -> str:
    text = (ROOT / path).read_text(encoding="utf-8")
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        raise SystemExit(f"error: no version found in {path} (pattern {pattern!r})")
    return match.group(1)


def collect() -> dict[str, str]:
    """Return {description: version} for every file that records the version."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    root_pkg = next(p for p in lock["package"] if p["name"] == "xraylabtool")
    return {
        "pyproject.toml": pyproject["project"]["version"],
        "xraylabtool/__init__.py": _search(
            "xraylabtool/__init__.py", r'^__version__ = "([^"]+)"'
        ),
        "uv.lock (root package)": root_pkg["version"],
        "docs/conf.py (version fallback)": _search(
            "docs/conf.py", r'^\s+version = "([^"]+)"'
        ),
        "docs/conf.py (release fallback)": _search(
            "docs/conf.py", r'^\s+release = "([^"]+)"'
        ),
        "README.md (BibTeX)": _search("README.md", r"version = \{([^}]+)\}"),
        "docs/index.rst (BibTeX)": _search("docs/index.rst", r"version=\{([^}]+)\}"),
        "CHANGELOG.md (newest release)": _search(
            "CHANGELOG.md", r"^## \[(\d+\.\d+\.\d+)\]"
        ),
        "docs/development/changelog.rst (newest)": _search(
            "docs/development/changelog.rst", r"^v(\d+\.\d+\.\d+) \("
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--expect", metavar="X.Y.Z", help="version the release must be")
    parser.add_argument(
        "--installed", action="store_true", help="also compare the installed metadata"
    )
    parser.add_argument(
        "--print", action="store_true", help="print the pyproject.toml version and exit"
    )
    args = parser.parse_args()

    versions = collect()
    if args.print:
        print(versions["pyproject.toml"])
        return 0
    if args.installed:
        try:
            versions["installed metadata"] = importlib.metadata.version("xraylabtool")
        except importlib.metadata.PackageNotFoundError:
            print("note: xraylabtool is not installed; skipping installed metadata")

    target = args.expect or versions["pyproject.toml"]
    width = max(map(len, versions))
    bad = 0
    for name, version in versions.items():
        ok = version == target
        bad += not ok
        print(f"  {'OK ' if ok else 'BAD'} {name:<{width}}  {version}")
    if bad:
        print(f"{bad} of {len(versions)} sources differ from {target}", file=sys.stderr)
        return 1
    print(f"all {len(versions)} sources agree on {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
