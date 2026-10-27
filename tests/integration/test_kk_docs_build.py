"""Sphinx build checks for the Kramers-Kronig documentation pages (slow)."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

import pytest

pytest.importorskip("sphinx")

ROOT = Path(__file__).resolve().parents[2]
# Warnings from the new pages, the kk package or the JAX-first core pages
# (calculators/backend API, ADR-006); other pre-existing warnings (e.g.
# untracked docs/superpowers/*.md) are not this test's concern.
KK_WARNING = re.compile(
    r"kramers_kronig|kk_guide|api/kk\b|xraylabtool[./]kk\b"
    r"|api/calculators|api/backend|006-adr"
    r"|xraylabtool[./]calculators[./](optics|tables|cache)\b"
    r"|xraylabtool[./]backend\b"
)


def _sphinx(builder: str, out: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", builder, "-q", "docs", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )


@pytest.mark.slow
@pytest.mark.integration
def test_kk_docs_build_adds_no_warnings(tmp_path: Path) -> None:
    proc = _sphinx("html", tmp_path)
    assert proc.returncode == 0, proc.stderr[-2000:]
    kk_lines = [
        line
        for line in proc.stderr.splitlines()
        if ("WARNING" in line or "ERROR" in line) and KK_WARNING.search(line)
    ]
    assert kk_lines == []
    for page in (
        "physics/kramers_kronig",
        "api/kk",
        "guides/kk_guide",
        "api/calculators",
        "api/backend",
        "architecture/006-adr-core-jax-only",
    ):
        assert (tmp_path / f"{page}.html").is_file(), page


@pytest.mark.slow
@pytest.mark.integration
def test_kk_docs_doctests_pass(tmp_path: Path) -> None:
    """Whole-docs gate: any doctest failure on any page fails this test."""
    proc = _sphinx("doctest", tmp_path)
    output = (tmp_path / "output.txt").read_text(encoding="utf-8")
    assert proc.returncode == 0, output[-3000:]
    assert re.search(r"^\s+0 failures in tests", output, re.MULTILINE), output[-3000:]
    # The kk API page contributes its docstring examples to the run.
    assert re.search(r"^Document: api/kk$", output, re.MULTILINE), output[-3000:]
