"""Completeness checks for the ``xraylabtool.kk`` documentation."""

from __future__ import annotations

import inspect
from pathlib import Path
import re

import pytest

from xraylabtool import kk
from xraylabtool.calculators.core import calculate_properties_from_factors

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs"
API_PAGE = DOCS / "api" / "kk.rst"
PHYSICS_PAGE = DOCS / "physics" / "kramers_kronig.rst"
GUIDE_PAGE = DOCS / "guides" / "kk_guide.rst"


def _public_objects() -> list[tuple[str, object]]:
    """Every name in ``kk.__all__``, its public methods, and the f1/f2 helper."""
    objects: list[tuple[str, object]] = []
    for name in kk.__all__:
        obj = getattr(kk, name)
        objects.append((name, obj))
        if inspect.isclass(obj):
            objects.extend(
                (f"{name}.{attr}", member)
                for attr, member in vars(obj).items()
                if not attr.startswith("_") and inspect.isfunction(member)
            )
    objects.append(
        ("calculate_properties_from_factors", calculate_properties_from_factors)
    )
    return objects


@pytest.mark.unit
@pytest.mark.parametrize("name", kk.__all__)
def test_every_public_kk_name_is_documented(name: str) -> None:
    assert re.search(rf"\b{re.escape(name)}\b", API_PAGE.read_text(encoding="utf-8"))


@pytest.mark.unit
def test_transform_docstrings_cite_paper_equations() -> None:
    assert "Eq. (9)" in (inspect.getdoc(kk.relativistic_correction) or "")
    assert "Eq. (10)" in (inspect.getdoc(kk.kk_piecewise_poly) or "")
    assert "Eq. (8)" in (inspect.getdoc(kk.evaluate_f2) or "")
    doc = inspect.getdoc(kk.refine_spectrum) or ""
    assert "References" in doc
    assert "not in B. Watts" in doc.split("References")[1]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "obj"), _public_objects(), ids=[n for n, _ in _public_objects()]
)
def test_public_docstrings_have_examples_and_references(name: str, obj: object) -> None:
    doc = inspect.getdoc(obj) or ""
    assert "Examples\n--------" in doc, name
    assert "References\n----------" in doc, name


@pytest.mark.unit
def test_docs_pages_toctree_linked() -> None:
    for page in (API_PAGE, PHYSICS_PAGE, GUIDE_PAGE):
        assert page.is_file(), page
    index = (DOCS / "index.rst").read_text(encoding="utf-8")
    assert re.search(r"^\s+physics/kramers_kronig\s*$", index, re.MULTILINE)
    assert re.search(r"^\s+guides/kk_guide\s*$", index, re.MULTILINE)
    api_index = (DOCS / "api" / "index.rst").read_text(encoding="utf-8")
    assert re.search(r"^\s+kk\s*$", api_index, re.MULTILINE)


@pytest.mark.unit
def test_docs_state_the_eq10_erratum() -> None:
    text = PHYSICS_PAGE.read_text(encoding="utf-8")
    assert "a₋₃" in text or "a_{-3}" in text or "a_{j,-3}" in text
    assert "erratum" in text.lower()
    # The published page prints x_{j+1}^{-1} - x_j^{-1}; a j-1 subscript there
    # is a low-resolution rendering artefact and must not be documented.
    assert "x_{j-1}^{-1}" not in text
    assert r"\left(x_j^{-1} - x_{j+1}^{-1}\right)" in text


@pytest.mark.unit
def test_docs_cite_the_paper_and_upstream_licence() -> None:
    citation = "doi:10.1364/OE.22.023628"
    for page in (API_PAGE, PHYSICS_PAGE, GUIDE_PAGE):
        text = page.read_text(encoding="utf-8")
        assert citation in text, page
        assert "\nReferences\n----------\n" in text, page
    guide = GUIDE_PAGE.read_text(encoding="utf-8")
    assert "f331ca31a654af946f7d550a22cf4e9c11fa8afc" in guide
    assert "Zlib" in guide
