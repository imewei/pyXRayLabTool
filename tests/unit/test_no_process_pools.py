"""The package starts no pool and imports neither ``multiprocessing`` nor ``concurrent``.

Batches run as device ``F``-blocks (spec section 7). Forking after JAX has started
threads can deadlock, so no pool may exist anywhere in ``xraylabtool/``. The GUI's
``QThreadPool`` is a Qt object, not a ``concurrent.futures`` pool, and is allowed.
The scan works on the AST, so ``import x``, ``import x as y``, ``from x import y``
and attribute use are all seen, and it is proven not to be vacuous by flagging
a snippet of each form.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import xraylabtool

POOLS = {"ProcessPoolExecutor", "ThreadPoolExecutor"}


def _banned_module(name: str) -> bool:
    """``multiprocessing`` and ``concurrent`` (``concurrent.futures``) in any form."""
    return name.split(".", maxsplit=1)[0] in {"multiprocessing", "concurrent"}


def forbidden(source: str) -> list[str]:
    """What in ``source`` imports ``multiprocessing`` or names a pool executor."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [f"import {a.name}" for a in node.names if _banned_module(a.name)]
        elif isinstance(node, ast.ImportFrom):
            if _banned_module(node.module or ""):
                found.append(f"from {node.module} import ...")
            found += [f"import {a.name}" for a in node.names if a.name in POOLS]
        elif isinstance(node, ast.Attribute) and node.attr in POOLS:
            found.append(f"attribute {node.attr}")
        elif isinstance(node, ast.Name) and node.id in POOLS:
            found.append(f"name {node.id}")
    return found


@pytest.mark.parametrize(
    "snippet",
    [
        "import multiprocessing",
        "import multiprocessing.pool as p",
        "from multiprocessing import Pool",
        "from multiprocessing.pool import ThreadPool",
        "from concurrent.futures import ProcessPoolExecutor",
        "from concurrent.futures import ThreadPoolExecutor as T",
        "import concurrent.futures\nconcurrent.futures.ProcessPoolExecutor(2)",
        "from concurrent import futures\nfutures.ThreadPoolExecutor(2)",
        "import concurrent.futures",
        "import concurrent.futures as cf",
        "from concurrent import futures",
        "from concurrent.futures import as_completed",
        "def f(ex=ThreadPoolExecutor):\n    return ex",
    ],
)
def test_the_scan_flags_every_form_of_a_pool(snippet: str) -> None:
    assert forbidden(snippet)


@pytest.mark.parametrize(
    "snippet",
    [
        "from PySide6.QtCore import QThreadPool\nQThreadPool.globalInstance()",
        "import threading\nthreading.Thread(target=print)",
        "x = 'multiprocessing ProcessPoolExecutor'  # in a string",
    ],
)
def test_the_scan_leaves_qt_thread_pools_and_plain_text_alone(snippet: str) -> None:
    assert forbidden(snippet) == []


def test_the_package_has_no_process_or_thread_pool() -> None:
    root = Path(xraylabtool.__file__).parent
    files = sorted(root.rglob("*.py"))
    assert len(files) > 50  # the scan really walked the package
    offenders = {
        str(path.relative_to(root)): hits
        for path in files
        if (hits := forbidden(path.read_text(encoding="utf-8")))
    }
    assert offenders == {}
