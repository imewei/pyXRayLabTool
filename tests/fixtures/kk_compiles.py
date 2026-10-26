"""Compile-event recorder for the ``xraylabtool.kk`` compile-budget tests.

On the installed jax the "Compiling jit(<name>) ..." message of
``jax.log_compiles`` comes from ``jax._src.interpreters.pxla`` (not
``dispatch``), so the recorder listens there. It counts compile EVENTS: a
function compiled at two shapes is two events. JAX's own trivial primitive
compiles (``jit(slice)``, ``jit(broadcast_in_dim)`` ...) show up under their
primitive names and are excluded by :func:`kk_programs`, which keeps only the
top-level numerical programs of ``xraylabtool.kk``.

The logger path is private, so ``tests/unit/kk/test_device_refine.py`` self-tests
the recorder with a freshly defined function compiled at two shapes: a silent
zero cannot pass.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import logging
import re

import jax

_LOGGER = "jax._src.interpreters.pxla"
_COMPILING = re.compile(r"Compiling jit\((.+?)\) with global shapes")

# Names of the jitted top-level programs defined in ``xraylabtool.kk``.
KK_PROGRAM_NAMES = frozenset(
    {
        "compound_spectrum_device",
        "evaluate_f2_device",
        "to_asf_scale_device",
        "merge_device",
        "_kk_f1_blocks",
        "_assemble",
        "_evaluate",
        "_start",
        "_first_pass",
        "_refine_pass",
        "_finalize",
        "_fit_columns",
    }
)

# Names under which JAX compiles eager primitives and tiny helper ops; every
# other compile event is a numerical program and must be in KK_PROGRAM_NAMES.
TRIVIAL_PRIMITIVES = frozenset(
    {
        "convert_element_type",
        "broadcast_in_dim",
        "iota",
        "slice",
        "dynamic_slice",
        "dynamic_update_slice",
        "concatenate",
        "squeeze",
        "reshape",
        "transpose",
        "select_n",
        "gather",
        "scatter",
        "_pad",
        "_where",
        "_unstack",
        "_reduce_sum",
        "_reduce_max",
        "_reduce_min",
        "_cumsum",
        "_argsort",
        "_arange",
        "_zeros",
        "_full",
        "_asarray",
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "eq",
        "lt",
        "le",
        "gt",
        "ge",
        "max",
        "min",
    }
)


class _Handler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.names: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        found = _COMPILING.search(record.getMessage())
        if found:
            self.names.append(found.group(1))


@contextmanager
def record_compiles() -> Iterator[list[str]]:
    """Collect the name of every XLA compile in the ``with`` block.

    Yields a list that fills as compiles happen (one entry per event, in
    order).
    """
    logger = logging.getLogger(_LOGGER)
    handler = _Handler()
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        with jax.log_compiles(True):
            yield handler.names
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


record = record_compiles


def kk_programs(names: list[str]) -> list[str]:
    """The entries of ``names`` that are top-level ``xraylabtool.kk`` programs."""
    return [n for n in names if n in KK_PROGRAM_NAMES]


def numerical_programs(names: list[str]) -> list[str]:
    """Every compile event that is not a trivial primitive (kk or not)."""
    return [n for n in names if n not in TRIVIAL_PRIMITIVES]
