"""Transfer and compile budgets of the optics programs (plan step 1.6, P13)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.fixtures.core_harness import run_program
from tests.fixtures.kk_compiles import record_compiles
from xraylabtool.backend import runtime
from xraylabtool.calculators import optics
from xraylabtool.calculators.tables import clear_tables, load_tables

MATS = [("SiO2", 2.2), ("Au", 19.32), ("H2O", 1.0)]
PROGRAMS = {"_optical_program", "_tail_program"}


def _mine(names: list[str]) -> list[str]:
    return [n for n in names if n in PROGRAMS]


@pytest.fixture(autouse=True)
def _warm():
    runtime.ensure_x64()
    load_tables()


def test_one_program_call_is_one_upload_and_one_fetch():
    run_program(MATS, np.linspace(1.0, 20.0, 50))  # warm: compile outside the count
    runtime.reset_transfer_counts()
    run_program(MATS, np.linspace(1.0, 20.0, 50))
    counts = runtime.transfer_counts()
    assert counts == {"upload": 1, "fetch_status": 0, "fetch": 1}


def test_first_call_compiles_then_same_buckets_compile_nothing():
    kw = {"m_bucket": 72}  # a shape no other test uses: guaranteed fresh compile
    with record_compiles() as names:
        run_program(MATS, np.linspace(1.0, 20.0, 50), **kw)
    assert _mine(names) == ["_optical_program"], names  # recorder self-check
    with record_compiles() as names:
        run_program(MATS, np.linspace(2.0, 25.0, 50), **kw)
    assert _mine(names) == [], names


def test_energy_count_inside_the_same_bucket_compiles_nothing():
    """P13: 37, 50 and 64 energies all land in the 64 bucket."""
    run_program(MATS, np.linspace(1.0, 20.0, 37))
    with record_compiles() as names:
        for m in (50, 64, 1, 63):
            run_program(MATS, np.linspace(1.0, 20.0, m))
    assert _mine(names) == [], names


def test_another_bucket_does_compile():
    with record_compiles() as names:
        run_program(MATS, np.linspace(1.0, 20.0, 50), m_bucket=88)
    assert _mine(names) == ["_optical_program"], names


def test_a_second_tables_object_hits_the_same_compiled_program():
    run_program(MATS, np.linspace(1.0, 20.0, 50))
    before = optics._optical_program._cache_size()
    first = load_tables()
    clear_tables()
    second = load_tables()
    assert second is not first
    with record_compiles() as names:
        run_program(MATS, np.linspace(1.0, 20.0, 50))
    assert _mine(names) == [], names
    assert optics._optical_program._cache_size() == before
