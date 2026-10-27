"""MemoryMonitor.force_gc collects garbage only (ADR-006: resident tables stay)."""

from __future__ import annotations

import xraylabtool as xlt
from xraylabtool.calculators import core
from xraylabtool.calculators.tables import load_tables
from xraylabtool.data_handling.batch_processing import MemoryMonitor


def test_force_gc_keeps_the_resident_tables_and_the_encoding_memo():
    xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)
    tables = load_tables()
    memo = core._encode_material.cache_info().currsize
    assert memo > 0

    MemoryMonitor(4.0).force_gc()

    assert load_tables() is tables
    assert core._encode_material.cache_info().currsize == memo
