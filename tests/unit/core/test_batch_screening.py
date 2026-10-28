"""Batch screening parity with the pre-switchover path (external review F1, F2, F3).

Expectations were read off ``9b18d70`` (the NumPy implementation) with the same
inputs; the energy-grouping and density cases fail without the fixes.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from xraylabtool.backend import shapes
from xraylabtool.calculators import host, optics
from xraylabtool.calculators.core import (
    calculate_multiple_xray_properties,
    calculate_single_material_properties,
    calculate_xray_properties,
)
from xraylabtool.data_handling import batch_processing as bp
from xraylabtool.data_handling.batch_processing import (
    BatchConfig,
    calculate_batch_properties,
    process_batch_chunk,
)

CONFIG = BatchConfig(enable_progress=False)
COMPLEX = "Complex energy values are not supported; pass real energies in keV"
TWO_D = "Energies must be a scalar or a 1-D array, got 2-D array with shape"
SEQUENCE = "can't multiply sequence by non-int of type 'float'"


def _chunk(chunk):
    with pytest.warns(UserWarning) as caught:
        got = process_batch_chunk(chunk, CONFIG)
    return got, [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


# ---- F1: equal bytes are not equal energies ---------------------------------------


def test_float32_pair_and_float64_view_keep_their_own_energies():
    a = np.array([8.0, 1.5], dtype=np.float32)
    b = a.view(np.float64)  # same bytes, one energy
    got = process_batch_chunk([("Si", a, 2.33), ("Au", b, 19.3)], CONFIG)
    assert got[0][1].energy_kev.tolist() == [8.0, 1.5]
    assert got[1][1].energy_kev.tolist() == b.tolist()


def test_a_complex_energy_does_not_share_a_group_with_real_bytes():
    real = np.array([8.0, 1.0])
    got, texts = _chunk([("Si", real, 2.33), ("Au", np.array([8.0 + 1.0j]), 19.3)])
    assert got[0][1].energy_kev.tolist() == [8.0, 1.0]
    assert got[1] == ("Au", None)
    assert len(texts) == 1
    assert texts[0].startswith(f"Failed to process formula 'Au': {COMPLEX}")


def test_a_reshaped_energy_grid_is_rejected_for_every_item():
    row = np.array([[8.0, 10.0]])
    col = np.array([[8.0], [10.0]])
    got, texts = _chunk([("Si", row, 2.33), ("Au", col, 19.3)])
    assert [r for _, r in got] == [None, None]
    assert len(texts) == 2
    assert all(TWO_D in t for t in texts)
    assert "(1, 2)" in texts[0]
    assert "(2, 1)" in texts[1]


def test_the_collisions_also_hold_through_process_chunks():
    a = np.array([8.0, 1.5], dtype=np.float32)
    data = [
        ("Si", a, 2.33),
        ("Au", a.view(np.float64), 19.3),
        ("Au", np.array([[8.0]]), 1.0),
    ]
    with pytest.warns(UserWarning, match="2-D"):
        got = bp._process_chunks(data, CONFIG, None)
    assert list(got) == ["Si@2.330", "Au@19.300", "Au"]
    assert got["Si@2.330"].energy_kev.tolist() == [8.0, 1.5]
    assert got["Au@19.300"].energy_kev.size == 1
    assert got["Au"] is None


def test_an_energy_input_numpy_cannot_arrayify_fails_alone():
    ragged = [[8.0], [8.0, 9.0]]
    got, texts = _chunk([("Si", ragged, 2.33), ("Au", np.array([8.0]), 19.3)])
    assert got[0] == ("Si", None)
    assert got[1][1] is not None
    assert len(texts) == 1


# ---- F2: a failed density conversion must not shift the encoded rows -----------------


def test_a_sequence_density_is_skipped_by_the_multiple_call():
    got = calculate_multiple_xray_properties(["SiO2", "Si"], [8.0], [[2.2], 2.33])
    assert list(got) == ["Si"]
    got = calculate_multiple_xray_properties(
        ["SiO2", "Si"], [8.0], [np.array([2.2, 3.0]), 2.33]
    )
    assert list(got) == ["Si"]


def test_a_sequence_density_is_rejected_up_front_by_calculate_xray_properties():
    with pytest.raises(
        ValueError, match=r"Density at index 0 must be a positive number"
    ):
        calculate_xray_properties(["SiO2", "Si"], [8.0], [[2.2], 2.33])


def test_a_sequence_density_warns_and_maps_to_none_in_the_batch():
    with pytest.warns(UserWarning) as caught:
        got = calculate_batch_properties(["SiO2", "Si"], [8.0], [[2.2], 2.33], CONFIG)
    assert list(got) == ["SiO2", "Si@2.330"]
    assert got["SiO2"] is None
    assert got["Si@2.330"].energy_kev.tolist() == [8.0]
    assert [str(w.message) for w in caught] == [
        f"Failed to process formula 'SiO2': {SEQUENCE}"
    ]


def test_a_sequence_density_in_the_middle_of_a_chunk_keeps_the_rows_aligned():
    energy = np.array([8.0])
    chunk = [("Si", energy, 2.33), ("SiO2", energy, [2.2]), ("Au", energy, 19.3)]
    got, texts = _chunk(chunk)
    assert [f for f, _ in got] == ["Si", "SiO2", "Au"]
    assert got[1][1] is None
    assert texts == [f"Failed to process formula 'SiO2': {SEQUENCE}"]
    # Au got Au's row, not SiO2's or Si's
    for index, (formula, density) in ((0, ("Si", 2.33)), (2, ("Au", 19.3))):
        ref = calculate_single_material_properties(formula, energy, density)
        assert got[index][1].molecular_weight_g_mol == ref.molecular_weight_g_mol
        np.testing.assert_array_equal(
            got[index][1].dispersion_delta, ref.dispersion_delta
        )


# ---- F3: a single oversized material is one F=1 block, never zero ---------------------


def test_the_planner_never_returns_a_block_of_zero():
    assert shapes.block_for(None, 10**7, 5) == 1
    assert shapes.block_for(None, 10**7, 1) == 1


def test_a_material_larger_than_the_budget_runs_as_blocks_of_exactly_one(monkeypatch):
    calls: list[int] = []
    real = host.run_optical

    def spy(encoded, densities, energies, **kw):
        calls.append(len(encoded))
        return real(encoded, densities, energies, **kw)

    monkeypatch.setattr(host, "run_optical", spy)
    # K=2 x M=64 = 128 elements per material: a budget of 100 cannot hold one.
    monkeypatch.setattr(
        host, "block_for", functools.partial(shapes.block_for, elements=100)
    )
    encoded = [optics.encode_formula("SiO2")] * 3
    got = host.run_optical_blocks(encoded, [2.2] * 3, np.linspace(2.0, 20.0, 50))
    assert calls == [1, 1, 1]
    assert got["ok"].all()
