"""Many materials as ``F``-blocks of the device program (plan steps 3.1-3.4).

``calculate_xray_properties``, ``calculate_multiple_xray_properties`` and the batch
pipeline group materials by element bucket ``K``, cut each group into ``F``-blocks
(``shapes.block_for``) and make one upload and one fetch per block. Rows go back to
their input positions: no sort, no restore.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import functools
import logging
from typing import Any

import numpy as np
import pytest

from tests.fixtures.core_rounding import assert_ulp_close
from tests.fixtures.kk_compiles import record_compiles
from xraylabtool.backend import runtime, shapes
from xraylabtool.backend.shapes import bucket, bucket_pow2
from xraylabtool.calculators import core, host, optics
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

FIELDS = (
    "wavelength_angstrom",
    "dispersion_delta",
    "absorption_beta",
    "scattering_factor_f1",
    "scattering_factor_f2",
    "critical_angle_degrees",
    "attenuation_length_cm",
    "real_sld_per_ang2",
    "imaginary_sld_per_ang2",
    "energy_kev",
)
# K = 1, 2, 3, 4, 5 elements: element buckets 2, 2, 4, 4, 8.
MIXED = [
    ("Si", 2.33),
    ("SiO2", 2.2),
    ("CaCO3", 2.71),
    ("Ca5(PO4)3F", 3.19),
    ("KAl3Si3O10(OH)2", 2.8),
    ("H2O", 1.0),
]
ENERGIES = np.linspace(2.0, 25.0, 50)


@pytest.fixture(autouse=True)
def _warm() -> None:
    runtime.ensure_x64()
    host.device_tables()


@contextmanager
def _logged(logger: logging.Logger, level: int) -> Iterator[list[logging.LogRecord]]:
    """Records of ``logger`` (the xraylabtool logger does not propagate to caplog)."""
    records: list[logging.LogRecord] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Collect(level)
    old = logger.level
    logger.addHandler(handler)
    logger.setLevel(level)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old)


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every ``host.run_optical`` call: its material count and element bucket."""
    calls: list[dict[str, Any]] = []
    real = host.run_optical

    def run_optical(encoded: Any, densities: Any, energies: Any, **kw: Any) -> Any:
        calls.append({"f": len(encoded), **kw})
        return real(encoded, densities, energies, **kw)

    monkeypatch.setattr(host, "run_optical", run_optical)
    return calls


def _same_bits(a: Any, b: Any) -> None:
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_results_follow_the_input_order_across_element_buckets(spy):
    formulas = [f for f, _ in MIXED]
    densities = [d for _, d in MIXED]
    got = calculate_xray_properties(formulas, ENERGIES, densities)
    assert list(got) == formulas
    assert {c["k_bucket"] for c in spy} == {2, 4, 8}
    assert len(spy) == 3  # one block per element bucket
    for formula, density in MIXED:
        assert got[formula].formula == formula
        assert got[formula].density_g_cm3 == density


def test_batch_equals_a_loop_of_single_calls():
    """Same program and tables; the batch only changes the ``F`` and ``K`` padding.

    Padding is invariant bit for bit (tests/unit/core/test_optics.py), so the batch
    matches the single calls exactly; ``assert_ulp_close`` is the fallback only for
    a backend that fuses the ``F=1`` and ``F=8`` programs differently.
    """
    formulas = [f for f, _ in MIXED]
    batch = calculate_xray_properties(formulas, ENERGIES, [d for _, d in MIXED])
    for formula, density in MIXED:
        single = calculate_single_material_properties(formula, ENERGIES, density)
        for name in FIELDS:
            assert_ulp_close(getattr(batch[formula], name), getattr(single, name))
        assert batch[formula].molecular_weight_g_mol == single.molecular_weight_g_mol
        assert batch[formula].total_electrons == single.total_electrons
        assert_ulp_close(
            batch[formula].electron_density_per_ang3, single.electron_density_per_ang3
        )


def test_a_material_is_bit_identical_whatever_its_batch_mates():
    """Public-API padding invariance: other materials and the ``F`` bucket change nothing."""
    alone = calculate_xray_properties(["SiO2"], ENERGIES, [2.2])["SiO2"]
    for others in (
        [("Au", 19.32)],
        [("Au", 19.32), ("H2O", 1.0), ("Si", 2.33)],  # F = 4
        [(f"Cu{i}O", 8.0 + i) for i in range(1, 6)],  # F = 8
    ):
        formulas = ["SiO2", *(f for f, _ in others)]
        densities = [2.2, *(d for _, d in others)]
        got = calculate_xray_properties(formulas, ENERGIES, densities)["SiO2"]
        for name in FIELDS:
            _same_bits(getattr(got, name), getattr(alone, name))
        _same_bits(got.electron_density_per_ang3, alone.electron_density_per_ang3)


def test_unsorted_energies_keep_their_order_without_a_sort():
    energies = np.array([20.0, 3.0, 9.0, 3.0, 12.5])
    got = calculate_xray_properties(["SiO2", "Au"], energies, [2.2, 19.32])
    ref = calculate_xray_properties(["SiO2", "Au"], np.sort(energies), [2.2, 19.32])
    order = np.argsort(energies, kind="stable")
    for formula in got:
        np.testing.assert_array_equal(got[formula].energy_kev, energies)
        for name in FIELDS:
            _same_bits(getattr(got[formula], name)[order], getattr(ref[formula], name))


def test_a_failing_formula_in_the_middle_is_skipped_and_logged():
    formulas = ["SiO2", "Si", "Xx999", "Al2O3", "Au"]
    densities = [2.2, 2.33, 1.0, 3.95, 19.32]
    with _logged(core._logger, logging.WARNING) as records:
        got = calculate_xray_properties(formulas, ENERGIES, densities)
    assert list(got) == ["SiO2", "Si", "Al2O3", "Au"]
    failed = [r for r in records if r.getMessage() == "Failed to process formula"]
    assert [r.formula for r in failed] == ["Xx999"]  # type: ignore[attr-defined]
    assert failed[0].error.startswith("Failed to process formula 'Xx999': ")  # type: ignore[attr-defined]
    for formula, density in zip(got, [2.2, 2.33, 3.95, 19.32], strict=True):
        _same_bits(
            got[formula].dispersion_delta,
            calculate_single_material_properties(
                formula, ENERGIES, density
            ).dispersion_delta,
        )


def test_every_formula_failing_raises_here_and_returns_nothing_there():
    with pytest.raises(RuntimeError, match="Failed to process any formulas"):
        calculate_xray_properties(["Xx1", "Yy2"], ENERGIES, [1.0, 1.0])
    assert (
        calculate_multiple_xray_properties(["Xx1", "Yy2"], ENERGIES, [1.0, 1.0]) == {}
    )


def test_multiple_materials_skip_log_and_keep_the_keys():
    formulas = ["SiO2", "Xx999", "SiO2", "SiO2"]
    densities = [2.2, 1.0, 2.5, 2.5]
    with _logged(core._logger, logging.WARNING) as records:
        got = calculate_multiple_xray_properties(formulas, ENERGIES, densities)
    assert list(got) == ["SiO2", "SiO2@2.500", "SiO2@2.500#2"]
    assert [r.formula for r in records] == ["Xx999"]  # type: ignore[attr-defined]
    single = calculate_single_material_properties("SiO2", ENERGIES, 2.5)
    _same_bits(got["SiO2@2.500"]["dispersion"], single.dispersion_delta)


def test_a_bad_energy_fails_every_formula_of_the_multiple_call():
    with _logged(core._logger, logging.WARNING) as records:
        got = calculate_multiple_xray_properties(["SiO2", "Si"], [0.001], [2.2, 2.33])
    assert got == {}
    assert len(records) == 2


def test_two_hundred_formulas_compile_at_most_one_program_and_transfer_once_per_block(
    spy,
):
    core.warm_up(("SiO2",))
    spy.clear()
    formulas = [f"Si{'O2' if i % 2 else 'C'}" for i in range(200)]
    densities = [1.0 + i / 100 for i in range(200)]
    energies = np.linspace(1.0, 29.0, 200)
    with record_compiles() as names:
        runtime.reset_transfer_counts()
        got = calculate_xray_properties(formulas, energies, densities)
        counts = runtime.transfer_counts()
    programs = [n for n in names if n == "_optical_program"]
    assert len(programs) <= 1, names
    assert len(got) == 200
    # 256 padded materials x 2 elements x 216 energies is far below the block budget
    blocks = len(spy)
    assert blocks == 1
    assert counts == {"upload": blocks, "fetch_status": 0, "fetch": blocks}
    # warm: the same run again compiles nothing
    with record_compiles() as names:
        calculate_xray_properties(formulas, energies, densities)
    assert "_optical_program" not in names


def test_the_compile_recorder_sees_a_fresh_shape():
    """Positive control for the budget test: a never-used (F, K, M) shape compiles."""
    with record_compiles() as names:
        calculate_xray_properties(
            ["Si", "SiO2", "H2O", "Au", "Fe2O3"],
            np.linspace(1.0, 20.0, 1000),
            [1.0] * 5,
        )
    assert names.count("_optical_program") == 1, names


def test_max_block_bounds_the_materials_per_block(spy):
    encoded = [optics.encode_formula("SiO2")] * 70
    got = host.run_optical_blocks(encoded, [2.2] * 70, ENERGIES, max_block=16)
    assert [c["f"] for c in spy] == [16, 16, 16, 16, 6]
    assert got["out"].shape == (70, len(optics.OUT_FIELDS), ENERGIES.size)
    assert got["ok"].all()


def test_a_block_is_one_upload_and_one_fetch(spy):
    encoded = [optics.encode_formula("SiO2"), optics.encode_formula("Si")]
    host.run_optical_blocks(encoded, [2.2, 2.33], ENERGIES)  # warm
    runtime.reset_transfer_counts()
    host.run_optical_blocks(encoded, [2.2, 2.33], ENERGIES)
    assert runtime.transfer_counts() == {"upload": 1, "fetch_status": 0, "fetch": 1}


def test_batch_keys_values_and_order_with_a_failing_item_in_the_middle():
    formulas = ["SiO2", "Xx999", "SiO2", "Al2O3", "SiO2"]
    densities = [2.2, 1.0, 2.2, 3.95, 2.5]
    with pytest.warns(UserWarning, match="Failed to process formula 'Xx999'"):
        got = calculate_batch_properties(
            formulas, ENERGIES, densities, BatchConfig(enable_progress=False)
        )
    assert list(got) == [
        "SiO2@2.200",
        "Xx999",
        "SiO2@2.200#2",
        "Al2O3@3.950",
        "SiO2@2.500",
    ]
    assert got["Xx999"] is None
    ref = calculate_single_material_properties("Al2O3", ENERGIES, 3.95)
    _same_bits(got["Al2O3@3.950"].dispersion_delta, ref.dispersion_delta)


def test_a_chunk_with_different_energies_per_item_follows_its_own_energies():
    e1, e2 = np.linspace(3.0, 9.0, 20), np.linspace(10.0, 20.0, 30)
    chunk = [("SiO2", e1, 2.2), ("Au", e2, 19.32), ("Si", e1, 2.33), ("Au", e1, 19.32)]
    got = process_batch_chunk(chunk, BatchConfig(enable_progress=False))
    assert [f for f, _ in got] == ["SiO2", "Au", "Si", "Au"]
    for (formula, energies, density), (_, result) in zip(chunk, got, strict=True):
        assert result is not None
        np.testing.assert_array_equal(result.energy_kev, energies)
        ref = calculate_single_material_properties(formula, energies, density)
        _same_bits(result.dispersion_delta, ref.dispersion_delta)


def test_chunk_size_bounds_the_blocks_and_progress_counts_every_item(spy):
    class Bar:
        def __init__(self) -> None:
            self.updates: list[int] = []

        def update(self, n: int) -> None:
            self.updates.append(n)

    bar = Bar()
    formulas = ["SiO2"] * 9 + ["Xx999"]
    data = [(f, ENERGIES, 2.2 + i / 10) for i, f in enumerate(formulas)]
    with pytest.warns(UserWarning):
        got = bp._process_chunks(
            data, BatchConfig(chunk_size=5, enable_progress=False), bar
        )
    assert len(got) == 10
    assert sum(bar.updates) == 10
    # chunks of 5 and 5 (one item fails the screening): blocks of 5 and 4
    assert sorted(c["f"] for c in spy) == [4, 5]


def test_max_workers_is_accepted_ignored_and_logged_once_at_debug():
    config = BatchConfig(enable_progress=False, max_workers=4)
    with _logged(bp._logger, logging.DEBUG) as records:
        got = calculate_batch_properties(["SiO2", "Si"], ENERGIES, [2.2, 2.33], config)
    assert list(got) == ["SiO2@2.200", "Si@2.330"]
    assert len([r for r in records if "max_workers" in r.getMessage()]) == 1


def test_a_binding_budget_splits_a_group_and_bounds_the_padded_blocks(spy, monkeypatch):
    """Spec section 3: the padded ``F·K·M`` of every block stays within the budget.

    The default budget is far too large to bind in a unit test, so ``block_for`` is
    patched to a small ``elements`` (patching ``CHUNK_ELEMENTS`` does nothing: it is a
    default argument). 700 rows fit the budget at the 64-energy bucket, which makes
    ``block_for`` return the non-power-of-two rung 640; padding that to 1024 rows
    would overshoot.
    """
    budget = 700 * 2 * 64
    n = 1000
    formulas = ["SiO2"] * n
    densities = [1.0 + i / 1000 for i in range(n)]
    whole = calculate_xray_properties(formulas, ENERGIES, densities)
    assert len(spy) == 1
    spy.clear()
    monkeypatch.setattr(
        host, "block_for", functools.partial(shapes.block_for, elements=budget)
    )
    split = calculate_xray_properties(formulas, ENERGIES, densities)
    assert len(spy) > 1
    assert sum(c["f"] for c in spy) == n
    for call in spy:
        assert bucket_pow2(call["f"]) * call["k_bucket"] * bucket(50, 64) <= budget
    assert list(split) == list(whole)
    for key, result in whole.items():
        for name in FIELDS:
            _same_bits(getattr(split[key], name), getattr(result, name))


def test_progress_counts_a_non_finite_item_once():
    """Screening failure + non-finite device result + good items: total == len(chunk)."""

    class Bar:
        total = 0

        def update(self, n: int) -> None:
            self.total += n

    bar = Bar()
    chunk = [
        ("SiO2", ENERGIES, 2.2),
        ("SiO2", ENERGIES, 1e308),  # passes screening, overflows on the device
        ("Xx999", ENERGIES, 1.0),  # fails screening
        ("Si", ENERGIES, 2.33),
    ]
    with pytest.warns(UserWarning) as caught:
        got = process_batch_chunk(chunk, BatchConfig(enable_progress=False), bar)
    texts = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("Non-finite values" in t for t in texts)
    assert [r is None for _, r in got] == [False, True, True, False]
    assert bar.total == len(chunk)


def test_comparator_takes_all_materials_from_one_call(spy):
    from xraylabtool.analysis.comparator import MaterialComparator

    formulas = ["SiO2", "Au", "SiO2", "CaCO3"]
    densities = [2.2, 19.32, 2.5, 2.71]
    data = MaterialComparator._calculate_material_data(formulas, densities, [8.0, 12.0])
    assert list(data) == [f"{f} ({d} g/cm³)" for f, d in zip(formulas, densities)]
    assert {c["k_bucket"] for c in spy} == {2, 4}
    assert sum(c["f"] for c in spy) == 4  # no retry loop on success
    ref = calculate_single_material_properties("SiO2", [8.0, 12.0], 2.5)
    _same_bits(data["SiO2 (2.5 g/cm³)"].dispersion_delta, ref.dispersion_delta)


def test_comparator_fallback_names_the_first_failing_formula():
    from xraylabtool.analysis.comparator import MaterialComparator

    with pytest.raises(
        ValueError,
        match=r"^Failed to calculate properties for Xx9: "
        r"Failed to process any formulas successfully$",
    ):
        MaterialComparator._calculate_material_data(
            ["SiO2", "Xx9", "Au"], [2.2, 1.0, 19.32], [8.0, 12.0]
        )
    with pytest.raises(ValueError, match=r"^Failed to calculate properties for Xx9: "):
        MaterialComparator._calculate_material_data(["Xx9"], [1.0], [8.0])
