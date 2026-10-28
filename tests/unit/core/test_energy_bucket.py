"""The core energy axis pads to coarse buckets, so few lengths mean few compiles."""

from __future__ import annotations

import numpy as np
import pytest

from tests.fixtures.kk_compiles import record_compiles
from xraylabtool.backend.shapes import bucket, bucket_energy
from xraylabtool.calculators import host, optics
from xraylabtool.calculators.optics import encode_formula


def test_powers_of_two_up_to_4096_then_the_ladder():
    assert [bucket_energy(n) for n in (0, 1, 64, 65, 500, 512, 513, 4096)] == [
        64,
        64,
        64,
        128,
        512,
        512,
        1024,
        4096,
    ]
    assert bucket_energy(4097) == bucket(4097, minimum=4096) == 5120


def test_a_bucket_is_never_smaller_than_the_count_and_wastes_at_most_a_factor_two():
    for n in [*list(range(1, 4200)), 10000, 123457]:
        b = bucket_energy(n)
        assert b >= n
        assert b <= max(64, 2 * n)


def test_above_4096_the_padding_stays_within_the_ladder_bound():
    for n in [*range(4097, 20001, 7), 123457]:
        assert n <= bucket_energy(n) <= 1.25 * n + 8


def test_few_rungs_cover_the_whole_small_range():
    assert len({bucket_energy(n) for n in range(1, 4097)}) == 7


def test_a_negative_count_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        bucket_energy(-1)


def test_two_lengths_in_one_bucket_share_a_compiled_program():
    optics._optical_program.clear_cache()  # the control must not depend on test order
    material = [encode_formula("SiO2")]
    host.run_optical(material, [2.2], np.linspace(5, 15, 300))  # warm
    with record_compiles() as names:
        host.run_optical(material, [2.2], np.linspace(5, 15, 500))
    assert [n for n in names if "_optical_program" in n] == []
    with record_compiles() as names:  # control: the next bucket does compile
        host.run_optical(material, [2.2], np.linspace(5, 15, 700))
    assert any("_optical_program" in n for n in names), names


def test_warm_up_precompiles_the_requested_energy_buckets():
    from xraylabtool.calculators import core

    optics._optical_program.clear_cache()
    optics._tail_program.clear_cache()
    with record_compiles() as names:
        core.warm_up(("SiO2",), energy_counts=(1, 3000))
    assert sum("_optical_program" in n for n in names) == 2, names
    assert sum("_tail_program" in n for n in names) == 2, names
    material = [encode_formula("SiO2")]
    with record_compiles() as names:
        host.run_optical(material, [2.2], np.linspace(5, 15, 2500))  # bucket 4096
        host.run_optical(material, [2.2], np.linspace(5, 15, 40))  # bucket 64
        host.run_tail(
            [1.0], [1.0], [1.0], np.zeros((1, 40)), np.zeros((1, 40)), np.ones(40)
        )
        host.run_tail(
            [1.0], [1.0], [1.0], np.zeros((1, 2500)), np.zeros((1, 2500)), np.ones(2500)
        )
    assert [n for n in names if "_program" in n] == []


@pytest.mark.parametrize("bad", [0, -3])
def test_warm_up_rejects_a_non_positive_energy_count(bad):
    from xraylabtool.calculators import core

    with pytest.raises(ValueError, match="energy_counts"):
        core.warm_up(("SiO2",), energy_counts=(bad,))


def test_warm_up_rejects_any_non_positive_count_in_a_sequence():
    from xraylabtool.calculators import core

    with pytest.raises(ValueError, match="energy_counts"):
        core.warm_up(("SiO2",), energy_counts=(1, 0))


@pytest.mark.parametrize("bad", [2.5, 1.9, "5"])
def test_warm_up_rejects_a_non_integer_energy_count(bad):
    from xraylabtool.calculators import core

    with pytest.raises(TypeError):
        core.warm_up(("SiO2",), energy_counts=(bad,))


def test_warm_up_compiles_one_program_for_counts_sharing_a_bucket():
    from xraylabtool.calculators import core

    optics._optical_program.clear_cache()
    with record_compiles() as names:
        core.warm_up(("SiO2",), energy_counts=(3000, 4000))
    assert sum("_optical_program" in n for n in names) == 1, names
