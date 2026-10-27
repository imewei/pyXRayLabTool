"""TableInterpolator, the scattering-factor helpers and the atomic provider on the tables."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from xraylabtool.backend import runtime
from xraylabtool.calculators import cache, host, optics
from xraylabtool.calculators.cache import (
    TableInterpolator,
    create_scattering_factor_interpolators,
)
from xraylabtool.calculators.derived_quantities import (
    calculate_derived_quantities,
    calculate_scattering_factors,
)
import xraylabtool.calculators.tables as tables_module
from xraylabtool.constants import ENERGY_TO_WAVELENGTH_FACTOR
from xraylabtool.data_handling.atomic_cache import get_atomic_data_provider

E_KEV = np.array([0.5, 1.0, 8.0, 20.0])


def _budget(call) -> tuple[int, int]:
    before = runtime.thread_transfer_counts()
    result = call()
    after = runtime.thread_transfer_counts()
    return result, (
        after["upload"] - before["upload"],
        after["fetch"] - before["fetch"],
    )


def _material(*symbols_counts):
    return [
        (count, *create_scattering_factor_interpolators(symbol))
        for symbol, count in symbols_counts
    ]


def test_factory_returns_table_interpolators_of_one_row():
    f1, f2 = create_scattering_factor_interpolators("Si")
    assert isinstance(f1, TableInterpolator)
    assert isinstance(f2, TableInterpolator)
    assert (f1.z, f2.z) == (13, 13)
    assert (f1.field, f2.field) == (0, 1)
    assert create_scattering_factor_interpolators("Si") == (f1, f2)  # cached


def test_interpolator_accepts_scalars_host_and_device_arrays():
    f1, _ = create_scattering_factor_interpolators("Si")
    host_q = np.array([100.0, 1000.0, 10000.0])
    from_host = np.asarray(f1(host_q))
    assert from_host.dtype == np.float64
    assert from_host.shape == (3,)
    with jax.enable_x64(True):
        device_q = jax.numpy.asarray(host_q)
    np.testing.assert_array_equal(np.asarray(f1(device_q)), from_host)
    assert np.asarray(f1(100.0)).shape == ()
    assert np.asarray(f1(100.0)) == from_host[0]


def test_interpolator_crosses_the_boundary_only_through_runtime_upload():
    f1, _ = create_scattering_factor_interpolators("Si")
    q = np.array([100.0, 1000.0, 10000.0])
    f1(q)  # warm
    with jax.enable_x64(True):
        device_q = runtime.upload(q)
    with jax.transfer_guard("disallow_explicit"):
        _, from_host = _budget(lambda: f1(q))
        _, from_scalar = _budget(lambda: f1(100.0))
        _, from_device = _budget(lambda: f1(device_q))
    assert from_host == (1, 0)
    assert from_scalar == (1, 0)
    assert from_device == (1, 0)  # only the row index


def test_interpolator_rejects_host_tables(monkeypatch):
    f1, _ = create_scattering_factor_interpolators("Si")
    monkeypatch.setattr(
        host,
        "load_tables",
        lambda: tables_module.ScatteringTables(*(np.zeros(1),) * 6, ()),
    )
    with pytest.raises(TypeError, match="device arrays"):
        f1(100.0)


def test_interpolator_is_nan_outside_the_table():
    f1, f2 = create_scattering_factor_interpolators("H")
    outside = np.array([1.0, 5.0e6])  # eV: below and above the H table
    assert np.isnan(np.asarray(f1(outside))).all()
    assert np.isnan(np.asarray(f2(outside))).all()


def test_insufficient_data_keeps_the_value_error(monkeypatch):
    from xraylabtool.calculators import scattering_data

    class OnePoint:
        def __len__(self) -> int:
            return 1

    create_scattering_factor_interpolators.cache_clear()
    cache._interpolator_cache.clear()
    monkeypatch.setattr(
        scattering_data, "load_scattering_factor_data", lambda element: OnePoint()
    )
    try:
        with pytest.raises(
            ValueError, match="Insufficient data points for element 'Si'"
        ):
            create_scattering_factor_interpolators("Si")
    finally:
        create_scattering_factor_interpolators.cache_clear()
        cache._interpolator_cache.clear()


def test_element_without_data_keeps_file_not_found():
    with pytest.raises(FileNotFoundError, match="'Pu'"):
        create_scattering_factor_interpolators("Pu")


def test_scattering_factors_accept_factory_entries_in_one_program_call():
    data = _material(("Si", 1.0), ("O", 2.0))
    wavelength = ENERGY_TO_WAVELENGTH_FACTOR / E_KEV
    calculate_scattering_factors(E_KEV * 1000.0, wavelength, 2.2, 60.08, data)  # warm
    out, spent = _budget(
        lambda: calculate_scattering_factors(
            E_KEV * 1000.0, wavelength, 2.2, 60.08, data
        )
    )
    assert spent == (1, 1)
    assert len(out) == 4
    for array in out:
        assert type(array) is np.ndarray
        assert array.dtype == np.float64
        assert array.shape == E_KEV.shape
        assert array.flags.writeable
    # f totals are the count-weighted sums of the element interpolators
    f1_si, f2_si = (np.asarray(f(E_KEV * 1000.0)) for f in data[0][1:])
    f1_o, f2_o = (np.asarray(f(E_KEV * 1000.0)) for f in data[1][1:])
    np.testing.assert_allclose(out[2], f1_si + 2.0 * f1_o, rtol=1e-13)
    np.testing.assert_allclose(out[3], f2_si + 2.0 * f2_o, rtol=1e-13)


def test_scattering_factors_reject_an_arbitrary_callable_without_calling_it():
    calls = []

    def rogue(energy):
        calls.append(energy)
        return np.ones_like(energy)

    wavelength = ENERGY_TO_WAVELENGTH_FACTOR / E_KEV
    with pytest.raises(TypeError, match="TableInterpolator"):
        calculate_scattering_factors(
            E_KEV * 1000.0, wavelength, 2.2, 28.0, [(1.0, rogue, rogue)]
        )
    mixed = [*_material(("Si", 1.0)), (2.0, rogue, rogue)]
    with pytest.raises(TypeError, match="TableInterpolator"):
        calculate_scattering_factors(E_KEV * 1000.0, wavelength, 2.2, 60.0, mixed)
    assert calls == []


def test_scattering_factors_reject_a_pair_of_two_elements():
    f1_si, _ = create_scattering_factor_interpolators("Si")
    _, f2_o = create_scattering_factor_interpolators("O")
    wavelength = ENERGY_TO_WAVELENGTH_FACTOR / E_KEV
    with pytest.raises(ValueError, match="one element"):
        calculate_scattering_factors(
            E_KEV * 1000.0, wavelength, 2.2, 44.0, [(1.0, f1_si, f2_o)]
        )


def test_scattering_factors_without_elements_are_zero_arrays():
    out = calculate_scattering_factors(E_KEV * 1000.0, E_KEV, 2.2, 1.0, [])
    assert len(out) == 4
    for array in out:
        np.testing.assert_array_equal(array, np.zeros(E_KEV.size))
    assert len({id(a) for a in out}) == 4


def test_derived_quantities_are_one_program_call_and_clamped():
    wavelength = np.array([1.5e-10, 1.0e-10])
    delta = np.array([-1e-6, 2e-6])
    beta = np.array([0.0, 1e-8])
    calculate_derived_quantities(wavelength, delta, beta, 2.2, 60.0, 30.0)  # warm
    out, spent = _budget(
        lambda: calculate_derived_quantities(wavelength, delta, beta, 2.2, 60.0, 30.0)
    )
    assert spent == (1, 1)
    electron_density, critical, attenuation, re_sld, im_sld = out
    assert isinstance(electron_density, float)
    assert critical[0] == 0.0  # delta < 0: no total external reflection
    assert np.isfinite(attenuation).all()  # beta = 0 is clamped at 1e-30
    assert re_sld[0] < 0.0 < re_sld[1]
    assert im_sld[0] == 0.0
    for array in (critical, attenuation, re_sld, im_sld):
        assert type(array) is np.ndarray
        assert array.shape == wavelength.shape


def test_derived_quantities_keep_the_scalar_electron_density_expression():
    out = calculate_derived_quantities(
        np.array([1.5e-10]), np.array([1e-6]), np.array([1e-8]), 2.2, 60.0, 30.0
    )
    from xraylabtool.constants import AVOGADRO

    expected = float(1e6 * 2.2 / 60.0 * AVOGADRO * 30.0 / 1e30)
    assert out[0] == pytest.approx(expected, rel=1e-14)


def test_atomic_provider_fills_complex128_from_the_tables_in_one_program_call():
    provider = get_atomic_data_provider()
    provider.get_scattering_factors("Si", E_KEV)  # warm
    out, spent = _budget(lambda: provider.get_scattering_factors("Si", E_KEV))
    assert spent == (1, 1)
    assert out.dtype == np.complex128
    assert out.shape == E_KEV.shape
    f1, f2 = create_scattering_factor_interpolators("Si")
    np.testing.assert_allclose(
        out, np.asarray(f1(E_KEV * 1000.0)) + 1j * np.asarray(f2(E_KEV * 1000.0))
    )


@pytest.mark.parametrize(
    "energies",
    [8.0, np.array(8.0), [8.0, 9.0], E_KEV, np.array([[8.0, 9.0]]), np.ones((2, 3))],
    ids=["float", "0d", "list", "1d", "2d_row", "2d_grid"],
)
def test_atomic_provider_keeps_the_shape_of_its_energies(energies):
    out = get_atomic_data_provider().get_scattering_factors("Si", energies)
    shape = np.asarray(energies).shape
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.complex128
    assert out.shape == shape
    f1, f2 = create_scattering_factor_interpolators("Si")
    ev = np.asarray(energies, dtype=np.float64) * 1000.0
    np.testing.assert_allclose(
        out, np.asarray(f1(ev)) + 1j * np.asarray(f2(ev)), rtol=1e-14
    )


def test_scattering_length_density_of_scalars_returns_floats():
    from xraylabtool.calculators.derived_quantities import (
        calculate_scattering_length_density,
    )

    real_sld, imag_sld = calculate_scattering_length_density(1.5e-6, 2.5e-8, 1.54)
    assert isinstance(real_sld, np.floating)
    assert isinstance(imag_sld, np.floating)
    assert real_sld == pytest.approx(2 * np.pi * 1.5e-6 / 1.54**2, rel=1e-14)
    assert imag_sld == pytest.approx(2 * np.pi * 2.5e-8 / 1.54**2, rel=1e-14)


def test_atomic_provider_unknown_element_keeps_its_error():
    provider = get_atomic_data_provider()
    with pytest.raises(FileNotFoundError):
        provider.get_scattering_factors("Xx", np.array([8.0]))


def test_helpers_keep_their_standalone_edges():
    from xraylabtool.calculators.derived_quantities import (
        calculate_attenuation_length,
        calculate_critical_angle,
        calculate_transmission,
    )

    assert np.isnan(calculate_critical_angle(np.array([-1e-6]))).all()
    assert np.isposinf(
        calculate_attenuation_length(np.array([1.0]), np.array([0.0]))
    ).all()
    np.testing.assert_allclose(
        calculate_transmission(1.0, np.array([1.0, np.inf])), [np.exp(-1.0), 1.0]
    )
    scalar = calculate_critical_angle(1e-6)
    assert np.ndim(scalar) == 0
    assert scalar == pytest.approx(np.degrees(np.sqrt(2e-6)), rel=1e-14)


_PADDING_ENERGIES = np.linspace(1000.0, 20000.0, 65)  # 64 and 65 fall in two M buckets


def test_padding_does_not_change_a_bit_of_the_pchip_programs():
    """64 and 65 energies are in different M buckets; shared entries must match."""
    from xraylabtool.backend.shapes import bucket

    assert bucket(64, minimum=64) != bucket(65, minimum=64)
    e64, e65 = _PADDING_ENERGIES[:64], _PADDING_ENERGIES
    wl64, wl65 = (
        ENERGY_TO_WAVELENGTH_FACTOR / (e64 / 1000),
        ENERGY_TO_WAVELENGTH_FACTOR / (e65 / 1000),
    )
    z, counts = [13, 7], [1.0, 2.0]

    short = host.run_scattering(z, counts, e64, wl64, 1.2e-5)
    long = host.run_scattering(z, counts, e65, wl65, 1.2e-5)
    np.testing.assert_array_equal(long[:, :64], short)

    np.testing.assert_array_equal(
        host.run_rows(13, e65)[:, :64], host.run_rows(13, e64)
    )

    ed64, out64 = host.run_derived(
        wl64, np.full(64, 1e-6), np.full(64, 1e-8), 2.2, 60.0, 30.0
    )
    ed65, out65 = host.run_derived(
        wl65, np.full(65, 1e-6), np.full(65, 1e-8), 2.2, 60.0, 30.0
    )
    assert ed64 == ed65
    np.testing.assert_array_equal(out65[:, :64], out64)

    mapped64 = host.run_map(
        optics.critical_angle_from_delta, 1e-6 * np.arange(1.0, 65.0)
    )
    mapped65 = host.run_map(
        optics.critical_angle_from_delta, 1e-6 * np.arange(1.0, 66.0)
    )
    np.testing.assert_array_equal(mapped65[:64], mapped64)
