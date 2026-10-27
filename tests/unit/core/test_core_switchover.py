"""The public calculators run on the device program (transfer budget, x64, warm-up)."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import jax
import numpy as np
import pytest

from tests.fixtures.kk_compiles import record_compiles
from xraylabtool.backend import runtime
from xraylabtool.calculators import core, optics
from xraylabtool.calculators.cache import clear_scattering_factor_cache
from xraylabtool.calculators.core import (
    calculate_properties_from_factors,
    calculate_single_material_properties,
    get_cached_elements,
    is_element_cached,
)
import xraylabtool.calculators.tables as tables_module

ENERGIES = np.linspace(1.0, 20.0, 50)
PACKAGE = Path(core.__file__).resolve().parents[1]


def _budget(call) -> tuple[int, int]:
    before = runtime.thread_transfer_counts()
    call()
    after = runtime.thread_transfer_counts()
    return after["upload"] - before["upload"], after["fetch"] - before["fetch"]


def test_single_material_call_is_one_upload_and_one_fetch():
    calculate_single_material_properties("SiO2", ENERGIES, 2.2)  # warm
    assert _budget(
        lambda: calculate_single_material_properties("SiO2", ENERGIES, 2.2)
    ) == (1, 1)


def test_from_factors_call_is_one_upload_and_one_fetch():
    f = np.ones(ENERGIES.size)
    calculate_properties_from_factors("SiO2", ENERGIES, 2.2, f, f)  # warm
    assert _budget(
        lambda: calculate_properties_from_factors("SiO2", ENERGIES, 2.2, f, f)
    ) == (1, 1)


def test_public_calls_run_under_an_outer_explicit_transfer_guard():
    f = np.ones(ENERGIES.size)
    calculate_single_material_properties("SiO2", ENERGIES, 2.2)
    calculate_properties_from_factors("SiO2", ENERGIES, 2.2, f, f)
    with jax.transfer_guard("disallow_explicit"):
        single = calculate_single_material_properties("SiO2", ENERGIES, 2.2)
        tail = calculate_properties_from_factors("SiO2", ENERGIES, 2.2, f, f)
    assert np.isfinite(single.dispersion_delta).all()
    assert np.isfinite(tail.dispersion_delta).all()


_NO_ENV_PROBE = """
import numpy as np
import os
import xraylabtool

os.environ.pop("JAX_ENABLE_X64", None)  # after the package import, before jax
import jax

assert not jax.config.jax_enable_x64
f = np.ones(3)
for res in (
    xraylabtool.calculate_single_material_properties("SiO2", [8.0, 9.0], 2.2),
    xraylabtool.calculators.core.calculate_properties_from_factors(
        "SiO2", np.array([8.0, 9.0, 10.0]), 2.2, f, f
    ),
):
    assert res.dispersion_delta.dtype == np.float64, res.dispersion_delta.dtype
    assert res.energy_kev.dtype == np.float64
ed, ca, al, re, im = xraylabtool.calculate_derived_quantities(
    np.array([1.5e-10]), np.array([1e-6]), np.array([1e-8]), 2.2, 60.0, 30.0
)
assert ca.dtype == np.float64 and isinstance(ed, float)
xraylabtool.warm_up(("SiO2",))
print("OK")
"""


def test_public_entries_give_float64_without_the_environment_variable():
    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    done = subprocess.run(
        [sys.executable, "-c", _NO_ENV_PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().endswith("OK")


def test_result_arrays_are_fresh_writable_numpy():
    energies = np.array([8.0, 9.0])
    result = calculate_single_material_properties("SiO2", energies, 2.2)
    arrays = [
        result.energy_kev,
        result.wavelength_angstrom,
        result.dispersion_delta,
        result.absorption_beta,
        result.scattering_factor_f1,
        result.scattering_factor_f2,
        result.critical_angle_degrees,
        result.attenuation_length_cm,
        result.real_sld_per_ang2,
        result.imaginary_sld_per_ang2,
    ]
    for array in arrays:
        assert type(array) is np.ndarray
        assert array.dtype == np.float64
        assert array.flags.writeable
        assert array.flags.owndata  # no view into one shared download buffer
    assert not np.shares_memory(result.energy_kev, energies)
    result.energy_kev[0] = -1.0
    assert energies[0] == 8.0
    assert isinstance(result.electron_density_per_ang3, float)
    assert isinstance(result.molecular_weight_g_mol, float)


def test_element_with_atomic_data_but_no_table_keeps_file_not_found():
    # Pu has atomic data but no scattering table; the pre-device path raised this.
    with pytest.raises(FileNotFoundError, match="'Pu'"):
        calculate_single_material_properties("PuO2", 8.0, 2.2)


def test_calculation_fills_the_element_cache_that_get_cached_elements_reports():
    clear_scattering_factor_cache()
    assert get_cached_elements() == []
    calculate_single_material_properties("SiO2", ENERGIES, 2.2)
    assert sorted(get_cached_elements()) == ["O", "Si"]
    assert is_element_cached("Si")


def test_non_finite_status_raises_the_legacy_value_error():
    with pytest.raises(ValueError, match=r"^Non-finite values \(NaN or Inf\) detected"):
        calculate_properties_from_factors(
            "SiO2", np.array([8.0]), 1e300, np.array([1e300]), np.array([1.0])
        )


def test_clear_cache_drops_the_resident_tables_and_the_next_call_reloads_them():
    calculate_single_material_properties("SiO2", ENERGIES, 2.2)
    assert tables_module._tables is not None
    clear_scattering_factor_cache()
    assert tables_module._tables is None
    calculate_single_material_properties("SiO2", ENERGIES, 2.2)
    assert tables_module._tables is not None


def test_warm_up_precompiles_the_default_bucket_programs():
    import xraylabtool

    assert xraylabtool.warm_up is core.warm_up
    # An element bucket no other test uses: 10 distinct elements -> K = 16.
    formula = "HLiBeBCNOFNaMg"
    optics._optical_program.clear_cache()  # independent of what ran before
    with record_compiles() as names:
        xraylabtool.warm_up((formula,))
    assert "_optical_program" in names, names
    with record_compiles() as names:
        calculate_single_material_properties(formula, ENERGIES, 1.5)
    assert [n for n in names if n in {"_optical_program", "_tail_program"}] == []


def test_the_first_call_after_a_cache_clear_runs_the_smart_warm_up():
    from xraylabtool.calculators import cache

    clear_scattering_factor_cache()
    assert not cache._CACHE_WARMED
    calculate_single_material_properties("SiO2", ENERGIES, 2.2)
    assert cache._CACHE_WARMED


def test_core_modules_do_not_use_the_backend_twin():
    for relative in (
        "calculators/core.py",
        "calculators/cache.py",
        "calculators/derived_quantities.py",
        "calculators/scattering_data.py",
        "data_handling/atomic_cache.py",
    ):
        source = (PACKAGE / relative).read_text()
        for needle in ("backend import ops", "array_ops", "InterpolationFactory"):
            assert needle not in source, (relative, needle)
