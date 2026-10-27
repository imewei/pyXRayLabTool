"""Pure device optics (spec section 4/5, plan step 1.4).

The Task 0 goldens (frozen from the NumPy path) are the oracle: each family is
recomputed through the thin harness of ``tests/fixtures/core_harness.py``
(``encode_formula`` + the device program) and compared at ``rtol=1e-12``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.characterization.core import cases
from tests.characterization.core.test_core_golden import (
    _inputs,
    _npz,
    compare_arrays,
)
from tests.fixtures.core_harness import GOLDEN_FIELDS, run_program, run_tail
from tests.fixtures.core_rounding import assert_ulp_close
from xraylabtool.backend import runtime
from xraylabtool.calculators import optics
from xraylabtool.calculators.tables import load_tables
from xraylabtool.constants import AVOGADRO

SEP = cases.SEP


@pytest.fixture(autouse=True, scope="module")
def _x64():
    runtime.ensure_x64()


def _assert_golden(key: str, actual: Any) -> None:
    problem, _ = compare_arrays(np.asarray(actual, dtype=np.float64), _npz()[key])
    assert problem is None, f"{key}: {problem}"


def _assert_material(prefix: str, run: dict[str, Any], i: int, mw, ne, rho) -> None:
    """Compare the material at row ``i`` with the golden entries under ``prefix``."""
    _assert_golden(f"{prefix}{SEP}molecular_weight_g_mol", mw)
    _assert_golden(f"{prefix}{SEP}total_electrons", ne)
    _assert_golden(f"{prefix}{SEP}density_g_cm3", rho)
    _assert_golden(
        f"{prefix}{SEP}electron_density_per_ang3", run["electron_density"][i]
    )
    for row, name in enumerate(GOLDEN_FIELDS):
        _assert_golden(f"{prefix}{SEP}{name}", run["out"][i, row])


def _check_formulas(
    family: str,
    materials: list[tuple[str, float]],
    energies: Any,
    keys: list[str],
) -> None:
    run = run_program(materials, energies)
    assert run["ok"].all()
    for i, ((_, rho), key) in enumerate(zip(materials, keys, strict=True)):
        _assert_material(
            f"{family}{SEP}{key}", run, i, run["mw"][i], run["n_electrons"][i], rho
        )


# ------------------------------------------------------------------ goldens


@pytest.mark.parametrize("grid", cases.GRID_NAMES)
def test_single_material_grids_match_golden(grid):
    _check_formulas(
        f"single_{grid}",
        list(cases.FORMULAS),
        _inputs()[f"grid_{grid}"],
        [f for f, _ in cases.FORMULAS],
    )


@pytest.mark.parametrize(
    ("label", "energy"), [("e8", 8.0), ("lo", 0.03), ("hi", 30.0), ("cu_ka", 8.048)]
)
def test_single_material_scalar_energies_match_golden(label, energy):
    _check_formulas(
        "single_scalar",
        list(cases.FORMULAS),
        energy,
        [f"{label}{SEP}{f}" for f, _ in cases.FORMULAS],
    )


@pytest.mark.parametrize("grid", ["lin200", "absedges60"])
def test_every_element_matches_golden(grid):
    materials = [
        (symbol, cases.element_density(z))
        for z, symbol in enumerate(cases.SYMBOLS, start=1)
    ]
    _check_formulas(
        "single_elements",
        materials,
        _inputs()[f"grid_{grid}"],
        [f"{grid}{SEP}{s}" for s, _ in materials],
    )


@pytest.mark.parametrize("grid", ["lin200", "absedges60", "unsorted200", "three"])
def test_interpolated_f_totals_match_scattering_factor_golden(grid):
    e_kev = _inputs()[f"grid_{grid}"]
    tables = load_tables()
    with jax.enable_x64(True):
        interpolate = jax.jit(optics.interpolate_f)
        for formula, _ in cases.FORMULAS_SMALL:
            z, counts, _, _ = optics.encode_formula(formula)
            f = interpolate(
                tables, jnp.asarray(z), jnp.asarray(counts), jnp.asarray(e_kev * 1e3)
            )
            _assert_golden(
                f"scattering_factors{SEP}{formula}{SEP}{grid}{SEP}f1_total", f[0]
            )
            _assert_golden(
                f"scattering_factors{SEP}{formula}{SEP}{grid}{SEP}f2_total", f[1]
            )


def test_from_factors_through_tail_program_matches_golden():
    inp = _inputs()
    e, f1, f2 = inp["ff_energy"], inp["ff_f1"], inp["ff_f2"]
    assert (f1 < 0).any()
    runs = [
        (f"smooth{SEP}{f}", f, d, e, f1, f2) for f, d in cases.FORMULAS_SMALL
    ] + [
        (
            "unsorted", "SiO2", 2.2, inp["ff_energy_unsorted"],
            inp["ff_f1_unsorted"], inp["ff_f2_unsorted"],
        ),
        (
            "wide_range_zero_f2", "Si", 2.33, inp["ff_energy_wide"],
            inp["ff_f1_wide"], inp["ff_f2_wide"],
        ),
        ("single_point_negative", "H2OH2", 1.0, np.array([8.0]),
         np.array([-5.0]), np.array([-0.1])),
        ("lists_int_density", "Au", 19, np.array([8.0, 9.0]),
         np.array([70.0, 71.0]), np.array([5.0, 4.0])),
    ]  # fmt: skip
    for key, formula, rho, en, a, b in runs:
        run = run_tail(formula, rho, en, a, b)
        _, _, mw, ne = optics.encode_formula(formula)
        _assert_material(f"from_factors{SEP}{key}", run, 0, mw, ne, float(rho))


def test_derived_quantities_golden_pins_clamped_rules():
    inp = _inputs()
    lam, delta, beta = inp["dq_wavelength_m"], inp["dq_delta"], inp["dq_beta"]
    assert (delta < 0).any()
    assert (beta <= 0).any()
    materials = {"SiO2": (2.2, 60.0843, 30.0), "Au": (19.32, 196.96657, 79.0)}
    with jax.enable_x64(True):
        for label, (rho, mw, ne) in materials.items():
            out = optics._derived(lam, delta, beta, rho, mw, ne)
            for name, value in zip(
                ("electron_density", "critical_angle", "attenuation_length",
                 "re_sld", "im_sld"),
                out,
                strict=True,
            ):  # fmt: skip
                _assert_golden(f"derived_quantities{SEP}{label}{SEP}{name}", value)
        out = optics._derived(lam[:1], delta[:1], np.abs(beta[1:2]), 2, 28, 14)
        for name, value in zip(
            ("electron_density", "critical_angle", "attenuation_length",
             "re_sld", "im_sld"),
            out,
            strict=True,
        ):  # fmt: skip
            _assert_golden(
                f"derived_quantities{SEP}one_point_int_args{SEP}{name}", value
            )


def test_standalone_primitives_match_golden_edges():
    inp = _inputs()
    delta, wl, beta = inp["prim_delta"], inp["prim_wavelength_a"], inp["prim_beta"]
    with jax.enable_x64(True):
        crit = np.asarray(optics.critical_angle_from_delta(delta))
        _assert_golden(f"primitives{SEP}critical_angle", crit)
        assert np.isnan(crit[delta < 0]).all()  # no clamp, unlike derive()
        _assert_golden(
            f"primitives{SEP}critical_angle_negative_one",
            optics.critical_angle_from_delta(np.array([-1e-6])),
        )
        assert np.isnan(
            np.asarray(optics.critical_angle_from_delta(np.array([-1e-6])))
        ).all()
        _assert_golden(
            f"primitives{SEP}critical_angle_zero",
            optics.critical_angle_from_delta(np.array([0.0])),
        )
        for i, w in enumerate(wl):
            _assert_golden(
                f"primitives{SEP}attenuation_length{SEP}wl{i}",
                optics.attenuation_length_from_beta(np.full(beta.size, w), beta),
            )
        zero = np.asarray(
            optics.attenuation_length_from_beta(np.array([1.0]), np.array([0.0]))
        )
        assert np.isposinf(zero).all()  # no clamp, unlike derive()
        _assert_golden(f"primitives{SEP}attenuation_length_beta_zero", zero)
        _assert_golden(
            f"primitives{SEP}attenuation_length_beta_negative",
            optics.attenuation_length_from_beta(np.array([1.0]), np.array([-1e-8])),
        )
        att = inp["prim_att_length"]
        for thickness in (0.0, 1e-4, 1.0, 10.0):
            _assert_golden(
                f"primitives{SEP}transmission{SEP}t{thickness:g}",
                optics.transmission(thickness, att),
            )


# ------------------------------------------------------------ contracts


MATERIALS = [
    ("H2O", 1.0),
    ("SiO2", 2.2),
    ("Fe2O3", 5.24),
    ("CuSO4(H2O)5", 2.286),
    ("KMg3AlSi3O10(OH)2", 2.8),
]
E37 = np.linspace(0.05, 29.0, 37)


def _same(a: dict[str, Any], b: dict[str, Any]) -> None:
    for key in ("out", "electron_density", "ok"):
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def test_padding_invariance_is_bit_exact():
    """Padded K rows, F rows and M energies never change a result bit."""
    base = run_program(MATERIALS, E37)
    assert np.isfinite(base["out"]).all()
    for kw in (
        {"k_bucket": 16},
        {"k_bucket": 32},
        {"k_bucket": 128},
        {"f_bucket": 8},
        {"f_bucket": 32},
        {"m_bucket": 72},
        {"m_bucket": 128},
        {"m_bucket": 256},
        {"m_bucket": 1024},
        {"k_bucket": 16, "f_bucket": 16, "m_bucket": 192},
    ):
        _same(base, run_program(MATERIALS, E37, **kw))


def test_tail_program_padding_invariance_is_bit_exact():
    e = np.linspace(0.5, 25.0, 41)
    f1 = 14.0 + np.sin(e)
    f2 = 0.4 + 0.3 * np.cos(2.0 * e)
    base = run_tail("SiO2", 2.2, e, f1, f2)
    for m, f in ((64, 2), (72, 1), (128, 4), (512, 16)):
        got = run_tail("SiO2", 2.2, e, f1, f2, m_bucket=m, f_bucket=f)
        np.testing.assert_array_equal(got["out"], base["out"])
        np.testing.assert_array_equal(got["electron_density"], base["electron_density"])


def test_each_material_is_independent_of_its_batch_mates():
    base = run_program(MATERIALS, E37)
    for i, material in enumerate(MATERIALS):
        alone = run_program([material], E37)
        np.testing.assert_array_equal(alone["out"][0], base["out"][i])


def test_padded_rows_are_finite_and_ok():
    """z=0, counts=0, mw=1, rho=1 rows must not poison anything (no NaN)."""
    run = run_program(MATERIALS, E37, k_bucket=16, f_bucket=8, m_bucket=64)
    assert run["ok"].all()
    assert run["out"].shape == (len(MATERIALS), 9, E37.size)
    # The unpadded outputs of other tests never see the padded rows; check them raw.
    encoded = optics.encode_formula("Si")
    z = np.zeros((4, 4), dtype=np.int32)
    counts = np.zeros((4, 4))
    z[0, :1], counts[0, :1] = encoded[0], encoded[1]
    batch = (z, counts, np.ones(4), np.ones(4), np.zeros(4))
    with jax.enable_x64(True):
        got = optics._optical_program(load_tables(), batch, jnp.asarray(E37))
    assert bool(np.asarray(got["ok"]).all())
    assert np.isfinite(np.asarray(got["out"])).all()


def test_out_rows_follow_out_fields():
    """The program's ``out`` rows are the ``OpticalProps`` fields in ``OUT_FIELDS`` order.

    ``assert_ulp_close``: one side is an eager ``optical_properties`` call and the
    other the jitted batched program, two compiles of the same expression whose
    fusion may differ by an ulp.
    """
    run = run_program([("SiO2", 2.2)], E37)
    props, ok = _props("SiO2", 2.2, E37)
    assert bool(ok)
    for row, name in enumerate(optics.OUT_FIELDS):
        assert_ulp_close(np.asarray(getattr(props, name)), run["out"][0, row])
    assert_ulp_close(np.asarray(props.electron_density), run["electron_density"][0])


def _props(formula: str, rho: float, e_kev: Any):
    z, counts, mw, ne = optics.encode_formula(formula)
    with jax.enable_x64(True):
        return optics.optical_properties(
            load_tables(),
            jnp.asarray(z),
            jnp.asarray(counts),
            rho,
            mw,
            ne,
            jnp.asarray(e_kev),
        )


def test_optical_properties_is_a_pytree_of_nine_arrays_and_a_scalar():
    props, ok = _props("SiO2", 2.2, E37)
    leaves = jax.tree_util.tree_leaves(props)
    assert len(leaves) == 10
    assert props.electron_density.shape == ()
    for name in optics.OUT_FIELDS:
        assert getattr(props, name).shape == (E37.size,), name
    assert ok.dtype == jnp.bool_
    assert ok.shape == ()
    rebuilt = jax.tree_util.tree_map(lambda x: x, props)
    assert isinstance(rebuilt, optics.OpticalProps)
    # f1, f2 and the electron density are the interpolated/derived quantities
    assert float(props.electron_density) == pytest.approx(
        1e6 * 2.2 / optics.encode_formula("SiO2")[2] * AVOGADRO * 30.0 / 1e30,
        rel=1e-14,
    )


def test_grad_with_respect_to_density():
    """δ is linear in ρ, so d(sum δ)/dρ = sum δ / ρ (spec section 5)."""
    z, counts, mw, ne = optics.encode_formula("Fe2O3")
    tables = load_tables()
    with jax.enable_x64(True):
        zj, cj, ej = jnp.asarray(z), jnp.asarray(counts), jnp.asarray(E37)

        def total_dispersion(rho):
            props, _ = optics.optical_properties(tables, zj, cj, rho, mw, ne, ej)
            return props.dispersion.sum()

        rho = 5.24
        grad = jax.grad(total_dispersion)(rho)
        value = total_dispersion(rho)
        assert np.isfinite(float(grad))
        np.testing.assert_allclose(float(grad), float(value) / rho, rtol=1e-12)


def test_vmap_over_densities_equals_the_batched_program():
    z, counts, mw, ne = optics.encode_formula("SiO2")
    rhos = np.array([1.0, 2.2, 2.65, 4.0])
    tables = load_tables()
    with jax.enable_x64(True):
        zj, cj, ej = jnp.asarray(z), jnp.asarray(counts), jnp.asarray(E37)

        def one(rho):
            return optics.optical_properties(tables, zj, cj, rho, mw, ne, ej)

        props, ok = jax.jit(jax.vmap(one))(jnp.asarray(rhos))
    assert np.asarray(ok).all()
    ref = run_program([("SiO2", float(r)) for r in rhos], E37)
    # vmap-of-jit and the program are different compiles of the same expression;
    # XLA may fuse one multiply differently, so allow a few ulp (never rtol 1e-12).
    for row, name in enumerate(optics.OUT_FIELDS):
        assert_ulp_close(np.asarray(getattr(props, name)), ref["out"][:, row, :])
    assert_ulp_close(np.asarray(props.electron_density), ref["electron_density"])


def test_negative_f1_keeps_negative_delta_and_zero_critical_angle():
    run = run_tail("H2OH2", 1.0, np.array([8.0]), np.array([-5.0]), np.array([-0.1]))
    out = dict(zip(optics.OUT_FIELDS, run["out"][0], strict=True))
    assert out["dispersion"][0] < 0
    assert out["critical_angle"][0] == 0.0
    assert out["absorption"][0] < 0  # kept; only the attenuation length clamps
    assert np.isfinite(out["attenuation_length"]).all()
    assert run["ok"][0]


def test_non_finite_input_sets_ok_false_only_for_that_material():
    e = np.array([8.0, 9.0, 10.0])
    got = _two_row_tail(
        [1.0, np.nan, 2.0], [0.1, 0.2, 0.3], [1.0, 2.0, 3.0], [0.1, 0.2, 0.3], e
    )
    assert got["ok"].tolist() == [False, True]
    got = _two_row_tail(
        [1.0, 2.0, 3.0], [0.1, np.inf, 0.3], [1.0, 2.0, 3.0], [0.1, 0.2, 0.3], e
    )
    assert got["ok"].tolist() == [False, True]


def _two_row_tail(f1a, f2a, f1b, f2b, e):
    with jax.enable_x64(True):
        batch = (
            np.array([28.0855, 28.0855]),
            np.array([2.3, 2.3]),
            np.array([14.0, 14.0]),
        )
        return jax.device_get(
            optics._tail_program(
                batch, np.array([f1a, f1b]), np.array([f2a, f2b]), np.asarray(e)
            )
        )


def test_energy_outside_the_table_sets_ok_false_not_an_error():
    run = run_program([("SiO2", 2.2), ("Si", 2.33)], np.array([0.02, 8.0]))
    assert run["ok"].tolist() == [False, False]
    assert np.isnan(run["out"][:, 1, 0]).all()  # δ at 0.02 keV (20 eV < 29.3 eV)
    assert np.isfinite(run["out"][:, 1, 1]).all()
    inside = run_program([("SiO2", 2.2)], np.array([8.0]))
    assert inside["ok"].tolist() == [True]


def test_program_does_not_embed_the_tables_in_the_hlo():
    tables = load_tables()
    z = np.zeros((2, 4), dtype=np.int32)
    batch = (z, np.zeros((2, 4)), np.ones(2), np.ones(2), np.zeros(2))
    with jax.enable_x64(True):
        text = optics._optical_program.lower(tables, batch, np.ones(64)).as_text()
    table_bytes = sum(a.nbytes for a in jax.tree_util.tree_leaves(tables))
    assert table_bytes > 1_500_000
    # A closed-over 1.7 MB table would put megabytes of constants in the text.
    assert len(text) < 100_000, len(text)


def test_x64_scope_overrides_a_disabled_global_setting():
    base = run_program([("SiO2", 2.2)], E37)
    with jax.enable_x64(False):
        again = run_program([("SiO2", 2.2)], E37)
    np.testing.assert_array_equal(base["out"], again["out"])
    assert again["out"].dtype == np.float64


_PROBE = """
import sys

import jax

import xraylabtool

assert "xraylabtool.calculators.optics" not in sys.modules, "package init imports optics"
assert "xraylabtool.calculators.tables" not in sys.modules, "package init imports tables"

# Until Task 2.5 the calculators package still imports the legacy ops twin, whose
# backend auto-select initialises JAX; load it first so the probe isolates optics.
import xraylabtool.calculators  # noqa: F401

jax.config.update("jax_enable_x64", False)
from jax._src import xla_bridge

before = (
    jax.config.jax_enable_x64,
    jax.config.jax_compilation_cache_dir,
    xla_bridge.backends_are_initialized(),
)
import xraylabtool.calculators.optics  # noqa: F401

after = (
    jax.config.jax_enable_x64,
    jax.config.jax_compilation_cache_dir,
    xla_bridge.backends_are_initialized(),
)
assert after == before, (before, after)
assert after[0] is False, "importing optics enabled x64"
print("OK")
"""


def test_importing_optics_has_no_side_effect_and_is_not_eager():
    import os
    import subprocess
    import sys

    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("JAX_ENABLE_X64", "JAX_COMPILATION_CACHE_DIR")
    }
    done = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().endswith("OK")
