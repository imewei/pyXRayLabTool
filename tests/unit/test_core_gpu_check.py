"""``scripts/core_gpu_check.py`` P12 verdict, with the fresh-process runs mocked (CPU-safe)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "core_gpu_check_under_test", SCRIPTS / "core_gpu_check.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    import core_bench

    return module, core_bench


def _fake(core_bench, monkeypatch, warm_compiles_batch200: float):
    def spawn_runs(case, backend, device, runs):
        if case == "compiles":
            return [
                {
                    "p12_same_shape": 0,
                    "p13_energy_200_to_201": 0,
                    "p13_energy_201_to_216": 0,
                }
            ] * runs
        return [
            {
                "warm_ms": 0.1,
                "first_ms": 1.0,
                "warm_compiles": warm_compiles_batch200 if case == "batch200" else 0,
            }
        ] * runs

    monkeypatch.setattr(core_bench, "spawn_runs", spawn_runs)


def _p12(module):
    return next(ok for name, ok, _ in module.rows if name.startswith("P12"))


def test_p12_fails_when_the_timed_warm_calls_compile(gate, monkeypatch):
    module, core_bench = gate
    _fake(core_bench, monkeypatch, 15)
    module.check_bench(2)
    assert _p12(module) is False


def test_p12_passes_when_nothing_compiles_after_warm_up(gate, monkeypatch):
    module, core_bench = gate
    _fake(core_bench, monkeypatch, 0)
    module.check_bench(2)
    assert _p12(module) is True
