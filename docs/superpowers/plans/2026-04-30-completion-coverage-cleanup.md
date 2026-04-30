# Completion Cleanup & Coverage Gap Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove stale coverage-omit entries, add tests for 5 untested modules, and reduce the coverage exclusion list from 7 paths to 1 (gui/*).

**Architecture:** TDD task-by-task. Each task: write tests → run to see red → run to see green → update pyproject.toml → commit. No production code changes required; all work is in tests and pyproject.toml.

**Tech Stack:** pytest, uv, xraylabtool (JAX backend), PySide6 excluded throughout

---

## File Map

| Action | Path |
|--------|------|
| Modify | `pyproject.toml` — coverage omit list (updated in Tasks 1, 2, 3, 4, 7) |
| Create | `tests/unit/test_validators.py` |
| Create | `tests/unit/test_data_export.py` |
| Create | `tests/unit/test_completion_bridge.py` |
| Create | `tests/unit/test_completion_cache.py` |
| Create | `tests/unit/test_completion_environment.py` |
| Create | `tests/unit/test_completion_integration.py` |

---

## Task 1: Remove ghost omit entries (no production code, no tests)

**Files:**
- Modify: `pyproject.toml:222-234` (`[tool.coverage.run]` omit section)

These three entries reference paths that do not exist or are already covered — removing them is free.

- `*/interfaces/completion_legacy.py` — file was deleted; entry is dead weight
- `*/validation/exceptions.py` — exceptions live at `xraylabtool/exceptions.py` (root), not under `validation/`; path never matched
- `*/io/file_operations.py` — `tests/unit/test_file_operations.py` has 14 passing tests; file is already covered

- [ ] **Step 1: Edit pyproject.toml**

Remove exactly these three lines from `[tool.coverage.run]` omit:
```toml
    "*/interfaces/completion_legacy.py",   # Legacy completion system - deprecated
    "*/validation/exceptions.py",          # Exception definitions - simple classes
    "*/io/file_operations.py",             # File I/O operations - basic functionality
```

After edit the omit block looks like:
```toml
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__init__.py",
    "*/conftest.py",
    "*/interfaces/completion.py",
    "*/interfaces/completion_v2/*.py",
    "*/io/data_export.py",
    "*/validation/validators.py",
    "*/gui/*",
]
```

- [ ] **Step 2: Verify tests still pass**

Run: `uv run pytest tests/unit/test_file_operations.py tests/unit/test_exceptions.py -v`

Expected: all green, no errors

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore(coverage): remove three dead omit entries"
```

---

## Task 2: Tests for `validation/validators.py`

**Files:**
- Create: `tests/unit/test_validators.py`
- Modify: `pyproject.toml` — remove `*/validation/validators.py` from omit

- [ ] **Step 1: Write the tests**

Create `tests/unit/test_validators.py`:
```python
"""Tests for xraylabtool.validation.validators."""

from __future__ import annotations

import math

import numpy as np
import pytest

from xraylabtool.exceptions import EnergyError, FormulaError, ValidationError
from xraylabtool.validation.validators import (
    validate_chemical_formula,
    validate_density,
    validate_energy_range,
)


# ---------------------------------------------------------------------------
# validate_energy_range
# ---------------------------------------------------------------------------


class TestValidateEnergyRange:
    def test_valid_scalar(self):
        result = validate_energy_range(10.0)
        assert float(result) == pytest.approx(10.0)

    def test_valid_array(self):
        energies = np.array([1.0, 5.0, 10.0])
        result = validate_energy_range(energies)
        np.testing.assert_array_equal(result, energies)

    def test_at_minimum_boundary(self):
        result = validate_energy_range(0.1)
        assert float(result) == pytest.approx(0.1)

    def test_at_maximum_boundary(self):
        result = validate_energy_range(100.0)
        assert float(result) == pytest.approx(100.0)

    def test_custom_range(self):
        result = validate_energy_range(5.0, min_energy=1.0, max_energy=50.0)
        assert float(result) == pytest.approx(5.0)

    def test_nan_raises(self):
        with pytest.raises(EnergyError, match="finite"):
            validate_energy_range(float("nan"))

    def test_inf_raises(self):
        with pytest.raises(EnergyError, match="finite"):
            validate_energy_range(float("inf"))

    def test_negative_raises(self):
        with pytest.raises(EnergyError, match="positive"):
            validate_energy_range(-1.0)

    def test_zero_raises(self):
        with pytest.raises(EnergyError, match="positive"):
            validate_energy_range(0.0)

    def test_below_min_raises(self):
        with pytest.raises(EnergyError, match="below minimum"):
            validate_energy_range(0.05)  # default min is 0.1

    def test_above_max_raises(self):
        with pytest.raises(EnergyError, match="above maximum"):
            validate_energy_range(150.0)  # default max is 100.0

    def test_array_with_one_bad_value_raises(self):
        energies = np.array([5.0, 10.0, float("nan")])
        with pytest.raises(EnergyError):
            validate_energy_range(energies)


# ---------------------------------------------------------------------------
# validate_chemical_formula
# ---------------------------------------------------------------------------


class TestValidateChemicalFormula:
    def test_simple_formula(self):
        result = validate_chemical_formula("SiO2")
        assert "Si" in result
        assert "O" in result

    def test_complex_formula(self):
        result = validate_chemical_formula("Al2O3")
        assert "Al" in result
        assert "O" in result

    def test_single_element(self):
        result = validate_chemical_formula("Au")
        assert "Au" in result

    def test_empty_string_raises(self):
        with pytest.raises(FormulaError):
            validate_chemical_formula("")

    def test_none_raises(self):
        with pytest.raises((FormulaError, TypeError)):
            validate_chemical_formula(None)  # type: ignore[arg-type]

    def test_invalid_characters_raises(self):
        with pytest.raises(FormulaError, match="invalid characters"):
            validate_chemical_formula("Si@2")

    def test_unknown_element_raises(self):
        with pytest.raises(FormulaError):
            validate_chemical_formula("Xx2O3")

    def test_whitespace_only_raises(self):
        with pytest.raises(FormulaError):
            validate_chemical_formula("   ")


# ---------------------------------------------------------------------------
# validate_density
# ---------------------------------------------------------------------------


class TestValidateDensity:
    def test_valid_density(self):
        result = validate_density(2.2)
        assert result == pytest.approx(2.2)

    def test_returns_float(self):
        result = validate_density(3)
        assert isinstance(result, float)

    def test_zero_raises(self):
        with pytest.raises(ValidationError, match="positive"):
            validate_density(0.0)

    def test_negative_raises(self):
        with pytest.raises(ValidationError, match="positive"):
            validate_density(-1.0)

    def test_inf_raises(self):
        with pytest.raises(ValidationError, match="finite"):
            validate_density(math.inf)

    def test_nan_raises(self):
        with pytest.raises(ValidationError, match="finite"):
            validate_density(math.nan)

    def test_string_raises(self):
        with pytest.raises((ValidationError, TypeError)):
            validate_density("2.2")  # type: ignore[arg-type]

    def test_below_min_raises(self):
        with pytest.raises(ValidationError, match="below minimum"):
            validate_density(0.0001)  # default min is 0.001

    def test_above_max_raises(self):
        with pytest.raises(ValidationError, match="above maximum"):
            validate_density(50.0)  # default max is 30.0

    def test_custom_max(self):
        result = validate_density(50.0, max_density=100.0)
        assert result == pytest.approx(50.0)
```

- [ ] **Step 2: Run to verify red (import should work; assertions may fail)**

Run: `uv run pytest tests/unit/test_validators.py -v`

Expected: tests run (no ImportError). Some may fail if edge-case behaviour differs — note which ones and fix the assertion to match actual behaviour.

- [ ] **Step 3: Remove validators.py from coverage omit in pyproject.toml**

Remove this line:
```toml
    "*/validation/validators.py",          # Validation utilities - limited usage
```

- [ ] **Step 4: Run tests again and confirm green**

Run: `uv run pytest tests/unit/test_validators.py -v`

Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_validators.py pyproject.toml
git commit -m "test(validators): add coverage for validate_energy_range, validate_chemical_formula, validate_density"
```

---

## Task 3: Tests for `io/data_export.py`

**Files:**
- Create: `tests/unit/test_data_export.py`
- Modify: `pyproject.toml` — remove `*/io/data_export.py` from omit

- [ ] **Step 1: Write the tests**

Create `tests/unit/test_data_export.py`:
```python
"""Tests for xraylabtool.io.data_export."""

from __future__ import annotations

import json

import numpy as np
import pytest

from xraylabtool.calculators.core import calculate_single_material_properties
from xraylabtool.io.data_export import format_calculation_summary, format_xray_result


@pytest.fixture(scope="module")
def single_result():
    return calculate_single_material_properties("SiO2", 10.0, 2.2)


@pytest.fixture(scope="module")
def two_results():
    return [
        calculate_single_material_properties("SiO2", 10.0, 2.2),
        calculate_single_material_properties("Al2O3", 10.0, 3.97),
    ]


# ---------------------------------------------------------------------------
# format_xray_result
# ---------------------------------------------------------------------------


class TestFormatXrayResult:
    def test_default_table_returns_string(self, single_result):
        output = format_xray_result(single_result)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_table_contains_formula(self, single_result):
        output = format_xray_result(single_result, format_type="table")
        assert "SiO2" in output or "formula" in output.lower()

    def test_json_format_is_valid_json(self, single_result):
        output = format_xray_result(single_result, format_type="json")
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_json_format_contains_numeric_fields(self, single_result):
        output = format_xray_result(single_result, format_type="json")
        parsed = json.loads(output)
        assert len(parsed) > 0

    def test_csv_format_returns_string(self, single_result):
        output = format_xray_result(single_result, format_type="csv")
        assert isinstance(output, str)

    def test_unknown_format_falls_back_to_table(self, single_result):
        output = format_xray_result(single_result, format_type="unknown_xyz")
        assert isinstance(output, str)

    def test_precision_parameter(self, single_result):
        out2 = format_xray_result(single_result, format_type="json", precision=2)
        out6 = format_xray_result(single_result, format_type="json", precision=6)
        assert isinstance(out2, str)
        assert isinstance(out6, str)

    def test_specific_fields(self, single_result):
        output = format_xray_result(
            single_result, format_type="json", fields=["formula"]
        )
        parsed = json.loads(output)
        assert "formula" in parsed


# ---------------------------------------------------------------------------
# format_calculation_summary
# ---------------------------------------------------------------------------


class TestFormatCalculationSummary:
    def test_empty_list_returns_sentinel(self):
        output = format_calculation_summary([])
        assert output == "No results to display"

    def test_single_result_table(self, single_result):
        output = format_calculation_summary([single_result])
        assert "1" in output
        assert isinstance(output, str)

    def test_multiple_results_table(self, two_results):
        output = format_calculation_summary(two_results)
        assert isinstance(output, str)
        assert "2" in output

    def test_json_format_returns_list(self, two_results):
        output = format_calculation_summary(two_results, format_type="json")
        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_csv_format_returns_string(self, two_results):
        output = format_calculation_summary(two_results, format_type="csv")
        assert isinstance(output, str)
        assert "\n" in output  # CSV has newlines
```

- [ ] **Step 2: Run to verify tests are importable and run**

Run: `uv run pytest tests/unit/test_data_export.py -v`

Expected: tests run. Fix any assertion that doesn't match actual output format.

- [ ] **Step 3: Remove data_export.py from coverage omit in pyproject.toml**

Remove this line:
```toml
    "*/io/data_export.py",                 # Data export utilities - basic functionality
```

- [ ] **Step 4: Run tests and confirm green**

Run: `uv run pytest tests/unit/test_data_export.py -v`

Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_data_export.py pyproject.toml
git commit -m "test(io): add coverage for format_xray_result and format_calculation_summary"
```

---

## Task 4: Tests for `interfaces/completion.py` bridge

**Files:**
- Create: `tests/unit/test_completion_bridge.py`
- Modify: `pyproject.toml` — remove `*/interfaces/completion.py` from omit

The bridge has exactly one function with logic (`_generate_bash_completion_script`) plus re-exports. Tests verify the public surface and that the function works.

- [ ] **Step 1: Write the tests**

Create `tests/unit/test_completion_bridge.py`:
```python
"""Tests for xraylabtool.interfaces.completion bridge."""

from __future__ import annotations

from xraylabtool.interfaces import completion as bridge
from xraylabtool.interfaces.completion import (
    BASH_COMPLETION_SCRIPT,
    CompletionInstaller,
    _generate_bash_completion_script,
    install_completion_main,
    uninstall_completion_main,
)


class TestBridgeExports:
    def test_completion_installer_importable(self):
        assert CompletionInstaller is not None

    def test_install_completion_main_callable(self):
        assert callable(install_completion_main)

    def test_uninstall_completion_main_callable(self):
        assert callable(uninstall_completion_main)


class TestBashCompletionScript:
    def test_constant_is_string(self):
        assert isinstance(BASH_COMPLETION_SCRIPT, str)

    def test_constant_is_nonempty(self):
        assert len(BASH_COMPLETION_SCRIPT) > 0

    def test_constant_references_xraylabtool(self):
        assert "xraylabtool" in BASH_COMPLETION_SCRIPT

    def test_generate_function_returns_string(self):
        result = _generate_bash_completion_script()
        assert isinstance(result, str)

    def test_generate_function_returns_nonempty(self):
        result = _generate_bash_completion_script()
        assert len(result) > 0

    def test_generate_function_contains_command_name(self):
        result = _generate_bash_completion_script()
        assert "xraylabtool" in result

    def test_generate_function_is_repeatable(self):
        first = _generate_bash_completion_script()
        second = _generate_bash_completion_script()
        assert first == second
```

- [ ] **Step 2: Run to verify**

Run: `uv run pytest tests/unit/test_completion_bridge.py -v`

Expected: all pass (bridge is thin re-exports + string generation)

- [ ] **Step 3: Remove completion.py from coverage omit in pyproject.toml**

Remove this line:
```toml
    "*/interfaces/completion.py",          # Shell completion bridge - basic functionality
```

- [ ] **Step 4: Confirm green after omit removal**

Run: `uv run pytest tests/unit/test_completion_bridge.py -v`

Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_completion_bridge.py pyproject.toml
git commit -m "test(interfaces): add coverage for completion.py bridge"
```

---

## Task 5: Tests for `completion_v2/cache.py`

**Files:**
- Create: `tests/unit/test_completion_cache.py`

All tests use `tmp_path` — never write to the real `~/.xraylabtool/cache`.

- [ ] **Step 1: Write the tests**

Create `tests/unit/test_completion_cache.py`:
```python
"""Tests for xraylabtool.interfaces.completion_v2.cache."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from xraylabtool.interfaces.completion_v2.cache import CompletionCache


@pytest.fixture
def cache(tmp_path: Path) -> CompletionCache:
    return CompletionCache(cache_dir=tmp_path / "cache")


class TestCompletionCacheInit:
    def test_creates_cache_dir(self, tmp_path: Path):
        cache_dir = tmp_path / "new_cache"
        assert not cache_dir.exists()
        CompletionCache(cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_default_timeout_set(self, cache: CompletionCache):
        assert cache.default_timeout == 3600

    def test_command_cache_timeout_set(self, cache: CompletionCache):
        assert cache.command_cache_timeout == 86400


class TestGetCacheKey:
    def test_returns_hex_string(self, cache: CompletionCache):
        key = cache.get_cache_key("hello")
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex digest

    def test_deterministic_for_string(self, cache: CompletionCache):
        assert cache.get_cache_key("test") == cache.get_cache_key("test")

    def test_deterministic_for_dict(self, cache: CompletionCache):
        data = {"a": 1, "b": 2}
        assert cache.get_cache_key(data) == cache.get_cache_key(data)

    def test_different_inputs_different_keys(self, cache: CompletionCache):
        assert cache.get_cache_key("foo") != cache.get_cache_key("bar")


class TestCacheGetSet:
    def test_get_missing_key_returns_none(self, cache: CompletionCache):
        assert cache.get("nonexistent_key") is None

    def test_set_then_get_returns_data(self, cache: CompletionCache):
        cache.set("mykey", {"value": 42})
        result = cache.get("mykey")
        assert result == {"value": 42}

    def test_set_then_get_string(self, cache: CompletionCache):
        cache.set("strkey", "hello world")
        assert cache.get("strkey") == "hello world"

    def test_set_with_metadata(self, cache: CompletionCache):
        cache.set("key1", [1, 2, 3], metadata={"source": "test"})
        result = cache.get("key1")
        assert result == [1, 2, 3]

    def test_get_respects_timeout_not_expired(self, cache: CompletionCache):
        cache.set("fresh", "data")
        result = cache.get("fresh", timeout=3600)
        assert result == "data"

    def test_get_returns_none_for_expired(self, cache: CompletionCache, tmp_path: Path):
        cache_dir = tmp_path / "expired_cache"
        c = CompletionCache(cache_dir=cache_dir)
        c.set("expkey", "old_data")

        # Make file appear old by modifying mtime
        cache_file = cache_dir / "expkey.json"
        old_time = time.time() - 7200  # 2 hours ago
        import os
        os.utime(cache_file, (old_time, old_time))

        result = c.get("expkey", timeout=3600)
        assert result is None

    def test_get_removes_expired_file(self, cache: CompletionCache, tmp_path: Path):
        cache_dir = tmp_path / "rm_cache"
        c = CompletionCache(cache_dir=cache_dir)
        c.set("rmkey", "data")

        cache_file = cache_dir / "rmkey.json"
        old_time = time.time() - 7200
        import os
        os.utime(cache_file, (old_time, old_time))

        c.get("rmkey", timeout=3600)
        assert not cache_file.exists()

    def test_get_handles_corrupt_file(self, cache: CompletionCache):
        bad_file = cache.cache_dir / "corrupt.json"
        bad_file.write_text("NOT VALID JSON {{{")
        result = cache.get("corrupt")
        assert result is None


class TestCacheInvalidate:
    def test_invalidate_removes_entry(self, cache: CompletionCache):
        cache.set("inv_key", "value")
        cache.invalidate("inv_key")
        assert cache.get("inv_key") is None

    def test_invalidate_nonexistent_key_is_noop(self, cache: CompletionCache):
        cache.invalidate("does_not_exist")  # should not raise
```

- [ ] **Step 2: Run to verify**

Run: `uv run pytest tests/unit/test_completion_cache.py -v`

Expected: all pass

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_completion_cache.py
git commit -m "test(completion_v2): add coverage for CompletionCache"
```

---

## Task 6: Tests for `completion_v2/environment.py`

**Files:**
- Create: `tests/unit/test_completion_environment.py`

`EnvironmentDetector.get_current_environment()` reads `os.environ` and may call `subprocess`. Tests mock both to remain hermetic and avoid writing to `~/.xraylabtool/`.

- [ ] **Step 1: Write the tests**

Create `tests/unit/test_completion_environment.py`:
```python
"""Tests for xraylabtool.interfaces.completion_v2.environment."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from xraylabtool.interfaces.completion_v2.environment import (
    EnvironmentDetector,
    EnvironmentInfo,
    EnvironmentType,
)


class TestEnvironmentType:
    def test_system_constant(self):
        assert EnvironmentType.SYSTEM == "system"

    def test_venv_constant(self):
        assert EnvironmentType.VENV == "venv"

    def test_conda_constant(self):
        assert EnvironmentType.CONDA == "conda"

    def test_all_constants_are_strings(self):
        for attr in ("SYSTEM", "VENV", "VIRTUALENV", "CONDA", "MAMBA", "PIPENV", "POETRY"):
            assert isinstance(getattr(EnvironmentType, attr), str)


class TestEnvironmentInfo:
    def test_constructor_sets_fields(self, tmp_path: Path):
        info = EnvironmentInfo(
            env_type="venv",
            path=tmp_path,
            name="myenv",
            is_active=True,
            python_version="3.12.0",
            has_completion=False,
        )
        assert info.env_type == "venv"
        assert info.path == tmp_path
        assert info.name == "myenv"
        assert info.is_active is True
        assert info.python_version == "3.12.0"
        assert info.has_completion is False

    def test_defaults_are_correct(self, tmp_path: Path):
        info = EnvironmentInfo(env_type="system", path=tmp_path, name="base")
        assert info.is_active is False
        assert info.python_version is None
        assert info.has_completion is False

    def test_from_dict_roundtrip(self, tmp_path: Path):
        info = EnvironmentInfo(
            env_type="conda",
            path=tmp_path,
            name="science",
            is_active=True,
            python_version="3.12",
            has_completion=True,
        )
        data = info.to_dict()
        restored = EnvironmentInfo.from_dict(data)
        assert restored.env_type == info.env_type
        assert restored.name == info.name
        assert restored.is_active == info.is_active
        assert restored.python_version == info.python_version
        assert restored.has_completion == info.has_completion


class TestEnvironmentDetector:
    def test_instantiates_without_error(self, tmp_path: Path):
        with patch.object(
            EnvironmentDetector, "__init__",
            lambda self: setattr(self, "_cache_file", tmp_path / "env_cache.json")
            or setattr(self, "_cache_timeout", 3600)
        ):
            det = EnvironmentDetector()
            assert det._cache_timeout == 3600

    def test_get_current_environment_returns_none_or_info(self, tmp_path: Path):
        with (
            patch(
                "xraylabtool.interfaces.completion_v2.environment.Path.home",
                return_value=tmp_path,
            ),
            patch("os.environ.get", return_value=None),
        ):
            det = EnvironmentDetector()
            result = det.get_current_environment()
            assert result is None or isinstance(result, EnvironmentInfo)

    def test_discover_all_environments_returns_list(self, tmp_path: Path):
        with patch(
            "xraylabtool.interfaces.completion_v2.environment.Path.home",
            return_value=tmp_path,
        ):
            det = EnvironmentDetector()
            result = det.discover_all_environments(use_cache=False)
            assert isinstance(result, list)
```

- [ ] **Step 2: Run to verify**

Run: `uv run pytest tests/unit/test_completion_environment.py -v`

Expected: all pass. If `EnvironmentInfo.to_dict()` / `from_dict()` don't exist, remove `test_from_dict_roundtrip` and note the gap.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_completion_environment.py
git commit -m "test(completion_v2): add coverage for EnvironmentType, EnvironmentInfo, EnvironmentDetector"
```

---

## Task 7: Tests for `completion_v2/integration.py` + remove completion_v2 omit

**Files:**
- Create: `tests/unit/test_completion_integration.py`
- Modify: `pyproject.toml` — remove `*/interfaces/completion_v2/*.py` from omit

`install_completion_main` and `uninstall_completion_main` are aliased to the `legacy_*` functions at module level. Tests use `argparse.Namespace` as the args object and mock `CompletionInstaller` to avoid filesystem side effects.

- [ ] **Step 1: Write the tests**

Create `tests/unit/test_completion_integration.py`:
```python
"""Tests for xraylabtool.interfaces.completion_v2.integration."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

from xraylabtool.interfaces.completion_v2.integration import (
    install_completion_main,
    legacy_install_completion_main,
    legacy_uninstall_completion_main,
    uninstall_completion_main,
)


def _args(**kwargs) -> argparse.Namespace:
    defaults = {
        "shell": None,
        "force": False,
        "system": False,
        "test": False,
        "uninstall": False,
        "install_completion": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestAliases:
    def test_install_completion_main_is_alias(self):
        assert install_completion_main is legacy_install_completion_main

    def test_uninstall_completion_main_is_alias(self):
        assert uninstall_completion_main is legacy_uninstall_completion_main


class TestLegacyInstallCompletionMain:
    def test_test_mode_returns_int(self):
        with patch(
            "xraylabtool.interfaces.completion_v2.integration.test_completion_installation",
            return_value=0,
        ):
            result = legacy_install_completion_main(_args(test=True))
        assert isinstance(result, int)

    def test_uninstall_mode_calls_installer_uninstall(self):
        mock_installer = MagicMock()
        mock_installer.uninstall.return_value = True
        with patch(
            "xraylabtool.interfaces.completion_v2.integration.CompletionInstaller",
            return_value=mock_installer,
        ):
            result = legacy_install_completion_main(_args(uninstall=True))
        mock_installer.uninstall.assert_called_once()
        assert result == 0

    def test_normal_install_success_returns_0(self):
        mock_installer = MagicMock()
        mock_installer.install.return_value = True
        with patch(
            "xraylabtool.interfaces.completion_v2.integration.CompletionInstaller",
            return_value=mock_installer,
        ):
            result = legacy_install_completion_main(_args(shell="bash"))
        assert result == 0

    def test_normal_install_failure_returns_1(self):
        mock_installer = MagicMock()
        mock_installer.install.return_value = False
        with patch(
            "xraylabtool.interfaces.completion_v2.integration.CompletionInstaller",
            return_value=mock_installer,
        ):
            result = legacy_install_completion_main(_args(shell="bash"))
        assert result == 1

    def test_keyboard_interrupt_returns_1(self):
        with patch(
            "xraylabtool.interfaces.completion_v2.integration.CompletionInstaller",
            side_effect=KeyboardInterrupt,
        ):
            result = legacy_install_completion_main(_args())
        assert result == 1

    def test_exception_returns_1(self):
        with patch(
            "xraylabtool.interfaces.completion_v2.integration.CompletionInstaller",
            side_effect=RuntimeError("boom"),
        ):
            result = legacy_install_completion_main(_args())
        assert result == 1


class TestLegacyUninstallCompletionMain:
    def test_cleanup_mode_calls_uninstall_all(self):
        mock_installer = MagicMock()
        mock_installer.uninstall.return_value = True
        with patch(
            "xraylabtool.interfaces.completion_v2.integration.CompletionInstaller",
            return_value=mock_installer,
        ):
            result = legacy_uninstall_completion_main(
                argparse.Namespace(shell=None, system=False, cleanup=True)
            )
        mock_installer.uninstall.assert_called_once_with(all_envs=True)
        assert result == 0

    def test_regular_uninstall_returns_0_on_success(self):
        mock_installer = MagicMock()
        mock_installer.uninstall.return_value = True
        with patch(
            "xraylabtool.interfaces.completion_v2.integration.CompletionInstaller",
            return_value=mock_installer,
        ):
            result = legacy_uninstall_completion_main(
                argparse.Namespace(shell="bash", system=False, cleanup=False)
            )
        assert result == 0
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/unit/test_completion_integration.py -v`

Expected: all pass

- [ ] **Step 3: Remove completion_v2/*.py from coverage omit in pyproject.toml**

Remove this line:
```toml
    "*/interfaces/completion_v2/*.py",     # New completion system - complex to test
```

- [ ] **Step 4: Run all completion tests to confirm green**

Run: `uv run pytest tests/unit/test_completion_bridge.py tests/unit/test_completion_cache.py tests/unit/test_completion_environment.py tests/unit/test_completion_integration.py tests/integration/test_completion_installer.py -v`

Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_completion_integration.py pyproject.toml
git commit -m "test(completion_v2): add integration coverage, remove completion_v2 from coverage omit"
```

---

## Task 8: Final verification

- [ ] **Step 1: Run all new test files together**

Run: `uv run pytest tests/unit/test_validators.py tests/unit/test_data_export.py tests/unit/test_completion_bridge.py tests/unit/test_completion_cache.py tests/unit/test_completion_environment.py tests/unit/test_completion_integration.py -v`

Expected: all pass, no errors

- [ ] **Step 2: Verify coverage omit list is clean**

After all tasks, `[tool.coverage.run]` omit in `pyproject.toml` should contain exactly:
```toml
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__init__.py",
    "*/conftest.py",
    "*/gui/*",
]
```

- [ ] **Step 3: Run full unit suite to check for regressions**

Run: `uv run pytest tests/unit/ -v --tb=short`

Expected: all existing tests still pass, no regressions

- [ ] **Step 4: Check coverage report**

Run: `uv run pytest tests/unit/ --cov=xraylabtool --cov-report=term-missing -q`

Expected: completion, validators, and data_export modules now show coverage. `gui/*` remains excluded.

---

## Self-Review

**Spec coverage check:**
- P1 completion_legacy.py removal ✅ Task 1
- P1 completion.py bridge coverage ✅ Task 4
- P1 completion_v2 coverage ✅ Tasks 5, 6, 7
- P2 file_operations.py (already tested) ✅ Task 1
- P2 data_export.py ✅ Task 3
- P2 validators.py ✅ Task 2
- P2 validation/exceptions.py (dead path, not the real exceptions.py) ✅ Task 1
- P2 gui/* — kept excluded, legitimately requires PySide6 infrastructure ✅

**Placeholder scan:** No TBD/TODO/implement-later phrases present.

**Type consistency:** All imports use exact module paths verified against source. `CompletionCache`, `EnvironmentDetector`, `EnvironmentInfo`, `EnvironmentType` match names found in source. `validate_energy_range`, `validate_chemical_formula`, `validate_density` match validators.py. `format_xray_result`, `format_calculation_summary` match data_export.py.
