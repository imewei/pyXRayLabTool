"""Tests for thread-safety of lru_cache-decorated functions in utils.py."""

from __future__ import annotations

from functools import lru_cache
import threading
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_element(
    atomic_number: int, atomic_weight: float, symbol: str
) -> MagicMock:
    elem = MagicMock()
    elem.atomic_number = atomic_number
    elem.atomic_weight = atomic_weight
    elem.symbol = symbol
    elem.name = symbol.lower()
    elem.density = 1.0
    return elem


_ELEMENTS: dict[str, MagicMock] = {
    "H": _make_mock_element(1, 1.008, "H"),
    "C": _make_mock_element(6, 12.011, "C"),
    "N": _make_mock_element(7, 14.007, "N"),
    "O": _make_mock_element(8, 15.999, "O"),
    "Si": _make_mock_element(14, 28.085, "Si"),
    "Fe": _make_mock_element(26, 55.845, "Fe"),
}


def _mock_get_element(symbol: str) -> MagicMock:
    if symbol not in _ELEMENTS:
        raise ValueError(f"Element not found: {symbol}")
    return _ELEMENTS[symbol]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_lru_caches():
    """Clear LRU caches before each test to avoid cross-test state."""
    from xraylabtool.utils import get_atomic_data, get_atomic_number, get_atomic_weight

    get_atomic_number.cache_clear()
    get_atomic_weight.cache_clear()
    get_atomic_data.cache_clear()
    yield
    get_atomic_number.cache_clear()
    get_atomic_weight.cache_clear()
    get_atomic_data.cache_clear()


# ---------------------------------------------------------------------------
# Item 4.4: Thread-safety of lru_cache-decorated functions
# ---------------------------------------------------------------------------


class TestConcurrentGetAtomicNumber:
    def test_all_threads_get_correct_result(self):
        """Multiple threads calling get_atomic_number simultaneously return correct values."""
        from xraylabtool.utils import get_atomic_number

        results: list[int | Exception] = [None] * 20  # type: ignore[list-item]
        errors: list[Exception] = []

        def worker(idx: int) -> None:
            try:
                results[idx] = get_atomic_number("Si")
            except Exception as exc:
                errors.append(exc)

        with patch("mendeleev.element", side_effect=_mock_get_element):
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert not errors, f"Threads raised exceptions: {errors}"
        assert all(r == 14 for r in results), f"Unexpected results: {results}"

    def test_concurrent_different_elements(self):
        """Threads calling get_atomic_number for different elements get correct values."""
        from xraylabtool.utils import get_atomic_number

        symbols = ["H", "C", "N", "O", "Si", "Fe"]
        expected = {"H": 1, "C": 6, "N": 7, "O": 8, "Si": 14, "Fe": 26}
        results: dict[str, list[int]] = {s: [] for s in symbols}
        lock = threading.Lock()

        def worker(symbol: str) -> None:
            result = get_atomic_number(symbol)
            with lock:
                results[symbol].append(result)

        with patch("mendeleev.element", side_effect=_mock_get_element):
            threads = [
                threading.Thread(target=worker, args=(sym,)) for sym in symbols * 5
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        for sym, values in results.items():
            assert all(v == expected[sym] for v in values), (
                f"Inconsistent results for {sym}: {values}"
            )

    def test_cache_clear_during_concurrent_access_does_not_crash(self):
        """Clearing the cache while threads are accessing it does not cause crashes."""
        from xraylabtool.utils import get_atomic_number

        stop_event = threading.Event()
        errors: list[Exception] = []

        def reader() -> None:
            while not stop_event.is_set():
                try:
                    get_atomic_number("O")
                except Exception as exc:
                    errors.append(exc)
                    break

        def cache_clearer() -> None:
            for _ in range(10):
                get_atomic_number.cache_clear()

        with patch("mendeleev.element", side_effect=_mock_get_element):
            reader_threads = [threading.Thread(target=reader) for _ in range(5)]
            clearer = threading.Thread(target=cache_clearer)

            for t in reader_threads:
                t.start()
            clearer.start()
            clearer.join()
            stop_event.set()
            for t in reader_threads:
                t.join()

        assert not errors, f"Threads raised exceptions during cache clear: {errors}"


class TestConcurrentGetAtomicWeight:
    def test_all_threads_get_correct_result(self):
        """Multiple threads calling get_atomic_weight simultaneously return correct values."""
        from xraylabtool.utils import get_atomic_weight

        results: list[float | None] = [None] * 20
        errors: list[Exception] = []

        def worker(idx: int) -> None:
            try:
                results[idx] = get_atomic_weight("C")
            except Exception as exc:
                errors.append(exc)

        with patch("mendeleev.element", side_effect=_mock_get_element):
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert not errors, f"Threads raised exceptions: {errors}"
        assert all(abs(r - 12.011) < 1e-6 for r in results), (
            f"Unexpected results: {results}"
        )

    def test_cache_clear_during_concurrent_weight_access_does_not_crash(self):
        """Clearing get_atomic_weight cache while threads are active does not crash."""
        from xraylabtool.utils import get_atomic_weight

        stop_event = threading.Event()
        errors: list[Exception] = []

        def reader() -> None:
            while not stop_event.is_set():
                try:
                    get_atomic_weight("H")
                except Exception as exc:
                    errors.append(exc)
                    break

        def cache_clearer() -> None:
            for _ in range(10):
                get_atomic_weight.cache_clear()

        with patch("mendeleev.element", side_effect=_mock_get_element):
            reader_threads = [threading.Thread(target=reader) for _ in range(5)]
            clearer = threading.Thread(target=cache_clearer)

            for t in reader_threads:
                t.start()
            clearer.start()
            clearer.join()
            stop_event.set()
            for t in reader_threads:
                t.join()

        assert not errors, f"Threads raised exceptions during cache clear: {errors}"


class TestCacheResultConsistency:
    def test_lru_cache_returns_same_object_for_same_input(self):
        """LRU cache returns the same cached object for repeated calls."""
        from xraylabtool.utils import get_atomic_number

        with patch("mendeleev.element", side_effect=_mock_get_element):
            result1 = get_atomic_number("Fe")
            result2 = get_atomic_number("Fe")

        assert result1 == result2 == 26

    def test_high_contention_mixed_functions(self):
        """Simultaneous access to both get_atomic_number and get_atomic_weight is safe."""
        from xraylabtool.utils import get_atomic_number, get_atomic_weight

        errors: list[Exception] = []
        results: list[tuple[int, float]] = []
        lock = threading.Lock()

        def worker() -> None:
            try:
                num = get_atomic_number("N")
                weight = get_atomic_weight("N")
                with lock:
                    results.append((num, weight))
            except Exception as exc:
                with lock:
                    errors.append(exc)

        with patch("mendeleev.element", side_effect=_mock_get_element):
            threads = [threading.Thread(target=worker) for _ in range(30)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert not errors, f"Errors in mixed-function concurrent test: {errors}"
        assert all(num == 7 and abs(w - 14.007) < 1e-6 for num, w in results)
