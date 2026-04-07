"""Tests for CalculationWorker error and success signal paths."""

from __future__ import annotations

import pytest

PySide6 = pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication

from xraylabtool.gui.workers import CalculationWorker


@pytest.fixture(scope="module")
def qt_app():
    app = QCoreApplication.instance() or QCoreApplication([])
    return app


def _run_worker(worker: CalculationWorker) -> None:
    """Run worker synchronously (no thread pool needed for unit tests)."""
    worker.run()


def test_worker_emits_result_on_success(qt_app):
    received = []

    def good_fn(x, y):
        return x + y

    worker = CalculationWorker(good_fn, 3, 4)
    worker.signals.finished.connect(received.append)

    _run_worker(worker)

    assert received == [7]


def test_worker_emits_error_on_exception(qt_app):
    errors = []

    def bad_fn():
        raise ValueError("boom")

    worker = CalculationWorker(bad_fn)
    worker.signals.error.connect(errors.append)

    _run_worker(worker)

    assert len(errors) == 1
    assert "boom" in errors[0]
    assert "ValueError" in errors[0]


def test_worker_does_not_emit_finished_on_exception(qt_app):
    finished = []
    errors = []

    def bad_fn():
        raise RuntimeError("fail")

    worker = CalculationWorker(bad_fn)
    worker.signals.finished.connect(finished.append)
    worker.signals.error.connect(errors.append)

    _run_worker(worker)

    assert finished == []
    assert len(errors) == 1


def test_worker_injects_progress_cb_when_accepted(qt_app):
    progress_values = []

    def fn_with_progress(progress_cb=None):
        if progress_cb:
            progress_cb(50)
        return "done"

    worker = CalculationWorker(fn_with_progress)
    worker.signals.progress.connect(progress_values.append)
    finished = []
    worker.signals.finished.connect(finished.append)

    _run_worker(worker)

    assert finished == ["done"]
    assert 50 in progress_values


def test_worker_passes_kwargs(qt_app):
    received = []

    def fn(a, b, c=0):
        return a + b + c

    worker = CalculationWorker(fn, 1, 2, c=10)
    worker.signals.finished.connect(received.append)

    _run_worker(worker)

    assert received == [13]
