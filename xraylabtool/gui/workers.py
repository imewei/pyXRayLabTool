"""Threaded workers for GUI calculations."""

from __future__ import annotations

from collections.abc import Callable
import traceback
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(int)


class CalculationWorker(QRunnable):
    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:  # pragma: no cover - thin wrapper
        try:
            # Inject progress callback if the target function accepts it
            if "progress_cb" in self.fn.__code__.co_varnames:
                self.kwargs.setdefault("progress_cb", self.signals.progress.emit)
            result = self.fn(*self.args, **self.kwargs)
        except Exception as exc:
            tb = traceback.format_exc()
            self.signals.error.emit(f"{exc}\n{tb}")
            return
        self.signals.finished.emit(result)
