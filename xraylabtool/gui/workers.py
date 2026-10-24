"""Threaded workers for GUI calculations."""

from __future__ import annotations

from collections.abc import Callable
import inspect
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

    def _accepts_progress_cb(self) -> bool:
        """True if the task declares a ``progress_cb`` parameter.

        Uses the signature, not ``__code__``: partials, builtins and callable objects
        have no code object, and ``co_varnames`` also lists local variables.
        """
        try:
            return "progress_cb" in inspect.signature(self.fn).parameters
        except (TypeError, ValueError):  # no introspectable signature
            return False

    def run(self) -> None:
        try:
            # Inject progress callback if the target function accepts it
            if self._accepts_progress_cb():
                self.kwargs.setdefault("progress_cb", self.signals.progress.emit)
            result = self.fn(*self.args, **self.kwargs)
        except Exception as exc:
            tb = traceback.format_exc()
            self.signals.error.emit(f"{exc}\n{tb}")
            return
        self.signals.finished.emit(result)
