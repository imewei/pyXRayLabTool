from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QWidget


class Toast(QLabel):
    """Lightweight, non-blocking toast overlay."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            "background: rgba(15,23,42,0.92); color: white; padding: 8px 12px;"
            "border-radius: 8px;"
        )
        self.setVisible(False)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)
        self._kind_colors = {
            "info": "#00f0ff",
            "success": "#00ff9d",
            "error": "#ff9d00",
        }
        self._durations = {"info": 2000, "success": 2400, "error": 3500}

    def show_toast(
        self, message: str, kind: str = "info", duration_ms: int | None = None
    ) -> None:
        color = self._kind_colors.get(kind, "#2563eb")
        self.setStyleSheet(
            f"background: rgba(15,23,42,0.92); color: white; padding: 8px 12px;"
            f"border: 1px solid {color}; border-radius: 8px;"
        )
        self.setText(message)
        self.adjustSize()
        self._reposition()
        self.show()
        self.raise_()
        duration = (
            duration_ms if duration_ms is not None else self._durations.get(kind, 2200)
        )
        self._timer.start(duration)

    def _reposition(self) -> None:
        parent = self.parentWidget()
        if not parent:
            return
        x = max(8, (parent.width() - self.width()) // 2)
        y = max(8, parent.height() - self.height() - 24)
        self.move(x, y)
