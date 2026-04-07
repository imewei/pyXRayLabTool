"""Overlay scrollbar margin helper for QScrollArea widgets."""

from __future__ import annotations

import contextlib
import types

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QTimer
from PySide6.QtWidgets import QScrollArea


class OverlayScrollbarMarginHelper(QObject):
    """Adjusts viewport margins so overlay scrollbars don't cover content.

    Install on a QScrollArea to automatically add right margin when the
    vertical scrollbar overlaps the viewport (common with overlay-style
    scrollbars on macOS).
    """

    def __init__(self, parent: QObject, target: QScrollArea) -> None:
        super().__init__(parent)
        self._scroll_area = target
        self._bar = target.verticalScrollBar()
        self._active = True
        self._scheduled = False
        target.destroyed.connect(lambda *_args: self._deactivate())
        self._bar.destroyed.connect(lambda *_args: self._deactivate())
        target.installEventFilter(self)
        target.viewport().installEventFilter(self)
        self._bar.installEventFilter(self)

        self._bar.rangeChanged.connect(lambda *_args: self._schedule())
        self._schedule()

    def _deactivate(self) -> None:
        self._active = False

    def _schedule(self) -> None:
        if not self._active or self._scheduled:
            return
        self._scheduled = True
        QTimer.singleShot(0, self.apply_margins)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.Hide):
            self._schedule()
        return super().eventFilter(watched, event)

    def apply_margins(self) -> None:
        self._scheduled = False
        if not self._active:
            return

        shiboken6_mod: types.ModuleType | None = None
        with contextlib.suppress(ImportError):
            import shiboken6 as shiboken6_mod
        shiboken6 = shiboken6_mod

        if shiboken6 is not None and (
            not shiboken6.isValid(self._scroll_area) or not shiboken6.isValid(self._bar)
        ):
            self._active = False
            return

        try:
            if not self._bar.isVisible():
                self._scroll_area.setViewportMargins(0, 0, 0, 0)
                return

            viewport_pos = self._scroll_area.viewport().mapTo(
                self._scroll_area, QPoint(0, 0)
            )
            viewport_rect = QRect(viewport_pos, self._scroll_area.viewport().size())
            overlaps = self._bar.geometry().intersects(viewport_rect)
            margin = self._bar.sizeHint().width() if overlaps else 0
            self._scroll_area.setViewportMargins(0, 0, margin, 0)
        except RuntimeError:
            # Underlying Qt objects may have been deleted during teardown.
            self._active = False
            return
