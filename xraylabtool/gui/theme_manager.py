from __future__ import annotations

from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtWidgets import QApplication

from .style import (
    DARK_THEME,
    LIGHT_THEME,
    ColorPalette,
    apply_matplotlib_theme,
    apply_theme,
)


class ThemeManager(QObject):
    """Manages application theme state and persistence."""

    theme_changed = Signal(str)  # Emits "light" or "dark"

    def __init__(self, app: QApplication) -> None:
        super().__init__()
        self._app = app
        self._settings = QSettings("AnlXray", "XRayLabTool")
        # Ensure we read as string, QSettings can be tricky with types
        self._current_mode = str(self._settings.value("gui/theme_mode", "light"))

        # Validate stored value
        if self._current_mode not in ("light", "dark"):
            self._current_mode = "light"

    @property
    def current_palette(self) -> ColorPalette:
        """Get the active ColorPalette object."""
        return DARK_THEME if self._current_mode == "dark" else LIGHT_THEME

    def set_theme(self, mode: str) -> None:
        """Set the active theme mode."""
        if mode not in ("light", "dark"):
            return

        if mode == self._current_mode:
            return

        self._current_mode = mode
        self._settings.setValue("gui/theme_mode", mode)
        self.apply(self._app)
        self.theme_changed.emit(mode)

    def get_theme(self) -> str:
        """Get current theme mode string."""
        return self._current_mode

    def toggle_theme(self) -> None:
        """Toggle between light and dark modes."""
        new_mode = "dark" if self._current_mode == "light" else "light"
        self.set_theme(new_mode)

    def apply(self, app_instance: QApplication) -> None:
        """Apply current theme to application and global resources."""
        # Update Qt styles
        apply_theme(app_instance, self.current_palette)

        # Update Matplotlib styles
        apply_matplotlib_theme(self.current_palette)

        # Force palette update on app (sometimes needed for dynamic switches)
        app_instance.setPalette(self.current_palette.to_qpalette())
