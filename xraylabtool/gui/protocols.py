from typing import Any, Literal, Protocol


class ThemeProtocol(Protocol):
    """Interface for the Theme Manager."""

    def set_theme(self, mode: Literal["light", "dark"]) -> None:
        """Switch the application theme."""
        ...

    def get_theme(self) -> Literal["light", "dark"]:
        """Get the current active theme."""
        ...

    def toggle_theme(self) -> None:
        """Toggle between light and dark modes."""
        ...

    def apply(self, app_instance: Any) -> None:
        """Apply the current theme stylesheet to the QApplication."""
        ...


class PlotThemeProtocol(Protocol):
    """Interface for Matplotlib Integration."""

    def apply_to_figure(self, figure: Any, mode: Literal["light", "dark"]) -> None:
        """Update a matplotlib figure to match the theme."""
        ...
