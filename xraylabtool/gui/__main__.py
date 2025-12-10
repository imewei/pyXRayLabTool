"""Launch the Qt GUI for XRayLabTool."""

from __future__ import annotations

import argparse
import os
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow
from .style import apply_matplotlib_theme, apply_styles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XRayLabTool desktop GUI")
    parser.add_argument(
        "--test-launch",
        action="store_true",
        help="Create and destroy the GUI immediately (for CI/headless smoke tests)",
    )
    parser.add_argument(
        "--platform",
        help="Override Qt platform (e.g. 'offscreen' for headless runs)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.platform:
        os.environ.setdefault("QT_QPA_PLATFORM", args.platform)
    # Silence noisy offscreen plugin info/debug logs when running headless
    os.environ.setdefault(
        "QT_LOGGING_RULES", "*.debug=false;*.info=false;qt.qpa.*=false"
    )

    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    apply_styles(app)
    apply_matplotlib_theme()

    window = MainWindow()
    window.show()

    if args.test_launch:
        # Close shortly after showing to allow CI smoke tests without hanging
        QTimer.singleShot(500, app.quit)

    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    main()
