"""Launch the Qt GUI for XRayLabTool."""

from __future__ import annotations

import argparse
import os
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from xraylabtool.logging_utils import configure_logging, get_logger, log_environment

from .main_window import MainWindow
from .theme_manager import ThemeManager


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

    # Configure logging early so Qt output is captured
    configure_logging()
    logger = get_logger("gui")
    log_environment(logger, component="gui")

    if args.platform:
        os.environ.setdefault("QT_QPA_PLATFORM", args.platform)
    # Silence noisy offscreen plugin info/debug logs when running headless
    os.environ.setdefault(
        "QT_LOGGING_RULES", "*.debug=false;*.info=false;qt.qpa.*=false"
    )

    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])

    # Initialize theme manager (handles styling and persistence)
    theme_manager = ThemeManager(app)

    window = MainWindow(theme_manager=theme_manager)
    window.show()

    if args.test_launch:
        # Close shortly after showing to allow CI smoke tests without hanging
        QTimer.singleShot(500, app.quit)

    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    main()
