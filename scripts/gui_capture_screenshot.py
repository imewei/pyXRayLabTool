from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication

from xraylabtool.gui.logging_filters import (
    enable_offscreen_quiet_env,
    suppress_qt_noise,
)
from xraylabtool.gui.main_window import MainWindow


def main() -> None:
    enable_offscreen_quiet_env()
    app = QApplication.instance() or QApplication([])

    with suppress_qt_noise():
        window = MainWindow()
        window.show()
        app.processEvents()

    out_path = Path("docs/_static/gui_main_offscreen.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pixmap = window.grab()
    pixmap.save(str(out_path))

    window.close()
    app.quit()


if __name__ == "__main__":
    main()
