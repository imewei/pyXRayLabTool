from __future__ import annotations

from typing import Any

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import (
    QFileDialog,  # re-exported: tests patch main_window.QFileDialog
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QStatusBar,
    QTabWidget,
)

from xraylabtool.logging_utils import get_logger

from .multi_tab import MultiTabMixin
from .shared import SharedHelpersMixin
from .single_tab import PROPERTIES, SingleTabMixin
from .toast import Toast

__all__ = ["PROPERTIES", "MainWindow", "Toast"]

logger = get_logger("xraylabtool.gui.main_window")


class MainWindow(SingleTabMixin, MultiTabMixin, SharedHelpersMixin, QMainWindow):
    def __init__(self, theme_manager: Any | None = None) -> None:
        super().__init__()
        self.theme_manager = theme_manager
        self.setWindowTitle("XRayLabTool GUI")
        self.resize(1100, 720)
        self.setMinimumSize(900, 620)

        self.threadpool: QThreadPool | None = (
            None  # Assigned on first use to avoid import cycles
        )

        self.status_bar = QStatusBar()
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(18)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        self.status_bar.addPermanentWidget(self.progress)

        self.log_path_label = QLabel()
        self.log_path_label.setVisible(False)
        self.status_bar.addPermanentWidget(self.log_path_label)

        self.log_path_toggle = QPushButton("Log path")
        self.log_path_toggle.setProperty("class", "secondary")
        self.log_path_toggle.setToolTip("Show or hide the current log file path")
        self.log_path_toggle.setCheckable(True)
        self.log_path_toggle.setChecked(False)
        self.log_path_toggle.clicked.connect(self._toggle_log_path)
        self.status_bar.addPermanentWidget(self.log_path_toggle)

        self.theme_toggle = QPushButton("Light Mode")
        self.theme_toggle.setProperty("class", "secondary")
        self.theme_toggle.setCheckable(True)
        if self.theme_manager:
            curr = self.theme_manager.get_theme()
            is_dark = curr == "dark"
            self.theme_toggle.setChecked(is_dark)
            self.theme_toggle.setText("Dark Mode" if is_dark else "Light Mode")
            self.theme_toggle.clicked.connect(self._handle_theme_toggle_click)
            self.theme_manager.theme_changed.connect(self._on_theme_changed)
        else:
            self.theme_toggle.setEnabled(False)
        self.status_bar.addPermanentWidget(self.theme_toggle)

        self.status_bar.setSizeGripEnabled(True)
        self.setStatusBar(self.status_bar)

        self.toast = Toast(self)

        self.single_result = None
        self.multi_results = None
        self.multi_comparison = None
        self._workers: list[Any] = []

        self.material_presets = {
            "Si": 2.33,
            "SiO2": 2.2,
            "Al2O3": 3.95,
            "C": 3.52,
            "Au": 19.3,
            "Pt": 21.45,
            "Rh": 12.4,
            "Pd": 12.0,
            "CaCO3": 2.71,
        }
        self.energy_presets = {
            "10 keV": (10.0, 10.0, 1, False),
            "Cu Kalpha (8.048 keV)": (8.048, 8.048, 1, False),
            "1-30 keV log (100)": (1.0, 30.0, 100, True),
            "5-25 keV log (50)": (5.0, 25.0, 50, True),
        }

        self.main_tabs = QTabWidget()
        self.main_tabs.addTab(self._build_single_tab(), "Single Material")
        self.main_tabs.addTab(self._build_multi_tab(), "Multiple Materials")
        self.setCentralWidget(self.main_tabs)
        self._set_tab_order()
        self._tune_table_headers()
