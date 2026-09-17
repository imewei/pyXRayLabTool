"""Typing-only stand-in for the full MainWindow surface, used by mixin methods.

MainWindow is composed from several mixins (single_tab, multi_tab, shared)
that each implement part of the window and read/write attributes owned by
the others (e.g. ``single_tab`` calls ``self._info(...)``, defined in
``shared``). At runtime this works via ordinary multiple inheritance, but
mypy checks each mixin class in isolation and doesn't know about its
siblings.

``MainWindowProtocol`` declares the combined surface: the extra attributes
and methods the mixins add, plus the handful of QMainWindow/QWidget builtins
they call directly (``findChildren``, ``setTabOrder``). It is a pure
``Protocol`` (structural, no concrete base) so the "swap self's type" trick
below is valid: mypy explicitly allows annotating a mixin method's ``self``
with an unrelated Protocol, without requiring the mixin to actually inherit
it. It is imported only under ``TYPE_CHECKING`` and used solely to annotate
mixin methods' ``self`` parameter, so it has no runtime effect and never
appears in a real MRO.

Because a Protocol isn't nominally a ``QWidget``/``QObject``, call sites that
hand ``self`` to a Qt API expecting one of those (``QMessageBox.critical``,
``QFileDialog.getSaveFileName``, ``OverlayScrollbarMarginHelper(self, ...)``)
still need a local ``cast(QWidget, self)`` — the protocol only fixes
attribute/method lookups, not nominal Qt typing.
"""

from __future__ import annotations

from typing import Any, Protocol

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTabWidget,
    QWidget,
)

from ..services import EnergyConfig
from ..widgets.material_form import MaterialInputForm
from ..widgets.material_table import MaterialTable
from ..widgets.plot_canvas import PlotCanvas
from ..widgets.sweep_plots import F1F2Plot, MultiF1F2Plot
from .toast import Toast


class MainWindowProtocol(Protocol):
    # Window chrome
    theme_manager: Any
    theme_toggle: QPushButton
    status_bar: QStatusBar
    progress: QProgressBar
    log_path_label: QLabel
    log_path_toggle: QPushButton
    toast: Toast
    threadpool: QThreadPool | None
    _workers: list[Any]
    _scroll_overlay_helpers: list[Any]

    # Shared data/config
    single_result: Any
    multi_results: Any
    multi_comparison: Any
    material_presets: dict[str, float]
    energy_presets: dict[str, tuple[float, float, int, bool]]

    # Single tab
    single_form: MaterialInputForm
    single_preset: QComboBox
    energy_preset: QComboBox
    single_property: QComboBox
    single_logx: QCheckBox
    single_logy: QCheckBox
    single_save_png: QPushButton
    single_export_csv: QPushButton
    single_summary: QTableWidget
    single_table: QTableWidget
    single_plot: PlotCanvas
    single_f1f2: F1F2Plot
    single_plot_tabs: QTabWidget
    single_plot_scroll: QScrollArea
    conv_energy: QDoubleSpinBox
    conv_wavelength: QDoubleSpinBox

    # Multi tab
    multi_formula: QLineEdit
    multi_density: QDoubleSpinBox
    multi_preset: QComboBox
    multi_table: MaterialTable
    multi_energy_start: QDoubleSpinBox
    multi_energy_end: QDoubleSpinBox
    multi_energy_points: QSpinBox
    multi_logspace: QCheckBox
    multi_property: QComboBox
    multi_compute_btn: QPushButton
    multi_save_png: QPushButton
    multi_export_csv: QPushButton
    multi_logx: QCheckBox
    multi_logy: QCheckBox
    multi_plot: PlotCanvas
    multi_f1f2_plot: MultiF1F2Plot
    multi_plot_tabs: QTabWidget
    multi_full_table: QTableWidget
    multi_plot_scroll: QScrollArea

    # Qt builtins the mixins call directly
    def findChildren(self, *args: Any, **kwargs: Any) -> list[QWidget]: ...
    def setTabOrder(self, first: QWidget, second: QWidget) -> None: ...

    # Cross-mixin methods
    def _info(self, message: str) -> None: ...
    def _error(self, message: str) -> None: ...
    def _show_progress(self, active: bool, value: int = ...) -> None: ...
    def _track_worker(self, worker: Any) -> None: ...
    def _cleanup_worker(self, worker: Any) -> None: ...
    def _label_for_property(self, prop: str) -> str: ...
    def _sanitize_filename(self, text: str) -> str: ...
    def _save_plot(self, plot_widget: QWidget, suggested: str) -> None: ...
    def _render_plot_to_png(self, plot_widget: QWidget, path: str) -> bool: ...
    def _reserve_overlay_scrollbar_space(self, scroll_area: QScrollArea) -> None: ...
    def _refresh_plots(self) -> None: ...

    # Single tab
    def _run_single(self) -> None: ...
    def _on_single_finished(self, result: Any) -> None: ...
    def _on_single_error(self, message: str) -> None: ...
    def _refresh_single_views(self) -> None: ...
    def _apply_single_preset(self, name: str) -> None: ...
    def _apply_energy_preset(self, name: str) -> None: ...
    def _convert_e2w(self) -> None: ...
    def _convert_w2e(self) -> None: ...
    def _save_single_png(self) -> None: ...
    def _export_single_csv(self) -> str | None: ...

    # Multi tab
    def _add_material(self) -> None: ...
    def _add_multi_preset(self, name: str) -> None: ...
    def _remove_material(self) -> None: ...
    def _multi_energy_cfg(self) -> EnergyConfig: ...
    def _run_multi(self) -> None: ...
    def _on_multi_finished(self, results: Any) -> None: ...
    def _on_multi_error(self, message: str) -> None: ...
    def _progress_multi(self, value: int) -> None: ...
    def _refresh_multi_views(self) -> None: ...
    def _save_multi_png(self) -> None: ...
    def _export_multi_csv(self) -> str | None: ...
