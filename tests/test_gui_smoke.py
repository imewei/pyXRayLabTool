from __future__ import annotations

import os
import time

from PySide6.QtCore import QThreadPool
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from xraylabtool.gui.logging_filters import suppress_qt_noise
from xraylabtool.gui.main_window import MainWindow
from xraylabtool.gui.style import apply_styles


def _ensure_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        # Silence noisy logs in offscreen mode
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault(
            "QT_LOGGING_RULES", "*.debug=false;*.info=false;qt.qpa.*=false"
        )
        app = QApplication([])
        apply_styles(app)
    return app


def test_gui_smoke_offscreen() -> None:
    app = _ensure_app()
    with suppress_qt_noise():
        win = MainWindow()
        win.show()

    # Prime with presets
    win.single_preset.setCurrentText("Si")
    win.energy_preset.setCurrentText("5-25 keV log (50)")
    win.multi_preset.setCurrentText("Si")
    win.multi_preset.setCurrentText("Au")

    # Compute synchronously for determinism in CI
    from xraylabtool.gui.services import EnergyConfig, compute_multiple, compute_single

    cfg = EnergyConfig(5.0, 25.0, 50, True)
    single_res = compute_single("Si", 2.33, cfg)
    win._on_single_finished(single_res)
    multi_res = compute_multiple(["Si", "Au"], [2.33, 19.3], cfg)
    win._on_multi_finished(multi_res)

    app.processEvents()

    # Scrollbars should have a real range at small window sizes (avoid clipped plots)
    win.resize(900, 620)
    app.processEvents()
    win.main_tabs.setCurrentIndex(0)
    app.processEvents()
    assert win.single_plot_scroll.verticalScrollBar().maximum() > 0
    win.main_tabs.setCurrentIndex(1)
    app.processEvents()
    assert win.multi_plot_scroll.verticalScrollBar().maximum() > 0

    # Basic sanity: results + tables populated
    assert win.single_result is not None, "Single calculation did not finish"
    assert win.single_table.rowCount() > 0, "Single table not populated"
    assert win.multi_results, "Multi calculation did not finish"
    assert win.multi_full_table.rowCount() > 0, "Multi full table not populated"

    # Cleanup window
    win.close()


def test_gui_smoke_threaded_offscreen() -> None:
    app = _ensure_app()
    with suppress_qt_noise():
        win = MainWindow()
        win.show()

    win.single_preset.setCurrentText("Si")
    win.energy_preset.setCurrentText("5-25 keV log (50)")
    win.multi_preset.setCurrentText("Si")
    win.multi_preset.setCurrentText("Au")

    # Trigger worker-based computations (threads via CalculationWorker)
    win.single_form.compute_button.click()
    win.multi_compute_btn.click()

    # Wait for threadpool to complete to avoid late signals during teardown
    QThreadPool.globalInstance().waitForDone(10000)

    # Wait until results arrive or timeout
    deadline = time.time() + 20
    while time.time() < deadline:
        app.processEvents()
        QTest.qWait(50)
        if (
            win.single_result is not None
            and win.single_table.rowCount() > 0
            and win.multi_results
            and win.multi_full_table.rowCount() > 0
        ):
            break

    assert win.single_result is not None, "Single threaded calculation did not finish"
    assert win.single_table.rowCount() > 0, "Single table not populated (threaded)"
    assert win.multi_results, "Multi threaded calculation did not finish"
    assert win.multi_full_table.rowCount() > 0, "Multi full table not populated"

    win.resize(900, 620)
    app.processEvents()
    win.main_tabs.setCurrentIndex(0)
    app.processEvents()
    assert win.single_plot_scroll.verticalScrollBar().maximum() > 0
    win.main_tabs.setCurrentIndex(1)
    app.processEvents()
    assert win.multi_plot_scroll.verticalScrollBar().maximum() > 0

    win.close()
