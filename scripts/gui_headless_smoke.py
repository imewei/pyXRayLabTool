"""Headless smoke check for the Qt GUI.

This script exercises key Single and Multi flows offscreen and writes PNG/CSV
exports to temporary files to ensure the handlers work without user dialogs.

Usage (headless friendly):

    QT_QPA_PLATFORM=offscreen python -m scripts.gui_headless_smoke

Exit code 0 means the basic flows completed.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PySide6.QtWidgets import QApplication, QFileDialog

from xraylabtool.gui.main_window import MainWindow
from xraylabtool.gui.services import EnergyConfig, compute_multiple, compute_single


def _run_smoke() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])

    window = MainWindow()

    # Prepare results for Single flow
    single_cfg = EnergyConfig(start_kev=8.0, end_kev=12.0, points=3, logspace=False)
    window.single_result = compute_single("Si", 2.33, single_cfg)
    window._refresh_single_views()

    # Prepare results for Multi flow
    multi_cfg = EnergyConfig(start_kev=8.0, end_kev=12.0, points=4, logspace=True)
    window.multi_results = compute_multiple(["Si", "Cu"], [2.33, 8.96], multi_cfg)
    window.multi_comparison = None  # optional comparator; not required for smoke
    window._refresh_multi_views()

    # Monkeypatch QFileDialog to return temporary paths for exports
    tmpdir = Path(tempfile.mkdtemp(prefix="xlt_gui_smoke_"))

    def _fake_get_save_file_name(*_args, **_kwargs):
        # Qt signature: (parent, caption, dir, filter, selectedFilter, options)
        cap_arg = _args[1] if len(_args) > 1 else ""
        dir_arg = _args[2] if len(_args) > 2 else ""
        filt_arg = _args[3] if len(_args) > 3 else ""
        flt = str(_kwargs.get("filter", filt_arg)).lower()
        cap = str(_kwargs.get("caption", cap_arg)).lower()
        base = (
            _kwargs.get("selectFile")
            or dir_arg
            or _kwargs.get("caption")
            or cap
            or "export"
        )
        base_clean = base.replace(" ", "_").lower()
        # If caller already passed an extension, honor it; otherwise append one
        if base_clean.endswith((".csv", ".png")):
            fname = base_clean
        else:
            suffix = ".csv" if "csv" in flt or "csv" in cap else ".png"
            fname = f"{base_clean}{suffix}"
        # ensure uniqueness if multiple calls use same caption
        path = tmpdir / fname
        counter = 1
        while path.exists():
            path = tmpdir / f"{path.stem}_{counter}{path.suffix}"
            counter += 1
        return str(path), ""

    orig_get = QFileDialog.getSaveFileName
    QFileDialog.getSaveFileName = _fake_get_save_file_name  # type: ignore
    try:
        window._save_single_png()
        window._save_multi_png()
        single_csv = window._export_single_csv()
        multi_csv = window._export_multi_csv()
    finally:
        QFileDialog.getSaveFileName = orig_get  # type: ignore

    # Basic CSV sanity checks
    for csv_path in [single_csv, multi_csv]:
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV export missing: {csv_file}")
        if csv_file.stat().st_size == 0:
            raise ValueError(f"CSV export empty: {csv_file}")
        # simple header+row check
        lines = csv_file.read_text().splitlines()
        if len(lines) < 2:
            raise ValueError(f"CSV export has insufficient rows: {csv_file}")

    # Close and clean up
    window.close()
    app.quit()


if __name__ == "__main__":  # pragma: no cover - smoke utility
    _run_smoke()
