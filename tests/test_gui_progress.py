import os
import platform
import subprocess
import sys

import pytest

from xraylabtool.gui.services import EnergyConfig, compute_multiple


def test_compute_multiple_reports_progress_to_100():
    values: list[int] = []

    def cb(v: int) -> None:
        values.append(v)

    cfg = EnergyConfig(start_kev=8.0, end_kev=10.0, points=2, logspace=False)
    compute_multiple(["Si", "Cu"], [2.33, 8.96], cfg, progress_cb=cb)

    assert values, "progress callback was not invoked"
    assert values[-1] == 100, "progress did not reach 100%"
    assert all(values[i] <= values[i + 1] for i in range(len(values) - 1)), (
        "progress not monotonic"
    )


@pytest.mark.skipif(
    platform.system().lower() != "linux",
    reason="offscreen Qt smoke limited to Linux runners",
)
def test_headless_gui_smoke_runs_offscreen(tmp_path):
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONPATH", env.get("PYTHONPATH", ""))
    # Ensure artifacts land somewhere harmless
    env.setdefault("TMPDIR", str(tmp_path))

    subprocess.run(
        [sys.executable, "-m", "scripts.gui_headless_smoke"],
        check=True,
        timeout=30,
        env=env,
    )
