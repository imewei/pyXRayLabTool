"""Unmocked ``xraylabtool kk``: in-process ``main()`` and a real subprocess."""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import jax
import numpy as np
import pytest

from xraylabtool.backend import runtime
from xraylabtool.interfaces.cli import main
from xraylabtool.kk import kk_calculate_real, load_nearedge

ROOT = Path(__file__).resolve().parents[2]
EDGE = ROOT / "tests" / "characterization" / "kk" / "nearedge_example.txt"
# main() reads sys.argv[1:], which after ``-c`` is exactly ``argv``.
# The reference kk_calculate_real runs on this process's default device, so the
# subprocess must use the same platform (a GPU host must not compare cpu to cuda).
_PLATFORM = {"gpu": "cuda"}.get(jax.default_backend(), jax.default_backend())
_ENTRY = "import sys; from xraylabtool.interfaces.cli import main; sys.exit(main())"
EDGE_ARGS = ["--nearedge", str(EDGE), "--input-type", "nexafs"]


def _in_process(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> tuple[int, str, str]:
    with patch("sys.argv", ["xraylabtool", *argv]):
        rc = main()
    out = capsys.readouterr()
    return rc, out.out, out.err


def _subprocess(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _ENTRY, *argv],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ, "JAX_PLATFORMS": _PLATFORM},
        timeout=300,
        check=False,
    )


def _table(text: str) -> dict[str, np.ndarray]:
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["energy_keV", "f1", "f2"]
    data = np.array(rows[1:], dtype=float)
    assert np.isfinite(data).all()
    return dict(zip(rows[0], data.T, strict=True))


def _assert_matches(table: dict[str, np.ndarray], ref) -> None:
    """Grid, f1 and f2 of a CLI table equal the API result at rtol 1e-12."""
    np.testing.assert_allclose(table["energy_keV"], ref.energy_kev, rtol=1e-12)
    np.testing.assert_allclose(table["f1"], ref.f1, rtol=1e-12)
    np.testing.assert_allclose(table["f2"], ref.f2, rtol=1e-12)


@pytest.mark.integration
@pytest.mark.parametrize("runner", ["in_process", "subprocess"])
class TestKKCliEndToEnd:
    @staticmethod
    def run(
        runner: str, argv: list[str], capsys: pytest.CaptureFixture[str]
    ) -> tuple[int, str, str]:
        if runner == "in_process":
            return _in_process(argv, capsys)
        done = _subprocess(argv)
        return done.returncode, done.stdout, done.stderr

    def test_compound_default_grid(self, runner, capsys):
        rc, out, _ = self.run(runner, ["kk", "SiO2"], capsys)
        assert rc == 0
        _assert_matches(_table(out), kk_calculate_real("SiO2"))

    def test_energy_list_matches_api(self, runner, capsys):
        rc, out, _ = self.run(runner, ["kk", "Si", "--energies", "1,2,5"], capsys)
        assert rc == 0
        ref = kk_calculate_real("Si", energies_kev=[1.0, 2.0, 5.0])
        np.testing.assert_allclose(_table(out)["f1"], ref.f1, rtol=1e-12)

    def test_refine(self, runner, capsys):
        rc, out, err = self.run(runner, ["kk", "Si", "--refine", "0.05"], capsys)
        assert rc == 0
        ref = kk_calculate_real("Si", refine_tolerance=0.05)
        assert f"Refinement added {ref.n_refined_points} points" in err
        table = _table(out)
        assert ref.n_refined_points > 0
        assert table["energy_keV"].size > kk_calculate_real("Si").energy_kev.size
        _assert_matches(table, ref)

    def test_nearedge_with_merge(self, runner, capsys):
        argv = ["kk", "C8H8", *EDGE_ARGS, "--merge", "280,320"]
        rc, out, _ = self.run(runner, argv, capsys)
        assert rc == 0
        ref = kk_calculate_real(
            "C8H8",
            nearedge=load_nearedge(str(EDGE), energy_unit="eV"),
            input_type="nexafs",
            merge_points_kev=(0.28, 0.32),
        )
        # Ignoring --nearedge/--merge would give the plain tabulated grid and f1.
        assert ref.merge_points_kev == (0.28, 0.32)
        assert ref.energy_kev.size != kk_calculate_real("C8H8").energy_kev.size
        _assert_matches(_table(out), ref)

    @pytest.mark.parametrize("fmt", ["csv", "json"])
    def test_output_file_round_trip(self, runner, capsys, tmp_path, fmt):
        path = tmp_path / f"out.{fmt}"
        argv = ["kk", "Si", "--energies", "1,2", "-o", str(path), "--format", fmt]
        rc, out, _ = self.run(runner, argv, capsys)
        assert rc == 0
        assert out.strip() == ""
        ref = kk_calculate_real("Si", energies_kev=[1.0, 2.0])
        if fmt == "json":
            data = json.loads(path.read_text())
            assert data["energy_keV"] == [1.0, 2.0]
            f1 = np.array(data["f1"])
        else:
            f1 = _table(path.read_text())["f1"]
        np.testing.assert_allclose(f1, ref.f1, rtol=1e-12)

    @pytest.mark.parametrize(
        ("argv", "needle"),
        [
            (["kk", "Xx", "--energies", "1"], "Xx"),
            (["kk", "Si", "--energies", "nan"], "nergies"),
            (["kk", "Si", "--energies", "1000"], "outside"),
            (["kk", "Si", "--energies", "0.001"], "outside"),
            (
                [
                    "kk",
                    "C",
                    "--nearedge",
                    "no-such-file.txt",
                    "--input-type",
                    "nexafs",
                ],
                "no-such-file",
            ),
            (["kk", "Si", "--merge", "1,2"], "--merge"),
        ],
    )
    def test_errors_exit_1_with_message(self, runner, capsys, argv, needle):
        rc, out, err = self.run(runner, argv, capsys)
        assert rc == 1
        assert out.strip() == ""
        assert needle.lower() in err.lower()

    def test_flat_nearedge_end_errors(self, runner, capsys, tmp_path):
        flat = tmp_path / "flat.txt"
        flat.write_text("".join(f"{e} 3.0\n" for e in range(270, 341, 10)))
        argv = ["kk", "C", "--nearedge", str(flat), "--input-type", "asf"]
        rc, _, err = self.run(runner, [*argv, "--merge", "280,320"], capsys)
        assert rc == 1
        assert "flat endpoints" in err

    def test_bad_format_exits_1(self, runner, capsys):
        # argparse rejects the choice; the CLI maps its exit to 1 with the message.
        rc, out, err = self.run(runner, ["kk", "Si", "--format", "xml"], capsys)
        assert rc == 1
        assert out.strip() == ""
        assert "invalid choice" in err


@pytest.mark.integration
def test_version_unaffected() -> None:
    done = _subprocess(["--version"])
    assert done.returncode == 0
    assert "jax" not in done.stderr.lower()


@pytest.mark.integration
def test_cli_writes_the_persistent_compile_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI enables the cache by default: compiled programs land in the cache dir."""
    # Every variable the OS-specific cache location reads (Linux: XDG_CACHE_HOME,
    # macOS: HOME, Windows: LOCALAPPDATA, with USERPROFILE for Path.home()), so
    # the subprocess and the expected path resolve under tmp_path on all three.
    redirect = {
        "XDG_CACHE_HOME": str(tmp_path),
        "HOME": str(tmp_path),
        "USERPROFILE": str(tmp_path),
        "LOCALAPPDATA": str(tmp_path),
    }
    for name, value in redirect.items():
        monkeypatch.setenv(name, value)
    cache_dir = runtime._user_cache_dir()
    env = {k: v for k, v in os.environ.items() if k != "XRAYLABTOOL_NO_JAX_CACHE"}
    env.update(JAX_PLATFORMS="cpu", PYTHONPATH=str(ROOT))
    done = subprocess.run(
        [sys.executable, "-c", _ENTRY, "kk", "Si", "--energies", "1,2"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert tmp_path in cache_dir.parents
    assert any(cache_dir.iterdir())
