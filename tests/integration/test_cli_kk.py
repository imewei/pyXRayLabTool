"""Integration tests for the ``kk`` CLI subcommand."""

import csv
from importlib.metadata import entry_points
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from xraylabtool.interfaces.cli import main
from xraylabtool.interfaces.completion_v2.shells import (
    BashCompletionGenerator,
    get_xraylabtool_commands,
)
from xraylabtool.kk import kk_calculate_real, load_nearedge

# Carbon K-edge style NEXAFS rows (eV, absorption), the shape the core docstring uses.
_EDGE_ROWS_EV = [
    (280.0, 0.05),
    (290.0, 0.1),
    (300.0, 0.3),
    (310.0, 0.45),
    (320.0, 0.5),
]


def _write_edge(path: Path, *, kev: bool = False, header: bool = True) -> Path:
    lines = ["% energy absorption"] if header else []
    scale = 1e-3 if kev else 1.0
    lines += [f"{e * scale!r} {v!r}" for e, v in _EDGE_ROWS_EV]
    path.write_text("\n".join(lines) + "\n")
    return path


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    with patch("sys.argv", ["xraylabtool", *argv]):
        rc = main()
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def _parse_csv(text: str) -> dict[str, np.ndarray]:
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["energy_keV", "f1", "f2"]
    data = np.array(rows[1:], dtype=float)
    return {"energy_keV": data[:, 0], "f1": data[:, 1], "f2": data[:, 2]}


@pytest.mark.integration
class TestKKCli:
    def test_kk_tabulated_only_matches_api(self, capsys):
        rc, out, _ = _run(["kk", "Si", "--energies", "1.0,2.0,5.0"], capsys)
        assert rc == 0
        table = _parse_csv(out)
        ref = kk_calculate_real("Si", energies_kev=[1.0, 2.0, 5.0])
        np.testing.assert_allclose(table["energy_keV"], ref.energy_kev, rtol=1e-12)
        np.testing.assert_allclose(table["f1"], ref.f1, rtol=1e-12)
        np.testing.assert_allclose(table["f2"], ref.f2, rtol=1e-12)
        assert np.isfinite(table["f1"]).all()

    def test_default_grid_without_energies(self, capsys):
        rc, out, _ = _run(["kk", "Si"], capsys)
        assert rc == 0
        table = _parse_csv(out)
        ref = kk_calculate_real("Si")
        assert table["energy_keV"].size == ref.energy_kev.size

    def test_nearedge_requires_input_type(self, capsys, tmp_path):
        edge = _write_edge(tmp_path / "edge.txt")
        rc, out, err = _run(["kk", "C", "--nearedge", str(edge)], capsys)
        assert rc == 1
        assert "--input-type" in err
        assert out == ""

    def test_input_type_without_nearedge_errors(self, capsys):
        rc, _, err = _run(["kk", "C", "--input-type", "nexafs"], capsys)
        assert rc == 1
        assert "--nearedge" in err

    def test_merge_without_nearedge_errors(self, capsys):
        rc, _, err = _run(["kk", "C", "--merge", "280,320"], capsys)
        assert rc == 1
        assert "--nearedge" in err

    def test_nearedge_matches_api(self, capsys, tmp_path):
        edge = _write_edge(tmp_path / "edge.txt")
        rc, out, _ = _run(
            [
                "kk",
                "C",
                "--nearedge",
                str(edge),
                "--input-type",
                "nexafs",
                "--energies",
                "0.1,0.29,0.3,1.0",
            ],
            capsys,
        )
        assert rc == 0
        table = _parse_csv(out)
        ref = kk_calculate_real(
            "C",
            energies_kev=[0.1, 0.29, 0.3, 1.0],
            nearedge=load_nearedge(edge),
            input_type="nexafs",
        )
        np.testing.assert_allclose(table["f1"], ref.f1, rtol=1e-12)
        np.testing.assert_allclose(table["f2"], ref.f2, rtol=1e-12)

    def test_energies_are_kev_regardless_of_energy_unit(self, capsys, tmp_path):
        ev_file = _write_edge(tmp_path / "edge_ev.txt")
        kev_file = _write_edge(tmp_path / "edge_kev.txt", kev=True)
        common = ["--input-type", "nexafs", "--energies", "0.29,0.3"]
        rc1, out1, _ = _run(
            ["kk", "C", "--nearedge", str(ev_file), "--energy-unit", "eV", *common],
            capsys,
        )
        rc2, out2, _ = _run(
            ["kk", "C", "--nearedge", str(kev_file), "--energy-unit", "keV", *common],
            capsys,
        )
        assert rc1 == rc2 == 0
        t1, t2 = _parse_csv(out1), _parse_csv(out2)
        np.testing.assert_allclose(t1["energy_keV"], [0.29, 0.3], rtol=1e-12)
        np.testing.assert_allclose(t2["energy_keV"], [0.29, 0.3], rtol=1e-12)
        np.testing.assert_allclose(t1["f1"], t2["f1"], rtol=1e-9)
        np.testing.assert_allclose(t1["f2"], t2["f2"], rtol=1e-9)

    def test_merge_units_follow_energy_unit(self, capsys, tmp_path):
        ev_file = _write_edge(tmp_path / "edge_ev.txt")
        kev_file = _write_edge(tmp_path / "edge_kev.txt", kev=True)
        base = ["--input-type", "nexafs", "--energies", "0.29,0.3"]
        rc1, out1, _ = _run(
            ["kk", "C", "--nearedge", str(ev_file), "--merge", "285,315", *base],
            capsys,
        )
        rc2, out2, _ = _run(
            [
                "kk",
                "C",
                "--nearedge",
                str(kev_file),
                "--energy-unit",
                "keV",
                "--merge",
                "0.285,0.315",
                *base,
            ],
            capsys,
        )
        assert rc1 == rc2 == 0
        ref = kk_calculate_real(
            "C",
            energies_kev=[0.29, 0.3],
            nearedge=load_nearedge(ev_file),
            input_type="nexafs",
            merge_points_kev=(0.285, 0.315),
        )
        np.testing.assert_allclose(_parse_csv(out1)["f1"], ref.f1, rtol=1e-12)
        np.testing.assert_allclose(_parse_csv(out2)["f1"], ref.f1, rtol=1e-9)

        # The same numbers read as keV merge points are far outside the eV file.
        rc3, _, err3 = _run(
            ["kk", "C", "--nearedge", str(ev_file), "--merge", "0.285,0.315", *base],
            capsys,
        )
        assert rc3 == 1
        assert err3.strip() != ""

    def test_bad_merge_spec_errors(self, capsys, tmp_path):
        edge = _write_edge(tmp_path / "edge.txt")
        for spec in ("285", "a,b", "285,315,320", "315,nan"):
            rc, _, err = _run(
                [
                    "kk",
                    "C",
                    "--nearedge",
                    str(edge),
                    "--input-type",
                    "nexafs",
                    "--merge",
                    spec,
                ],
                capsys,
            )
            assert rc == 1, spec
            assert "--merge" in err, spec

    def test_kev_file_declared_ev_errors_cleanly(self, capsys, tmp_path):
        kev_file = _write_edge(tmp_path / "edge_kev.txt", kev=True)
        rc, out, err = _run(
            ["kk", "C", "--nearedge", str(kev_file), "--input-type", "nexafs"], capsys
        )
        assert rc == 1
        assert out == ""
        assert "Error" in err
        assert "eV" in err
        assert "Traceback" not in err

    def test_density_does_not_change_beta_result(self, capsys, tmp_path):
        beta = tmp_path / "beta.txt"
        beta.write_text(
            "\n".join(f"{e!r} {v * 1e-4!r}" for e, v in _EDGE_ROWS_EV) + "\n"
        )
        argv = [
            "kk",
            "C",
            "--nearedge",
            str(beta),
            "--input-type",
            "beta",
            "--energies",
            "0.29,0.3",
        ]
        rc1, out1, _ = _run(argv, capsys)
        rc2, out2, _ = _run([*argv, "--density", "2.2"], capsys)
        assert rc1 == rc2 == 0
        t1, t2 = _parse_csv(out1), _parse_csv(out2)
        np.testing.assert_allclose(t1["f1"], t2["f1"], rtol=1e-9)
        np.testing.assert_allclose(t1["f2"], t2["f2"], rtol=1e-9)

    @pytest.mark.parametrize("density", ["0", "-1", "nan"])
    def test_bad_density_errors(self, capsys, density):
        rc, _, err = _run(["kk", "C", "--density", density], capsys)
        assert rc == 1
        assert "--density" in err

    def test_json_format(self, capsys):
        rc, out, _ = _run(
            ["kk", "Si", "--energies", "1.0,2.0", "--format", "json"], capsys
        )
        assert rc == 0
        payload = json.loads(out)
        assert set(payload) == {"energy_keV", "f1", "f2"}
        ref = kk_calculate_real("Si", energies_kev=[1.0, 2.0])
        np.testing.assert_allclose(payload["f1"], ref.f1, rtol=1e-12)
        np.testing.assert_allclose(payload["energy_keV"], [1.0, 2.0], rtol=1e-12)

    def test_output_file(self, capsys, tmp_path):
        target = tmp_path / "out.csv"
        rc, out, _ = _run(
            ["kk", "Si", "--energies", "1.0,2.0", "-o", str(target)], capsys
        )
        assert rc == 0
        assert out == ""
        table = _parse_csv(target.read_text())
        assert table["energy_keV"].tolist() == [1.0, 2.0]

    def test_output_file_json_inferred_from_extension(self, capsys, tmp_path):
        target = tmp_path / "out.json"
        rc, _, _ = _run(["kk", "Si", "--energies", "1.0", "-o", str(target)], capsys)
        assert rc == 0
        assert set(json.loads(target.read_text())) == {"energy_keV", "f1", "f2"}

    def test_output_file_unwritable_errors(self, capsys, tmp_path):
        rc, _, err = _run(
            ["kk", "Si", "--energies", "1.0", "-o", str(tmp_path / "no" / "out.csv")],
            capsys,
        )
        assert rc == 1
        assert "Error" in err

    @pytest.mark.parametrize("spec", ["nan", "-1", "0", "1.0,nan", "inf"])
    def test_energies_nan_rejected(self, capsys, spec):
        rc, out, err = _run(["kk", "Si", "--energies", spec], capsys)
        assert rc == 1
        assert out == ""
        assert "Error" in err

    def test_energies_out_of_range_errors(self, capsys):
        rc, _, err = _run(["kk", "Si", "--energies", "1000"], capsys)
        assert rc == 1
        assert "Error" in err

    def test_bad_formula_errors(self, capsys):
        rc, _, err = _run(["kk", "Xx9", "--energies", "1.0"], capsys)
        assert rc == 1
        assert "Error" in err

    def test_refine_reports_added_points(self, capsys):
        rc, out, err = _run(
            ["kk", "Si", "--energies", "5.0,0.1", "--refine", "0.05"], capsys
        )
        assert rc == 0
        ref = kk_calculate_real("Si", energies_kev=[5.0, 0.1], refine_tolerance=0.05)
        assert ref.n_refined_points > 0
        assert f"{ref.n_refined_points} point" in err
        # data stays on stdout, the report does not
        assert "refine" not in out.lower()
        table = _parse_csv(out)
        assert table["energy_keV"].size == ref.energy_kev.size

    @pytest.mark.parametrize("tol", ["0", "-0.1", "nan"])
    def test_bad_refine_errors(self, capsys, tol):
        rc, _, err = _run(["kk", "Si", "--energies", "1.0", "--refine", tol], capsys)
        assert rc == 1
        assert "Error" in err

    def test_xlt_alias_runs_kk(self, capsys):
        (script,) = [
            ep for ep in entry_points(group="console_scripts") if ep.name == "xlt"
        ]
        entry = script.load()
        with patch("sys.argv", ["xlt", "kk", "Si", "--energies", "1.0"]):
            rc = entry()
        assert rc == 0
        assert _parse_csv(capsys.readouterr().out)["energy_keV"].tolist() == [1.0]

    def test_completion_lists_kk(self):
        commands = get_xraylabtool_commands()
        assert "kk" in commands
        for option in ("--nearedge", "--input-type", "--energy-unit", "--refine"):
            assert option in commands["kk"]["options"]
        script = BashCompletionGenerator().generate(commands, ["--help"])
        assert "kk" in script.split()

    def test_help_exits_zero(self):
        with (
            patch("sys.argv", ["xraylabtool", "kk", "--help"]),
            patch("sys.stdout"),
            pytest.raises(SystemExit) as excinfo,
        ):
            main()
        assert excinfo.value.code == 0
