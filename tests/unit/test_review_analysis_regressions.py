"""Comparison statistics must support zero and negative optical properties."""

from xraylabtool.analysis.comparator import MaterialComparator


def test_zero_critical_angles_do_not_abort_material_comparison():
    result = MaterialComparator().compare_materials(["Si", "Si2"], [2.33, 2.4], [0.1])

    assert result.summary_stats["critical_angle_degrees"]["mean"] == 0.0
    assert result.summary_stats["critical_angle_degrees"]["std"] == 0.0
    assert len(result.materials) == 2


def test_negative_dispersion_still_reports_large_material_variation():
    result = MaterialComparator().compare_materials(
        ["Si", "Si"], [1.0, 10.0], [0.1], properties=["dispersion_delta"]
    )

    assert result.summary_stats["dispersion_delta"]["mean"] < 0.0
    assert any(
        "Large variation in dispersion delta" in r for r in result.recommendations
    )
