"""Tests for xraylabtool.data_handling.compound_analysis."""

from __future__ import annotations

import pytest

from xraylabtool.data_handling.compound_analysis import (
    COMMON_COMPOUNDS,
    COMPOUND_FAMILIES,
    analyze_element_associations,
    find_similar_compounds,
    get_compound_complexity_score,
    get_compound_family,
    get_compound_frequency_score,
    get_elements_for_compound,
    get_recommended_elements_for_warming,
    parse_chemical_formula,
)

# ---------------------------------------------------------------------------
# parse_chemical_formula
# ---------------------------------------------------------------------------


class TestParseChemicalFormula:
    def test_simple_binary_oxide(self):
        result = parse_chemical_formula("SiO2")
        assert result == {"Si": 1, "O": 2}

    def test_simple_ternary(self):
        result = parse_chemical_formula("Al2O3")
        assert result == {"Al": 2, "O": 3}

    def test_single_element(self):
        result = parse_chemical_formula("Fe")
        assert result == {"Fe": 1}

    def test_parentheses_hydroxyapatite(self):
        # Ca5(PO4)3OH — uses the canonical parser; O count verified via parse_formula
        result = parse_chemical_formula("Ca5(PO4)3OH")
        assert result["Ca"] == 5
        assert result["P"] == 3
        assert result["H"] == 1
        # O = 4*3 = 12 from (PO4)3, plus 1 from OH → 13
        assert result["O"] == 13

    def test_parentheses_magnesium_hydroxide(self):
        result = parse_chemical_formula("Mg(OH)2")
        assert result == {"Mg": 1, "O": 2, "H": 2}

    def test_nested_parentheses(self):
        # Ca10(PO4)6(OH)2 — Ca:10, P:6, O=4*6+2=26, H:2
        result = parse_chemical_formula("Ca10(PO4)6(OH)2")
        assert result["Ca"] == 10
        assert result["P"] == 6
        assert result["H"] == 2
        assert result["O"] == 26

    def test_returns_integer_counts(self):
        result = parse_chemical_formula("Fe2O3")
        for v in result.values():
            assert isinstance(v, int)


# ---------------------------------------------------------------------------
# get_elements_for_compound
# ---------------------------------------------------------------------------


class TestGetElementsForCompound:
    def test_known_compound_sio2(self):
        elements = get_elements_for_compound("SiO2")
        assert elements == ["Si", "O"]

    def test_known_compound_al2o3(self):
        elements = get_elements_for_compound("Al2O3")
        assert "Al" in elements
        assert "O" in elements

    def test_unknown_compound_parsed(self):
        # ZnO is not in COMMON_COMPOUNDS; should be parsed
        assert "ZnO" not in COMMON_COMPOUNDS
        elements = get_elements_for_compound("ZnO")
        assert "Zn" in elements
        assert "O" in elements

    def test_complex_compound(self):
        elements = get_elements_for_compound("CaAl2Si2O8")
        for expected in ("Ca", "Al", "Si", "O"):
            assert expected in elements

    def test_returns_list(self):
        result = get_elements_for_compound("NaCl")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# get_compound_frequency_score
# ---------------------------------------------------------------------------


class TestGetCompoundFrequencyScore:
    def test_sio2_high_score(self):
        score = get_compound_frequency_score("SiO2")
        assert score > 0.8

    def test_known_compound_returns_positive(self):
        score = get_compound_frequency_score("Al2O3")
        assert score > 0.0

    def test_unknown_compound_low_score(self):
        # XeF6 — all rare/exotic elements not in common set
        score = get_compound_frequency_score("XeF6")
        assert score <= 0.5

    def test_score_in_range(self):
        for formula in list(COMMON_COMPOUNDS.keys())[:5]:
            score = get_compound_frequency_score(formula)
            assert 0.0 <= score <= 1.0

    def test_unknown_all_common_elements_capped(self):
        # SiAlO (unknown) has Si, Al, O — all common → score ≤ 0.5
        assert "SiAlO" not in COMMON_COMPOUNDS
        score = get_compound_frequency_score("SiAlO")
        assert score <= 0.5


# ---------------------------------------------------------------------------
# find_similar_compounds
# ---------------------------------------------------------------------------


class TestFindSimilarCompounds:
    def test_sio2_finds_si_o_compounds(self):
        results = find_similar_compounds("SiO2", similarity_threshold=0.3)
        # SiO2 itself should appear (Jaccard = 1.0)
        assert "SiO2" in results
        # Other Si-O compounds should appear
        si_o_compounds = [
            c
            for c in results
            if "Si" in COMMON_COMPOUNDS.get(c, {})
            and "O" in COMMON_COMPOUNDS.get(c, {})
        ]
        assert len(si_o_compounds) > 1

    def test_lower_threshold_returns_more(self):
        low = find_similar_compounds("SiO2", similarity_threshold=0.1)
        high = find_similar_compounds("SiO2", similarity_threshold=0.9)
        assert len(low) >= len(high)

    def test_unknown_exotic_returns_empty_or_small(self):
        # A compound with very rare elements should match few/none
        results = find_similar_compounds("OsIr", similarity_threshold=0.9)
        # Expect very few matches at high threshold
        assert len(results) <= 2

    def test_result_is_list(self):
        result = find_similar_compounds("NaCl")
        assert isinstance(result, list)

    def test_similarity_at_threshold_1_returns_exact_only(self):
        # Only exact element-set matches at threshold=1.0
        results = find_similar_compounds("NaCl", similarity_threshold=1.0)
        # NaCl has {Na, Cl}; only compounds with exactly {Na, Cl} qualify
        for formula in results:
            elements = set(COMMON_COMPOUNDS.get(formula, {}).keys())
            assert elements == {"Na", "Cl"}


# ---------------------------------------------------------------------------
# get_compound_family
# ---------------------------------------------------------------------------


class TestGetCompoundFamily:
    def test_sio2_is_silicate(self):
        assert get_compound_family("SiO2") == "silicates"

    def test_nacl_is_halide(self):
        assert get_compound_family("NaCl") == "halides"

    def test_caco3_is_carbonate(self):
        assert get_compound_family("CaCO3") == "carbonates"

    def test_al2o3_is_oxide(self):
        assert get_compound_family("Al2O3") == "oxides"

    def test_unknown_with_si_and_o(self):
        # A novel silicate not in the dict
        assert "MgSi2O5" not in COMPOUND_FAMILIES.get("silicates", [])
        family = get_compound_family("MgSi2O5")
        assert family == "silicates"

    def test_unknown_binary_oxide(self):
        # Two-element compound with O → oxides
        family = get_compound_family("SnO2")
        assert family == "oxides"

    def test_unknown_returns_none_or_str(self):
        result = get_compound_family("UnknownXY")
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# get_recommended_elements_for_warming
# ---------------------------------------------------------------------------


class TestGetRecommendedElementsForWarming:
    def test_includes_elements_from_compounds(self):
        result = get_recommended_elements_for_warming(["SiO2", "Al2O3"])
        assert "Si" in result
        assert "Al" in result
        assert "O" in result

    def test_empty_list_returns_common_elements(self):
        result = get_recommended_elements_for_warming([])
        # Common elements get a baseline score of 0.1
        common = {"Si", "O", "Al", "Fe", "Ca", "Mg", "Ti", "C", "N", "H"}
        assert len(set(result) & common) > 0

    def test_max_elements_respected(self):
        many_compounds = list(COMMON_COMPOUNDS.keys())[:10]
        result = get_recommended_elements_for_warming(many_compounds, max_elements=5)
        assert len(result) <= 5

    def test_returns_list_of_strings(self):
        result = get_recommended_elements_for_warming(["SiO2"])
        assert isinstance(result, list)
        assert all(isinstance(e, str) for e in result)


# ---------------------------------------------------------------------------
# analyze_element_associations
# ---------------------------------------------------------------------------


class TestAnalyzeElementAssociations:
    def test_oxygen_associates_with_si_and_al(self):
        result = analyze_element_associations(["SiO2", "Al2O3"])
        assert "O" in result
        assert "Si" in result["O"]
        assert "Al" in result["O"]

    def test_self_association_excluded(self):
        result = analyze_element_associations(["SiO2", "Al2O3"])
        for elem, associates in result.items():
            assert elem not in associates

    def test_empty_input(self):
        result = analyze_element_associations([])
        assert result == {}

    def test_single_element_compound(self):
        # Fe has no other elements to associate with
        result = analyze_element_associations(["Fe"])
        assert result == {}

    def test_returns_dict_of_lists(self):
        result = analyze_element_associations(["NaCl"])
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, list)


# ---------------------------------------------------------------------------
# get_compound_complexity_score
# ---------------------------------------------------------------------------


class TestGetCompoundComplexityScore:
    def test_simple_less_than_complex(self):
        simple = get_compound_complexity_score("NaCl")
        complex_ = get_compound_complexity_score("CaAl2Si2O8")
        assert simple < complex_

    def test_returns_float(self):
        score = get_compound_complexity_score("SiO2")
        assert isinstance(score, float)

    def test_positive_for_valid_formula(self):
        assert get_compound_complexity_score("Fe2O3") > 0.0

    def test_more_elements_increases_score(self):
        binary = get_compound_complexity_score("NaCl")  # 2 elements
        ternary = get_compound_complexity_score("CaCO3")  # 3 elements
        assert ternary > binary

    def test_anorthite_greater_than_quartz(self):
        quartz = get_compound_complexity_score("SiO2")
        anorthite = get_compound_complexity_score("CaAl2Si2O8")
        assert anorthite > quartz
