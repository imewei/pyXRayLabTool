"""
Element combination analysis for intelligent cache warming.

This module analyzes common chemical compounds and their constituent elements
to enable intelligent cache pre-loading based on typical usage patterns in
materials science applications.
"""

from __future__ import annotations

from collections import defaultdict
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Common compounds in materials science and their constituent elements
# This data is based on frequent usage patterns in X-ray analysis
COMMON_COMPOUNDS: dict[str, dict[str, int]] = {
    # Silicates and ceramics
    "SiO2": {"Si": 1, "O": 2},  # Quartz, silica
    "Al2O3": {"Al": 2, "O": 3},  # Alumina
    "TiO2": {"Ti": 1, "O": 2},  # Titania
    "ZrO2": {"Zr": 1, "O": 2},  # Zirconia
    "CaO": {"Ca": 1, "O": 1},  # Lime
    "MgO": {"Mg": 1, "O": 1},  # Magnesia
    "FeO": {"Fe": 1, "O": 1},  # Iron oxide (wüstite)
    "Fe2O3": {"Fe": 2, "O": 3},  # Iron oxide (hematite)
    "Fe3O4": {"Fe": 3, "O": 4},  # Magnetite
    "Cr2O3": {"Cr": 2, "O": 3},  # Chromium oxide
    # Complex silicates
    "CaSiO3": {"Ca": 1, "Si": 1, "O": 3},  # Wollastonite
    "MgSiO3": {"Mg": 1, "Si": 1, "O": 3},  # Enstatite
    "Al2SiO5": {"Al": 2, "Si": 1, "O": 5},  # Andalusite/Sillimanite
    "CaAl2Si2O8": {"Ca": 1, "Al": 2, "Si": 2, "O": 8},  # Anorthite
    "NaAlSi3O8": {"Na": 1, "Al": 1, "Si": 3, "O": 8},  # Albite
    "KAlSi3O8": {"K": 1, "Al": 1, "Si": 3, "O": 8},  # Orthoclase
    # Carbonates
    "CaCO3": {"Ca": 1, "C": 1, "O": 3},  # Calcite/Limestone
    "MgCO3": {"Mg": 1, "C": 1, "O": 3},  # Magnesite
    "FeCO3": {"Fe": 1, "C": 1, "O": 3},  # Siderite
    "CaMg(CO3)2": {"Ca": 1, "Mg": 1, "C": 2, "O": 6},  # Dolomite
    # Nitrides and carbides
    "Si3N4": {"Si": 3, "N": 4},  # Silicon nitride
    "AlN": {"Al": 1, "N": 1},  # Aluminum nitride
    "TiN": {"Ti": 1, "N": 1},  # Titanium nitride
    "SiC": {"Si": 1, "C": 1},  # Silicon carbide
    "TiC": {"Ti": 1, "C": 1},  # Titanium carbide
    "WC": {"W": 1, "C": 1},  # Tungsten carbide
    # Sulfides
    "ZnS": {"Zn": 1, "S": 1},  # Zinc sulfide
    "CuS": {"Cu": 1, "S": 1},  # Copper sulfide
    "FeS": {"Fe": 1, "S": 1},  # Iron sulfide
    "FeS2": {"Fe": 1, "S": 2},  # Pyrite
    "PbS": {"Pb": 1, "S": 1},  # Galena
    # Halides
    "NaCl": {"Na": 1, "Cl": 1},  # Halite
    "CaF2": {"Ca": 1, "F": 2},  # Fluorite
    "MgF2": {"Mg": 1, "F": 2},  # Magnesium fluoride
    # Phosphates
    "Ca3(PO4)2": {"Ca": 3, "P": 2, "O": 8},  # Calcium phosphate
    "Ca5(PO4)3OH": {"Ca": 5, "P": 3, "O": 13, "H": 1},  # Hydroxyapatite
    # Sulfates
    "CaSO4": {"Ca": 1, "S": 1, "O": 4},  # Gypsum (anhydrite)
    "BaSO4": {"Ba": 1, "S": 1, "O": 4},  # Barite
    "SrSO4": {"Sr": 1, "S": 1, "O": 4},  # Celestine
    # Polymers and organics
    "C2H4": {"C": 2, "H": 4},  # Ethylene (polymer precursor)
    "C8H8": {"C": 8, "H": 8},  # Styrene
    "C6H6": {"C": 6, "H": 6},  # Benzene
    # Alloys and intermetallics (represented as compounds)
    "FeC": {"Fe": 1, "C": 1},  # Steel approximation
    "CuZn": {"Cu": 1, "Zn": 1},  # Brass approximation
    "AlCu": {"Al": 1, "Cu": 1},  # Aluminum alloy
}

# Element families for group-based cache warming
ELEMENT_FAMILIES: dict[str, list[str]] = {
    "alkali_metals": ["Li", "Na", "K", "Rb", "Cs"],
    "alkaline_earth": ["Be", "Mg", "Ca", "Sr", "Ba"],
    "transition_metals": ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"],
    "noble_metals": ["Ru", "Rh", "Pd", "Ag", "Os", "Ir", "Pt", "Au"],
    "lanthanides": [
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
    ],
    "common_ceramics": ["Si", "Al", "O", "Ti", "Zr", "Ca", "Mg"],
    "common_metals": ["Fe", "Al", "Cu", "Ni", "Ti", "Zn", "Pb", "Sn"],
    "semiconductor_elements": ["Si", "Ge", "Ga", "As", "In", "P", "Sb"],
    "optical_materials": ["Si", "Ge", "ZnSe", "CaF2", "BaF2", "Al2O3"],
}

# Compound families for intelligent warming
COMPOUND_FAMILIES: dict[str, list[str]] = {
    "silicates": [
        "SiO2",
        "Al2SiO5",
        "CaSiO3",
        "MgSiO3",
        "CaAl2Si2O8",
        "NaAlSi3O8",
        "KAlSi3O8",
    ],
    "oxides": ["Al2O3", "TiO2", "ZrO2", "CaO", "MgO", "FeO", "Fe2O3", "Cr2O3"],
    "carbonates": ["CaCO3", "MgCO3", "FeCO3", "CaMg(CO3)2"],
    "nitrides": ["Si3N4", "AlN", "TiN", "BN"],
    "carbides": ["SiC", "TiC", "WC", "B4C"],
    "sulfides": ["ZnS", "CuS", "FeS", "FeS2", "PbS"],
    "halides": ["NaCl", "CaF2", "MgF2", "LiF"],
    "phosphates": ["Ca3(PO4)2", "Ca5(PO4)3OH"],
    "sulfates": ["CaSO4", "BaSO4", "SrSO4"],
}


def parse_chemical_formula(formula: str) -> dict[str, int]:
    """
    Parse a chemical formula into element counts.

    This function uses regular expressions to extract element symbols and their
    counts from chemical formulas, handling parentheses and complex structures.

    Args:
        formula: Chemical formula string (e.g., "Ca5(PO4)3OH")

    Returns:
        Dictionary mapping element symbols to their counts

    Examples:
        >>> parse_chemical_formula("SiO2")
        {'Si': 1, 'O': 2}
        >>> parse_chemical_formula("Ca5(PO4)3OH")
        {'Ca': 5, 'P': 3, 'O': 13, 'H': 1}
    """
    element_counts: dict[str, int] = defaultdict(int)

    # Remove spaces and normalize formula
    formula = formula.replace(" ", "")

    # Handle parentheses by expanding them
    while "(" in formula:
        # Find the innermost parentheses
        start = formula.rfind("(")
        if start == -1:
            break

        # Find the matching closing parenthesis
        end = formula.find(")", start)
        if end == -1:
            break

        # Extract content inside parentheses
        inside = formula[start + 1 : end]

        # Find the multiplier after the parentheses
        rest = formula[end + 1 :]
        multiplier_match = re.match(r"(\d+)", rest)
        multiplier = int(multiplier_match.group(1)) if multiplier_match else 1

        # Parse elements inside parentheses
        inside_elements = parse_chemical_formula(inside)

        # Create expanded string
        expanded = ""
        for element, count in inside_elements.items():
            total_count = count * multiplier
            if total_count > 1:
                expanded += f"{element}{total_count}"
            else:
                expanded += element

        # Replace parentheses group with expanded form
        if multiplier_match:
            formula = (
                formula[:start]
                + expanded
                + formula[end + 1 + len(multiplier_match.group(1)) :]
            )
        else:
            formula = formula[:start] + expanded + formula[end + 1 :]

    # Parse elements and their counts using regex
    # Matches: Capital letter + optional lowercase letter + optional digit(s)
    pattern = r"([A-Z][a-z]?)(\d*)"
    matches = re.findall(pattern, formula)

    for element, count_str in matches:
        count = int(count_str) if count_str else 1
        element_counts[element] += count

    return dict(element_counts)


def get_elements_for_compound(formula: str) -> list[str]:
    """
    Get the list of elements for a given compound formula.

    Args:
        formula: Chemical formula string

    Returns:
        List of unique element symbols in the compound

    Examples:
        >>> get_elements_for_compound("SiO2")
        ['Si', 'O']
        >>> get_elements_for_compound("CaAl2Si2O8")
        ['Ca', 'Al', 'Si', 'O']
    """
    # First check if it's a known compound
    if formula in COMMON_COMPOUNDS:
        return list(COMMON_COMPOUNDS[formula].keys())

    # Otherwise parse the formula
    element_counts = parse_chemical_formula(formula)
    return list(element_counts.keys())


def get_compound_frequency_score(formula: str) -> float:
    """
    Get a frequency score for a compound based on its commonality in materials science.

    Args:
        formula: Chemical formula string

    Returns:
        Frequency score (0.0 to 1.0, higher = more common)
    """
    # Known common compounds get high scores
    if formula in COMMON_COMPOUNDS:
        # Score based on position in list and usage patterns
        common_formulas = list(COMMON_COMPOUNDS.keys())
        position = common_formulas.index(formula)
        base_score = 1.0 - (position / len(common_formulas)) * 0.5  # 0.5 to 1.0 range

        # Boost scores for very common compounds
        high_frequency_compounds = [
            "SiO2",
            "Al2O3",
            "TiO2",
            "CaCO3",
            "Fe2O3",
            "FeO",
            "CaO",
            "MgO",
        ]
        if formula in high_frequency_compounds:
            base_score = min(1.0, base_score + 0.3)

        return base_score

    # Unknown compounds get moderate scores based on element commonality
    elements = get_elements_for_compound(formula)

    # Common elements in materials science
    common_elements = {"Si", "O", "Al", "Fe", "Ca", "Mg", "Ti", "C", "N"}
    common_count = sum(1 for elem in elements if elem in common_elements)

    # Base score on fraction of common elements
    if elements:
        return min(0.5, (common_count / len(elements)) * 0.5)

    return 0.0


def find_similar_compounds(
    formula: str, similarity_threshold: float = 0.5
) -> list[str]:
    """
    Find compounds with similar element composition.

    Args:
        formula: Target chemical formula
        similarity_threshold: Minimum similarity score (0.0 to 1.0)

    Returns:
        List of similar compound formulas
    """
    target_elements = set(get_elements_for_compound(formula))
    similar_compounds = []

    for compound in COMMON_COMPOUNDS:
        compound_elements = set(get_elements_for_compound(compound))

        if not target_elements or not compound_elements:
            continue

        # Calculate Jaccard similarity
        intersection = len(target_elements & compound_elements)
        union = len(target_elements | compound_elements)
        similarity = intersection / union if union > 0 else 0.0

        if similarity >= similarity_threshold:
            similar_compounds.append(compound)

    # Sort by similarity score (descending)
    similar_compounds.sort(
        key=lambda x: len(set(get_elements_for_compound(x)) & target_elements),
        reverse=True,
    )

    return similar_compounds


def get_compound_family(formula: str) -> str | None:
    """
    Determine which compound family a formula belongs to.

    Args:
        formula: Chemical formula string

    Returns:
        Family name or None if not found
    """
    for family, compounds in COMPOUND_FAMILIES.items():
        if formula in compounds:
            return family

    # Classify based on elements present
    elements = get_elements_for_compound(formula)

    if "Si" in elements and "O" in elements:
        return "silicates"
    elif "C" in elements and "O" in elements and len(elements) <= 3:
        return "carbonates"
    elif "O" in elements and len(elements) == 2:
        return "oxides"
    elif "N" in elements and "O" not in elements:
        return "nitrides"
    elif "C" in elements and "O" not in elements:
        return "carbides"
    elif "S" in elements and "O" not in elements:
        return "sulfides"
    elif any(hal in elements for hal in ["F", "Cl", "Br", "I"]):
        return "halides"
    elif "P" in elements and "O" in elements:
        return "phosphates"
    elif "S" in elements and "O" in elements:
        return "sulfates"

    return None


def get_recommended_elements_for_warming(
    recent_compounds: list[str], max_elements: int = 20
) -> list[str]:
    """
    Get recommended elements for cache warming based on recent compound usage.

    Args:
        recent_compounds: List of recently used compound formulas
        max_elements: Maximum number of elements to recommend

    Returns:
        List of element symbols ranked by importance for warming
    """
    element_scores: dict[str, float] = defaultdict(float)

    # Score elements based on compound frequency and usage
    for compound in recent_compounds:
        elements = get_elements_for_compound(compound)
        frequency_score = get_compound_frequency_score(compound)

        # Distribute score among elements in compound
        element_score = frequency_score / len(elements) if elements else 0.0

        for element in elements:
            element_scores[element] += element_score

    # Add bonus for very common elements
    common_elements = {"Si", "O", "Al", "Fe", "Ca", "Mg", "Ti", "C", "N", "H"}
    for element in common_elements:
        if element in element_scores:
            element_scores[element] *= 1.5  # 50% bonus
        else:
            element_scores[element] = 0.1  # Base score for common elements

    # Sort by score and return top elements
    sorted_elements = sorted(element_scores.items(), key=lambda x: x[1], reverse=True)
    return [element for element, score in sorted_elements[:max_elements]]


def analyze_element_associations(compound_list: list[str]) -> dict[str, list[str]]:
    """
    Analyze which elements are frequently associated together.

    Args:
        compound_list: List of compound formulas to analyze

    Returns:
        Dictionary mapping elements to their frequently associated elements
    """
    associations: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Count co-occurrences
    for compound in compound_list:
        elements = get_elements_for_compound(compound)

        # Record all pairwise associations
        for i, elem1 in enumerate(elements):
            for j, elem2 in enumerate(elements):
                if i != j:  # Don't count self-associations
                    associations[elem1][elem2] += 1

    # Convert to ranked lists
    element_associations = {}
    for element, associated in associations.items():
        # Sort by frequency and take top associations
        sorted_associations = sorted(
            associated.items(), key=lambda x: x[1], reverse=True
        )
        element_associations[element] = [assoc[0] for assoc in sorted_associations[:5]]

    return element_associations


def get_compound_complexity_score(formula: str) -> float:
    """
    Calculate a complexity score for a compound (more complex = higher score).

    Args:
        formula: Chemical formula string

    Returns:
        Complexity score (higher values indicate more complex compounds)
    """
    element_counts = parse_chemical_formula(formula)

    if not element_counts:
        return 0.0

    # Factors contributing to complexity
    num_elements = len(element_counts)
    total_atoms = sum(element_counts.values())
    max_count = max(element_counts.values())

    # Complexity score based on multiple factors
    complexity = (
        num_elements * 2.0  # More elements = more complex
        + total_atoms * 0.1  # More atoms = more complex
        + max_count * 0.5  # High stoichiometry = more complex
        + (1.0 if num_elements > 3 else 0.0)  # Bonus for >3 elements
    )

    return complexity
