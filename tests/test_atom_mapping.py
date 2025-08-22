import pytest
from rdkit import Chem

from cgr_smiles.atom_mapping import add_atom_mapping, is_fully_atom_mapped


@pytest.mark.parametrize(
    "rxn_smiles, expected",
    [
        ("[CH3:1][OH:2]>>[CH3:1][Cl:2]", True),  # fully mapped, single mol
        (
            "[CH3:1][OH:2].[Na+:3]>>[CH3:1][Na+:3].[O-:2]",
            True,
        ),  # fully mapped, multiple mols
        ("CCO>>CCCl", False),  # no mapping at all
        ("CO>>[CH3:1][OH:2]", False),  # no mapping in reac
        ("[CH3:1][OH:2]>>CO", False),  # no mapping in prod
        ("C[OH:2]>>[CH3:1][OH:2]", False),  # partial mapping in reac
        ("[CH3:1][OH:2]>>C[OH:2]", False),  # partial mapping in prod
        ("[CH3:1][OH:2]>>[CH3:1][Cl:99]", False),  # mismatched map nums
        ("[CH3:0][OH:2]>>[CH3:0][OH:2]", False),  # invalid zero mapping
    ],
)
def test_is_fully_mapped_rxn(rxn_smiles, expected):
    assert is_fully_atom_mapped(rxn_smiles) == expected


REACTION_CASES = [
    ("CCO>>CCO", "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3]"),  # balanced
    ("CCO>>OCC", "[CH3:1][CH2:2][OH:3]>>[OH:3][CH2:2][CH3:1]"),  # balanced
    ("CCO>>CCO.O", "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3].[OH2:4]"),  # unbalanced
    (
        "CC[C:1]>>CC[C:1]",
        "[CH3:2][CH2:3][C:1]>>[CH3:2][CH2:3][C:1]",
    ),  # partially mapped
    ("CC[C:1]>>[C:1]", "[CH3:2][CH2:3][C:1]>>[C:1]"),  # partially mapped
    # (
    #     "C([H])([H])([H])[H]>>C([H])([H])([H])[H]",
    #     "[C:1]([H:2])([H:3])([H:4])[H:5]>>[C:1]([H:2])([H:3])([H:4])[H:5]",
    # ),  # unmapped, explicit hydrogens
    (
        "[C:1]([H:2])([H:3])([H:4])[H:5]>>[C:1]([H:2])([H:3])([H:4])[H:5]",
        "[C:1]([H:2])([H:3])([H:4])[H:5]>>[C:1]([H:2])([H:3])([H:4])[H:5]",
    ),  # fully mapped
]


@pytest.mark.parametrize("rxn,expected", REACTION_CASES)
def test_add_atom_mapping(rxn, expected):
    """
    Test that add_atom_mapping returns a mapped reaction SMILES
    for both rxnmapper and graph_overlay methods.
    """
    mapped_smi = add_atom_mapping(rxn, method="graph_overlay")

    assert isinstance(mapped_smi, str)
    assert mapped_smi == expected


# for a, b in REACTION_CASES:
#     test_add_atom_mapping(a, b)

# mapped_smi = add_atom_mapping(rxn, method="graph_overlay")


@pytest.mark.parametrize("rxn,expected", REACTION_CASES)
def test_add_atom_mapping_with_rxn_mapper(rxn, expected):
    """
    Test that add_atom_mapping returns a mapped reaction SMILES for rxnmapper.
    """
    mapped_smi = add_atom_mapping(rxn, method="rxnmapper")

    assert isinstance(mapped_smi, str)
    assert mapped_smi == expected


def mapping_pattern(smi: str):
    """Extracts the mapping pattern from a mapped reaction SMILES."""
    reac_smi, prod_smi = smi.split(">>")
    mol_reac = Chem.MolFromSmiles(reac_smi)
    mol_prod = Chem.MolFromSmiles(prod_smi)

    # Map number -> list of atom indices in reac/prod
    reactant_map = {}
    for atom in mol_reac.GetAtoms():
        amap = atom.GetAtomMapNum()
        if amap:
            reactant_map[amap] = reactant_map.get(amap, []) + [("R", atom.GetIdx())]

    product_map = {}
    for atom in mol_prod.GetAtoms():
        amap = atom.GetAtomMapNum()
        if amap:
            product_map[amap] = product_map.get(amap, []) + [("P", atom.GetIdx())]

    # Build a normalized pattern: for each map number, store counts of atoms in R and P
    # This ignores the actual map number and focuses on the "shape" of the mapping
    pattern = []
    for amap in sorted(set(reactant_map) | set(product_map)):
        r_count = len([x for x in reactant_map.get(amap, []) if x[0] == "R"])
        p_count = len([x for x in product_map.get(amap, []) if x[0] == "P"])
        pattern.append((r_count, p_count))

    # Sort pattern so that different numbering orders still match
    return sorted(pattern)


def mapping_matches(smi: str, ref_smi: str) -> bool:
    """
    Checks if the mapping pattern in smi matches that in ref_smi,
    ignoring actual atom map numbers but preserving mapping relationships.
    """
    return mapping_pattern(smi) == mapping_pattern(ref_smi)
