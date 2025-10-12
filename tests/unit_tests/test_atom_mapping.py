import pytest
from conftest import equivalent_reactions

from cgr_smiles.atom_mapping import (
    add_atom_mapping,
    add_atom_mapping_to_cgr,
    is_cgr_smiles_fully_atom_mapped,
    is_fully_atom_mapped,
)


@pytest.mark.parametrize(
    "rxn_smiles, expected",
    [
        ("[CH3:1][OH:2]>>[CH3:1][Cl:2]", True),  # fully mapped, single mol
        ("[CH3:1][OH:2].[Na+:3]>>[CH3:1][Na+:3].[O-:2]", True),  # fully mapped, multiple mols
        ("CCO>>CCCl", False),  # no mapping at all
        ("CO>>[CH3:1][OH:2]", False),  # no mapping in reac
        ("[CH3:1][OH:2]>>CO", False),  # no mapping in prod
        ("C[OH:2]>>[CH3:1][OH:2]", False),  # partial mapping in reac
        ("[CH3:1][OH:2]>>C[OH:2]", False),  # partial mapping in prod
        ("[CH3:1][OH:2]>>[CH3:1][Cl:99]", False),  # mismatched map nums
        ("[CH3:0][OH:2]>>[CH3:0][OH:2]", False),  # invalid zero mapping
    ],
)
def test_is_fully_atom_mapped(rxn_smiles, expected):
    """Test that `is_fully_atom_mapped` correctly identifies fully mapped reactions."""
    assert is_fully_atom_mapped(rxn_smiles) == expected


@pytest.mark.parametrize(
    "rxn_smiles, expected",
    [
        ("[CH3:1][OH:2]", True),  # fully mapped, single mol
        ("[CH3:1][OH:2]{[CH3:3]|[CH2:3]}", True),  # fully mapped, single mol
        ("[CH3:1][OH:2].[Na+:3]", True),  # fully mapped, multiple mols
        ("CCCl", False),  # no mapping at all
        ("CC{Cl|Br}", False),  # no mapping at all
        ("C[C:1]{Cl|Br}", False),  # no mapping at all
        ("CC{[Cl:1]|[Br:1]}", False),  # no mapping at all
    ],
)
def test_is_cgr_smiles_fully_atom_mapped(rxn_smiles, expected):
    """Test that `is_cgr_smiles_fully_atom_mapped` correctly identifies fully mapped reactions."""
    assert is_cgr_smiles_fully_atom_mapped(rxn_smiles) == expected


REACTION_CASES = [
    ("CCO>>CCO", "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3]"),  # balanced
    ("CCO>>OCC", "[CH3:1][CH2:2][OH:3]>>[OH:3][CH2:2][CH3:1]"),  # balanced
    ("CCO>>CCO.O", "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3].[OH2:4]"),  # unbalanced
    (
        "CC[C:1]>>CC[C:1]",
        "[CH3:2][CH2:3][C:1]>>[CH3:2][CH2:3][C:1]",
    ),  # partially mapped
    # ("CC[C:1]>>[C:1]", "[CH3:2][CH2:3][C:1]>>[C:1]"),  # partially mapped
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
def test_add_atom_mapping_with_rdkit_graph_overlay(rxn, expected):
    """Test that add_atom_mapping returns a mapped reaction SMILES for `graph_overlay` method."""
    mapped_smi = add_atom_mapping(rxn, method="graph_overlay")

    assert isinstance(mapped_smi, str)
    assert mapped_smi == expected


@pytest.mark.parametrize("rxn,expected", REACTION_CASES)
def test_add_atom_mapping_with_rxn_mapper(rxn, expected):
    """Test that add_atom_mapping returns a mapped reaction SMILES for `rxnmapper` method."""
    mapped_smi = add_atom_mapping(rxn, method="rxnmapper")

    assert isinstance(mapped_smi, str)
    assert equivalent_reactions(mapped_smi, expected)


@pytest.mark.parametrize(
    "cgr, expected",
    [
        ("C=O", "[C:1]=[O:2]"),
        ("{C|[C-]}", "{[C:1]|[C-:1]}"),
        ("{C|N}=O", "{[C:1]|[N:1]}=[O:2]"),
        ("[C:2]O{-|=}{[CH3]|[CH2]}", "[C:2][O:1]{-|=}{[CH3:3]|[CH2:3]}"),
    ],
)
def test_add_atom_mapping_to_cgr(cgr, expected):
    """Ensure add_atom_mapping_to_cgr assigns unique continuous map numbers and groups share indices."""
    result = add_atom_mapping_to_cgr(cgr)
    assert result == expected, f"For '{cgr}': expected '{expected}', got '{result}'"
