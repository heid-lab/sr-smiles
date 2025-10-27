from typing import Tuple

import pytest
from rdkit import Chem


@pytest.fixture
def simple_mol() -> Chem.Mol:
    """Provides a simple molecule."""
    smiles = "[C:1][C@:2]([Cl:3])([Br:4])[I:5]"
    mol = Chem.MolFromSmiles(smiles)
    return mol


@pytest.fixture
def reaction_mols() -> Tuple[Chem.Mol, Chem.Mol]:
    """Provides a simple reactant and product molecule pair."""
    reac_smi = "[CH3:1][C:2](=[O:3])[O:4].[CH3:5][NH2:6]"
    prod_smi = "[CH3:1][C:2](=[O:3])[NH:6][CH3:5].[OH2:4]"
    mol_reac = Chem.MolFromSmiles(reac_smi)
    mol_prod = Chem.MolFromSmiles(prod_smi)
    return mol_reac, mol_prod


@pytest.fixture
def simple_smiles() -> str:
    """Provides a simple atom-mapped SMILES string with explicit hydrogens."""
    return "[C:1][C@:2]([Cl:3])([Br:4])[I:5]"


@pytest.fixture
def complex_smiles() -> Tuple[Chem.Mol, Chem.Mol]:
    """Provides a complex smiles, including multiple rings and stereochemistry."""
    smiles = "[C:1]([C@@:2]1([H:11])[O:3][C@@:4]2([H:12])[C:5]([H:13])([H:14])[C:6]([H:15])([H:16])[C@@:7]12[H:17])([H:8])([H:9])[H:10]"  # noqa: E501
    return smiles


# def test_remove_non_participating_hydrogens_unbalanced():
#     """
#     Test that the input is returned unchanged when H count changes.
#     """
#     input_rxn = "[CH3][OH]>>[CH3][O]"
#     result = remove_non_participating_hydrogens(input_rxn)
#     assert result == input_rxn


# def test_remove_non_participating_hydrogens_balanced():
#     """
#     Test that non-essential hydrogens are removed when H count is conserved.
#     """
#     input_rxn = "[CH3][OH]>>[CH3][O][H]"
#     result = remove_non_participating_hydrogens(input_rxn)
#     assert result == "[C][OH]>>[C][O][H]"


# def test_remove_redundant_square_brackets_rxn_smiles():
#     """Test removal of square brackets from a reaction smiles."""
#     rxn_smiles = "[O]=[C]([H])[C]#[C][H]>>[H][C]#[C][H].[O+]#[C-]"
#     rxn_wo_am = remove_redundant_square_brackets(rxn_smiles)
#     assert rxn_wo_am == "O=C([H])C#C[H]>>[H]C#C[H].[O+]#[C-]"


# def test_remove_redundant_square_brackets_sr_smiles():
#     """Test removal of square brackets from a SR smiles."""
#     sr_smiles = "{[O]|[O+]}{=|#}{[C]|[C-]}1{-|~}[H]{~|-}[C]{-|~}1#[C][H]"
#     sr_wo_am = remove_redundant_square_brackets(sr_smiles)
#     assert sr_wo_am == "{O|[O+]}{=|#}{C|[C-]}1{-|~}[H]{~|-}C{-|~}1#C[H]"


# @pytest.mark.parametrize(
#     "smiles, expected",
#     [
#         ("[O:1]([H:2])[H:3]", True),
#         ("[CH3:1]=[O:2]", False),
#     ],
# )
# def test_includes_individually_mapped_hydrogens(smiles, expected):
#     """Test detection of individually mapped hydrogens in SMILES."""
#     assert includes_individually_mapped_hydrogens(smiles) == expected


# def test_get_atom_map_num(simple_mol):
#     """Check that each atom's map number matches its index + 1."""
#     num_atoms = simple_mol.GetNumAtoms()
#     for idx in range(num_atoms):
#         map_num = get_atom_map_num(simple_mol, idx)
#         assert map_num == idx + 1


# def test_get_atom_indices_and_smarts(simple_mol):
#     """Check that each atom index is paired with its correct SMARTS pattern."""
#     indexed_smarts = get_atom_indices_and_smarts(simple_mol)
#     assert indexed_smarts == [
#         (0, "[C:1]"),
#         (1, "[C@:2]"),
#         (2, "[Cl:3]"),
#         (3, "[Br:4]"),
#         (4, "[I:5]"),
#     ]


# def test_get_bond_idx(simple_mol):
#     """Check that the bond index is returned for connected atoms, otherwise None."""
#     assert get_bond_idx(simple_mol, 0, 1) == 0

#     assert get_bond_idx(simple_mol, 1, 2) == 1
#     assert get_bond_idx(simple_mol, 1, 3) == 2
#     assert get_bond_idx(simple_mol, 1, 4) == 3

#     assert get_bond_idx(simple_mol, 2, 3) is None
