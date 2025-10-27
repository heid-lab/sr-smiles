from typing import Tuple

import pytest
from rdkit import Chem

from sr_smiles.chem_utils.mol_utils import (
    get_atom_by_map_num,
    get_atom_map_nums_of_mol,
    get_reac_to_prod_mapping,
    make_mol,
    remove_bonds_by_atom_map_nums,
    reorder_mol,
)


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


@pytest.mark.parametrize(
    "smi_in,smi_out,kekulize",
    [
        ("CCO", "CCO", False),
        ("CCO", "CCO", True),
        ("c1c[nH]cc1", "c1c[nH]cc1", False),
        ("c1c[nH]cc1", "C1=CNC=C1", True),
        ("[C:1]1=[C:5][N:4][C:3]=[C:2]1", "[C:1]1=[C:5][N:4][C:3]=[C:2]1", True),
        ("C[C@H](O)[C@@H](O)C", "C[C@H](O)[C@@H](O)C", True),
    ],
)
def test_make_mol_roundtrip(smi_in, smi_out, kekulize):
    """Ensure `make_mol()` creates a molecule equivalent to the original SMILES."""
    mol = make_mol(smi_in, kekulize=kekulize)
    roundtrip_smi = Chem.MolToSmiles(mol, canonical=False)

    assert roundtrip_smi == smi_out, f"Roundtrip failed: expected {smi_out}, got {roundtrip_smi}"


def test_reorder_mol_matches_atommap_order():
    """Ensure `reorder_mol()` aligns the atom map order of the target molecule with that of the reference."""
    # reference molecule: atom maps 1–4 in order
    ref_smi = "[C:1][C:2][O:3][N:4]"
    reference = Chem.MolFromSmiles(ref_smi)

    # target molecule with shuffled atom-map order (3-2-1-4)
    target_smi = "[O:3]([C:2][C:1])[N:4]"
    target = Chem.MolFromSmiles(target_smi)

    # reorder target to match reference
    reordered = reorder_mol(target, reference)

    reordered_maps = [a.GetAtomMapNum() for a in reordered.GetAtoms()]
    reference_maps = [a.GetAtomMapNum() for a in reference.GetAtoms()]

    assert reordered_maps == reference_maps, f"Expected atom order {reference_maps}, got {reordered_maps}"

    assert Chem.MolToSmiles(reference, canonical=True) == Chem.MolToSmiles(reordered, canonical=True)


def test_get_atom_by_map_num(simple_mol):
    """Check that atoms are correctly retrieved by their map number."""
    num_atoms = simple_mol.GetNumAtoms()
    for idx in range(num_atoms):
        atom = get_atom_by_map_num(simple_mol, atom_map_num=idx + 1)
        assert atom.GetIdx() == idx


@pytest.mark.parametrize(
    "smi, expected",
    [
        ("[C:1][C:2][O:3]", [1, 2, 3]),  # all atoms mapped
        ("[C:1]CO", [1, 0, 0]),  # some atoms unmapped
    ],
)
def test_get_atom_map_nums_of_mol(smi, expected):
    """Verify get_atom_map_nums_of_mol correctly extracts map numbers, including unmapped atoms."""
    mol = Chem.MolFromSmiles(smi)
    result = get_atom_map_nums_of_mol(mol)
    assert result == expected, f"For {smi}: expected {expected}, got {result}"


def test_get_reac_to_prod_mapping(reaction_mols):
    """Tests that reactant indices are correctly mapped to product indices."""
    mol_reac, mol_prod = reaction_mols

    expected_map = {0: 0, 1: 1, 2: 2, 3: 5, 4: 4, 5: 3}
    assert get_reac_to_prod_mapping(mol_reac, mol_prod) == expected_map


def test_get_reac_to_prod_mapping_with_key_error(reaction_mols):
    """Tests that reactant indices are correctly mapped to product indices."""
    _, mol_prod = reaction_mols

    mol_reac_bad = Chem.MolFromSmiles("[C:1][H:99]", sanitize=False)
    with pytest.raises(KeyError):
        get_reac_to_prod_mapping(mol_reac_bad, mol_prod)


def test_remove_bonds_by_atom_map_nums_removes_existing_bond():
    """Check that a specified existing bond is successfully removed."""
    smi = "[C:1]([O:2])[C:3]"
    mol = Chem.MolFromSmiles(smi)

    modified = remove_bonds_by_atom_map_nums(mol, [(1, 3)])

    assert mol.GetNumBonds() == 2
    assert modified.GetNumBonds() == 1, "Bond between atom maps (1,3) should be removed"


def test_remove_bonds_by_atom_map_nums_handles_missing_bond(caplog):
    """Check that attempting to remove a non-existent bond leaves mol unchanged."""
    smi = "[C:1].[O:2]"
    mol = Chem.MolFromSmiles(smi)

    modified = remove_bonds_by_atom_map_nums(mol, [(1, 2)])

    # both mols have 0 bonds
    assert mol.GetNumBonds() == 0
    assert modified.GetNumBonds() == 0, "Mol should be unchanged when bond does not exist"
