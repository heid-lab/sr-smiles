import pytest
from rdkit import Chem

from cgr_smiles.chem_utils.stereo_chem_utils import (
    find_e_z_stereo_bonds,
    flip_e_z_stereo,
    get_chiral_center_map_nums,
)
from cgr_smiles.transforms.rxn_to_cgr import is_chiral_center  # adjust import path as needed


@pytest.mark.parametrize(
    "smi, chiral_center_indices",
    [
        ("C[C@H](F)Br", [1]),  # chiral carbon: @
        ("C[C@@H](F)Br", [1]),  # chiral carbon: @@
        ("CCO", []),  # non-chiral carbon
    ],
)
def test_is_chiral_center(smi, chiral_center_indices):
    """Ensure is_chiral_center correctly detects tetrahedral chiral centers."""
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        if atom.GetIdx() in chiral_center_indices:
            assert is_chiral_center(atom)
        else:
            assert not is_chiral_center(atom)


@pytest.mark.parametrize(
    "smi, expected",
    [
        ("[C@H:1](F)(Cl)Br", [1]),  # one chiral center
        ("[C@:1]([F:2])([Br:3])([Cl:4])[C@@:5]([Cl:6])([F:7])[Br:8]", [1, 5]),  # two chiral centers
        ("[C:1](F)(Cl)Br", []),  # no chiral centers
    ],
)
def test_get_chiral_center_map_nums(smi, expected):
    """Verify get_chiral_center_map_nums correctly returns map numbers of chiral centers."""
    mol = Chem.MolFromSmiles(smi)
    result = get_chiral_center_map_nums(mol)
    assert result == expected, f"For {smi}: expected {expected}, got {result}"


@pytest.mark.parametrize(
    "bond_dict, expected_stereo",
    [
        (
            {
                (0, 1): "/",
                (1, 2): "=",
                (2, 3): "/",
            },
            Chem.BondStereo.STEREOE,
        ),
        (
            {
                (0, 1): "/",
                (1, 2): "=",
                (2, 3): "\\",
            },
            Chem.BondStereo.STEREOZ,
        ),
        (
            {
                (1, 0): "/",
                (1, 2): "=",
                (2, 3): "/",
            },
            Chem.BondStereo.STEREOZ,
        ),
    ],
)
def test_find_e_z_stereo_bonds(bond_dict, expected_stereo):
    """Verify find_e_z_stereo_bonds detects E/Z stereo correctly from bond dictionaries."""
    result = find_e_z_stereo_bonds(bond_dict)

    if not bond_dict:
        assert result == {}, "Expected empty dict for empty input"
    else:
        # Should contain entries for both (1,2) and (2,1)
        assert (1, 2) in result and (2, 1) in result
        stereo = result[(1, 2)]["stereo"]
        assert stereo == expected_stereo, f"Expected {expected_stereo}, got {stereo}"

        term_a, term_b = result[(1, 2)]["terminal_atoms"]
        assert isinstance(term_a, int) and isinstance(term_b, int)


def test_flip_e_z_stereo():
    r"""Ensure flip_e_z_stereo correctly swaps '/' and '\\' in SMILES strings."""
    input = "/"
    result = flip_e_z_stereo(input)
    expected = "\\"
    assert result == expected, f"For {input}: expected {expected}, got {result}"
