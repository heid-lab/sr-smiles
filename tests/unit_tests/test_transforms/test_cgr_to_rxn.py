import pandas as pd
import pytest
from rdkit import Chem

from cgr_smiles.transforms.cgr_to_rxn import (
    cgrsmiles_to_rxnsmiles,
    find_cis_trans_stereo_bonds,
    get_chiral_center_map_nums,
    get_reac_prod_scaffold_smiles_from_cgr,
    parse_bonds_from_smiles,
    remove_bonds_by_atom_map_nums,
    update_chirality_tags,
    update_cis_trans_stereo_chem,
)
from cgr_smiles.utils import ROOT_DIR, canonicalize

# TODO: test case: F/C=C/C=C/C
# TODO test for correct parsing of the bonds (all are valid):
# C1CCCCC=1
# C=1CCCCC1
# C=1CCCCC=1
# C0CCCCC0


TEST_DATA_PATH = ROOT_DIR / "tests" / "data" / "cgr_test_cases.csv"


def cgr_test_cases():
    """Loads test cases from a CSV file once per test module."""
    df = pd.read_csv(TEST_DATA_PATH)
    return list(zip(df["rxn"], df["rxn_smiles"], df["cgr_smiles"]))


CGR_CASES = cgr_test_cases()


@pytest.mark.parametrize("rxn_id, rxn_smiles, cgr_smiles", CGR_CASES)
def test_cgrsmiles_to_rxnsmiles(rxn_id, rxn_smiles, cgr_smiles):
    """Check that CGR to RXN conversion reproduces the original reaction SMILES."""
    res1 = cgrsmiles_to_rxnsmiles(cgr_smiles)
    rxn2 = canonicalize(rxn_smiles)
    res2 = canonicalize(res1)
    assert rxn2 == res2, f"Assertion error for reaction with id {rxn_id}"


def e_z_stereo_test_cases():
    """Provide E/Z stereochemistry test cases."""
    return [
        (
            1,
            "[F:1][CH:2]=[CH:3][F:4]>>[F:1]/[CH:2]=[CH:3]/[F:4]",
            "[F:1]{-|/}[CH:2]=[CH:3]{-|/}[F:4]",
        ),  # trans
        (
            2,
            "[F:1][CH:2]=[CH:3][F:4]>>[CH:2](\\[F:1])=[CH:3]/[F:4]",
            "[F:1]{-|/}[CH:2]=[CH:3]{-|/}[F:4]",
        ),  # trans
        (
            3,
            "[F:1][CH:2]=[CH:3][F:4]>>[F:1]\\[CH:2]=[CH:3]/[F:4]",
            "[F:1]{-|\\}[CH:2]=[CH:3]{-|/}[F:4]",
        ),  # cis
        (
            4,
            "[F:1][CH:2]=[CH:3][F:4]>>[CH:2](/[F:1])=[CH:3]/[F:4]",
            "[F:1]{-|\\}[CH:2]=[CH:3]{-|/}[F:4]",
        ),  # cis
        (
            5,
            "[F:1]/[CH:2]=[CH:3]/[F:4]>>[F:1][CH:2]=[CH:3][F:4]",
            "[F:1]{/|-}[CH:2]=[CH:3]{/|-}[F:4]",
        ),  # trans
        (
            6,
            "[CH:2](\\[F:1])=[CH:3]/[F:4]>>[F:1][CH:2]=[CH:3][F:4]",
            "[CH:2]({\\|-}[F:1])=[CH:3]{/|-}[F:4]",
        ),  # trans
        (
            7,
            "[F:1]\\[CH:2]=[CH:3]/[F:4]>>[F:1][CH:2]=[CH:3][F:4]",
            "[F:1]{\\|-}[CH:2]=[CH:3]{/|-}[F:4]",
        ),  # cis
        (
            8,
            "[CH:2](\\[F:1])=[CH:3]/[F:4]>>[F:1][CH:2]=[CH:3][F:4]",
            "[CH:2]({\\|-}[F:1])=[CH:3]{/|-}[F:4]",
        ),  # cis
        (
            9,
            "[CH:2](\\[F:1])=[CH:3]/[F:4]>>[CH:2]([F:1])=[CH:3][F:4]",
            "[CH:2]({\\|-}[F:1])=[CH:3]{/|-}[F:4]",
        ),  # cis
    ]


E_Z_STEREO_CASES = e_z_stereo_test_cases()


@pytest.mark.parametrize("idx, rxn_smiles, cgr_smiles", E_Z_STEREO_CASES)
def test_rxnsmiles_to_cgrsmiles_e_z_stereo(idx, rxn_smiles, cgr_smiles):
    """Check that E/Z stereochemistry is correctly preserved in RXN->CGR->RXN conversion."""
    result = cgrsmiles_to_rxnsmiles(cgr_smiles)
    can_res = canonicalize(result)
    can_rxn = canonicalize(rxn_smiles)
    assert can_res == can_rxn, f"Assertion error for reaction with id {idx}"


def test_parse_bonds_from_smiles():
    """Verify correct parsing of bonds and stereochemistry from a SMILES string."""
    smiles = "[c:1]1([n:2][n:3][c:4]([H:8])[n:5][n:6]1)[CH3:9]/[N:10]=[N:11]\\[CH3:12]"
    expected_output = {
        (1, 2): "-",
        (2, 3): "-",
        (3, 4): "-",
        (4, 8): "-",
        (4, 5): "-",
        (5, 6): "-",
        (6, 1): "-",
        (1, 9): "-",
        (9, 10): "/",
        (10, 11): "=",
        (11, 12): "\\",
    }

    assert parse_bonds_from_smiles(smiles) == expected_output


def test_remove_bonds_by_atom_map_nums():
    """Tests successful removal of specified bonds using atom map numbers."""
    mol = Chem.MolFromSmiles("[C:1][C:2]=[C:3][C:4]")
    bonds_to_remove = [(2, 3)]
    modified_mol = remove_bonds_by_atom_map_nums(mol, bonds_to_remove)

    expected_output = "[C:1][C:2].[C:3][C:4]"
    assert Chem.MolToSmiles(modified_mol) == expected_output


def test_remove_bonds_by_atom_map_nums_invalid_key():
    """Test that remove_bonds_by_atom_map_nums raises a KeyError when atom map numberis invalid."""
    mol = Chem.MolFromSmiles("[C:1][C:2]=[C:3][C:4]")
    bonds_to_remove = [(99, 100)]

    with pytest.raises(KeyError, match="99"):
        remove_bonds_by_atom_map_nums(mol, bonds_to_remove)


def test_update_chirality_tags_without_flip():
    """Test that chirality is preserved when the neighbor order implies no flip.

    The chirality in the SMILES and CGR scaffold differs in tag (@ vs @@),
    but the neighbor order leads to an even permutation, so no flip is needed.
    The original SMILES should be returned unchanged.
    """
    smiles = "[C:1][C@:2]([Br:3])([Cl:4])[N:5]"
    cgr_scaffold = "[C:1][C@@:2]([Br:3])([N:5])[Cl:4]"
    chiral_centers = [2]

    modified_smiles = update_chirality_tags(smiles, cgr_scaffold, chiral_centers)
    assert modified_smiles == smiles


def test_update_chirality_tags_with_flip():
    """Test that chirality tag is flipped when the neighbor order implies inversion.

    The input SMILES has a '@' tag, and the CGR scaffold has '@',
    but the neighbor order is an odd permutation, so chirality must be flipped.
    The output should have '@@' instead of '@'.
    """
    smiles = "[C:1][C@:2]([Br:3])([Cl:4])[N:5]"
    cgr_scaffold = "[C:1][C@:2]([Br:3])([N:5])[Cl:4]"
    chiral_centers = [2]

    modified_smiles = update_chirality_tags(smiles, cgr_scaffold, chiral_centers)
    expected_output = "[C:1][C@@:2]([Br:3])([Cl:4])[N:5]"

    assert modified_smiles == expected_output


def test_find_cis_trans_stereo_bonds():
    """Verify detection of cis/trans stereochemistry bonds in a bond dictionary."""
    bonds = {
        (1, 2): "-",
        (2, 3): "-",
        (3, 4): "-",
        (1, 4): "-",
        (4, 5): "-",
        (5, 6): "/",
        (6, 7): "=",
        (7, 8): "\\",
    }
    result = find_cis_trans_stereo_bonds(bonds)
    expected_output = {
        (6, 7): {"stereo": Chem.BondStereo.STEREOZ, "terminal_atoms": (5, 8)},
        (7, 6): {"stereo": Chem.BondStereo.STEREOZ, "terminal_atoms": (8, 5)},
    }
    assert result == expected_output


def test_update_cis_trans_stereo_chem_cis_update():
    """Tests that a double bond's stereochemistry is correctly updated to cis."""
    smiles = "[c:1]1([CH2:9]/[N:10]=[N:11]/[CH3:12])[n:2][n:3][cH:4][n:5][n:6]1"
    mol = Chem.MolFromSmiles(smiles)

    parsed_bonds_data = {
        (1, 9): "-",
        (9, 10): "/",
        (10, 11): "=",
        (11, 12): "\\",
        (1, 2): "-",
        (2, 3): "-",
        (3, 4): "-",
        (4, 8): "-",
        (4, 5): "-",
        (5, 6): "-",
        (1, 6): "-",
    }

    modified_mol = update_cis_trans_stereo_chem(mol, parsed_bonds_data)
    modified_smiles = Chem.MolToSmiles(modified_mol, canonical=False)
    assert modified_smiles == "[c:1]1([CH2:9]/[N:10]=[N:11]\\[CH3:12])[n:2][n:3][cH:4][n:5][n:6]1"


def test_update_cis_trans_stereo_chem_trans_update():
    """Tests that a double bond's stereochemistry is correctly updated to trans."""
    smiles = "[c:1]1([CH2:9][N:10]=[N:11][CH3:12])[n:2][n:3][cH:4][n:5][n:6]1"
    mol = Chem.MolFromSmiles(smiles)

    parsed_bonds_data = {
        (1, 9): "-",
        (9, 10): "/",
        (10, 11): "=",
        (11, 12): "/",
        (1, 2): "-",
        (2, 3): "-",
        (3, 4): "-",
        (4, 8): "-",
        (4, 5): "-",
        (5, 6): "-",
        (1, 6): "-",
    }

    modified_mol = update_cis_trans_stereo_chem(mol, parsed_bonds_data)
    modified_smiles = Chem.MolToSmiles(modified_mol, canonical=False)
    assert modified_smiles == "[c:1]1([CH2:9]/[N:10]=[N:11]/[CH3:12])[n:2][n:3][cH:4][n:5][n:6]1"


def test_update_cis_trans_stereo_chem_with_disconnected_molecule():
    """Tests that a double bond's stereochemistry is correctly updated to trans."""
    smiles = "[c:1]1([CH2:9][N:10]=[N:11][CH3:12])[n:2][n:3][cH:4][n:5][n:6]1.[CH4:13]"
    mol = Chem.MolFromSmiles(smiles)

    parsed_bonds_data = {
        (1, 9): "-",
        (9, 10): "/",
        (10, 11): "=",
        (11, 12): "/",
        (1, 2): "-",
        (2, 3): "-",
        (3, 4): "-",
        (4, 8): "-",
        (4, 5): "-",
        (5, 6): "-",
        (1, 6): "-",
    }

    modified_mol = update_cis_trans_stereo_chem(mol, parsed_bonds_data)
    modified_smiles = Chem.MolToSmiles(modified_mol, canonical=False)
    assert modified_smiles == "[c:1]1([CH2:9]/[N:10]=[N:11]/[CH3:12])[n:2][n:3][cH:4][n:5][n:6]1.[CH4:13]"


def test_get_reac_prod_scaffold_smiles_from_cgr():
    """Tests parsing of a CGR string into reactant and product scaffold SMILES."""
    cgr = "{[O:1]|[O+:1]}{=|#}{[C:2]|[C-:2]}1{-|~}[C:3](#[C:4][H:6]){~|-}[H:5]{-|~}1"

    expected_reac = "[O:1]=[C:2]1-[C:3](#[C:4][H:6])~[H:5]-1"
    expected_prod = "[O+:1]#[C-:2]1~[C:3](#[C:4][H:6])-[H:5]~1"

    reac, prod = get_reac_prod_scaffold_smiles_from_cgr(cgr)

    assert reac == expected_reac, f"Reactant SMILES mismatch: got {reac}"
    assert prod == expected_prod, f"Product SMILES mismatch: got {prod}"


def test_get_chiral_center_map_nums():
    """Tests identification of chiral centers in a complex molecule."""
    smiles = "[O:1]([C@@:2]1([H:9])[C@@:3]2([H:10])[C@@:4]3([H:11])[C:5]([H:12])([H:13])[C@:6]1([H:14])[N:7]23)[H:8]"  # noqa: E501
    mol = Chem.MolFromSmiles(smiles)
    chiral_centers = get_chiral_center_map_nums(mol)

    expected = [2, 3, 4, 6]
    assert chiral_centers == expected, f"Expected {expected}, but got {chiral_centers}"


def test_get_chiral_center_map_nums_no_chirality():
    """Test that get_chiral_center_map_nums returns an empty list when mol has no chiral centers."""
    smiles = "[C:1]([H:2])([H:3])[C:4]([H:5])([H:6])[H:7]"
    mol = Chem.MolFromSmiles(smiles)
    chiral_centers = get_chiral_center_map_nums(mol)

    assert chiral_centers == [], f"Expected no chiral centers, but got {chiral_centers}"
