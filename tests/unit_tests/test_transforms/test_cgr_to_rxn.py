import pandas as pd
import pytest
from conftest import equivalent_reactions
from rdkit import Chem

from cgr_smiles import ROOT_DIR
from cgr_smiles.chem_utils.smiles_utils import canonicalize
from cgr_smiles.transforms.cgr_to_rxn import (
    CgrToRxn,
    add_atom_mapping_to_cgr,
    cgr_to_rxn,
    find_e_z_stereo_bonds,
    get_chiral_center_map_nums,
    get_reac_prod_scaffold_smiles_from_cgr,
    is_cgr_smiles_fully_atom_mapped,
    is_kekule,
    parse_bonds_in_order_from_smiles,
    remove_bonds_by_atom_map_nums,
    update_chirality_tags,
    update_e_z_stereo_chem,
)
from cgr_smiles.transforms.rxn_to_cgr import rxn_to_cgr

# TODO: test case: F/C=C/C=C/C
# TODO test for correct parsing of the bonds (all are valid):
# C1CCCCC=1
# C=1CCCCC1
# C=1CCCCC=1
# C0CCCCC0


TEST_DATA_PATH = ROOT_DIR / "tests" / "data" / "cgr_test_cases.csv"


def load_cgr_test_cases():
    """Loads test cases from a CSV file once per test module."""
    df = pd.read_csv(TEST_DATA_PATH)
    return df


DF = load_cgr_test_cases()
CGR_TEST_CASES = list(zip(DF["rxn"], DF["rxn_smiles"], DF["cgr_smiles"]))


@pytest.mark.parametrize("rxn_id, rxn_smiles, cgr_smiles", CGR_TEST_CASES)
def test_mapped_cgr_to_rxn(rxn_id, rxn_smiles, cgr_smiles):
    """Check that CGR (with mapping!) to RXN conversion reproduces the original reaction SMILES."""
    res1 = cgr_to_rxn(cgr_smiles)
    rxn2 = canonicalize(rxn_smiles)
    res2 = canonicalize(res1)
    assert rxn2 == res2, f"Assertion error for reaction with id {rxn_id}"


@pytest.mark.parametrize("rxn_id, rxn_smiles, cgr_smiles", CGR_TEST_CASES)
def test_unmapped_cgr_to_rxn(rxn_id, rxn_smiles, cgr_smiles):
    """Check that CGR (unmapped) to RXN conversion reproduces an equivalent to the original RXN SMILES."""
    cgr = rxn_to_cgr(rxn_smiles, remove_brackets=True, remove_hydrogens=True)
    assert not is_cgr_smiles_fully_atom_mapped(cgr)

    rxn = cgr_to_rxn(cgr, add_atom_mapping=True)
    assert equivalent_reactions(rxn_smiles, rxn)


@pytest.mark.parametrize(
    "cgr, expected",
    [
        (
            "CC(C)(C)OC(=O)O{-|~}C(OC(C)(C)C)(=O){~|-}{[nH]|n}1c2ccc(C(C)=O)cc2cc1",
            "[C:1][C:2]([C:3])([C:4])[O:5][C:6](=[O:7])[O:8]{-|~}[C:9]([O:10][C:11]([C:12])([C:13])[C:14])(=[O:15]){~|-}{[nH:16]|[n:16]}1[c:17]2[c:18][c:19][c:20]([C:21]([C:22])=[O:23])[c:24][c:25]2[c:26][c:27]1",
        ),
        (
            "F{-|/}[CH]=[CH]{-|/}{F|[F+]}",
            "[F:1]{-|/}[CH:2]=[CH:3]{-|/}{[F:4]|[F+:4]}",
        ),
        (
            "C[Sc]C{C|[CH]}",
            "[C:1][Sc:2][C:3]{[C:4]|[CH:4]}",
        ),
        (
            "CScc{C|[CH]}",
            "[C:1][S:2][c:3][c:4]{[C:5]|[CH:5]}",
        ),
        (
            "CClc{C|[CH]}",
            "[C:1][Cl:2][c:3]{[C:4]|[CH:4]}",
        ),
        (
            "ccc{C|[C+]}",
            "[c:1][c:2][c:3]{[C:4]|[C+:4]}",
        ),
        (
            "ccc{[C-]|[C+]}",
            "[c:1][c:2][c:3]{[C-:4]|[C+:4]}",
        ),
        (
            "ccc{[C@H]|C}(C)(Br)N",
            "[c:1][c:2][c:3]{[C@H:4]|[C:4]}([C:5])([Br:6])[N:7]",
        ),
    ],
)
def test_add_atom_mapping_to_cgr(cgr, expected):
    """Add atom mapping to a CGR SMILES."""
    assert add_atom_mapping_to_cgr(cgr) == expected


def test_cgr_to_rxn_invalid_smiles(propagated_logger, caplog):
    """Verify that invalid CGR SMILES input logs a warning and returns an empty string."""
    bad_smi = "INVALID-CGR-SMILES"

    with caplog.at_level("WARNING", logger=propagated_logger.name):
        result = cgr_to_rxn(bad_smi)

    assert result == ""
    # assert len(caplog.records) == 1, caplog.records

    record = caplog.records[0]
    assert record.levelname == "WARNING"
    assert "Failed to process CGR SMILES" in record.message, record.message
    # assert "Returning empty string." in record.message


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
def test_cgr_to_rxn_e_z_stereo(idx, rxn_smiles, cgr_smiles):
    """Check that E/Z stereochemistry is correctly preserved in RXN->CGR->RXN conversion."""
    result = cgr_to_rxn(cgr_smiles)
    can_res = canonicalize(result)
    can_rxn = canonicalize(rxn_smiles)
    assert can_res == can_rxn, f"Assertion error for reaction with id {idx}"


def test_parse_bonds_in_order_from_smiles():
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

    assert parse_bonds_in_order_from_smiles(smiles) == expected_output


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


def test_find_e_z_stereo_bonds():
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
    result = find_e_z_stereo_bonds(bonds)
    expected_output = {
        (6, 7): {"stereo": Chem.BondStereo.STEREOZ, "terminal_atoms": (5, 8)},
        (7, 6): {"stereo": Chem.BondStereo.STEREOZ, "terminal_atoms": (8, 5)},
    }
    assert result == expected_output


def test_update_e_z_stereo_chem_cis_update():
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

    modified_mol = update_e_z_stereo_chem(mol, parsed_bonds_data)
    modified_smiles = Chem.MolToSmiles(modified_mol, canonical=False)
    assert modified_smiles == "[c:1]1([CH2:9]/[N:10]=[N:11]\\[CH3:12])[n:2][n:3][cH:4][n:5][n:6]1"


def test_update_e_z_stereo_chem_trans_update():
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

    modified_mol = update_e_z_stereo_chem(mol, parsed_bonds_data)
    modified_smiles = Chem.MolToSmiles(modified_mol, canonical=False)
    assert modified_smiles == "[c:1]1([CH2:9]/[N:10]=[N:11]/[CH3:12])[n:2][n:3][cH:4][n:5][n:6]1"


def test_update_e_z_stereo_chem_with_disconnected_molecule():
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

    modified_mol = update_e_z_stereo_chem(mol, parsed_bonds_data)
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


def test_CgrToRxn_single_string():
    """Test CgrToRxn class for single CGR SMILES."""
    transform = CgrToRxn()
    cgr_smiles = DF.iloc[0]["cgr_smiles"]
    exp_output = DF.iloc[0]["rxn_smiles"]
    result = transform(cgr_smiles)

    assert isinstance(result, str)
    assert result == exp_output


def test_CgrToRxn_list_of_strings():
    """Test CgrToRxn class for a list of CGR SMILES."""
    transform = CgrToRxn()
    cgr_smiles = DF["cgr_smiles"].tolist()
    exp_output = DF["rxn_smiles"].tolist()

    results = transform(cgr_smiles)

    assert isinstance(results, list)
    assert all(isinstance(r, str) for r in results)
    for res, gt in zip(results, exp_output):
        assert canonicalize(res) == canonicalize(gt)


def test_CgrToRxn_pd_series():
    """Test CgrToRxn class for a pd.Series input."""
    transform = CgrToRxn()
    cgr_smiles = DF["cgr_smiles"]
    exp_output = DF["rxn_smiles"]

    results = transform(cgr_smiles)

    assert isinstance(results, pd.Series)
    assert all(isinstance(r, str) for r in results)
    for res, gt in zip(results.tolist(), exp_output.tolist()):
        assert canonicalize(res) == canonicalize(gt)


def test_CgrToRxn_pd_df():
    """Test CgrToRxn class for a pd.DataFrame input."""
    transform = CgrToRxn(cgr_col="cgr_smiles")
    df_cgr_smiles = DF
    exp_output = DF["rxn_smiles"]
    results = transform(df_cgr_smiles)

    assert isinstance(results, pd.Series)
    assert all(isinstance(r, str) for r in results)
    for res, gt in zip(results.tolist(), exp_output.tolist()):
        assert canonicalize(res) == canonicalize(gt)


def test_dataframe_without_cgr_col_raises():
    """Test that CgrToRxn raises ValueError if `self.cgr_col` not set."""
    transform = CgrToRxn()
    df = pd.DataFrame({"cgr": ["CGR>>SMILES"]})
    with pytest.raises(ValueError, match="`self.cgr_col` is not set"):
        transform(df)


def test_invalid_input_type():
    """Test that CgrToRxn raises TypeError if input type is not valid."""
    transform = CgrToRxn()
    with pytest.raises(TypeError):
        transform(12345)


@pytest.mark.parametrize(
    "empty_input,expected_type",
    [
        ("", str),
        ([], list),
        (pd.Series([], dtype=object), pd.Series),
        (pd.DataFrame({"cgr_smiles": []}), pd.Series),
    ],
)
def test_CgrToRxn_empty_inputs(empty_input, expected_type):
    """Test CgrToRxn call for empty inputs."""
    transform = CgrToRxn(cgr_col="cgr_smiles")

    result = transform(empty_input)

    assert isinstance(result, expected_type)

    if isinstance(result, (list, pd.Series)):
        assert len(result) == 0
    elif isinstance(result, str):
        assert result == ""


@pytest.mark.parametrize(
    "cgr_smiles,is_fully_mapped",
    [
        ("{[O:1]|[O+:1]}[C:2]([C:3])", True),
        ("{[O:1]|[O+:1]}[C:5]([C:2])", True),
        ("{O|[O+]}C(C)", False),
        ("{[O:1]|[O+:1]}C([C:2])", False),
    ],
)
def test_is_cgr_smiles_fully_atom_mapped(cgr_smiles, is_fully_mapped):
    """Check if a CGR SMILES is fully atom-mapped."""
    assert is_cgr_smiles_fully_atom_mapped(cgr_smiles) == is_fully_mapped


def test_cgr_to_rxn_keep_atom_mapping():
    """Test cgr2rxn with unmapped cgr smiles."""
    cgr = "F{-|/}[CH]=[CH]{-|/}{F|[F+]}"
    cgr_with_map = cgr_to_rxn(cgr, add_atom_mapping=True)
    expected = "[F:1][CH:2]=[CH:3][F:4]>>[F:1]/[CH:2]=[CH:3]/[F+:4]"
    assert cgr_with_map == expected


def test_cgr_to_rxn_do_not_keep_atom_mapping():
    """Test cgr2rxn with unmapped cgr smiles."""
    cgr = "F{-|/}[CH]=[CH]{-|/}{F|[F+]}"
    cgr_with_map = cgr_to_rxn(cgr, add_atom_mapping=False)
    expected = "F[CH]=[CH]F>>F/[CH]=[CH]/[F+]"
    assert cgr_with_map == expected


def test_cgr_to_rxn_in_kekule_form():
    """Check CGR to RXN conversion on a non-aromatic CGR string."""
    cgr = "C1(C(C)=O)=CC=C2C(=C1)C=CN2{-|~}C(=O)(OC(C)(C)C){~|-}O"
    r, p = cgr_to_rxn(cgr).split(">>")

    exp_r = "C1(C(C)=O)=CC=C2C(=C1)C=CN2C(=O)OC(C)(C)C.O"
    exp_p = "C1(C(C)=O)=CC=C2C(=C1)C=CN2.C(=O)(OC(C)(C)C)O"

    assert r == exp_r
    assert p == exp_p


def test_cgr_to_rxn_cgr_not_in_kekule_form():
    """Check CGR to RXN conversion on an aromatic CGR string."""
    cgr = "c1(C(C)=O)ccc2c(c1){c|C}{:|=}{c|C}{n|N}2{-|~}C(=O)(OC(C)(C)C){~|-}O"
    r, p = cgr_to_rxn(cgr).split(">>")

    exp_r = "c1(C(C)=O)ccc2c(c1)ccn2C(=O)OC(C)(C)C.O"
    exp_p = "c1(C(C)=O)ccc2c(c1)C=CN2.C(=O)(OC(C)(C)C)O"

    assert r == exp_r
    assert p == exp_p


@pytest.mark.parametrize(
    "smiles,kekule",
    [
        ("{[O:1]|[O+:1]}[C:2]([C:3])", True),
        ("{[O:1]|[O+:1]}[C:5]([c:2])", False),
    ],
)
def test_is_rxn_smiles_kekule(smiles, kekule):
    """Check if a given RXN SMILES is kekulized."""
    assert is_kekule(smiles) == kekule


@pytest.mark.parametrize(
    "smiles,kekule",
    [
        ("[O:1][C:2]([C:3])>>[O+:1][C:2]([C:3])", True),
        ("[O:1][C:2]([Cl:3])>>[O+:1][C:2]([Br:3])", True),
        ("[O:1][C:2]([C:3])>>[O+:1][C:2]([c:3])", False),
        ("[O:1][C:2]([c:3])>>[O+:1][C:2]([C:3])", False),
        ("[O:1][C:2]([c:3])>>[O+:1][C:2]([c:3])", False),
        ("[O:1][C:2]([1c:3])>>[O+:1][C:2]([C:3])", False),
        ("[O:1][C:2]([Cl:3])>>[O+:1][C:2]([c:3])", False),
    ],
)
def test_is_cgr_smiles_kekule(smiles, kekule):
    """Check if a given CGR SMILES is kekulized."""
    assert is_kekule(smiles) == kekule
