import pandas as pd
import pytest

from cgr_smiles import ROOT_DIR
from cgr_smiles.transforms.rxn_to_cgr import (
    RxnToCgr,
    remove_redundant_brackets,
    remove_redundant_brackets_and_hydrogens,
    rxn_to_cgr,
)

TEST_DATA_PATH = ROOT_DIR / "tests" / "data" / "cgr_test_cases.csv"


def load_cgr_test_cases():
    """Loads test cases from a CSV file once per test module."""
    df = pd.read_csv(TEST_DATA_PATH)
    return df


DF = load_cgr_test_cases()
CGR_TEST_CASES = list(zip(DF["rxn"], DF["rxn_smiles"], DF["cgr_smiles"]))


@pytest.mark.parametrize("rxn_id, rxn_smiles, cgr_smiles", CGR_TEST_CASES)
def test_rxn_to_cgr(rxn_id, rxn_smiles, cgr_smiles):
    """Test the RXN to CGR (forward) transformation."""
    result = rxn_to_cgr(rxn_smiles, keep_atom_mapping=True)
    assert result == cgr_smiles, f"Assertion error for reaction with id {rxn_id}"


def e_z_stereo_test_cases():
    """Return test cases for E/Z stereochemistry transformations."""
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


@pytest.mark.parametrize("idx, rxn_smiles, cgr_smiles", e_z_stereo_test_cases())
def test_rxn_to_cgr_e_z_stereo(idx, rxn_smiles, cgr_smiles):
    """Test E/Z stereo changes in RXN to CGR transformation."""
    result = rxn_to_cgr(rxn_smiles, keep_atom_mapping=True)
    assert result == cgr_smiles, f"Assertion error for reaction with id {idx}"


# def a():
#     rxn = r"[F:1]/[CH:2]=[CH:3]/[F:4]>>[F:1]/[CH:2]=[CH:3]\[F:4]"
#     cgr = rxn_to_cgr(rxn, keep_atom_mapping=True)


# a()


@pytest.mark.parametrize("kekulize", [(True), (False)])
def test_rxn_to_cgr_kekulize(kekulize):
    """Test kekulize flag (True, False) in `rxn_to_cgr()`."""
    rxn_smi = "[CH:5]1=[C:1]([C:2]([CH3:3])=[O:4])[CH:9]=[C:8]2[C:7](=[CH:6]1)[NH:12][CH:11]=[CH:10]2.[O:20]([C:21]([O:22][C:23]([CH3:24])([CH3:26])[CH3:25])=[O:27])[C:13](=[O:14])[O:15][C:16]([CH3:17])([CH3:18])[CH3:19]>>[C:1]1([C:2]([CH3:3])=[O:4])=[CH:5][CH:6]=[C:7]2[C:8](=[CH:9]1)[CH:10]=[CH:11][N:12]2[C:13](=[O:14])[O:15][C:16]([CH3:17])([CH3:18])[CH3:19]"  # noqa: E501
    cgr_smi = rxn_to_cgr(rxn_smi, balance_rxn=True, kekulize=kekulize)
    assert kekulize == ("c" not in cgr_smi)


def test_rxn_to_cgr_keep_aromatic_bonds():
    """Test `keep_aromatic_bonds=True` in `rxn_to_cgr()`."""
    rxn_smi = "[C:6]([O:5][C:2]([CH3:4])([CH3:3])[CH3:1])(=[O:7])OC(=O)OC(C)(C)C.[cH:13]1[cH:12][c:11]([C:9]([CH3:8])=[O:10])[cH:19][c:18]2[cH:17][cH:16][nH:15][c:14]12>>[c:14]12[c:18]([cH:19][c:11]([C:9](=[O:10])[CH3:8])[cH:12][cH:13]1)[cH:17][cH:16][n:15]2[C:6]([O:5][C:2]([CH3:1])([CH3:3])[CH3:4])=[O:7]"  # noqa: E501
    cgr_smi = rxn_to_cgr(
        rxn_smi, balance_rxn=True, kekulize=False, keep_aromatic_bonds=True, remove_brackets=True
    )
    expected = "C(OC([CH3])([CH3])[CH3])(=O)({-|~}{O|[OH]}C(=O)OC([CH3])([CH3])[CH3]){~|-}{[nH]|n}1:[cH]:[cH]:c2:[cH]:c(C([CH3])=O):[cH]:[cH]:c:2:1"  # noqa: E501
    assert cgr_smi == expected


def test_rxn_to_cgr_do_not_keep_aromatic_bonds():
    """Test `keep_aromatic_bonds=False` in `rxn_to_cgr()`."""
    rxn_smi = "[C:6]([O:5][C:2]([CH3:4])([CH3:3])[CH3:1])(=[O:7])OC(=O)OC(C)(C)C.[cH:13]1[cH:12][c:11]([C:9]([CH3:8])=[O:10])[cH:19][c:18]2[cH:17][cH:16][nH:15][c:14]12>>[c:14]12[c:18]([cH:19][c:11]([C:9](=[O:10])[CH3:8])[cH:12][cH:13]1)[cH:17][cH:16][n:15]2[C:6]([O:5][C:2]([CH3:1])([CH3:3])[CH3:4])=[O:7]"  # noqa: E501
    cgr_smi = rxn_to_cgr(
        rxn_smi, balance_rxn=True, kekulize=False, keep_aromatic_bonds=False, remove_brackets=True
    )
    expected = "C(OC([CH3])([CH3])[CH3])(=O)({-|~}{O|[OH]}C(=O)OC([CH3])([CH3])[CH3]){~|-}{[nH]|n}1[cH][cH]c2[cH]c(C([CH3])=O)[cH][cH]c21"  # noqa: E501
    assert cgr_smi == expected


# test_rxn_to_cgr_do_not_keep_aromatic_bonds()


def test_rxn_to_cgr_invalid_smiles(propagated_logger, caplog):
    """Verify that invalid RXN SMILES input logs a warning and returns an empty string."""
    bad_smi = "INVALID-RXN-SMILES"

    with caplog.at_level("WARNING", logger=propagated_logger.name):
        result = rxn_to_cgr(bad_smi)

    assert result == ""
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "WARNING"
    assert f"Failed to process RXN-SMILES '{bad_smi}'" in record.message
    assert "Returning empty string." in record.message


@pytest.mark.parametrize(
    "cgr, expected",
    [
        ("[C]", "C"),
        ("[CH3]", "C"),
        ("[Fe]", "[Fe]"),
        ("[CH3][O]{[CH3]|[NH2]}", "CO{C|N}"),
        ("[CH3][O]{[CH3]|[NH3+]}", "CO{C|[NH3+]}"),
        ("{[CH3]|[CH2]}", "C"),
    ],
)
def test_remove_redundant_brackets_and_hydrogens(cgr, expected):
    """Test removal of redundant brackets and explicit hydrogens."""
    assert remove_redundant_brackets_and_hydrogens(cgr) == expected


@pytest.mark.parametrize(
    "cgr, expected",
    [
        ("[C]", "C"),
        ("[N]", "N"),
        ("[O]", "O"),
        ("[Fe]", "[Fe]"),
        ("[C+]", "[C+]"),
        ("[CH3]", "[CH3]"),
        ("[C][Fe][O]", "C[Fe]O"),
        ("[C]{[Fe]|[O]}", "C{[Fe]|O}"),
    ],
)
def test_remove_redundant_brackets(cgr, expected):
    """Test removal of redundant brackets."""
    assert remove_redundant_brackets(cgr) == expected


def test_RxnToCgr_single_string():
    """Test RxnToCgr class for single reaction."""
    transform = RxnToCgr(keep_atom_mapping=True)
    rxn_smiles = DF.iloc[0]["rxn_smiles"]
    exp_output = DF.iloc[0]["cgr_smiles"]
    result = transform(rxn_smiles)

    assert isinstance(result, str)
    assert result == exp_output


def test_RxnToCgr_list_of_strings():
    """Test RxnToCgr class for a list of reactions."""
    transform = RxnToCgr(keep_atom_mapping=True)
    rxn_smiles = DF["rxn_smiles"].tolist()
    exp_output = DF["cgr_smiles"].tolist()

    results = transform(rxn_smiles)

    assert isinstance(results, list)
    assert all(isinstance(r, str) for r in results)
    assert results == exp_output


def test_RxnToCgr_pd_series():
    """Test RxnToCgr class for a pd.Series input."""
    transform = RxnToCgr(keep_atom_mapping=True)
    rxn_smiles = DF["rxn_smiles"]
    exp_output = DF["cgr_smiles"]

    results = transform(rxn_smiles)

    assert isinstance(results, pd.Series)
    assert all(isinstance(r, str) for r in results)
    assert results.equals(exp_output)


def test_RxnToCgr_pd_df():
    """Test RxnToCgr class for a pd.Series object."""
    transform = RxnToCgr(keep_atom_mapping=True, rxn_col="rxn_smiles")
    df_rxn_smiles = DF
    exp_output = DF["cgr_smiles"]
    results = transform(df_rxn_smiles)

    assert isinstance(results, pd.Series)
    assert all(isinstance(r, str) for r in results)
    assert results.equals(exp_output)


def test_dataframe_without_rxn_col_raises():
    """Test that RxnToCgr call raises a ValueError isf `self.rxn_col` not set."""
    transform = RxnToCgr(keep_atom_mapping=True)
    df = pd.DataFrame({"rxn": ["A>>B"]})
    with pytest.raises(ValueError, match="`self.rxn_col` is not set"):
        transform(df)


def test_invalid_input_type():
    """Test that RxnToCgr call raises a TypeError if input type is not valid."""
    transform = RxnToCgr(keep_atom_mapping=True)
    with pytest.raises(TypeError):
        transform(12345)


@pytest.mark.parametrize(
    "empty_input,expected_type",
    [
        ("", str),
        ([], list),
        (pd.Series([], dtype=object), pd.Series),
        (pd.DataFrame({"rxn_smiles": []}), pd.Series),
    ],
)
def test_RxnToCgr_empty_inputs(empty_input, expected_type):
    """Test RxnToCgr call for empty inputs."""
    transform = RxnToCgr(keep_atom_mapping=True, rxn_col="rxn_smiles")

    result = transform(empty_input)

    assert isinstance(result, expected_type)

    if isinstance(result, (list, pd.Series)):
        assert len(result) == 0
    elif isinstance(result, str):
        assert result == ""
