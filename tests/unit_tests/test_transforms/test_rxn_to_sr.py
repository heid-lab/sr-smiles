import pandas as pd
import pytest

from sr_smiles import ROOT_DIR
from sr_smiles.transforms.rxn_to_sr import (
    RxnToSr,
    remove_redundant_brackets,
    remove_redundant_brackets_and_hydrogens,
    rxn_to_sr,
)

TEST_DATA_PATH = ROOT_DIR / "tests" / "data" / "sr_test_cases.csv"


def load_sr_test_cases():
    """Loads test cases from a CSV file once per test module."""
    df = pd.read_csv(TEST_DATA_PATH)
    return df


DF = load_sr_test_cases()
SR_TEST_CASES = list(zip(DF["rxn"], DF["rxn_smiles"], DF["sr_smiles"]))


@pytest.mark.parametrize("rxn_id, rxn_smiles, sr_smiles", SR_TEST_CASES)
def test_rxn_to_sr(rxn_id, rxn_smiles, sr_smiles):
    """Test the RXN to sr-SMILES (forward) transformation."""
    result = rxn_to_sr(rxn_smiles, keep_atom_mapping=True)
    assert result == sr_smiles, f"Assertion error for reaction with id {rxn_id}"


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


@pytest.mark.parametrize("idx, rxn_smiles, sr_smiles", e_z_stereo_test_cases())
def test_rxn_to_sr_e_z_stereo(idx, rxn_smiles, sr_smiles):
    """Test E/Z stereo changes in RXN to sr-SMILES transformation."""
    result = rxn_to_sr(rxn_smiles, keep_atom_mapping=True)
    assert result == sr_smiles, f"Assertion error for reaction with id {idx}"


@pytest.mark.parametrize("kekulize", [(True), (False)])
def test_rxn_to_sr_kekulize(kekulize):
    """Test kekulize flag (True, False) in `rxn_to_sr()`."""
    rxn_smi = "[CH:5]1=[C:1]([C:2]([CH3:3])=[O:4])[CH:9]=[C:8]2[C:7](=[CH:6]1)[NH:12][CH:11]=[CH:10]2.[O:20]([C:21]([O:22][C:23]([CH3:24])([CH3:26])[CH3:25])=[O:27])[C:13](=[O:14])[O:15][C:16]([CH3:17])([CH3:18])[CH3:19]>>[C:1]1([C:2]([CH3:3])=[O:4])=[CH:5][CH:6]=[C:7]2[C:8](=[CH:9]1)[CH:10]=[CH:11][N:12]2[C:13](=[O:14])[O:15][C:16]([CH3:17])([CH3:18])[CH3:19]"  # noqa: E501
    sr_smi = rxn_to_sr(rxn_smi, balance_rxn=True, kekulize=kekulize)
    assert kekulize == ("c" not in sr_smi)


def test_rxn_to_sr_keep_aromatic_bonds():
    """Test `keep_aromatic_bonds=True` in `rxn_to_sr()`."""
    rxn_smi = "[C:6]([O:5][C:2]([CH3:4])([CH3:3])[CH3:1])(=[O:7])OC(=O)OC(C)(C)C.[cH:13]1[cH:12][c:11]([C:9]([CH3:8])=[O:10])[cH:19][c:18]2[cH:17][cH:16][nH:15][c:14]12>>[c:14]12[c:18]([cH:19][c:11]([C:9](=[O:10])[CH3:8])[cH:12][cH:13]1)[cH:17][cH:16][n:15]2[C:6]([O:5][C:2]([CH3:1])([CH3:3])[CH3:4])=[O:7]"  # noqa: E501
    sr_smi = rxn_to_sr(
        rxn_smi,
        balance_rxn=True,
        kekulize=False,
        keep_aromatic_bonds=True,
    )
    expected = "C(OC([CH3])([CH3])[CH3])(=O)({-|~}{O|[OH]}C(=O)OC([CH3])([CH3])[CH3]){~|-}{[nH]|n}1:[cH]:[cH]:c2:[cH]:c(C([CH3])=O):[cH]:[cH]:c:2:1"  # noqa: E501
    assert sr_smi == expected


def test_rxn_to_sr_do_not_keep_aromatic_bonds():
    """Test `keep_aromatic_bonds=False` in `rxn_to_sr()`."""
    rxn_smi = "[C:6]([O:5][C:2]([CH3:4])([CH3:3])[CH3:1])(=[O:7])OC(=O)OC(C)(C)C.[cH:13]1[cH:12][c:11]([C:9]([CH3:8])=[O:10])[cH:19][c:18]2[cH:17][cH:16][nH:15][c:14]12>>[c:14]12[c:18]([cH:19][c:11]([C:9](=[O:10])[CH3:8])[cH:12][cH:13]1)[cH:17][cH:16][n:15]2[C:6]([O:5][C:2]([CH3:1])([CH3:3])[CH3:4])=[O:7]"  # noqa: E501
    sr_smi = rxn_to_sr(
        rxn_smi,
        balance_rxn=True,
        kekulize=False,
        keep_aromatic_bonds=False,
    )
    expected = "C(OC([CH3])([CH3])[CH3])(=O)({-|~}{O|[OH]}C(=O)OC([CH3])([CH3])[CH3]){~|-}{[nH]|n}1[cH][cH]c2[cH]c(C([CH3])=O)[cH][cH]c21"  # noqa: E501
    assert sr_smi == expected


# test_rxn_to_sr_do_not_keep_aromatic_bonds()


def test_rxn_to_sr_invalid_smiles(propagated_logger, caplog):
    """Verify that invalid RXN SMILES input logs a warning and returns an empty string."""
    bad_smi = "INVALID-RXN-SMILES"

    with caplog.at_level("WARNING", logger=propagated_logger.name):
        result = rxn_to_sr(bad_smi)

    assert result == ""
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "WARNING"
    assert f"Failed to process RXN SMILES '{bad_smi}'" in record.message
    assert "Returning empty string." in record.message


@pytest.mark.parametrize(
    "sr_smi, expected",
    [
        ("[C]", "C"),
        ("[CH3]", "C"),
        ("[Fe]", "[Fe]"),
        ("[CH3][O]{[CH3]|[NH2]}", "CO{C|N}"),
        ("[CH3][O]{[CH3]|[NH3+]}", "CO{C|[NH3+]}"),
        ("{[CH3]|[CH2]}", "C"),
    ],
)
def test_remove_redundant_brackets_and_hydrogens(sr_smi, expected):
    """Test removal of redundant brackets and explicit hydrogens."""
    assert remove_redundant_brackets_and_hydrogens(sr_smi) == expected


@pytest.mark.parametrize(
    "sr_smi, expected",
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
def test_remove_redundant_brackets(sr_smi, expected):
    """Test removal of redundant brackets."""
    assert remove_redundant_brackets(sr_smi) == expected


def test_RxnToSr_single_string():
    """Test RxnToSr class for single reaction."""
    transform = RxnToSr(keep_atom_mapping=True)
    rxn_smiles = DF.iloc[0]["rxn_smiles"]
    exp_output = DF.iloc[0]["sr_smiles"]
    result = transform(rxn_smiles)

    assert isinstance(result, str)
    assert result == exp_output


def test_RxnToSr_list_of_strings():
    """Test RxnToSr class for a list of reactions."""
    transform = RxnToSr(keep_atom_mapping=True)
    rxn_smiles = DF["rxn_smiles"].tolist()
    exp_output = DF["sr_smiles"].tolist()

    results = transform(rxn_smiles)

    assert isinstance(results, list)
    assert all(isinstance(r, str) for r in results)
    assert results == exp_output


def test_RxnToSr_pd_series():
    """Test RxnToSr class for a pd.Series input."""
    transform = RxnToSr(keep_atom_mapping=True)
    rxn_smiles = DF["rxn_smiles"]
    exp_output = DF["sr_smiles"]

    results = transform(rxn_smiles)

    assert isinstance(results, pd.Series)
    assert all(isinstance(r, str) for r in results)
    assert results.equals(exp_output)


def test_RxnToSr_pd_df():
    """Test RxnToSr class for a pd.Series object."""
    transform = RxnToSr(keep_atom_mapping=True, rxn_col="rxn_smiles")
    df_rxn_smiles = DF
    exp_output = DF["sr_smiles"]
    results = transform(df_rxn_smiles)

    assert isinstance(results, pd.Series)
    assert all(isinstance(r, str) for r in results)
    assert results.equals(exp_output)


def test_RxnToSr_with_rxnmapper():
    """Test RxnToSr class with rxnmapper for unmapped reaction."""
    try:
        transform = RxnToSr(use_rxnmapper=True, balance_rxn=True)
    except ImportError:
        pytest.skip("rxnmapper not installed or not compatible")

    # Unmapped reaction: ethanol oxidation
    rxn_smiles = "CCO>>CC=O"
    result = transform(rxn_smiles)

    assert isinstance(result, str)
    assert result != ""  # Should produce a valid sr-SMILES
    assert "{" in result  # Should contain sr notation for bond changes


def test_dataframe_without_rxn_col_raises():
    """Test that RxnToSr call raises a ValueError isf `self.rxn_col` not set."""
    transform = RxnToSr(keep_atom_mapping=True)
    df = pd.DataFrame({"rxn": ["A>>B"]})
    with pytest.raises(ValueError, match="`self.rxn_col` is not set"):
        transform(df)


def test_invalid_input_type():
    """Test that RxnToSr call raises a TypeError if input type is not valid."""
    transform = RxnToSr(keep_atom_mapping=True)
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
def test_RxnToSr_empty_inputs(empty_input, expected_type):
    """Test RxnToSr call for empty inputs."""
    transform = RxnToSr(keep_atom_mapping=True, rxn_col="rxn_smiles")

    result = transform(empty_input)

    assert isinstance(result, expected_type)

    if isinstance(result, (list, pd.Series)):
        assert len(result) == 0
    elif isinstance(result, str):
        assert result == ""
