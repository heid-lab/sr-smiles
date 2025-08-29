import pandas as pd
import pytest

from cgr_smiles.transforms.rxn_to_cgr import (
    remove_redundant_brackets,
    remove_redundant_brackets_and_hydrogens,
    rxnsmiles_to_cgrsmiles,
)
from cgr_smiles.utils import ROOT_DIR

TEST_DATA_PATH = ROOT_DIR / "tests" / "data" / "cgr_test_cases.csv"


def cgr_test_cases():
    """Loads test cases from a CSV file once per test module."""
    df = pd.read_csv(TEST_DATA_PATH)
    test_cases = list(zip(df["rxn"], df["rxn_smiles"], df["cgr_smiles"]))
    return test_cases


@pytest.mark.parametrize("rxn_id, rxn_smiles, cgr_smiles", cgr_test_cases())
def test_rxnsmiles_to_cgrsmiles(rxn_id, rxn_smiles, cgr_smiles):
    """Test the RXN to CGR (forward) transformation."""
    result = rxnsmiles_to_cgrsmiles(rxn_smiles, keep_atom_mapping=True)
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
def test_rxnsmiles_to_cgrsmiles_e_z_stereo(idx, rxn_smiles, cgr_smiles):
    """Test E/Z stereo changes in RXN to CGR transformation."""
    result = rxnsmiles_to_cgrsmiles(rxn_smiles, keep_atom_mapping=True)
    assert result == cgr_smiles, f"Assertion error for reaction with id {idx}"


def test_rxn_to_cgr_invalid_smiles(propagated_logger, caplog):
    """Verify that invalid RXN SMILES input logs a warning and returns an empty string."""
    bad_smi = "INVALID-RXN-SMILES"

    with caplog.at_level("WARNING", logger=propagated_logger.name):
        result = rxnsmiles_to_cgrsmiles(bad_smi)

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
