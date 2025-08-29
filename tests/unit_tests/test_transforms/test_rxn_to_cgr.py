import pandas as pd
import pytest

from cgr_smiles.transforms.rxn_to_cgr import rxnsmiles_to_cgrsmiles
from cgr_smiles.utils import ROOT_DIR

TEST_DATA_PATH = ROOT_DIR / "tests" / "data" / "cgr_test_cases.csv"


# Load the test cases once
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
