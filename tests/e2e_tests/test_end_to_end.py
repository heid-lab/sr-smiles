import csv

import pandas as pd
import pytest

from cgr_smiles.transforms.cgr_to_rxn import cgr_to_rxn
from cgr_smiles.transforms.rxn_to_cgr import rxn_to_cgr
from cgr_smiles.utils import ROOT_DIR, canonicalize

TEST_DATA_PATH = ROOT_DIR / "tests" / "data"


def generate_individual_tests(max_samples: int = 1000):
    """Load and prepare individual reaction test cases for the entire test session.

    Reads multiple CSV datasets and returns a list of test cases and corresponding IDs.
    Each test case is a tuple containing:
        - file_path (str): Path to the CSV file.
        - idx (int): Row index in the CSV.
        - rxn (str): Reaction SMILES string.
        - rxn_col (str): Column name containing the reaction SMILES.

    Returns:
        Tuple[List[Tuple[str, int, str, str]], List[str]]:
            A tuple containing the list of test cases and a list of test IDs.
    """
    test_cases = []
    ids = []

    dataset_configs = [
        ("rgd1/rgd1_full.csv", "smiles"),
        ("e2/test.csv", "AAM"),
        ("sn2/test.csv", "AAM"),
        ("rdb7/test.csv", "smiles"),
        ("rdb7/val.csv", "smiles"),
        ("rdb7/train.csv", "smiles"),
        ("cycloaddition/full_dataset.csv", "rxn_smiles"),
    ]

    for file_path, rxn_col in dataset_configs:
        full_path = TEST_DATA_PATH / file_path
        df = pd.read_csv(full_path)
        df = df.sample(n=min(max_samples, len(df)), random_state=42)

        for idx, row in df.iterrows():
            rxn = row[rxn_col]
            test_cases.append((file_path, idx, rxn, rxn_col))
            ids.append(f"{file_path}:{idx}")

    return test_cases, ids


E2E_FAILURES_CSV = ROOT_DIR / "tests" / "e2e_tests" / "roundtrip_failures.csv"


def initialize_failures_csv_file():
    """Create file to store failed e2e test cases to."""
    with open(E2E_FAILURES_CSV, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "file_path",
                "rxn_smiles",
                "e2e_rxn_smiles",
                "can_rxn_smiles",
                "can_e2e_rxn_smiles",
                "cgr_smiles",
            ]
        )


# Call this function once before running the tests
initialize_failures_csv_file()
test_cases, ids = generate_individual_tests()


@pytest.mark.e2e
@pytest.mark.parametrize("file_path, idx, rxn_smiles, rxn_col", test_cases, ids=ids)
def test_roundtrip_per_sample(file_path, idx, rxn_smiles, rxn_col):
    """Test single sample roundtrip (RXN -> CGR -> RXN)."""
    rxn_can = canonicalize(rxn_smiles)
    cgr = rxn_to_cgr(rxn_smiles, keep_atom_mapping=True)
    res = cgr_to_rxn(cgr)
    res_can = canonicalize(res)

    if res_can != rxn_can:
        # Log the failing case to the CSV file
        with open(E2E_FAILURES_CSV, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([file_path, rxn_smiles, res, rxn_can, res_can, cgr])  # Write the failing case

    assert (
        res_can == rxn_can
    ), f"Mismatch at {file_path}:{idx}, cgr={cgr}, rxn_can={rxn_can}, res_can={res_can}"
