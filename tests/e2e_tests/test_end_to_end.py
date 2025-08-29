import csv
from pathlib import Path

import pandas as pd
import pytest

from cgr_smiles.transforms.cgr_to_rxn import cgrsmiles_to_rxnsmiles
from cgr_smiles.transforms.rxn_to_cgr import rxnsmiles_to_cgrsmiles
from cgr_smiles.utils import ROOT_DIR, canonicalize

TEST_DATA_PATH = ROOT_DIR / "tests" / "data"


DATASET_CONFIGS = [
    ("rgd1/rgd1_full.csv", "smiles"),
    ("e2/test.csv", "AAM"),
    ("sn2/test.csv", "AAM"),
    ("rdb7/test.csv", "smiles"),
    ("rdb7/val.csv", "smiles"),
    ("rdb7/train.csv", "smiles"),
    ("cycloaddition/full_dataset.csv", "rxn_smiles"),
]


def generate_individual_tests(file_path: Path, rxn_col: str, max_samples: int = 1000):
    """Load and prepare individual reaction test cases for the entire test session.

    Reads multiple CSV datasets and returns a list of test cases and corresponding IDs.
    Each test case is a tuple containing:
        - file_path (str): Path to the CSV file.
        - idx (int): Row index in the CSV.
        - rxn (str): Reaction SMILES string.
        - rxn_col (str): Column name containing the reaction SMILES.

    Returns:
        tuple[list[tuple[str, int, str, str]], list[str]]:
            A tuple containing the list of test cases and a list of test IDs.
    """
    test_cases = []
    ids = []

    df = pd.read_csv(file_path)
    df = df.sample(n=min(max_samples, len(df)), random_state=42)

    for idx, row in df.iterrows():
        rxn = row[rxn_col]
        test_cases.append((file_path, idx, rxn, rxn_col))
        ids.append(f"{file_path}:{idx}")

    return test_cases, ids


E2E_FAILURES_CSV = ROOT_DIR / "tests" / "e2e" / "roundtrip_failures.csv"


# Ensure the CSV file is initialized with headers (optional, but useful)
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
        )  # Column headers


# Call this function once before running the tests
initialize_failures_csv_file()

for file, col in DATASET_CONFIGS:
    test_cases, ids = generate_individual_tests(TEST_DATA_PATH / file, col)

    @pytest.mark.parametrize("file_path, idx, rxn_smiles, rxn_col", test_cases, ids=ids)
    def test_roundtrip_per_sample(file_path, idx, rxn_smiles, rxn_col):
        """Test single sample roundtrip (RXN -> CGR -> RXN)."""
        rxn_can = canonicalize(rxn_smiles)
        cgr = rxnsmiles_to_cgrsmiles(rxn_smiles, keep_atom_mapping=True)
        res = cgrsmiles_to_rxnsmiles(cgr)
        res_can = canonicalize(res)

        if res_can != rxn_can:
            # Log the failing case to the CSV file
            with open(E2E_FAILURES_CSV, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([file_path, rxn_smiles, res, rxn_can, res_can, cgr])  # Write the failing case

        assert (
            res_can == rxn_can
        ), f"Mismatch at {file_path}:{idx}, cgr={cgr}, rxn_can={rxn_can}, res_can={res_can}"
