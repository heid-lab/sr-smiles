import pandas as pd
import pytest
import csv
import os

from cgr_smiles.cgr_to_rxn import cgrsmiles_to_rxnsmiles
from cgr_smiles.rxn_to_cgr import rxnsmiles_to_cgrsmiles
from cgr_smiles.utils import ROOT_DIR, canonicalize

TEST_DATA_PATH = ROOT_DIR / "data" / "cgr"


def generate_individual_tests():
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

        for idx, row in df.iterrows():
            rxn = row[rxn_col]
            test_cases.append((file_path, idx, rxn, rxn_col))
            ids.append(f"{file_path}:{idx}")

    return test_cases, ids


test_cases, ids = generate_individual_tests()


@pytest.mark.parametrize("file_path, idx, rxn_smiles, rxn_col", test_cases, ids=ids)
def test_roundtrip_per_sample(file_path, idx, rxn_smiles, rxn_col):
    """
    Test single sample roundtrip (rxn -> CGR -> rxn).
    Failures show which exact sample failed, even on internal errors.
    """
    rxn_can = canonicalize(rxn_smiles)
    cgr = rxnsmiles_to_cgrsmiles(rxn_smiles, keep_atom_mapping=True)
    res = cgrsmiles_to_rxnsmiles(cgr)
    res_can = canonicalize(res)
    if res_can != rxn_can:
        file = "/home/charlotte.gerhaher/projects/chemtorch/tests/data/failed_testcases_25-08-11.csv"
        # Check if file exists to decide whether to write header
        file_exists = os.path.isfile(file)

        with open(file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["dataset", "idx", "rxn_smiles", "cgr"])
            writer.writerow([file_path, idx, rxn_smiles, cgr])

    assert (
        res_can == rxn_can
    ), f"Mismatch at {file_path}:{idx}, cgr={cgr}, rxn_can={rxn_can}, res_can={res_can}"


# @pytest.mark.parametrize(
#     "test_config_mapped_and_balanced",
#     [
#         ("e2/test.csv", "AAM"),  # Test case for dataset E2, preserve_maps=True
#         ("sn2/test.csv", "AAM"),
#         ("rdb7/test.csv", "smiles"),
#         ("rdb7/val.csv", "smiles"),
#         ("rdb7/train.csv", "smiles"),
#         ("cycloaddition/full_dataset.csv", "rxn_smiles"),

#         # ("uspto_50k_test", "uspto_50k/raw_test.csv", "reactants>reagents>production", True),
#         # ("uspto_50k_val", "uspto_50k/raw_val.csv", "reactants>reagents>production", True),
#         # ("uspto_50k_train", "uspto_50k/raw_train.csv", "reactants>reagents>production", True),

#         # ("rgd1_dataset", "test_data_e2.csv", "rxn_smiles", True),

#     ],
#     ids=[
#         "e2",
#         "sn2",
#         "rdb7_test",
#         "rdb7_val",
#         "rdb7_train",
#         "cycloadd_no_maps",
#         # "uspto_50k_test",
#         # "uspto_50k_val",
#         # "uspto_50k_train",
#         # "rgd1_no_maps",
#     ]
# )
# def test_roundtrip_mapped_and_balanced(
#     test_config_mapped_and_balanced,
# ):
#     """
#     Tests the end-to-end roundtrip (rxn -> CGR -> rxn) for various datasets and flag settings.
#     Records results to a CSV file.
#     """
#     file_path, rxn_col = test_config_mapped_and_balanced
#     df = pd.read_csv(TEST_DATA_PATH / file_path)

#     for idx, row in df.iterrows():
#         rxn_smiles = row[rxn_col]
#         rxn_can = canonicalize(rxn_smiles)
#         cgr = rxnsmiles_to_cgrsmiles(rxn_smiles, keep_atom_mapping=True)
#         res = cgrsmiles_to_rxnsmiles(cgr)
#         res_can = canonicalize(res)
#         assert res_can == rxn_can, f"Assertion error for reaction {idx}"


# def test1_rxnsmiles_to_cgr_to_rxn_roundtrip(
#     # test_config,
# ):
#     """
#     Tests the end-to-end roundtrip (rxn -> CGR -> rxn) for various datasets and flag settings.
#     Records results to a CSV file.
#     """
#     # file_path, rxn_col = test_config
#     # file_path, rxn_col = ("rdb7/test.csv", "smiles")
#     # file_path, rxn_col = ("e2/test.csv", "AAM")
#     # file_path, rxn_col = ("sn2/test.csv", "AAM")
#     # file_path, rxn_col = ("cycloaddition/full_dataset.csv", "rxn_smiles")
#     # file_path, rxn_col = ("rgd1/rgd1_full.csv", "smiles")
#     file_path, rxn_col = ("rgd1/52265.csv", "smiles")
#     df = pd.read_csv(TEST_DATA_PATH / file_path)

#     for idx, row in df.iterrows():
#         if idx == 0:
#             print(f"idx = {idx}")
#             rxn_smiles = row[rxn_col]
#             rxn_can = canonicalize(rxn_smiles)
#             cgr = rxnsmiles_to_cgrsmiles(rxn_smiles, keep_atom_mapping=True)
#             res = cgrsmiles_to_rxnsmiles(cgr)
#             res_can = canonicalize(res)
#             assert res_can == rxn_can, f"Assertion error for reaction {idx}"


# test1_rxnsmiles_to_cgr_to_rxn_roundtrip()
