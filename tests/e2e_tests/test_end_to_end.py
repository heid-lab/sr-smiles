import csv

import pandas as pd
import pytest
from conftest import equivalent_reactions
from rdkit import Chem

from sr_smiles import ROOT_DIR
from sr_smiles.chem_utils.smiles_utils import canonicalize
from sr_smiles.transforms.rxn_to_sr import RxnToSr, rxn_to_sr
from sr_smiles.transforms.sr_to_rxn import SrToRxn, sr_to_rxn

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
                "sr_smiles",
            ]
        )


# Call this function once before running the tests
initialize_failures_csv_file()
test_cases, ids = generate_individual_tests()


@pytest.mark.e2e
@pytest.mark.parametrize("file_path, idx, rxn_smiles, rxn_col", test_cases, ids=ids)
def test_roundtrip_per_sample(file_path, idx, rxn_smiles, rxn_col):
    """Test single sample roundtrip (RXN -> sr -> RXN)."""
    rxn_can = canonicalize(rxn_smiles)
    sr = rxn_to_sr(rxn_smiles, keep_atom_mapping=True)

    res = sr_to_rxn(sr)
    res_can = canonicalize(res)

    if res_can != rxn_can:
        # Log the failing case to the CSV file
        with open(E2E_FAILURES_CSV, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([file_path, rxn_smiles, res, rxn_can, res_can, sr])  # Write the failing case

    assert res_can == rxn_can, f"Mismatch at {file_path}:{idx}, sr={sr}, rxn_can={rxn_can}, res_can={res_can}"


@pytest.mark.e2e
@pytest.mark.parametrize("file_path, idx, rxn_smiles, rxn_col", test_cases, ids=ids)
def test_roundtrip_per_sample_with_unmapped_sr_smiles(file_path, idx, rxn_smiles, rxn_col):
    """Test single sample roundtrip (RXN -> sr -> RXN)."""
    rxn_can = canonicalize(rxn_smiles)
    sr = rxn_to_sr(rxn_smiles, keep_atom_mapping=False)

    res = sr_to_rxn(sr, add_atom_mapping=True)
    res_can = canonicalize(res)

    assert equivalent_reactions(rxn_smiles, res)
    assert equivalent_reactions(rxn_can, res_can)


def canonicalize_without_mapping(smi: str) -> str:
    """Remove atom mapping and return canonical (Kekulé) SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")

    # Strip atom-map numbers
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    # Canonicalize
    return Chem.MolToSmiles(mol, canonical=True)


def are_equivalent_rxn_smiles(smi1: str, smi2: str) -> bool:
    """Return True if two SMILES encode the same structure, ignoring atom mapping."""
    reac1, prod1 = smi1.split(">>")
    can1_reac = canonicalize_without_mapping(reac1)
    can1_prod = canonicalize_without_mapping(prod1)

    reac2, prod2 = smi2.split(">>")
    can2_reac = canonicalize_without_mapping(reac2)
    can2_prod = canonicalize_without_mapping(prod2)

    return can1_reac == can2_reac and can1_prod == can2_prod


subset_test_cases, subset_ids = generate_individual_tests(max_samples=10)


@pytest.fixture(scope="module")
def transform_back():
    """Single backward transformer instance (sr → RXN)."""
    return SrToRxn()


@pytest.fixture(params=[True, False], ids=["rxnmapper", "no_mapper"])
def forward_transformer(request):
    """Forward RxnToSr transformer, parameterized by use_rxnmapper."""
    return RxnToSr(use_rxnmapper=request.param, keep_atom_mapping=True)


@pytest.mark.parametrize("file_path, idx, rxn_smiles, rxn_col", subset_test_cases, ids=subset_ids)
def test_RxnToSr_roundtrip_with_mapping(
    forward_transformer, transform_back, file_path, idx, rxn_smiles, rxn_col
):
    """Ensure round-trip RXN → sr → RXN equivalence with/without RxnMapper."""
    sr = forward_transformer(rxn_smiles)
    rxn_back = transform_back(sr)

    assert are_equivalent_rxn_smiles(rxn_smiles, rxn_back), (
        f"Round-trip mismatch for sample {file_path}:{idx} "
        f"using use_rxnmapper={forward_transformer.use_rxnmapper}"
    )


@pytest.mark.parametrize("n_jobs", [-1, 2, 1, 0])
def test_RxnToSr_parallel_vs_serial(n_jobs):
    """Verify that parallel processing (n_jobs != 1) yields identical results to serial."""
    # Use the first 5 reactions from the pre‑defined subset
    batch_size = 5
    rxn_smiles_list = [case[2] for case in subset_test_cases[:batch_size]]

    # Create transformers with identical settings except n_jobs
    # Use keep_atom_mapping=True for consistency, use_rxnmapper=False for speed
    transformer_serial = RxnToSr(
        keep_atom_mapping=True,
        remove_hydrogens=False,
        balance_rxn=False,
        kekulize=False,
        keep_aromatic_bonds=True,
        use_rxnmapper=False,
        n_jobs=1,
    )
    transformer_parallel = RxnToSr(
        keep_atom_mapping=True,
        remove_hydrogens=False,
        balance_rxn=False,
        kekulize=False,
        keep_aromatic_bonds=True,
        use_rxnmapper=False,
        n_jobs=n_jobs,
    )

    # Transform the batch with both transformers
    results_serial = transformer_serial(rxn_smiles_list)
    results_parallel = transformer_parallel(rxn_smiles_list)

    # Compare results
    assert len(results_serial) == len(results_parallel) == batch_size
    for i, (sr_serial, sr_parallel) in enumerate(zip(results_serial, results_parallel)):
        assert sr_serial == sr_parallel, (
            f"Parallel (n_jobs={n_jobs}) output differs from serial at index {i}. "
            f"Reaction: {rxn_smiles_list[i]}, serial: {sr_serial}, parallel: {sr_parallel}"
        )


@pytest.mark.parametrize("n_jobs", [-1, 2, 1, 0])
def test_SrToRxn_parallel_vs_serial(n_jobs):
    """Verify parallel processing yields identical results to serial for backward transformation."""
    # Use the first 5 reactions from the pre‑defined subset, convert them to sr-SMILES first
    batch_size = 5
    rxn_smiles_list = [case[2] for case in subset_test_cases[:batch_size]]

    # Convert to sr-SMILES using serial transformer
    forward_transformer = RxnToSr(
        keep_atom_mapping=True,
        remove_hydrogens=False,
        balance_rxn=False,
        kekulize=False,
        keep_aromatic_bonds=True,
        use_rxnmapper=False,
        n_jobs=1,
    )
    sr_smiles_list = forward_transformer(rxn_smiles_list)

    # Create backward transformers with identical settings except n_jobs
    transformer_serial = SrToRxn(add_atom_mapping=True, n_jobs=1)
    transformer_parallel = SrToRxn(add_atom_mapping=True, n_jobs=n_jobs)

    # Transform the batch with both transformers
    results_serial = transformer_serial(sr_smiles_list)
    results_parallel = transformer_parallel(sr_smiles_list)

    # Compare results
    assert len(results_serial) == len(results_parallel) == batch_size
    for i, (rxn_serial, rxn_parallel) in enumerate(zip(results_serial, results_parallel)):
        assert rxn_serial == rxn_parallel, (
            f"Parallel (n_jobs={n_jobs}) output differs from serial at index {i}. "
            f"sr-SMILES: {sr_smiles_list[i]}, serial: {rxn_serial}, parallel: {rxn_parallel}"
        )


@pytest.mark.parametrize("n_jobs_forward", [-1, 2])
@pytest.mark.parametrize("n_jobs_backward", [-1, 2])
def test_roundtrip_with_parallel_processing(n_jobs_forward, n_jobs_backward):
    """Ensure round-trip RXN → sr → RXN equivalence with parallel processing matches serial output."""
    # Use a single reaction for simplicity
    test_case = subset_test_cases[0]
    rxn_smiles = test_case[2]

    # Create serial transformers (baseline)
    forward_serial = RxnToSr(
        keep_atom_mapping=True,
        remove_hydrogens=False,
        balance_rxn=False,
        kekulize=False,
        keep_aromatic_bonds=True,
        use_rxnmapper=False,
        n_jobs=1,
    )
    backward_serial = SrToRxn(add_atom_mapping=True, n_jobs=1)

    # Create parallel transformers
    forward_parallel = RxnToSr(
        keep_atom_mapping=True,
        remove_hydrogens=False,
        balance_rxn=False,
        kekulize=False,
        keep_aromatic_bonds=True,
        use_rxnmapper=False,
        n_jobs=n_jobs_forward,
    )
    backward_parallel = SrToRxn(add_atom_mapping=True, n_jobs=n_jobs_backward)

    # Perform serial roundtrip (baseline)
    sr_serial = forward_serial(rxn_smiles)
    rxn_back_serial = backward_serial(sr_serial)

    # Perform parallel roundtrip
    sr_parallel = forward_parallel(rxn_smiles)
    rxn_back_parallel = backward_parallel(sr_parallel)

    # 1. Verify both roundtrips are correct (equivalent to original)
    assert are_equivalent_rxn_smiles(
        rxn_smiles, rxn_back_serial
    ), f"Serial round-trip mismatch. Original: {rxn_smiles}, Roundtrip: {rxn_back_serial}"
    assert are_equivalent_rxn_smiles(rxn_smiles, rxn_back_parallel), (
        f"Parallel round-trip mismatch (n_jobs_forward={n_jobs_forward}, "
        f"n_jobs_backward={n_jobs_backward}). Original: {rxn_smiles}, Roundtrip: {rxn_back_parallel}"
    )

    # 2. Verify parallel output matches serial output
    assert are_equivalent_rxn_smiles(rxn_back_serial, rxn_back_parallel), (
        f"Parallel output differs from serial (n_jobs_forward={n_jobs_forward}, "
        f"n_jobs_backward={n_jobs_backward}). Serial: {rxn_back_serial}, Parallel: {rxn_back_parallel}"
    )
