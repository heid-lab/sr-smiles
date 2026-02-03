import subprocess
from pathlib import Path

import pandas as pd
import pytest

from sr_smiles import ROOT_DIR

TEST_DATA_PATH = ROOT_DIR / "tests" / "data"


@pytest.mark.e2e
def test_cli_rxn2sr_and_sr2rxn(tmp_path: Path):
    """Test the end-to-end CLI workflow from reaction to sr-SMILES and back."""
    input_file = TEST_DATA_PATH / "rdb7" / "test.csv"
    forward_file = tmp_path / "cli_test_cases_forward.csv"
    backward_file = tmp_path / "cli_test_cases_backward.csv"

    # run forward transformation (rxn -> sr)
    subprocess.run(
        [
            "rxn2sr",
            str(input_file),
            "-o",
            str(forward_file),
            "--rxn-col=smiles",
            "--keep-atom-mapping",
        ],
        check=True,
    )

    # check forward output
    df_forward = pd.read_csv(forward_file)
    assert "sr_smiles" in df_forward.columns
    assert not df_forward["sr_smiles"].isnull().any()

    # run backward transformation (sr -> rxn)
    subprocess.run(
        [
            "sr2rxn",
            str(forward_file),
            "-o",
            str(backward_file),
            "--sr-col=sr_smiles",
        ],
        check=True,
    )

    # check backward output
    df_backward = pd.read_csv(backward_file)
    assert "rxn_smiles" in df_backward.columns
    assert not df_backward["rxn_smiles"].isnull().any()
