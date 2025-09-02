import subprocess
from pathlib import Path

import pandas as pd
import pytest

from cgr_smiles.utils import ROOT_DIR

TEST_DATA_PATH = ROOT_DIR / "tests" / "data"


@pytest.mark.e2e
def test_cli_rxn2cgr_and_cgr2rxn(tmp_path: Path):
    """Test the end-to-end CLI workflow from reaction to CGR and back."""
    input_file = TEST_DATA_PATH / "rdb7" / "test.csv"
    forward_file = tmp_path / "cli_test_cases_forward.csv"
    backward_file = tmp_path / "cli_test_cases_backward.csv"

    # run forward transformation (rxn -> cgr)
    subprocess.run(
        [
            "rxn2cgr",
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
    assert "cgr_smiles" in df_forward.columns
    assert not df_forward["cgr_smiles"].isnull().any()

    # run backward transformation (cgr -> rxn)
    subprocess.run(
        [
            "cgr2rxn",
            str(forward_file),
            "-o",
            str(backward_file),
            "--cgr-col=cgr_smiles",
        ],
        check=True,
    )

    # check backward output
    df_backward = pd.read_csv(backward_file)
    assert "rxn_smiles" in df_backward.columns
    assert not df_backward["rxn_smiles"].isnull().any()
