"""CGR SMILES representation of chemical reactions."""

from pathlib import Path

from cgr_smiles.io.logger import logger, set_verbose
from cgr_smiles.transforms.cgr_to_rxn import CgrToRxn, cgr_to_rxn
from cgr_smiles.transforms.rxn_to_cgr import RxnToCgr, rxn_to_cgr

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
__version__ = "0.0.1"

__all__ = [
    "cgr_smiles",
    "cgr_to_rxn",
    "CgrToRxn",
    "RxnToCgr",
    "rxn_to_cgr",
    "logger",
    "set_verbose",
]
