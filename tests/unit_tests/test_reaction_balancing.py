import csv

import pytest
from rxnmapper import RXNMapper

from cgr_smiles.reaction_balancing import balance_reaction
from cgr_smiles.utils import ROOT_DIR, get_list_of_atom_map_numbers


@pytest.fixture(scope="session")
def load_unbalanced_reaction_cases():
    """Load and map 100 unbalanced reaction SMILES from USPTO-50k dataset."""
    n_samples = 100
    csv_path = ROOT_DIR / "tests" / "data" / "uspto_1k" / "test.csv"
    cases = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(row["reaction"])
            if len(cases) == n_samples:
                break
    mapper = RXNMapper()
    result = mapper.get_attention_guided_atom_maps(cases, canonicalize_rxns=False)
    return [entry["mapped_rxn"] for entry in result]


@pytest.mark.parametrize("rxn_input", range(100))
def test_balance_reaction(load_unbalanced_reaction_cases, rxn_input):
    """Check that unbalanced reactions are being balanced."""
    rxn = load_unbalanced_reaction_cases[rxn_input]
    bln1 = balance_reaction(rxn)
    reac, prod = bln1.split(">>")
    map_nums_reac = get_list_of_atom_map_numbers(reac)
    map_nums_prod = get_list_of_atom_map_numbers(prod)

    assert set(map_nums_reac) == set(map_nums_prod)
