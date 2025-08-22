from rxnmapper import RXNMapper


import csv

import pytest
from cgr_smiles.reaction_balancing import balance_reaction
from cgr_smiles.utils import get_list_of_atom_map_numbers


# Load CSV data once
def load_reaction_cases():
    csv_path = "/home/charlotte.gerhaher/projects/chemtorch/data/uspto_1k/test.csv"
    cases = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append((row["reaction"]))
            if len(cases) == 10:
                break
    mapper = RXNMapper()
    result = mapper.get_attention_guided_atom_maps(cases, canonicalize_rxns=False)
    result = [entry["mapped_rxn"] for entry in result]
    return result


@pytest.mark.parametrize("rxn_input", load_reaction_cases())
def test_balance_reaction(rxn_input):
    # for rxn1 in rxn_input:
    # rxn1 = "[H:1][O:3][H:2]>>[H:1][H:2].[O:3]=[O:4]"
    bln1 = balance_reaction(rxn_input)
    reac, prod = bln1.split(">>")
    map_nums_reac = get_list_of_atom_map_numbers(reac)
    map_nums_prod = get_list_of_atom_map_numbers(prod)

    assert set(map_nums_reac) == set(map_nums_prod)


# test_balance_reaction(load_reaction_cases())
