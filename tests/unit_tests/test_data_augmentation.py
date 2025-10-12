from collections import Counter
from typing import List

import pytest
from rdkit import Chem

from cgr_smiles.chem_utils.mol_utils import make_mol
from cgr_smiles.data_augmentation import (
    augment_atom_traversal_order,
    augment_reassign_atom_map_nums,
    augment_rxn_smiles,
)


@pytest.fixture
def rxn_smiles() -> str:
    """Return a sample reaction SMILES string for testing."""
    r_smi = "[C:1]([C:2]([C:3]([C:4]([C:5]([C:6]([H:18])([H:19])[H:20])([H:16])[H:17])([H:14])[H:15])([H:12])[H:13])([H:10])[H:11])([H:7])([H:8])[H:9]"  # noqa: E501
    p_smi = "[C:1]([C:2](=[C:3]([H:12])[H:13])[H:11])([H:7])([H:8])[H:9].[C:4](=[C:5]([C:6]([H:18])([H:19])[H:20])[H:16])([H:14])[H:15].[H:10][H:17]"  # noqa: E501
    return f"{r_smi}>>{p_smi}"


def mols_equal(smiles1, smiles2):
    """Check if two SMILES strings represent the same molecule."""
    mol1 = make_mol(smiles1)
    mol2 = make_mol(smiles2)
    return Chem.MolToSmiles(mol1, canonical=True) == Chem.MolToSmiles(mol2, canonical=True)


def test_augment_atom_traversal_order(rxn_smiles):
    """Verify that atom traversal order augmentation changes SMILES but preserves molecules."""
    aug_rxn_smiles = augment_atom_traversal_order(rxn_smiles)

    assert rxn_smiles != aug_rxn_smiles, "Augmentation did not change the rxn smiles string."

    r, _, p = rxn_smiles.split(">")
    r_aug, _, p_aug = aug_rxn_smiles.split(">")

    assert mols_equal(r, r_aug), "Augmented reac mol is not the same as the original reac mol."
    assert mols_equal(p, p_aug), "Augmented prod mol is not the same as the original prod mol."


def get_atom_map_nums(smi: str) -> List[int]:
    """Return the list of atom map numbers from a SMILES string."""
    mol = make_mol(smi)
    return [a.GetAtomMapNum() for a in mol.GetAtoms()]


def test_augment_reassign_atom_map_nums(rxn_smiles):
    """Test that atom map numbers are reassigned correctly while preserving counts and uniqueness."""
    new_rxn = augment_reassign_atom_map_nums(rxn_smiles)

    r_new, _, p_new = new_rxn.split(">")
    reac_nums = get_atom_map_nums(r_new)
    prod_nums = get_atom_map_nums(p_new)

    # assert atom map counts are unchanged and values are same in reactant/product
    assert Counter(reac_nums) == Counter(prod_nums)
    assert len(set(reac_nums)) == len(reac_nums), "Atom map numbers should be unique."
    assert reac_nums != get_atom_map_nums(rxn_smiles.split(">")[0]), "Atom maps have not been reassigned."


@pytest.mark.parametrize(
    "aug_atom_traversal_order, aug_atom_mapping, should_differ",
    [
        (False, False, False),
        (False, True, True),
        (True, False, True),
        (True, True, True),
    ],
)
def test_augment_rxn_smiles_parametrized(
    rxn_smiles, aug_atom_traversal_order, aug_atom_mapping, should_differ
):
    """Test data augmentation with different configurations."""
    augmented = augment_rxn_smiles(
        rxn_smiles,
        aug_atom_traversal_order=aug_atom_traversal_order,
        aug_atom_mapping=aug_atom_mapping,
    )

    assert isinstance(augmented, str)
    assert ">>" in augmented

    if should_differ:
        assert (
            rxn_smiles != augmented
        ), f"Augmented SMILES should differ from input\nrxn_smiles = {rxn_smiles}\naug_smiles = {augmented}"
    else:
        assert (
            rxn_smiles == augmented
        ), "Augmentation was set to `False, output should not differ from original"


# def test_augment_atom_traversal_order_deterministic(rxn_smiles):
#     """Check that atom traversal order augmentation is deterministic with a fixed RNG seed."""
#     initial_seed = 123
#     num_augmentation = 10

#     rng_run1 = random.Random(initial_seed)
#     results_run1 = []
#     for _ in range(num_augmentation):
#         results_run1.append(augment_atom_traversal_order(rxn_smiles, random_state=rng_run1))

#     rng_run2 = random.Random(initial_seed)
#     results_run2 = []
#     for _ in range(num_augmentation):
#         results_run2.append(augment_atom_traversal_order(rxn_smiles, random_state=rng_run2))

#     # check that the entire sequence of results from run1 matches run2
#     assert (
#         results_run1 == results_run2
#     ), "Augmentations with the same RNG seed should produce identical sequences, but differences were found"


# test_augment_atom_traversal_order_deterministic("[C:1]([C:2]([C:3]([C:4]([C:5]([C:6]([H:18])([H:19])[H:20])([H:16])[H:17])([H:14])[H:15])([H:12])[H:13])([H:10])[H:11])([H:7])([H:8])[H:9]>>[C:1]([C:2](=[C:3]([H:12])[H:13])[H:11])([H:7])([H:8])[H:9].[C:4](=[C:5]([C:6]([H:18])([H:19])[H:20])[H:16])([H:14])[H:15].[H:10][H:17]")  # noqa: E501
