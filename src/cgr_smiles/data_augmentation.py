import random
from typing import Optional

from rdkit import Chem

from cgr_smiles.utils import make_mol


def augment_atom_traversal_order(
    rxn_smiles: str,
    random_state: Optional[random.Random] = None,
) -> str:
    """Randomizes atom traversal order independently in both reactants and products of a reaction SMILES.

    This augmentation preserves chemical validity while introducing non-canonical
    SMILES representations, which can improve model generalization in SMILES-based
    learning tasks.

    Args:
        rxn_smiles (str): Reaction SMILES string.
        random_state (Optional[random.Random]): Optional random state for reproducibility.

    Returns:
        str: Augmented reaction SMILES.
    """
    r, _, p = rxn_smiles.split(">")
    mol_reac = make_mol(r)
    mol_prod = make_mol(p)

    rng = random_state if random_state is not None else random.Random()

    # reac shuffle
    atom_nums_reac = list(range(mol_reac.GetNumAtoms()))
    rng.shuffle(atom_nums_reac)
    mol_reac = Chem.RenumberAtoms(mol_reac, atom_nums_reac)

    # prod shuffle
    atom_nums_prod = list(range(mol_prod.GetNumAtoms()))
    rng.shuffle(atom_nums_prod)
    mol_prod = Chem.RenumberAtoms(mol_prod, atom_nums_prod)

    r_smi_shuffled = Chem.MolToSmiles(mol_reac, canonical=False)
    p_smi_shuffled = Chem.MolToSmiles(mol_prod, canonical=False)

    return ">>".join([r_smi_shuffled, p_smi_shuffled])


def augment_reassign_atom_map_nums(rxn_smiles: str) -> str:
    """Randomly reassigns atom map numbers in both reactants and products of a reaction SMILES.

    Atom mappings are preserved one-to-one but shuffled to introduce
    non-canonical numbering.

    Args:
        rxn_smiles (str): Reaction SMILES string.

    Returns:
        str: Augmented reaction SMILES with reassigned atom map numbers.
    """
    r, _, p = rxn_smiles.split(">")
    mol_reac, mol_prod = make_mol(r), make_mol(p)
    atom_nums = [a.GetAtomMapNum() for a in mol_reac.GetAtoms()]
    atom_nums_shuffled = random.sample(atom_nums, k=len(atom_nums))

    atom_num_old2new = {old_num: new_num for old_num, new_num in zip(atom_nums, atom_nums_shuffled)}

    for atom in mol_reac.GetAtoms():
        atom.SetAtomMapNum(atom_num_old2new[atom.GetAtomMapNum()])

    for atom in mol_prod.GetAtoms():
        atom.SetAtomMapNum(atom_num_old2new[atom.GetAtomMapNum()])

    r_smi_shuffled = Chem.MolToSmiles(mol_reac, canonical=False)
    p_smi_shuffled = Chem.MolToSmiles(mol_prod, canonical=False)

    return ">>".join([r_smi_shuffled, p_smi_shuffled])


def augment_rxn_smiles(rxn_smiles, aug_atom_traversal_order: bool, aug_atom_mapping: bool) -> str:
    """Apply augmentation to a reaction SMILES string.

    The function can optionally shuffle the atom traversal order and/or
    reassign atom map numbers to generate an augmented version of the reaction.

    Args:
        rxn_smiles (str): The reaction SMILES string to augment.
        aug_atom_traversal_order (bool): Whether to shuffle atom traversal order.
        aug_atom_mapping (Callable): A function to reassign atom map numbers.

    Returns:
        str: Augmented reaction SMILES string.
    """
    if aug_atom_traversal_order:
        rxn_smiles = augment_atom_traversal_order(rxn_smiles)

    if aug_atom_mapping:
        rxn_smiles = augment_reassign_atom_map_nums(rxn_smiles)

    return rxn_smiles
