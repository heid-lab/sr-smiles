import random
from typing import Optional

from rdkit import Chem

from sr_smiles.chem_utils.mol_utils import make_mol


def augment_atom_traversal_order(
    rxn_smiles: str,
    random_state: Optional[random.Random] = None,
    kekulize: bool = False,
) -> str:
    """Randomizes atom traversal order independently in both reactants and products of a reaction SMILES.

    This augmentation preserves chemical validity while introducing non-canonical
    SMILES representations, which can improve model generalization in SMILES-based
    learning tasks.

    Note:
        RDKit's internal randomization (`doRandom=True`) is not reseeded between calls,
        so repeated calls in the same session may yield different outputs.

    Args:
        rxn_smiles (str): Reaction SMILES string.
        random_state (Optional[random.Random]): Optional random state for reproducibility.
        kekulize (bool): defaults to False.

    Returns:
        str: Augmented reaction SMILES.
    """
    r, _, p = rxn_smiles.split(">")
    mol_r = make_mol(r, kekulize=kekulize)
    mol_p = make_mol(p, kekulize=kekulize)

    # pick a random atom map shared by both as root
    map_nums_r = set(a.GetAtomMapNum() for a in mol_r.GetAtoms() if a.GetAtomMapNum() > 0)
    map_nums_p = set(a.GetAtomMapNum() for a in mol_p.GetAtoms() if a.GetAtomMapNum() > 0)
    shared_map_nums = list(map_nums_r & map_nums_p)

    if not shared_map_nums:
        return r + ">>" + p  # nothing to align on

    rng = random_state or random
    root_map = rng.choice(shared_map_nums)

    # get the atom indices corresponding to that map on each side
    root_idx_r = next(a.GetIdx() for a in mol_r.GetAtoms() if a.GetAtomMapNum() == root_map)
    root_idx_p = next(a.GetIdx() for a in mol_p.GetAtoms() if a.GetAtomMapNum() == root_map)

    # produce smiles rooted at those atoms with random traversal
    r_smi = Chem.MolToSmiles(mol_r, canonical=False, rootedAtAtom=root_idx_r, doRandom=True)
    p_smi = Chem.MolToSmiles(mol_p, canonical=False, rootedAtAtom=root_idx_p, doRandom=True)

    return r_smi + ">>" + p_smi


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
