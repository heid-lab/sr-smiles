from typing import Dict, List, Optional, Tuple

import rdkit
from rdkit import Chem

from sr_smiles.io.logger import logger


def make_mol(smi: str, sanitize: bool = True, kekulize: bool = True) -> Chem.Mol:
    """Creates an RDKit molecule object from a SMILES string, with optional sanitization.

    Args:
        smi (str): A SMILES string representing the molecule.
        sanitize (bool, optional): Whether to sanitize the molecule after parsing. Defaults to True.
            Sanitization applies all standard checks except hydrogen adjustment.
        kekulize (bool, optional): If True, converts all aromatic atoms/bonds into a Kekulé form with
            explicit single/double bonds. Defaults to False (keep aromatic SMILES notation).

    Returns:
        Chem.Mol: An RDKit molecule object constructed from the SMILES string.
    """
    if ">>" in smi:
        raise ValueError(
            "`make_mol()` received a reaction SMILES, but it expects a SMILES string for a single molecule."
        )

    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if sanitize:
        sanitize_flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS
        if kekulize:
            sanitize_flags ^= Chem.SanitizeFlags.SANITIZE_SETAROMATICITY

        Chem.SanitizeMol(mol, sanitizeOps=sanitize_flags)

    Chem.AssignStereochemistry(mol)

    try:
        test_mol = Chem.Mol(mol)  # make a copy
        rdkit.RDLogger.DisableLog("rdApp.error")
        Chem.Kekulize(test_mol)
        rdkit.RDLogger.EnableLog("rdApp.error")

    except Chem.KekulizeException:
        # workaround to prevent aromatic [nH] from being converted to [n]
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        # record which atoms had explicit H
        mol.UpdatePropertyCache(strict=False)
        nH_atoms = {
            atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0
        }

        if sanitize:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
            )
        Chem.AssignStereochemistry(mol)

        # re-add explicit H to those atoms
        for idx in nH_atoms:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetTotalNumHs() == 0:
                atom.SetNumExplicitHs(1)
                atom.UpdatePropertyCache()

    return mol


def reorder_mol(target_mol: Chem.Mol, reference_mol: Chem.Mol) -> Chem.Mol:
    """Reorder atoms in a target molecule to match the atom map order of a reference molecule.

    Args:
        target_mol (Chem.Mol): Molecule to reorder.
        reference_mol (Chem.Mol): Molecule whose atom map order defines the desired ordering.

    Returns:
        Chem.Mol: Target molecule with atoms reordered according to the reference.
    """
    target_map_to_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in target_mol.GetAtoms()}
    new_order = [target_map_to_idx[a.GetAtomMapNum()] for a in reference_mol.GetAtoms()]
    return Chem.RenumberAtoms(target_mol, new_order)


def get_atom_by_map_num(mol: Chem.Mol, atom_map_num: int) -> Optional[Chem.Atom]:
    """Retrieve the atom from the molecule that has the specified atom map number.

    Args:
        mol (Chem.Mol): The RDKit molecule to search.
        atom_map_num (int): The atom map number to find.

    Returns:
        Optional[Chem.Atom]: The atom with the matching atom map number if found,
        otherwise None.
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == atom_map_num:
            return atom
    return None


def get_atom_map_nums_of_mol(mol: Chem.Mol) -> List[int]:
    """Retrieves the atom map numbers of all atoms in an RDKit molecule.

    Args:
        mol (Chem.Mol): The RDKit molecule.

    Returns:
        List[int]: The list of atom map numbers of the molecule.
    """
    map_nums = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    return map_nums


def get_reac_to_prod_mapping(mol_reac: Chem.Mol, mol_prod: Chem.Mol) -> Dict[int, int]:
    """Creates mapping from reactant atom indices to product atom indices based on atom mapping numbers.

    Args:
        mol_reac (Chem.Mol): RDKit molecule object for the reactants, with atom mapping numbers.
        mol_prod (Chem.Mol): RDKit molecule object for the products, with atom mapping numbers.

    Returns:
        Dict[int, int]: A dictionary mapping reactant atom indices (keys) to corresponding product atom
            indices (values).
    """
    prod_map_to_id = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol_prod.GetAtoms()}
    reac_id_to_prod_id = {atom.GetIdx(): prod_map_to_id[atom.GetAtomMapNum()] for atom in mol_reac.GetAtoms()}
    return reac_id_to_prod_id


def remove_bonds_by_atom_map_nums(mol: Chem.Mol, atom_map_pairs: List[Tuple[int, int]]) -> Chem.Mol:
    """Removes specified bonds from an RDKit molecule based on atom map number pairs.

    Args:
        mol (Chem.Mol): The input RDKit molecule object.
        atom_map_pairs (List[Tuple[int, int]]): A list of atom map tuples to be removed.

    Returns:
        Chem.Mol: A new RDKit molecule object with the specified bonds removed.
            If a bond corresponding to a given pair of atom map numbers does not exist,
            a warning is logged, and that pair is skipped.

    """
    atom_map_to_idx = {}
    for atom in mol.GetAtoms():
        atom_map_to_idx[atom.GetAtomMapNum()] = atom.GetIdx()

    emol = Chem.EditableMol(mol)

    bonds_to_remove_by_idx = []
    for am1, am2 in atom_map_pairs:
        idx1 = atom_map_to_idx[am1]
        idx2 = atom_map_to_idx[am2]

        bond = mol.GetBondBetweenAtoms(idx1, idx2)
        if bond:
            bonds_to_remove_by_idx.append((idx1, idx2))
        else:
            logger.warning(f"No bond found between atom map numbers {am1} and {am2}. Skipping removal.")

    for idx1, idx2 in bonds_to_remove_by_idx:
        emol.RemoveBond(idx1, idx2)

    final_mol = emol.GetMol()
    return final_mol


# def get_atom_map_num(mol: Chem.Mol, atom_idx: int) -> int:
#     """Retrieves the atom mapping number for a specified atom in an RDKit molecule.

#     Args:
#         mol (Chem.Mol): The RDKit molecule.
#         atom_idx (int): The index of the atom within the molecule.

#     Returns:
#         int: The atom mapping number of the specified atom.
#     """
#     num_atoms = mol.GetNumAtoms()
#     assert (
#         0 <= atom_idx < mol.GetNumAtoms()
#     ), f"Error: Atom index {atom_idx} is out of bounds for mol with {num_atoms} atoms."

#     atom = mol.GetAtomWithIdx(atom_idx)
#     return atom.GetAtomMapNum()

# def get_atom_indices_and_smarts(mol: Chem.Mol) -> List[Tuple[int, str]]:
#     """Extract each atom's index and its corresponding SMARTS representation from the molecule.

#     Args:
#         mol (Chem.Mol): The molecule to process.

#     Returns:
#         List[Tuple[int, str]]: A list of tuples where each contains an atom's index and its SMARTS string.
#     """
#     num_atoms = mol.GetNumAtoms()
#     atom_indices = [
#         (i, mol.GetAtomWithIdx(i).GetSmarts(isomericSmiles=True, allHsExplicit=True))
#         for i in range(num_atoms)
#     ]
#     return atom_indices


# def get_bond_idx(mol: Chem.Mol, atom_idx1: int, atom_idx2: int) -> Optional[int]:
#     """Get the bond index between two atoms in a molecule.

#     Args:
#         mol (Chem.Mol): The molecule containing the atoms.
#         atom_idx1 (int): Index of the first atom.
#         atom_idx2 (int): Index of the second atom.

#     Returns:
#         Optional[int]: The bond index if a bond exists between the two atoms; otherwise None.
#     """
#     bond = mol.GetBondBetweenAtoms(atom_idx1, atom_idx2)
#     return bond.GetIdx() if bond is not None else None
