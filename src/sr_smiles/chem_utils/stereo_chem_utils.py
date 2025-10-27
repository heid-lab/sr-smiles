from collections import defaultdict
from typing import Dict, List, Tuple

from rdkit import Chem

from sr_smiles.chem_utils.list_utils import common_elements_preserving_order, is_num_permutations_even
from sr_smiles.chem_utils.smiles_utils import (
    extract_chiral_tag_by_atom_map_num,
    get_atom_map_adjacency_list_from_smiles,
)


def is_chiral_center(atom: Chem.Atom) -> bool:
    """Checks whether an RDKit atom is a tetrahedral chiral center.

    Args:
        atom (Chem.Atom): RDKit atom object to check.

    Returns:
        bool: True if the atom is a tetrahedral chiral center, False otherwise.
    """
    return atom.GetChiralTag() in (
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    )


def get_chiral_center_map_nums(mol: Chem.Mol) -> List[int]:
    """Returns the atom map numbers of chiral centers in an RDKit molecule.

    Args:
        mol (Chem.Mol): The input RDKit molecule object.

    Returns:
        List[int]: A list of integer atom map numbers corresponding to the chiral centers
            found in the molecule.
    """
    atom_map_nums = []
    for atom in mol.GetAtoms():
        if is_chiral_center(atom):
            atom_map_nums.append(atom.GetAtomMapNum())
    return atom_map_nums


def find_e_z_stereo_bonds(
    bond_dict: Dict[Tuple[int, int], str],
) -> Dict[Tuple[int, int], Dict[str, any]]:
    r"""Identifies E/Z stereochemistry of double bonds from bond data.

    Parses a dictionary representing molecule bonds and their types. It identifies
    double bonds and determines their E/Z stereochemistry by examiningstereo bond
    specifiers ('/' or '\') on adjacent bonds connected to the double bond atoms.

    Args:
        bond_dict (Dict[Tuple[int, int], str]): A dictionary where keys are tuples of atom indices,
            and values are strings indicating the bond type.

    Returns:
        Dict[Tuple[int, int], Dict[str, any]: A dictionary where keys are tuples of atom indices
            `(idx_a, idx_b)` representing the double bond, and values are dictionaries containing:
                - "stereo": An RDKit `Chem.BondStereo` enum value (STEREOE or STEREOZ).
                - "terminal_atoms": A tuple of the two atom indices `(neighbor_a_idx, neighbor_b_idx)`
                    that define the stereochemistry for each side of the double bond.
                    The dictionary contains entries for both `(a, b)` and `(b, a)` for the same
                    double bond.
    """
    if not bond_dict:
        return {}

    neighbors_map = defaultdict(list)
    stereo_bond_map = {}
    double_bonds = set()

    for (a1, a2), bond_type in bond_dict.items():
        neighbors_map[a1].append((a2, bond_type))
        neighbors_map[a2].append((a1, flip_e_z_stereo(bond_type)))

        if bond_type == "=":
            double_bonds.add(frozenset((a1, a2)))
        elif bond_type in ("/", "\\"):
            stereo_bond_map[(a1, a2)] = bond_type
            stereo_bond_map[(a2, a1)] = flip_e_z_stereo(bond_type)

    results = {}

    for db_pair in double_bonds:
        atom_a, atom_b = tuple(db_pair)

        stereo_arms_a = []
        for neighbor_atom_a, _ in neighbors_map.get(atom_a, []):
            if (atom_a, neighbor_atom_a) in stereo_bond_map:
                stereo_arms_a.append(
                    (
                        atom_a,
                        neighbor_atom_a,
                        stereo_bond_map[(atom_a, neighbor_atom_a)],
                    )
                )

        stereo_arms_b = []
        for neighbor_atom_b, _ in neighbors_map.get(atom_b, []):
            if (atom_b, neighbor_atom_b) in stereo_bond_map:
                stereo_arms_b.append(
                    (
                        atom_b,
                        neighbor_atom_b,
                        stereo_bond_map[(atom_b, neighbor_atom_b)],
                    )
                )

        if stereo_arms_a and stereo_arms_b:
            arm_a = stereo_arms_a[0]
            arm_b = stereo_arms_b[0]

            neighbor_a, slash_type_a = arm_a[1], flip_e_z_stereo(arm_a[2])
            neighbor_b, slash_type_b = arm_b[1], arm_b[2]

            if slash_type_a == slash_type_b:
                stereo = Chem.BondStereo.STEREOE
            else:
                stereo = Chem.BondStereo.STEREOZ

            results[(atom_a, atom_b)] = {
                "stereo": stereo,
                "terminal_atoms": (neighbor_a, neighbor_b),
            }
            results[(atom_b, atom_a)] = {
                "stereo": stereo,
                "terminal_atoms": (neighbor_b, neighbor_a),
            }

    return results


def flip_e_z_stereo(smiles: str) -> str:
    r"""Flips E/Z stereochemistry in a string by switching '/' and '\\' bond E/Z-stereochemistry markers.

    Args:
        smiles (str): A SMILES string that may contain E/Z stereochemistry indicators ('/' or '\\').

    Returns:
        str: The SMILES string with flipped E/Z stereochemistry.
    """
    flipped = []
    for ch in smiles:
        if ch == "/":
            flipped.append("\\")
        elif ch == "\\":
            flipped.append("/")
        else:
            flipped.append(ch)
    return "".join(flipped)


def update_chirality_in_mol_from_smiles(mol: Chem.Mol, smi: str, smi_ref: str) -> None:
    """Updates the stereochemistry of chiral centers in a molecule.

    Updates the stereochemistry of chiral centers in a molecule based on changes in atom
    neighbor ordering between the original and refernce SMILES (e.g. a SR-scaffold SMILES).

    This function is intended to fix incorrect stereochemistry caused by atom reordering
    during transformations (e.g., from reaction SMILES to SR and back). It works by:
      - Extracting the neighbor atom map number orderings from both the original SMILES (`smi`)
        and the SR-generated SMILES (`smi_sr`).
      - Iterating over each chiral atom in the molecule.
      - Comparing the neighbor orderings from `smi` and `smi_sr` for each atom.
      - If the permutation between the two orderings is odd (i.e., the parity has flipped),
        the atom's stereochemistry is inverted (CW ↔ CCW).
      - The updated stereochemistry is set directly on the RDKit molecule (`mol`), in place.

    Args:
        mol (Chem.Mol): The RDKit molecule object to update in place.
        smi (str): The original mapped reaction SMILES (or molecule SMILES) string.
        smi_ref (str): The SR-transformed SMILES string that may have altered atom orderings.
    """
    d_smi = get_atom_map_adjacency_list_from_smiles(smi)
    d_sr = get_atom_map_adjacency_list_from_smiles(smi_ref)

    for atom in mol.GetAtoms():
        chiral_tag = atom.GetChiralTag()
        if chiral_tag in (
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        ):
            atom_map_num = atom.GetAtomMapNum()
            smiles_chiral_tag = extract_chiral_tag_by_atom_map_num(smi, atom_map_num)

            l1 = d_smi[atom_map_num]
            l2 = d_sr[atom_map_num]
            l1, l2 = common_elements_preserving_order(l1, l2)

            if not is_num_permutations_even(l1, l2):
                smiles_chiral_tag = "@@" if smiles_chiral_tag == "@" else "@"

            if smiles_chiral_tag == "@@":
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            elif smiles_chiral_tag == "@":
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
