import enum
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import rdkit
from rdkit import Chem

ROOT_DIR = Path(__file__).parent.parent.parent

BOND_TYPES = {"-", "~", "=", "#", "$", ":", "."}
ORGANIC_SUBSET = {
    "B",
    "C",
    "N",
    "O",
    "S",
    "P",
    "F",
    "Cl",
    "Br",
    "I",
    "*",
    "b",
    "c",
    "n",
    "o",
    "s",
    "p",
}  # "*" is a "wildcard" or unknown atom


@enum.unique
class TokenType(enum.Enum):
    """Possible SMILES token types."""

    ATOM = 1
    BOND_TYPE = 2
    BRANCH_START = 3
    BRANCH_END = 4
    RING_NUM = 5
    EZSTEREO = 6
    CHIRAL = 7
    OTHER = 8


def _tokenize(smiles: str) -> Iterator[Tuple[TokenType, int, Union[str, int]]]:
    """Tokenize a SMILES string into atoms, bonds, branches, and stereochemistry.

    Recognizes standard organic atoms, bond types, ring numbers, branching,
    cis/trans stereochemistry, and `{reac|prod}` blocks.

    Args:
        smiles (str): The SMILES string to tokenize.

    Yields:
        Iterator[Tuple[TokenType, int, Union[str, int]]]: Tuples of
            (token_type, original_index_in_string, token_value).
    """
    s_iter = iter(smiles)
    token = ""
    idx = -1
    peek: Optional[str] = None

    while True:
        idx += 1
        char = peek if peek is not None else next(s_iter, "")
        peek = None

        if not char:
            break

        if char == "[":
            token = char
            for c in s_iter:
                token += c
                if c == "]":
                    break
            else:
                raise ValueError(f"Unmatched '[' in SMILES string at index {idx}")
            yield TokenType.ATOM, idx, token
        elif char in ORGANIC_SUBSET:
            next_char = next(s_iter, "")
            if (
                char + next_char in ORGANIC_SUBSET and next_char.islower()
            ):  # for aromatic 'Cl', 'cl', 'Br', 'br'
                yield TokenType.ATOM, idx, char + next_char
            else:
                yield TokenType.ATOM, idx, char
                peek = next_char  # put back the char if it wasn't part of a 2-char atom
        elif char in BOND_TYPES:
            yield TokenType.BOND_TYPE, idx, char
        elif char == "(":
            yield TokenType.BRANCH_START, idx, "("
        elif char == ")":
            yield TokenType.BRANCH_END, idx, ")"
        elif char == "%":
            # For two-digit ring numbers, e.g., %10
            digit1 = next(s_iter, "")
            digit2 = next(s_iter, "")
            if not (digit1 and digit2 and digit1.isdigit() and digit2.isdigit()):
                raise ValueError(f"Malformed two-digit ring number at index {idx}")
            yield TokenType.RING_NUM, idx, f"%{digit1}{digit2}"
        elif char in ("/", "\\"):
            yield TokenType.EZSTEREO, idx, char
        # elif char in ("@", "@@"):
        #     next_char = next(s_iter, "")
        #     if char + next_char == "@@":
        #         yield TokenType.CHIRAL, idx, char + next_char
        #     else:
        #         yield TokenType.CHIRAL, idx, char
        #         peek = next_char
        elif char.isdigit():
            yield TokenType.RING_NUM, idx, char
        else:
            # Handle any characters not explicitly covered.
            yield TokenType.OTHER, idx, char


def remove_atom_mapping(rxn_smiles: str) -> str:
    """Removes atom mapping numbers from a reaction SMILES string.

    Args:
        rxn_smiles (str): A reaction SMILES string with atom mappings.

    Returns:
        str: Reaction SMILES string without atom mappings.
    """
    # return re.sub(r":\d+", "", rxn_smiles)
    toks = []
    for tokentype, token_original_idx, token_val in _tokenize(rxn_smiles):
        if tokentype == TokenType.ATOM:
            token_val = re.sub(r":\d+", "", token_val)
        toks.append(str(token_val))

    return "".join(toks)


def remove_redundant_square_brackets(rxn_smiles: str) -> str:
    """Removes only the redundant square brackets in a reaction SMILES string.

    Redundant square brackets are those that enclose uncharged, non-chiral atoms from
    the organic subset. For example, [C] becomes C, while brackets are retained for
    cases like [C-], [C@H], and [H], where additional chemical information is present.

    Args:
        rxn_smiles (str): A reaction SMILES string possibly containing redundant brackets.

    Returns:
        str: Reaction SMILES string with unnecessary square brackets removed.
    """
    escaped_subset = [re.escape(elem) for elem in sorted(ORGANIC_SUBSET, key=len, reverse=True)]
    pattern = rf"\[({'|'.join(escaped_subset)})\]"

    def replacer(match):
        return match.group(1)

    return re.sub(pattern, replacer, rxn_smiles)


def parse_bonds_in_order_from_smiles(smiles: str) -> Dict[Tuple[int, int], str]:
    """Parses SMILES to map bond atom-map pairs to their bond specifiers.

    This function traverses the SMILES token by token, identifying bonds by
    their connecting atom map numbers and extracting their explicit bond type.

    Args:
        smiles (str): SMILES string of a molecule.

    Returns:
        Dict[Tuple[int, int], str]: A dict mapping sorted `(atom_map_num_1, atom_map_num_2)`
            Tuples to their bond specifier string.

    Raises:
        ValueError: If the CGR SMILES string has malformed syntax.
    """
    replace_dict_bonds = {}
    anchor_logical_idx = None
    next_bond_specifier = None
    branches = []
    ring_open_bonds = {}

    logical_idx_to_map_num = {}
    current_logical_idx = 0

    for tokentype, token_original_idx, token_val in _tokenize(smiles):
        if tokentype == TokenType.ATOM:
            # Extract atom map number or assign a temporary one if none.
            atom_map_match = re.search(r":(\d+)", str(token_val))
            current_atom_map_num = (
                int(atom_map_match.group(1)) if atom_map_match else (current_logical_idx + 1000)
            )

            logical_idx_to_map_num[current_logical_idx] = current_atom_map_num

            if anchor_logical_idx is not None:
                # We have a bond between anchor_logical_idx and current_logical_idx
                bond_map_num_pair = (
                    logical_idx_to_map_num[anchor_logical_idx],
                    current_atom_map_num,
                )

                # Determine the bond specification
                # If next_bond_specifier is None, it implies a single bond by default.
                bond_val = next_bond_specifier if next_bond_specifier is not None else "-"

                # Store the bond
                replace_dict_bonds[bond_map_num_pair] = bond_val

            current_logical_idx += 1
            anchor_logical_idx = current_logical_idx - 1  # This new atom becomes the anchor for next bond
            next_bond_specifier = None  # Clear any pending bond specifier

        elif tokentype == TokenType.BOND_TYPE or tokentype == TokenType.EZSTEREO:
            # These are standard bond types (-,=,#,:,. or E/Z stereo)
            next_bond_specifier = str(token_val)

        elif tokentype == TokenType.BRANCH_START:
            branches.append(anchor_logical_idx)  # Push current anchor onto stack

        elif tokentype == TokenType.BRANCH_END:
            if not branches:
                raise ValueError(f"Unmatched ')' in SMILES string at index {token_original_idx}")
            anchor_logical_idx = branches.pop()  # Pop anchor from stack
            next_bond_specifier = (
                None  # Branch closure typically implies implicit single bond if no explicit one.
            )

        elif tokentype == TokenType.RING_NUM:
            ring_num_val = str(token_val)  # Ring numbers can be int or string (for %XX)

            if ring_num_val in ring_open_bonds:  # Ring closure (we found a matching number)
                logical_idx_opener, bond_opener_specifier = ring_open_bonds[ring_num_val]

                # Bond is between the current atom (anchor_logical_idx) and the atom that opened the ring
                bond_map_num_pair = (
                    logical_idx_to_map_num[anchor_logical_idx],
                    logical_idx_to_map_num[logical_idx_opener],
                )

                # Determine the bond specification for this ring closure
                # If there's a bond specifier immediately before this ring_num_val, use it.
                # Otherwise, use the bond specifier that was active when the ring *opened*.
                bond_val = (
                    next_bond_specifier
                    if next_bond_specifier is not None
                    else (bond_opener_specifier if bond_opener_specifier is not None else "-")
                )

                # Store the bond
                replace_dict_bonds[bond_map_num_pair] = bond_val

                del ring_open_bonds[ring_num_val]  # Remove from open rings
                next_bond_specifier = None  # Clear any pending bond specifier

            else:
                ring_open_bonds[ring_num_val] = (
                    anchor_logical_idx,
                    next_bond_specifier,
                )
                next_bond_specifier = None

    return replace_dict_bonds


def get_atom_map_adjacency_list_from_smiles(smi: str) -> Dict[int, List[int]]:
    """Creates a dictionary, mapping each atom map number to a list of adjacent map numbers.

    The list of adjacent atom map numbers reflects the order in which bonds are encountered
    during the SMILES traversal.

    Args:
        smi (str): A SMILES string with atom mapping.

    Returns:
        Dict[int, List[int]]: A dictionary where keys are atom map numbers and values are
            lists of adjacent atom map numbers, representing the molecular graph connectivity.
    """
    bonds = parse_bonds_in_order_from_smiles(smi)
    adj_dict = {}
    for (map_num_1, map_num_2), bond_type in bonds.items():
        if bond_type != ".":
            if map_num_1 not in adj_dict:
                adj_dict[map_num_1] = []
            if map_num_2 not in adj_dict:
                adj_dict[map_num_2] = []
            adj_dict[map_num_1].append(map_num_2)
            adj_dict[map_num_2].append(map_num_1)
    return adj_dict


def extract_chiral_tag_by_atom_map_num(smiles: str, atom_map_num: int) -> str:
    """Extracts the chiral tag ("@" or "@@") for an atom in a SMILES string based on its atom map number.

    Args:
        smiles (str): A SMILES string where atoms may have chiral tags and mapping numbers.
        atom_map_num (int): The atom map number to search for within the SMILES.

    Returns:
        str: The chiral tag associated with the atom (either "@", "@@", or "" if no tag is found).

    Example:
        >>> extract_chiral_tag_by_atom_map_num("[C@H:1](F)Cl", 1)
        "@"
    """
    atom_tokens = [tok for tok_type, _, tok in _tokenize(smiles) if tok_type == TokenType.ATOM]
    for tok in atom_tokens:
        match = re.search(r":(\d+)\]", tok)
        if match:
            curr_atom_map_num = int(match.group(1))
            if curr_atom_map_num == atom_map_num:
                chiral_match = re.search(r"@{1,2}", tok)
                if chiral_match:
                    return chiral_match.group(0)
                else:
                    return ""
    return ""


def update_all_atom_stereo(mol: Chem.Mol, smi: str, smi_ref: str) -> None:
    """Updates the stereochemistry of chiral centers in a molecule.

    Updates the stereochemistry of chiral centers in a molecule based on changes in atom
    neighbor ordering between the original and refernce SMILES (e.g. a CGR-scaffold SMILES).

    This function is intended to fix incorrect stereochemistry caused by atom reordering
    during transformations (e.g., from reaction SMILES to CGR and back). It works by:
      - Extracting the neighbor atom map number orderings from both the original SMILES (`smi`)
        and the CGR-generated SMILES (`smi_cgr`).
      - Iterating over each chiral atom in the molecule.
      - Comparing the neighbor orderings from `smi` and `smi_cgr` for each atom.
      - If the permutation between the two orderings is odd (i.e., the parity has flipped),
        the atom's stereochemistry is inverted (CW ↔ CCW).
      - The updated stereochemistry is set directly on the RDKit molecule (`mol`), in place.

    Args:
        mol (Chem.Mol): The RDKit molecule object to update in place.
        smi (str): The original mapped reaction SMILES (or molecule SMILES) string.
        smi_ref (str): The CGR-transformed SMILES string that may have altered atom orderings.
    """
    d_smi = get_atom_map_adjacency_list_from_smiles(smi)
    d_cgr = get_atom_map_adjacency_list_from_smiles(smi_ref)

    for atom in mol.GetAtoms():
        chiral_tag = atom.GetChiralTag()
        if chiral_tag in (
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        ):
            atom_map_num = atom.GetAtomMapNum()
            smiles_chiral_tag = extract_chiral_tag_by_atom_map_num(smi, atom_map_num)

            l1 = d_smi[atom_map_num]
            l2 = d_cgr[atom_map_num]
            l1, l2 = common_elements_preserving_order(l1, l2)

            if not is_num_permutations_even(l1, l2):
                smiles_chiral_tag = "@@" if smiles_chiral_tag == "@" else "@"

            if smiles_chiral_tag == "@@":
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            elif smiles_chiral_tag == "@":
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)


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


def is_num_permutations_even(l1: list, l2: list) -> bool:
    """Determines if the permutation to transform list `l1` into `l2` involves an even min number of swaps.

    Args:
        l1 (list): The original list of elements.
        l2 (list): The target list of elements, a permutation of `l1`.

    Returns:
        bool: True if the permutation from `l1` to `l2` is even (i.e., involves an even number of swaps),
              False if it is odd.
    """
    target_map = {val: i for i, val in enumerate(l2)}

    visited_indices = [False] * len(l1)
    num_cycles = 0

    for i in range(len(l1)):
        if not visited_indices[i]:
            num_cycles += 1
            current_idx = i
            while not visited_indices[current_idx]:
                visited_indices[current_idx] = True
                element_in_list1 = l1[current_idx]
                current_idx = target_map[element_in_list1]

    num_swaps = len(l1) - num_cycles
    return num_swaps % 2 == 0


def common_elements_preserving_order(list1: list, list2: list) -> Tuple[list, list]:
    """Returns the common elements between two lists, preserving the order of each original list.

    Args:
        list1 (list): The first list of elements.
        list2 (list): The second list of elements.

    Returns:
        Tuple[list, list]: Two lists containing only the common elements from `list1` and `list2`,
        respectively, with their original order maintained.
    """
    set2 = set(list2)
    set1 = set(list1)
    filtered1 = [x for x in list1 if x in set2]
    filtered2 = [x for x in list2 if x in set1]
    return filtered1, filtered2


def includes_individually_mapped_hydrogens(smiles: str) -> bool:
    """Check if a SMILES string includes individually mapped hydrogens, e.g., [H:2].

    Args:
        smiles (str): The SMILES string to check.

    Returns:
        bool: True if mapped hydrogens are found, False otherwise.
    """
    # Regular expression to match [H:<number>], optionally with isotope or charge
    pattern = r"\[H(?::\d+)\]"

    res = bool(re.search(pattern, smiles))
    if not res:
        # check if any hydrogens at all
        res_h = "H" in smiles

        return not res_h

    else:
        return res


def make_mol(smi: str, sanitize: bool = True) -> Chem.Mol:
    """Creates an RDKit molecule object from a SMILES string, with optional sanitization.

    Args:
        smi (str): A SMILES string representing the molecule.
        sanitize (bool, optional): Whether to sanitize the molecule after parsing. Defaults to True.
            Sanitization applies all standard checks except hydrogen adjustment.

    Returns:
        Chem.Mol: An RDKit molecule object constructed from the SMILES string.
    """
    # includes_individually_mapped_hs = includes_individually_mapped_hydrogens(smi)
    # if includes_individually_mapped_hs:

    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if sanitize:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
        )
    Chem.AssignStereochemistry(mol)

    try:
        test_mol = Chem.Mol(mol)  # make a copy
        # Chem.Kekulize(test_mol)
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


def canonicalize(rxn_smi: str) -> Chem.Mol:
    """Converts a reaction SMILES string into its canonical form for both reactants and products.

    Args:
        rxn_smi (str): A reaction SMILES string formatted as "reactants>>products".

    Returns:
        str: The reaction SMILES with reactants and products converted to their canonical SMILES.
    """
    smi_reac, _, smi_prod = rxn_smi.split(">")
    mol_reac = make_mol(smi_reac)
    mol_prod = make_mol(smi_prod)
    smi_reac_new = Chem.MolToSmiles(mol_reac, canonical=True)
    smi_prod_new = Chem.MolToSmiles(mol_prod, canonical=True)
    return f"{smi_reac_new}>>{smi_prod_new}"


def map_reac_to_prod(mol_reac: Chem.Mol, mol_prod: Chem.Mol) -> Dict[int, int]:
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


def get_atom_map_num(mol: Chem.Mol, atom_idx: int) -> int:
    """Retrieves the atom mapping number for a specified atom in an RDKit molecule.

    Args:
        mol (Chem.Mol): The RDKit molecule.
        atom_idx (int): The index of the atom within the molecule.

    Returns:
        int: The atom mapping number of the specified atom.
    """
    num_atoms = mol.GetNumAtoms()
    assert (
        0 <= atom_idx < mol.GetNumAtoms()
    ), f"Error: Atom index {atom_idx} is out of bounds for mol with {num_atoms} atoms."

    atom = mol.GetAtomWithIdx(atom_idx)
    return atom.GetAtomMapNum()


def get_atom_map_num_of_mol(mol: Chem.Mol) -> List[int]:
    """Retrieves the atom map numbers of all atoms in an RDKit molecule.

    Args:
        mol (Chem.Mol): The RDKit molecule.

    Returns:
        List[int]: The list of atom map numbers of the molecule.
    """
    map_nums = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    return map_nums


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


def get_list_of_atom_map_numbers(smi: str) -> List[str]:
    """Extract all atom map numbers from a SMILES string in traversal order.

    Args:
        smi (str): A SMILES string potentially containing atom map numbers.

    Returns:
        List[int]: A list of atom map numbers as integers, extracted from left to right.
    """
    pattern = r":(\d+)"
    matches = re.findall(pattern, smi)
    ams = [int(m) for m in matches]
    return ams


def get_atom_indices_and_smarts(mol: Chem.Mol) -> List[Tuple[int, str]]:
    """Extract each atom's index and its corresponding SMARTS representation from the molecule.

    Args:
        mol (Chem.Mol): The molecule to process.

    Returns:
        List[Tuple[int, str]]: A list of tuples where each contains an atom's index and its SMARTS string.
    """
    num_atoms = mol.GetNumAtoms()
    atom_indices = [
        (i, mol.GetAtomWithIdx(i).GetSmarts(isomericSmiles=True, allHsExplicit=True))
        for i in range(num_atoms)
    ]
    return atom_indices


def get_bond_idx(mol: Chem.Mol, atom_idx1: int, atom_idx2: int) -> Optional[int]:
    """Get the bond index between two atoms in a molecule.

    Args:
        mol (Chem.Mol): The molecule containing the atoms.
        atom_idx1 (int): Index of the first atom.
        atom_idx2 (int): Index of the second atom.

    Returns:
        Optional[int]: The bond index if a bond exists between the two atoms; otherwise None.
    """
    bond = mol.GetBondBetweenAtoms(atom_idx1, atom_idx2)
    return bond.GetIdx() if bond is not None else None
