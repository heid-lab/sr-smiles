import enum
import itertools
import re
from typing import Dict, Iterator, List, Match, Optional, Tuple, Union

from rdkit import Chem

from sr_smiles.chem_utils.mol_utils import make_mol


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


def remove_redundant_brackets_and_hydrogens(smi_sr: str) -> str:
    """Remove redundant square brackets and explicit hydrogens from an SR-SMILES string.

    This function cleans an SR-SMILES string by removing brackets that contain only atoms
    from the `ORGANIC_SUBSET` and by eliminating explicit hydrogen atoms where possible,
    while preserving charges, isotopes, and other annotations.

    Args:
        smi_sr (str): An SR-SMILES string potentially containing redundant brackets or hydrogens.

    Returns:
        str: The cleaned SR-SMILES string with redundant brackets and hydrogens removed.
    """
    # Special explicit-H patterns first
    specials = {
        "CH4": "C",
        "CH3": "C",
        "CH2": "C",
        "CH": "C",
        "OH2": "O",
        "OH": "O",
        "NH3": "N",
        "NH2": "N",
        "NH": "N",
        "SH2": "S",
        "SH": "S",
        "PH3": "P",
        "PH2": "P",
        "PH": "P",
        "cH": "c",
    }

    def _replace_bracketed(match: re.Match) -> str:
        atom = match.group(1)
        if atom in specials:
            return specials[atom]
        if atom in ORGANIC_SUBSET:
            return atom
        return f"[{atom}]"

    # replace bracketed atoms when they can be safely simplified
    smi_sr = re.sub(r"\[([^\]]+)\]", _replace_bracketed, smi_sr)

    # collapse identical reac|prod pairs, e.g. {X|X} → X
    smi_sr = re.sub(r"\{([A-Za-z0-9@+\-]+)\|\1\}", r"\1", smi_sr)

    return smi_sr


def remove_redundant_brackets(smi_sr: str) -> str:
    """Removes redundant square brackets from an SR-SMILES string.

    Brackets are removed only if they enclose atoms from the `ORGANIC_SUBSET`. Brackets
    that include explicit hydrogens, charges, isotopes, or other annotations are preserved.

    Args:
        smi_sr (str): An SR-SMILES string potentially containing redundant brackets.

    Returns:
        str: SR-SMILES string with redundant brackets removed.
    """

    def _replace_bracketed(match: Match[str]) -> str:
        atom = match.group(1)
        if atom in ORGANIC_SUBSET:
            return atom
        return f"[{atom}]"

    # replace [X] with X if X is in the organic subset
    smi_sr = re.sub(r"\[([^\]]+)\]", _replace_bracketed, smi_sr)

    # collapse identical reac|prod pairs, e.g. {X|X} → X
    smi_sr = re.sub(r"\{([A-Za-z0-9@+\-]+)\|\1\}", r"\1", smi_sr)

    return smi_sr


def remove_aromatic_bonds(smiles: str) -> str:
    """Remove aromatic bond symbols (':') outside brackets and SR bond blocks.

    Handles plain molecular SMILES, reaction SMILES, and SR-SMILES strings.
    The colon (`:`) denotes aromatic bonds in SMILES but also appears
    *inside* brackets (for atom maps, e.g. `[C:1]`) and inside curly braces
    for SR bond descriptors (e.g. `{:|=}`). This function removes only those
    colons that represent aromatic bonds **between atoms**, not those used
    within `[...]` or `{...}` groups.

    Args:
        smiles (str): A SMILES, reaction SMILES, or SR-SMILES string that may
            contain colon (':') characters.

    Returns:
        str: The same SMILES string with aromatic bond colons removed, while
        preserving colons inside atom brackets and SR bond descriptors.

    Example:
        >>> remove_aromatic_bonds("c1:c:c:c:c:c:1")
        'c1ccccc1'
        >>> remove_aromatic_bonds("[c:1]:[c:2]>>[C:1][C:2]")
        '[c:1][c:2]>>[C:1][C:2]'
        >>> remove_aromatic_bonds("c:c{:|=}{c|C}")
        'cc{:|=}{c|C}'
    """
    result = []
    in_brackets = False  # inside [ ... ]
    in_sr_brackets = False  # inside { ... }

    for char in smiles:
        if char == "[":
            in_brackets = True
        elif char == "]":
            in_brackets = False
        elif char == "{":
            in_sr_brackets = True
        elif char == "}":
            in_sr_brackets = False

        # Skip ':' when it's an aromatic bond (outside both brackets & braces)
        if char == ":" and not in_brackets and not in_sr_brackets:
            continue

        result.append(char)

    return "".join(result)


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


def parse_bonds_in_order_from_smiles(smiles: str) -> Dict[Tuple[int, int], str]:
    """Parses SMILES to map bond atom map pairs to their bond specifiers.

    This function traverses the SMILES token by token, identifying bonds by
    their connecting atom map numbers and extracting their explicit bond type.

    Args:
        smiles (str): SMILES string of a molecule.

    Returns:
        Dict[Tuple[int, int], str]: A dict mapping sorted `(atom_map_num_1, atom_map_num_2)`
            Tuples to their bond specifier string.

    Raises:
        ValueError: If the SR-SMILES string has malformed syntax.
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
                replace_dict_bonds[bond_map_num_pair] = bond_val

            anchor_logical_idx = current_logical_idx
            current_logical_idx += 1
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


def get_fragment_permutations(smiles: str, max_permutations: int = None) -> List[List[int]]:
    """Generates all index permutations for disconnected fragments in a SMILES string.

    The SMILES string is split on '.' (dot), which separates disconnected
    fragments. The function returns all possible index orderings of these fragments.

    Args:
        smiles (str): Input SMILES string, potentially containing multiple
            disconnected components, e.g. "CC.O.N".
        max_permutations (int, optional): Maximum number of permutations to
            generate. If None, all possible permutations are returned.

    Returns:
        List[List[int]]: A list of index permutations, where each sublist
        defines a possible ordering of fragment indices.

    Example:
        >>> get_fragment_permutations("A.B.C")
        [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]

        >>> get_fragment_permutations("A.B.C", max_permutations=3)
        [[0, 1, 2], [0, 2, 1], [1, 0, 2]]
    """
    fragments = smiles.split(".")
    n_frag = len(fragments)

    if n_frag <= 1:
        return [[0]]

    result = []
    for i, perm in enumerate(itertools.permutations(range(n_frag))):
        if max_permutations is not None and i >= max_permutations:
            break
        result.append(list(perm))

    return result


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


def is_kekule(atom_mapped_rxn_smi: str) -> bool:
    """Check if a iven RXN SMILES is kekulized.

    Return True if all bracketed atoms in a mapped reaction SMILES string
    use Kekulé (uppercase) element symbols, i.e. no lowercase aromatic
    symbols like [c], [n], [o], etc. are present. Returns False otherwise.
    """
    for match in re.finditer(r"\[([^\]]+)\]", atom_mapped_rxn_smi):
        content = match.group(1)
        m = re.search(r"[A-Za-z]", content)
        if m:
            first = m.group(0)
            if not first.isupper():
                return False
    return True


# def get_list_of_atom_map_numbers_from_sr_smiles(sr_smi: str) -> List[str]:
#     """Extract all atom map numbers from a SMILES string in traversal order.

#     Args:
#         sr_smi (str): A SMILES string potentially containing atom map numbers.

#     Returns:
#         List[int]: A list of atom map numbers as integers, extracted from left to right.
#     """
#     result = []

#     # Pattern for {...} groups
#     group_pattern = re.compile(r"\{([^}]*)\}")
#     # Pattern to capture :num] map indices
#     mapnum_pattern = re.compile(r":(\d+)\]")

#     pos = 0
#     for m in group_pattern.finditer(sr_smi):
#         # process text before the group
#         before = sr_smi[pos : m.start()]
#         result.extend(int(x) for x in mapnum_pattern.findall(before))

#         # process inside the group -> deduplicated numbers
#         inside = mapnum_pattern.findall(m.group(1))
#         seen = set()
#         for n in inside:
#             if n not in seen:
#                 seen.add(n)
#                 result.append(int(n))

#         pos = m.end()

#     # process the tail after the last group
#     tail = sr_smi[pos:]
#     result.extend(int(x) for x in mapnum_pattern.findall(tail))

#     return result


# def includes_individually_mapped_hydrogens(smiles: str) -> bool:
#     """Check if a SMILES string includes individually mapped hydrogens, e.g., [H:2].

#     Args:
#         smiles (str): The SMILES string to check.

#     Returns:
#         bool: True if mapped hydrogens are found, False otherwise.
#     """
#     # Regular expression to match [H:<number>], optionally with isotope or charge
#     pattern = r"\[H(?::\d+)\]"

#     res = bool(re.search(pattern, smiles))
#     if not res:
#         # check if any hydrogens at all
#         res_h = "H" in smiles

#         return not res_h

#     else:
#         return res


# def remove_redundant_square_brackets(rxn_smiles: str) -> str:
#     """Removes only the redundant square brackets in a reaction SMILES string.

#     Redundant square brackets are those that enclose uncharged, non-chiral atoms from
#     the organic subset. For example, [C] becomes C, while brackets are retained for
#     cases like [C-], [C@H], and [H], where additional chemical information is present.

#     Args:
#         rxn_smiles (str): A reaction SMILES string possibly containing redundant brackets.

#     Returns:
#         str: Reaction SMILES string with unnecessary square brackets removed.
#     """
#     escaped_subset = [re.escape(elem) for elem in sorted(ORGANIC_SUBSET, key=len, reverse=True)]
#     pattern = rf"\[({'|'.join(escaped_subset)})\]"

#     def replacer(match):
#         return match.group(1)

#     return re.sub(pattern, replacer, rxn_smiles)
