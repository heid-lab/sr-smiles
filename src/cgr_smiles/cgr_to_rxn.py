import re
from collections import defaultdict

from rdkit import Chem

from cgr_smiles.utils import (
    TokenType,
    _tokenize,
    common_elements_preserving_order,
    extract_chiral_tag_by_atom_map_num,
    flip_e_z_stereo,
    get_atom_map_adjacency_list_from_smiles,
    is_num_permutations_even,
)


def parse_bonds_from_smiles(smiles: str) -> dict[tuple[int, int], str]:
    """
    Parses SMILES to map bond atom-map pairs to their bond specifiers.

    This function traverses the SMILES token by token, identifying bonds by
    their connecting atom map numbers and extracting their explicit bond type.

    Args:
        smiles: SMILES string of a molecule.

    Returns:
        A dictionary mapping sorted `(atom_map_num_1, atom_map_num_2)` tuples
        to their bond specifier string.

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
                int(atom_map_match.group(1))
                if atom_map_match
                else (current_logical_idx + 1000)
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
                bond_val = (
                    next_bond_specifier if next_bond_specifier is not None else "-"
                )

                # Store the bond
                replace_dict_bonds[bond_map_num_pair] = bond_val

            current_logical_idx += 1
            anchor_logical_idx = (
                current_logical_idx - 1
            )  # This new atom becomes the anchor for next bond
            next_bond_specifier = None  # Clear any pending bond specifier

        elif tokentype == TokenType.BOND_TYPE or tokentype == TokenType.EZSTEREO:
            # These are standard bond types (-,=,#,: or E/Z stereo)
            next_bond_specifier = str(token_val)

        elif tokentype == TokenType.BRANCH_START:
            branches.append(anchor_logical_idx)  # Push current anchor onto stack

        elif tokentype == TokenType.BRANCH_END:
            if not branches:
                raise ValueError(
                    f"Unmatched ')' in SMILES string at index {token_original_idx}"
                )
            anchor_logical_idx = branches.pop()  # Pop anchor from stack
            next_bond_specifier = None  # Branch closure typically implies implicit single bond if no explicit one.

        elif tokentype == TokenType.RING_NUM:
            ring_num_val = str(token_val)  # Ring numbers can be int or string (for %XX)

            if (
                ring_num_val in ring_open_bonds
            ):  # Ring closure (we found a matching number)
                logical_idx_opener, bond_opener_specifier = ring_open_bonds[
                    ring_num_val
                ]

                # Bond is between the current atom (which is anchor_logical_idx) and the atom that opened the ring
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
                    else (
                        bond_opener_specifier
                        if bond_opener_specifier is not None
                        else "-"
                    )
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


def remove_bonds_by_atom_map_nums(
    mol: Chem.Mol, atom_map_pairs: list[tuple[int, int]]
) -> Chem.Mol:
    """
    Removes specified bonds from an RDKit molecule based on atom map numbers.

    Args:
        mol: The input RDKit molecule object.
        atom_map_pairs: A list of atom map tuples to be removed.

    Returns:
        A new RDKit molecule object with the specified bonds removed.
        If a bond corresponding to a given pair of atom map numbers does not exist,
        a warning is printed, and that pair is skipped.

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
            print(
                f"Warning: No bond found between atom map numbers {am1} and {am2}. Skipping removal."
            )

    for idx1, idx2 in bonds_to_remove_by_idx:
        emol.RemoveBond(idx1, idx2)

    final_mol = emol.GetMol()
    return final_mol


def update_chirality_tags(
    smiles: str, cgr_scaffold: str, chiral_center_map_nums: list[int]
) -> str:
    """
    Updates chirality tags in a SMILES string based on a CGR scaffold.

    Identifies chiral centers in the provided RDKit molecule (`mol`) by their atom
    map numbers. It then compares the neighborhood of these chiral centers in
    both the input SMILES (`smiles`) and a reference CGR scaffold (`cgr_scaffold`)
    to determine the chirality tags (@ or @@). If the chirality appears inverted
    between the SMILES and scaffold, the tag is flipped.

    Args:
        smiles: The input SMILES string of the molecule.
        cgr_scaffold: A reference CGR SMILES string containing correct chirality
                      information for comparison.
        chiral_center_map_nums: List of the atom map numbers of the chiral centers.

    Returns:
        A new SMILES string with updated or corrected chirality tags.

    """
    reac_adj = get_atom_map_adjacency_list_from_smiles(smiles)
    cgr_adj = get_atom_map_adjacency_list_from_smiles(cgr_scaffold)

    reac_tokens = [[tok_type, tok] for tok_type, _, tok in _tokenize(smiles)]
    for i, (tok_type, tok) in enumerate(reac_tokens):
        if tok_type == TokenType.ATOM:
            match = re.search(r":(\d+)", tok)
            map_num = int(match.group(1))
            if map_num in chiral_center_map_nums:
                reac_nbrs = reac_adj[map_num]
                cgr_nbrs = cgr_adj[map_num]
                reac_nbrs, cgr_nbrs = common_elements_preserving_order(
                    reac_nbrs, cgr_nbrs
                )

                current_tag = extract_chiral_tag_by_atom_map_num(cgr_scaffold, map_num)

                if is_num_permutations_even(reac_nbrs, cgr_nbrs):
                    chirality_tag = current_tag
                else:
                    if current_tag == "@":
                        chirality_tag = "@@"
                    elif current_tag == "@@":
                        chirality_tag = "@"

                replace_pattern = rf"(\[[A-Z][a-z]?)(@{{1,2}})?(:{map_num}\])"
                reac_tokens[i][1] = re.sub(
                    replace_pattern, rf"\1{chirality_tag}\3", reac_tokens[i][1]
                )

    return "".join([str(tok[1]) for tok in reac_tokens])


def find_cis_trans_stereo_bonds(
    bond_dict: dict[tuple[int, int], str],
) -> dict[tuple[int, int], dict[str, any]]:
    """Identifies cis/trans stereochemistry of double bonds from bond data.

    Parses a dictionary representing molecule bonds and their types. It identifies
    double bonds and determines their cis/trans stereochemistry by examining
    stereo bond specifiers ('/' or '\') on adjacent bonds connected to the
    double bond atoms.

    Args:
        bond_dict: A dictionary where keys are tuples of atom indices, and values
                   are strings indicating the bond type.

    Returns:
        A dictionary where keys are tuples of atom indices `(atom_idx_a, atom_idx_b)`
        representing the double bond, and values are dictionaries containing:
        - "stereo": An RDKit `Chem.BondStereo` enum value (STEREOCIS or STEREOTRANS).
        - "terminal_atoms": A tuple of the two atom indices `(neighbor_atom_c_idx, neighbor_atom_d_idx)`
                            that define the stereochemistry for each side of the double bond.
                            The dictionary contains entries for both `(a, b)` and `(b, a)`
                            for the same double bond.
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

        # stereo_arms_a = [(a, n, stereo_bonds[(a, n)]) for n, bt in neighbors[a] if (a, n) in stereo_bonds]
        # stereo_arms_b = [(b, n, stereo_bonds[(b, n)]) for n, bt in neighbors[b] if (b, n) in stereo_bonds]

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
            arm_a = stereo_arms_a[0]  # (atom_a, neighbor_c, slash_type_a)
            arm_b = stereo_arms_b[0]  # (atom_b, neighbor_d, slash_type_b)

            neighbor_c, slash_type_a = arm_a[1], flip_e_z_stereo(arm_a[2])
            neighbor_d, slash_type_b = arm_b[1], arm_b[2]

            if slash_type_a == slash_type_b:
                stereo = Chem.BondStereo.STEREOE  # Chem.BondStereo.STEREOTRANS
            else:
                stereo = Chem.BondStereo.STEREOZ  # Chem.BondStereo.STEREOCIS

            results[(atom_a, atom_b)] = {
                "stereo": stereo,
                "terminal_atoms": (neighbor_c, neighbor_d),
            }
            results[(atom_b, atom_a)] = {
                "stereo": stereo,
                "terminal_atoms": (neighbor_d, neighbor_c),
            }

    return results


def update_cis_trans_stereo_chem(mol: Chem.Mol, parsed_bonds: dict) -> Chem.Mol:
    """
    Updates cis/trans stereochemistry for double bonds in a molecule,
    using pre-parsed bond and stereochemistry data.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule with atom map numbers.
        parsed_bonds (dict): Dictionary with bond keys (am1, am2) and values including
                             'terminal_atoms' (tuple of atom map nums) and 'stereo' (RDKit stereo enum).

    Returns:
        rdkit.Chem.Mol: Molecule with updated stereochemistry.
    """
    b = find_cis_trans_stereo_bonds(parsed_bonds)

    # assigning stereochem manually only works for individual molecules, i.e. smiles
    # does not include ".". Therefore, iterate over the fragments of the molecule.
    frags = Chem.GetMolFrags(mol, asMols=True)

    for frag in frags:
        map_num2idx = {a.GetAtomMapNum(): a.GetIdx() for a in frag.GetAtoms()}
        for bond in frag.GetBonds():
            am1 = bond.GetBeginAtom().GetAtomMapNum()
            am2 = bond.GetEndAtom().GetAtomMapNum()

            if (am1, am2) in b.keys():
                nbr1, nbr2 = b[(am1, am2)]["terminal_atoms"]
                nbr1, nbr2 = map_num2idx[nbr1], map_num2idx[nbr2]
                bond.SetStereoAtoms(nbr1, nbr2)

                stereo = b[(am1, am2)]["stereo"]
                bond.SetStereo(stereo)

            elif (am2, am1) in b.keys():
                nbr1, nbr2 = b[(am2, am1)]["terminal_atoms"]
                nbr1, nbr2 = map_num2idx[nbr1], map_num2idx[nbr2]
                bond.SetStereoAtoms(nbr2, nbr1)

                stereo = b[(am2, am1)]["stereo"]
                bond.SetStereo(stereo)

        Chem.AssignStereochemistry(frag, force=True)

    smiles = [Chem.MolToSmiles(f, canonical=False) for f in frags]
    m = Chem.MolFromSmiles(".".join(smiles), sanitize=False)
    return m


def get_reac_prod_scaffold_smiles_from_cgr(cgr_smiles: str) -> tuple[str, str]:
    """
    Extracts the reactant and product scaffold SMILES from a CGR SMILES string.

    The CGR SMILES encodes atom-level differences between reactants and products using
    substitution patterns in the form `{reactant|product}`.
    This function decodes those patterns by replacing each `{...|...}` block with the
    appropriate fragment in two parallel SMILES strings: one for the reactant, one for the product.

    Args:
        cgr_smiles (str): A CGR SMILES string containing substitution patterns.

    Returns:
        tuple[str, str]: A tuple containing the reactant SMILES and product SMILES
                         with all substitution patterns resolved.
    """
    reac_smi = cgr_smiles
    prod_smi = cgr_smiles

    cgr_pattern = r"\{([^{|}]*)\|([^{|}]*)\}"

    while "{" in reac_smi:
        match = re.search(cgr_pattern, reac_smi)
        if match is None:
            break

        full_match = match.group(0)
        reac_fragment = match.group(1)
        prod_fragment = match.group(2)

        # replace the first match occurrence
        reac_smi = reac_smi.replace(full_match, reac_fragment, 1)
        prod_smi = prod_smi.replace(full_match, prod_fragment, 1)

    return reac_smi, prod_smi


def get_chiral_center_map_nums(mol: Chem.Mol) -> list[int]:
    """Returns the atom map numbers of chiral centers in an RDKit molecule.

    Args:
        mol: The input RDKit molecule object.

    Returns:
        A list of integer atom map numbers corresponding to the chiral centers
        found in the molecule.
    """
    atom_map_nums = []
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() in (
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        ):
            atom_map_nums.append(atom.GetAtomMapNum())
    return atom_map_nums


def cgrsmiles_to_rxnsmiles(cgr_smiles: str) -> str:
    # TODO: start with a validity check, especially that each substitution pattern follows `{...|...}`.

    # extract reac and prod smiles scaffold from cgr smiles
    reac_smi1, prod_smi1 = get_reac_prod_scaffold_smiles_from_cgr(cgr_smiles)

    cgr_reac_scaffold = reac_smi1.replace("~", "")
    cgr_prod_scaffold = prod_smi1.replace("~", "")

    # extract the bonds, parsed from the smiles
    reac_parsed_bonds = parse_bonds_from_smiles(reac_smi1)
    prod_parsed_bonds = parse_bonds_from_smiles(prod_smi1)

    # extract the bond information of the unspecified bonds, as those are the ones we want to delete from the molecule.
    reac_map_nums_unspecified_bonds = [
        key for key, val in reac_parsed_bonds.items() if val == "~"
    ]
    prod_map_nums_unspecified_bonds = [
        key for key, val in prod_parsed_bonds.items() if val == "~"
    ]

    # then create the molecules from the smiles and manually remove the bonds that are labelled as unspecified
    reac_mol = Chem.MolFromSmiles(reac_smi1.replace("~", ""), sanitize=False)
    prod_mol = Chem.MolFromSmiles(prod_smi1.replace("~", ""), sanitize=False)
    Chem.SanitizeMol(
        prod_mol, Chem.SanitizeFlags.SANITIZE_ADJUSTHS
    )  # TODO also for reac
    Chem.SanitizeMol(
        reac_mol, Chem.SanitizeFlags.SANITIZE_ADJUSTHS
    )  # TODO also for reac
    reac_mol = remove_bonds_by_atom_map_nums(
        reac_mol, reac_map_nums_unspecified_bonds
    )  # smiles based patch needed while bug not fixed on rdkits end.
    prod_mol = remove_bonds_by_atom_map_nums(prod_mol, prod_map_nums_unspecified_bonds)

    # update stereochem bonds / and \ (because apparently they get messed up along the way. this might not e neccessary anymore if previous bug gets fixed.)
    reac_mol = update_cis_trans_stereo_chem(reac_mol, reac_parsed_bonds)
    prod_mol = update_cis_trans_stereo_chem(prod_mol, prod_parsed_bonds)

    # then get the resulting smiles again and check if the chirality tags are correct.
    reac_smi3 = Chem.MolToSmiles(reac_mol, canonical=False)
    prod_smi3 = Chem.MolToSmiles(prod_mol, canonical=False)

    reac_map_num_of_chiral_centers = get_chiral_center_map_nums(reac_mol)
    prod_map_num_of_chiral_centers = get_chiral_center_map_nums(prod_mol)

    reac_smi4 = update_chirality_tags(
        reac_smi3, cgr_reac_scaffold, reac_map_num_of_chiral_centers
    )
    prod_smi4 = update_chirality_tags(
        prod_smi3, cgr_prod_scaffold, prod_map_num_of_chiral_centers
    )

    return f"{reac_smi4}>>{prod_smi4}"
