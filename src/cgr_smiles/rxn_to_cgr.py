import re

from rdkit import Chem

from cgr_smiles.utils import (
    ORGANIC_SUBSET,
    TokenType,
    _tokenize,
    flip_e_z_stereo,
    make_mol,
    map_reac_to_prod,
    remove_atom_mapping,
    update_all_atom_stereo,
)


def remove_redundant_brackets_and_hydrogens(cgr: str) -> str:
    """
    Clean a CGR-SMILES string by removing redundant square brackets and explicit hydrogens.
    """

    # Special explicit-H patterns first
    specials = {
        # Carbon
        "CH4": "C",
        "CH3": "C",
        "CH2": "C",
        "CH": "C",
        # Oxygen
        "OH": "O",
        # Nitrogen
        "NH2": "N",
        "NH": "N",
        "nH": "n",
        # Sulfur
        "SH": "S",
        "sH": "s",
        # Phosphorus
        "PH2": "P",
        "PH": "P",
        # Aromatic carbons
        "cH": "c",
    }

    def replace_bracketed(match):
        atom = match.group(1)
        # Special cases like CH3, cH, OH
        if atom in specials:
            return specials[atom]
        # Pure organic subset atoms (no isotope/charge/explicit H)
        if atom in ORGANIC_SUBSET:
            return atom
        # Otherwise keep brackets
        return f"[{atom}]"

    # Replace all [..] with cleaned version
    cgr = re.sub(r"\[([^\]]+)\]", replace_bracketed, cgr)

    # Collapse {X|X} → X
    cgr = re.sub(r"\{([A-Za-z0-9@+\-]+)\|\1\}", r"\1", cgr)

    return cgr


def remove_redundant_brackets(cgr: str) -> str:
    """
    Remove only redundant square brackets in a CGR-SMILES string.
    Keeps explicit H, charges, isotopes, etc.
    """

    def replace_bracketed(match):
        atom = match.group(1)
        if atom in ORGANIC_SUBSET:
            return atom
        return f"[{atom}]"

    # Replace [X] with X if X is in organic subset
    cgr = re.sub(r"\[([^\]]+)\]", replace_bracketed, cgr)

    return cgr


# TODO: do some checks according to our assumptions (atom mapping, balanced etc)
# TODO: also make version that has the {..|..} for all atoms and bonds (not only those changing)
# TODO: standardize atom order depending on unmapped reactants, then add mappings again. Make the cgr molecule from this canonicalized reactant molecule to get maximum reproducibility
# TODO: Make this also work for unbalanced rxns
# DONE: als make unmapped version of the cgrsmiles
def rxnsmiles_to_cgrsmiles(
    rxn_smi: str,
    keep_atom_mapping: bool = False,
    remove_brackets: bool = False,
    remove_hydrogens: bool = False,
) -> str:
    """
    Converts a reaction SMILES string into a Condensed Graph of Reaction (CGR) SMILES.

    A CGR SMILES encodes the transformation between reactant and product molecules
    as a single, compact string representation, where atoms and bonds are annotated to
    show differences in atom types, bond orders, and stereochemistry.

    Args:
        rxn_smi (str): A reaction SMILES string in the format "reactant>>product".
        keep_atom_mapping: If True, atom map numbers will be removed in the output
                           CGR SMILES. Otherwise they will be retained (default).
    Returns:
        str: A CGR SMILES string representing the reaction as a single molecule
        with annotations of changes using `{reac|prod}` syntax.

    Notes:
        - Requires all atoms in the SMILES to be atom-mapped.
        - Requires balanced reactions.

    Example:
        >>> rxn_smiles = "[C:1]([H:3])([H:4])([H:5])[H:6].[Cl:2][H:7]>>[C:1]([H:3])([H:4])([H:5])[Cl:2].[H:6][H:7]"
        >>> rxnsmiles_to_cgrsmiles(rxn_smiles)
        "[C:1]1([H:3])([H:4])([H:5]){-|~}[H:6]{~|-}[H:7]{-|~}[Cl:2]{~|-}1"

        # In the resulting CGR SMILES, the `{reac|prod}` notation encodes how atoms and bonds
        # change from reactants to products. For example, '[H:6]{~|-}[H:7]' means that while there
        # was no bond between these two hydrogen atoms in the reactants, a single bond has been
        # formed between them in the product molecule.
    """
    # TODO: check if rxn_smiles is atom mapped, if not, add mapping.
    # TODO: maybe let this function make the assumption of balanced, fully mapped reactions.
    # Handling of the preparation, shall the wrapper do.
    # fully_atom_mapped = is_fully_atom_mapped(rxn_smi)
    # if not fully_atom_mapped:
    #     print(f"WARNING: given reaction smiles is not fully atom mapped: {rxn_smi}")
    #     rxn_smi = add_atom_mapping(rxn_smi)

    smi_reac, _, smi_prod = rxn_smi.split(">")
    mol_reac, mol_prod = make_mol(smi_reac), make_mol(smi_prod)

    ri2pi = map_reac_to_prod(mol_reac, mol_prod)

    # add missing bonds to the cgr mol
    mol_cgr = Chem.EditableMol(mol_reac)
    n_atoms = mol_reac.GetNumAtoms()
    unspecified_bonds = []
    for idx1 in range(n_atoms):
        for idx2 in range(idx1 + 1, n_atoms):
            bond_reac = mol_reac.GetBondBetweenAtoms(idx1, idx2)
            bond_prod = mol_prod.GetBondBetweenAtoms(ri2pi[idx1], ri2pi[idx2])
            if bond_reac is None and bond_prod is not None:
                mol_cgr.AddBond(idx1, idx2, order=Chem.rdchem.BondType.UNSPECIFIED)
                unspecified_bonds.append((idx1, idx2))

    mol_cgr = mol_cgr.GetMol()
    # bonds1 = [(b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum(), b.GetStereo(), ) for b in mol_cgr.GetBonds()]

    smi_cgr = Chem.MolToSmiles(mol_cgr, canonical=False)
    # mol_cgr = Chem.MolFromSmiles(smi_cgr, sanitize=False)
    mol_cgr = make_mol(smi_cgr, sanitize=False)
    # bonds2 = [(b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum(), b.GetStereo(), ) for b in mol_cgr.GetBonds()]

    # d_cgr = get_atom_map_adjacency_list_from_mol(mol_cgr)

    # reorder reac and prod molecule so we get the relative stereochemistry tags right:
    # TODO: by doing the reordering, we basically canonicalize and make it a non-injective maping from rxn_smi to cgr_smi
    # TODO: maybe instead just align the mapping of the product with the one in the reactant
    # bonds_prod_1 = [(b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum(), b.GetStereo(), ) for b in mol_prod.GetBonds()]
    # atoms_prod_1 = [(a.GetAtomMapNum(), a.GetChiralTag(), a.GetSmarts(isomericSmiles=True)) for a in mol_prod.GetAtoms()]
    prod_map_to_id = dict(
        [(atom.GetAtomMapNum(), atom.GetIdx()) for atom in mol_prod.GetAtoms()]
    )
    prod_reorder = [prod_map_to_id[a.GetAtomMapNum()] for a in mol_cgr.GetAtoms()]
    mol_prod = Chem.RenumberAtoms(mol_prod, prod_reorder)
    smi_prod = Chem.MolToSmiles(mol_prod, canonical=False)
    # bonds_prod_2 = [(b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum(), b.GetStereo(), ) for b in mol_prod.GetBonds()]
    # atoms_prod_2 = [(a.GetAtomMapNum(), a.GetChiralTag(), a.GetSmarts(isomericSmiles=True)) for a in mol_prod.GetAtoms()]

    # bonds_reac_1 = [(b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum(), b.GetStereo(), ) for b in mol_reac.GetBonds()]
    # atoms_reac_1 = [(a.GetAtomMapNum(), a.GetChiralTag(), a.GetSmarts(isomericSmiles=True)) for a in mol_reac.GetAtoms()]
    reac_map_to_id = dict(
        [(atom.GetAtomMapNum(), atom.GetIdx()) for atom in mol_reac.GetAtoms()]
    )
    reac_reorder = [reac_map_to_id[a.GetAtomMapNum()] for a in mol_cgr.GetAtoms()]
    mol_reac = Chem.RenumberAtoms(mol_reac, reac_reorder)
    # bonds_reac_2 = [(b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum(), b.GetStereo(), ) for b in mol_reac.GetBonds()]
    # atoms_reac_2 = [(a.GetAtomMapNum(), a.GetChiralTag(), a.GetSmarts(isomericSmiles=True)) for a in mol_reac.GetAtoms()]
    smi_reac = Chem.MolToSmiles(mol_reac, canonical=False)

    update_all_atom_stereo(mol_reac, smi_reac, smi_cgr)
    update_all_atom_stereo(mol_prod, smi_prod, smi_cgr)

    replace_dict_atoms = {}
    replace_dict_bonds = {}
    for i1 in range(n_atoms):
        atom_reac = mol_reac.GetAtomWithIdx(i1)
        atom_cgr = mol_cgr.GetAtomWithIdx(i1)
        atom_prod = mol_prod.GetAtomWithIdx(i1)

        reac_smarts = atom_reac.GetSmarts(isomericSmiles=True)
        prod_smarts = atom_prod.GetSmarts(isomericSmiles=True)

        if reac_smarts != prod_smarts:
            replace_dict_atoms[atom_cgr.GetAtomMapNum()] = (
                f"{{{reac_smarts}|{prod_smarts}}}"
            )
        else:
            replace_dict_atoms[atom_cgr.GetAtomMapNum()] = reac_smarts

        for i2 in range(i1 + 1, n_atoms):
            atom2_cgr = mol_cgr.GetAtomWithIdx(i2)
            map_num_1 = atom_cgr.GetAtomMapNum()
            map_num_2 = atom2_cgr.GetAtomMapNum()
            bond_reac = mol_reac.GetBondBetweenAtoms(i1, i2)
            bond_prod = mol_prod.GetBondBetweenAtoms(i1, i2)

            reac_begin, reac_end = map_num_1, map_num_2
            smarts_bond_reac = "~"
            if bond_reac is not None:
                smarts_bond_reac = bond_reac.GetSmarts(allBondsExplicit=True)
                reac_begin, reac_end = (
                    bond_reac.GetBeginAtom().GetAtomMapNum(),
                    bond_reac.GetEndAtom().GetAtomMapNum(),
                )

            prod_begin, prod_end = map_num_1, map_num_2
            smarts_bond_prod = "~"
            if bond_prod is not None:
                smarts_bond_prod = bond_prod.GetSmarts(allBondsExplicit=True)
                prod_begin, prod_end = (
                    bond_prod.GetBeginAtom().GetAtomMapNum(),
                    bond_prod.GetEndAtom().GetAtomMapNum(),
                )

                if (
                    reac_begin == prod_end and reac_end == prod_begin
                ):  # TODO: maybe actually compare to cgr begin end atom, not reac vs. prod.
                    # need to flip!
                    smarts_bond_prod = flip_e_z_stereo(smarts_bond_prod)

            if bond_reac is None and bond_prod is None:
                continue

            if smarts_bond_reac != smarts_bond_prod:
                val = f"{{{smarts_bond_reac}|{smarts_bond_prod}}}"
            else:
                val = smarts_bond_reac if smarts_bond_reac != "-" else ""

            replace_dict_bonds[(reac_begin, reac_end)] = val
            replace_dict_bonds[(reac_end, reac_begin)] = flip_e_z_stereo(val)

    # change bonds
    smiles = ""
    anchor = None
    idx = 0
    next_bond = None
    branches = []
    ring_nums = {}
    i2m = {}
    for tokentype, token_idx, token in _tokenize(smi_cgr):
        if tokentype == TokenType.ATOM:
            i2m[idx] = int(token[:-1].split(":")[1])
            if anchor is not None:
                if next_bond is None:
                    next_bond = ""
                smiles += replace_dict_bonds.get((i2m[anchor], i2m[idx]), next_bond)
                next_bond = None
            smiles += token
            anchor = idx
            idx += 1
        elif tokentype == TokenType.BRANCH_START:
            branches.append(anchor)
            smiles += token
        elif tokentype == TokenType.BRANCH_END:
            anchor = branches.pop()
            smiles += token
        elif tokentype == TokenType.BOND_TYPE:
            next_bond = token
        elif tokentype == TokenType.EZSTEREO:
            next_bond = token
        elif tokentype == TokenType.RING_NUM:
            if token in ring_nums:
                jdx, order = ring_nums[token]
                if next_bond is None and order is None:
                    next_bond = ""
                elif order is None:
                    next_bond = next_bond
                elif next_bond is None:
                    next_bond = order
                smiles += replace_dict_bonds.get((i2m[idx - 1], i2m[jdx]), next_bond)
                smiles += str(token)
                next_bond = None
                del ring_nums[token]

            else:
                ring_nums[token] = (idx - 1, next_bond)
                next_bond = None
                smiles += str(token)

    smi_cgr = smiles

    # change atoms
    for k in replace_dict_atoms.keys():
        smi_cgr = smi_cgr.replace(
            re.findall(rf"\[[^):]*:{k}\]", smi_cgr)[0], replace_dict_atoms[k]
        )

    if not keep_atom_mapping:
        smi_cgr = remove_atom_mapping(smi_cgr)

    if remove_brackets and remove_hydrogens:
        smi_cgr = remove_redundant_brackets_and_hydrogens(smi_cgr)

    elif remove_brackets:
        smi_cgr = remove_redundant_brackets(smi_cgr)

    return smi_cgr
