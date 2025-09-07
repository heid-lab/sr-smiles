import re
from typing import List, Optional, Union

import pandas as pd
from rdkit import Chem

from cgr_smiles.logger import logger
from cgr_smiles.reaction_balancing import balance_reaction, is_rxn_balanced
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
    """Remove redundant square brackets and explicit hydrogens from a CGR-SMILES string.

    This function cleans a CGR-SMILES string by removing brackets that contain only atoms
    from the ORGANIC_SUBSET and by eliminating explicit hydrogen atoms where possible,
    while preserving charges, isotopes, and other annotations.

    Args:
        cgr (str): A CGR-SMILES string potentially containing redundant brackets and explicit hydrogens.

    Returns:
        str: The cleaned CGR-SMILES string with redundant brackets and hydrogens removed.
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
        "nH": "n",
        "SH2": "S",
        "SH": "S",
        "sH": "s",
        "PH3": "P",
        "PH2": "P",
        "PH": "P",
        "cH": "c",
    }

    def replace_bracketed(match):
        atom = match.group(1)
        if atom in specials:
            return specials[atom]
        if atom in ORGANIC_SUBSET:
            return atom
        return f"[{atom}]"

    # replace all [..] with cleaned version
    cgr = re.sub(r"\[([^\]]+)\]", replace_bracketed, cgr)

    # collapse {X|X} → X
    cgr = re.sub(r"\{([A-Za-z0-9@+\-]+)\|\1\}", r"\1", cgr)

    return cgr


def remove_redundant_brackets(cgr: str) -> str:
    """Remove redundant square brackets from a CGR-SMILES string.

    Only brackets containing atoms from the ORGANIC_SUBSET are removed.
    Brackets that include explicit hydrogens, charges, isotopes, or other annotations are preserved.

    Args:
        cgr (str): A CGR-SMILES string potentially containing redundant brackets.

    Returns:
        str: The CGR-SMILES string with redundant brackets removed.
    """

    def replace_bracketed(match):
        atom = match.group(1)
        if atom in ORGANIC_SUBSET:
            return atom
        return f"[{atom}]"

    # Replace [X] with X if X is in organic subset
    cgr = re.sub(r"\[([^\]]+)\]", replace_bracketed, cgr)

    return cgr


class RxnToCgr:
    """Transform reaction SMILES into CGR SMILES.

    This class provides a callable interface to convert reaction SMILES into
    CGR SMILES. It supports single strings, lists of strings, pandas Series,
    and pandas DataFrames.

    Attributes:
        keep_atom_mapping (bool): Whether to preserve atom mapping in the output.
        remove_brackets (bool): Whether to remove brackets from the SMILES.
        remove_hydrogens (bool): Whether to remove explicit hydrogens.
        balance_rxn (bool): Whether to balance the given reaction.
        rxn_col (Optional[str]): Column name in a DataFrame containing reaction SMILES.

    Examples:
        Transform a pandas DataFrame of reactions into CGR SMILES:

        >>> import pandas as pd
        >>> df = pd.read_csv("path/to/file.csv")
        >>> transform = RxnToCgr(rxn_col="rxn_smiles")
        >>> df["cgr_smiles"] = transform(df)
    """

    def __init__(
        self,
        keep_atom_mapping: bool = False,
        remove_brackets: bool = False,
        remove_hydrogens: bool = False,
        balance_rxn: bool = False,
        rxn_col: Optional[str] = None,
        use_aromaticity: bool = True,
    ) -> None:
        """Initialize the transformation object.

        Args:
            keep_atom_mapping (bool, optional): If True, preserve atom mapping
                in the output. Defaults to False.
            remove_brackets (bool, optional): If True, remove brackets from
                the SMILES. Defaults to False.
            remove_hydrogens (bool, optional): If True, remove explicit
                hydrogens. Defaults to False.
            balance_rxn (bool, optional): If True, attempts to balance the reaction
                before generating the CGR. Defaults to False.
            rxn_col (str, optional): Column name in a DataFrame containing
                reaction SMILES. Required if passing a DataFrame. Defaults to None.
            use_aromaticity (bool, optional): If True, RDKit aromaticity perception is applied
                during sanitization, and aromatic atoms will be written in lowercase (e.g. "c").
                If False, aromaticity perception is skipped, and atoms will be written in
                uppercase (e.g. "C"). Defaults to True.
        """
        self.keep_atom_mapping = keep_atom_mapping
        self.remove_brackets = remove_brackets
        self.remove_hydrogens = remove_hydrogens
        self.balance_rxn = balance_rxn
        self.rxn_col = rxn_col
        self.use_aromaticity = use_aromaticity

    def __call__(
        self, data: Union[str, List[str], pd.Series, pd.DataFrame]
    ) -> Union[str, List[str], pd.Series, pd.DataFrame]:
        """Apply the transformation to reaction SMILES.

        Args:
            data (Union[str, List[str], pd.Series, pd.DataFrame]): Input data
                containing reaction SMILES. Can be a single string, a list of
                strings, a pandas Series, or a pandas DataFrame.

        Returns:
            Union[str, List[str], pd.Series, pd.DataFrame]: Transformed CGR SMILES
            in the same structure as the input.

        Raises:
            ValueError: If a DataFrame is provided but `self.rxn_col` is not set.
            TypeError: If the input type is not supported.
        """
        if isinstance(data, str):
            return rxn_to_cgr(
                data,
                keep_atom_mapping=self.keep_atom_mapping,
                remove_brackets=self.remove_brackets,
                remove_hydrogens=self.remove_hydrogens,
                balance_rxn=self.balance_rxn,
                use_aromaticity=self.use_aromaticity,
            )

        elif isinstance(data, list):
            result = [self(d) for d in data]
            n_empty = sum(1 for item in result if item == "")
            if len(result) > 0:
                logger.warning(
                    f"Failed for {n_empty} out of {len(result)} samples ({n_empty / len(result) * 100} %). "
                )
            return result

        elif isinstance(data, pd.Series):
            return data.apply(self)

        elif isinstance(data, pd.DataFrame):
            if self.rxn_col is None:
                raise ValueError(
                    f"A pandas DataFrame was provided, but `self.rxn_col` is not set.\n"
                    f"Available columns are: {list(data.columns)}\n"
                    "Please specify the column name containing the reactions by setting "
                    "`rxn_col` at time of initialization."
                )
            return data[self.rxn_col].apply(self)

        else:
            raise TypeError("Input must be str, list, pandas Series, or DataFrame.")


# TODO: do some checks according to our assumptions (atom mapping, balanced etc)
# TODO: also make version that has the {..|..} for all atoms and bonds (not only those changing)
# TODO: standardize atom order depending on unmapped reactants, then add mappings again. Make the cgr molecule from this canonicalized reactant molecule to get maximum reproducibility  # noqa: E501
# TODO: Make this also work for unbalanced rxns
# DONE: als make unmapped version of the cgrsmiles
def rxn_to_cgr(
    rxn_smi: str,
    keep_atom_mapping: bool = False,
    remove_brackets: bool = False,
    remove_hydrogens: bool = False,
    balance_rxn: bool = False,
    use_aromaticity: bool = True,
) -> str:
    """Converts a reaction SMILES string into a Condensed Graph of Reaction (CGR) SMILES.

    A CGR SMILES encodes the transformation between reactant and product molecules
    as a single, compact string representation, where atoms and bonds are annotated to
    show differences in atom types, bond orders, and stereochemistry.

    Args:
        rxn_smi (str): A reaction SMILES string in the format "reactant>>product".
        keep_atom_mapping (bool): If True, atom map numbers will be removed in the
            output CGR SMILES. Otherwise they will be retained (default).
        remove_brackets (bool): If True, redundant square brackets will be removed
            in the output CGR SMILES. Otherwise they will be kept (default).
        remove_hydrogens (bool): If True, explicit hydrogens will be removed in the
            output CGR SMILES. Otherwise they will be kept (default).
        balance_rxn (bool, optional): If True, attempts to balance the reaction
            before generating the CGR. Defaults to False.
        use_aromaticity (bool, optional): If True, RDKit aromaticity perception is applied
            during sanitization, and aromatic atoms will be written in lowercase (e.g. "c").
            If False, aromaticity perception is skipped, and atoms will be written in
            uppercase (e.g. "C"). Defaults to True.

    Returns:
        str: A CGR SMILES string representing the reaction as a single molecule
        with annotations of changes using `{reac|prod}` syntax.

    Notes:
        - Requires all atoms in the SMILES to be atom-mapped.
        - Requires balanced reactions.

    Example:
        >>> smi_reac = "[C:1]([H:3])([H:4])([H:5])[H:6].[Cl:2][H:7]"
        >>> smi_prod = "[C:1]([H:3])([H:4])([H:5])[Cl:2].[H:6][H:7]"
        >>> rxn_smiles = f"{smi_reac}>>{smi_prod}"
        >>> rxn_to_cgr(rxn_smiles)
        "[C:1]1([H:3])([H:4])([H:5]){-|~}[H:6]{~|-}[H:7]{-|~}[Cl:2]{~|-}1"

        # In the resulting CGR SMILES, the `{reac|prod}` notation encodes how atoms and bonds
        # change from reactants to products. For example, '[H:6]{~|-}[H:7]' means that while there
        # was no bond between these two hydrogen atoms in the reactants, a single bond has been
        # formed between them in the product molecule.
    """
    try:
        if not is_rxn_balanced(rxn_smi):
            if balance_rxn:
                rxn_smi = balance_reaction(rxn_smi, use_aromaticity=use_aromaticity)
            else:
                raise ValueError(
                    "The given rxn is not balanced. To enable cgr transform, set `balance_reaction=True`."
                )

        # TODO: check if rxn_smiles is atom mapped, if not, add mapping.
        # TODO: maybe let this function make the assumption of balanced, fully mapped reactions.
        # Handling of the preparation, shall the wrapper do.
        # fully_atom_mapped = is_fully_atom_mapped(rxn_smi)
        # if not fully_atom_mapped:
        #     print(f"WARNING: given reaction smiles is not fully atom mapped: {rxn_smi}")
        #     rxn_smi = add_atom_mapping(rxn_smi)

        smi_reac, _, smi_prod = rxn_smi.split(">")
        mol_reac, mol_prod = (
            make_mol(smi_reac, use_aromaticity=use_aromaticity),
            make_mol(smi_prod, use_aromaticity=use_aromaticity),
        )

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
        smi_cgr = Chem.MolToSmiles(mol_cgr, canonical=False)
        mol_cgr = make_mol(smi_cgr, sanitize=False, use_aromaticity=use_aromaticity)

        # reorder reac and prod molecule so we get the relative stereochemistry tags right:
        # TODO: by doing the reordering, we basically canonicalize and make it a non-injective mapping
        # TODO: maybe instead just align the mapping of the product with the one in the reactant
        prod_map_to_id = dict([(atom.GetAtomMapNum(), atom.GetIdx()) for atom in mol_prod.GetAtoms()])
        prod_reorder = [prod_map_to_id[a.GetAtomMapNum()] for a in mol_cgr.GetAtoms()]
        mol_prod = Chem.RenumberAtoms(mol_prod, prod_reorder)
        smi_prod = Chem.MolToSmiles(mol_prod, canonical=False)

        reac_map_to_id = dict([(atom.GetAtomMapNum(), atom.GetIdx()) for atom in mol_reac.GetAtoms()])
        reac_reorder = [reac_map_to_id[a.GetAtomMapNum()] for a in mol_cgr.GetAtoms()]
        mol_reac = Chem.RenumberAtoms(mol_reac, reac_reorder)
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
                replace_dict_atoms[atom_cgr.GetAtomMapNum()] = f"{{{reac_smarts}|{prod_smarts}}}"
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
            smi_cgr = smi_cgr.replace(re.findall(rf"\[[^):]*:{k}\]", smi_cgr)[0], replace_dict_atoms[k])

        if not keep_atom_mapping:
            smi_cgr = remove_atom_mapping(smi_cgr)

        if remove_brackets and remove_hydrogens:
            smi_cgr = remove_redundant_brackets_and_hydrogens(smi_cgr)

        elif remove_brackets:
            smi_cgr = remove_redundant_brackets(smi_cgr)

        return smi_cgr

    except Exception as e:
        logger.warning(f"Failed to process RXN-SMILES '{rxn_smi}'. Error: {e}. Returning empty string.")
        return ""
