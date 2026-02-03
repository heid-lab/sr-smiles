import re
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from rdkit import Chem

from sr_smiles.atom_mapping import IdentityMapper, RxnMapperWrapper
from sr_smiles.chem_utils.list_utils import is_num_permutations_even, mask_nonshared_with_neg1
from sr_smiles.chem_utils.mol_utils import (
    get_atom_by_map_num,
    get_reac_to_prod_mapping,
    make_mol,
    reorder_mol,
)
from sr_smiles.chem_utils.smiles_utils import (
    TokenType,
    _tokenize,
    get_atom_map_adjacency_list_from_smiles,
    get_fragment_permutations,
    get_unchanged_explicit_hydrogen_map_nums,
    has_individually_mapped_hydrogens,
    remove_aromatic_bonds,
    remove_atom_mapping,
    remove_explicit_hydrogens_from_sr_smiles,
    remove_redundant_brackets,
    remove_redundant_brackets_and_hydrogens,
)
from sr_smiles.chem_utils.stereo_chem_utils import (
    flip_e_z_stereo,
    is_chiral_center,
    update_chirality_in_mol_from_smiles,
)
from sr_smiles.io.logger import logger
from sr_smiles.reaction_balancing import balance_reaction, is_balanced


class RxnToSr:
    """Transform reaction SMILES into sr-SMILES.

    This class provides a callable interface for converting reaction SMILES into
    superimposed reaction (sr) SMILES. It supports single strings, lists
    of strings, pandas Series, and pandas DataFrames.

    Transformation options include atom mapping preservation, bracket and
    hydrogen removal, reaction balancing, and Kekulé expansion control.

    Attributes:
        keep_atom_mapping (bool): Preserve atom mapping numbers in the output.
        remove_hydrogens (bool): Remove hydrogens from the output sr-SMILES.
            Automatically detects the hydrogen representation in the input:
            - If hydrogens are individually mapped (e.g., [H:1]), removes unchanged
              explicit hydrogens (those not involved in bond/charge/radical changes).
            - If hydrogens are implicit (e.g., [CH3]), simplifies to bare atoms (C).
        balance_rxn (bool): Attempt to balance the reaction stoichiometry before
            sr generation.
        rxn_col (Optional[str]): Column name in a DataFrame containing reaction SMILES.
        kekulize (bool): Convert aromatic atoms/bonds into explicit Kekulé notation.
            Defaults to False (keep aromatic notation).
        keep_aromatic_bonds (bool):  If True and used together with
            `kekulize=False`, aromatic bonds will be explicitly retained in the
            Kekulé-expanded sr (where supported). If False under `kekulize=False`,
            aromaticity is fully converted into alternating single/double bonds.
            Has no effect if `kekulize=True`. Defaults to True.
        use_rxnmapper (bool): If True, use RXNMapper for atom mapping before
            sr transformation. Requires the rxnmapper package to be installed.
            Defaults to False.
    """

    def __init__(
        self,
        keep_atom_mapping: bool = False,
        remove_hydrogens: bool = False,
        balance_rxn: bool = False,
        rxn_col: Optional[str] = None,
        kekulize: bool = False,
        keep_aromatic_bonds: bool = True,
        use_rxnmapper: bool = False,
    ) -> None:
        """Initializes the RXN to sr transformation settings."""
        self.keep_atom_mapping = keep_atom_mapping
        self.remove_hydrogens = remove_hydrogens
        self.balance_rxn = balance_rxn
        self.rxn_col = rxn_col
        self.kekulize = kekulize
        self.keep_aromatic_bonds = keep_aromatic_bonds
        self.use_rxnmapper = use_rxnmapper
        self.init_mapper()

    def init_mapper(self) -> None:
        """Initialize the reaction mapper.

        Creates and assigns a mapper instance to ``self.rxnmapper`` based on
        ``self.use_rxnmapper``.

        If True, uses the external RxnMapper package for atom mapping.
        If False, returns input reactions unchanged (identity mapping).
        """
        if self.use_rxnmapper:
            try:
                self.rxnmapper = RxnMapperWrapper()
            except ImportError:
                raise ImportError("RxnMapper is not installed. Run: pip install rxnmapper")

        else:
            self.rxnmapper = IdentityMapper()

    def __call__(
        self, data: Union[str, List[str], pd.Series, pd.DataFrame]
    ) -> Union[str, List[str], pd.Series, pd.DataFrame]:
        """Applies the transformation of RXN to sr-SMILES.

        Args:
            data (Union[str, List[str], pd.Series, pd.DataFrame]): Input reaction SMILES.
                Can be a single string, a list, a pandas Series, or a pandas DataFrame.

        Returns:
            Union[str, List[str], pd.Series, pd.DataFrame]: Output data of the same type
            as `data`, with each entry converted into its sr-SMILES representation.

        Raises:
            ValueError: If a DataFrame is provided but `self.rxn_col` is not set.
            TypeError: If the input type is not supported.
        """
        if isinstance(data, str):
            mapped_rxn = self.rxnmapper(data)
            return rxn_to_sr(
                mapped_rxn,
                keep_atom_mapping=self.keep_atom_mapping,
                remove_hydrogens=self.remove_hydrogens,
                balance_rxn=self.balance_rxn,
                kekulize=self.kekulize,
                keep_aromatic_bonds=self.keep_aromatic_bonds,
            )

        elif isinstance(data, list):
            result = [self(d) for d in data]
            n_res = len(result)
            n_empty = sum(item == "" for item in result)
            if result and n_empty:
                logger.warning(
                    f"sr transform failed for {n_empty}/{n_res} samples ({n_empty / n_res * 100:.1f}%)."
                )
            return result

        elif isinstance(data, pd.Series):
            return data.apply(self)

        elif isinstance(data, pd.DataFrame):
            if self.rxn_col is None:
                raise ValueError(
                    f"A pandas DataFrame was provided, but `self.rxn_col` is not set.\n"
                    f"Available columns are: {list(data.columns)}\n"
                    "Specify the correct column name in the constructor."
                )
            return data[self.rxn_col].apply(self)

        else:
            raise TypeError("Input must be str, list, pandas Series, or DataFrame.")


def rxn_to_sr(
    rxn_smi: str,
    keep_atom_mapping: bool = False,
    remove_hydrogens: bool = False,
    balance_rxn: bool = False,
    kekulize: bool = False,
    keep_aromatic_bonds: bool = False,
) -> str:
    """Converts a reaction SMILES string into a superimposed reaction (sr) SMILES.

    An sr-SMILES encodes the transformation between reactant and product molecules
    as a single, compact string representation, where atoms and bonds are annotated to
    show differences in atom types, bond orders, and stereochemistry.

    Args:
        rxn_smi (str): A reaction SMILES string in the format "reactant>>product".
        keep_atom_mapping (bool): If True, atom map numbers will be retained in the
            output sr-SMILES. Otherwise they will be removed (default).
        remove_hydrogens (bool): If True, removes hydrogens from the output sr-SMILES.
            Automatically detects the hydrogen representation in the input:
            - If hydrogens are individually mapped (e.g., [H:1]), removes unchanged
              explicit hydrogens (those not involved in bond/charge/radical changes).
            - If hydrogens are implicit (e.g., [CH3]), simplifies to bare atoms (C).
        balance_rxn (bool, optional): If True, attempts to balance the reaction
            before generating the sr. Defaults to False.
        kekulize (bool, optional): If True, converts all aromatic atoms/bonds into a
            specific Kekulé representation with alternating single/double bonds.
            Defaults to False (keep aromatic notation).
        keep_aromatic_bonds (bool, optional): If True and used together with
            `kekulize=False`, aromatic bonds will be explicitly retained in the
            Kekulé-expanded sr (where supported). If False under `kekulize=False`,
            aromaticity is fully converted into alternating single/double bonds.
            Has no effect if `kekulize=True`. Defaults to True.

    Returns:
        str: An sr-SMILES string representing the reaction as a single molecule
        with annotations of changes using `{reac|prod}` syntax.

    Notes:
        - Requires all atoms in the SMILES to be atom-mapped.
        - Requires balanced reactions.

    Example:
        >>> smi_reac = "[C:1]([H:3])([H:4])([H:5])[H:6].[Cl:2][H:7]"
        >>> smi_prod = "[C:1]([H:3])([H:4])([H:5])[Cl:2].[H:6][H:7]"
        >>> rxn_smiles = f"{smi_reac}>>{smi_prod}"
        >>> rxn_to_sr(rxn_smiles)
        "[C:1]1([H:3])([H:4])([H:5]){-|~}[H:6]{~|-}[H:7]{-|~}[Cl:2]{~|-}1"

        # In the resulting sr-SMILES, the `{reac|prod}` notation encodes how atoms and bonds
        # change from reactants to products. For example, '[H:6]{~|-}[H:7]' means that while there
        # was no bond between these two hydrogen atoms in the reactants, a single bond has been
        # formed between them in the product molecule.
    """
    try:
        # check if given rxn smiles is balanced
        if not is_balanced(rxn_smi):
            if balance_rxn:
                rxn_smi = balance_reaction(rxn_smi, kekulize=kekulize)
            else:
                raise ValueError(
                    "The given rxn is not balanced. "
                    "Set `balance_rxn=True` to apply automatic balancing before sr transformation"
                )

        # detect if hydrogens are individually mapped (e.g., [H:1]) or implicit (e.g., [CH3])
        has_explicit_h = has_individually_mapped_hydrogens(rxn_smi) if remove_hydrogens else False

        # start rxn to sr transformation
        rxn_smi, smi_sr_scaffold, mol_reac, mol_prod, mol_sr = get_chirality_aligned_smiles_and_mols(
            rxn_smi, kekulize
        )

        replace_dict_atoms, replace_dict_bonds = extract_atom_and_bond_changes(mol_reac, mol_prod, mol_sr)
        smi_sr = build_sr_smiles(smi_sr_scaffold, replace_dict_atoms, replace_dict_bonds)

        if remove_hydrogens and has_explicit_h:
            unchanged_h_map_nums = get_unchanged_explicit_hydrogen_map_nums(
                mol_reac, replace_dict_atoms, replace_dict_bonds
            )
            smi_sr = remove_explicit_hydrogens_from_sr_smiles(smi_sr, unchanged_h_map_nums)

        if not keep_atom_mapping:
            smi_sr = remove_atom_mapping(smi_sr)

        # remove redundant brackets
        if remove_hydrogens and not has_explicit_h:
            smi_sr = remove_redundant_brackets_and_hydrogens(smi_sr)
        else:
            smi_sr = remove_redundant_brackets(smi_sr)

        if not kekulize and not keep_aromatic_bonds:
            smi_sr = remove_aromatic_bonds(smi_sr)

        return smi_sr

    except Exception as e:
        logger.warning(
            f"Failed to process RXN SMILES '{rxn_smi}'.\n" f"Error: {e}.\n" "Returning empty string."
        )
        return ""


def get_sr_scaffold(mol_reac: Chem.Mol, mol_prod: Chem.Mol, kekulize: bool) -> Chem.Mol:
    """Builds an sr scaffold molecule from aligned reactant and product RDKit molecules.

    Combines the reactant and product molecules into a superimposed reaction scaffold
    by merging both bond sets according to atom map correspondence. Any bonds
    present in the product but missing in the reactant are added as unspecified
    bonds (`BondType.UNSPECIFIED`).

    Args:
        mol_reac (Chem.Mol): Atom-mapped RDKit molecule representing the reactant.
        mol_prod (Chem.Mol): Atom-mapped RDKit molecule representing the product.
            Atom map numbers must correspond to those in `mol_reac`.
        kekulize (bool): Whether to apply Kekulé expansion when creating the
            final RDKit molecule.

    Returns:
        Tuple[str, Chem.Mol]:
            - smi_sr (str): SMILES string of the generated sr scaffold.
            - mol_sr (Chem.Mol): RDKit molecule object of the sr scaffold.
    """
    # extract atom indices between reac and prod via atom map numbers
    ri2pi = get_reac_to_prod_mapping(mol_reac, mol_prod)

    # add missing bonds to the sr mol
    mol_sr = Chem.EditableMol(mol_reac)
    n_atoms = mol_reac.GetNumAtoms()

    for idx1 in range(n_atoms):
        for idx2 in range(idx1 + 1, n_atoms):
            bond_reac = mol_reac.GetBondBetweenAtoms(idx1, idx2)
            bond_prod = mol_prod.GetBondBetweenAtoms(ri2pi[idx1], ri2pi[idx2])

            if bond_reac is None and bond_prod is not None:
                mol_sr.AddBond(idx1, idx2, order=Chem.rdchem.BondType.UNSPECIFIED)

    mol_sr = mol_sr.GetMol()
    smi_sr = Chem.MolToSmiles(mol_sr, canonical=False)
    mol_sr = make_mol(smi_sr, sanitize=False, kekulize=kekulize)

    return smi_sr, mol_sr


def get_chirality_aligned_smiles_and_mols(
    rxn_smi: str, kekulize: bool, max_permutations: int = 5
) -> Tuple[str, str, Chem.Mol, Chem.Mol, Chem.Mol]:
    """Build reactant, product, and sr molecules from a reaction SMILES.

    Parses a reaction SMILES, builds RDKit molecule objects, and reorders
    them so that atom mapping and tetrahedral stereochemistry (chirality)
    are consistent between reactant and product sides.

    The routine performs the following steps:
      1. Parses the reaction SMILES into separate reactant and product strings.
      2. Builds RDKit Mol objects with optional kekulization.
      3. Constructs a superimposed reaction (sr) and uses its
         atom mapping to reorder reactant and product atoms.
      4. Detects tetrahedral centers present in both molecules and checks
         neighbor permutation parity.
      5. If an odd permutation is detected, reorders fragments in the reactant
         SMILES to restore consistent stereochemistry.
      6. Returns the possibly updated reaction SMILES along with intermediate
         SMILES strings and Mol objects.

    Args:
        rxn_smi (str): Reaction SMILES string with atom mappings.
        kekulize (bool): Whether to kekulize molecules when creating them.
        max_permutations (int, optional): Maximum number of fragment
            permutations to test for chirality alignment. Defaults to 3.

    Returns:
        Tuple[str, str, str, str, Chem.Mol, Chem.Mol, Chem.Mol]:
            A tuple of:
                * **rxn_smi_aligned** (`str`): Possibly reordered reaction SMILES.
                * **smi_sr** (`str`): SMILES of the superimposed reaction (sr).
                * **mol_reac** (`Chem.Mol`): Reactant molecule.
                * **mol_prod** (`Chem.Mol`): Product molecule.
                * **mol_sr** (`Chem.Mol`): sr molecule.
    """
    frag_permutations = get_fragment_permutations(rxn_smi.split(">>")[0], max_permutations)

    while frag_permutations:
        smi_reac, _, smi_prod = rxn_smi.split(">")
        mol_reac = make_mol(smi_reac, kekulize=kekulize)
        mol_prod = make_mol(smi_prod, kekulize=kekulize)
        smi_sr, mol_sr = get_sr_scaffold(mol_reac, mol_prod, kekulize)

        mol_reac = reorder_mol(mol_reac, mol_sr)
        mol_prod = reorder_mol(mol_prod, mol_sr)

        smi_reac = Chem.MolToSmiles(mol_reac, canonical=False)
        smi_prod = Chem.MolToSmiles(mol_prod, canonical=False)

        adj_reac = get_atom_map_adjacency_list_from_smiles(smi_reac)
        adj_prod = get_atom_map_adjacency_list_from_smiles(smi_prod)

        if len(frag_permutations) > 1:
            frag_permutations.pop(0)
            for atom_reac in mol_reac.GetAtoms():
                # check if atom is a chiral center in both reactant and product
                if is_chiral_center(atom_reac):
                    map_num = atom_reac.GetAtomMapNum()
                    atom_prod = get_atom_by_map_num(mol_prod, map_num)
                    if is_chiral_center(atom_prod):
                        nbrs_reac = adj_reac[map_num]
                        nbrs_prod = adj_prod[map_num]

                        nbrs_reac, nbrs_prod = mask_nonshared_with_neg1(nbrs_reac, nbrs_prod)

                        if not is_num_permutations_even(nbrs_reac, nbrs_prod):
                            frag_reac = smi_reac.split(".")
                            smi_reac = ".".join(frag_reac[i] for i in frag_permutations[0])
                            rxn_smi = f"{smi_reac}>>{smi_prod}"
                            break
            else:
                break
        else:
            frag_permutations.pop(0)

    update_chirality_in_mol_from_smiles(mol_reac, smi_reac, smi_sr)
    update_chirality_in_mol_from_smiles(mol_prod, smi_prod, smi_sr)

    return rxn_smi, smi_sr, mol_reac, mol_prod, mol_sr


def add_radical_sign(smarts_string: str, n_radicals: int) -> str:
    """Adds caret(s) '^' to a SMARTS string based on the number of radical electrons.

    One '^' for 1 radical, '^^' for 2 or more radicals.
    If the string is in square brackets, inserts inside: [C] -> [C^]
    """
    if n_radicals == 0:
        return smarts_string

    carets = "^" if n_radicals == 1 else "^^"

    if smarts_string.endswith("]"):
        return f"{smarts_string[:-1]}{carets}]"
    else:
        return f"{smarts_string}{carets}"


def extract_atom_and_bond_changes(
    mol_reac: Chem.Mol, mol_prod: Chem.Mol, mol_sr: Chem.Mol
) -> tuple[dict[int, str], dict[tuple[int, int], str]]:
    """Extract atom- and bond-level transformation SMARTS between reactant, product, and sr molecules.

    Args:
        mol_reac (Chem.Mol): reactant molecule.
        mol_prod (Chem.Mol): product molecule.
        mol_sr (Chem.Mol): superimposed reaction (sr) molecule.

    Returns:
        Tuple[
            Dict[int, str],
            Dict[Tuple[int, int], str]
        ]:
            - replace_dict_atoms: atom map → SMARTS replacement.
            - replace_dict_bonds: (begin_map, end_map) → bond SMARTS replacement.
    """
    replace_dict_atoms = {}
    replace_dict_bonds = {}
    n_atoms = mol_reac.GetNumAtoms()

    for i1 in range(n_atoms):
        atom_reac = mol_reac.GetAtomWithIdx(i1)
        atom_prod = mol_prod.GetAtomWithIdx(i1)
        atom_sr = mol_sr.GetAtomWithIdx(i1)

        reac_smarts = atom_reac.GetSmarts(isomericSmiles=True)
        prod_smarts = atom_prod.GetSmarts(isomericSmiles=True)

        if atom_reac.GetNumRadicalElectrons() > 0:
            reac_smarts = add_radical_sign(reac_smarts, n_radicals=atom_reac.GetNumRadicalElectrons())

        if atom_prod.GetNumRadicalElectrons() > 0:
            prod_smarts = add_radical_sign(prod_smarts, n_radicals=atom_prod.GetNumRadicalElectrons())

        if reac_smarts != prod_smarts:
            replace_dict_atoms[atom_sr.GetAtomMapNum()] = f"{{{reac_smarts}|{prod_smarts}}}"
        else:
            replace_dict_atoms[atom_sr.GetAtomMapNum()] = reac_smarts

        for i2 in range(i1 + 1, n_atoms):
            atom2_sr = mol_sr.GetAtomWithIdx(i2)
            map_num_1, map_num_2 = atom_sr.GetAtomMapNum(), atom2_sr.GetAtomMapNum()

            bond_reac = mol_reac.GetBondBetweenAtoms(i1, i2)
            bond_prod = mol_prod.GetBondBetweenAtoms(i1, i2)

            if bond_reac is None and bond_prod is None:
                continue

            # handle reactant bond
            if bond_reac is not None:
                smarts_bond_reac = bond_reac.GetSmarts(allBondsExplicit=True)
                reac_begin, reac_end = (
                    bond_reac.GetBeginAtom().GetAtomMapNum(),
                    bond_reac.GetEndAtom().GetAtomMapNum(),
                )
            else:
                smarts_bond_reac = "~"
                reac_begin, reac_end = map_num_1, map_num_2

            # handle product bond
            if bond_prod is not None:
                smarts_bond_prod = bond_prod.GetSmarts(allBondsExplicit=True)
                prod_begin, prod_end = (
                    bond_prod.GetBeginAtom().GetAtomMapNum(),
                    bond_prod.GetEndAtom().GetAtomMapNum(),
                )
                if reac_begin == prod_end and reac_end == prod_begin:
                    smarts_bond_prod = flip_e_z_stereo(smarts_bond_prod)
            else:
                smarts_bond_prod = "~"

            if smarts_bond_reac != smarts_bond_prod:
                val = f"{{{smarts_bond_reac}|{smarts_bond_prod}}}"
            else:
                val = smarts_bond_reac if smarts_bond_reac != "-" else ""

            replace_dict_bonds[(reac_begin, reac_end)] = val
            replace_dict_bonds[(reac_end, reac_begin)] = flip_e_z_stereo(val)

    return replace_dict_atoms, replace_dict_bonds


def build_sr_smiles(
    smi_sr_scaffold: str,
    replace_dict_atoms: Dict[int, str],
    replace_dict_bonds: Dict[Tuple[int, int], str],
) -> str:
    """Converts an sr-SMILES scaffold into an sr-SMILES with explicit atom and bond replacements.

    Args:
        smi_sr_scaffold (str): Scaffold of the sr-SMILES.
        replace_dict_atoms (dict[int, str]): Map number → replacement SMARTS for atoms.
        replace_dict_bonds (dict[tuple[int, int], str]):
            (begin_map, end_map) → replacement SMARTS for bonds.

    Returns:
        str: The constructed sr-SMILES string.
    """
    smiles = ""
    anchor = None
    idx = 0
    next_bond = None
    branches = []
    ring_nums = {}
    i2m = {}

    # change bonds
    for tokentype, _, token in _tokenize(smi_sr_scaffold):
        if tokentype == TokenType.ATOM:
            # extract atom map number
            i2m[idx] = int(token[:-1].split(":")[1])

            # connect current atom to previous (anchor)
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

        elif tokentype in (TokenType.BOND_TYPE, TokenType.EZSTEREO):
            next_bond = token

        elif tokentype == TokenType.RING_NUM:
            # handle ring closing
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

            # handle ring opening
            else:
                ring_nums[token] = (idx - 1, next_bond)
                next_bond = None
                smiles += str(token)

    # change atoms
    for k in replace_dict_atoms.keys():
        smiles = smiles.replace(re.findall(rf"\[[^):]*:{k}\]", smiles)[0], replace_dict_atoms[k])

    return smiles
