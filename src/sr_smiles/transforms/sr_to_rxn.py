import re
from typing import List, Optional, Tuple, Union

import pandas as pd
from rdkit import Chem

from sr_smiles.atom_mapping import add_atom_mapping_to_sr, is_sr_smiles_fully_atom_mapped
from sr_smiles.chem_utils.list_utils import (
    common_elements_preserving_order,
    is_num_permutations_even,
)
from sr_smiles.chem_utils.mol_utils import remove_bonds_by_atom_map_nums
from sr_smiles.chem_utils.smiles_utils import (
    TokenType,
    _tokenize,
    extract_chiral_tag_by_atom_map_num,
    get_atom_map_adjacency_list_from_smiles,
    is_kekule,
    parse_bonds_in_order_from_smiles,
    remove_atom_mapping,
    remove_redundant_brackets,
)
from sr_smiles.chem_utils.stereo_chem_utils import find_e_z_stereo_bonds, get_chiral_center_map_nums
from sr_smiles.io.logger import logger


class SrToRxn:
    """Transform reaction SMILES into sr-SMILES.

    This class provides a callable interface to convert sr-SMILES into reaction
    SMILES. It supports single strings, lists of strings, pandas Series, and
    pandas DataFrames.

    Attributes:
        sr_col (Optional[str]): Column name in a DataFrame containing sr-SMILES.
        add_atom_mapping (bool, optional): If True, ensures atom mappings are
            present in the output RXN SMILES. If False, atom mappings are stripped
            unless they were already present in the input. Default is False.

    Examples:
        Transform a pandas DataFrame of reactions into sr-SMILES:

        >>> import pandas as pd
        >>> df = pd.read_csv("path/to/file.csv")
        >>> transform = SrToRxn(sr_col="sr_smiles")
        >>> df["rxn_smiles"] = transform(df)
    """

    def __init__(
        self,
        sr_col: Optional[str] = None,
        add_atom_mapping: bool = False,
    ) -> None:
        """Initializes the RXN to sr transformation settings."""
        self.sr_col = sr_col
        self.add_atom_mapping = add_atom_mapping

    def __call__(
        self, data: Union[str, List[str], pd.Series, pd.DataFrame]
    ) -> Union[str, List[str], pd.Series, pd.DataFrame]:
        """Apply the transformation to sr-SMILES.

        Args:
            data (Union[str, List[str], pd.Series, pd.DataFrame]): Input data
                containing sr-SMILES. Can be a single string, a list of strings,
                a pandas Series, or a pandas DataFrame.

        Returns:
            Union[str, List[str], pd.Series, pd.DataFrame]: Transformed sr-SMILES
            in the same structure as the input.

        Raises:
            ValueError: If a DataFrame is provided but `self.sr_col` is not set.
            TypeError: If the input type is not supported.
        """
        if isinstance(data, str):
            return sr_to_rxn(data, self.add_atom_mapping)

        elif isinstance(data, list):
            return [self(d) for d in data]

        elif isinstance(data, pd.Series):
            return data.apply(self)

        elif isinstance(data, pd.DataFrame):
            if self.sr_col is None:
                raise ValueError(
                    f"A pandas DataFrame was provided, but `self.sr_col` is not set.\n"
                    f"Available columns are: {list(data.columns)}\n"
                    "Please specify the column name containing the reactions by setting "
                    "`sr_col` at time of initialization."
                )
            return data[self.sr_col].apply(self)

        else:
            raise TypeError("Input must be str, list, pandas Series, or DataFrame.")


def sr_to_rxn(sr_smiles: str, add_atom_mapping: bool = False) -> str:
    """Converts an sr-SMILES string back into a reaction SMILES string.

    This function reverses a Superimposed Reaction (sr) SMILES representation
    into standard reaction SMILES (`reactants>>products`). It reconstructs reactant
    and product molecules by removing unspecified bonds, updating stereochemistry,
    and restoring chirality tags based on the sr annotations.

    Args:
        sr_smiles (str): An sr-SMILES string representing a reaction, where changes
            between reactants and products are encoded using `{reac|prod}` syntax.
        add_atom_mapping (bool, optional): If True, ensures atom mappings are
            present in the output RXN SMILES. If False, atom mappings are stripped
            unless they were already present in the input. Default is False.

    Returns:
        str: The corresponding reaction SMILES string in the format "reactants>>products".

    Notes:
        - Each substitution pattern in the sr-SMILES should follow `{...|...}`.
        - Unspecified bonds (labeled as "~") are removed in the resulting molecules.
        - This function is the reverse transformation of `rxn_to_sr`.
    """
    if sr_smiles == "":
        return ""

    try:
        sr_smiles = remove_radical_annotations(sr_smiles)

        if not is_sr_smiles_fully_atom_mapped(sr_smiles):
            sr_smi = add_atom_mapping_to_sr(sr_smiles)
            input_atom_mapped = False
        else:
            sr_smi = sr_smiles
            input_atom_mapped = True

        kekulized = is_kekule(sr_smi)

        # extract reac and prod smiles scaffold from sr smiles
        reac_smi, prod_smi = get_reac_prod_scaffold_smiles_from_sr_smiles(sr_smi)
        sr_reac_scaffold = reac_smi.replace("~", "")
        sr_prod_scaffold = prod_smi.replace("~", "")

        # try each side independently
        reac_smi_final = _rebuild_side_from_sr(reac_smi, sr_reac_scaffold, "reactant", kekulized)
        prod_smi_final = _rebuild_side_from_sr(prod_smi, sr_prod_scaffold, "product", kekulized)

        # if both sides failed, return empty string
        if reac_smi_final == "" and prod_smi_final == "":
            return ""

        rxn_smiles = f"{reac_smi_final}>>{prod_smi_final}"

        if not input_atom_mapped and not add_atom_mapping:
            rxn_smiles = remove_atom_mapping(rxn_smiles)
            rxn_smiles = remove_redundant_brackets(rxn_smiles)

        return rxn_smiles

    except Exception as e:
        logger.warning(
            f"Failure in `sr_to_rxn()` for input '{sr_smiles}'. " f"Error: {e}. Returning empty string."
        )
        return ""


def remove_radical_annotations(rxn_smi: str) -> str:
    """Removes radical signs ('^') from a reaction SMILES string."""
    return rxn_smi.replace("^", "")


def _rebuild_side_from_sr(
    side_smi: str,
    scaffold: str,
    sr_side_name: str,
    kekulized: bool,
) -> str:
    """Rebuilds either the reac or prod molecule from an sr-SMILES scaffold.

    This function reconstructs a standard SMILES representation from an sr-side SMILES string by:
      1. Parsing bond specifications (including unspecified bonds, '~').
      2. Removing any bonds that are marked as unspecified.
      3. Correcting E/Z double-bond stereochemistry based on parsed bond data.
      4. Updating chiral tags to ensure stereochemistry consistency with the sr-SMILES scaffold.

    It is used internally when decoding sr-SMILES representations back into reactant or product
    molecules during reaction reconstruction.

    Args:
        side_smi (str): The sr-side SMILES string representing one part (reactant or product).
        scaffold (str): The sr-SMILES scaffold used as reference for correcting chirality.
        sr_side_name (str): The name of the molecule side being processed
            (e.g., "reactant" or "product"), used for logging context.
        kekulized (bool): Whether to output a Kekulé (explicit bond) SMILES.

    Returns:
        str: The rebuilt SMILES string for the given sr-SMILES side.
             Returns an empty string if reconstruction fails for any reason.
    """
    try:
        parsed_bonds = parse_bonds_in_order_from_smiles(side_smi)

        # extract unspecified bonds to be deleted
        map_nums_unspecified_bonds = [key for key, val in parsed_bonds.items() if val == "~"]

        # build rdkit mol
        mol = Chem.MolFromSmiles(side_smi.replace("~", ""), sanitize=False)
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
        mol = remove_bonds_by_atom_map_nums(mol, map_nums_unspecified_bonds)

        # fix E/Z stereo
        mol = update_e_z_stereo_chem(mol, parsed_bonds)
        smi = Chem.MolToSmiles(mol, canonical=False, kekuleSmiles=kekulized)

        # fix chirality
        chiral_map_nums = get_chiral_center_map_nums(mol)
        smi = update_chirality_tags(smi, scaffold, chiral_map_nums)

        return smi

    except Exception as e:
        logger.warning(f"Failed to process sr-SMILES {sr_side_name} side '{side_smi}'. Error: {e}")
        return ""


def update_chirality_tags(smiles: str, sr_scaffold: str, chiral_center_map_nums: List[int]) -> str:
    """Updates chirality tags in a SMILES string based on an sr scaffold.

    Identifies chiral centers in the provided RDKit molecule (`mol`) by their atom
    map numbers. It then compares the neighborhood of these chiral centers in
    both the input SMILES (`smiles`) and a reference sr scaffold (`sr_scaffold`)
    to determine the chirality tags (@ or @@). If the chirality appears inverted
    between the SMILES and scaffold, the tag is flipped.

    Args:
        smiles (str): The input SMILES string of the molecule.
        sr_scaffold (List[int]): A reference sr-SMILES string containing correct chirality
            information for comparison.
        chiral_center_map_nums: List of the atom map numbers of the chiral centers.

    Returns:
        A new SMILES string with updated or corrected chirality tags.

    """
    reac_adj = get_atom_map_adjacency_list_from_smiles(smiles)
    sr_adj = get_atom_map_adjacency_list_from_smiles(sr_scaffold)

    reac_tokens = [[tok_type, tok] for tok_type, _, tok in _tokenize(smiles)]
    for i, (tok_type, tok) in enumerate(reac_tokens):
        if tok_type == TokenType.ATOM:
            match = re.search(r":(\d+)", tok)
            map_num = int(match.group(1))
            if map_num in chiral_center_map_nums:
                reac_nbrs = reac_adj[map_num]
                sr_nbrs = sr_adj[map_num]
                reac_nbrs, sr_nbrs = common_elements_preserving_order(reac_nbrs, sr_nbrs)

                current_tag = extract_chiral_tag_by_atom_map_num(sr_scaffold, map_num)

                if is_num_permutations_even(reac_nbrs, sr_nbrs):
                    chirality_tag = current_tag
                else:
                    if current_tag == "@":
                        chirality_tag = "@@"
                    elif current_tag == "@@":
                        chirality_tag = "@"

                # replace_pattern = rf"(\[[A-Z][a-z]?)(@{{1,2}})?(:{map_num}\])"
                replace_pattern = rf"(\[[A-Z][a-z]?)(@{{1,2}})?([+-]*:{map_num}\])"
                old_tok = reac_tokens[i][1]
                tok = re.sub(replace_pattern, rf"\1{chirality_tag}\3", old_tok)
                reac_tokens[i][1] = tok

    return "".join([str(tok[1]) for tok in reac_tokens])


def update_e_z_stereo_chem(mol: Chem.Mol, parsed_bonds: dict) -> Chem.Mol:
    """Update E/Z stereochemistry for double bonds in a molecule.

    This function uses pre-parsed bond and stereochemistry information to correct
    the E/Z configuration of double bonds in an RDKit molecule. Atom map numbers
    are preserved, and stereochemistry is updated according to the provided bond data.

    Args:
        mol (Chem.Mol): An RDKit molecule object with atom map numbers.
        parsed_bonds (dict): A dictionary where keys are bond identifiers (tuple of atom map numbers),
            and values are dictionaries containing:
                - 'terminal_atoms' (Tuple[int, int]): The atom map numbers of the bond ends.
                - 'stereo' (Chem.rdchem.BondStereo): The desired stereochemistry for the bond.

    Returns:
        Chem.Mol: The input molecule with updated E/Z stereochemistry on relevant bonds.
    """
    b = find_e_z_stereo_bonds(parsed_bonds)

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


def get_reac_prod_scaffold_smiles_from_sr_smiles(sr_smiles: str) -> Tuple[str, str]:
    """Extracts the reactant and product scaffold SMILES from an sr-SMILES string.

    The sr-SMILES encodes atom-level differences between reactants and products using
    substitution patterns in the form `{reactant|product}`.
    This function decodes those patterns by replacing each `{...|...}` block with the
    appropriate fragment in two parallel SMILES strings: one for the reactant, one for the product.

    Args:
        sr_smiles (str): An sr-SMILES string containing substitution patterns.

    Returns:
        Tuple[str, str]: A tuple containing the reactant SMILES and product SMILES
            with all substitution patterns resolved.
    """
    reac_smi = sr_smiles
    prod_smi = sr_smiles

    sr_pattern = r"\{([^{|}]*)\|([^{|}]*)\}"

    while "{" in reac_smi:
        match = re.search(sr_pattern, reac_smi)
        if match is None:
            break

        full_match = match.group(0)
        reac_fragment = match.group(1)
        prod_fragment = match.group(2)

        # replace the first match occurrence
        reac_smi = reac_smi.replace(full_match, reac_fragment, 1)
        prod_smi = prod_smi.replace(full_match, prod_fragment, 1)

    return reac_smi, prod_smi
