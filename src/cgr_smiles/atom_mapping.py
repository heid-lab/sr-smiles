from typing import Literal

from rdkit import Chem
from rdkit.Chem import rdFMCS

from cgr_smiles.utils import (
    get_atom_map_num_of_mol,
    make_mol,
)


def is_fully_atom_mapped(rxn_smiles: str) -> bool:
    """
    Checks if a reaction smiles if ully atom mapped.
    Fully atom mapped means, that each reactant atom has a non-zero numer assigned to it, which has a corresponding partner in the product.
    """
    smi_reac, _, smi_prod = rxn_smiles.split(">")
    mol_reac, mol_prod = make_mol(smi_reac), make_mol(smi_prod)
    reac_map_nums = get_atom_map_num_of_mol(mol_reac)
    prod_map_nums = get_atom_map_num_of_mol(mol_prod)
    if reac_map_nums is None or prod_map_nums is None:
        raise ValueError("Invalid SMILES in reaction")

    # all atoms have non-zero map numbers
    if not all(reac_map_nums) or not all(prod_map_nums):
        return False

    # mapping numbers match between reactants and products
    return set(reac_map_nums) == set(prod_map_nums)


def add_atom_mapping(
    rxn_smiles: str,
    method: Literal["rxnmapper", "graph_overlay"] = "rxnmapper",
    canonical: bool = False,
) -> str:
    # TODO: maybe even do a hybrid approach, where we do rxn_mapper, but if the confidence is low, we do a rule-based mapping.

    if method == "rxnmapper":
        print(
            "WARNING: Calling the RXNmapper for a single reaction SMILES is very inefficient. For multiple reactions, use the wrapper class instead (TODO), which runs the mapper more efficiently in batch."
        )
        try:
            from rxnmapper import RXNMapper
        except ImportError:
            raise ImportError(
                "RxnMapper is not installed. Install with: `pip install rxnmapper`"
            )

        mapper = RXNMapper()
        # TODO: check if the rxn_smiles is partially mapped, if so print a warning, that the rxnmapper was trained to predict mapping for unmapped reaction, therefore the present mapping will be stripped.
        # NOTE: RXNMapper handles only unmapped reactions correctly. erformance drops for unbalanced reactions
        result = mapper.get_attention_guided_atom_maps(
            [rxn_smiles], canonicalize_rxns=canonical
        )
        print(f"result = {result}")
        if not result or "mapped_rxn" not in result[0]:
            raise ValueError(f"RxnMapper failed to map reaction: {rxn_smiles}")
        return result[0]["mapped_rxn"]

    elif method == "graph_overlay":
        # assumption: rxn is either
        # 1. balanced and unmapped
        # 2. unbalanced and unmapped -> returns unbalanced and partially (intersection) mapped
        return maximum_common_substructure_mapping(rxn_smiles)

    else:
        raise ValueError(f"Unknown method: {method}")


def _maximum_common_substructure_mapping(rxn_smiles: str) -> str:
    """
    Generate atom-mapped reaction SMILES using a Maximum Common Substructure (MCS)
    graph overlay approach.

    This function attempts to assign atom map numbers to a reaction by:
      1. Parsing the reactants and products into RDKit molecule objects.
      2. Finding the Maximum Common Substructure (MCS) between reactants and products.
      3. Assigning the same atom map number to corresponding atoms in the MCS.
      4. Assigning new map numbers to any unmatched atoms (e.g., added/removed atoms).

    The result is a reaction SMILES string where each atom has an atom map number,
    allowing tracking of atom correspondence between reactants and products.

    Parameters
    ----------
    rxn_smiles : str
        Reaction SMILES string in the form "reactants>>products".
        Multiple reactants or products can be separated by '.' (dot).
        Example: "C([H])([H])[H]>>C([H])([H])[H]"

    Returns
    -------
    str
        Atom-mapped reaction SMILES string, e.g.:
        "[C:1]([H:2])([H:3])[H:4]>>[C:1]([H:2])([H:3])[H:4]"

    Notes
    -----
    - This method is deterministic and does not rely on machine learning.
    - It works best for reactions where most atoms are preserved between
      reactants and products.
    - For large rearrangements or reactions with no clear MCS, the mapping
      may be incomplete or arbitrary for unmatched atoms.
    - Multi-molecule reactions are supported, but the MCS is computed on
      the combined molecular graph of all reactants vs. all products.

    Examples
    --------
    >>> mcs_atom_mapping("C([H])([H])[H]>>C([H])([H])[H]")
    '[C:1]([H:2])([H:3])[H:4]>>[C:1]([H:2])([H:3])[H:4]'

    >>> mcs_atom_mapping("CCO>>CC=O")
    '[C:1][C:2][O:3]>>[C:1][C:2]=[O:3]'
    """
    reactants_smiles, products_smiles = rxn_smiles.split(">>")

    reactants = make_mol(reactants_smiles)
    products = make_mol(products_smiles)

    # find maximum common substructure
    mcs = rdFMCS.FindMCS(
        [reactants, products],
        matchValences=False,  # allow valence changes
        ringMatchesRingOnly=False,  # allow ring opening
        completeRingsOnly=False,  # allow partial ring matches
        matchChiralTag=False,  # ignore stereochemistry
        atomCompare=rdFMCS.AtomCompare.CompareElements,  # match same element
        bondCompare=rdFMCS.BondCompare.CompareAny,  # allow bond order changes
    )

    patt = Chem.MolFromSmarts(mcs.smartsString)
    react_match = reactants.GetSubstructMatch(patt)
    prod_match = products.GetSubstructMatch(patt)

    # Assign map numbers to matched atoms
    map_num = 1
    for r_idx, p_idx in zip(react_match, prod_match):
        reactants.GetAtomWithIdx(r_idx).SetAtomMapNum(map_num)
        products.GetAtomWithIdx(p_idx).SetAtomMapNum(map_num)
        map_num += 1

    # Assign map numbers to unmatched atoms
    for mol in (reactants, products):
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == 0:
                atom.SetAtomMapNum(map_num)
                map_num += 1

    mapped_reactants = Chem.MolToSmiles(reactants, allHsExplicit=True)
    mapped_products = Chem.MolToSmiles(products, allHsExplicit=True)

    return f"{mapped_reactants}>>{mapped_products}"


def maximum_common_substructure_mapping(rxn_smiles: str) -> str:
    """
    Generate atom-mapped reaction SMILES using a Maximum Common Substructure (MCS)
    approach, preserving any existing mapping and extending it to unmapped atoms.
    """
    reactants_smiles, products_smiles = rxn_smiles.split(">>")

    reactants = make_mol(reactants_smiles)
    products = make_mol(products_smiles)

    # 1. Identify already mapped atoms
    mapped_reac = {
        a.GetIdx(): a.GetAtomMapNum()
        for a in reactants.GetAtoms()
        if a.GetAtomMapNum() > 0
    }
    mapped_prod = {
        a.GetIdx(): a.GetAtomMapNum()
        for a in products.GetAtoms()
        if a.GetAtomMapNum() > 0
    }

    # 2. Get unmapped atom indices
    unmapped_reac_idx = [
        a.GetIdx() for a in reactants.GetAtoms() if a.GetAtomMapNum() == 0
    ]
    unmapped_prod_idx = [
        a.GetIdx() for a in products.GetAtoms() if a.GetAtomMapNum() == 0
    ]

    max_map_num = (
        max(set(mapped_reac.values()) | set(mapped_prod.values()), default=0) + 1
    )
    if not unmapped_reac_idx or not unmapped_prod_idx:
        for p_idx in unmapped_prod_idx:
            products.GetAtomWithIdx(p_idx).SetAtomMapNum(max_map_num)
            max_map_num += 1

        for r_idx in unmapped_reac_idx:
            reactants.GetAtomWithIdx(r_idx).SetAtomMapNum(max_map_num)
            max_map_num += 1

        mapped_reactants = Chem.MolToSmiles(
            reactants, allHsExplicit=True, canonical=False
        )
        mapped_products = Chem.MolToSmiles(
            products, allHsExplicit=True, canonical=False
        )
        return f"{mapped_reactants}>>{mapped_products}"

    # 3. Extract unmapped submols and keep index maps
    frag_reac_idx_map = {
        new_idx: orig_idx for new_idx, orig_idx in enumerate(unmapped_reac_idx)
    }
    frag_prod_idx_map = {
        new_idx: orig_idx for new_idx, orig_idx in enumerate(unmapped_prod_idx)
    }

    unmapped_reac = Chem.MolFromSmiles(
        Chem.MolFragmentToSmiles(
            reactants, atomsToUse=unmapped_reac_idx, isomericSmiles=True
        )
    )
    unmapped_prod = Chem.MolFromSmiles(
        Chem.MolFragmentToSmiles(
            products, atomsToUse=unmapped_prod_idx, isomericSmiles=True
        )
    )

    # 4. Run MCS
    params = rdFMCS.MCSParameters()
    params.matchValences = True
    params.ringMatchesRingOnly = True
    params.completeRingsOnly = True
    params.atomCompare = rdFMCS.AtomCompare.CompareElements
    params.bondCompare = rdFMCS.BondCompare.CompareAny
    # params.matchChiralTag = False

    mcs = rdFMCS.FindMCS([unmapped_reac, unmapped_prod], params)
    patt = Chem.MolFromSmarts(mcs.smartsString)
    reac_match = reactants.GetSubstructMatch(patt)
    prod_match = products.GetSubstructMatch(patt)

    # 5. Assign new map numbers
    map_num = max(set(mapped_reac.values()) | set(mapped_prod.values()), default=0) + 1
    for r_idx, p_idx in zip(reac_match, prod_match):
        reactants.GetAtomWithIdx(frag_reac_idx_map[r_idx]).SetAtomMapNum(map_num)
        products.GetAtomWithIdx(frag_prod_idx_map[p_idx]).SetAtomMapNum(map_num)
        map_num += 1

    # 6. Assign map numbers to any remaining unmatched atoms
    next_map = map_num
    for mol in (reactants, products):
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == 0:
                atom.SetAtomMapNum(next_map)
                next_map += 1

    # 7. Return mapped reaction SMILES
    mapped_reactants = Chem.MolToSmiles(reactants, allHsExplicit=True, canonical=False)
    mapped_products = Chem.MolToSmiles(products, allHsExplicit=True, canonical=False)
    return f"{mapped_reactants}>>{mapped_products}"


# def maximum_common_substructure_mapping(rxn_smiles: str) -> str:
#     reactants_smiles, products_smiles = rxn_smiles.split(">>")
#     reactants = Chem.MolFromSmiles(reactants_smiles)
#     products = Chem.MolFromSmiles(products_smiles)

#     if reactants is None or products is None:
#         raise ValueError("Invalid SMILES in reaction.")

#     # Find Maximum Common Substructure
#     mcs_result = rdFMCS.FindMCS(
#         [reactants, products],
#         matchValences=True,
#         ringMatchesRingOnly=True,
#         completeRingsOnly=True,
#         timeout=10
#     )

#     mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
#     if mcs_mol is None:
#         raise ValueError("No MCS found between reactants and products.")

#     # Get atom matches
#     react_match = reactants.GetSubstructMatch(mcs_mol)
#     prod_match = products.GetSubstructMatch(mcs_mol)

#     # Assign atom map numbers
#     for map_num, (r_idx, p_idx) in enumerate(zip(react_match, prod_match), start=1):
#         reactants.GetAtomWithIdx(r_idx).SetAtomMapNum(map_num)
#         products.GetAtomWithIdx(p_idx).SetAtomMapNum(map_num)

#     # Return mapped reaction SMILES
#     return f"{Chem.MolToSmiles(reactants)}>>{Chem.MolToSmiles(products)}"
