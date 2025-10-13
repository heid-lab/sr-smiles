import re
from typing import Literal

from rdkit import Chem
from rdkit.Chem import rdFMCS

from cgr_smiles.chem_utils.mol_utils import (
    get_atom_map_nums_of_mol,
    make_mol,
)
from cgr_smiles.chem_utils.smiles_utils import ORGANIC_SUBSET, TokenType, _tokenize
from cgr_smiles.io.logger import logger


class BaseMapper:
    """Abstract base class defining a callable interface for reaction mappers."""

    def __call__(self, rxn: str) -> str:
        """Return a mapped reaction SMILES string."""
        raise NotImplementedError


class RxnMapperWrapper(BaseMapper):
    """Wrapper around the Schwaller et al. RXNMapper model for atom mapping reactions."""

    def __init__(self):
        """Initialize the RXNMapper backend, raising an error if it's not installed."""
        try:
            from rxnmapper import RXNMapper
        except ImportError:
            raise ImportError("RxnMapper is not installed. Install with: `pip install rxnmapper`")

        self.mapper = RXNMapper()

    def __call__(self, rxn: str) -> str:
        """Map atoms in a reaction string using RXNMapper."""
        res = self.mapper.get_attention_guided_atom_maps([rxn], canonicalize_rxns=False)[0]
        return res["mapped_rxn"]


class GraphOverlayWrapper(BaseMapper):
    """Mapper using a graph‑overlay (MCS) approach for atom mapping."""

    def __init__(self):
        """Initialize the graph overlay mapping function."""
        self.mapping_func = maximum_common_substructure_mapping

    def __call__(self, rxn: str) -> str:
        """Map a reaction by aligning atoms via maximum common substructure overlap."""
        return self.mapping_func(rxn)


class IdentityMapper(BaseMapper):
    """A no-op reaction mapper that returns the input reaction unchanged."""

    def __call__(self, rxn: str) -> str:
        """Return the input reaction SMILES without modification."""
        return rxn


def is_fully_atom_mapped(rxn_smiles: str) -> bool:
    """Check if a reaction SMILES string is fully atom-mapped.

    A reaction is considered fully atom-mapped if every atom in the reactants
    has a non-zero atom map number that corresponds to an atom in the products,
    and vice versa.

    Args:
        rxn_smiles (str): A reaction SMILES string in the format "reactants>>products".

    Returns:
        bool: True if all atoms are mapped and correspond between reactants and products, False otherwise.
    """
    smi_reac, _, smi_prod = rxn_smiles.split(">")
    mol_reac, mol_prod = make_mol(smi_reac), make_mol(smi_prod)
    reac_map_nums = get_atom_map_nums_of_mol(mol_reac)
    prod_map_nums = get_atom_map_nums_of_mol(mol_prod)
    if reac_map_nums is None or prod_map_nums is None:
        raise ValueError("Invalid SMILES in reaction")

    # all atoms have non-zero map numbers
    if not all(reac_map_nums) or not all(prod_map_nums):
        return False

    # mapping numbers match between reactants and products
    return set(reac_map_nums) == set(prod_map_nums)


def is_cgr_smiles_fully_atom_mapped(cgr_smiles: str) -> bool:
    """Checks if a CGR-SMILES string is fully atom-mapped.

    Checks according to the following definition:
    - All CGRTOKENs ({...|...}) must have both alternatives atom-mapped with
      the SAME map number.
    - All TOKENs ([...]) must be atom-mapped.
    """
    atom_map_pattern = re.compile(r":\d+")

    for tok_type, _, tok in _tokenize(cgr_smiles):
        if tok_type == TokenType.ATOM:
            if not atom_map_pattern.search(tok):
                return False

    return True


def add_atom_mapping(
    rxn_smiles: str,
    method: Literal["rxnmapper", "graph_overlay"] = "rxnmapper",
    canonical: bool = False,
) -> str:
    """Add atom mapping to a reaction SMILES string using the specified method.

    This function maps all atoms in a reaction SMILES string using either a machine-learning
    approach (`rxnmapper`) or a rule-based graph overlay approach (`graph_overlay`). Existing
    mappings may be preserved or replaced depending on the method.

    Args:
        rxn_smiles (str): A reaction SMILES string in the format "reactants>>products".
        method (Literal["rxnmapper", "graph_overlay"]): The mapping method to use.
            - "rxnmapper": Uses the NN-RXNMapper (by Schwaller et. al) to predict atom mapping.
                Best for unmapped reactions.
            - "graph_overlay": Uses a maximum common substructure approach to extend or assign mappings.
        canonical (bool): If True, reactions are canonicalized before mapping (only affects RXNMapper).

    Returns:
        str: A reaction SMILES string with complete atom mapping.

    Raises:
        ImportError: If `rxnmapper` is requested but not installed.
        ValueError: If the reaction cannot be mapped or an unknown method is specified.

    Notes:
        - RXNMapper is inefficient for single reactions; batch processing is recommended.
        - RXNMapper may overwrite existing atom mapping, and performance may drop for unbalanced reactions.
        - Graph overlay mapping works for both balanced and unbalanced reactions.
    """
    # TODO: introduce a hybrid approach, where we do rxn_mapper, but if the confidence is low, we do a rule-based mapping.  # noqa: E501

    if method == "rxnmapper":
        logger.warning(
            "Calling the RXNmapper for a single reaction SMILES is very inefficient. "
            "For multiple reactions, use the wrapper class instead (TODO), "
            "which runs the mapper more efficiently in batch."
        )
        try:
            from rxnmapper import RXNMapper
        except ImportError:
            raise ImportError("RxnMapper is not installed. Install with: `pip install rxnmapper`")

        mapper = RXNMapper()
        # TODO: check if the rxn_smiles is partially mapped, if so print a warning, that the rxnmapper was trained to predict mapping for unmapped reaction, therefore the present mapping will be stripped.  # noqa: E501
        # NOTE: RXNMapper handles only unmapped reactions correctly. erformance drops for unbalanced reactions
        result = mapper.get_attention_guided_atom_maps([rxn_smiles], canonicalize_rxns=canonical)
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


def add_atom_mapping_to_cgr(cgr: str) -> str:
    """Add atom mapping numbers to a CGR-SMILES string.

    Each atom gets a continuous unique index: 1, 2, 3, ...
    Atoms inside the same {...|...} group share one index.
    """
    atom_pattern = re.compile(r"(\[[^\]]+\]|[A-Z][a-z]?|[cnops])")
    mapping_counter = 1

    def insert_mapping(atom, idx):
        if atom.startswith("["):
            if re.search(r":\d+", atom) is not None:  # already has mapping
                return atom
            return atom[:-1] + f":{idx}]"
        else:
            return f"[{atom}:{idx}]"

    out = []
    i = 0

    preexisting_map_nums = set()

    while i < len(cgr):
        # before mapping any new atom, make sure counter is unused
        while mapping_counter in preexisting_map_nums:
            mapping_counter += 1

        if cgr[i] == "{":  # handle group {...|...}
            j = cgr.find("}", i)
            group_content = cgr[i + 1 : j]

            # recursively map inside group using same index
            group_mapped = atom_pattern.sub(
                lambda m: insert_mapping(m.group(), mapping_counter), group_content
            )
            if re.search(r":\d+", group_mapped) is not None:
                mapping_counter += 1
            out.append("{" + group_mapped + "}")
            i = j + 1
        else:
            m = atom_pattern.match(cgr, i)
            # check if we have an atom token
            if m:
                token = m.group()

                # check if token already has a mapping
                existing_map = re.search(r":(\d+)\]", token)
                if existing_map:
                    map_num = int(existing_map.group(1))
                    preexisting_map_nums.add(map_num)
                    out.append(token)
                    i = m.end()
                    continue

                # check case of uppercase + lowercase (like "Sc")
                if (
                    len(token) == 2
                    and token[0].isupper()
                    and token[1].islower()
                    and token not in ORGANIC_SUBSET
                ):
                    # split into separate atoms
                    out.append(insert_mapping(token[0], mapping_counter))
                    mapping_counter += 1

                    while mapping_counter in preexisting_map_nums:
                        mapping_counter += 1

                    out.append(insert_mapping(token[1], mapping_counter))
                    mapping_counter += 1
                else:
                    out.append(insert_mapping(token, mapping_counter))
                    mapping_counter += 1

                i = m.end()
            else:
                out.append(cgr[i])
                i += 1
    return "".join(out)


def maximum_common_substructure_mapping(rxn_smiles: str) -> str:
    """Generate an atom-mapped reaction SMILES using a Maximum Common Substructure (MCS) approach.

    This function preserves any existing atom mapping in the reactants and products,
    identifies unmapped atoms, and extends the mapping using a maximum common substructure
    algorithm. Any remaining unmapped atoms after MCS matching are assigned new atom map numbers.

    Args:
        rxn_smiles (str): A reaction SMILES string in the format "reactants>>products".

    Returns:
        str: A reaction SMILES string where all atoms in reactants and products are atom-mapped.

    Notes:
        - Uses RDKit's MCS algorithm with valence and ring constraints to match unmapped substructures.
        - Existing atom mappings are preserved.
        - Explicit hydrogens are retained in the output (`allHsExplicit=True`).
    """
    reactants_smiles, products_smiles = rxn_smiles.split(">>")

    reactants = make_mol(reactants_smiles)
    products = make_mol(products_smiles)

    # 1. Identify already mapped atoms
    mapped_reac = {a.GetIdx(): a.GetAtomMapNum() for a in reactants.GetAtoms() if a.GetAtomMapNum() > 0}
    mapped_prod = {a.GetIdx(): a.GetAtomMapNum() for a in products.GetAtoms() if a.GetAtomMapNum() > 0}

    # 2. Get unmapped atom indices
    unmapped_reac_idx = [a.GetIdx() for a in reactants.GetAtoms() if a.GetAtomMapNum() == 0]
    unmapped_prod_idx = [a.GetIdx() for a in products.GetAtoms() if a.GetAtomMapNum() == 0]

    max_map_num = max(set(mapped_reac.values()) | set(mapped_prod.values()), default=0) + 1
    if not unmapped_reac_idx or not unmapped_prod_idx:
        for p_idx in unmapped_prod_idx:
            products.GetAtomWithIdx(p_idx).SetAtomMapNum(max_map_num)
            max_map_num += 1

        for r_idx in unmapped_reac_idx:
            reactants.GetAtomWithIdx(r_idx).SetAtomMapNum(max_map_num)
            max_map_num += 1

        mapped_reactants = Chem.MolToSmiles(reactants, allHsExplicit=True, canonical=False)
        mapped_products = Chem.MolToSmiles(products, allHsExplicit=True, canonical=False)
        return f"{mapped_reactants}>>{mapped_products}"

    # 3. Extract unmapped submols and keep index maps
    frag_reac_idx_map = {new_idx: orig_idx for new_idx, orig_idx in enumerate(unmapped_reac_idx)}
    frag_prod_idx_map = {new_idx: orig_idx for new_idx, orig_idx in enumerate(unmapped_prod_idx)}

    unmapped_reac = Chem.MolFromSmiles(
        Chem.MolFragmentToSmiles(reactants, atomsToUse=unmapped_reac_idx, isomericSmiles=True)
    )
    unmapped_prod = Chem.MolFromSmiles(
        Chem.MolFragmentToSmiles(products, atomsToUse=unmapped_prod_idx, isomericSmiles=True)
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
