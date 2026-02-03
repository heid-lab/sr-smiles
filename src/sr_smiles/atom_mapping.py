import re

from sr_smiles.chem_utils.mol_utils import (
    get_atom_map_nums_of_mol,
    make_mol,
)
from sr_smiles.chem_utils.smiles_utils import ORGANIC_SUBSET, TokenType, _tokenize
from sr_smiles.io.logger import logger


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


def is_sr_smiles_fully_atom_mapped(sr_smiles: str) -> bool:
    """Checks if an sr-SMILES string is fully atom-mapped.

    Checks according to the following definition:
    - All sr-TOKENs ({...|...}) must have both alternatives atom-mapped with
      the SAME map number.
    - All TOKENs ([...]) must be atom-mapped.
    """
    atom_map_pattern = re.compile(r":\d+")

    for tok_type, _, tok in _tokenize(sr_smiles):
        if tok_type == TokenType.ATOM:
            if not atom_map_pattern.search(tok):
                return False

    return True


def add_atom_mapping(
    rxn_smiles: str,
    canonical: bool = False,
) -> str:
    """Add atom mapping to a reaction SMILES string using the specified method.

    This function maps all atoms in a reaction SMILES using RXNMapper. Existing
    mappings may be preserved or replaced depending on the method.

    Args:
        rxn_smiles (str): A reaction SMILES string in the format "reactants>>products".
        canonical (bool): If True, reactions are canonicalized before mapping (only affects RXNMapper).

    Returns:
        str: A reaction SMILES string with complete atom mapping.

    Raises:
        ImportError: If `rxnmapper` is requested but not installed.
        ValueError: If the reaction cannot be mapped or an unknown method is specified.

    Notes:
        - RXNMapper is inefficient for single reactions; batch processing is recommended.
        - RXNMapper may overwrite existing atom mapping, and performance may drop for unbalanced reactions.
    """
    logger.warning(
        "Calling the RXNmapper for a single reaction SMILES is very inefficient. "
        "For multiple reactions, use the wrapper class instead, "
        "which runs the mapper more efficiently in batch."
    )
    try:
        from rxnmapper import RXNMapper
    except ImportError:
        raise ImportError("RXNMapper is not installed. Install with: `pip install rxnmapper`")

    mapper = RXNMapper()
    result = mapper.get_attention_guided_atom_maps([rxn_smiles], canonicalize_rxns=canonical)
    if not result or "mapped_rxn" not in result[0]:
        raise ValueError(f"RXNMapper failed to map reaction: {rxn_smiles}")
    return result[0]["mapped_rxn"]


def add_atom_mapping_to_sr(sr: str) -> str:
    """Add atom mapping numbers to a sr-SMILES string.

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

    while i < len(sr):
        # before mapping any new atom, make sure counter is unused
        while mapping_counter in preexisting_map_nums:
            mapping_counter += 1

        if sr[i] == "{":  # handle group {...|...}
            j = sr.find("}", i)
            group_content = sr[i + 1 : j]

            # recursively map inside group using same index
            group_mapped = atom_pattern.sub(
                lambda m: insert_mapping(m.group(), mapping_counter), group_content
            )
            if re.search(r":\d+", group_mapped) is not None:
                mapping_counter += 1
            out.append("{" + group_mapped + "}")
            i = j + 1
        else:
            m = atom_pattern.match(sr, i)
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
                out.append(sr[i])
                i += 1
    return "".join(out)
