import pytest
from rdkit import Chem

from sr_smiles.io.logger import logger


@pytest.fixture
def propagated_logger():
    """A fixture to temporarily enable propagation on the logger for testing."""
    # fixture to enable propagation for the test
    original_propagation = logger.propagate
    logger.propagate = True
    yield logger

    # restore the original setting after testing
    logger.propagate = original_propagation


def mapping_pattern(smi: str):
    """Extracts the mapping pattern from a mapped reaction SMILES."""
    reac_smi, prod_smi = smi.split(">>")
    mol_reac = Chem.MolFromSmiles(reac_smi)
    mol_prod = Chem.MolFromSmiles(prod_smi)

    # Map number -> list of atom indices in reac/prod
    reactant_map = {}
    for atom in mol_reac.GetAtoms():
        amap = atom.GetAtomMapNum()
        if amap:
            reactant_map[amap] = reactant_map.get(amap, []) + [("R", atom.GetIdx())]

    product_map = {}
    for atom in mol_prod.GetAtoms():
        amap = atom.GetAtomMapNum()
        if amap:
            product_map[amap] = product_map.get(amap, []) + [("P", atom.GetIdx())]

    # Build a normalized pattern: for each map number, store counts of atoms in R and P
    # This ignores the actual map number and focuses on the "shape" of the mapping
    pattern = []
    for amap in sorted(set(reactant_map) | set(product_map)):
        r_count = len([x for x in reactant_map.get(amap, []) if x[0] == "R"])
        p_count = len([x for x in product_map.get(amap, []) if x[0] == "P"])
        pattern.append((r_count, p_count))

    # Sort pattern so that different numbering orders still match
    return sorted(pattern)


def equivalent_reactions(smi: str, ref_smi: str) -> bool:
    """Check if the mapping pattern matches ref_smi, ignoring map numbers."""
    return mapping_pattern(smi) == mapping_pattern(ref_smi)
