from rdkit import Chem
from rdkit.Chem import rdchem
from cgr_smiles.atom_mapping import add_atom_mapping
from cgr_smiles.utils import get_list_of_atom_map_numbers, make_mol

from collections import Counter


def get_element_counts(smiles: str) -> Counter:
    """
    Given a dot-separated SMILES string, return a Counter of element counts.
    Implicit hydrogens are included.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    counts = Counter()
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] += 1
    return counts


def is_rxn_balanced(rxn_smi: str) -> str:
    """
    Check if a reaction SMILES is atom-balanced.
    """
    if ">>" not in rxn_smi:
        raise ValueError("Not a valid reaction SMILES (missing '>>').")

    reac_smi, prod_smi = rxn_smi.split(">>")
    reac_counts = get_element_counts(reac_smi) if reac_smi else Counter()
    prod_counts = get_element_counts(prod_smi) if prod_smi else Counter()

    return reac_counts == prod_counts


def is_rxn_mapped(rxn_smi: str) -> str:
    reac_smi, prod_smi = rxn_smi.split(">>")
    map_nums_reac = get_list_of_atom_map_numbers(reac_smi)
    map_nums_prod = get_list_of_atom_map_numbers(prod_smi)
    if len(map_nums_reac) == 0 or len(map_nums_prod) == 0:
        return False

    return True


def make_balanced_and_fully_mapped(rxn_smi: str) -> str:
    is_mapped = is_rxn_mapped(rxn_smi)
    is_balanced = is_rxn_balanced(rxn_smi)
    if is_balanced and is_mapped:
        return rxn_smi
    elif is_balanced and not is_mapped:
        # add atom mapping
        rxn_smi = add_atom_mapping(rxn_smi, method="graph_overlay")  # or rxn_mapper
        return rxn_smi
    elif not is_balanced:
        # check if their is a partial mapping:
        # add a partial mapping
        rxn_smi = add_atom_mapping(rxn_smi, method="graph_overlay")  # or rxn_mapper
        rxn_smi = balance_reaction(rxn_smi)
        return rxn_smi


# our assumption: if we receive an unbalanced, partially mapped reaction,
# we assume the mapping is present for the intersection of atoms that occour
# in both reac and prod. Otherwise, we delete the mapping, and create a new one.
def balance_reaction(rxn_smiles: str) -> str:
    smi_reac, smi_prod = rxn_smiles.split(">>")
    mol_reac, mol_prod = make_mol(smi_reac), make_mol(smi_prod)

    # Editable version of reactant mol
    rw_reac = Chem.RWMol(mol_reac)
    rw_prod = Chem.RWMol(mol_prod)

    # Map numbers present in each side
    map_nums_reac = {a.GetAtomMapNum() for a in mol_reac.GetAtoms()}
    map_nums_prod = {a.GetAtomMapNum() for a in mol_prod.GetAtoms()}

    max_map_num = max(max(map_nums_reac), max(map_nums_prod))

    # if any unmapped ones in reac or mol, add new atom map numbers
    for atom in rw_reac.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num == 0:
            atom.SetAtomMapNum(max_map_num + 1)
            # new_num = atom.GetAtomMapNum()
            max_map_num += 1

    for atom in rw_prod.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num == 0:
            atom.SetAtomMapNum(max_map_num + 1)
            # new_num = atom.GetAtomMapNum()
            max_map_num += 1

    # smi_reac2, smi_prod2 = (
    #     Chem.MolToSmiles(rw_reac, canonical=False),
    #     Chem.MolToSmiles(rw_prod, canonical=False),
    # )

    # Find atoms missing from reactants
    map_nums_reac = {a.GetAtomMapNum() for a in rw_reac.GetAtoms()}
    map_nums_prod = {a.GetAtomMapNum() for a in rw_prod.GetAtoms()}

    missing_map_nums_reac = [num for num in map_nums_prod if num not in map_nums_reac]
    missing_map_nums_prod = [num for num in map_nums_reac if num not in map_nums_prod]

    # Store new atom indices for later bond creation
    new_atom_indices_reac = {}
    new_atom_indices_prod = {}

    # Store bonds to add later: (map1, map2, bond_type)
    bonds_to_add_reac = []
    bonds_to_add_prod = []

    frag_reac = Chem.RWMol()
    frag_prod = Chem.RWMol()
    mapnum_to_frag_idx = {}

    for atom in rw_prod.GetAtoms():
        atom_map_num = atom.GetAtomMapNum()
        if atom_map_num in missing_map_nums_reac:
            new_atom = rdchem.Atom(atom)  # copy atom
            frag_idx = frag_reac.AddAtom(new_atom)
            mapnum_to_frag_idx[atom_map_num] = frag_idx

            for bond in atom.GetBonds():
                other_atom = bond.GetOtherAtom(atom)
                other_map_num = other_atom.GetAtomMapNum()
                if other_map_num in missing_map_nums_reac:
                    bonds_to_add_reac.append(
                        (atom_map_num, other_map_num, bond.GetBondType())
                    )

    # add missing atoms to reac and prod
    for atom in rw_reac.GetAtoms():
        atom_map_num = atom.GetAtomMapNum()
        if atom_map_num in missing_map_nums_prod:
            new_atom = rdchem.Atom(atom)
            idx = frag_prod.AddAtom(new_atom)
            new_atom_indices_prod[atom_map_num] = idx

            for bond in atom.GetBonds():
                other_atom = bond.GetOtherAtom(atom)
                other_map_num = other_atom.GetAtomMapNum()
                if other_map_num in missing_map_nums_prod:
                    bonds_to_add_prod.append(
                        (atom_map_num, other_map_num, bond.GetBondType())
                    )

    # add bonds to reac mol
    for m1, m2, b_type in bonds_to_add_reac:
        if m1 in new_atom_indices_reac:
            idx1 = new_atom_indices_reac[m1]
        else:
            idx1 = next(
                a.GetIdx() for a in frag_reac.GetAtoms() if a.GetAtomMapNum() == m1
            )

        if m2 in new_atom_indices_reac:
            idx2 = new_atom_indices_reac[m2]
        else:
            idx2 = next(
                a.GetIdx() for a in frag_reac.GetAtoms() if a.GetAtomMapNum() == m2
            )

        if frag_reac.GetBondBetweenAtoms(idx1, idx2) is None:
            frag_reac.AddBond(idx1, idx2, b_type)

    # add bonds to prod mol
    for m1, m2, b_type in bonds_to_add_prod:
        if m1 in new_atom_indices_prod:
            idx1 = new_atom_indices_prod[m1]
        else:
            idx1 = next(
                a.GetIdx() for a in frag_prod.GetAtoms() if a.GetAtomMapNum() == m1
            )

        if m2 in new_atom_indices_prod:
            idx2 = new_atom_indices_prod[m2]
        else:
            idx2 = next(
                a.GetIdx() for a in frag_prod.GetAtoms() if a.GetAtomMapNum() == m2
            )

        if frag_prod.GetBondBetweenAtoms(idx1, idx2) is None:
            frag_prod.AddBond(idx1, idx2, b_type)

    # for atom in frag_reac.GetAtoms():
    #     if atom.GetSymbol() == "N" and atom.GetTotalValence() < atom.GetValence(which=ValenceType.EXPLICIT):
    #         atom.SetNumExplicitHs(1)
    #         atom.UpdatePropertyCache()

    # for atom in frag_prod.GetAtoms():
    #     if atom.GetSymbol() == "N" and atom.GetTotalValence() < atom.GetValence(which=ValenceType.EXPLICIT):
    #         atom.SetNumExplicitHs(1)
    #         atom.UpdatePropertyCache()

    # hs_reac_mol = count_total_hydrogens(rw_reac)
    # hs_prod_mol = count_total_hydrogens(rw_prod)

    # hs_reac_frg = count_total_hydrogens(frag_reac)
    # hs_prod_frg = count_total_hydrogens(frag_prod)
    # print(f"hs_reac_mol = {hs_reac_mol}")
    # print(f"hs_reac_frg = {hs_reac_frg}")
    # print(f"hs_prod_mol = {hs_prod_mol}")
    # print(f"hs_prod_frg = {hs_prod_frg}")

    rw_reac = Chem.RWMol(Chem.CombineMols(rw_reac, frag_reac))
    rw_prod = Chem.RWMol(Chem.CombineMols(rw_prod, frag_prod))

    # Step 4: Return the balanced reaction SMILES
    balanced_reac_smi = Chem.MolToSmiles(rw_reac, canonical=False)
    balanced_prod_smi = Chem.MolToSmiles(rw_prod, canonical=False)
    return balanced_reac_smi + ">>" + balanced_prod_smi


def count_total_hydrogens(mol):
    return sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
