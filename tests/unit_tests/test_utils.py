from typing import List, Tuple

import pytest
from rdkit import Chem

from cgr_smiles.utils import (
    TokenType,
    _tokenize,
    canonicalize,
    common_elements_preserving_order,
    get_atom_by_map_num,
    get_atom_indices_and_smarts,
    get_atom_map_adjacency_list_from_smiles,
    get_atom_map_num,
    get_bond_idx,
    get_list_of_atom_map_numbers,
    includes_individually_mapped_hydrogens,
    is_num_permutations_even,
    map_reac_to_prod,
    parse_bonds_in_order_from_smiles,
    remove_atom_mapping,
    remove_redundant_square_brackets,
    update_all_atom_stereo,
)


@pytest.fixture
def simple_mol() -> Chem.Mol:
    """Provides a simple molecule."""
    smiles = "[C:1][C@:2]([Cl:3])([Br:4])[I:5]"
    mol = Chem.MolFromSmiles(smiles)
    return mol


@pytest.fixture
def reaction_mols() -> Tuple[Chem.Mol, Chem.Mol]:
    """Provides a simple reactant and product molecule pair."""
    reac_smi = "[CH3:1][C:2](=[O:3])[O:4].[CH3:5][NH2:6]"
    prod_smi = "[CH3:1][C:2](=[O:3])[NH:6][CH3:5].[OH2:4]"
    mol_reac = Chem.MolFromSmiles(reac_smi)
    mol_prod = Chem.MolFromSmiles(prod_smi)
    return mol_reac, mol_prod


@pytest.fixture
def simple_smiles() -> str:
    """Provides a simple atom-mapped SMILES string with explicit hydrogens."""
    return "[C:1][C@:2]([Cl:3])([Br:4])[I:5]"


@pytest.fixture
def complex_smiles() -> Tuple[Chem.Mol, Chem.Mol]:
    """Provides a complex smiles, including multiple rings and stereochemistry."""
    smiles = "[C:1]([C@@:2]1([H:11])[O:3][C@@:4]2([H:12])[C:5]([H:13])([H:14])[C:6]([H:15])([H:16])[C@@:7]12[H:17])([H:8])([H:9])[H:10]"  # noqa: E501
    return smiles


def test_remove_atom_mapping_rxn_smiles():
    """Test removal of atom mapping from a reaction smiles."""
    rxn_smiles = "[CH3:1][C:2](=[O:3])[O:4].[CH3:5][NH2:6]>>[CH3:1][C:2](=[O:3])[NH:6][CH3:5].[OH2:4]"
    rxn_wo_am = remove_atom_mapping(rxn_smiles)
    assert rxn_wo_am == "[CH3][C](=[O])[O].[CH3][NH2]>>[CH3][C](=[O])[NH][CH3].[OH2]"


def test_remove_atom_mapping_cgr_smiles():
    """Test removal of atom mapping from a CGR smiles."""
    rxn_smiles = "{[O:1]|[O+:1]}{=|#}{[C:2]|[C-:2]}1{-|~}[H:5]{~|-}[C:3]{-|~}1#[C:4][H:6]"
    rxn_wo_am = remove_atom_mapping(rxn_smiles)
    assert rxn_wo_am == "{[O]|[O+]}{=|#}{[C]|[C-]}1{-|~}[H]{~|-}[C]{-|~}1#[C][H]"


# def test_remove_non_participating_hydrogens_unbalanced():
#     """
#     Test that the input is returned unchanged when H count changes.
#     """
#     input_rxn = "[CH3][OH]>>[CH3][O]"
#     result = remove_non_participating_hydrogens(input_rxn)
#     assert result == input_rxn


# def test_remove_non_participating_hydrogens_balanced():
#     """
#     Test that non-essential hydrogens are removed when H count is conserved.
#     """
#     input_rxn = "[CH3][OH]>>[CH3][O][H]"
#     result = remove_non_participating_hydrogens(input_rxn)
#     assert result == "[C][OH]>>[C][O][H]"


def test_remove_redundant_square_brackets_rxn_smiles():
    """Test removal of square brackets from a reaction smiles."""
    rxn_smiles = "[O]=[C]([H])[C]#[C][H]>>[H][C]#[C][H].[O+]#[C-]"
    rxn_wo_am = remove_redundant_square_brackets(rxn_smiles)
    assert rxn_wo_am == "O=C([H])C#C[H]>>[H]C#C[H].[O+]#[C-]"


def test_remove_redundant_square_brackets_cgr_smiles():
    """Test removal of square brackets from a CGR smiles."""
    cgr_smiles = "{[O]|[O+]}{=|#}{[C]|[C-]}1{-|~}[H]{~|-}[C]{-|~}1#[C][H]"
    cgr_wo_am = remove_redundant_square_brackets(cgr_smiles)
    assert cgr_wo_am == "{O|[O+]}{=|#}{C|[C-]}1{-|~}[H]{~|-}C{-|~}1#C[H]"


def test_parse_bonds_in_order_from_smiles_simple(simple_smiles):
    """Verify that SMILES-parsed bonds are in the same order they appear in the input."""
    bond_order = parse_bonds_in_order_from_smiles(simple_smiles)
    assert bond_order == {(1, 2): "-", (2, 3): "-", (2, 4): "-", (2, 5): "-"}


def test_parse_bonds_in_order_from_smiles_complex(complex_smiles):
    """Verify that SMILES-parsed bonds are in the same order they appear in the input."""
    bond_order = parse_bonds_in_order_from_smiles(complex_smiles)
    assert bond_order == {
        (1, 2): "-",
        (2, 11): "-",
        (2, 3): "-",
        (3, 4): "-",
        (4, 12): "-",
        (4, 5): "-",
        (5, 13): "-",
        (5, 14): "-",
        (5, 6): "-",
        (6, 15): "-",
        (6, 16): "-",
        (6, 7): "-",
        (7, 2): "-",
        (7, 4): "-",
        (7, 17): "-",
        (1, 8): "-",
        (1, 9): "-",
        (1, 10): "-",
    }


def test_get_atom_map_adjacency_list_from_smiles(simple_smiles):
    """Test that the SMILES atom-adjacency list matches the expected atom mapping."""
    expected = {
        1: [2],
        2: [1, 3, 4, 5],
        3: [2],
        4: [2],
        5: [2],
    }
    result = get_atom_map_adjacency_list_from_smiles(simple_smiles)
    assert result == expected


test_data = [
    (
        "mapped_mol",
        "[C:1](=[O:2])[O:3][C@H:4]([F:5])[B:6]",
        [
            (TokenType.ATOM, "[C:1]"),
            (TokenType.BRANCH_START, "("),
            (TokenType.BOND_TYPE, "="),
            (TokenType.ATOM, "[O:2]"),
            (TokenType.BRANCH_END, ")"),
            (TokenType.ATOM, "[O:3]"),
            (TokenType.ATOM, "[C@H:4]"),
            (TokenType.BRANCH_START, "("),
            (TokenType.ATOM, "[F:5]"),
            (TokenType.BRANCH_END, ")"),
            (TokenType.ATOM, "[B:6]"),
        ],
    ),
    (
        "unmapped_mol",
        "C(=O)OC(F)B",
        [
            (TokenType.ATOM, "C"),
            (TokenType.BRANCH_START, "("),
            (TokenType.BOND_TYPE, "="),
            (TokenType.ATOM, "O"),
            (TokenType.BRANCH_END, ")"),
            (TokenType.ATOM, "O"),
            (TokenType.ATOM, "C"),
            (TokenType.BRANCH_START, "("),
            (TokenType.ATOM, "F"),
            (TokenType.BRANCH_END, ")"),
            (TokenType.ATOM, "B"),
        ],
    ),
    (
        "partially_mapped_mol",
        "[C:1](=O)O[C:2](F)B",
        [
            (TokenType.ATOM, "[C:1]"),
            (TokenType.BRANCH_START, "("),
            (TokenType.BOND_TYPE, "="),
            (TokenType.ATOM, "O"),
            (TokenType.BRANCH_END, ")"),
            (TokenType.ATOM, "O"),
            (TokenType.ATOM, "[C:2]"),
            (TokenType.BRANCH_START, "("),
            (TokenType.ATOM, "F"),
            (TokenType.BRANCH_END, ")"),
            (TokenType.ATOM, "B"),
        ],
    ),
    (
        "ez_stereochemistry",
        "F/C=C/B",
        [
            (TokenType.ATOM, "F"),
            (TokenType.EZSTEREO, "/"),
            (TokenType.ATOM, "C"),
            (TokenType.BOND_TYPE, "="),
            (TokenType.ATOM, "C"),
            (TokenType.EZSTEREO, "/"),
            (TokenType.ATOM, "B"),
        ],
    ),
    (
        "chirality",
        "Cl[C@](N)(Br)F",
        [
            (TokenType.ATOM, "Cl"),
            (TokenType.ATOM, "[C@]"),
            (TokenType.BRANCH_START, "("),
            (TokenType.ATOM, "N"),
            (TokenType.BRANCH_END, ")"),
            (TokenType.BRANCH_START, "("),
            (TokenType.ATOM, "Br"),
            (TokenType.BRANCH_END, ")"),
            (TokenType.ATOM, "F"),
        ],
    ),
    (
        "mol_with_ring",
        "C1=CC(=O)C=C1",
        [
            (TokenType.ATOM, "C"),
            (TokenType.RING_NUM, "1"),
            (TokenType.BOND_TYPE, "="),
            (TokenType.ATOM, "C"),
            (TokenType.ATOM, "C"),
            (TokenType.BRANCH_START, "("),
            (TokenType.BOND_TYPE, "="),
            (TokenType.ATOM, "O"),
            (TokenType.BRANCH_END, ")"),
            (TokenType.ATOM, "C"),
            (TokenType.BOND_TYPE, "="),
            (TokenType.ATOM, "C"),
            (TokenType.RING_NUM, "1"),
        ],
    ),
    (
        "two_digit_ring_numbers",
        "C%10=CC%10",
        [
            (TokenType.ATOM, "C"),
            (TokenType.RING_NUM, "%10"),
            (TokenType.BOND_TYPE, "="),
            (TokenType.ATOM, "C"),
            (TokenType.ATOM, "C"),
            (TokenType.RING_NUM, "%10"),
        ],
    ),
    (
        "disconnected_mols",
        "[CH3:1].[CH3:2]",
        [
            (TokenType.ATOM, "[CH3:1]"),
            (TokenType.BOND_TYPE, "."),
            (TokenType.ATOM, "[CH3:2]"),
        ],
    ),
    (
        "isotopes_and_charges",
        "[13CH3+][12CH3-]",
        [(TokenType.ATOM, "[13CH3+]"), (TokenType.ATOM, "[12CH3-]")],
    ),
    (
        "two_character_elements",
        "CClC=BrC",
        [
            (TokenType.ATOM, "C"),
            (TokenType.ATOM, "Cl"),
            (TokenType.ATOM, "C"),
            (TokenType.BOND_TYPE, "="),
            (TokenType.ATOM, "Br"),
            (TokenType.ATOM, "C"),
        ],
    ),
    (
        "aromatic_carbons",
        "c1ccccc1",
        [
            (TokenType.ATOM, "c"),
            (TokenType.RING_NUM, "1"),
            (TokenType.ATOM, "c"),
            (TokenType.ATOM, "c"),
            (TokenType.ATOM, "c"),
            (TokenType.ATOM, "c"),
            (TokenType.ATOM, "c"),
            (TokenType.RING_NUM, "1"),
        ],
    ),
]


@pytest.mark.parametrize("test_name, smiles, expected_tokens", test_data)
def test_tokenize_valid_smiles(test_name: str, smiles: str, expected_tokens: List[Tuple[TokenType, str]]):
    """Tests the `_tokenize()` function with various valid SMILES strings."""
    actual_tokens = [(t[0], t[2]) for t in _tokenize(smiles)]
    assert (
        actual_tokens == expected_tokens
    ), f"Test '{test_name}' failed: \nExpected: {expected_tokens}\nGot:      {actual_tokens}"


# error test cases
error_test_data = [
    ("unmatched_bracket_error", "C[CH2", ValueError),
    ("malformed_ring_number_error", "C%1", ValueError),
]


@pytest.mark.parametrize("test_name, smiles, expected_exception_type", error_test_data)
def test_tokenize_invalid_smiles(test_name: str, smiles: str, expected_exception_type: type):
    """Tests the `_tokenize()` function with invalid SMILES strings to ensure correct error handling."""
    with pytest.raises(expected_exception_type):
        list(_tokenize(smiles))


def test_update_all_atom_stereo_without_chirality_change(simple_mol):
    """Check that stereo tags stay the same when number of flips is even."""
    chiral_atom = simple_mol.GetAtomWithIdx(
        Chem.FindMolChiralCenters(simple_mol, includeUnassigned=True)[0][0]
    )
    chiral_tag_before = chiral_atom.GetChiralTag()

    update_all_atom_stereo(
        simple_mol,
        "[C:1][C@:2]([Cl:3])([Br:4])[I:5]",
        "[C:1][C@:2]([I:5])([Cl:3])[Br:4]",
    )  # 2 flips

    chiral_tag_after = chiral_atom.GetChiralTag()
    assert chiral_tag_before == chiral_tag_after


def test_update_all_atom_stereo_with_chirality_change(simple_mol):
    """Check that stereo tags changes the same when number of flips is odd."""
    chiral_atom = simple_mol.GetAtomWithIdx(
        Chem.FindMolChiralCenters(simple_mol, includeUnassigned=True)[0][0]
    )
    chiral_tag_before = chiral_atom.GetChiralTag()

    update_all_atom_stereo(
        simple_mol,
        "[C:1][C@:2]([Cl:3])([Br:4])[I:5]",
        "[Cl:3][C@:2]([C:1])([Br:4])[I:5]",
    )  # 1 flip

    chiral_tag_after = chiral_atom.GetChiralTag()
    assert chiral_tag_before != chiral_tag_after


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("[O:1]([H:2])[H:3]", True),
        ("[CH3:1]=[O:2]", False),
    ],
)
def test_includes_individually_mapped_hydrogens(smiles, expected):
    """Test detection of individually mapped hydrogens in SMILES."""
    assert includes_individually_mapped_hydrogens(smiles) == expected


@pytest.mark.parametrize(
    "l1, l2, expected",
    [
        ([1, 2, 3], [1, 2, 3], True),  # identical (0 swaps)
        ([1, 2, 3], [2, 1, 3], False),  # 1 swap
        ([1, 2, 3], [3, 1, 2], True),  # 2 swaps
        ([1, 2, 3, 4], [2, 1, 4, 3], True),  # even-length list
        ([], [], True),  # empty lists
    ],
)
def test_is_num_permutations_even(l1, l2, expected):
    """Test `is_num_permutations_even` with various list pairs."""
    assert is_num_permutations_even(l1, l2) == expected


@pytest.mark.parametrize(
    "l1, l2, expected1, expected2",
    [
        ([1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]),
        ([1, 2, 3], [1, 2, 4, 3], [1, 2, 3], [1, 2, 3]),
        ([1, 2, 3, 4], [3, 4, 5, 1], [1, 3, 4], [3, 4, 1]),
        ([1, 2], [3, 4], [], []),
        ([1, 2, 2, 3], [2, 3, 3, 4], [2, 2, 3], [2, 3, 3]),
        ([], [1, 2], [], []),
        ([1, 2], [], [], []),
        ([], [], [], []),
    ],
)
def test_common_elements_preserving_order(l1, l2, expected1, expected2):
    """Check that common elements of two lists are returned in correct order."""
    result1, result2 = common_elements_preserving_order(l1, l2)
    assert result1 == expected1
    assert result2 == expected2


canonicalization_test_cases = [
    # 1. simple test case
    pytest.param(
        "C(C)C>>CCO",
        "CCC>>CCO",
        id="non_canonical_smiles",
    ),
    # 2. with atom mapping
    pytest.param(
        "[CH2:2]([CH3:3])[CH3:1]>>[CH3:3][CH2:2][OH:1]",
        "[CH3:1][CH2:2][CH3:3]>>[OH:1][CH2:2][CH3:3]",
        id="with_atom_mapping",
    ),
    # 3. more complex rxn smiles with explicit hydrogens
    pytest.param(
        "[C:3](#[C:4][H:9])[C@:2]([H:8])([O:1][H:7])[C:5]#[N:6]>>[C:4]([H:9])#[C:3][C@@:2]1([H:8])[N:6]=[C:5]1[O:1][H:7]",
        "[O:1]([C@@:2]([C:3]#[C:4][H:9])([C:5]#[N:6])[H:8])[H:7]>>[O:1]([C:5]1=[N:6][C@@:2]1([C:3]#[C:4][H:9])[H:8])[H:7]",
        id="complex_with_explicit_hydrogens",
    ),
    # 4. already canonical, should remain unchanged
    pytest.param(
        "[CH3:1][CH2:2][CH3:3]>>[OH:1][CH2:2][CH3:3]",
        "[CH3:1][CH2:2][CH3:3]>>[OH:1][CH2:2][CH3:3]",
        id="already_canonical",
    ),
]


@pytest.mark.parametrize("test_input_rxn, expected_output_rxn", canonicalization_test_cases)
def test_canonicalize_variants(test_input_rxn, expected_output_rxn):
    """Tests that various SMILES formats are correctly canonicalized."""
    assert canonicalize(test_input_rxn) == expected_output_rxn


def test_canonicalize_malformed_string():
    """Tests that canonicalize raises a ValueError for a malformed reaction string."""
    with pytest.raises(ValueError):
        canonicalize("just one molecule")


def test_map_reac_to_prod(reaction_mols):
    """Tests that reactant indices are correctly mapped to product indices."""
    mol_reac, mol_prod = reaction_mols

    expected_map = {0: 0, 1: 1, 2: 2, 3: 5, 4: 4, 5: 3}
    assert map_reac_to_prod(mol_reac, mol_prod) == expected_map


def test_map_reac_to_prod_with_key_error(reaction_mols):
    """Tests that reactant indices are correctly mapped to product indices."""
    _, mol_prod = reaction_mols

    mol_reac_bad = Chem.MolFromSmiles("[C:1][H:99]", sanitize=False)
    with pytest.raises(KeyError):
        map_reac_to_prod(mol_reac_bad, mol_prod)


def test_get_atom_map_num(simple_mol):
    """Check that each atom's map number matches its index + 1."""
    num_atoms = simple_mol.GetNumAtoms()
    for idx in range(num_atoms):
        map_num = get_atom_map_num(simple_mol, idx)
        assert map_num == idx + 1


def test_get_atom_by_map_num(simple_mol):
    """Check that atoms are correctly retrieved by their map number."""
    num_atoms = simple_mol.GetNumAtoms()
    for idx in range(num_atoms):
        atom = get_atom_by_map_num(simple_mol, atom_map_num=idx + 1)
        assert atom.GetIdx() == idx


def test_get_list_of_atom_map_numbers():
    """Check that all atom map numbers are extracted in SMILES traversal order."""
    smiles = "[C:4]([H:9])#[C:3][C@@:2]1([H:8])[N:6]=[C:5]1[O:1][H:7]"
    map_nums = get_list_of_atom_map_numbers(smiles)
    assert map_nums == [4, 9, 3, 2, 8, 6, 5, 1, 7]


def test_get_atom_indices_and_smarts(simple_mol):
    """Check that each atom index is paired with its correct SMARTS pattern."""
    indexed_smarts = get_atom_indices_and_smarts(simple_mol)
    assert indexed_smarts == [
        (0, "[C:1]"),
        (1, "[C@:2]"),
        (2, "[Cl:3]"),
        (3, "[Br:4]"),
        (4, "[I:5]"),
    ]


def test_get_bond_idx(simple_mol):
    """Check that the bond index is returned for connected atoms, otherwise None."""
    assert get_bond_idx(simple_mol, 0, 1) == 0

    assert get_bond_idx(simple_mol, 1, 2) == 1
    assert get_bond_idx(simple_mol, 1, 3) == 2
    assert get_bond_idx(simple_mol, 1, 4) == 3

    assert get_bond_idx(simple_mol, 2, 3) is None
