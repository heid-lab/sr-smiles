from typing import List, Tuple

import pytest
from rdkit import Chem

from sr_smiles.chem_utils.smiles_utils import (
    TokenType,
    _tokenize,
    canonicalize,
    extract_chiral_tag_by_atom_map_num,
    get_atom_map_adjacency_list_from_smiles,
    get_fragment_permutations,
    get_list_of_atom_map_numbers,
    is_kekule,
    parse_bonds_in_order_from_smiles,
    remove_aromatic_bonds,
    remove_atom_mapping,
    remove_redundant_brackets,
    remove_redundant_brackets_and_hydrogens,
)


@pytest.fixture
def simple_smiles() -> str:
    """Provides a simple atom-mapped SMILES string with explicit hydrogens."""
    return "[C:1][C@:2]([Cl:3])([Br:4])[I:5]"


@pytest.fixture
def complex_smiles() -> Tuple[Chem.Mol, Chem.Mol]:
    """Provides a complex smiles, including multiple rings and stereochemistry."""
    smiles = "[C:1]([C@@:2]1([H:11])[O:3][C@@:4]2([H:12])[C:5]([H:13])([H:14])[C:6]([H:15])([H:16])[C@@:7]12[H:17])([H:8])([H:9])[H:10]"  # noqa: E501
    return smiles


@pytest.mark.parametrize(
    "smi_sr, expected",
    [
        # basic redundant bracket removal
        ("[C][C]", "CC"),
        ("[O][H]", "O[H]"),
        ("[C][C][O]", "CCO"),
        ("[CH3][CH2][OH]>>{[CH3]|[CH3]}", "CCO>>C"),
        # hydrogens removal from explicit patterns
        ("[CH3]", "C"),
        ("[CH2]", "C"),
        ("[NH3]", "N"),
        ("[OH2]", "O"),
        ("[SH]", "S"),
        ("[nH]", "[nH]"),
        # preserving annotation/charge/isotope
        ("[CH3+]", "[CH3+]"),
        ("[13CH]", "[13CH]"),
        ("Fe+2].[Cl-].[Cl-]", "Fe+2].[Cl-].[Cl-]"),
        # collapsing {X|X} pairs
        ("{C|C}", "C"),
        ("C{O|O}N", "CON"),
    ],
)
def test_remove_redundant_brackets_and_hydrogens(smi_sr, expected):
    """Check core functionality for standard cases and hydrogens simplifications."""
    result = remove_redundant_brackets_and_hydrogens(smi_sr)
    assert result == expected


@pytest.mark.parametrize(
    "smi_sr, expected",
    [
        # basic redundant bracket removal
        ("[C][C]", "CC"),
        ("[O][H]", "O[H]"),
        ("[C][C][O]", "CCO"),
        ("[CH3][CH2][OH]>>{[CH3]|[CH3]}", "[CH3][CH2][OH]>>{[CH3]|[CH3]}"),
        # hydrogens removal from explicit patterns
        ("[CH3]", "[CH3]"),
        ("[C]", "C"),
        ("[OH]", "[OH]"),
        ("[O]", "O"),
        ("[nH]", "[nH]"),
        # preserving annotation/charge/isotope
        ("[CH3+]", "[CH3+]"),
        ("[13CH]", "[13CH]"),
        ("Fe+2].[Cl-].[Cl-]", "Fe+2].[Cl-].[Cl-]"),
        # collapsing {X|X} pairs
        ("{C|C}", "C"),
        ("C{O|O}N", "CON"),
    ],
)
def test_remove_redundant_brackets(smi_sr, expected):
    """Check core functionality for standard cases and hydrogens simplifications."""
    result = remove_redundant_brackets(smi_sr)
    assert result == expected


@pytest.mark.parametrize(
    "smi, expected",
    [
        ("c1:c:c:[nH]:c:1", "c1cc[nH]c1"),
        ("[c:1]1:[c:2]:[c:3]:[nH:4]:[c:5]:1", "[c:1]1[c:2][c:3][nH:4][c:5]1"),
    ],
)
def test_remove_aromatic_bonds(smi, expected):
    """Verify remove_aromatic_bonds removes colons from aromatic SMILES correctly."""
    res = remove_aromatic_bonds(smi)
    assert res == expected, f"For {smi}: expected {expected}, got {res}"


def test_remove_atom_mapping_rxn_smiles():
    """Test removal of atom mapping from a reaction smiles."""
    rxn_smiles = "[CH3:1][C:2](=[O:3])[O:4].[CH3:5][NH2:6]>>[CH3:1][C:2](=[O:3])[NH:6][CH3:5].[OH2:4]"
    rxn_wo_am = remove_atom_mapping(rxn_smiles)
    assert rxn_wo_am == "[CH3][C](=[O])[O].[CH3][NH2]>>[CH3][C](=[O])[NH][CH3].[OH2]"


def test_remove_atom_mapping_sr_smiles():
    """Test removal of atom mapping from an sr-SMILES."""
    rxn_smiles = "{[O:1]|[O+:1]}{=|#}{[C:2]|[C-:2]}1{-|~}[H:5]{~|-}[C:3]{-|~}1#[C:4][H:6]"
    rxn_wo_am = remove_atom_mapping(rxn_smiles)
    assert rxn_wo_am == "{[O]|[O+]}{=|#}{[C]|[C-]}1{-|~}[H]{~|-}[C]{-|~}1#[C][H]"


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


@pytest.mark.parametrize(
    "smi, atom_map_num, expected",
    [
        ("[C@H:1](F)Cl", 1, "@"),  # single chirality tag
        ("[C@@H:2](Br)Cl", 2, "@@"),  # double chirality tag
        ("[C:3](F)(Cl)Br", 3, ""),  # no chirality tag
        ("[C@H:1](F)[C@:2](Cl)Br", 2, "@"),  # multiple chiral centers
    ],
)
def test_extract_chiral_tag_by_atom_map_num(smi, atom_map_num, expected):
    """Ensure correct extraction of @, @@, or '' for a given atom map number."""
    result = extract_chiral_tag_by_atom_map_num(smi, atom_map_num)
    assert result == expected, f"For {smi} and atom map {atom_map_num}, expected '{expected}', got '{result}'"


@pytest.mark.parametrize(
    "smi, expected",
    [
        ("CC", [[0]]),
        (
            "A.B.C",
            [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
            ],
        ),
    ],
)
def test_get_fragment_permutations(smi, expected):
    """Ensure get_fragment_permutations returns correct index permutations based on fragment count."""
    result = get_fragment_permutations(smi)

    # compare as sets since permutations can appear in any order
    assert set(tuple(x) for x in result) == set(tuple(x) for x in expected)
    assert all(isinstance(p, list) for p in result)


def test_get_list_of_atom_map_numbers():
    """Check that all atom map numbers are extracted in SMILES traversal order."""
    smiles = "[C:4]([H:9])#[C:3][C@@:2]1([H:8])[N:6]=[C:5]1[O:1][H:7]"
    map_nums = get_list_of_atom_map_numbers(smiles)
    assert map_nums == [4, 9, 3, 2, 8, 6, 5, 1, 7]


@pytest.mark.parametrize(
    "rxn_smi, expected",
    [
        # positive cases
        ("[C:1][C:2]>>[C:1]=[C:2]", True),
        ("[C:1]([H])([H])[C:2]>>[C:1]=[O:2]", True),
        ("CC[O:1]>>CC[O:1][H]", True),
        ("CCO>>CC=O", True),
        # negative cases
        ("[c:1]1ccccc1>>[C:1]1CCCCC1", False),
        ("[n:1]1ccccc1>>[N:1]1CCCCC1", False),
        ("[C:1][c:2]>>[C:1][C:2]", False),
    ],
)
def test_is_kekule(rxn_smi, expected):
    """Test is_kekule across Kekulé, aromatic, and malformed reaction SMILES."""
    assert is_kekule(rxn_smi) is expected


def test_is_kekule_invariant_to_extra_annotations():
    """Ensure it ignores numeric tags, charges, isotopes, etc., focusing only on element case."""
    smi = "[C@@H:1]([O-:2])[N+:3](=O)[O-:4]>>[C@@H:1]([OH:2])[NH2:3]"
    assert is_kekule(smi)


# def test_get_list_of_atom_map_numbers_from_sr_smiles():
#     """Check that all atom map numbers are extracted in SMILES traversal order."""
#     sr = "{[C:4]|[CH:4]}([H:9])#[C:3][C@@:2]1([H:8]){[N:6]|[N+:6]}{=|-}[C:5]1[O:1][H:7]"
#     map_nums = get_list_of_atom_map_numbers_from_sr_smiles(sr)
#     assert map_nums == [4, 9, 3, 2, 8, 6, 5, 1, 7]
