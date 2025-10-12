import pytest

from cgr_smiles.chem_utils.list_utils import (
    common_elements_preserving_order,
    is_num_permutations_even,
    mask_nonshared_with_neg1,
)


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


def test_mask_nonshared_with_neg1():
    """Test that values that are not shared between lists are replaced with -1."""
    list1 = [1, 2, 2, 3]
    list2 = [2, 2, 5, 3]
    new1, new2 = mask_nonshared_with_neg1(list1, list2)
    assert new1 == [-1, 2, 2, 3]
    assert new2 == [2, 2, -1, 3]


def test_mask_nonshared_with_neg1_no_shared_values():
    """Test that no values are replaced, if no values are shared."""
    list1 = [1, 1, 1]
    list2 = [2, 2, 2]
    new1, new2 = mask_nonshared_with_neg1(list1, list2)
    assert new1 == [-1, -1, -1]
    assert new2 == [-1, -1, -1]


def test_mask_nonshared_with_neg1_all_shared_same_counts():
    """If all values are shared, lists remain unchanged."""
    list1 = [1, 2, 3]
    list2 = [1, 2, 3]
    new1, new2 = mask_nonshared_with_neg1(list1, list2)
    assert new1 == [1, 2, 3]
    assert new2 == [1, 2, 3]


def test_mask_nonshared_with_neg1_empty_lists():
    """Handles empty input lists without errors."""
    new1, new2 = mask_nonshared_with_neg1([], [])
    assert new1 == []
    assert new2 == []
