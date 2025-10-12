from collections import Counter
from typing import List, Tuple


def is_num_permutations_even(l1: list, l2: list) -> bool:
    """Determines if the permutation to transform list `l1` into `l2` involves an even min number of swaps.

    Args:
        l1 (list): The original list of elements.
        l2 (list): The target list of elements, a permutation of `l1`.

    Returns:
        bool: True if the permutation from `l1` to `l2` is even (i.e., involves an even number of swaps),
              False if it is odd.
    """
    target_map = {val: i for i, val in enumerate(l2)}

    visited_indices = [False] * len(l1)
    num_cycles = 0

    for i in range(len(l1)):
        if not visited_indices[i]:
            num_cycles += 1
            current_idx = i
            while not visited_indices[current_idx]:
                visited_indices[current_idx] = True
                element_in_list1 = l1[current_idx]
                current_idx = target_map[element_in_list1]

    num_swaps = len(l1) - num_cycles
    return num_swaps % 2 == 0


def common_elements_preserving_order(list1: list, list2: list) -> Tuple[list, list]:
    """Returns the common elements between two lists, preserving the order of each original list.

    Args:
        list1 (list): The first list of elements.
        list2 (list): The second list of elements.

    Returns:
        Tuple[list, list]: Two lists containing only the common elements from `list1` and `list2`,
        respectively, with their original order maintained.
    """
    set2 = set(list2)
    set1 = set(list1)
    filtered1 = [x for x in list1 if x in set2]
    filtered2 = [x for x in list2 if x in set1]
    return filtered1, filtered2


def mask_nonshared_with_neg1(list1: List[int], list2: List[int]) -> Tuple[List[int], List[int]]:
    """Masks elements not shared one-to-one between two lists with -1.

    Each element is kept only as many times as it appears in both lists
    (based on the smaller count between them). All other occurrences are
    replaced by -1, preserving list length and order.

    Args:
        list1 (List[int]): First list of elements.
        list2 (List[int]): Second list of elements.

    Returns:
        Tuple[List[int], List[int], List[int], List[int]]:
            - new1 (List[int]): Copy of `list1` with unmatched elements set to -1.
            - new2 (List[int]): Copy of `list2` with unmatched elements set to -1.

    Example:
        >>> mask_nonshared_with_neg1([1, 2, 2, 3], [2, 2, 5, 3])
        ([ -1, 2, 2, 3 ], [ 2, 2, -1, 3])
    """
    c1, c2 = Counter(list1), Counter(list2)
    shared_counts = {k: min(c1[k], c2[k]) for k in c1 & c2}

    def mask_and_index(seq, shared):
        """Return masked list and indices of kept values."""
        remaining = shared.copy()
        masked = []
        for i, x in enumerate(seq):
            if remaining.get(x, 0) > 0:
                masked.append(x)
                remaining[x] -= 1
            else:
                masked.append(-1)
        return masked

    new1 = mask_and_index(list1, shared_counts)
    new2 = mask_and_index(list2, shared_counts)

    return new1, new2
