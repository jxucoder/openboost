import numpy as np
import pytest
from benchmarks.v1.ranking import groups, validate


def test_group_weights_have_explicit_identity_and_sizes():
    ids, sizes, weights = groups(np.array([8, 8, 3, 3, 3]), 5, [2, 0.5])
    assert ids.tolist() == [8, 3]
    assert sizes.tolist() == [2, 3]
    assert weights.tolist() == [2, 0.5]


@pytest.mark.parametrize(
    "query,weight", [([1, 2, 1], None), ([1, 1, 2], [1, 1, 1]), ([1, 1, 2], [0, 0])]
)
def test_fragmented_groups_and_invalid_query_weights_fail(query, weight):
    with pytest.raises(ValueError):
        groups(query, 3, weight)


def test_query_overlap_and_row_weights_are_rejected():
    a = dict(
        x_train=np.ones((3, 2)),
        x_validation=np.ones((2, 2)),
        query_train=np.array([1, 1, 2]),
        query_validation=np.array([2, 2]),
        y_train=np.array([0, 1, 2]),
    )
    with pytest.raises(ValueError, match="overlap"):
        validate(a, None)
    a["query_validation"] = np.array([3, 3])
    a["weight_train"] = np.ones(3)
    with pytest.raises(ValueError, match="row weights"):
        validate(a, None)
