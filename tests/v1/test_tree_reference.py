"""Hand-worked splits, different growth policies, and two-round traces."""

import numpy as np
import pytest

from tests.v1.reference.tree import best_split, boost_squared, enumerate_splits, fit_tree


def test_half_gain_and_regularized_leaves():
    x, g = np.array([[0.0], [0.0], [1.0], [1.0]]), [2.0, 2.0, -2.0, -2.0]
    split = best_split(enumerate_splits(x, g, np.ones(4)))
    assert split.left == (0, 1) and split.right == (2, 3)
    assert split.gain == pytest.approx(16 / 3)
    tree = fit_tree(x, g, np.ones(4), max_depth=1)
    np.testing.assert_allclose(tree.predict(x), [-4 / 3, -4 / 3, 4 / 3, 4 / 3])
    assert best_split(enumerate_splits(x, g, np.ones(4), split_penalty=6.0)) is None


def test_two_rounds_recompute_gradient_and_accumulate_coefficients_once():
    x, y = np.array([[0.0], [0.0], [1.0], [1.0]]), [-2.0, -2.0, 2.0, 2.0]
    trace = boost_squared(x, y, rounds=2, learning_rate=0.1, max_depth=1)
    assert trace.base == 0.0
    first, second = trace.steps
    np.testing.assert_array_equal(first.gradient, [2.0, 2.0, -2.0, -2.0])
    np.testing.assert_allclose(first.raw_after, [-2 / 15, -2 / 15, 2 / 15, 2 / 15])
    np.testing.assert_allclose(second.gradient, [28 / 15, 28 / 15, -28 / 15, -28 / 15])
    np.testing.assert_allclose(second.tree.predict(x), [-56 / 45, -56 / 45, 56 / 45, 56 / 45])
    np.testing.assert_allclose(second.raw_after, [-58 / 225, -58 / 225, 58 / 225, 58 / 225])
    np.testing.assert_allclose(trace.predict(x), second.raw_after)
    assert second.loss_after < first.loss_after < first.loss_before


@pytest.mark.parametrize("missing_gradient,missing_left", [(2.0, True), (-2.0, False)])
def test_missing_direction_is_chosen_from_routed_rows(missing_gradient, missing_left):
    x = np.array([[0.0], [1.0], [np.nan]])
    split = best_split(enumerate_splits(x, [2.0, -2.0, missing_gradient], np.ones(3)))
    assert split.condition.missing_left is missing_left
    assert 2 in (split.left if missing_left else split.right)
    assert set(split.left).isdisjoint(split.right)
    assert sorted(split.left + split.right) == [0, 1, 2]


def test_exact_ties_choose_first_feature_threshold_and_missing_right():
    x = np.column_stack([np.arange(3), np.arange(3)])
    split = best_split(enumerate_splits(x, [-2.0, 0.0, 2.0], np.ones(3)))
    assert (split.condition.feature, split.condition.threshold, split.condition.missing_left) == (
        0,
        0,
        False,
    )


def test_missing_only_split_and_all_missing_feature():
    split = best_split(enumerate_splits([[0.0], [np.nan]], [-2.0, 2.0], [1.0, 1.0]))
    assert split.left == (0,) and split.right == (1,)
    assert best_split(enumerate_splits([[np.nan], [np.nan]], [-2.0, 2.0], [1.0, 1.0])) is None


def test_zero_weight_or_zero_curvature_child_is_not_feasible():
    for kwargs in ({"weight": [1.0, 0.0]}, {"curvature": [1.0, 0.0]}):
        h = kwargs.pop("curvature", [1.0, 1.0])
        assert best_split(enumerate_splits([[0.0], [1.0]], [-2.0, 2.0], h, **kwargs)) is None


def test_empty_node_has_no_candidate():
    assert enumerate_splits([[0.0], [1.0]], [-2.0, 2.0], [1.0, 1.0], rows=()) == ()


def test_integer_weights_match_duplicated_rows_with_fixed_bins():
    x, y, w = np.array([[0.0], [1.0], [2.0]]), np.array([-2.0, 1.0, 3.0]), np.array([2, 1, 3])
    weighted = boost_squared(x, y, weight=w, rounds=2)
    repeated = boost_squared(np.repeat(x, w, axis=0), np.repeat(y, w), rounds=2)
    np.testing.assert_allclose(weighted.predict(x), repeated.predict(x), rtol=1e-12, atol=1e-12)


def test_cohort_constraint_changes_best_split_and_is_not_training_weight():
    x, g = np.arange(1, 7)[:, None], [-6.0, 1.0, 1.0, 1.0, 1.0, 2.0]
    info = np.eye(2)[[0, 1, 0, 1, 0, 1]]
    plain = best_split(enumerate_splits(x, g, np.ones(6)))
    constrained = best_split(
        enumerate_splits(x, g, np.ones(6), information=info, min_information=1.0)
    )
    assert plain.condition.threshold == 1 and constrained.condition.threshold == 2
    assert constrained.left == (0, 1) and constrained.right == (2, 3, 4, 5)
    # Zero train weight on row 0 must not erase its independent information mass.
    candidates = enumerate_splits(
        x,
        g,
        np.ones(6),
        weight=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        information=info,
        min_information=1.0,
    )
    assert any(c.condition.threshold == 2 for c in candidates)
    separated = np.eye(2)[[0, 0, 0, 1, 1, 1]]
    assert (
        best_split(enumerate_splits(x, g, np.ones(6), information=separated, min_information=1.0))
        is None
    )


def grid():
    return np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])


def test_best_first_can_grow_deeper_before_a_shallower_leaf():
    x, y = grid(), np.array([-101.0, -99.0, -101.0, -99.0, 80.0, 100.0, 100.0, 120.0])
    depthwise = fit_tree(x, -y, np.ones(8), reg_lambda=0.0, max_depth=3, max_leaves=4)
    best_first = fit_tree(
        x, -y, np.ones(8), reg_lambda=0.0, max_depth=3, max_leaves=4, policy="best_first"
    )
    assert depthwise.nodes[0].condition.feature == best_first.nodes[0].condition.feature == 0
    assert depthwise.nodes[1].condition is not None
    assert best_first.nodes[1].condition is None
    assert best_first.nodes[3].condition is not None
    assert (
        sum(n.condition is None for n in depthwise.nodes)
        == sum(n.condition is None for n in best_first.nodes)
        == 4
    )


def test_symmetric_aggregates_common_candidates_not_node_winners():
    x, y = grid(), np.array([-12.0, -12.0, -8.0, -8.0, 6.0, 14.0, 6.0, 14.0])
    regular = fit_tree(x, -y, np.ones(8), reg_lambda=0.0, max_depth=2)
    symmetric = fit_tree(x, -y, np.ones(8), reg_lambda=0.0, max_depth=2, policy="symmetric")
    assert regular.nodes[1].condition.feature == 1
    assert regular.nodes[2].condition.feature == 2
    assert symmetric.nodes[1].condition == symmetric.nodes[2].condition
    assert symmetric.nodes[1].condition.feature == 2
    np.testing.assert_allclose(
        symmetric.predict(x), [-10.0, -10.0, -10.0, -10.0, 6.0, 14.0, 6.0, 14.0]
    )


def test_symmetric_stops_when_full_level_does_not_fit_budget():
    tree = fit_tree(
        grid(),
        [-12.0, -12.0, -8.0, -8.0, 6.0, 14.0, 6.0, 14.0],
        np.ones(8),
        reg_lambda=0.0,
        policy="symmetric",
        max_leaves=3,
    )
    assert len(tree.nodes) == 3


@pytest.mark.parametrize("policy", ["depthwise", "best_first", "symmetric"])
def test_constant_target_and_root_only_budget(policy):
    trace = boost_squared([[0.0], [1.0]], [3.0, 3.0], policy=policy)
    np.testing.assert_array_equal(trace.predict([[0.0], [1.0]]), [3.0, 3.0])
    assert all(len(step.tree.nodes) == 1 for step in trace.steps)
    tree = fit_tree([[0.0], [1.0]], [2.0, -2.0], [1.0, 1.0], max_leaves=1, policy=policy)
    assert len(tree.nodes) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_depth": -1},
        {"max_leaves": 0},
        {"policy": "unknown"},
        {"split_penalty": -1.0},
        {"min_child_h": -1.0},
        {"information": [[1.0], [-1.0]]},
        {"min_information": 1.0},
    ],
)
def test_invalid_tree_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        fit_tree([[0.0], [1.0]], [2.0, -2.0], [1.0, 1.0], **kwargs)


@pytest.mark.parametrize("x", [[[np.inf], [0.0]], [[0.5], [1.0]], [[-1.0], [0.0]], [0.0, 1.0]])
def test_invalid_fixed_numeric_bins_rejected(x):
    with pytest.raises(ValueError, match="bins"):
        fit_tree(x, [2.0, -2.0], [1.0, 1.0])


def test_child_leaves_use_actual_routed_rows_and_trace_owns_values():
    x, y = grid(), np.array([-12.0, -12.0, -8.0, -8.0, 6.0, 14.0, 6.0, 14.0])
    trace = boost_squared(x, y, reg_lambda=0.0)
    tree = trace.steps[0].tree
    for node in tree.nodes:
        if node.condition is None:
            assert node.value == pytest.approx(sum(y[i] for i in node.rows) / len(node.rows))
    expected = trace.predict(x)
    y[:] = 1000.0
    x[:] = 1000
    np.testing.assert_array_equal(trace.predict(grid()), expected)


def test_minimum_curvature_and_prediction_schema_are_enforced():
    x = [[0.0], [0.0], [1.0], [1.0]]
    assert (
        best_split(enumerate_splits(x, [2.0, 2.0, -2.0, -2.0], np.ones(4), min_child_h=2.1)) is None
    )
    tree = fit_tree(x, [2.0, 2.0, -2.0, -2.0], np.ones(4))
    with pytest.raises(ValueError, match="feature"):
        tree.predict([[0.0, 1.0]])


@pytest.mark.parametrize(
    "kwargs", [{"rounds": -1}, {"learning_rate": np.nan}, {"learning_rate": -0.1}]
)
def test_invalid_boosting_options_rejected(kwargs):
    with pytest.raises(ValueError):
        boost_squared([[0.0], [1.0]], [-1.0, 1.0], **kwargs)
