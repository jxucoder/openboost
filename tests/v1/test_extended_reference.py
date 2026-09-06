"""Independent mathematical counterexamples for A4/A5/A6."""

import numpy as np
import pytest

from .reference.quantile import fit_quantile_tree, pinball, weighted_quantile
from .reference.ranking import pairwise, query_ndcg
from .reference.tree import fit_tree
from .reference.vector import TargetScale, fit_vector_stump


def test_pair_hand_normalization_and_query_isolation():
    result = pairwise(
        [0, 0, 0, 0], [2, 0, 1, 1], ["a", "a", "b", "b"], query_weight={"a": 3, "b": 7}
    )
    assert result.loss == pytest.approx(3 * np.log(2))
    np.testing.assert_allclose(result.gradient, [-1.5, 1.5, 0, 0])
    np.testing.assert_allclose(result.curvature, [0.75, 0.75, 0, 0])
    assert result.pairs == ((0, 1, 3.0),)
    assert pairwise([0, 0, 0], [2, 1, 0], [0, 0, 0]).loss == pytest.approx(np.log(2))


def test_pair_gradient_finite_difference_shift_and_weights():
    raw = np.array([0.2, -0.4, 1.1, 2.0])
    rel = [2, 0, 1, 0]
    groups = [0, 0, 1, 1]
    opts = {"query_weight": {0: 2.0, 1: 0.5}, "pair_weight": {(0, 1): 3.0, (2, 3): 2.0}}
    result = pairwise(raw, rel, groups, **opts)
    for i in range(4):
        delta = np.eye(4)[i] * 1e-5
        plus, minus = (
            pairwise(raw + delta, rel, groups, **opts),
            pairwise(raw - delta, rel, groups, **opts),
        )
        assert (plus.loss - minus.loss) / 2e-5 == pytest.approx(result.gradient[i], abs=1e-9)
        assert (plus.gradient[i] - minus.gradient[i]) / 2e-5 == pytest.approx(
            result.curvature[i], abs=1e-9
        )
    shifted = pairwise(raw + [100, 100, -30, -30], rel, groups, **opts)
    np.testing.assert_allclose(result.gradient, shifted.gradient)
    assert result.gradient.sum() == pytest.approx(0)
    with pytest.raises(ValueError):
        pairwise(raw, rel, groups, weight=np.ones(4))


def test_ndcg_ties_and_zero_ideal():
    value = query_ndcg([0, 0], [0, 1], row_ids=[20, 10], k=1)
    assert value == 1
    assert query_ndcg([0, 0], [0, 1], row_ids=[10, 20], k=1) == 0
    assert query_ndcg([1, 0], [0, 0]) == 1


def test_lambda_weights_change_with_new_ranking():
    first = pairwise([0, 0, 0], [0, 2, 1], [0, 0, 0], lambdas=True, k=1)
    second = pairwise([0, 2, 1], [0, 2, 1], [0, 0, 0], lambdas=True, k=1)
    # At k=1 swapping two rows outside the top position has zero NDCG delta.
    assert dict(((i, j), w) for i, j, w in first.pairs)[(1, 2)] == 0
    assert dict(((i, j), w) for i, j, w in second.pairs)[(1, 2)] == pytest.approx(2 / 9)


def test_pair_two_rounds_recompute_scores():
    raw = np.zeros(2)
    for step in range(2):
        result = pairwise(raw, [1, 0], [0, 0])
        tree = fit_tree([[0], [1]], result.gradient, result.curvature, max_depth=1)
        q = 0.5 if step == 0 else 1 / (1 + np.exp(0.08))
        value = q / (1 + q * (1 - q))
        np.testing.assert_allclose(tree.predict([[0], [1]]), [value, -value])
        raw += 0.1 * tree.predict([[0], [1]])
    assert raw[0] > 0 > raw[1]


@pytest.mark.parametrize("q,expected", [(0.1, 0), (0.5, 2), (0.9, 10)])
def test_weighted_quantile_hand_and_optimality(q, expected):
    residual = np.array([0.0, 2.0, 10.0])
    weight = [1, 3, 1]
    value = weighted_quantile(residual, q, weight)
    assert value == expected
    optimum = pinball(np.full(3, value), residual, q, weight)[0]
    for candidate in [-1, 0, 1, 2, 5, 10, 11]:
        assert pinball(np.full(3, candidate), residual, q, weight)[0] >= optimum - 1e-14
    assert weighted_quantile([-100, 0, 2], 0.5, [0, 1, 1]) == 0


def test_quantile_two_rounds_uses_residual_leaf_not_newton():
    y = np.array([0.0, 2.0, 10.0])
    raw = np.full(3, weighted_quantile(y, 0.5, [2, 1, 2]))
    for step in range(2):
        tree = fit_quantile_tree([[0], [0], [1]], raw, y, 0.5, weight=[2, 1, 2], max_depth=1)
        assert tree.predict([[1]])[0] == pytest.approx(8 if step == 0 else 7.2)
        raw += 0.1 * tree.predict([[0], [0], [1]])
    assert raw[-1] == pytest.approx(3.52)


def test_vector_shared_topology_and_projection_keep_full_leaves():
    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = np.array([[-1.0, -3.0], [-1.0, 3.0], [1.0, -3.0], [1.0, 3.0]])
    shared = fit_vector_stump(x, np.zeros_like(y), y)
    projected = fit_vector_stump(x, np.zeros_like(y), y, projection=[[1.0], [0.0]])
    assert shared.condition.feature == 1
    assert projected.condition.feature == 0
    np.testing.assert_allclose(shared.predict(x), [[0, -2], [0, 2], [0, -2], [0, 2]])
    assert projected.predict(x).shape == (4, 2)
    np.testing.assert_allclose(
        projected.predict(x), [[-2 / 3, 0], [-2 / 3, 0], [2 / 3, 0], [2 / 3, 0]]
    )
    independent = [fit_tree(x, -y[:, k], np.ones(4), max_depth=1) for k in range(2)]
    assert [t.nodes[0].condition.feature for t in independent] == [0, 1]


def test_vector_k1_and_two_round_permutation():
    x = [[0], [0], [1], [1]]
    y = np.array([[-2.0, 1.0], [-2.0, 1.0], [2.0, -1.0], [2.0, -1.0]])
    raw = np.zeros_like(y)
    for step in range(2):
        tree = fit_vector_stump(x, raw, y)
        permuted = fit_vector_stump(x, raw[:, ::-1], y[:, ::-1])
        np.testing.assert_allclose(tree.predict(x)[:, ::-1], permuted.predict(x))
        expected = np.array([-4 / 3, 2 / 3]) if step == 0 else np.array([-56 / 45, 28 / 45])
        np.testing.assert_allclose(tree.predict([[0]])[0], expected)
        scalar = fit_tree(x, raw[:, 0] - y[:, 0], np.ones(4), max_depth=1)
        single = fit_vector_stump(x, raw[:, :1], y[:, :1])
        np.testing.assert_allclose(single.predict(x)[:, 0], scalar.predict(x))
        raw += 0.1 * tree.predict(x)
    np.testing.assert_allclose(raw[0], [-58 / 225, 29 / 225])


def test_target_scaling_training_only_and_constant_axis():
    y = np.array([[0.0, 5.0], [2.0, 5.0]])
    scale = TargetScale.fit(y)
    y[:] = -100
    np.testing.assert_allclose(scale.transform([[0, 5], [2, 5]]), [[-1, 0], [1, 0]])
    np.testing.assert_allclose(scale.inverse([[9, 0]]), [[10, 5]])
    assert scale.constant == (False, True)


def test_quantile_ties_may_have_no_positive_pseudo_split():
    # The residual quantile does not imply the pseudo-Newton split is profitable.
    tree = fit_quantile_tree(
        [[0], [0], [1]], [2, 2, 2], [0, 2, 10], 0.5, weight=[1, 3, 1], max_depth=1
    )
    assert tree.nodes[0].condition is None
    np.testing.assert_array_equal(tree.predict([[0], [1]]), [0, 0])


@pytest.mark.parametrize("q", [0, 1, -0.1, np.nan])
def test_quantile_rejects_invalid_level(q):
    with pytest.raises(ValueError):
        weighted_quantile([0, 1], q)


def test_quantile_zero_weights_and_replication():
    for weight in ([0, 0], [-1, 2], [1, np.inf]):
        with pytest.raises(ValueError):
            weighted_quantile([0, 1], 0.5, weight)
    for q in (0.1, 0.5, 0.9):
        assert weighted_quantile([0, 2, 10], q, [1, 3, 1]) == weighted_quantile([0, 2, 2, 2, 10], q)
    loss, g, h = pinball([0, 3, 10], [0, 2, 10], 0.9, [1, 3, 1])
    assert loss == pytest.approx(0.06)
    np.testing.assert_allclose(g, [-0.9, 0.1, -0.9])
    np.testing.assert_array_equal(h, [1, 1, 1])  # named pseudo curvature


def test_ranking_rejects_misaligned_or_silently_unused_metadata():
    bad = [
        dict(query_weight={9: 1}),
        dict(pair_weight={(1, 0): 1}),
        dict(query_weight={0: -1}),
        dict(row_ids=[1, 1]),
        dict(k=0),
    ]
    for options in bad:
        with pytest.raises(ValueError):
            pairwise([0, 0], [1, 0], [0, 0], **options)
    for relevance in ([1, -0.5], [np.nan, 0], [1024, 0]):
        with pytest.raises(ValueError):
            pairwise([0, 0], relevance, [0, 0])
    with pytest.raises(ValueError):
        pairwise([0, 0], [1, 0], [0])


def test_lambda_permutation_uses_stable_row_ids():
    raw, rel, ids = np.zeros(3), np.array([2, 0, 1]), np.array([30, 10, 20])
    original = pairwise(raw, rel, [0, 0, 0], row_ids=ids, lambdas=True)
    perm = [2, 0, 1]
    changed = pairwise(raw[perm], rel[perm], [0, 0, 0], row_ids=ids[perm], lambdas=True)
    assert changed.loss == pytest.approx(original.loss)
    np.testing.assert_allclose(changed.gradient, original.gradient[perm])
    np.testing.assert_allclose(changed.curvature, original.curvature[perm])


def test_vector_weight_replication_and_no_split_root():
    x, y = [[0], [1]], np.array([[-2.0, 1.0], [2.0, 3.0]])
    weighted = fit_vector_stump(x, np.zeros_like(y), y, weight=[1, 3])
    ids = [0, 1, 1, 1]
    replicated = fit_vector_stump(np.array(x)[ids], np.zeros((4, 2)), y[ids])
    np.testing.assert_allclose(weighted.predict(x), replicated.predict(x))
    root = fit_vector_stump([[0], [0]], np.zeros_like(y), y)
    assert root.condition is None
    np.testing.assert_allclose(root.predict([[0]]), [[0, 4 / 3]])


@pytest.mark.parametrize("projection", [[[0], [0]], [[1]], [[np.nan], [1]]])
def test_vector_rejects_invalid_projection(projection):
    with pytest.raises(ValueError):
        fit_vector_stump([[0], [1]], [[0, 0], [0, 0]], [[1, 2], [3, 4]], projection=projection)


def test_lambda_geometry_is_a_frozen_weighted_pair_surrogate():
    raw = np.array([0.1, 0.8, -0.3])
    result = pairwise(raw, [0, 2, 1], [0, 0, 0], lambdas=True)
    frozen = {(i, j): factor * 3 for i, j, factor in result.pairs}
    # Hold ranking-derived weights fixed for differentiation; no derivative of NDCG.
    for i in range(3):
        delta = np.eye(3)[i] * 1e-5
        plus = pairwise(raw + delta, [0, 2, 1], [0, 0, 0], pair_weight=frozen)
        minus = pairwise(raw - delta, [0, 2, 1], [0, 0, 0], pair_weight=frozen)
        assert (plus.loss - minus.loss) / 2e-5 == pytest.approx(result.gradient[i], abs=1e-9)


def test_vector_general_projection_affects_only_split_statistics():
    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = np.array([[-1.0, -3.0], [-1.0, 3.0], [1.0, -3.0], [1.0, 3.0]])
    tree = fit_vector_stump(x, np.zeros_like(y), y, projection=[[1.0], [1.0]])
    assert tree.condition.feature == 1
    np.testing.assert_allclose(tree.predict(x), [[0, -2], [0, 2], [0, -2], [0, 2]])


def test_ranking_extreme_score_and_zero_weight_query():
    result = pairwise([-1000, 1000], [1, 0], [0, 0])
    assert result.loss == 2000
    np.testing.assert_array_equal(result.gradient, [-1, 1])
    np.testing.assert_array_equal(result.curvature, [0, 0])
    zero = pairwise([-1000, 1000], [1, 0], [0, 0], query_weight={0: 0})
    assert zero.loss == 0
    np.testing.assert_array_equal(zero.gradient, [0, 0])
