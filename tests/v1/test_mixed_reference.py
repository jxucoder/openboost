"""Raw mixed inputs, full-depth trees and vector payload conformance fixtures."""

import numpy as np
import pytest

from .reference.classification import binary
from .reference.mixed import Transformer, grow
from .reference.tree import fit_tree
from .reference.vector import fit_vector_stump


def fixture():
    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = np.array([[-1.0, -3.0], [-1.0, 3.0], [1.0, -3.0], [1.0, 3.0]])
    transformer = Transformer.fit(x, names=("a", "b"), kinds=("numeric", "numeric"), bins=2)
    return x, y, transformer


@pytest.mark.parametrize("policy", ["depthwise", "best_first", "symmetric"])
def test_multilevel_vector_two_rounds_and_row_conservation(policy):
    x, y, transformer = fixture()
    y[:, 0] *= 2  # Positive summed gain at the second layer despite ridge cost.
    raw = np.zeros_like(y)
    for iteration in range(2):
        tree = grow(x, raw - y, np.ones_like(y), transformer, policy=policy, max_depth=2)
        assert len(tree.nodes) == 7
        leaves = [node for node in tree.nodes if node.condition is None]
        assert sorted(i for node in leaves for i in node.rows) == list(range(4))
        assert all(len(node.rows) == 1 for node in leaves)
        expected = y / 2 if iteration == 0 else 0.95 * y / 2
        np.testing.assert_allclose(tree.predict(x), expected)
        raw += 0.1 * tree.predict(x)
    np.testing.assert_allclose(raw, 0.0975 * y)


@pytest.mark.parametrize("policy", ["depthwise", "best_first", "symmetric"])
def test_k1_matches_existing_scalar_reference(policy):
    x = [[0], [1], [2], [3], [np.nan]]
    y = np.array([0.0, -1.0, 3.0, 1.0, 4.0])
    tr = Transformer.fit(x, names=("x",), kinds=("numeric",), bins=4)
    tree = grow(x, -y[:, None], np.ones((5, 1)), tr, policy=policy, max_depth=3)
    scalar = fit_tree(tr.transform(x), -y, np.ones(5), policy=policy, max_depth=3)
    np.testing.assert_allclose(tree.predict(x)[:, 0], scalar.predict(tr.transform(x)))


def test_category_middle_value_and_unknown_missing_routing():
    x = [["a"], ["a"], ["m"], ["m"], ["z"], ["z"], [None]]
    y = np.array([1, 1, -2, -2, 1, 1, -2.0])[:, None]
    tr = Transformer.fit(x, names=("category",), kinds=("categorical",))
    tree = grow(x, -y, np.ones_like(y), tr, max_depth=2)
    root = tree.nodes[0]
    assert root.condition == (0, 1, True)
    np.testing.assert_allclose(tree.predict([["m"], ["unseen"], [None]]), [[-1.5], [-1.5], [-1.5]])
    np.testing.assert_allclose(tree.predict([["a"], ["z"]]), [[0.8], [0.8]])


def test_raw_mixed_binary_two_rounds_train_only_transform():
    x = [[0, "a"], [1, "a"], [0, "b"], [1, "b"], [np.nan, None]]
    y = np.array([0, 0, 1, 1, 1])
    tr = Transformer.fit(x, names=("numeric", "category"), kinds=("numeric", "categorical"), bins=2)
    raw = np.zeros(5)
    test_x = [[100, "new"], [0, "b"], [np.nan, None]]
    prediction = np.zeros(3)
    trees = []
    for _ in range(2):
        _, g, h = binary(raw, y)
        tree = grow(x, g[:, None], h[:, None], tr, max_depth=2)
        raw += 0.1 * tree.predict(x)[:, 0]
        prediction += 0.1 * tree.predict(test_x)[:, 0]
        trees.append(tree)
    np.testing.assert_allclose(prediction, sum(0.1 * t.predict(test_x)[:, 0] for t in trees))
    assert binary(raw, y)[0] < np.log(2)
    assert tr.encoders[1].values == ("a", "b")
    assert tr.encoders[0].cuts == (0.5,)
    assert prediction[0] == prediction[2]  # unknown follows fitted missing branch


def test_vector_projection_preserves_payload_and_matches_stump():
    x, y, tr = fixture()
    projected = grow(x, -y, np.ones_like(y), tr, max_depth=1, projection=[[1], [0]])
    stump = fit_vector_stump(tr.transform(x), np.zeros_like(y), y, projection=[[1], [0]])
    assert projected.nodes[0].condition[0] == 0
    np.testing.assert_allclose(projected.predict(x), stump.predict(tr.transform(x)))
    permuted = grow(x, -y[:, ::-1], np.ones_like(y), tr, max_depth=2)
    original = grow(x, -y, np.ones_like(y), tr, max_depth=2)
    np.testing.assert_allclose(original.predict(x)[:, ::-1], permuted.predict(x))


def test_weight_replication_and_input_mutation_do_not_change_tree():
    x, y, tr = fixture()
    original = grow(x, -y, np.ones_like(y), tr, weight=[1, 2, 1, 2], max_depth=2)
    ids = [0, 1, 1, 2, 3, 3]
    repeated = grow(np.array(x)[ids], -y[ids], np.ones((6, 2)), tr, max_depth=2)
    np.testing.assert_allclose(original.predict(x), repeated.predict(x))
    before = original.predict(x)
    y[:] = 100
    np.testing.assert_array_equal(original.predict(x), before)


def test_all_missing_or_zero_information_has_no_split():
    for kind in ("numeric", "categorical"):
        x = [[None], [None]]
        tr = Transformer.fit(x, names=("x",), kinds=(kind,))
        tree = grow(x, [[1], [1]], [[0], [0]], tr)
        assert len(tree.nodes) == 1
        np.testing.assert_allclose(tree.predict(x), [[-2], [-2]])


def test_vector_regularization_can_stop_a_split_helpful_to_one_output():
    x, y, tr = fixture()
    tree = grow(x, -y, np.ones_like(y), tr, max_depth=2)
    assert len(tree.nodes) == 3
    # Second-layer gain: .5 from separating +/-1, minus 1.5 for splitting
    # the other channel's constant magnitude 3 into separately penalized leaves.
    np.testing.assert_allclose(tree.predict(x), [[0, -2], [0, 2], [0, -2], [0, 2]])


@pytest.mark.parametrize("policy", ["depthwise", "best_first"])
def test_multilevel_category_numeric_routing_with_exact_unregularized_leaves(policy):
    x = [[0, "a"], [1, "a"], [0, "m"], [1, "m"], [0, "z"], [1, "z"]]
    y = np.array([[-2, -3], [2, -3], [-2, 0], [2, 0], [-2, 3], [2, 3.0]])
    tr = Transformer.fit(x, names=("number", "category"), kinds=("numeric", "categorical"), bins=2)
    tree = grow(x, -y, np.ones_like(y), tr, max_depth=3, policy=policy, reg_lambda=0)
    leaves = [node for node in tree.nodes if node.condition is None]
    assert len(leaves) == 6 and max(node.depth for node in leaves) == 3
    np.testing.assert_allclose(tree.predict(x), y)
    assert sorted(i for node in leaves for i in node.rows) == list(range(6))
    assert {node.condition[0] for node in tree.nodes if node.condition} == {0, 1}


@pytest.mark.parametrize("policy", ["depthwise", "best_first", "symmetric"])
def test_missing_direction_and_zero_weight_rows(policy):
    x = [[0], [1], [np.nan], [np.nan]]
    tr = Transformer.fit(x, names=("x",), kinds=("numeric",), bins=2)
    for target, missing_left in (([-2, 2, -2, 99], True), ([-2, 2, 2, 99], False)):
        y = np.array(target, dtype=float)[:, None]
        tree = grow(x, -y, np.ones_like(y), tr, weight=[1, 1, 1, 0], max_depth=1, policy=policy)
        assert tree.nodes[0].condition[2] is missing_left
        assert tree.predict([[np.nan]])[0, 0] == pytest.approx(-4 / 3 if missing_left else 4 / 3)


@pytest.mark.parametrize("max_depth", [-1, True, 1.5])
def test_invalid_growth_depth(max_depth):
    x, y, tr = fixture()
    with pytest.raises(ValueError):
        grow(x, -y, np.ones_like(y), tr, max_depth=max_depth)


def test_schema_and_projection_rejection():
    x, y, tr = fixture()
    with pytest.raises(ValueError):
        Transformer.fit(x, names=("a", "a"), kinds=("numeric", "numeric"))
    with pytest.raises(ValueError):
        tr.transform([[0]])
    with pytest.raises(ValueError):
        grow(x, -y, np.ones_like(y), tr, projection=[[0], [0]])
    with pytest.raises(ValueError):
        grow(x, -y, -np.ones_like(y), tr)


def test_mixed_pipeline_identity_includes_fitted_transformer_and_raw_content():
    from .reference.runs import bind_identity

    x = [[0, "a"], [1, "b"], [2, "a"]]
    tr = Transformer.fit(x, names=("n", "c"), kinds=("numeric", "categorical"), bins=2)
    identity = tr.identity([10, 20, 30], x)
    assert identity == tr.identity([10, 20, 30], x)
    changed_cuts = Transformer.fit(x, names=("n", "c"), kinds=("numeric", "categorical"), bins=3)
    assert identity != changed_cuts.identity([10, 20, 30], x)
    problem = bind_identity(identity, [10, 20, 30], target=([10, 20, 30], [0, 1, 0]))
    assert len(problem) == 64
    with pytest.raises(ValueError):
        bind_identity(identity, [30, 20, 10], target=([30, 20, 10], [0, 1, 0]))


def test_symmetric_vector_uses_one_shared_condition_per_layer():
    x, y, tr = fixture()
    y[:, 0] *= 2
    tree = grow(x, -y, np.ones_like(y), tr, policy="symmetric", max_depth=2)
    for depth in (0, 1):
        conditions = {
            n.condition for n in tree.nodes if n.depth == depth and n.condition is not None
        }
        assert len(conditions) == 1


def test_nonfinite_combined_gain_rejected():
    x = [[0], [0], [1], [1]]
    tr = Transformer.fit(x, names=("x",), kinds=("numeric",), bins=2)
    gradient = np.array([[1e154, 1e154]] * 2 + [[-1e154, -1e154]] * 2)
    with pytest.raises(ValueError, match="non-finite"):
        grow(x, gradient, np.ones((4, 2)), tr)
