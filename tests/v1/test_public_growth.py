"""Independent topology checks for all public numeric growth policies."""

import numpy as np
import pytest

from openboost import NumericData, Problem
from openboost.binning import NumericBinning
from openboost.stats import newton
from openboost.tree import best_first, depthwise, symmetric
from tests.v1.reference.tree import fit_tree


@pytest.mark.parametrize("grow", [depthwise, best_first, symmetric])
@pytest.mark.parametrize("depth,leaves", [(0, 1), (1, 4), (2, 3), (3, 5), (3, 8)])
def test_policy_matches_independent_reference(grow, depth, leaves):
    rng = np.random.default_rng(19)
    values = rng.integers(0, 4, (20, 3)).astype(float)
    values[::4, 1] = np.nan
    x = NumericData(values, np.arange(20), ("a", "b", "c"))
    p = Problem(x, np.zeros((20, 1)), x.row_ids, weight=rng.integers(0, 4, 20))
    g, h = rng.normal(size=20), rng.uniform(0.5, 2, 20)
    b = NumericBinning.fit(x, bins=4).transform(x)
    tree = grow(b, newton(p, g, h), max_depth=depth, max_leaves=leaves)
    bins = np.where(b.missing.T, np.nan, b.codes.T)
    ref = fit_tree(
        bins, g, h, weight=p.weight, policy=grow.__name__, max_depth=depth, max_leaves=leaves
    )
    assert len(tree.value) == len(ref.nodes)
    for i, node in enumerate(ref.nodes):
        assert tree.value[i] == pytest.approx(node.value)
        assert (tree.left[i], tree.right[i]) == (node.left, node.right)
        if node.condition:
            assert (tree.feature[i], tree.threshold[i], tree.missing_left[i]) == (
                node.condition.feature,
                node.condition.threshold,
                node.condition.missing_left,
            )
        else:
            assert tree.feature[i] == -1
    np.testing.assert_allclose(tree.predict(x)[:, 0], ref.predict(bins))


def test_symmetric_uses_common_gain_not_individual_winners():
    x = NumericData([[0, 0], [0, 1], [1, 0], [1, 1]], [1, 2, 3, 4], ("a", "b"))
    p = Problem(x, [[0]] * 4, x.row_ids)
    b = NumericBinning.fit(x, bins=2).transform(x)
    fields = newton(p, [-4, -4, 1, 3], [1] * 4)
    seen = set()

    def scoring(c):
        key = (c.rows_identity, c.key)
        assert key not in seen
        seen.add(key)
        if c.left_count + c.right_count == 4:
            return 10 if c.feature == 0 else 0
        return (-1 if c.parent[0] < 0 else 3) if c.feature == 1 else 0

    tree = symmetric(b, fields, max_depth=2, scoring=scoring)
    assert tree.feature.tolist() == [0, 1, 1, -1, -1, -1, -1]
    assert tree.threshold[1] == tree.threshold[2]
    assert tree.missing_left[1] == tree.missing_left[2]
    seen.clear()
    limited = symmetric(b, fields, max_depth=2, max_leaves=3, scoring=scoring)
    assert len(limited.value) == 3  # Cannot partially split a symmetric layer.


@pytest.mark.parametrize("grow", [best_first, symmetric])
def test_callbacks_leaf_values_and_scores_are_used_once(grow):
    from openboost.ops import feasible, newton_leaf, score

    rng = np.random.default_rng(23)
    x = NumericData(rng.normal(size=(24, 2)), np.arange(24), ("a", "b"))
    p = Problem(x, np.zeros((24, 1)), x.row_ids)
    b = NumericBinning.fit(x, bins=6).transform(x)
    fields = newton(p, rng.normal(size=24), np.ones(24))
    seen, leaf_calls = set(), []

    def scoring(c):
        key = (c.rows_identity, c.key)
        assert key not in seen
        seen.add(key)
        return score(c)

    def leaf(total, names):
        leaf_calls.append(1)
        return 2 * newton_leaf(total, names)

    def legality(c):
        return c.feature == 1 and feasible(c)

    tree = grow(b, fields, max_depth=4, max_leaves=8, scoring=scoring, legality=legality, leaf=leaf)
    baseline = grow(b, fields, max_depth=4, max_leaves=8, legality=legality)
    assert seen and len(leaf_calls) == len(tree.value)
    assert set(tree.feature) <= {-1, 1}
    np.testing.assert_allclose(tree.predict(x), 2 * baseline.predict(x))


@pytest.mark.parametrize("grow", [best_first, symmetric])
def test_recipe_substitution_and_persistence(grow, tmp_path):
    from functools import partial

    from openboost import RunContext
    from openboost.artifacts import Model
    from openboost.recipes import squared
    from tests.v1.reference.tree import boost_squared

    rng = np.random.default_rng(41)
    x = NumericData(rng.integers(0, 5, (20, 3)), np.arange(20), ("a", "b", "c"))
    p = Problem(x, rng.normal(size=(20, 1)), x.row_ids, weight=rng.uniform(0.5, 2, 20))
    result = squared(
        p,
        p,
        context=RunContext(grow.__name__, 1),
        rounds=3,
        bins=5,
        learner=partial(grow, max_depth=3, max_leaves=5),
    )
    b = NumericBinning.fit(x, bins=5).transform(x)
    expected = boost_squared(
        b.codes.T,
        p.target[:, 0],
        weight=p.weight,
        rounds=3,
        policy=grow.__name__,
        max_depth=3,
        max_leaves=5,
    )
    np.testing.assert_allclose(result.state.train_raw[:, 0], expected.predict(b.codes.T))
    path = tmp_path / "ensemble.json"
    result.state.model.save(path)
    restored = Model.load(path)
    unseen = NumericData(
        [[-100, 1, 0], [100, 3, 5], [np.nan, np.nan, np.nan]], [1, 2, 3], ("a", "b", "c")
    )
    np.testing.assert_array_equal(restored.predict(unseen), result.state.model.predict(unseen))
    assert restored.identity == result.state.model.identity


@pytest.mark.parametrize("grow", [best_first, symmetric])
def test_invalid_callbacks_and_capacity_rejected(grow):
    x = NumericData([[0], [1]], [1, 2], ("a",))
    p = Problem(x, [[0], [0]], x.row_ids)
    b = NumericBinning.fit(x, bins=2).transform(x)
    fields = newton(p, [-1, 1], [1, 1])
    for kwargs in (
        {"max_depth": -1},
        {"max_leaves": 0},
        {"max_leaves": 2**31},
        {"scoring": lambda _: np.nan},
        {"leaf": lambda *_: np.inf},
    ):
        with pytest.raises(ValueError):
            grow(b, fields, **kwargs)
