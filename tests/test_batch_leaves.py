"""Direct row-sum and Newton references for leaf reduction and user rules."""

import numpy as np
import pytest

from openboost.experimental import TrainerConfig, leaf_values, reduce_leaves


class BoundedLeaf:
    supported_devices = frozenset({"cpu", "cuda"})

    def values(self, grad, hess, *, config, context):
        from openboost.experimental import NewtonLeafRule

        return context.xp.clip(
            NewtonLeafRule().values(grad, hess, config=config, context=context), -0.5, 0.5
        )


def example():
    g = np.array([-2, 0, -12, 5, -3, 8], np.float32)
    h = np.array([1, 0, 3, 2, 1, 4], np.float32)
    ids = np.array([0, 0, 0, 1, 2, -1], np.int32)
    active = np.array([True, False, True, True])
    return g, h, ids, active


def direct(g, h, ids, active):
    G, H, counts = np.zeros(len(active)), np.zeros(len(active)), np.zeros(len(active), np.int32)
    for i, node in enumerate(ids):
        if node >= 0 and active[node]:
            G[node] += float(g[i])
            H[node] += float(h[i])
            counts[node] += 1
    return G, H, counts


def test_reduction_and_weighted_newton():
    args = example()
    stats = reduce_leaves(*args)
    for actual, expected in zip((stats.grad, stats.hess, stats.counts), direct(*args), strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(leaf_values(*args), [14 / 5, 0, 3 / 2, 0], rtol=1e-6)
    np.testing.assert_array_equal(leaf_values(*args, leaf_rule=BoundedLeaf()), [0.5, 0, 0.5, 0])
    assert stats.counts.tolist() == [3, 0, 1, 0]


def test_two_rounds_bounded_values_change_next_gradient():
    y, weights = np.array([2, 4], np.float32), np.array([1, 3], np.float32)
    ids, active = np.zeros(2, np.int32), np.array([True])
    results = []
    for rule in (None, BoundedLeaf()):
        raw = np.zeros(2, np.float32)
        history = []
        for _ in range(2):
            g = (raw - y) * weights
            history.append(g.copy())
            raw += leaf_values(g, weights.copy(), ids, active, leaf_rule=rule)[ids]
        results.append((raw, history))
    np.testing.assert_allclose(results[0][0], [3.36, 3.36], rtol=1e-6)
    np.testing.assert_array_equal(results[1][0], [1, 1])
    np.testing.assert_allclose(results[0][1][1], [0.8, -3.6], atol=1e-6)
    np.testing.assert_array_equal(results[1][1][1], [-1.5, -10.5])


def test_empty_and_zero_curvature():
    z = np.zeros(0, np.float32)
    np.testing.assert_array_equal(
        leaf_values(
            z, z, np.zeros(0, np.int32), np.ones(3, bool), config=TrainerConfig(reg_lambda=0)
        ),
        [0, 0, 0],
    )
    g, h, ids, active = (
        np.zeros(2, np.float32),
        np.zeros(2, np.float32),
        np.zeros(2, np.int32),
        np.array([True]),
    )
    assert leaf_values(g, h, ids, active, config=TrainerConfig(reg_lambda=0))[0] == 0
    g[:] = 1
    with pytest.raises(ValueError, match="curvature"):
        leaf_values(g, h, ids, active, config=TrainerConfig(reg_lambda=0))


@pytest.mark.parametrize("kind", ["dtype", "shape", "nan", "inactive", "mutation", "device"])
def test_bad_rule(kind):
    class Bad:
        supported_devices = frozenset({"cuda"} if kind == "device" else {"cpu"})

        def values(self, g, h, *, config, context):
            if kind == "mutation":
                g[0] = 123
            if kind == "dtype":
                return np.zeros(len(g), np.float64)
            if kind == "shape":
                return np.zeros(len(g) + 1, np.float32)
            if kind == "nan":
                return np.full(len(g), np.nan, np.float32)
            return np.ones(len(g), np.float32)

    with pytest.raises((TypeError, ValueError)):
        leaf_values(*example(), leaf_rule=Bad())


@pytest.mark.parametrize(
    "index,value",
    [
        (0, np.ones(6, np.float64)),
        (1, np.full(6, -1, np.float32)),
        (2, np.full(6, 4, np.int32)),
        (0, np.full(6, np.inf, np.float32)),
    ],
)
def test_bad_statistics(index, value):
    args = list(example())
    args[index] = value
    with pytest.raises((ValueError, TypeError)):
        reduce_leaves(*args)


def test_default_rule_rejects_unsupported_l1():
    with pytest.raises(ValueError, match="L2"):
        leaf_values(*example(), config=TrainerConfig(reg_alpha=1))


def test_bounded_leaf_in_actual_cpu_trainer():
    from openboost.experimental import Booster, BuiltTree
    from tests.test_experimental_dispatch import constant
    from tests.test_experimental_objective import TwoSquared

    class Observe(TwoSquared):
        def __init__(self):
            self.history = []

        def step(self, raw, y, sample_weight=None, extra=None, *, context):
            result = super().step(raw, y, sample_weight, extra, context=context)
            self.history.append(result["a"][0].copy())
            return result

    class RootBuilder:
        supported_devices = frozenset({"cpu"})

        def __init__(self, rule):
            self.rule = rule

        def build(self, binned, grad, hess, *, config, context):
            values = leaf_values(
                grad,
                hess,
                np.zeros(len(grad), np.int32),
                np.array([True]),
                config=config,
                context=context,
                leaf_rule=self.rule,
            )
            return BuiltTree(constant(float(values[0]), binned.n_features))

    X, y, w = (
        np.zeros((2, 1), np.float32),
        np.array([2, 4], np.float32),
        np.array([1, 3], np.float32),
    )
    models = []
    for rule in (None, BoundedLeaf()):
        model = Booster(
            objective=Observe(),
            tree_builder=RootBuilder(rule),
            config=TrainerConfig(n_trees=2, max_depth=0, learning_rate=1),
        ).fit(X, y, sample_weight=w)
        models.append(model)
    np.testing.assert_allclose(models[0].predict_raw(X)["a"], [3.36, 3.36], rtol=1e-6)
    np.testing.assert_array_equal(models[1].predict_raw(X)["a"], [1, 1])
    np.testing.assert_allclose(models[0].objective.history[1], [0.8, -3.6], atol=1e-6)
    np.testing.assert_array_equal(models[1].objective.history[1], [-1.5, -10.5])


def test_rule_scratch_ownership_and_reduction_overflow():
    class Scratch(BoundedLeaf):
        def values(self, *args, **kwargs):
            self.buffer = super().values(*args, **kwargs)
            return self.buffer

    rule = Scratch()
    actual = leaf_values(*example(), leaf_rule=rule)
    rule.buffer[:] = 17
    np.testing.assert_array_equal(actual, [0.5, 0, 0.5, 0])
    args = list(example())
    args[0] = np.full(6, np.finfo(np.float32).max, np.float32)
    with pytest.raises(ValueError, match="overflow"):
        reduce_leaves(*args)
