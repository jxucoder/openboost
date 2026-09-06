"""Whole-tree oracle derives every split and leaf from original rows."""

import numpy as np
import pytest

import openboost as ob
from openboost.experimental import ExecutionContext, LevelWiseBuilder, TrainerConfig


def example():
    bins = np.array([[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 0, 1, 0, 1, 0, 1]], np.uint8)
    g = np.array([4, 0, 4, 2, -1, -1, -15, -5], np.float32)
    h = np.array([1, 0, 2, 1, 0.5, 1, 3, 1], np.float32)
    binned = ob.BinnedArray(bins, [np.array([0.5, 1.5, 2.5]), np.array([0.5])], 2, 8, "cpu")
    return binned, g, h


def context(xp=np, round_idx=0, channel="a"):
    return ExecutionContext(
        "cpu" if xp is np else "cuda", xp, np.random.default_rng(7), round_idx, channel
    )


def oracle_tree(bins, g, h, config, bound=None):
    slots = 2 ** (config.max_depth + 1) - 1
    arrays = {
        k: np.full(slots, -1, np.int32)
        for k in ("features", "thresholds", "left_children", "right_children")
    }
    values, predictions = np.zeros(slots, np.float32), np.zeros(len(g), np.float32)

    def grow(node, rows, depth):
        G, H = sum(float(v) for v in g[rows]), sum(float(v) for v in h[rows])
        best = None
        if depth < config.max_depth and H > 0:
            for f in range(len(bins)):
                for t in range(255):
                    left_rows = rows[bins[f, rows] <= t]
                    right_rows = rows[bins[f, rows] > t]
                    GL, HL = sum(float(v) for v in g[left_rows]), sum(float(v) for v in h[left_rows])
                    GR, HR = sum(float(v) for v in g[right_rows]), sum(float(v) for v in h[right_rows])
                    if HL <= 0 or HR <= 0 or min(HL, HR) < config.min_child_weight:
                        continue
                    gain = (
                        GL**2 / (HL + config.reg_lambda)
                        + GR**2 / (HR + config.reg_lambda)
                        - G**2 / (H + config.reg_lambda)
                    )
                    if gain > 0 and gain >= config.min_gain and (best is None or gain > best[0]):
                        best = (gain, f, t, left_rows, right_rows)
        if best is None:
            denominator = H + config.reg_lambda
            value = -G / denominator if denominator else 0.0
            if bound is not None:
                value = np.clip(value, -bound, bound)
            values[node], predictions[rows] = value, value
            return
        _, f, t, left_rows, right_rows = best
        arrays["features"][node], arrays["thresholds"][node] = f, t
        arrays["left_children"][node], arrays["right_children"][node] = 2 * node + 1, 2 * node + 2
        grow(2 * node + 1, left_rows, depth + 1)
        grow(2 * node + 2, right_rows, depth + 1)

    grow(0, np.arange(len(g)), 0)
    return {**arrays, "values": values}, predictions


@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_whole_tree_row_oracle(depth):
    binned, g, h = example()
    cfg = TrainerConfig(max_depth=depth)
    built = LevelWiseBuilder().build(binned, g, h, config=cfg, context=context())
    arrays, prediction = oracle_tree(binned.data, g, h, cfg)
    for k, v in arrays.items():
        np.testing.assert_allclose(getattr(built.tree, k), v, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(built.train_prediction, built.tree(binned))
    np.testing.assert_allclose(built.train_prediction, prediction, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "kind", ["missing", "categorical", "subsample", "colsample_bytree", "reg_alpha", "budget"]
)
def test_preflight_before_histogram(monkeypatch, kind):
    import openboost.experimental._levelwise as module

    binned, g, h = example()
    cfg = TrainerConfig(max_depth=2)
    builder = LevelWiseBuilder(memory_budget_bytes=0 if kind == "budget" else 256 * 1024**2)
    if kind == "missing":
        binned.data[0, 0] = 255
    if kind == "categorical":
        binned.is_categorical = np.array([True, False])
    if kind in ("subsample", "colsample_bytree"):
        setattr(cfg, kind, 0.5)
    if kind == "reg_alpha":
        cfg.reg_alpha = 1

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must precede histograms")

    monkeypatch.setattr(module, "build_histograms", forbidden)
    with pytest.raises((ValueError, MemoryError)):
        builder.build(binned, g, h, config=cfg, context=context())


def test_two_channel_trainer_schedule_and_persistence(tmp_path):
    from openboost.experimental import Booster
    from tests.test_batch_leaves import BoundedLeaf
    from tests.test_experimental_dispatch import Decay
    from tests.test_experimental_objective import TwoSquared

    X = np.arange(16, dtype=np.float32).reshape(8, 2)
    y = np.array([3, 3, 2, 2, -1, -1, -4, -4], np.float32)
    cfg = TrainerConfig(n_trees=2, max_depth=2, learning_rate=0.5)
    models = []
    for rule in (None, BoundedLeaf()):
        model = Booster(
            objective=TwoSquared(),
            tree_builder=LevelWiseBuilder(leaf_rule=rule),
            step_schedule=Decay(),
            config=cfg,
        ).fit(X, y)
        before = model.predict_raw(X)
        path = tmp_path / f"model{len(models)}.ob"
        model.save(path)
        with pytest.warns(UserWarning, match="trusted"):
            loaded = Booster.load(path)
        for channel, raw in before.items():
            np.testing.assert_array_equal(loaded.predict_raw(X)[channel], raw)
        assert model.coefficients_ == {"a": [0.5, 0.25], "b": [0.25, 0.125]}
        models.append(model)
    assert not np.allclose(models[0].predict_raw(X)["a"], models[1].predict_raw(X)["a"])


def test_zero_curvature_and_early_leaf():
    binned, g, h = example()
    cfg = TrainerConfig(max_depth=3, reg_lambda=0)
    result = LevelWiseBuilder().build(
        binned, np.zeros_like(g), np.zeros_like(h), config=cfg, context=context()
    )
    assert result.tree.left_children[0] == -1
    np.testing.assert_array_equal(result.train_prediction, 0)
    with pytest.raises(ValueError, match="curvature"):
        LevelWiseBuilder().build(
            binned, np.ones_like(g), np.zeros_like(h), config=cfg, context=context()
        )


@pytest.mark.parametrize("early_leaf", [False, True])
def test_fixed_slot_growth_does_not_compact_split_arrays(monkeypatch, early_leaf):
    """Fixed-slot growth must not materialize variable-length masked split arrays."""
    from dataclasses import replace

    import openboost.experimental._levelwise as module

    class FixedSlots(np.ndarray):
        def __getitem__(self, index):
            if isinstance(index, np.ndarray) and index.dtype == np.bool_:
                raise AssertionError("Boolean compaction of fixed-slot split arrays")
            return super().__getitem__(index)

    original = module.find_splits

    def splits(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(
            result,
            **{
                name: getattr(result, name).view(FixedSlots)
                for name in ("feature", "threshold", "left_child", "right_child", "valid")
            },
        )

    monkeypatch.setattr(module, "find_splits", splits)
    binned, g, h = example()
    if early_leaf:
        g = np.zeros_like(g)
    cfg = TrainerConfig(max_depth=3)
    built = LevelWiseBuilder().build(binned, g, h, config=cfg, context=context())
    expected, prediction = oracle_tree(binned.data, g, h, cfg)
    for name, values in expected.items():
        np.testing.assert_allclose(getattr(built.tree, name), values, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(built.train_prediction, prediction, rtol=1e-6, atol=1e-6)
