"""Builder priority and exactly-once channel updates, with independent oracles."""

import numpy as np
import pytest

from openboost.experimental import (
    Booster,
    BuiltTree,
    CPUHistogramBuilder,
    TrainerConfig,
    TreeStructure,
)
from tests.test_experimental_objective import TwoSquared


def constant(value, n_features=1):
    return TreeStructure(features=np.array([-1], dtype=np.int32), thresholds=np.array([-1], dtype=np.int32),
                         left_children=np.array([-1], dtype=np.int32), right_children=np.array([-1], dtype=np.int32),
                         values=np.array([value], dtype=np.float32), n_nodes=1, depth=0, n_features=n_features)


class PresetBuilder:
    supported_devices = frozenset({'cpu'})

    def __init__(self):
        self.calls = []

    def build(self, binned, grad, hess, *, config, context):
        assert not grad.flags.writeable and not hess.flags.writeable
        assert not binned.data.flags.writeable
        self.calls.append((context.round_idx, context.channel))
        value = 2. if context.channel == 'a' else 4.
        return BuiltTree(constant(value), np.full(len(grad), value, dtype=np.float32))


class Decay:
    def coefficients(self, round_idx, channel_names, base_learning_rate):
        return {'a': base_learning_rate / (round_idx+1), 'b': base_learning_rate / (2*(round_idx+1))}


def test_builder_priority_and_exact_updates(monkeypatch):
    import openboost._trainer as trainer
    from openboost._callbacks import Callback
    # Even an eligible input must never bypass the explicitly selected builder.
    monkeypatch.setattr(trainer, '_gpu_native_eligible', lambda *args: True)
    def wrong(*args, **kwargs):
        raise AssertionError('Native dispatch bypassed explicit builder')
    monkeypatch.setattr(trainer, 'fit_tree_gpu_native', wrong)
    class Recording(TwoSquared):
        def loss_value(self, raw, *args, **kwargs):
            self.final_raw = {k: v.copy() for k, v in raw.items()}
            return super().loss_value(raw, *args, **kwargs)
    objective, builder = Recording(), PresetBuilder()
    X = np.zeros((4, 1), dtype=np.float32)
    model = Booster(objective=objective, tree_builder=builder, step_schedule=Decay(), config=TrainerConfig(n_trees=2, learning_rate=.5)).fit(X, np.arange(4, dtype=np.float32), callbacks=[Callback()])
    assert builder.calls == [(0,'a'), (0,'b'), (1,'a'), (1,'b')]
    assert model.coefficients_ == {'a': [.5,.25], 'b': [.25,.125]}
    for k, expected in [('a', 1.5), ('b', 2.5)]:
        np.testing.assert_array_equal(model.predict_raw(X)[k], np.full(4, expected, dtype=np.float32))
        np.testing.assert_array_equal(model.predict_raw(X)[k], objective.final_raw[k])


@pytest.mark.parametrize('values', [{'a': .1}, {'a': -.1, 'b': .1}, {'a': np.nan, 'b': .1}])
def test_schedule_rejected_before_build(values):
    class Bad:
        def coefficients(self, *args):
            return values
    builder = PresetBuilder()
    with pytest.raises(ValueError):
        Booster(objective=TwoSquared(), tree_builder=builder, step_schedule=Bad(), config=TrainerConfig(n_trees=1)).fit(np.zeros((4,1)), np.ones(4))
    assert builder.calls == []


def test_cached_prediction_must_match_tree():
    class Bad(PresetBuilder):
        def build(self, binned, grad, hess, **kwargs):
            return BuiltTree(constant(1), np.zeros(len(grad), dtype=np.float32))
    with pytest.raises(ValueError, match='prediction'):
        Booster(objective=TwoSquared(), tree_builder=Bad(), config=TrainerConfig(n_trees=1)).fit(np.zeros((4,1)), np.ones(4))


def test_cpu_split_gain_scale_and_newton():
    X = np.array([[0], [0], [1], [1]], dtype=np.float32)
    y = np.array([-2,-2,2,2], dtype=np.float32)
    # At channel a: G_left=4, G_right=-4, H_child=2, lambda=1.
    # Unhalved gain is 16/3 + 16/3 = 32/3. min_gain=10 splits; 11 rejects.
    for gain, leaves in [(10., [-4/3,-4/3,4/3,4/3]), (11., [0,0,0,0])]:
        model = Booster(objective=TwoSquared(), config=TrainerConfig(n_trees=1, max_depth=1, min_gain=gain, learning_rate=1)).fit(X,y)
        np.testing.assert_allclose(model.predict_raw(X)['a'], leaves, rtol=1e-6)


def test_zero_curvature_root():
    class Zero(TwoSquared):
        def step(self, raw, y, *args, **kwargs):
            return {k: (np.zeros_like(y), np.zeros_like(y)) for k in self.channel_names}
    X, y = np.zeros((4,1)), np.ones(4)
    model = Booster(objective=Zero(), config=TrainerConfig(n_trees=1, reg_lambda=0)).fit(X,y)
    np.testing.assert_array_equal(model.predict_raw(X)['a'], np.zeros(4))
    class Invalid(Zero):
        def step(self, raw, y, *args, **kwargs):
            return {k: (np.ones_like(y), np.zeros_like(y)) for k in self.channel_names}
    with pytest.raises(ValueError, match='curvature'):
        Booster(objective=Invalid(), config=TrainerConfig(n_trees=1, reg_lambda=0)).fit(X,y)


def test_default_builder_is_cpu_only():
    assert CPUHistogramBuilder.supported_devices == frozenset({'cpu'})


def test_builder_scratch_tree_is_detached():
    class Reuse(PresetBuilder):
        def __init__(self):
            self.tree = constant(0)
        def build(self, binned, grad, hess, *, config, context):
            self.tree.values[0] = 2 if context.channel == 'a' else 4
            return BuiltTree(self.tree)
    model = Booster(objective=TwoSquared(), tree_builder=Reuse(), config=TrainerConfig(n_trees=1, learning_rate=1)).fit(np.zeros((4,1)), np.ones(4))
    assert model.trees_['a'][0].values[0] == 2
    assert model.trees_['b'][0].values[0] == 4


def test_invalid_tree_routing_rejected():
    class Bad(PresetBuilder):
        def build(self, binned, grad, hess, **kwargs):
            tree = constant(1)
            tree.left_children[0] = tree.right_children[0] = 0
            tree.features[0] = tree.thresholds[0] = 0
            return BuiltTree(tree)
    with pytest.raises(ValueError, match='cyclic'):
        Booster(objective=TwoSquared(), tree_builder=Bad(), config=TrainerConfig(n_trees=1)).fit(np.zeros((4,1)), np.ones(4))


def test_unsupported_zero_regularization_pair():
    with pytest.raises(ValueError, match='positive min_child_weight'):
        Booster(objective=TwoSquared(), config=TrainerConfig(reg_lambda=0, min_child_weight=0)).fit(np.zeros((4,1)), np.ones(4))
