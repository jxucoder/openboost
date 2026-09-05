"""Independent CPU objective and contract failures for the experimental facade."""

import numpy as np
import pytest

from openboost.experimental import Booster, DistributionObjectiveAdapter, TrainerConfig


class TwoSquared:
    channel_names = ('a', 'b')
    supported_devices = frozenset({'cpu'})

    def init_raw(self, y, sample_weight=None, extra=None):
        return {'a': 0., 'b': 1.}

    def step(self, raw, y, sample_weight=None, extra=None, *, context):
        assert context.device == 'cpu' and context.channel is None
        assert not raw['a'].flags.writeable
        w = np.ones_like(y) if sample_weight is None else sample_weight
        return {k: ((raw[k] - y) * w, w.copy()) for k in self.channel_names}

    def loss_value(self, raw, y, sample_weight=None, extra=None, *, context):
        return np.average(sum((raw[k] - y) ** 2 for k in self.channel_names) / 2, weights=sample_weight)

    def constrain(self, raw, extra=None):
        return raw


@pytest.fixture
def data():
    return np.zeros((4, 1), dtype=np.float32), np.array([1, 2, 3, 4], dtype=np.float32)


def test_two_channel_weighted_newton(data):
    X, y = data
    weights = np.array([0, 1, 2, 4], dtype=np.float32)
    model = Booster(objective=TwoSquared(), config=TrainerConfig(n_trees=1, max_depth=0, learning_rate=.5)).fit(X, y, sample_weight=weights)
    # G_a=-24, G_b=-17, H=7, lambda=1. Weight is applied exactly once.
    result = model.predict_raw(X)
    np.testing.assert_array_equal(result['a'], np.full(4, 1.5, dtype=np.float32))
    np.testing.assert_array_equal(result['b'], np.full(4, 2.0625, dtype=np.float32))
    assert model.fit_report_['actual_device'] == 'cpu'


@pytest.mark.parametrize('fault', ['keys', 'shape', 'dtype', 'nan', 'negative_hess', 'alias', 'raw_alias'])
def test_bad_objective_output(data, fault):
    class Bad(TwoSquared):
        def step(self, raw, y, *args, **kwargs):
            output = super().step(raw, y, *args, **kwargs)
            g, h = output['a']
            if fault == 'keys':
                output.pop('b')
            elif fault == 'shape':
                output['a'] = (g[:, None], h)
            elif fault == 'dtype':
                output['a'] = (g.astype('float64'), h)
            elif fault == 'nan':
                g[0] = np.nan
            elif fault == 'negative_hess':
                h[0] = -1
            elif fault == 'alias':
                output['b'] = output['a']
            else:
                output['a'] = (raw['a'], h)
            return output
    model = Booster(objective=Bad(), config=TrainerConfig(n_trees=1))
    with pytest.raises((ValueError, TypeError)):
        model.fit(*data)
    with pytest.raises(RuntimeError, match='not fitted'):
        model.predict_raw(data[0])


@pytest.mark.parametrize('weight', [[0]*4, [-1,1,1,1], [np.nan,1,1,1], [np.inf,1,1,1], [1,1]])
def test_invalid_weights(data, weight):
    with pytest.raises(ValueError):
        Booster(objective=TwoSquared()).fit(*data, sample_weight=weight)


def test_target_shape_and_capabilities(data):
    with pytest.raises(ValueError, match='1D'):
        Booster(objective=TwoSquared()).fit(data[0], data[1][:, None])
    class Missing(TwoSquared):
        supported_devices = None
    with pytest.raises(ValueError, match='supported_devices'):
        Booster(objective=Missing()).fit(*data)


def test_builtin_adapter_parity(data):
    import openboost as ob
    from openboost._trainer import predict_raw
    X, y = data
    config = TrainerConfig(n_trees=2, max_depth=1)
    actual = Booster(objective=DistributionObjectiveAdapter('normal', natural=True), config=config).fit(X, y)
    reference = ob.NaturalBoostNormal(n_trees=2, max_depth=1).fit(X, y)
    for k, v in predict_raw(reference, X).items():
        np.testing.assert_array_equal(actual.predict_raw(X)[k], v)


def test_cuda_preflight_and_report(data):
    with pytest.raises(ValueError, match='CPU'):
        Booster(objective=TwoSquared(), device='cuda').fit(*data)
    with pytest.warns(RuntimeWarning, match='CPU'):
        model = Booster(objective=TwoSquared(), device='cuda', fallback='warn', config=TrainerConfig(n_trees=1)).fit(*data)
    assert model.fit_report_['requested_device'] == 'cuda'
    assert model.fit_report_['actual_device'] == 'cpu'


@pytest.mark.parametrize('kwargs', [{'n_trees': 0}, {'max_depth': 9}, {'n_bins': 255}, {'reg_lambda': -1}, {'min_gain': float('inf')}, {'subsample': 0}])
def test_unsupported_config_preflight(data, kwargs):
    model = Booster(objective=TwoSquared(), config=TrainerConfig(**kwargs))
    with pytest.raises(ValueError):
        model.fit(*data)
    assert not model.trees_


def test_plugin_inputs_and_context_are_readonly(data):
    class Inspect(TwoSquared):
        def step(self, raw, y, sample_weight=None, extra=None, *, context):
            with pytest.raises(ValueError):
                raw['a'][0] = 10
            with pytest.raises(AttributeError):
                context.round_idx = 20
            with pytest.raises(TypeError):
                raw['other'] = y
            return super().step(raw, y, sample_weight, extra, context=context)
    Booster(objective=Inspect(), config=TrainerConfig(n_trees=1)).fit(*data)


def test_seed_shared_with_builder_and_global_rng_unchanged():
    class RandomObjective(TwoSquared):
        def step(self, raw, y, sample_weight, extra, *, context):
            self.draw = context.rng.random()
            return super().step(raw, y, sample_weight, extra, context=context)

    from openboost.experimental import CPUHistogramBuilder

    class WatchBuilder(CPUHistogramBuilder):
        def build(self, binned, grad, hess, *, config, context):
            self.draw = context.rng.random()
            return super().build(binned, grad, hess, config=config, context=context)

    rng = np.random.default_rng(42)
    X = rng.normal(size=(32, 3)).astype(np.float32)
    y = X[:, 0].copy()
    before = np.random.get_state()
    models = []
    for _ in range(2):
        objective, builder = RandomObjective(), WatchBuilder()
        models.append(Booster(objective=objective, tree_builder=builder,
                              config=TrainerConfig(n_trees=2, max_depth=2, random_state=13,
                                                   subsample=.75, colsample_bytree=.67)).fit(X, y))
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    assert models[0].objective.draw == models[1].objective.draw
    assert models[0].tree_builder.draw == models[1].tree_builder.draw
    for k, v in models[0].predict_raw(X).items():
        np.testing.assert_array_equal(v, models[1].predict_raw(X)[k])
