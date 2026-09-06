"""Real CUDA weighted Newton regression, using fixed bins and host oracles."""

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_weighted_newton(checks, monkeypatch):
    from numba import cuda

    import openboost as ob
    import openboost._trainer as trainer
    from openboost._array import BinnedArray
    from openboost._backends._cuda import _build_histogram_shared_kernel
    from openboost._core._histogram import build_histogram

    bins = np.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8)
    weights = np.array([0, 1, 2, 4, 0, 2, 3, 5], dtype=np.float32)
    direction = np.array([-3, -3, -3, -3, 2, 2, 2, 2], dtype=np.float32)
    grad, hess = direction * weights, weights.copy()
    expected_leaves = np.array([21 / 8, -20 / 11], dtype=np.float32)
    expected_raw = 0.1 * np.repeat(expected_leaves, 4)
    binned = BinnedArray(bins, [np.array([0.5])], 1, 8, 'cpu')

    class FixedObjective:
        channel_names = ['value']
        device_capable = False
        unit_hessian = True

        def init_raw(self, *args):
            return {'value': 0.0}

        def step(self, raw, y, sample_weight, extra):
            return {'value': (direction * sample_weight, sample_weight.copy())}

    with ob.backend_context('cpu'):
        hist_cpu = build_histogram(bins, grad, hess)
        cpu = SimpleNamespace()
        trainer.fit_boosting(cpu, FixedObjective(), binned, direction,
                             config=trainer.TrainerConfig(n_trees=1, max_depth=1), sample_weight=weights)
        raw_cpu = trainer.predict_raw(cpu, binned)['value']
    original = trainer.fit_tree_gpu_native
    captured = {}

    def native(x, g, h, **kwargs):
        hist = cuda.to_device(np.zeros((1, 1, 256, 2), dtype=np.float32))
        nodes = cuda.to_device(np.zeros(8, dtype=np.int32))
        _build_histogram_shared_kernel[(1, 1), 256](
            x, g, h, nodes, 0, 1, 0, hist, np.float32(kwargs['const_hess']))
        captured['hist'] = hist.copy_to_host()[0, 0]
        return original(x, g, h, **kwargs)

    monkeypatch.setattr(trainer, 'fit_tree_gpu_native', native)
    with ob.backend_context('cuda'):
        gpu = SimpleNamespace()
        trainer.fit_boosting(gpu, FixedObjective(), binned, direction,
                             config=trainer.TrainerConfig(n_trees=1, max_depth=1), sample_weight=weights)
        raw_gpu = trainer.predict_raw(gpu, binned)['value']
    checks['weighted_newton'] = {
        'cpu_hist_hess': hist_cpu[1][0, :2].tolist(),
        'gpu_hist_hess': captured['hist'][:2, 1].tolist(),
        'expected_leaves': expected_leaves.tolist(),
        'cpu_raw': raw_cpu.tolist(), 'gpu_raw': raw_gpu.tolist(),
    }
    np.testing.assert_allclose(raw_cpu, expected_raw, rtol=1e-6)
    np.testing.assert_allclose(captured['hist'][:, 0], hist_cpu[0][0], atol=1e-6)
    np.testing.assert_allclose(captured['hist'][:, 1], hist_cpu[1][0], atol=1e-6)
    np.testing.assert_allclose(raw_gpu, expected_raw, rtol=1e-6)
    assert gpu.trees_['value'][0].features[0] == 0
    assert gpu.trees_['value'][0].thresholds[0] == 0


@pytest.mark.parametrize('distribution', ['normal', 'poisson'])
def test_weighted_distribution(distribution, checks):
    from numba import cuda

    import openboost as ob
    from openboost._objectives import DistributionObjective
    from openboost._trainer import predict_raw

    # Two distinct bins and no near-tied alternative splits. Reuse CPU bins.
    X = np.repeat(np.array([[0.], [1.]], dtype=np.float32), 32, axis=0)
    y = np.tile(np.array([1, 2, 3, 4], dtype=np.float32), 16) + X[:, 0] * 5
    weights = np.tile(np.array([0, 1, 2, 4], dtype=np.float32), 16)
    models, raws, metrics, gradients = {}, {}, {}, {}
    with ob.backend_context('cpu'):
        binned = ob.array(X)
    for backend in ('cpu', 'cuda'):
        with ob.backend_context(backend):
            model = ob.NaturalBoost(distribution=distribution, n_trees=3, max_depth=1)
            model.fit(binned, y, sample_weight=weights)
            models[backend] = model
            raws[backend] = predict_raw(model, binned)
            metrics[backend] = float(model.nll(binned, y))
            objective = DistributionObjective(model.distribution_, natural=True)
            raw = {k: np.full(len(y), v, dtype=np.float32) for k, v in model._base_scores.items()}
            if backend == 'cuda':
                output = objective.step({k: cuda.to_device(v) for k, v in raw.items()},
                                        cuda.to_device(y), cuda.to_device(weights))
                gradients[backend] = {k: tuple(a.copy_to_host() for a in pair) for k, pair in output.items()}
            else:
                gradients[backend] = objective.step(raw, y, weights)
    checks[f'weighted_{distribution}'] = {
        'nll': metrics,
        'raw_max_abs_error': max(float(np.max(np.abs(raws['cpu'][k] - raws['cuda'][k]))) for k in raws['cpu']),
    }
    for channel in raws['cpu']:
        for a, b in zip(gradients['cpu'][channel], gradients['cuda'][channel], strict=True):
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(raws['cpu'][channel], raws['cuda'][channel], rtol=2e-5, atol=2e-6)
        for a, b in zip(models['cpu'].trees_[channel], models['cuda'].trees_[channel], strict=True):
            assert a.features[0] == b.features[0]
            assert a.thresholds[0] == b.thresholds[0]
    np.testing.assert_allclose(metrics['cpu'], metrics['cuda'], rtol=2e-5)
