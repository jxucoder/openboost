"""Real-device execution boundaries for the existing unified trainer."""

import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def fixture_data():
    X = np.repeat(np.array([[0.], [1.]], dtype=np.float32), 32, axis=0)
    y = np.tile(np.array([1, 2, 3, 4], dtype=np.float32), 16) + 5 * X[:, 0]
    return X, y


@pytest.mark.parametrize('mode', ['custom', 'exposure', 'generic'])
def test_visible_fallback(mode, checks):
    import openboost as ob
    from openboost._distributions import Normal as BuiltinNormal

    class Normal(BuiltinNormal):
        def natural_gradient(self, y, params):
            result = super().natural_gradient(y, params)
            g, h = result['loc']
            result['loc'] = (g * 0.5, h)
            return result

    X, y = fixture_data()
    kwargs, extra = {}, {}
    if mode == 'custom':
        kwargs['distribution'] = Normal()
    elif mode == 'exposure':
        kwargs['distribution'] = 'poisson'
        extra['exposure'] = np.tile(np.array([1, 2], dtype=np.float32), 32)
    else:
        kwargs['reg_alpha'] = 0.2
    predictions = {}
    for backend in ('cpu', 'cuda'):
        with ob.backend_context(backend), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = ob.NaturalBoost(n_trees=3, max_depth=1, **kwargs).fit(X, y, **extra)
            predictions[backend] = model.predict(X, **extra)
        if backend == 'cuda':
            messages = [str(w.message) for w in caught if 'fallback' in str(w.message)]
            assert messages, 'Fallback was silent'
            checks[f'fallback_{mode}'] = messages
    np.testing.assert_allclose(predictions['cpu'], predictions['cuda'], rtol=2e-5, atol=2e-6)


def test_device_error_rolls_back(monkeypatch, checks):
    import openboost as ob
    import openboost._backends._cuda as kernels

    def fail(*args, **kwargs):
        raise RuntimeError('deliberate kernel failure')

    monkeypatch.setattr(kernels, 'normal_step_gpu', fail)
    X, y = fixture_data()
    with ob.backend_context('cuda'):
        model = ob.NaturalBoost(n_trees=2)
        with pytest.raises(RuntimeError, match='deliberate kernel failure'):
            model.fit(X, y)
        assert not model.trees_ and model.X_binned_ is None
        with pytest.raises(RuntimeError, match='not fitted'):
            model.predict(X)
    checks['device_error_propagated'] = True


@pytest.mark.parametrize('parameter', ['subsample', 'colsample_bytree'])
def test_device_sampling_preflight(parameter, checks):
    import openboost as ob

    X, y = fixture_data()
    with ob.backend_context('cuda'):
        model = ob.NaturalBoost(**{parameter: 0.5})
        with pytest.raises(ValueError, match='sampling'):
            model.fit(X, y)
        assert not model.trees_ and model.X_binned_ is None
    checks[f'preflight_{parameter}'] = True


@pytest.mark.parametrize('distribution', ['normal', 'poisson'])
def test_eval_callback_persistence(distribution, monkeypatch, tmp_path, checks):
    import openboost as ob
    import openboost._trainer as trainer
    from openboost._callbacks import Callback

    X, y = fixture_data()
    host_raw_calls = []
    original = trainer._as_host_raw

    def host_raw(raw):
        assert all(hasattr(v, '__cuda_array_interface__') for v in raw.values())
        result = original(raw)
        assert all(isinstance(v, np.ndarray) for v in result.values())
        host_raw_calls.append(len(result))
        return result

    class Recorder(Callback):
        def __init__(self):
            self.values = []

        def on_round_end(self, state):
            assert np.isfinite(state.train_loss) and np.isfinite(state.val_loss)
            self.values.append((state.train_loss, state.val_loss))
            return True

    recorder = Recorder()
    with ob.backend_context('cuda'):
        with monkeypatch.context() as patch:
            patch.setattr(trainer, '_as_host_raw', host_raw)
            plain = ob.NaturalBoost(distribution=distribution, n_trees=3, max_depth=1).fit(X, y)
            assert host_raw_calls == [], 'Training raw scores downloaded without callback/eval'
            evaluated = ob.NaturalBoost(distribution=distribution, n_trees=3, max_depth=1).fit(
                X, y, callbacks=[recorder], eval_set=[(X, y)])
            assert len(host_raw_calls) == 3
        expected = plain.predict_params(X)
        actual = evaluated.predict_params(X)
        for k in expected:
            np.testing.assert_allclose(actual[k], expected[k], rtol=2e-5, atol=2e-6)
        assert len(recorder.values) == 3
        np.testing.assert_allclose(recorder.values[-1], evaluated.nll(X, y), rtol=2e-5)
        gpu_path = tmp_path / 'gpu.ob'
        evaluated.save(gpu_path)
    with ob.backend_context('cpu'):
        with pytest.warns(UserWarning, match='pickle'):
            loaded = ob.load(gpu_path)
        for k, v in loaded.predict_params(X).items():
            np.testing.assert_allclose(v, expected[k], rtol=2e-5, atol=2e-6)
        cpu_model = ob.NaturalBoost(distribution=distribution, n_trees=3, max_depth=1).fit(X, y)
        cpu_expected = cpu_model.predict_params(X)
        for k in expected:
            np.testing.assert_allclose(cpu_expected[k], expected[k], rtol=2e-5, atol=2e-6)
        cpu_path = tmp_path / 'cpu.ob'
        cpu_model.save(cpu_path)
    with ob.backend_context('cuda'):
        with pytest.warns(UserWarning, match='pickle'):
            loaded = ob.load(cpu_path)
        for k, v in loaded.predict_params(X).items():
            np.testing.assert_allclose(v, cpu_expected[k], rtol=2e-5, atol=2e-6)
    checks[f'boundaries_{distribution}'] = {
        'callback_rounds': 3, 'raw_host_calls_with_callback': len(host_raw_calls),
        'bidirectional_persistence': True, 'eval_final_nll': recorder.values[-1][1],
    }
