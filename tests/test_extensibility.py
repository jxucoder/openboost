"""Tests for the extensibility layer.

Covers:
- register_loss / register_distribution / register_growth_strategy
- duplicate-name and override behavior
- input validation errors
- loss_value_fn: true loss values in history and early stopping
- device_loss decorator marker plumbing (no-op on CPU; device contract on GPU)
"""

import numpy as np
import pytest

import openboost as ob
from openboost._core._growth import _GROWTH_REGISTRY
from openboost._distributions import DISTRIBUTIONS
from openboost._loss import _LOSS_REGISTRY, _LOSS_VALUE_REGISTRY

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def _restore_registries():
    """Snapshot and restore all extension registries around each test."""
    loss_snap = dict(_LOSS_REGISTRY)
    loss_value_snap = dict(_LOSS_VALUE_REGISTRY)
    growth_snap = dict(_GROWTH_REGISTRY)
    dist_snap = dict(DISTRIBUTIONS)
    yield
    _LOSS_REGISTRY.clear()
    _LOSS_REGISTRY.update(loss_snap)
    _LOSS_VALUE_REGISTRY.clear()
    _LOSS_VALUE_REGISTRY.update(loss_value_snap)
    _GROWTH_REGISTRY.clear()
    _GROWTH_REGISTRY.update(growth_snap)
    DISTRIBUTIONS.clear()
    DISTRIBUTIONS.update(dist_snap)


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(300, 5)).astype(np.float32)
    y = (2.0 * X[:, 0] - X[:, 1] + rng.normal(scale=0.1, size=300)).astype(np.float32)
    return X, y


def _mse_grad(pred, y):
    """MSE gradients: grad = pred - y, hess = 1."""
    return (pred - y).astype(np.float32), np.ones_like(pred, dtype=np.float32)


# =============================================================================
# register_loss
# =============================================================================

class TestRegisterLoss:
    def test_train_end_to_end_by_name(self, data):
        X, y = data
        ob.register_loss('xt_mse', _mse_grad)
        model = ob.GradientBoosting(n_trees=30, max_depth=3, loss='xt_mse')
        model.fit(X, y)
        pred = model.predict(X)
        # Custom MSE clone should actually fit the data
        assert np.mean((pred - y) ** 2) < np.var(y) * 0.5

    def test_get_loss_function_resolves_registered_name(self):
        ob.register_loss('xt_loss_resolve', _mse_grad)
        assert ob.get_loss_function('xt_loss_resolve') is _mse_grad

    def test_unknown_loss_error_lists_registered_name(self):
        ob.register_loss('xt_listed', _mse_grad)
        with pytest.raises(ValueError, match='xt_listed'):
            ob.get_loss_function('definitely_not_registered')

    def test_duplicate_name_raises(self):
        ob.register_loss('xt_dup', _mse_grad)
        with pytest.raises(ValueError, match="'xt_dup'"):
            ob.register_loss('xt_dup', _mse_grad)

    def test_duplicate_builtin_name_raises(self):
        with pytest.raises(ValueError, match="'mse'"):
            ob.register_loss('mse', _mse_grad)

    def test_override_replaces(self):
        ob.register_loss('xt_ovr', _mse_grad)

        def other(pred, y):
            return _mse_grad(pred, y)

        ob.register_loss('xt_ovr', other, override=True)
        assert ob.get_loss_function('xt_ovr') is other

    def test_invalid_fn_raises_typeerror(self):
        with pytest.raises(TypeError, match='callable'):
            ob.register_loss('xt_bad_fn', 42)

    def test_invalid_name_raises_typeerror(self):
        with pytest.raises(TypeError, match='string'):
            ob.register_loss(123, _mse_grad)

    def test_invalid_loss_value_fn_raises_typeerror(self):
        with pytest.raises(TypeError, match='loss_value_fn'):
            ob.register_loss('xt_bad_lv', _mse_grad, loss_value_fn='not-callable')

    def test_builtin_losses_unchanged(self, data):
        """Built-in names keep working (registry refactor is behavior-preserving)."""
        X, y = data
        for loss in ('mse', 'mae', 'huber', 'quantile'):
            model = ob.GradientBoosting(n_trees=3, max_depth=2, loss=loss)
            model.fit(X, y)
            assert len(model.trees_) == 3

    def test_parameterized_builtin_kwargs_still_work(self):
        fn = ob.get_loss_function('quantile', quantile_alpha=0.9)
        pred = np.zeros(4, dtype=np.float32)
        y = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
        grad, hess = fn(pred, y)
        # Quantile gradient at alpha=0.9: -0.9 where y > pred
        assert np.allclose(grad[:2], -0.9, atol=1e-6)


# =============================================================================
# register_distribution
# =============================================================================

class TestRegisterDistribution:
    def test_train_end_to_end_by_name(self, data):
        X, y = data

        class MyNormal(ob.Normal):
            pass

        ob.register_distribution('xt_normal', MyNormal)
        model = ob.NaturalBoost(distribution='xt_normal', n_trees=5, max_depth=3)
        model.fit(X, y)
        assert isinstance(model.distribution_, MyNormal)
        out = model.predict_distribution(X)
        assert np.all(np.isfinite(out.params['loc']))

    def test_lookup_is_case_insensitive(self):
        class MyNormal(ob.Normal):
            pass

        ob.register_distribution('XT_CaseNormal', MyNormal)
        dist = ob.get_distribution('xt_casenormal')
        assert isinstance(dist, MyNormal)

    def test_duplicate_name_raises(self):
        class MyNormal(ob.Normal):
            pass

        ob.register_distribution('xt_dupdist', MyNormal)
        with pytest.raises(ValueError, match="'xt_dupdist'"):
            ob.register_distribution('xt_dupdist', MyNormal)

    def test_duplicate_builtin_name_raises(self):
        class MyNormal(ob.Normal):
            pass

        with pytest.raises(ValueError, match="'normal'"):
            ob.register_distribution('normal', MyNormal)

    def test_override_replaces(self):
        class A(ob.Normal):
            pass

        class B(ob.Normal):
            pass

        ob.register_distribution('xt_ovrdist', A)
        ob.register_distribution('xt_ovrdist', B, override=True)
        assert isinstance(ob.get_distribution('xt_ovrdist'), B)

    def test_invalid_class_raises_typeerror(self):
        class NotADistribution:
            pass

        with pytest.raises(TypeError, match='Distribution'):
            ob.register_distribution('xt_baddist', NotADistribution)

    def test_invalid_name_raises_typeerror(self):
        with pytest.raises(TypeError, match='string'):
            ob.register_distribution(None, ob.Normal)


# =============================================================================
# register_growth_strategy
# =============================================================================

class TestRegisterGrowthStrategy:
    def test_train_end_to_end_by_name(self, data):
        X, y = data

        class CountingGrowth(ob.LevelWiseGrowth):
            calls = 0

            def grow(self, *args, **kwargs):
                CountingGrowth.calls += 1
                return super().grow(*args, **kwargs)

        ob.register_growth_strategy('xt_growth', CountingGrowth)
        model = ob.GradientBoosting(n_trees=4, max_depth=3, growth='xt_growth')
        model.fit(X, y)
        assert CountingGrowth.calls == 4  # one grow per tree
        assert len(model.trees_) == 4
        pred = model.predict(X)
        assert np.all(np.isfinite(pred))

    def test_low_level_fit_tree_by_name(self, data):
        X, y = data

        class MyGrowth(ob.LevelWiseGrowth):
            pass

        ob.register_growth_strategy('xt_lowlevel', MyGrowth)
        X_binned = ob.array(X)
        grad, hess = _mse_grad(np.zeros(len(y), dtype=np.float32), y)
        tree = ob.fit_tree(X_binned, grad, hess, growth='xt_lowlevel')
        assert np.all(np.isfinite(tree(X_binned)))

    def test_lookup_is_case_insensitive(self):
        class MyGrowth(ob.LevelWiseGrowth):
            pass

        ob.register_growth_strategy('XT_CaseGrowth', MyGrowth)
        assert isinstance(ob.get_growth_strategy('xt_casegrowth'), MyGrowth)

    def test_duplicate_name_raises(self):
        class MyGrowth(ob.LevelWiseGrowth):
            pass

        ob.register_growth_strategy('xt_dupgrowth', MyGrowth)
        with pytest.raises(ValueError, match="'xt_dupgrowth'"):
            ob.register_growth_strategy('xt_dupgrowth', MyGrowth)

    def test_duplicate_builtin_name_raises(self):
        class MyGrowth(ob.LevelWiseGrowth):
            pass

        with pytest.raises(ValueError, match="'levelwise'"):
            ob.register_growth_strategy('levelwise', MyGrowth)

    def test_override_replaces(self):
        class A(ob.LevelWiseGrowth):
            pass

        class B(ob.LevelWiseGrowth):
            pass

        ob.register_growth_strategy('xt_ovrgrowth', A)
        ob.register_growth_strategy('xt_ovrgrowth', B, override=True)
        assert isinstance(ob.get_growth_strategy('xt_ovrgrowth'), B)

    def test_invalid_class_raises_typeerror(self):
        class NotAStrategy:
            pass

        with pytest.raises(TypeError, match='GrowthStrategy'):
            ob.register_growth_strategy('xt_badgrowth', NotAStrategy)

    def test_invalid_name_raises_typeerror(self):
        with pytest.raises(TypeError, match='string'):
            ob.register_growth_strategy(b'bytes', ob.LevelWiseGrowth)


# =============================================================================
# loss_value_fn: true loss values for custom objectives
# =============================================================================

class TestLossValueFn:
    def test_history_uses_registered_loss_value_fn(self, data):
        """train/val history must equal the true metric, not the Taylor proxy."""
        X, y = data

        # True value: shifted MAE — cannot coincide with the Taylor proxy of
        # MSE gradients (which equals MSE/2).
        def true_value(pred, y_):
            return float(np.mean(np.abs(pred - y_)) + 7.0)

        ob.register_loss('xt_valued', _mse_grad, loss_value_fn=true_value)

        hist = ob.HistoryCallback()
        model = ob.GradientBoosting(n_trees=5, max_depth=3, loss='xt_valued')
        model.fit(X, y, callbacks=[hist], eval_set=[(X, y)])

        final_pred = model.predict(X).astype(np.float64)
        expected = true_value(final_pred, y.astype(np.float64))
        proxy = float(np.mean((final_pred - y) ** 2) / 2.0)

        assert np.isclose(hist.history['train_loss'][-1], expected, rtol=1e-5)
        assert np.isclose(hist.history['val_loss'][-1], expected, rtol=1e-5)
        assert not np.isclose(hist.history['train_loss'][-1], proxy, rtol=1e-3)
        # All history entries carry the +7 shift, so they came from true_value
        assert all(v > 7.0 for v in hist.history['train_loss'])

    def test_taylor_proxy_still_used_without_value_fn(self, data):
        """Custom loss without a value hook keeps the Taylor proxy fallback."""
        X, y = data
        ob.register_loss('xt_proxied', _mse_grad)

        hist = ob.HistoryCallback()
        model = ob.GradientBoosting(n_trees=3, max_depth=3, loss='xt_proxied')
        model.fit(X, y, callbacks=[hist])

        final_pred = model.predict(X).astype(np.float64)
        proxy = float(np.mean((final_pred - y) ** 2) / 2.0)
        assert np.isclose(hist.history['train_loss'][-1], proxy, rtol=1e-5)

    def test_loss_value_attribute_on_direct_callable(self, data):
        """fn.loss_value on a callable passed directly to the model is used."""
        X, y = data

        def my_loss(pred, y_):
            return _mse_grad(pred, y_)

        my_loss.loss_value = lambda pred, y_: float(np.mean(np.abs(pred - y_)) + 3.0)

        hist = ob.HistoryCallback()
        model = ob.GradientBoosting(n_trees=4, max_depth=3, loss=my_loss)
        model.fit(X, y, callbacks=[hist])

        final_pred = model.predict(X).astype(np.float64)
        expected = float(np.mean(np.abs(final_pred - y)) + 3.0)
        assert np.isclose(hist.history['train_loss'][-1], expected, rtol=1e-5)

    def test_registered_value_fn_takes_precedence_over_attribute(self, data):
        """Documented precedence: register_loss kwarg > fn.loss_value attribute."""
        X, y = data

        def my_loss(pred, y_):
            return _mse_grad(pred, y_)

        my_loss.loss_value = lambda pred, y_: -123.0  # should be shadowed

        ob.register_loss(
            'xt_precedence', my_loss,
            loss_value_fn=lambda pred, y_: 456.0,
        )

        hist = ob.HistoryCallback()
        model = ob.GradientBoosting(n_trees=2, max_depth=2, loss='xt_precedence')
        model.fit(X, y, callbacks=[hist])
        assert hist.history['train_loss'] == [456.0, 456.0]

    def test_early_stopping_on_true_metric(self, data):
        """Early stopping must track loss_value_fn, not the (decreasing) proxy.

        MSE gradients make the Taylor proxy strictly decrease, so under the
        proxy training would never stop early. A strictly increasing
        loss_value_fn must therefore trigger early stopping quickly.
        """
        X, y = data

        counter = {'n': 0}

        def increasing_value(pred, y_):
            counter['n'] += 1
            return float(counter['n'])

        ob.register_loss('xt_early', _mse_grad, loss_value_fn=increasing_value)

        es = ob.EarlyStopping(patience=2, restore_best=True)
        model = ob.GradientBoosting(n_trees=50, max_depth=3, loss='xt_early')
        model.fit(X, y, callbacks=[es], eval_set=[(X, y)])

        assert es.stopped_round is not None
        assert es.stopped_round < 49
        # Best round is round 0 (metric only ever increases); restore_best
        # truncates the model to that round.
        assert es.best_round == 0
        assert len(model.trees_) < 50
        # The monitored best score is a value produced by loss_value_fn
        assert es.best_score == 2.0  # round 0: train call -> 1, val call -> 2


# =============================================================================
# device_loss decorator
# =============================================================================

class TestDeviceLoss:
    def test_decorator_sets_marker_and_returns_fn(self):
        def my_loss(pred, y_):
            return _mse_grad(pred, y_)

        marked = ob.device_loss(my_loss)
        assert marked is my_loss
        assert marked.__openboost_device__ is True

    def test_decorator_rejects_non_callable(self):
        with pytest.raises(TypeError, match='callable'):
            ob.device_loss('not-a-function')

    def test_marker_is_noop_on_cpu(self, data):
        """On the CPU backend a device-marked loss receives plain numpy arrays
        and training matches an unmarked copy of the same loss exactly."""
        X, y = data
        seen_types = []

        @ob.device_loss
        def marked_loss(pred, y_):
            seen_types.append((type(pred), type(y_)))
            return _mse_grad(pred, y_)

        def plain_loss(pred, y_):
            return _mse_grad(pred, y_)

        m1 = ob.GradientBoosting(n_trees=5, max_depth=3, loss=marked_loss, random_state=0)
        m1.fit(X, y)
        m2 = ob.GradientBoosting(n_trees=5, max_depth=3, loss=plain_loss, random_state=0)
        m2.fit(X, y)

        # Called once per round for gradients (plus once per round for the
        # Taylor-proxy loss value) — always with numpy arrays on CPU.
        assert len(seen_types) >= 5
        assert all(issubclass(p, np.ndarray) and issubclass(t, np.ndarray)
                   for p, t in seen_types)
        np.testing.assert_allclose(m1.predict(X), m2.predict(X), rtol=1e-6)

    def test_registered_device_marked_loss_on_cpu(self, data):
        """A device-marked loss registered by name also trains fine on CPU."""
        X, y = data

        @ob.device_loss
        def marked_loss(pred, y_):
            return _mse_grad(pred, y_)

        ob.register_loss('xt_device_cpu', marked_loss)
        model = ob.GradientBoosting(n_trees=5, max_depth=3, loss='xt_device_cpu')
        model.fit(X, y)
        assert np.all(np.isfinite(model.predict(X)))

    @pytest.mark.gpu
    def test_device_contract_on_gpu(self, data):
        """On CUDA, a device-marked loss receives DEVICE arrays and must
        return device (grad, hess); unmarked losses keep the host round-trip."""
        from numba import cuda

        X, y = data
        seen = {}

        @ob.device_loss
        def dev_mse(pred, y_):
            seen['pred_is_device'] = hasattr(pred, '__cuda_array_interface__')
            seen['y_is_device'] = hasattr(y_, '__cuda_array_interface__')
            pred_h = pred.copy_to_host()
            y_h = y_.copy_to_host()
            grad = (pred_h - y_h).astype(np.float32)
            hess = np.ones_like(grad, dtype=np.float32)
            return cuda.to_device(grad), cuda.to_device(hess)

        host_seen = {}

        def host_mse(pred, y_):
            host_seen['pred_is_host'] = isinstance(pred, np.ndarray)
            return _mse_grad(pred, y_)

        with ob.backend_context('cuda'):
            m_dev = ob.GradientBoosting(n_trees=5, max_depth=3, loss=dev_mse)
            m_dev.fit(X, y)
            m_host = ob.GradientBoosting(n_trees=5, max_depth=3, loss=host_mse)
            m_host.fit(X, y)

        assert seen['pred_is_device'] is True
        assert seen['y_is_device'] is True
        assert host_seen['pred_is_host'] is True
        np.testing.assert_allclose(
            m_dev.predict(X), m_host.predict(X), rtol=1e-4, atol=1e-4
        )
