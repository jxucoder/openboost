"""Tests for sklearn-compatible wrappers and Phase 13 features.

Tests cover:
- OpenBoostRegressor and OpenBoostClassifier
- Callback system (EarlyStopping, Logger, etc.)
- Feature importance utilities
- sklearn compatibility (GridSearchCV, Pipeline, etc.)
"""

import numpy as np
import pytest

import openboost as ob
from openboost import (
    EarlyStopping,
    GradientBoosting,
    HistoryCallback,
    OpenBoostClassifier,
    OpenBoostDistributionalRegressor,
    OpenBoostGAMClassifier,
    OpenBoostGAMRegressor,
    OpenBoostLinearLeafRegressor,
    OpenBoostRegressor,
    compute_feature_importances,
    get_feature_importance_dict,
)

# Skip tests if sklearn not installed
sklearn = pytest.importorskip("sklearn")
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TestOpenBoostRegressor:
    """Tests for OpenBoostRegressor."""
    
    def test_fit_predict(self):
        """Basic fit and predict works."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(100).astype(np.float32) * 0.1
        
        reg = OpenBoostRegressor(n_estimators=10, max_depth=3)
        reg.fit(X, y)
        pred = reg.predict(X)
        
        assert pred.shape == y.shape
        assert pred.dtype == np.float32
    
    def test_score(self):
        """R² score works."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0] + 0.5 * X[:, 1]
        
        reg = OpenBoostRegressor(n_estimators=20, max_depth=4)
        reg.fit(X, y)
        score = reg.score(X, y)
        
        # Should fit well on training data
        assert score > 0.5
    
    def test_get_set_params(self):
        """sklearn get_params/set_params work."""
        reg = OpenBoostRegressor(n_estimators=50, max_depth=5, learning_rate=0.2)
        
        params = reg.get_params()
        assert params['n_estimators'] == 50
        assert params['max_depth'] == 5
        assert params['learning_rate'] == 0.2
        
        reg.set_params(n_estimators=100)
        assert reg.n_estimators == 100
    
    def test_clone(self):
        """sklearn clone works."""
        reg = OpenBoostRegressor(n_estimators=50, max_depth=5)
        reg_clone = clone(reg)
        
        assert reg_clone.n_estimators == 50
        assert reg_clone.max_depth == 5
        assert reg_clone is not reg
    
    def test_feature_importances(self):
        """Feature importances work."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0] * 2 + X[:, 1]  # Features 0 and 1 are important
        
        reg = OpenBoostRegressor(n_estimators=20, max_depth=4)
        reg.fit(X, y)
        
        importances = reg.feature_importances_
        
        assert importances.shape == (5,)
        assert np.isclose(importances.sum(), 1.0)  # Normalized
        # First two features should be most important
        assert importances[0] > importances[4] or importances[1] > importances[4]
    
    def test_early_stopping(self):
        """Early stopping works with eval_set."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] + np.random.randn(200).astype(np.float32) * 0.1
        
        X_train, X_val = X[:150], X[150:]
        y_train, y_val = y[:150], y[150:]
        
        reg = OpenBoostRegressor(
            n_estimators=1000,  # High number
            max_depth=3,
            early_stopping_rounds=10,
        )
        reg.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        
        # Should have stopped early
        assert len(reg.booster_.trees_) < 1000
        assert hasattr(reg, 'best_iteration_')
    
    def test_gridsearch(self):
        """Works with GridSearchCV."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0] + np.random.randn(100).astype(np.float32) * 0.1
        
        param_grid = {
            'n_estimators': [5, 10],
            'max_depth': [2, 3],
        }
        
        search = GridSearchCV(
            OpenBoostRegressor(learning_rate=0.3),
            param_grid,
            cv=2,
            scoring='r2',
        )
        search.fit(X, y)
        
        assert search.best_params_ is not None
        assert 'n_estimators' in search.best_params_
    
    def test_cross_val_score(self):
        """Works with cross_val_score."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0] + np.random.randn(100).astype(np.float32) * 0.1
        
        scores = cross_val_score(
            OpenBoostRegressor(n_estimators=10, max_depth=3),
            X, y,
            cv=3,
        )
        
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)
    
    def test_pipeline(self):
        """Works in sklearn Pipeline."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0] + np.random.randn(100).astype(np.float32) * 0.1
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', OpenBoostRegressor(n_estimators=10, max_depth=3)),
        ])
        
        pipe.fit(X, y)
        pred = pipe.predict(X)
        score = pipe.score(X, y)
        
        assert pred.shape == y.shape
        assert isinstance(score, float)


class TestOpenBoostClassifier:
    """Tests for OpenBoostClassifier."""
    
    def test_binary_classification(self):
        """Binary classification works."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int32)
        
        clf = OpenBoostClassifier(n_estimators=10, max_depth=3)
        clf.fit(X, y)
        pred = clf.predict(X)
        
        assert pred.shape == y.shape
        assert set(pred).issubset({0, 1})
    
    def test_predict_proba_binary(self):
        """Probability predictions work for binary."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int32)
        
        clf = OpenBoostClassifier(n_estimators=10, max_depth=3)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        
        assert proba.shape == (100, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all() and (proba <= 1).all()
    
    def test_multiclass_classification(self):
        """Multi-class classification works."""
        np.random.seed(42)
        X = np.random.randn(150, 5).astype(np.float32)
        y = np.array([0] * 50 + [1] * 50 + [2] * 50)
        
        clf = OpenBoostClassifier(n_estimators=10, max_depth=3)
        clf.fit(X, y)
        pred = clf.predict(X)
        proba = clf.predict_proba(X)
        
        assert set(pred).issubset({0, 1, 2})
        assert proba.shape == (150, 3)
        assert clf.n_classes_ == 3
    
    def test_classes_attribute(self):
        """classes_ attribute set correctly."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.array(['cat'] * 50 + ['dog'] * 50)
        
        clf = OpenBoostClassifier(n_estimators=10, max_depth=3)
        clf.fit(X, y)
        
        assert hasattr(clf, 'classes_')
        assert len(clf.classes_) == 2
        assert set(clf.classes_) == {'cat', 'dog'}
        
        # Predictions should return original labels
        pred = clf.predict(X)
        assert set(pred).issubset({'cat', 'dog'})
    
    def test_score(self):
        """Accuracy score works."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int32)
        
        clf = OpenBoostClassifier(n_estimators=20, max_depth=4)
        clf.fit(X, y)
        score = clf.score(X, y)
        
        # Should classify well on training data
        assert score > 0.7
    
    def test_feature_importances_classifier(self):
        """Feature importances work for classifier."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int32)
        
        clf = OpenBoostClassifier(n_estimators=20, max_depth=4)
        clf.fit(X, y)
        
        importances = clf.feature_importances_
        
        assert importances.shape == (5,)
        assert np.isclose(importances.sum(), 1.0)


class TestCallbacks:
    """Tests for callback system."""
    
    def test_early_stopping_callback(self):
        """EarlyStopping callback works."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] + np.random.randn(200).astype(np.float32) * 0.1
        
        X_train, X_val = X[:150], X[150:]
        y_train, y_val = y[:150], y[150:]
        
        callback = EarlyStopping(patience=5, verbose=False)
        
        model = GradientBoosting(n_trees=1000, max_depth=3)
        model.fit(X_train, y_train, 
                  callbacks=[callback],
                  eval_set=[(X_val, y_val)])
        
        # Should have stopped early
        assert len(model.trees_) < 1000
        assert hasattr(model, 'best_iteration_')
        assert callback.best_round >= 0
    
    def test_history_callback(self):
        """HistoryCallback records losses."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0] + np.random.randn(100).astype(np.float32) * 0.1
        
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]
        
        history = HistoryCallback()
        
        model = GradientBoosting(n_trees=20, max_depth=3)
        model.fit(X_train, y_train,
                  callbacks=[history],
                  eval_set=[(X_val, y_val)])
        
        assert len(history.history['train_loss']) == 20
        assert len(history.history['val_loss']) == 20
        
        # Loss should generally decrease
        assert history.history['train_loss'][-1] < history.history['train_loss'][0]
    
    def test_multiple_callbacks(self):
        """Multiple callbacks work together."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0] + np.random.randn(100).astype(np.float32) * 0.1
        
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]
        
        history = HistoryCallback()
        early_stop = EarlyStopping(patience=5, verbose=False, restore_best=False)
        
        model = GradientBoosting(n_trees=100, max_depth=3)
        model.fit(X_train, y_train,
                  callbacks=[history, early_stop],
                  eval_set=[(X_val, y_val)])
        
        # Both callbacks should have run
        assert len(history.history['train_loss']) > 0
        # History length equals actual training rounds (trees count)
        assert len(history.history['train_loss']) == len(model.trees_)


class TestFeatureImportance:
    """Tests for feature importance utilities."""
    
    def test_compute_importances_gradient_boosting(self):
        """compute_feature_importances works with GradientBoosting."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0] * 2 + X[:, 1]
        
        model = GradientBoosting(n_trees=20, max_depth=4)
        model.fit(X, y)
        
        importances = compute_feature_importances(model)
        
        assert importances.shape == (5,)
        assert np.isclose(importances.sum(), 1.0)
    
    def test_compute_importances_dart(self):
        """compute_feature_importances works with DART."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0] * 2 + X[:, 1]
        
        model = ob.DART(n_trees=20, max_depth=4, seed=42)
        model.fit(X, y)
        
        importances = compute_feature_importances(model)
        
        assert importances.shape == (5,)
        assert np.isclose(importances.sum(), 1.0)
    
    def test_importance_dict(self):
        """get_feature_importance_dict works."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0] * 2 + X[:, 1]
        
        model = GradientBoosting(n_trees=20, max_depth=4)
        model.fit(X, y)
        
        importance_dict = get_feature_importance_dict(
            model,
            feature_names=['a', 'b', 'c', 'd', 'e'],
            top_n=3
        )
        
        assert len(importance_dict) == 3
        assert all(isinstance(k, str) for k in importance_dict)
        # Values are numpy float32, check they're numeric
        assert all(np.issubdtype(type(v), np.floating) for v in importance_dict.values())
    
    def test_importance_types(self):
        """Different importance types work."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0] * 2 + X[:, 1]
        
        model = GradientBoosting(n_trees=20, max_depth=4)
        model.fit(X, y)
        
        # Frequency (default)
        imp_freq = compute_feature_importances(model, importance_type='frequency')
        assert imp_freq.shape == (5,)
        
        # Gain (falls back to frequency if not stored)
        imp_gain = compute_feature_importances(model, importance_type='gain')
        assert imp_gain.shape == (5,)


class TestOpenBoostDistributionalRegressor:
    """Tests for OpenBoostDistributionalRegressor fit kwargs and early stopping."""

    def _make_data(self, n=200, seed=42):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, 5).astype(np.float32)
        y = X[:, 0] + rng.randn(n).astype(np.float32) * 0.1
        return X, y

    def test_fit_predict(self):
        """Basic fit and predict works."""
        X, y = self._make_data(100)
        reg = OpenBoostDistributionalRegressor(n_estimators=10, max_depth=3)
        reg.fit(X, y)
        pred = reg.predict(X)
        assert pred.shape == y.shape

    def test_sample_weight_forwarded(self):
        """Skewed sample weights actually change the fitted model."""
        rng = np.random.RandomState(42)
        n = 200
        X = rng.randn(n, 5).astype(np.float32)
        y = X[:, 0] + rng.randn(n).astype(np.float32) * 0.1
        y[:100] += 3.0  # index-based shift X cannot explain

        weights = np.array([50.0] * 100 + [0.02] * 100, dtype=np.float32)

        base = OpenBoostDistributionalRegressor(n_estimators=20, max_depth=3)
        weighted = clone(base)
        base.fit(X, y)
        weighted.fit(X, y, sample_weight=weights)

        pred_base = base.predict(X)
        pred_weighted = weighted.predict(X)
        assert not np.allclose(pred_base, pred_weighted)
        # The weighted fit must track the upweighted half more closely
        mse_weighted = np.mean((pred_weighted[:100] - y[:100]) ** 2)
        mse_base = np.mean((pred_base[:100] - y[:100]) ** 2)
        assert mse_weighted < mse_base

    def test_unknown_kwarg_raises(self):
        """Unknown fit kwargs raise TypeError, like sklearn."""
        X, y = self._make_data(100)
        reg = OpenBoostDistributionalRegressor(n_estimators=5)
        with pytest.raises(TypeError):
            reg.fit(X, y, not_a_real_kwarg=1)

    def test_callbacks_and_eval_set_forwarded(self):
        """callbacks and eval_set reach the underlying booster."""
        X, y = self._make_data(150)
        X_train, X_val = X[:100], X[100:]
        y_train, y_val = y[:100], y[100:]

        history = HistoryCallback()
        reg = OpenBoostDistributionalRegressor(n_estimators=10, max_depth=3)
        reg.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[history])

        assert len(history.history['train_loss']) == 10
        assert len(history.history['val_loss']) == 10

    def test_early_stopping(self):
        """eval_set + early_stopping_rounds actually stops early."""
        X, y = self._make_data(200)
        X_train, X_val = X[:150], X[150:]
        y_train, y_val = y[:150], y[150:]

        n_estimators = 300
        reg = OpenBoostDistributionalRegressor(
            n_estimators=n_estimators,  # High number; val NLL plateaus long before
            max_depth=2,
            early_stopping_rounds=5,
        )
        reg.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        assert hasattr(reg, 'best_iteration_')
        assert reg.best_iteration_ == reg.booster_.best_iteration_
        # One ensemble per distribution parameter; each restored to best round
        for trees in reg.booster_.trees_.values():
            assert len(trees) < n_estimators
            assert len(trees) == reg.best_iteration_ + 1

    def test_exposure_poisson_end_to_end(self):
        """exposure flows through fit, eval_set 3-tuples, and predict methods."""
        rng = np.random.RandomState(0)
        n = 300
        X = rng.randn(n, 3).astype(np.float32)
        exposure = rng.uniform(0.5, 2.0, n).astype(np.float32)
        rate = np.exp(0.4 * X[:, 0])
        y = rng.poisson(rate * exposure).astype(np.float32)

        reg = OpenBoostDistributionalRegressor(
            distribution='poisson', n_estimators=20, max_depth=3,
        )
        reg.fit(
            X[:200], y[:200],
            exposure=exposure[:200],
            eval_set=[(X[200:], y[200:], exposure[200:])],
        )

        # Exposure-aware (3-tuple) eval set is evaluated every round
        assert len(reg.evals_result_['eval_0']['nll']) == 20

        # Mean scales multiplicatively with exposure
        ones = np.ones(100, dtype=np.float32)
        pred_1 = reg.predict(X[200:], exposure=ones)
        pred_2 = reg.predict(X[200:], exposure=2 * ones)
        assert np.allclose(pred_2, 2 * pred_1, rtol=1e-4)

        # exposure=None means exposure 1
        assert np.allclose(reg.predict(X[200:]), pred_1, rtol=1e-6)

        # Other predict methods accept exposure through the wrapper
        lower, upper = reg.predict_interval(X[200:], alpha=0.2, exposure=ones)
        assert (lower <= upper).all()
        q90 = reg.predict_quantile(X[200:], 0.9, exposure=2 * ones)
        assert q90.shape == (100,)
        samples = reg.sample(X[200:], n_samples=3, seed=0, exposure=ones)
        assert samples.shape == (100, 3)
        nll = reg.nll_score(X[200:], y[200:], exposure=exposure[200:])
        assert np.isfinite(nll)

    def test_exposure_unsupported_family_raises(self):
        """exposure with a non-log-link family raises ValueError."""
        X, y = self._make_data(100)
        reg = OpenBoostDistributionalRegressor(distribution='normal', n_estimators=5)
        with pytest.raises(ValueError, match="exposure is not supported"):
            reg.fit(X, y, exposure=np.ones(100, dtype=np.float32))

    def test_eval_metric_and_early_stopping(self):
        """Non-default eval_metric records evals_result_ and stops early."""
        X, y = self._make_data(300)
        X_train, X_val = X[:200], X[200:]
        y_train, y_val = y[:200], y[200:]

        n_estimators = 300
        reg = OpenBoostDistributionalRegressor(
            n_estimators=n_estimators,  # High number; val CRPS plateaus long before
            max_depth=2,
            eval_metric='crps',
            early_stopping_rounds=5,
        )
        reg.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        history = reg.evals_result_['eval_0']['crps']
        rounds_run = len(history)
        assert 0 < rounds_run < n_estimators  # stopped early
        assert all(np.isfinite(v) for v in history)

        assert reg.best_iteration_ == reg.booster_.best_iteration_
        assert reg.best_score_ == reg.booster_.best_score_
        for trees in reg.booster_.trees_.values():
            assert len(trees) == reg.best_iteration_ + 1

    def test_evals_result_requires_fit(self):
        """evals_result_ raises NotFittedError before fit."""
        reg = OpenBoostDistributionalRegressor(n_estimators=5)
        with pytest.raises(NotFittedError):
            _ = reg.evals_result_

    def test_no_early_stopping_attributes_absent(self):
        """Without early stopping, best_iteration_/best_score_ stay absent."""
        X, y = self._make_data(80)
        reg = OpenBoostDistributionalRegressor(n_estimators=5)
        reg.fit(X, y)
        assert not hasattr(reg, 'best_iteration_')
        assert not hasattr(reg, 'best_score_')
        assert reg.evals_result_ == {}

    def test_clone_get_params_with_new_params(self):
        """clone/get_params/set_params work with the new constructor params."""
        reg = OpenBoostDistributionalRegressor(
            distribution='gamma',
            n_estimators=15,
            eval_metric='interval_score',
            quantiles=[0.1, 0.9],
            interval_alpha=0.2,
        )
        params = reg.get_params()
        assert params['eval_metric'] == 'interval_score'
        assert params['quantiles'] == [0.1, 0.9]
        assert params['interval_alpha'] == 0.2

        reg_clone = clone(reg)
        assert reg_clone is not reg
        assert reg_clone.eval_metric == 'interval_score'
        assert reg_clone.quantiles == [0.1, 0.9]
        assert reg_clone.interval_alpha == 0.2

        reg.set_params(eval_metric='nll', quantiles=None)
        assert reg.eval_metric == 'nll'
        assert reg.quantiles is None


class TestOpenBoostLinearLeafRegressorSklearnCompat:
    """sklearn-compat tests for OpenBoostLinearLeafRegressor."""

    def _make_data(self, n=80, seed=42):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, 3).astype(np.float32)
        y = 2.0 * X[:, 0] + X[:, 1] + rng.randn(n).astype(np.float32) * 0.1
        return X, y

    def test_clone(self):
        """sklearn clone works."""
        reg = OpenBoostLinearLeafRegressor(n_estimators=7, max_depth=2)
        reg_clone = clone(reg)
        assert reg_clone.n_estimators == 7
        assert reg_clone.max_depth == 2
        assert reg_clone is not reg

    def test_get_set_params(self):
        """get_params/set_params roundtrip works."""
        reg = OpenBoostLinearLeafRegressor(n_estimators=7, learning_rate=0.2)
        params = reg.get_params()
        assert params['n_estimators'] == 7
        assert params['learning_rate'] == 0.2

        reg.set_params(**params)
        assert reg.get_params() == params
        reg.set_params(n_estimators=15)
        assert reg.n_estimators == 15

    def test_pipeline(self):
        """fit/predict works in a Pipeline."""
        X, y = self._make_data()
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', OpenBoostLinearLeafRegressor(n_estimators=5, max_depth=2)),
        ])
        pipe.fit(X, y)
        pred = pipe.predict(X)
        assert pred.shape == y.shape
        assert isinstance(pipe.score(X, y), float)

    def test_gridsearch(self):
        """Works with GridSearchCV (2-point grid)."""
        X, y = self._make_data()
        search = GridSearchCV(
            OpenBoostLinearLeafRegressor(max_depth=2),
            {'n_estimators': [3, 5]},
            cv=2,
        )
        search.fit(X, y)
        assert search.best_params_['n_estimators'] in (3, 5)

    def test_not_fitted_error(self):
        """predict before fit raises NotFittedError."""
        X, _ = self._make_data()
        reg = OpenBoostLinearLeafRegressor(n_estimators=5)
        with pytest.raises(NotFittedError):
            reg.predict(X)

    def test_unsupported_fit_kwargs_raise(self):
        """Unsupported fit kwargs raise instead of being silently dropped."""
        X, y = self._make_data()
        reg = OpenBoostLinearLeafRegressor(n_estimators=5)
        with pytest.raises(NotImplementedError, match="sample_weight"):
            reg.fit(X, y, sample_weight=np.ones(len(y), dtype=np.float32))
        with pytest.raises(TypeError):
            reg.fit(X, y, not_a_real_kwarg=1)

    def _make_es_data(self):
        """Train/val split prone to overfitting, for early-stopping tests."""
        rng = np.random.RandomState(123)
        X = rng.randn(250, 6).astype(np.float32)
        y = (1.5 * X[:, 0] - X[:, 2] + 0.3 * rng.randn(250)).astype(np.float32)
        X_val = rng.randn(80, 6).astype(np.float32)
        y_val = (1.5 * X_val[:, 0] - X_val[:, 2] + 0.3 * rng.randn(80)).astype(
            np.float32
        )
        return X, y, X_val, y_val

    def test_callbacks_and_eval_set_forwarded(self):
        """callbacks and eval_set reach the underlying booster."""
        X, y, X_val, y_val = self._make_es_data()

        history = HistoryCallback()
        reg = OpenBoostLinearLeafRegressor(n_estimators=8, max_depth=3)
        reg.fit(X, y, eval_set=[(X_val, y_val)], callbacks=[history])

        assert len(history.history['train_loss']) == 8
        assert len(history.history['val_loss']) == 8
        assert len(reg.evals_result_['eval_0']['mse']) == 8

    def test_early_stopping(self):
        """early_stopping_rounds constructor param stops training early."""
        X, y, X_val, y_val = self._make_es_data()

        n_estimators, patience = 200, 5
        reg = OpenBoostLinearLeafRegressor(
            n_estimators=n_estimators,
            max_depth=3,
            early_stopping_rounds=patience,
        )
        reg.fit(X, y, eval_set=[(X_val, y_val)])

        history = reg.evals_result_['eval_0']['mse']
        rounds_trained = len(history)
        assert rounds_trained < n_estimators  # stopped early

        # Restored to best iteration: trees truncated, attrs passed through
        assert reg.best_iteration_ == int(np.argmin(history))
        assert reg.best_iteration_ == reg.booster_.best_iteration_
        assert len(reg.booster_.trees_) == reg.best_iteration_ + 1
        assert reg.best_score_ == min(history)

    def test_no_early_stopping_attributes_absent(self):
        """Without early stopping, best_iteration_/best_score_ stay absent."""
        X, y = self._make_data()
        reg = OpenBoostLinearLeafRegressor(n_estimators=5, max_depth=2)
        reg.fit(X, y)
        assert not hasattr(reg, 'best_iteration_')
        assert not hasattr(reg, 'best_score_')
        assert reg.evals_result_ == {}

    def test_evals_result_requires_fit(self):
        """evals_result_ raises NotFittedError before fit."""
        reg = OpenBoostLinearLeafRegressor(n_estimators=5)
        with pytest.raises(NotFittedError):
            _ = reg.evals_result_

    def test_early_stopping_rounds_param_roundtrip(self):
        """early_stopping_rounds survives get_params/set_params/clone."""
        reg = OpenBoostLinearLeafRegressor(n_estimators=5, early_stopping_rounds=7)
        assert reg.get_params()['early_stopping_rounds'] == 7
        reg_clone = clone(reg)
        assert reg_clone.early_stopping_rounds == 7
        reg.set_params(early_stopping_rounds=None)
        assert reg.early_stopping_rounds is None


class TestOpenBoostGAMRegressorSklearnCompat:
    """sklearn-compat tests for OpenBoostGAMRegressor."""

    def _make_data(self, n=60, seed=42):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, 3).astype(np.float32)
        y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + rng.randn(n).astype(np.float32) * 0.1
        return X, y

    def test_clone(self):
        """sklearn clone works."""
        reg = OpenBoostGAMRegressor(n_estimators=8, learning_rate=0.1)
        reg_clone = clone(reg)
        assert reg_clone.n_estimators == 8
        assert reg_clone.learning_rate == 0.1
        assert reg_clone is not reg

    def test_get_set_params(self):
        """get_params/set_params roundtrip works."""
        reg = OpenBoostGAMRegressor(n_estimators=8, reg_lambda=2.0)
        params = reg.get_params()
        assert params['n_estimators'] == 8
        assert params['reg_lambda'] == 2.0

        reg.set_params(**params)
        assert reg.get_params() == params
        reg.set_params(n_estimators=12)
        assert reg.n_estimators == 12

    def test_pipeline(self):
        """fit/predict works in a Pipeline."""
        X, y = self._make_data()
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', OpenBoostGAMRegressor(n_estimators=10, learning_rate=0.1)),
        ])
        pipe.fit(X, y)
        pred = pipe.predict(X)
        assert pred.shape == y.shape
        assert isinstance(pipe.score(X, y), float)

    def test_gridsearch(self):
        """Works with GridSearchCV (2-point grid)."""
        X, y = self._make_data()
        search = GridSearchCV(
            OpenBoostGAMRegressor(learning_rate=0.1),
            {'n_estimators': [5, 10]},
            cv=2,
        )
        search.fit(X, y)
        assert search.best_params_['n_estimators'] in (5, 10)

    def test_not_fitted_error(self):
        """predict before fit raises NotFittedError."""
        X, _ = self._make_data()
        reg = OpenBoostGAMRegressor(n_estimators=5)
        with pytest.raises(NotFittedError):
            reg.predict(X)

    def test_unknown_fit_kwarg_raises(self):
        """fit has an explicit signature: unknown kwargs raise TypeError."""
        X, y = self._make_data()
        reg = OpenBoostGAMRegressor(n_estimators=5)
        with pytest.raises(TypeError):
            reg.fit(X, y, sample_weight=np.ones(len(y), dtype=np.float32))

    def _make_es_data(self, n=240, seed=21):
        """Train/val split prone to overfitting, for early-stopping tests."""
        rng = np.random.RandomState(seed)
        X = rng.uniform(-2, 2, (n, 3)).astype(np.float32)
        y = (X[:, 0] + 0.5 * np.sin(2 * X[:, 1]) + 0.5 * rng.randn(n)).astype(
            np.float32
        )
        n_train = 2 * n // 3
        return X[:n_train], y[:n_train], X[n_train:], y[n_train:]

    def test_new_params_roundtrip(self):
        """New constructor params survive get_params/set_params/clone."""
        reg = OpenBoostGAMRegressor(
            n_estimators=10,
            interactions=2,
            interaction_rounds=5,
            smoothing=1.5,
            monotone={0: 1, 2: -1},
            early_stopping_rounds=4,
        )
        params = reg.get_params()
        assert params['interactions'] == 2
        assert params['interaction_rounds'] == 5
        assert params['smoothing'] == 1.5
        assert params['monotone'] == {0: 1, 2: -1}
        assert params['early_stopping_rounds'] == 4

        reg_clone = clone(reg)
        assert reg_clone is not reg
        assert reg_clone.interactions == 2
        assert reg_clone.interaction_rounds == 5
        assert reg_clone.smoothing == 1.5
        assert reg_clone.monotone == {0: 1, 2: -1}
        assert reg_clone.early_stopping_rounds == 4

        reg.set_params(interactions=0, smoothing=0.0, monotone=None)
        assert reg.interactions == 0
        assert reg.smoothing == 0.0
        assert reg.monotone is None

    def test_gridsearch_new_params(self):
        """GridSearchCV over a new param (2-point grid) works."""
        X, y = self._make_data()
        search = GridSearchCV(
            OpenBoostGAMRegressor(n_estimators=10, learning_rate=0.1),
            {'smoothing': [0.0, 2.0]},
            cv=2,
        )
        search.fit(X, y)
        assert search.best_params_['smoothing'] in (0.0, 2.0)

    def test_callbacks_and_eval_set_forwarded(self):
        """callbacks and eval_set reach the underlying booster."""
        X_tr, y_tr, X_va, y_va = self._make_es_data()

        history = HistoryCallback()
        reg = OpenBoostGAMRegressor(n_estimators=12, learning_rate=0.1)
        reg.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[history])

        assert len(history.history['train_loss']) == 12
        assert len(history.history['val_loss']) == 12
        assert len(reg.evals_result_['eval_0']['mse']) == 12

    def test_early_stopping(self):
        """early_stopping_rounds constructor param stops training early."""
        X_tr, y_tr, X_va, y_va = self._make_es_data()

        n_estimators, patience = 400, 15
        reg = OpenBoostGAMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.5,
            early_stopping_rounds=patience,
        )
        reg.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])

        history = reg.evals_result_['eval_0']['mse']
        assert 0 < len(history) < n_estimators  # stopped early
        assert reg.best_iteration_ == reg.booster_.best_iteration_
        assert np.isclose(reg.best_score_, min(history), rtol=1e-6)

        # Restored to best: current val MSE matches the best recorded score
        val_mse = float(np.mean((reg.predict(X_va) - y_va) ** 2))
        assert np.isclose(val_mse, reg.best_score_, rtol=1e-5)

    def test_no_early_stopping_attributes_absent(self):
        """Without early stopping, best_iteration_/best_score_ stay absent."""
        X, y = self._make_data()
        reg = OpenBoostGAMRegressor(n_estimators=5)
        reg.fit(X, y)
        assert not hasattr(reg, 'best_iteration_')
        assert not hasattr(reg, 'best_score_')
        assert reg.evals_result_ == {}

    def test_interactions_forwarded(self):
        """interactions param reaches the booster and selects pairs."""
        rng = np.random.RandomState(2)
        X = rng.uniform(-2, 2, (300, 3)).astype(np.float32)
        y = (X[:, 0] * X[:, 1]).astype(np.float32)

        reg = OpenBoostGAMRegressor(
            n_estimators=15, learning_rate=0.1, n_bins=16,
            interactions=1, interaction_rounds=5,
        )
        reg.fit(X, y)

        assert reg.booster_.interactions == 1
        assert reg.booster_.interaction_rounds == 5
        assert reg.interaction_pairs_ == [(0, 1)]

    def test_monotone_forwarded(self):
        """monotone constraint reaches the booster and shapes the fit."""
        rng = np.random.RandomState(7)
        X = rng.uniform(-2, 2, (300, 2)).astype(np.float32)
        y = (X[:, 0] + 0.3 * rng.randn(300)).astype(np.float32)

        reg = OpenBoostGAMRegressor(
            n_estimators=50, learning_rate=0.1, monotone={0: 1},
        )
        reg.fit(X, y)

        assert reg.booster_.monotone == {0: 1}
        n_used = len(reg.booster_.X_binned_.bin_edges[0])
        diffs = np.diff(reg.shape_values_[0, :n_used])
        assert np.all(diffs >= -1e-7)

    def test_smoothing_forwarded(self):
        """smoothing param reaches the booster and changes the fit."""
        X, y = self._make_data(n=120)
        plain = OpenBoostGAMRegressor(n_estimators=20, learning_rate=0.1)
        smoothed = OpenBoostGAMRegressor(
            n_estimators=20, learning_rate=0.1, smoothing=5.0,
        )
        plain.fit(X, y)
        smoothed.fit(X, y)

        assert smoothed.booster_.smoothing == 5.0
        assert not np.allclose(plain.shape_values_, smoothed.shape_values_)


class TestOpenBoostGAMClassifier:
    """sklearn-compat battery for OpenBoostGAMClassifier."""

    def _make_data(self, n=200, seed=42):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, 5).astype(np.float32)
        y = (X[:, 0] + 0.5 * rng.randn(n) > 0).astype(np.int32)
        return X, y

    def test_binary_classification(self):
        """Binary classification works."""
        X, y = self._make_data()
        clf = OpenBoostGAMClassifier(n_estimators=30, learning_rate=0.1)
        clf.fit(X, y)
        pred = clf.predict(X)

        assert pred.shape == y.shape
        assert set(pred).issubset({0, 1})

    def test_predict_proba(self):
        """Probability predictions are a valid (n, 2) simplex."""
        X, y = self._make_data()
        clf = OpenBoostGAMClassifier(n_estimators=30, learning_rate=0.1)
        clf.fit(X, y)
        proba = clf.predict_proba(X)

        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all() and (proba <= 1).all()

        # predict is argmax over predict_proba, mapped back to classes_
        pred = clf.predict(X)
        assert np.array_equal(pred, clf.classes_[np.argmax(proba, axis=1)])

    def test_classes_attribute(self):
        """classes_ set correctly and predictions use original labels."""
        X, y_num = self._make_data()
        y = np.where(y_num == 1, 'dog', 'cat')

        clf = OpenBoostGAMClassifier(n_estimators=20, learning_rate=0.1)
        clf.fit(X, y)

        assert set(clf.classes_) == {'cat', 'dog'}
        assert clf.n_classes_ == 2
        pred = clf.predict(X)
        assert set(pred).issubset({'cat', 'dog'})

    def test_score(self):
        """Accuracy score works and beats chance on separable data."""
        X, y = self._make_data()
        clf = OpenBoostGAMClassifier(n_estimators=50, learning_rate=0.1)
        clf.fit(X, y)
        assert clf.score(X, y) > 0.7

    def test_multiclass_raises(self):
        """More than 2 classes raises a clear ValueError."""
        rng = np.random.RandomState(42)
        X = rng.randn(90, 3).astype(np.float32)
        y = np.array([0] * 30 + [1] * 30 + [2] * 30)

        clf = OpenBoostGAMClassifier(n_estimators=5)
        with pytest.raises(ValueError, match="binary"):
            clf.fit(X, y)

    def test_clone(self):
        """sklearn clone works, including new GAM params."""
        clf = OpenBoostGAMClassifier(
            n_estimators=25, learning_rate=0.2, interactions=1,
            smoothing=0.5, monotone={1: -1}, early_stopping_rounds=3,
        )
        clf_clone = clone(clf)
        assert clf_clone is not clf
        assert clf_clone.n_estimators == 25
        assert clf_clone.learning_rate == 0.2
        assert clf_clone.interactions == 1
        assert clf_clone.smoothing == 0.5
        assert clf_clone.monotone == {1: -1}
        assert clf_clone.early_stopping_rounds == 3

    def test_get_set_params(self):
        """get_params/set_params roundtrip works."""
        clf = OpenBoostGAMClassifier(n_estimators=25, reg_lambda=2.0)
        params = clf.get_params()
        assert params['n_estimators'] == 25
        assert params['reg_lambda'] == 2.0
        assert params['interactions'] == 0

        clf.set_params(**params)
        assert clf.get_params() == params
        clf.set_params(n_estimators=40)
        assert clf.n_estimators == 40

    def test_gridsearch(self):
        """Works with GridSearchCV (2-point grid)."""
        X, y = self._make_data()
        search = GridSearchCV(
            OpenBoostGAMClassifier(learning_rate=0.1),
            {'n_estimators': [10, 20]},
            cv=2,
        )
        search.fit(X, y)
        assert search.best_params_['n_estimators'] in (10, 20)

    def test_pipeline(self):
        """fit/predict works in a Pipeline."""
        X, y = self._make_data()
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', OpenBoostGAMClassifier(n_estimators=15, learning_rate=0.1)),
        ])
        pipe.fit(X, y)
        pred = pipe.predict(X)
        assert pred.shape == y.shape
        assert isinstance(pipe.score(X, y), float)

    def test_not_fitted_error(self):
        """predict/predict_proba before fit raise NotFittedError."""
        X, _ = self._make_data()
        clf = OpenBoostGAMClassifier(n_estimators=5)
        with pytest.raises(NotFittedError):
            clf.predict(X)
        with pytest.raises(NotFittedError):
            clf.predict_proba(X)

    def test_early_stopping(self):
        """early_stopping_rounds + eval_set stops early, records logloss."""
        rng = np.random.RandomState(21)
        X = rng.uniform(-2, 2, (240, 3)).astype(np.float32)
        y = (X[:, 0] + 0.75 * rng.randn(240) > 0).astype(np.int32)
        X_tr, y_tr = X[:160], y[:160]
        X_va, y_va = X[160:], y[160:]

        n_estimators, patience = 400, 15
        clf = OpenBoostGAMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.5,
            early_stopping_rounds=patience,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])

        history = clf.evals_result_['eval_0']['logloss']
        assert 0 < len(history) < n_estimators  # stopped early
        assert all(np.isfinite(v) for v in history)
        assert clf.best_iteration_ == clf.booster_.best_iteration_
        assert np.isclose(clf.best_score_, min(history), rtol=1e-6)

    def test_string_labels_eval_set_encoded(self):
        """eval_set labels are encoded with the training label encoder."""
        X, y_num = self._make_data()
        y = np.where(y_num == 1, 'dog', 'cat')
        X_tr, y_tr = X[:150], y[:150]
        X_va, y_va = X[150:], y[150:]

        clf = OpenBoostGAMClassifier(n_estimators=10, learning_rate=0.1)
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])

        history = clf.evals_result_['eval_0']['logloss']
        assert len(history) == 10
        assert all(np.isfinite(v) for v in history)

    def test_unknown_fit_kwarg_raises(self):
        """fit has an explicit signature: unknown kwargs raise TypeError."""
        X, y = self._make_data()
        clf = OpenBoostGAMClassifier(n_estimators=5)
        with pytest.raises(TypeError):
            clf.fit(X, y, sample_weight=np.ones(len(y), dtype=np.float32))


class TestSampleWeight:
    """Tests for sample weight support."""
    
    def test_sample_weight_regressor(self):
        """Sample weights work for regressor."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0] + np.random.randn(100).astype(np.float32) * 0.1
        
        # Higher weights on first half
        weights = np.array([10.0] * 50 + [1.0] * 50, dtype=np.float32)
        
        reg = OpenBoostRegressor(n_estimators=10, max_depth=3)
        reg.fit(X, y, sample_weight=weights)
        
        pred = reg.predict(X)
        assert pred.shape == y.shape
    
    def test_sample_weight_gradient_boosting(self):
        """Sample weights work for GradientBoosting."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0] + np.random.randn(100).astype(np.float32) * 0.1
        
        weights = np.ones(100, dtype=np.float32)
        
        model = GradientBoosting(n_trees=10, max_depth=3)
        model.fit(X, y, sample_weight=weights)
        
        assert len(model.trees_) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
