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
    OpenBoostRegressor,
    OpenBoostClassifier,
    GradientBoosting,
    EarlyStopping,
    Logger,
    HistoryCallback,
    compute_feature_importances,
    get_feature_importance_dict,
)


# Skip tests if sklearn not installed
sklearn = pytest.importorskip("sklearn")
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone


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
        assert all(isinstance(k, str) for k in importance_dict.keys())
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
