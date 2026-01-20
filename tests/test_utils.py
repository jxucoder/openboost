"""Tests for OpenBoost utility functions.

Phase 20.6: Tests for suggest_params, cross_val_predict, etc.
Phase 22: Tests for evaluation metrics with sample weight support.
"""

import numpy as np
import pytest

from openboost import (
    OpenBoostRegressor,
    OpenBoostClassifier,
    OpenBoostDistributionalRegressor,
    suggest_params,
    cross_val_predict,
    cross_val_predict_proba,
    cross_val_predict_interval,
    evaluate_coverage,
    get_param_grid,
    PARAM_GRID_REGRESSION,
    PARAM_GRID_CLASSIFICATION,
    PARAM_GRID_DISTRIBUTIONAL,
    # Phase 22: Evaluation metrics
    roc_auc_score,
    accuracy_score,
    log_loss_score,
    mse_score,
    r2_score,
    mae_score,
    rmse_score,
    f1_score,
    precision_score,
    recall_score,
    # Phase 22 Sprint 2: Probabilistic metrics
    crps_gaussian,
    crps_empirical,
    brier_score,
    pinball_loss,
    interval_score,
    expected_calibration_error,
    calibration_curve,
    negative_log_likelihood,
)


class TestSuggestParams:
    """Tests for suggest_params function."""
    
    def test_suggest_params_small_dataset(self):
        """Small datasets get fewer trees and more regularization."""
        np.random.seed(42)
        X = np.random.randn(500, 10).astype(np.float32)
        y = np.random.randn(500).astype(np.float32)
        
        params = suggest_params(X, y, task='regression')
        
        assert params['n_estimators'] <= 100
        assert params['reg_lambda'] >= 1.0
        assert 'max_depth' in params
        assert 'learning_rate' in params
    
    def test_suggest_params_large_dataset(self):
        """Large datasets get more trees and lower learning rate."""
        np.random.seed(42)
        X = np.random.randn(50000, 20).astype(np.float32)
        y = np.random.randn(50000).astype(np.float32)
        
        params = suggest_params(X, y, task='regression')
        
        assert params['n_estimators'] >= 200
        assert params['learning_rate'] <= 0.1
        assert 'subsample' in params  # Sampling for large datasets
    
    def test_suggest_params_high_dimensional(self):
        """High-dimensional data gets column sampling and shallower trees."""
        np.random.seed(42)
        X = np.random.randn(5000, 200).astype(np.float32)
        y = np.random.randn(5000).astype(np.float32)
        
        params = suggest_params(X, y, task='regression')
        
        assert params['colsample_bytree'] < 1.0
        assert params['max_depth'] <= 6
    
    def test_suggest_params_distributional(self):
        """Distributional task gets shallower trees."""
        np.random.seed(42)
        X = np.random.randn(5000, 20).astype(np.float32)
        y = np.random.randn(5000).astype(np.float32)
        
        params = suggest_params(X, y, task='distributional')
        
        assert params['max_depth'] <= 5
    
    def test_suggest_params_with_model(self):
        """Suggested params work with actual model."""
        np.random.seed(42)
        X = np.random.randn(200, 10).astype(np.float32)
        y = np.random.randn(200).astype(np.float32)
        
        params = suggest_params(X, y, task='regression')
        model = OpenBoostRegressor(**params)
        model.fit(X, y)
        
        pred = model.predict(X)
        assert pred.shape == y.shape


class TestCrossValPredict:
    """Tests for cross_val_predict function."""
    
    def test_cross_val_predict_regression(self):
        """OOF predictions for regression."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0] * 2 + np.random.randn(100).astype(np.float32) * 0.1
        
        model = OpenBoostRegressor(n_estimators=20, max_depth=3)
        oof_pred = cross_val_predict(model, X, y, cv=3)
        
        assert oof_pred.shape == y.shape
        # OOF predictions should be somewhat correlated with targets
        corr = np.corrcoef(oof_pred, y)[0, 1]
        assert corr > 0.5  # Reasonable correlation
    
    def test_cross_val_predict_all_samples_predicted(self):
        """Every sample gets a prediction."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        
        model = OpenBoostRegressor(n_estimators=10)
        oof_pred = cross_val_predict(model, X, y, cv=5)
        
        # All samples should have predictions (no zeros from uninitialized)
        # Check variance - all predictions shouldn't be the same
        assert np.std(oof_pred) > 0.01


class TestCrossValPredictProba:
    """Tests for cross_val_predict_proba function."""
    
    def test_cross_val_predict_proba_binary(self):
        """OOF probabilities for binary classification."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int32)
        
        model = OpenBoostClassifier(n_estimators=20, max_depth=3)
        oof_proba = cross_val_predict_proba(model, X, y, cv=3)
        
        assert oof_proba.shape == (100, 2)
        # Probabilities should sum to 1
        assert np.allclose(oof_proba.sum(axis=1), 1.0, atol=1e-5)
        # Probabilities should be in [0, 1]
        assert np.all(oof_proba >= 0) and np.all(oof_proba <= 1)
    
    def test_cross_val_predict_proba_multiclass(self):
        """OOF probabilities for multi-class classification."""
        np.random.seed(42)
        X = np.random.randn(150, 5).astype(np.float32)
        y = np.repeat([0, 1, 2], 50).astype(np.int32)
        np.random.shuffle(y)
        
        model = OpenBoostClassifier(n_estimators=20, max_depth=3)
        oof_proba = cross_val_predict_proba(model, X, y, cv=3)
        
        assert oof_proba.shape == (150, 3)
        assert np.allclose(oof_proba.sum(axis=1), 1.0, atol=1e-5)
    
    def test_cross_val_predict_proba_error_on_regressor(self):
        """Should raise error when used with regressor."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        
        model = OpenBoostRegressor(n_estimators=10)
        
        with pytest.raises(AttributeError, match="predict_proba"):
            cross_val_predict_proba(model, X, y, cv=3)


class TestCrossValPredictInterval:
    """Tests for cross_val_predict_interval function."""
    
    def test_cross_val_predict_interval_basic(self):
        """OOF prediction intervals."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0] * 2 + np.random.randn(100).astype(np.float32) * 0.5
        
        model = OpenBoostDistributionalRegressor(
            distribution='normal',
            n_estimators=20,
            max_depth=3
        )
        lower, upper = cross_val_predict_interval(model, X, y, alpha=0.1, cv=3)
        
        assert lower.shape == y.shape
        assert upper.shape == y.shape
        assert np.all(lower <= upper)
    
    def test_cross_val_predict_interval_coverage(self):
        """OOF intervals should have reasonable coverage."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] * 2 + np.random.randn(200).astype(np.float32) * 0.5
        
        model = OpenBoostDistributionalRegressor(
            distribution='normal',
            n_estimators=50,
            max_depth=3
        )
        lower, upper = cross_val_predict_interval(model, X, y, alpha=0.1, cv=5)
        
        coverage = np.mean((y >= lower) & (y <= upper))
        # Coverage should be somewhat close to 90%, allow wide margin for OOF
        assert coverage > 0.5  # At least half should be covered


class TestEvaluateCoverage:
    """Tests for evaluate_coverage function."""
    
    def test_evaluate_coverage_perfect(self):
        """100% coverage when all points in interval."""
        y = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.0, 1.0, 2.0])
        upper = np.array([2.0, 3.0, 4.0])
        
        metrics = evaluate_coverage(y, lower, upper, alpha=0.1)
        
        assert metrics['coverage'] == 1.0
        assert metrics['expected_coverage'] == 0.9
        assert metrics['mean_width'] == 2.0
    
    def test_evaluate_coverage_partial(self):
        """Partial coverage."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        lower = np.array([0.5, 1.5, 3.5, 4.5])  # 2 out of 4 covered
        upper = np.array([1.5, 2.5, 4.5, 5.5])
        
        metrics = evaluate_coverage(y, lower, upper, alpha=0.1)
        
        assert metrics['coverage'] == 0.5


class TestGetParamGrid:
    """Tests for get_param_grid function."""
    
    def test_get_param_grid_regression(self):
        """Regression param grid."""
        grid = get_param_grid('regression')
        
        assert 'n_estimators' in grid
        assert 'max_depth' in grid
        assert 'learning_rate' in grid
        assert grid == PARAM_GRID_REGRESSION
    
    def test_get_param_grid_classification(self):
        """Classification param grid."""
        grid = get_param_grid('classification')
        
        assert grid == PARAM_GRID_CLASSIFICATION
    
    def test_get_param_grid_distributional(self):
        """Distributional param grid."""
        grid = get_param_grid('distributional')
        
        assert grid == PARAM_GRID_DISTRIBUTIONAL
        # Distributional should have shallower trees
        assert max(grid['max_depth']) <= 5
    
    def test_get_param_grid_invalid(self):
        """Invalid task raises error."""
        with pytest.raises(ValueError, match="Unknown task"):
            get_param_grid('invalid')
    
    def test_param_grid_with_gridsearch(self):
        """Param grid works with sklearn GridSearchCV."""
        pytest.importorskip("sklearn")
        from sklearn.model_selection import GridSearchCV
        
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        
        # Use a small subset of the grid for speed
        small_grid = {
            'n_estimators': [20, 50],
            'max_depth': [3, 4],
        }
        
        search = GridSearchCV(
            OpenBoostRegressor(),
            small_grid,
            cv=2,
            scoring='neg_mean_squared_error'
        )
        search.fit(X, y)
        
        assert search.best_params_ is not None
        assert 'n_estimators' in search.best_params_


class TestParamGridConstants:
    """Tests for parameter grid constants."""
    
    def test_regression_grid_structure(self):
        """Regression grid has expected structure."""
        assert isinstance(PARAM_GRID_REGRESSION, dict)
        for key, values in PARAM_GRID_REGRESSION.items():
            assert isinstance(values, list)
            assert len(values) > 0
    
    def test_classification_grid_structure(self):
        """Classification grid has expected structure."""
        assert isinstance(PARAM_GRID_CLASSIFICATION, dict)
        for key, values in PARAM_GRID_CLASSIFICATION.items():
            assert isinstance(values, list)
    
    def test_distributional_grid_structure(self):
        """Distributional grid has expected structure."""
        assert isinstance(PARAM_GRID_DISTRIBUTIONAL, dict)
        for key, values in PARAM_GRID_DISTRIBUTIONAL.items():
            assert isinstance(values, list)


# =============================================================================
# Phase 22: Evaluation Metrics Tests
# =============================================================================

class TestRocAucScore:
    """Tests for roc_auc_score function."""
    
    def test_roc_auc_perfect_separation(self):
        """Perfect classifier gets AUC = 1.0."""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        
        assert roc_auc_score(y_true, y_score) == 1.0
    
    def test_roc_auc_random(self):
        """Random classifier gets AUC ~ 0.5."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_score = np.random.random(1000)
        
        auc = roc_auc_score(y_true, y_score)
        assert 0.4 < auc < 0.6  # Should be close to 0.5
    
    def test_roc_auc_with_sample_weight(self):
        """AUC with sample weights."""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])
        
        # Without weights
        auc_no_weight = roc_auc_score(y_true, y_score)
        
        # With weights emphasizing correct predictions
        weights = np.array([1.0, 0.1, 0.1, 1.0])
        auc_weighted = roc_auc_score(y_true, y_score, sample_weight=weights)
        
        # Weighted should be higher (emphasizes good predictions)
        assert auc_weighted >= auc_no_weight


class TestAccuracyScore:
    """Tests for accuracy_score function."""
    
    def test_accuracy_perfect(self):
        """Perfect predictions get accuracy = 1.0."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        
        assert accuracy_score(y_true, y_pred) == 1.0
    
    def test_accuracy_half(self):
        """50% accuracy."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        
        assert accuracy_score(y_true, y_pred) == 0.5
    
    def test_accuracy_with_sample_weight(self):
        """Accuracy with sample weights."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1])  # 2 correct, 2 wrong
        
        # Uniform weights: 50% accuracy
        assert accuracy_score(y_true, y_pred) == 0.5
        
        # Weight correct predictions more heavily
        weights = np.array([2.0, 2.0, 1.0, 1.0])  # Correct: weight 4, Wrong: weight 2
        weighted_acc = accuracy_score(y_true, y_pred, sample_weight=weights)
        assert weighted_acc == pytest.approx(4.0 / 6.0)


class TestLogLossScore:
    """Tests for log_loss_score function."""
    
    def test_log_loss_perfect(self):
        """Near-perfect predictions have low log loss."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.01, 0.01, 0.99, 0.99])
        
        loss = log_loss_score(y_true, y_pred)
        assert loss < 0.1
    
    def test_log_loss_random(self):
        """Random predictions (0.5) have log loss ~0.693."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        
        loss = log_loss_score(y_true, y_pred)
        assert 0.68 < loss < 0.7  # ln(2) ≈ 0.693
    
    def test_log_loss_with_sample_weight(self):
        """Log loss with sample weights."""
        y_true = np.array([0, 1])
        y_pred = np.array([0.1, 0.9])
        
        loss_no_weight = log_loss_score(y_true, y_pred)
        
        # Weight first sample more (which is well predicted)
        weights = np.array([10.0, 1.0])
        loss_weighted = log_loss_score(y_true, y_pred, sample_weight=weights)
        
        # Should still be reasonable
        assert loss_weighted > 0


class TestMseScore:
    """Tests for mse_score function."""
    
    def test_mse_perfect(self):
        """Perfect predictions have MSE = 0."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        
        assert mse_score(y_true, y_pred) == 0.0
    
    def test_mse_known_value(self):
        """MSE with known errors."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])  # Errors: 1, 0, 1
        
        # MSE = (1 + 0 + 1) / 3 = 2/3
        assert mse_score(y_true, y_pred) == pytest.approx(2.0 / 3.0)
    
    def test_mse_with_sample_weight(self):
        """MSE with sample weights."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])  # Errors: 1, 0, 1
        
        # Weight the zero-error sample heavily
        weights = np.array([1.0, 10.0, 1.0])
        mse_weighted = mse_score(y_true, y_pred, sample_weight=weights)
        
        # Should be lower than unweighted
        mse_unweighted = mse_score(y_true, y_pred)
        assert mse_weighted < mse_unweighted


class TestR2Score:
    """Tests for r2_score function."""
    
    def test_r2_perfect(self):
        """Perfect predictions have R² = 1.0."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        
        assert r2_score(y_true, y_pred) == 1.0
    
    def test_r2_mean_baseline(self):
        """Predicting mean has R² = 0."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([2.5, 2.5, 2.5, 2.5])  # Mean
        
        assert r2_score(y_true, y_pred) == pytest.approx(0.0)
    
    def test_r2_worse_than_mean(self):
        """Worse than mean predictions have R² < 0."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([4.0, 3.0, 2.0, 1.0])  # Inverted
        
        assert r2_score(y_true, y_pred) < 0
    
    def test_r2_with_sample_weight(self):
        """R² with sample weights."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9])  # Good predictions
        
        r2_unweighted = r2_score(y_true, y_pred)
        
        # Weight samples with smaller errors more
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        r2_weighted = r2_score(y_true, y_pred, sample_weight=weights)
        
        # Both should be close to 1
        assert r2_unweighted > 0.9
        assert r2_weighted > 0.9


class TestMaeScore:
    """Tests for mae_score function."""
    
    def test_mae_perfect(self):
        """Perfect predictions have MAE = 0."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        
        assert mae_score(y_true, y_pred) == 0.0
    
    def test_mae_known_value(self):
        """MAE with known errors."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])  # Errors: 1, 0, 1
        
        # MAE = (1 + 0 + 1) / 3 = 2/3
        assert mae_score(y_true, y_pred) == pytest.approx(2.0 / 3.0)
    
    def test_mae_with_sample_weight(self):
        """MAE with sample weights."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        
        # Weight the zero-error sample heavily
        weights = np.array([1.0, 10.0, 1.0])
        mae_weighted = mae_score(y_true, y_pred, sample_weight=weights)
        
        # Should be lower than unweighted
        mae_unweighted = mae_score(y_true, y_pred)
        assert mae_weighted < mae_unweighted


class TestRmseScore:
    """Tests for rmse_score function."""
    
    def test_rmse_perfect(self):
        """Perfect predictions have RMSE = 0."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        
        assert rmse_score(y_true, y_pred) == 0.0
    
    def test_rmse_is_sqrt_mse(self):
        """RMSE = sqrt(MSE)."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        
        mse = mse_score(y_true, y_pred)
        rmse = rmse_score(y_true, y_pred)
        
        assert rmse == pytest.approx(np.sqrt(mse))
    
    def test_rmse_with_sample_weight(self):
        """RMSE with sample weights."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        
        weights = np.array([1.0, 10.0, 1.0])
        rmse_weighted = rmse_score(y_true, y_pred, sample_weight=weights)
        
        rmse_unweighted = rmse_score(y_true, y_pred)
        assert rmse_weighted < rmse_unweighted


class TestF1Score:
    """Tests for f1_score function."""
    
    def test_f1_perfect(self):
        """Perfect predictions have F1 = 1.0."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        
        assert f1_score(y_true, y_pred) == 1.0
    
    def test_f1_known_value(self):
        """F1 with known precision and recall."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])  # TP=2, FP=0, FN=1
        
        # Precision = 2/2 = 1.0, Recall = 2/3
        # F1 = 2 * (1.0 * 2/3) / (1.0 + 2/3) = 0.8
        assert f1_score(y_true, y_pred) == pytest.approx(0.8)
    
    def test_f1_multiclass_macro(self):
        """F1 with macro averaging for multi-class."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1])  # 4/6 correct
        
        f1_macro = f1_score(y_true, y_pred, average='macro')
        assert 0 < f1_macro < 1
    
    def test_f1_with_sample_weight(self):
        """F1 with sample weights."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        f1_unweighted = f1_score(y_true, y_pred)
        
        # Weight correct predictions more
        weights = np.array([1.0, 2.0, 1.0, 1.0, 2.0])
        f1_weighted = f1_score(y_true, y_pred, sample_weight=weights)
        
        # Both should be reasonable
        assert f1_unweighted > 0
        assert f1_weighted > 0


class TestPrecisionScore:
    """Tests for precision_score function."""
    
    def test_precision_perfect(self):
        """Perfect predictions have precision = 1.0."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        
        assert precision_score(y_true, y_pred) == 1.0
    
    def test_precision_no_false_positives(self):
        """No false positives means precision = 1.0."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])  # TP=2, FP=0
        
        assert precision_score(y_true, y_pred) == 1.0
    
    def test_precision_with_false_positives(self):
        """Precision with false positives."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 1])  # TP=2, FP=1
        
        # Precision = 2/3
        assert precision_score(y_true, y_pred) == pytest.approx(2.0 / 3.0)
    
    def test_precision_multiclass(self):
        """Precision with multi-class."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1])
        
        prec_macro = precision_score(y_true, y_pred, average='macro')
        assert 0 < prec_macro <= 1


class TestRecallScore:
    """Tests for recall_score function."""
    
    def test_recall_perfect(self):
        """Perfect predictions have recall = 1.0."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        
        assert recall_score(y_true, y_pred) == 1.0
    
    def test_recall_no_false_negatives(self):
        """No false negatives means recall = 1.0."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([1, 1, 1, 1, 1])  # All predicted positive
        
        assert recall_score(y_true, y_pred) == 1.0
    
    def test_recall_with_false_negatives(self):
        """Recall with false negatives."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])  # TP=2, FN=1
        
        # Recall = 2/3
        assert recall_score(y_true, y_pred) == pytest.approx(2.0 / 3.0)
    
    def test_recall_multiclass(self):
        """Recall with multi-class."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1])
        
        rec_macro = recall_score(y_true, y_pred, average='macro')
        assert 0 < rec_macro <= 1


class TestMetricsWithOpenBoost:
    """Integration tests: metrics with actual OpenBoost models."""
    
    def test_regression_metrics_with_model(self):
        """Regression metrics on OpenBoost predictions."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] * 2 + np.random.randn(200).astype(np.float32) * 0.5
        
        model = OpenBoostRegressor(n_estimators=20, max_depth=3)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # All regression metrics should work
        mse = mse_score(y, y_pred)
        mae = mae_score(y, y_pred)
        rmse = rmse_score(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        assert mse >= 0
        assert mae >= 0
        assert rmse >= 0
        assert rmse == pytest.approx(np.sqrt(mse))
        assert r2 > 0  # Should fit reasonably well
    
    def test_classification_metrics_with_model(self):
        """Classification metrics on OpenBoost predictions."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int32)
        
        model = OpenBoostClassifier(n_estimators=20, max_depth=3)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        # All classification metrics should work
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_proba)
        logloss = log_loss_score(y, y_proba)
        f1 = f1_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        
        assert 0 <= acc <= 1
        assert 0 <= auc <= 1
        assert logloss >= 0
        assert 0 <= f1 <= 1
        assert 0 <= prec <= 1
        assert 0 <= rec <= 1
        
        # Model should do better than random
        assert acc > 0.6
        assert auc > 0.6
    
    def test_weighted_metrics_with_model(self):
        """Weighted metrics on OpenBoost predictions."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] * 2 + np.random.randn(200).astype(np.float32) * 0.5
        
        model = OpenBoostRegressor(n_estimators=20, max_depth=3)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Create sample weights
        weights = np.abs(y)  # Weight by target magnitude
        
        # All metrics should accept weights
        mse_w = mse_score(y, y_pred, sample_weight=weights)
        mae_w = mae_score(y, y_pred, sample_weight=weights)
        rmse_w = rmse_score(y, y_pred, sample_weight=weights)
        r2_w = r2_score(y, y_pred, sample_weight=weights)
        
        assert mse_w >= 0
        assert mae_w >= 0
        assert rmse_w >= 0
        # R2 can be anything when weighted


# =============================================================================
# Phase 22 Sprint 2: Probabilistic/Distributional Metrics Tests
# =============================================================================

class TestCRPSGaussian:
    """Tests for CRPS (Continuous Ranked Probability Score) for Gaussian."""
    
    def test_crps_perfect_predictions(self):
        """Perfect mean predictions with small std have low CRPS."""
        y_true = np.array([1.0, 2.0, 3.0])
        mean = np.array([1.0, 2.0, 3.0])  # Perfect means
        std = np.array([0.01, 0.01, 0.01])  # Very small std
        
        crps = crps_gaussian(y_true, mean, std)
        assert crps < 0.01  # Should be very small
    
    def test_crps_larger_std_higher_crps(self):
        """Larger std leads to higher CRPS for same mean predictions."""
        y_true = np.array([1.0, 2.0, 3.0])
        mean = np.array([1.0, 2.0, 3.0])
        
        crps_small_std = crps_gaussian(y_true, mean, np.array([0.1, 0.1, 0.1]))
        crps_large_std = crps_gaussian(y_true, mean, np.array([1.0, 1.0, 1.0]))
        
        assert crps_small_std < crps_large_std
    
    def test_crps_wrong_mean_higher_crps(self):
        """Wrong mean predictions have higher CRPS."""
        y_true = np.array([1.0, 2.0, 3.0])
        std = np.array([0.5, 0.5, 0.5])
        
        crps_good = crps_gaussian(y_true, np.array([1.0, 2.0, 3.0]), std)
        crps_bad = crps_gaussian(y_true, np.array([2.0, 3.0, 4.0]), std)
        
        assert crps_good < crps_bad
    
    def test_crps_positive(self):
        """CRPS should always be non-negative."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        mean = y_true + np.random.randn(100) * 0.5
        std = np.abs(np.random.randn(100)) + 0.1
        
        crps = crps_gaussian(y_true, mean, std)
        assert crps >= 0
    
    def test_crps_with_sample_weight(self):
        """CRPS with sample weights."""
        y_true = np.array([1.0, 2.0, 3.0])
        mean = np.array([1.0, 2.5, 3.0])  # Middle prediction is off
        std = np.array([0.5, 0.5, 0.5])
        
        crps_unweighted = crps_gaussian(y_true, mean, std)
        
        # Weight down the bad prediction
        weights = np.array([1.0, 0.1, 1.0])
        crps_weighted = crps_gaussian(y_true, mean, std, sample_weight=weights)
        
        assert crps_weighted < crps_unweighted
    
    def test_crps_invalid_std_raises(self):
        """Negative or zero std should raise error."""
        y_true = np.array([1.0])
        mean = np.array([1.0])
        
        with pytest.raises(ValueError, match="std must be positive"):
            crps_gaussian(y_true, mean, np.array([0.0]))
        
        with pytest.raises(ValueError, match="std must be positive"):
            crps_gaussian(y_true, mean, np.array([-1.0]))


class TestCRPSEmpirical:
    """Tests for empirical CRPS from Monte Carlo samples."""
    
    def test_crps_empirical_basic(self):
        """Basic empirical CRPS computation."""
        np.random.seed(42)
        y_true = np.array([1.0, 2.0, 3.0])
        
        # Generate samples centered around true values
        samples = np.stack([
            np.random.randn(1000) * 0.1 + 1.0,
            np.random.randn(1000) * 0.1 + 2.0,
            np.random.randn(1000) * 0.1 + 3.0,
        ])
        
        crps = crps_empirical(y_true, samples)
        assert crps >= 0
        assert crps < 0.5  # Should be small for good predictions
    
    def test_crps_empirical_poor_predictions(self):
        """Poor predictions have higher empirical CRPS."""
        np.random.seed(42)
        y_true = np.array([1.0, 2.0, 3.0])
        
        # Good samples - centered on true values
        good_samples = np.stack([
            np.random.randn(1000) * 0.1 + 1.0,
            np.random.randn(1000) * 0.1 + 2.0,
            np.random.randn(1000) * 0.1 + 3.0,
        ])
        
        # Bad samples - centered away from true values
        bad_samples = np.stack([
            np.random.randn(1000) * 0.1 + 5.0,
            np.random.randn(1000) * 0.1 + 6.0,
            np.random.randn(1000) * 0.1 + 7.0,
        ])
        
        crps_good = crps_empirical(y_true, good_samples)
        crps_bad = crps_empirical(y_true, bad_samples)
        
        assert crps_good < crps_bad


class TestBrierScore:
    """Tests for Brier score."""
    
    def test_brier_perfect_predictions(self):
        """Perfect probability predictions have Brier = 0."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        
        assert brier_score(y_true, y_prob) == pytest.approx(0.0)
    
    def test_brier_random_predictions(self):
        """Random predictions (p=0.5) have Brier = 0.25."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        
        assert brier_score(y_true, y_prob) == pytest.approx(0.25)
    
    def test_brier_worst_predictions(self):
        """Completely wrong predictions have Brier = 1."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0])
        
        assert brier_score(y_true, y_prob) == pytest.approx(1.0)
    
    def test_brier_intermediate(self):
        """Intermediate predictions have Brier between 0 and 0.25."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        
        brier = brier_score(y_true, y_prob)
        assert 0 < brier < 0.25
    
    def test_brier_with_sample_weight(self):
        """Brier score with sample weights."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.9])  # Second pred is wrong
        
        brier_unweighted = brier_score(y_true, y_prob)
        
        # Weight down the bad prediction
        weights = np.array([1.0, 0.1, 1.0, 1.0])
        brier_weighted = brier_score(y_true, y_prob, sample_weight=weights)
        
        assert brier_weighted < brier_unweighted


class TestPinballLoss:
    """Tests for pinball loss (quantile loss)."""
    
    def test_pinball_median_equals_mae_scaled(self):
        """At quantile=0.5, pinball loss = MAE / 2."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.0, 2.5])
        
        pinball = pinball_loss(y_true, y_pred, quantile=0.5)
        mae = mae_score(y_true, y_pred)
        
        assert pinball == pytest.approx(mae / 2)
    
    def test_pinball_perfect_predictions(self):
        """Perfect predictions have pinball loss = 0."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        
        assert pinball_loss(y_true, y_pred, quantile=0.5) == pytest.approx(0.0)
    
    def test_pinball_asymmetric(self):
        """Pinball loss is asymmetric around quantile."""
        y_true = np.array([2.0])
        
        # Underpredict
        y_pred_under = np.array([1.0])
        loss_under_q90 = pinball_loss(y_true, y_pred_under, quantile=0.9)
        
        # Overpredict by same amount
        y_pred_over = np.array([3.0])
        loss_over_q90 = pinball_loss(y_true, y_pred_over, quantile=0.9)
        
        # For q=0.9, underprediction is penalized more
        assert loss_under_q90 > loss_over_q90
    
    def test_pinball_lower_quantile(self):
        """Lower quantile penalizes overprediction more."""
        y_true = np.array([2.0])
        
        # Overpredict
        y_pred_over = np.array([3.0])
        loss_q10 = pinball_loss(y_true, y_pred_over, quantile=0.1)
        loss_q50 = pinball_loss(y_true, y_pred_over, quantile=0.5)
        
        # q=0.1 penalizes overprediction more than q=0.5
        assert loss_q10 > loss_q50
    
    def test_pinball_invalid_quantile(self):
        """Invalid quantile raises error."""
        y_true = np.array([1.0])
        y_pred = np.array([1.0])
        
        with pytest.raises(ValueError, match="quantile must be in"):
            pinball_loss(y_true, y_pred, quantile=0.0)
        
        with pytest.raises(ValueError, match="quantile must be in"):
            pinball_loss(y_true, y_pred, quantile=1.0)
        
        with pytest.raises(ValueError, match="quantile must be in"):
            pinball_loss(y_true, y_pred, quantile=1.5)
    
    def test_pinball_with_sample_weight(self):
        """Pinball loss with sample weights."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.0, 2.5])  # Errors on first and third
        
        loss_unweighted = pinball_loss(y_true, y_pred, quantile=0.5)
        
        # Weight the accurate prediction more
        weights = np.array([0.1, 10.0, 0.1])
        loss_weighted = pinball_loss(y_true, y_pred, quantile=0.5, sample_weight=weights)
        
        assert loss_weighted < loss_unweighted


class TestIntervalScore:
    """Tests for interval score."""
    
    def test_interval_score_perfect_coverage(self):
        """All points inside interval contribute only width."""
        y_true = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.0, 1.0, 2.0])
        upper = np.array([2.0, 3.0, 4.0])
        
        score = interval_score(y_true, lower, upper, alpha=0.1)
        
        # Only width contributes (all points inside)
        expected_width = 2.0
        assert score == pytest.approx(expected_width)
    
    def test_interval_score_missed_below(self):
        """Point below lower bound adds penalty."""
        y_true = np.array([0.0])  # Below lower bound
        lower = np.array([1.0])
        upper = np.array([2.0])
        alpha = 0.1
        
        score = interval_score(y_true, lower, upper, alpha=alpha)
        
        # Width (1) + penalty for below (2/0.1 * 1 = 20)
        expected = 1.0 + (2 / alpha) * 1.0
        assert score == pytest.approx(expected)
    
    def test_interval_score_missed_above(self):
        """Point above upper bound adds penalty."""
        y_true = np.array([3.0])  # Above upper bound
        lower = np.array([1.0])
        upper = np.array([2.0])
        alpha = 0.1
        
        score = interval_score(y_true, lower, upper, alpha=alpha)
        
        # Width (1) + penalty for above (2/0.1 * 1 = 20)
        expected = 1.0 + (2 / alpha) * 1.0
        assert score == pytest.approx(expected)
    
    def test_interval_score_narrower_is_better(self):
        """Narrower intervals (with same coverage) are better."""
        y_true = np.array([1.5, 2.5, 3.5])
        
        wide_lower = np.array([0.0, 1.0, 2.0])
        wide_upper = np.array([3.0, 4.0, 5.0])
        
        narrow_lower = np.array([1.0, 2.0, 3.0])
        narrow_upper = np.array([2.0, 3.0, 4.0])
        
        score_wide = interval_score(y_true, wide_lower, wide_upper, alpha=0.1)
        score_narrow = interval_score(y_true, narrow_lower, narrow_upper, alpha=0.1)
        
        assert score_narrow < score_wide
    
    def test_interval_score_invalid_alpha(self):
        """Invalid alpha raises error."""
        y_true = np.array([1.0])
        lower = np.array([0.0])
        upper = np.array([2.0])
        
        with pytest.raises(ValueError, match="alpha must be in"):
            interval_score(y_true, lower, upper, alpha=0.0)
        
        with pytest.raises(ValueError, match="alpha must be in"):
            interval_score(y_true, lower, upper, alpha=1.0)
    
    def test_interval_score_with_sample_weight(self):
        """Interval score with sample weights."""
        y_true = np.array([1.5, 5.0])  # Second point outside interval
        lower = np.array([1.0, 1.0])
        upper = np.array([2.0, 2.0])
        
        score_unweighted = interval_score(y_true, lower, upper, alpha=0.1)
        
        # Weight the covered point more heavily
        weights = np.array([10.0, 0.1])
        score_weighted = interval_score(y_true, lower, upper, alpha=0.1, sample_weight=weights)
        
        assert score_weighted < score_unweighted


class TestExpectedCalibrationError:
    """Tests for Expected Calibration Error (ECE)."""
    
    def test_ece_perfectly_calibrated(self):
        """Perfectly calibrated model has ECE = 0."""
        # 10% of samples with prob=0.1 have label=1, etc.
        np.random.seed(42)
        n = 1000
        y_prob = np.random.random(n)
        y_true = (np.random.random(n) < y_prob).astype(int)
        
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        # Won't be exactly 0 due to finite sample, but should be small
        assert ece < 0.1
    
    def test_ece_overconfident(self):
        """Overconfident model has high ECE."""
        # Model predicts 0.9 but actual rate is 0.5
        y_true = np.array([0, 1] * 50)  # 50% positive rate
        y_prob = np.array([0.9] * 100)  # Always predict 0.9
        
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        # ECE should be approximately |0.5 - 0.9| = 0.4
        assert ece > 0.3
    
    def test_ece_underconfident(self):
        """Underconfident model has high ECE."""
        # Model predicts 0.3 for negatives and 0.6 for positives
        # but actual rates differ significantly
        y_true = np.array([0] * 50 + [1] * 50)
        # Underconfident: predicts 0.3 but 0% are positive (among those predictions)
        # and 0.6 but 100% are positive
        y_prob = np.array([0.3] * 50 + [0.6] * 50)
        
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        # ECE should reflect the miscalibration (|0 - 0.3| and |1 - 0.6|)
        assert ece > 0.2
    
    def test_ece_uniform_vs_quantile(self):
        """Uniform and quantile strategies give different results."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.beta(2, 5, 100)  # Skewed towards low probs
        
        ece_uniform = expected_calibration_error(y_true, y_prob, strategy='uniform')
        ece_quantile = expected_calibration_error(y_true, y_prob, strategy='quantile')
        
        # Both should be reasonable
        assert 0 <= ece_uniform <= 1
        assert 0 <= ece_quantile <= 1


class TestCalibrationCurve:
    """Tests for calibration curve computation."""
    
    def test_calibration_curve_basic(self):
        """Basic calibration curve computation."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        frac_pos, mean_pred, counts = calibration_curve(y_true, y_prob, n_bins=5)
        
        assert len(frac_pos) > 0
        assert len(mean_pred) == len(frac_pos)
        assert len(counts) == len(frac_pos)
        
        # Fractions should be in [0, 1]
        assert np.all(frac_pos >= 0) and np.all(frac_pos <= 1)
        
        # Mean predictions should be in [0, 1]
        assert np.all(mean_pred >= 0) and np.all(mean_pred <= 1)
        
        # Counts should sum to n_samples
        assert np.sum(counts) == len(y_true)
    
    def test_calibration_curve_perfectly_calibrated(self):
        """Perfectly calibrated model has diagonal curve."""
        np.random.seed(42)
        n = 10000
        
        # Generate perfectly calibrated predictions
        y_prob = np.random.random(n)
        y_true = (np.random.random(n) < y_prob).astype(int)
        
        frac_pos, mean_pred, _ = calibration_curve(y_true, y_prob, n_bins=10)
        
        # For perfect calibration, frac_pos ≈ mean_pred
        differences = np.abs(frac_pos - mean_pred)
        assert np.mean(differences) < 0.1
    
    def test_calibration_curve_quantile_strategy(self):
        """Quantile strategy ensures equal bin sizes."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.beta(2, 5, 100)  # Skewed distribution
        
        _, _, counts = calibration_curve(y_true, y_prob, n_bins=5, strategy='quantile')
        
        # Bin sizes should be roughly equal with quantile strategy
        expected_per_bin = len(y_true) / 5
        for count in counts:
            assert abs(count - expected_per_bin) < expected_per_bin * 0.5


class TestNegativeLogLikelihood:
    """Tests for negative log-likelihood."""
    
    def test_nll_perfect_predictions_small(self):
        """Perfect mean predictions with small std have low NLL."""
        y_true = np.array([1.0, 2.0, 3.0])
        mean = np.array([1.0, 2.0, 3.0])
        std = np.array([0.1, 0.1, 0.1])
        
        nll = negative_log_likelihood(y_true, mean, std)
        # NLL should be reasonable for good predictions
        assert nll < 5
    
    def test_nll_larger_std(self):
        """Larger std increases NLL even for perfect means."""
        y_true = np.array([1.0, 2.0, 3.0])
        mean = np.array([1.0, 2.0, 3.0])
        
        nll_small_std = negative_log_likelihood(y_true, mean, np.array([0.1, 0.1, 0.1]))
        nll_large_std = negative_log_likelihood(y_true, mean, np.array([1.0, 1.0, 1.0]))
        
        # Larger std → higher NLL for same mean
        assert nll_small_std < nll_large_std
    
    def test_nll_wrong_mean(self):
        """Wrong mean predictions increase NLL."""
        y_true = np.array([1.0, 2.0, 3.0])
        std = np.array([0.5, 0.5, 0.5])
        
        nll_good = negative_log_likelihood(y_true, np.array([1.0, 2.0, 3.0]), std)
        nll_bad = negative_log_likelihood(y_true, np.array([2.0, 3.0, 4.0]), std)
        
        assert nll_good < nll_bad
    
    def test_nll_invalid_std_raises(self):
        """Negative or zero std should raise error."""
        y_true = np.array([1.0])
        mean = np.array([1.0])
        
        with pytest.raises(ValueError, match="std must be positive"):
            negative_log_likelihood(y_true, mean, np.array([0.0]))
    
    def test_nll_with_sample_weight(self):
        """NLL with sample weights."""
        y_true = np.array([1.0, 2.0, 3.0])
        mean = np.array([1.0, 5.0, 3.0])  # Middle prediction is bad
        std = np.array([0.5, 0.5, 0.5])
        
        nll_unweighted = negative_log_likelihood(y_true, mean, std)
        
        # Weight down the bad prediction
        weights = np.array([1.0, 0.1, 1.0])
        nll_weighted = negative_log_likelihood(y_true, mean, std, sample_weight=weights)
        
        assert nll_weighted < nll_unweighted


class TestProbabilisticMetricsWithOpenBoost:
    """Integration tests: probabilistic metrics with NaturalBoost models."""
    
    def test_crps_with_naturalboost(self):
        """CRPS evaluation on NaturalBoost predictions."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] * 2 + np.random.randn(200).astype(np.float32) * 0.5
        
        model = OpenBoostDistributionalRegressor(
            distribution='normal',
            n_estimators=20,
            max_depth=3
        )
        model.fit(X, y)
        
        # Get distribution parameters (dict with 'loc' and 'scale')
        output = model.predict_distribution(X)
        mean = output['loc']
        std = np.maximum(output['scale'], 1e-6)  # Ensure positive
        
        crps = crps_gaussian(y, mean, std)
        
        assert crps >= 0
        assert crps < 2  # Should be reasonable for a fitted model
    
    def test_interval_score_with_naturalboost(self):
        """Interval score evaluation on NaturalBoost predictions."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] * 2 + np.random.randn(200).astype(np.float32) * 0.5
        
        model = OpenBoostDistributionalRegressor(
            distribution='normal',
            n_estimators=20,
            max_depth=3
        )
        model.fit(X, y)
        
        lower, upper = model.predict_interval(X, alpha=0.1)
        
        score = interval_score(y, lower, upper, alpha=0.1)
        
        assert score >= 0
    
    def test_nll_with_naturalboost(self):
        """NLL evaluation on NaturalBoost predictions."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] * 2 + np.random.randn(200).astype(np.float32) * 0.5
        
        model = OpenBoostDistributionalRegressor(
            distribution='normal',
            n_estimators=20,
            max_depth=3
        )
        model.fit(X, y)
        
        # Get distribution parameters (dict with 'loc' and 'scale')
        output = model.predict_distribution(X)
        mean = output['loc']
        std = np.maximum(output['scale'], 1e-6)
        
        nll = negative_log_likelihood(y, mean, std)
        
        # NLL should be reasonable
        assert nll < 10
    
    def test_brier_with_classifier(self):
        """Brier score with OpenBoostClassifier."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int32)
        
        model = OpenBoostClassifier(n_estimators=20, max_depth=3)
        model.fit(X, y)
        
        y_proba = model.predict_proba(X)[:, 1]
        
        brier = brier_score(y, y_proba)
        
        assert 0 <= brier <= 1
        assert brier < 0.25  # Better than random
    
    def test_calibration_with_classifier(self):
        """Calibration metrics with OpenBoostClassifier."""
        np.random.seed(42)
        X = np.random.randn(500, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int32)
        
        model = OpenBoostClassifier(n_estimators=30, max_depth=3)
        model.fit(X, y)
        
        y_proba = model.predict_proba(X)[:, 1]
        
        ece = expected_calibration_error(y, y_proba)
        frac_pos, mean_pred, counts = calibration_curve(y, y_proba, n_bins=5)
        
        assert 0 <= ece <= 1
        assert len(frac_pos) > 0
        assert np.sum(counts) == len(y)
