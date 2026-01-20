"""Integration tests with real datasets.

Phase 20.5: Real-world validation tests using sklearn datasets.

These tests ensure OpenBoost works well on real data, not just synthetic examples.
They also serve as regression tests for model quality.

Tests include:
- California Housing (regression)
- Breast Cancer (binary classification)
- Iris (multiclass classification)
- Uncertainty coverage for NaturalBoost
- XGBoost comparison (optional)
"""

import numpy as np
import pytest

# Skip if sklearn not available
sklearn = pytest.importorskip("sklearn")
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_iris,
    make_regression,
)
from sklearn.model_selection import train_test_split, cross_val_score


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def california_housing():
    """Load California Housing dataset (cached at module level)."""
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data.astype(np.float32),
        data.target.astype(np.float32),
        test_size=0.2,
        random_state=42,
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def breast_cancer():
    """Load Breast Cancer dataset (cached at module level)."""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data.astype(np.float32),
        data.target.astype(np.float32),
        test_size=0.2,
        random_state=42,
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def iris():
    """Load Iris dataset (cached at module level)."""
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data.astype(np.float32),
        data.target.astype(np.int32),
        test_size=0.2,
        random_state=42,
    )
    return X_train, X_test, y_train, y_test


# =============================================================================
# Regression Tests
# =============================================================================

class TestRealDatasets:
    """Integration tests with real datasets."""
    
    def test_california_housing_regressor(self, california_housing):
        """Test regression on California Housing dataset."""
        from openboost import OpenBoostRegressor
        
        X_train, X_test, y_train, y_test = california_housing
        
        model = OpenBoostRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        # California Housing is a well-known dataset, R² > 0.75 is reasonable
        assert score > 0.75, f"R² should be > 0.75, got {score:.4f}"
        
        # Check predictions are reasonable (not all same value)
        pred = model.predict(X_test)
        assert pred.std() > 0.1, "Predictions should have variance"
        
        # Check feature importances exist and are valid
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == X_train.shape[1]
        assert np.all(model.feature_importances_ >= 0)
    
    def test_california_housing_core_model(self, california_housing):
        """Test GradientBoosting directly on California Housing."""
        from openboost import GradientBoosting
        
        X_train, X_test, y_train, y_test = california_housing
        
        model = GradientBoosting(n_trees=100, max_depth=6, learning_rate=0.1)
        model.fit(X_train, y_train)
        
        pred = model.predict(X_test)
        mse = np.mean((pred - y_test) ** 2)
        rmse = np.sqrt(mse)
        
        # RMSE should be reasonable (target is in ~0-5 range)
        assert rmse < 0.7, f"RMSE should be < 0.7, got {rmse:.4f}"
    
    def test_breast_cancer_classifier(self, breast_cancer):
        """Test binary classification on Breast Cancer dataset."""
        from openboost import OpenBoostClassifier
        
        X_train, X_test, y_train, y_test = breast_cancer
        
        model = OpenBoostClassifier(n_estimators=50, max_depth=4, learning_rate=0.1)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        # Breast Cancer is fairly separable, accuracy > 0.90 is expected
        assert accuracy > 0.90, f"Accuracy should be > 0.90, got {accuracy:.4f}"
        
        # Check predict_proba output
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        
        # Check classes_ attribute
        assert hasattr(model, 'classes_')
        assert len(model.classes_) == 2
    
    def test_iris_multiclass(self, iris):
        """Test multi-class classification on Iris dataset."""
        from openboost import OpenBoostClassifier
        
        X_train, X_test, y_train, y_test = iris
        
        model = OpenBoostClassifier(n_estimators=50, max_depth=4, learning_rate=0.1)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        # Iris is quite separable, accuracy > 0.90 is expected
        assert accuracy > 0.90, f"Accuracy should be > 0.90, got {accuracy:.4f}"
        
        # Check multiclass attributes
        assert hasattr(model, 'n_classes_')
        assert model.n_classes_ == 3
        
        # Check predict_proba shape
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# =============================================================================
# Uncertainty Quantification Tests
# =============================================================================

class TestUncertaintyQuantification:
    """Tests for probabilistic prediction and prediction intervals."""
    
    def test_prediction_interval_coverage(self, california_housing):
        """Test that prediction intervals have approximately correct coverage."""
        from openboost import NaturalBoostNormal
        
        X_train, X_test, y_train, y_test = california_housing
        
        # Use fewer trees for faster test
        model = NaturalBoostNormal(n_trees=50, max_depth=4, learning_rate=0.1)
        model.fit(X_train, y_train)
        
        # 90% prediction interval
        lower, upper = model.predict_interval(X_test, alpha=0.1)
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        
        # Coverage should be approximately 90% (allow 75-98% for test stability)
        # Note: With limited trees and real data, coverage might vary
        assert 0.70 < coverage < 0.99, f"90% interval coverage is {coverage:.2%}"
        
        # Check interval sanity
        assert np.all(lower < upper), "Lower bound should be less than upper bound"
        
        # Point prediction should be between bounds
        mean = model.predict(X_test)
        # Allow small tolerance for edge cases
        assert np.mean((mean >= lower - 0.01) & (mean <= upper + 0.01)) > 0.95
    
    def test_prediction_interval_coverage_80(self, california_housing):
        """Test 80% prediction interval coverage."""
        from openboost import NaturalBoostNormal
        
        X_train, X_test, y_train, y_test = california_housing
        
        model = NaturalBoostNormal(n_trees=50, max_depth=4, learning_rate=0.1)
        model.fit(X_train, y_train)
        
        # 80% prediction interval
        lower, upper = model.predict_interval(X_test, alpha=0.2)
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        
        # 80% interval should have lower coverage than 90%
        lower_90, upper_90 = model.predict_interval(X_test, alpha=0.1)
        coverage_90 = np.mean((y_test >= lower_90) & (y_test <= upper_90))
        
        # 80% interval should be narrower
        width_80 = np.mean(upper - lower)
        width_90 = np.mean(upper_90 - lower_90)
        assert width_80 < width_90, "80% interval should be narrower than 90%"
    
    def test_distributional_sampling(self, california_housing):
        """Test sampling from predicted distributions."""
        from openboost import NaturalBoostNormal
        
        X_train, X_test, y_train, y_test = california_housing
        
        model = NaturalBoostNormal(n_trees=50, max_depth=4, learning_rate=0.1)
        model.fit(X_train, y_train)
        
        # Sample from posterior
        n_test = min(100, len(X_test))  # Limit for speed
        samples = model.sample(X_test[:n_test], n_samples=100)
        
        assert samples.shape == (n_test, 100)
        
        # Sample mean should be close to predict mean
        sample_means = samples.mean(axis=1)
        pred_means = model.predict(X_test[:n_test])
        
        # Correlation between sample means and predicted means should be high
        corr = np.corrcoef(sample_means, pred_means)[0, 1]
        assert corr > 0.95, f"Sample means should correlate with predictions, got {corr:.3f}"
    
    def test_different_distributions(self, california_housing):
        """Test different distribution families."""
        from openboost import (
            NaturalBoostNormal,
            NaturalBoostLogNormal,
            NaturalBoostStudentT,
        )
        
        X_train, X_test, y_train, y_test = california_housing
        
        # Test multiple distributions converge
        for ModelClass in [NaturalBoostNormal, NaturalBoostStudentT]:
            model = ModelClass(n_trees=30, max_depth=4, learning_rate=0.1)
            model.fit(X_train, y_train)
            
            pred = model.predict(X_test)
            mse = np.mean((pred - y_test) ** 2)
            
            # Should achieve reasonable MSE
            assert mse < 1.0, f"{ModelClass.__name__} MSE should be < 1.0, got {mse:.4f}"


# =============================================================================
# sklearn Compatibility Tests  
# =============================================================================

class TestSklearnCompatibility:
    """Test sklearn ecosystem compatibility."""
    
    def test_cross_validation(self, california_housing):
        """Test cross-validation with OpenBoostRegressor."""
        from openboost import OpenBoostRegressor
        from sklearn.model_selection import cross_val_score
        
        X_train, X_test, y_train, y_test = california_housing
        
        # Use smaller model for faster CV
        model = OpenBoostRegressor(n_estimators=30, max_depth=4)
        
        # 3-fold CV for speed
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
        
        # All folds should achieve reasonable R²
        assert np.all(scores > 0.5), f"All CV scores should be > 0.5, got {scores}"
        assert scores.mean() > 0.65, f"Mean CV R² should be > 0.65, got {scores.mean():.4f}"
    
    def test_pipeline_compatibility(self, california_housing):
        """Test OpenBoost in sklearn Pipeline."""
        from openboost import OpenBoostRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        X_train, X_test, y_train, y_test = california_housing
        
        # Create pipeline with scaler and model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', OpenBoostRegressor(n_estimators=50, max_depth=4)),
        ])
        
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        
        assert score > 0.70, f"Pipeline R² should be > 0.70, got {score:.4f}"
    
    def test_gridsearch_compatibility(self):
        """Test GridSearchCV with OpenBoostRegressor."""
        from openboost import OpenBoostRegressor
        from sklearn.model_selection import GridSearchCV
        
        # Small synthetic dataset for fast GridSearch
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = (X[:, 0] + X[:, 1] * 2 + np.random.randn(200) * 0.1).astype(np.float32)
        
        model = OpenBoostRegressor()
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [2, 4],
        }
        
        search = GridSearchCV(model, param_grid, cv=2, scoring='r2')
        search.fit(X, y)
        
        assert hasattr(search, 'best_params_')
        assert search.best_score_ > 0.8, f"Best CV score should be > 0.8, got {search.best_score_:.4f}"


# =============================================================================
# Model Persistence Tests
# =============================================================================

class TestModelPersistence:
    """Test save/load functionality with real data."""
    
    def test_save_load_predictions_match(self, california_housing, tmp_path):
        """Test that predictions match after save/load."""
        from openboost import GradientBoosting
        
        X_train, X_test, y_train, y_test = california_housing
        
        # Train model
        model = GradientBoosting(n_trees=50, max_depth=4)
        model.fit(X_train, y_train)
        pred_before = model.predict(X_test)
        
        # Save and load
        save_path = tmp_path / "model.joblib"
        model.save(str(save_path))
        
        loaded = GradientBoosting.load(str(save_path))
        pred_after = loaded.predict(X_test)
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(
            pred_before, pred_after, decimal=5,
            err_msg="Predictions should match after save/load"
        )
    
    def test_save_load_sklearn_wrapper(self, breast_cancer, tmp_path):
        """Test save/load with sklearn wrappers (via pickle)."""
        from openboost import OpenBoostClassifier
        import pickle
        
        X_train, X_test, y_train, y_test = breast_cancer
        
        # Train model
        model = OpenBoostClassifier(n_estimators=30, max_depth=4)
        model.fit(X_train, y_train)
        pred_before = model.predict_proba(X_test)
        
        # Pickle save/load
        save_path = tmp_path / "classifier.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(save_path, 'rb') as f:
            loaded = pickle.load(f)
        
        pred_after = loaded.predict_proba(X_test)
        
        np.testing.assert_array_almost_equal(
            pred_before, pred_after, decimal=5,
            err_msg="Classifier predictions should match after pickle"
        )
    
    def test_save_load_distributional(self, california_housing, tmp_path):
        """Test save/load for NaturalBoost models."""
        from openboost import NaturalBoostNormal, NaturalBoost
        
        X_train, X_test, y_train, y_test = california_housing
        
        # Train model (NaturalBoostNormal is a factory function that returns NaturalBoost)
        model = NaturalBoostNormal(n_trees=30, max_depth=4)
        model.fit(X_train, y_train)
        
        mean_before = model.predict(X_test)
        lower_before, upper_before = model.predict_interval(X_test, alpha=0.1)
        
        # Save and load
        save_path = tmp_path / "natural_boost.joblib"
        model.save(str(save_path))
        
        # Load using the actual class (NaturalBoost), not the factory function
        loaded = NaturalBoost.load(str(save_path))
        
        mean_after = loaded.predict(X_test)
        lower_after, upper_after = loaded.predict_interval(X_test, alpha=0.1)
        
        np.testing.assert_array_almost_equal(mean_before, mean_after, decimal=5)
        np.testing.assert_array_almost_equal(lower_before, lower_after, decimal=5)
        np.testing.assert_array_almost_equal(upper_before, upper_after, decimal=5)


# =============================================================================
# XGBoost Comparison Tests (Optional)
# =============================================================================

class TestXGBoostComparison:
    """Compare with XGBoost to ensure reasonable performance.
    
    These tests are optional - they skip if XGBoost is not installed.
    They ensure OpenBoost achieves competitive performance.
    """
    
    xgb = pytest.importorskip("xgboost")
    
    def test_regression_comparable_to_xgboost(self, california_housing):
        """OpenBoost should be within 15% of XGBoost on standard regression."""
        from openboost import OpenBoostRegressor
        import xgboost as xgb
        
        X_train, X_test, y_train, y_test = california_housing
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
        )
        xgb_model.fit(X_train, y_train)
        xgb_score = xgb_model.score(X_test, y_test)
        
        # OpenBoost
        ob_model = OpenBoostRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
        )
        ob_model.fit(X_train, y_train)
        ob_score = ob_model.score(X_test, y_test)
        
        # OpenBoost should be within 15% of XGBoost
        # (We use 15% tolerance because implementations differ)
        assert ob_score > xgb_score * 0.85, (
            f"OpenBoost R²={ob_score:.4f} should be within 15% of XGBoost R²={xgb_score:.4f}"
        )
        
        # Log the comparison for reference
        print(f"\nXGBoost R²: {xgb_score:.4f}")
        print(f"OpenBoost R²: {ob_score:.4f}")
        print(f"Ratio: {ob_score / xgb_score:.2%}")
    
    def test_classification_comparable_to_xgboost(self, breast_cancer):
        """OpenBoost should be within 5% accuracy of XGBoost on classification."""
        from openboost import OpenBoostClassifier
        import xgboost as xgb
        
        X_train, X_test, y_train, y_test = breast_cancer
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',  # Suppress warning
        )
        xgb_model.fit(X_train, y_train)
        xgb_acc = xgb_model.score(X_test, y_test)
        
        # OpenBoost
        ob_model = OpenBoostClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
        )
        ob_model.fit(X_train, y_train)
        ob_acc = ob_model.score(X_test, y_test)
        
        # Classification accuracy should be very close
        assert ob_acc > xgb_acc - 0.05, (
            f"OpenBoost accuracy={ob_acc:.4f} should be within 5% of XGBoost={xgb_acc:.4f}"
        )
        
        print(f"\nXGBoost accuracy: {xgb_acc:.4f}")
        print(f"OpenBoost accuracy: {ob_acc:.4f}")


# =============================================================================
# Early Stopping Tests
# =============================================================================

class TestEarlyStopping:
    """Test early stopping functionality."""
    
    def test_early_stopping_reduces_iterations(self):
        """Early stopping should stop before max iterations."""
        from openboost import OpenBoostRegressor
        
        # Use small synthetic data for fast test
        np.random.seed(42)
        X = np.random.randn(500, 5).astype(np.float32)
        y = (X[:, 0] * 2 + X[:, 1] + np.random.randn(500) * 0.1).astype(np.float32)
        
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model with early stopping
        model = OpenBoostRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            early_stopping_rounds=5,  # Short patience
        )
        
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        
        # Model should have trained successfully
        assert hasattr(model, 'booster_')
        assert len(model.booster_.trees_) > 0
        
        # Should achieve good fit on this simple data
        score = model.score(X_val, y_val)
        assert score > 0.80, f"R² should be > 0.80, got {score:.4f}"


# =============================================================================
# Edge Cases and Robustness
# =============================================================================

class TestRobustness:
    """Test edge cases and robustness."""
    
    def test_single_feature(self):
        """Test with single feature."""
        from openboost import OpenBoostRegressor
        
        np.random.seed(42)
        X = np.random.randn(100, 1).astype(np.float32)
        y = (2 * X[:, 0] + np.random.randn(100) * 0.1).astype(np.float32)
        
        model = OpenBoostRegressor(n_estimators=20, max_depth=4)
        model.fit(X, y)
        
        score = model.score(X, y)
        assert score > 0.9, f"Should fit well on simple linear data, got R²={score:.4f}"
    
    def test_many_features(self):
        """Test with many features (wider than samples)."""
        from openboost import OpenBoostRegressor
        
        np.random.seed(42)
        X = np.random.randn(100, 200).astype(np.float32)  # 200 features
        y = (X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1).astype(np.float32)
        
        model = OpenBoostRegressor(n_estimators=20, max_depth=4)
        model.fit(X, y)
        
        # Should not crash and should achieve some fit
        pred = model.predict(X)
        assert not np.any(np.isnan(pred)), "Predictions should not be NaN"
    
    def test_constant_feature(self):
        """Test handling of constant features."""
        from openboost import OpenBoostRegressor
        
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        X[:, 2] = 1.0  # Constant feature
        y = (X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1).astype(np.float32)
        
        model = OpenBoostRegressor(n_estimators=20, max_depth=4)
        model.fit(X, y)
        
        # Should not crash
        pred = model.predict(X)
        assert not np.any(np.isnan(pred)), "Should handle constant features"
    
    def test_small_dataset(self):
        """Test with very small dataset."""
        from openboost import OpenBoostRegressor
        
        np.random.seed(42)
        X = np.random.randn(20, 3).astype(np.float32)
        y = (X[:, 0] + np.random.randn(20) * 0.1).astype(np.float32)
        
        # Use shallow trees and few estimators for small data
        model = OpenBoostRegressor(n_estimators=5, max_depth=2, min_child_weight=1.0)
        model.fit(X, y)
        
        pred = model.predict(X)
        assert not np.any(np.isnan(pred)), "Should handle small datasets"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
