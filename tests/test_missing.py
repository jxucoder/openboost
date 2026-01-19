"""Tests for Phase 14: Missing Value Handling.

Tests cover:
- NaN detection and encoding in BinnedArray
- Histogram building with missing values
- Split finding with optimal missing direction
- Tree prediction with missing values
- End-to-end training with missing data
"""

import numpy as np
import pytest

import openboost as ob
from openboost import (
    array,
    BinnedArray,
    MISSING_BIN,
    fit_tree,
    GradientBoosting,
)


class TestBinningWithMissing:
    """Tests for NaN handling in ob.array()."""
    
    def test_missing_bin_constant(self):
        """MISSING_BIN is correctly defined."""
        assert MISSING_BIN == 255
    
    def test_nan_detection(self):
        """NaN values are detected in BinnedArray."""
        X = np.array([
            [1.0, np.nan, 3.0],
            [2.0, 2.0, np.nan],
            [np.nan, 3.0, 5.0],
        ], dtype=np.float32)
        
        X_binned = array(X)
        
        # Check has_missing is set correctly
        assert len(X_binned.has_missing) == 3
        assert X_binned.has_missing[0] == True   # Feature 0 has NaN
        assert X_binned.has_missing[1] == True   # Feature 1 has NaN
        assert X_binned.has_missing[2] == True   # Feature 2 has NaN
    
    def test_nan_encoded_as_missing_bin(self):
        """NaN values are encoded as bin 255."""
        X = np.array([
            [1.0, np.nan],
            [2.0, 3.0],
            [np.nan, 4.0],
        ], dtype=np.float32)
        
        X_binned = array(X)
        
        # Check that NaN positions have bin 255
        # Data is transposed: (n_features, n_samples)
        assert X_binned.data[0, 2] == MISSING_BIN  # X[2,0] was NaN
        assert X_binned.data[1, 0] == MISSING_BIN  # X[0,1] was NaN
        
        # Non-NaN values should not be 255
        assert X_binned.data[0, 0] != MISSING_BIN
        assert X_binned.data[1, 1] != MISSING_BIN
    
    def test_no_missing_values(self):
        """BinnedArray handles data without NaN."""
        X = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ], dtype=np.float32)
        
        X_binned = array(X)
        
        # has_missing should all be False
        assert not np.any(X_binned.has_missing)
        assert not X_binned.any_missing
    
    def test_all_missing_feature(self):
        """Handle feature with all NaN values."""
        X = np.array([
            [1.0, np.nan],
            [2.0, np.nan],
            [3.0, np.nan],
        ], dtype=np.float32)
        
        X_binned = array(X)
        
        # Feature 1 is all NaN
        assert X_binned.has_missing[1] == True
        # All values in feature 1 should be MISSING_BIN
        assert np.all(X_binned.data[1, :] == MISSING_BIN)
    
    def test_binned_array_repr_shows_missing(self):
        """BinnedArray repr shows missing info."""
        X = np.array([
            [1.0, np.nan],
            [2.0, 3.0],
        ], dtype=np.float32)
        
        X_binned = array(X)
        repr_str = repr(X_binned)
        
        assert "features_with_missing" in repr_str


class TestFitTreeWithMissing:
    """Tests for tree fitting with missing values."""
    
    def test_fit_tree_with_missing(self):
        """fit_tree works with missing values."""
        np.random.seed(42)
        n_samples = 100
        
        # Create data with missing values
        X = np.random.randn(n_samples, 3).astype(np.float32)
        X[np.random.rand(n_samples, 3) < 0.1] = np.nan  # 10% missing
        
        y = X[:, 0].copy()
        y[np.isnan(y)] = 0  # Use 0 for missing in y for this test
        y = y + np.random.randn(n_samples).astype(np.float32) * 0.1
        
        X_binned = array(X)
        
        # Compute gradients (MSE)
        pred = np.zeros(n_samples, dtype=np.float32)
        grad = 2 * (pred - y)
        hess = np.ones_like(grad) * 2
        
        # Should not raise
        tree = fit_tree(X_binned, grad, hess, max_depth=3)
        
        assert tree is not None
        assert tree.n_nodes > 0
    
    def test_tree_has_missing_directions(self):
        """Tree stores missing direction for each split."""
        np.random.seed(42)
        
        X = np.array([
            [1.0, np.nan],
            [2.0, 1.0],
            [np.nan, 2.0],
            [4.0, 3.0],
        ], dtype=np.float32)
        y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        
        X_binned = array(X)
        grad = -2 * y
        hess = np.ones_like(grad) * 2
        
        tree = fit_tree(X_binned, grad, hess, max_depth=2)
        
        # Tree should have missing_go_left array
        assert tree.missing_go_left is not None
        assert len(tree.missing_go_left) == tree.n_nodes
    
    def test_predict_with_missing(self):
        """Prediction handles missing values correctly."""
        np.random.seed(42)
        
        # Create training data
        X_train = np.random.randn(100, 2).astype(np.float32)
        y_train = X_train[:, 0] + 0.1 * np.random.randn(100).astype(np.float32)
        
        X_binned = array(X_train)
        grad = 2 * (np.zeros(100) - y_train)
        hess = np.ones(100, dtype=np.float32) * 2
        
        tree = fit_tree(X_binned, grad, hess, max_depth=3)
        
        # Create test data with missing values
        X_test = np.array([
            [1.0, np.nan],
            [np.nan, 2.0],
            [1.5, 2.5],
        ], dtype=np.float32)
        
        X_test_binned = array(X_test)
        
        # Should not raise
        predictions = tree(X_test_binned)
        
        assert predictions.shape == (3,)
        assert not np.any(np.isnan(predictions))


class TestGradientBoostingWithMissing:
    """Tests for GradientBoosting with missing values."""
    
    def test_gradient_boosting_fit_with_missing(self):
        """GradientBoosting fits with missing values."""
        np.random.seed(42)
        n_samples = 200
        
        X = np.random.randn(n_samples, 5).astype(np.float32)
        X[np.random.rand(n_samples, 5) < 0.1] = np.nan  # 10% missing
        y = X[:, 0].copy()
        y[np.isnan(y)] = 0
        
        model = GradientBoosting(n_trees=10, max_depth=3)
        model.fit(X, y)
        
        assert len(model.trees_) == 10
    
    def test_gradient_boosting_predict_with_missing(self):
        """GradientBoosting predicts with missing values."""
        np.random.seed(42)
        
        # Train on clean data
        X_train = np.random.randn(100, 3).astype(np.float32)
        y_train = X_train[:, 0] + 0.1 * np.random.randn(100).astype(np.float32)
        
        model = GradientBoosting(n_trees=10, max_depth=3)
        model.fit(X_train, y_train)
        
        # Predict on data with missing values
        X_test = np.random.randn(20, 3).astype(np.float32)
        X_test[np.random.rand(20, 3) < 0.2] = np.nan  # 20% missing
        
        predictions = model.predict(X_test)
        
        assert predictions.shape == (20,)
        assert not np.any(np.isnan(predictions))
    
    def test_missing_learns_optimal_direction(self):
        """Model learns optimal direction for missing values.
        
        Create data where missing values have a clear pattern:
        - When feature 0 is missing, target is high
        - This should make the model learn to send missing values
          to the branch with higher predictions
        """
        np.random.seed(42)
        n_samples = 500
        
        X = np.random.randn(n_samples, 2).astype(np.float32)
        y = np.zeros(n_samples, dtype=np.float32)
        
        # When feature 0 is "missing" (we'll mark some values), y is high
        missing_mask = np.random.rand(n_samples) < 0.3
        y[missing_mask] = 5.0
        y[~missing_mask] = X[~missing_mask, 0]
        
        # Set the marked values to NaN
        X[missing_mask, 0] = np.nan
        
        # Train model
        model = GradientBoosting(n_trees=50, max_depth=4, learning_rate=0.1)
        model.fit(X, y)
        
        # Test: create data where feature 0 is missing
        X_test = np.array([
            [np.nan, 0.0],
            [np.nan, 1.0],
            [np.nan, -1.0],
        ], dtype=np.float32)
        
        predictions = model.predict(X_test)
        
        # Missing values should predict closer to 5.0 (the high target for missing)
        assert np.mean(predictions) > 2.0  # Should be biased towards high values


class TestEdgeCases:
    """Edge case tests for missing value handling."""
    
    def test_single_sample_with_missing(self):
        """Handle single sample with missing value."""
        X = np.array([[np.nan, 1.0]], dtype=np.float32)
        
        X_binned = array(X)
        
        assert X_binned.n_samples == 1
        assert X_binned.has_missing[0] == True
    
    def test_all_features_missing_in_sample(self):
        """Handle sample with all features missing."""
        X = np.array([
            [1.0, 2.0],
            [np.nan, np.nan],  # All missing
            [3.0, 4.0],
        ], dtype=np.float32)
        
        X_binned = array(X)
        
        # Sample 1 should have all MISSING_BIN
        assert X_binned.data[0, 1] == MISSING_BIN
        assert X_binned.data[1, 1] == MISSING_BIN
    
    def test_mixed_inf_nan(self):
        """Handle inf values (treated as normal, not missing)."""
        X = np.array([
            [1.0, np.inf],
            [np.nan, -np.inf],
            [3.0, 4.0],
        ], dtype=np.float32)
        
        X_binned = array(X)
        
        # NaN should be MISSING_BIN
        assert X_binned.data[0, 1] == MISSING_BIN  # X[1,0] was NaN
        
        # inf values are not missing
        assert X_binned.has_missing[1] == False  # Feature 1 only has inf


class TestSplitFindingWithMissing:
    """Tests for split finding logic with missing values."""
    
    def test_split_evaluates_both_directions(self):
        """Split finding evaluates missing going left vs right."""
        from openboost._backends._cpu import find_best_split_with_missing_cpu
        
        # Create histogram where missing should go right for better gain
        n_features = 1
        hist_grad = np.zeros((n_features, 256), dtype=np.float64)
        hist_hess = np.zeros((n_features, 256), dtype=np.float64)
        
        # Bin 0: gradient=-10, hessian=10 (want to go left)
        hist_grad[0, 0] = -10.0
        hist_hess[0, 0] = 10.0
        
        # Bin 1: gradient=10, hessian=10 (want to go right)
        hist_grad[0, 1] = 10.0
        hist_hess[0, 1] = 10.0
        
        # Missing: gradient=5, hessian=5 (should go with bin 1 to right)
        hist_grad[0, 255] = 5.0
        hist_hess[0, 255] = 5.0
        
        total_grad = float(np.sum(hist_grad))
        total_hess = float(np.sum(hist_hess))
        
        has_missing = np.array([True], dtype=np.bool_)
        
        feature, threshold, gain, missing_left = find_best_split_with_missing_cpu(
            hist_grad, hist_hess, total_grad, total_hess,
            reg_lambda=1.0, min_child_weight=1.0,
            has_missing=has_missing,
        )
        
        # Should find a valid split
        assert feature >= 0
        assert gain > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
