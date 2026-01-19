"""Tests for Phase 11 parameters: gamma, reg_alpha, subsample, colsample_bytree."""

import numpy as np
import pytest

import openboost as ob


class TestGamma:
    """Tests for gamma (min_split_gain) parameter."""
    
    def test_gamma_alias_works(self):
        """Test that gamma is an alias for min_gain."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0].astype(np.float32)
        binned = ob.array(X)
        
        grad = (np.zeros(100) - y).astype(np.float32) * 2
        hess = np.ones(100, dtype=np.float32) * 2
        
        # Both should work
        tree1 = ob.fit_tree(binned, grad, hess, gamma=0.1, max_depth=3)
        tree2 = ob.fit_tree(binned, grad, hess, min_gain=0.1, max_depth=3)
        
        assert tree1.n_nodes > 0
        assert tree2.n_nodes > 0
    
    def test_high_gamma_reduces_splits(self):
        """Test that high gamma results in fewer splits."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0].astype(np.float32)
        binned = ob.array(X)
        
        grad = (np.zeros(200) - y).astype(np.float32) * 2
        hess = np.ones(200, dtype=np.float32) * 2
        
        tree_low_gamma = ob.fit_tree(binned, grad, hess, gamma=0.0, max_depth=4)
        tree_high_gamma = ob.fit_tree(binned, grad, hess, gamma=10.0, max_depth=4)
        
        # High gamma should result in fewer nodes (more pruning)
        assert tree_high_gamma.n_nodes <= tree_low_gamma.n_nodes
    
    def test_gamma_in_gradient_boosting(self):
        """Test gamma parameter in GradientBoosting."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0].astype(np.float32)
        
        model = ob.GradientBoosting(n_trees=5, max_depth=3, gamma=0.5)
        model.fit(X, y)
        
        pred = model.predict(X)
        assert pred.shape == y.shape


class TestRegAlpha:
    """Tests for reg_alpha (L1 regularization) parameter."""
    
    def test_reg_alpha_produces_sparse_leaves(self):
        """Test that high reg_alpha produces zero leaf values."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        # Small gradients that should be zeroed by L1
        y = np.random.randn(100).astype(np.float32) * 0.01
        binned = ob.array(X)
        
        grad = (np.zeros(100) - y).astype(np.float32) * 2
        hess = np.ones(100, dtype=np.float32) * 2
        
        # Very high reg_alpha should zero out leaves
        tree = ob.fit_tree(binned, grad, hess, reg_alpha=100.0, max_depth=2)
        
        # Check that some leaf values are exactly zero
        leaf_values = tree.values
        n_zeros = np.sum(leaf_values == 0.0)
        assert n_zeros > 0, "High reg_alpha should produce some zero leaf values"
    
    def test_reg_alpha_soft_thresholding(self):
        """Test that reg_alpha applies soft-thresholding correctly."""
        from openboost._core._split import compute_leaf_value
        
        # Test case: gradient exactly at threshold
        val = compute_leaf_value(sum_grad=0.5, sum_hess=1.0, reg_lambda=1.0, reg_alpha=0.5)
        assert val == 0.0, "Gradient at threshold should give zero"
        
        # Test case: gradient above threshold
        val = compute_leaf_value(sum_grad=1.5, sum_hess=1.0, reg_lambda=1.0, reg_alpha=0.5)
        expected = -(1.5 - 0.5) / (1.0 + 1.0)  # -(G - alpha) / (H + lambda)
        assert abs(val - expected) < 1e-6
        
        # Test case: negative gradient
        val = compute_leaf_value(sum_grad=-1.5, sum_hess=1.0, reg_lambda=1.0, reg_alpha=0.5)
        expected = -(-1.5 + 0.5) / (1.0 + 1.0)  # -(G + alpha) / (H + lambda)
        assert abs(val - expected) < 1e-6
    
    def test_reg_alpha_in_gradient_boosting(self):
        """Test reg_alpha parameter in GradientBoosting."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0].astype(np.float32)
        
        model = ob.GradientBoosting(n_trees=5, max_depth=3, reg_alpha=0.1)
        model.fit(X, y)
        
        pred = model.predict(X)
        assert pred.shape == y.shape


class TestSubsample:
    """Tests for subsample (row sampling) parameter."""
    
    def test_subsample_affects_training(self):
        """Test that subsample produces different results."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] + 0.5 * X[:, 1]
        y = y.astype(np.float32)
        
        # Train with full data
        model_full = ob.GradientBoosting(n_trees=10, max_depth=3, subsample=1.0)
        model_full.fit(X, y)
        
        # Train with subsampling
        np.random.seed(123)  # Different seed for sampling
        model_sub = ob.GradientBoosting(n_trees=10, max_depth=3, subsample=0.5)
        model_sub.fit(X, y)
        
        # Predictions should be different
        pred_full = model_full.predict(X)
        pred_sub = model_sub.predict(X)
        
        assert not np.allclose(pred_full, pred_sub), "Subsample should affect predictions"
    
    def test_subsample_reduces_overfitting(self):
        """Test that subsample can reduce overfitting."""
        np.random.seed(42)
        # Small dataset prone to overfitting
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)
        
        # Both should train without error
        model_full = ob.GradientBoosting(n_trees=20, max_depth=5, subsample=1.0)
        model_full.fit(X, y)
        
        model_sub = ob.GradientBoosting(n_trees=20, max_depth=5, subsample=0.7)
        model_sub.fit(X, y)
        
        # Just check they produce valid predictions
        assert model_full.predict(X).shape == y.shape
        assert model_sub.predict(X).shape == y.shape
    
    def test_subsample_at_fit_tree_level(self):
        """Test subsample works at fit_tree level."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        binned = ob.array(X)
        
        grad = np.random.randn(100).astype(np.float32)
        hess = np.ones(100, dtype=np.float32)
        
        tree = ob.fit_tree(binned, grad, hess, subsample=0.5, max_depth=3)
        
        assert tree.n_nodes > 0


class TestColsampleBytree:
    """Tests for colsample_bytree (column sampling) parameter."""
    
    def test_colsample_in_gradient_boosting(self):
        """Test colsample_bytree parameter in GradientBoosting."""
        np.random.seed(42)
        X = np.random.randn(100, 10).astype(np.float32)
        y = X[:, 0].astype(np.float32)
        
        model = ob.GradientBoosting(n_trees=5, max_depth=3, colsample_bytree=0.5)
        model.fit(X, y)
        
        pred = model.predict(X)
        assert pred.shape == y.shape
    
    def test_colsample_at_fit_tree_level(self):
        """Test colsample_bytree works at fit_tree level."""
        np.random.seed(42)
        X = np.random.randn(100, 10).astype(np.float32)
        binned = ob.array(X)
        
        grad = np.random.randn(100).astype(np.float32)
        hess = np.ones(100, dtype=np.float32)
        
        tree = ob.fit_tree(binned, grad, hess, colsample_bytree=0.5, max_depth=3)
        
        assert tree.n_nodes > 0


class TestCombinedParameters:
    """Tests for combining multiple Phase 11 parameters."""
    
    def test_stochastic_gradient_boosting(self):
        """Test combining subsample and colsample (stochastic GB)."""
        np.random.seed(42)
        X = np.random.randn(200, 10).astype(np.float32)
        y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(200).astype(np.float32) * 0.1
        
        model = ob.GradientBoosting(
            n_trees=20,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        model.fit(X, y)
        
        pred = model.predict(X)
        mse = np.mean((pred - y) ** 2)
        
        # Should still learn something
        baseline_mse = np.mean(y ** 2)  # Predicting zeros
        assert mse < baseline_mse, "Model should improve over baseline"
    
    def test_regularized_stochastic_gb(self):
        """Test combining all Phase 11 parameters."""
        np.random.seed(42)
        X = np.random.randn(200, 10).astype(np.float32)
        y = X[:, 0] + 0.5 * X[:, 1]
        y = y.astype(np.float32)
        
        model = ob.GradientBoosting(
            n_trees=20,
            max_depth=4,
            gamma=0.1,
            reg_alpha=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        model.fit(X, y)
        
        pred = model.predict(X)
        r2 = 1 - np.mean((pred - y) ** 2) / np.var(y)
        
        # Should achieve reasonable R²
        assert r2 > 0.5, f"Model should learn pattern, got R²={r2:.3f}"


class TestMultiClassWithPhase11:
    """Tests for Phase 11 parameters with MultiClassGradientBoosting."""
    
    def test_multiclass_with_all_params(self):
        """Test MultiClassGradientBoosting with Phase 11 params."""
        np.random.seed(42)
        n_samples, n_features, n_classes = 200, 10, 3
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, n_classes, n_samples).astype(np.int32)
        
        model = ob.MultiClassGradientBoosting(
            n_classes=n_classes,
            n_trees=10,
            max_depth=3,
            gamma=0.1,
            reg_alpha=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        model.fit(X, y)
        
        pred = model.predict(X)
        assert pred.shape == y.shape
        assert np.all((pred >= 0) & (pred < n_classes))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
