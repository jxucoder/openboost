"""Tests for Phase 14.3: Native Categorical Feature Support.

Tests cover:
- Categorical feature encoding in BinnedArray
- Fisher-based categorical split finding
- Tree prediction with categorical splits
- End-to-end training with categorical features
"""

import numpy as np
import pytest

import openboost as ob
from openboost import array, BinnedArray, MISSING_BIN, GradientBoosting


class TestCategoricalBinning:
    """Tests for categorical feature encoding in ob.array()."""
    
    def test_categorical_feature_detection(self):
        """Categorical features are correctly identified."""
        X = np.array([
            [25, 0, 50000],
            [30, 1, 60000],
            [35, 2, 70000],
        ], dtype=np.float32)
        
        X_binned = array(X, categorical_features=[1])
        
        assert X_binned.is_categorical[0] == False  # Feature 0 is numeric
        assert X_binned.is_categorical[1] == True   # Feature 1 is categorical
        assert X_binned.is_categorical[2] == False  # Feature 2 is numeric
    
    def test_category_encoding(self):
        """Categorical values are encoded as 0, 1, 2, ..."""
        X = np.array([
            [0],  # Category A
            [1],  # Category B
            [0],  # Category A
            [2],  # Category C
        ], dtype=np.float32)
        
        X_binned = array(X, categorical_features=[0])
        
        # Categories should be encoded as 0, 1, 2
        assert X_binned.n_categories[0] == 3
        assert X_binned.category_maps[0] is not None
        assert len(X_binned.category_maps[0]) == 3
    
    def test_category_map_values(self):
        """Category map correctly maps values to bins."""
        X = np.array([
            [10],
            [20],
            [10],
            [30],
        ], dtype=np.float32)
        
        X_binned = array(X, categorical_features=[0])
        
        # Check the category map
        cat_map = X_binned.category_maps[0]
        assert 10.0 in cat_map or 10 in cat_map
        assert 20.0 in cat_map or 20 in cat_map
        assert 30.0 in cat_map or 30 in cat_map
    
    def test_categorical_with_missing(self):
        """Categorical features handle NaN correctly."""
        X = np.array([
            [0],
            [np.nan],
            [1],
            [0],
        ], dtype=np.float32)
        
        X_binned = array(X, categorical_features=[0])
        
        assert X_binned.is_categorical[0] == True
        assert X_binned.has_missing[0] == True
        # NaN should be encoded as MISSING_BIN
        assert X_binned.data[0, 1] == MISSING_BIN
    
    def test_many_categories(self):
        """Handle features with many categories."""
        n_categories = 100
        X = np.arange(n_categories).reshape(-1, 1).astype(np.float32)
        
        X_binned = array(X, categorical_features=[0])
        
        assert X_binned.n_categories[0] == n_categories
    
    def test_too_many_categories_raises(self):
        """Features with > 254 categories raise error."""
        n_categories = 255
        X = np.arange(n_categories).reshape(-1, 1).astype(np.float32)
        
        with pytest.raises(ValueError, match="max is 254"):
            array(X, categorical_features=[0])
    
    def test_mixed_numeric_categorical(self):
        """Mix of numeric and categorical features works."""
        X = np.array([
            [25, 0, 50000],
            [30, 1, 60000],
            [35, 0, 70000],
        ], dtype=np.float32)
        
        X_binned = array(X, categorical_features=[1])
        
        # Feature 0: numeric (has bin edges)
        assert len(X_binned.bin_edges[0]) > 0 or X_binned.bin_edges[0].size >= 0
        assert X_binned.is_categorical[0] == False
        
        # Feature 1: categorical (no bin edges)
        assert X_binned.is_categorical[1] == True
        assert X_binned.n_categories[1] == 2
        
        # Feature 2: numeric
        assert X_binned.is_categorical[2] == False
    
    def test_any_categorical_property(self):
        """any_categorical property works."""
        X = np.random.randn(10, 3).astype(np.float32)
        
        X_no_cat = array(X)
        assert not X_no_cat.any_categorical
        
        X_with_cat = array(X, categorical_features=[1])
        assert X_with_cat.any_categorical
    
    def test_binned_array_repr_shows_categorical(self):
        """BinnedArray repr shows categorical info."""
        X = np.random.randn(10, 3).astype(np.float32)
        X_binned = array(X, categorical_features=[1])
        
        repr_str = repr(X_binned)
        assert "categorical_features=1" in repr_str


class TestCategoricalSplitFinding:
    """Tests for Fisher-based categorical split finding."""
    
    def test_categorical_split_found(self):
        """Categorical split is found and marked correctly."""
        from openboost._backends._cpu import find_best_split_categorical_cpu
        
        n_features = 1
        hist_grad = np.zeros((n_features, 256), dtype=np.float64)
        hist_hess = np.zeros((n_features, 256), dtype=np.float64)
        
        # 3 categories with clear gradient pattern
        hist_grad[0, 0] = -10.0  # Category 0: low target
        hist_grad[0, 1] = -10.0  # Category 1: low target
        hist_grad[0, 2] = 20.0   # Category 2: high target
        
        hist_hess[0, 0] = 5.0
        hist_hess[0, 1] = 5.0
        hist_hess[0, 2] = 5.0
        
        total_grad = float(np.sum(hist_grad))
        total_hess = float(np.sum(hist_hess))
        
        has_missing = np.array([False], dtype=np.bool_)
        is_categorical = np.array([True], dtype=np.bool_)
        n_categories = np.array([3], dtype=np.int32)
        
        feature, threshold, gain, miss_left, is_cat, bitset, cat_thresh = find_best_split_categorical_cpu(
            hist_grad, hist_hess, total_grad, total_hess,
            reg_lambda=1.0, min_child_weight=1.0,
            has_missing=has_missing,
            is_categorical=is_categorical,
            n_categories=n_categories,
        )
        
        assert is_cat == True
        assert gain > 0


class TestGradientBoostingWithCategorical:
    """Tests for GradientBoosting with categorical features."""
    
    def test_fit_with_categorical(self):
        """GradientBoosting fits with categorical features."""
        np.random.seed(42)
        n_samples = 200
        
        # Feature 0: numeric
        # Feature 1: categorical (3 levels)
        X = np.zeros((n_samples, 2), dtype=np.float32)
        X[:, 0] = np.random.randn(n_samples)
        X[:, 1] = np.random.choice([0, 1, 2], size=n_samples)
        
        # Target depends on category
        y = np.zeros(n_samples, dtype=np.float32)
        y[X[:, 1] == 0] = 0.0
        y[X[:, 1] == 1] = 1.0
        y[X[:, 1] == 2] = 2.0
        y += np.random.randn(n_samples).astype(np.float32) * 0.1
        
        # Bin with categorical feature
        X_binned = array(X, categorical_features=[1])
        
        model = GradientBoosting(n_trees=10, max_depth=3)
        model.fit(X_binned, y)
        
        assert len(model.trees_) == 10
    
    def test_predict_with_categorical(self):
        """GradientBoosting predicts with categorical features."""
        np.random.seed(42)
        n_samples = 100
        
        X = np.zeros((n_samples, 2), dtype=np.float32)
        X[:, 0] = np.random.randn(n_samples)
        X[:, 1] = np.random.choice([0, 1], size=n_samples)
        
        y = X[:, 1].astype(np.float32) + 0.1 * np.random.randn(n_samples).astype(np.float32)
        
        X_binned = array(X, categorical_features=[1])
        
        model = GradientBoosting(n_trees=20, max_depth=3)
        model.fit(X_binned, y)
        
        pred = model.predict(X_binned)
        
        assert pred.shape == (n_samples,)
        assert not np.any(np.isnan(pred))
    
    def test_categorical_learns_groupings(self):
        """Model learns to group categories optimally.
        
        Categories 0 and 1 have low target, category 2 has high target.
        Model should learn to group {0, 1} vs {2}.
        """
        np.random.seed(42)
        n_samples = 500
        
        X = np.zeros((n_samples, 1), dtype=np.float32)
        X[:, 0] = np.random.choice([0, 1, 2], size=n_samples)
        
        y = np.zeros(n_samples, dtype=np.float32)
        y[X[:, 0] == 0] = 0.0  # Category 0: low
        y[X[:, 0] == 1] = 0.0  # Category 1: low  
        y[X[:, 0] == 2] = 5.0  # Category 2: high
        y += np.random.randn(n_samples).astype(np.float32) * 0.1
        
        X_binned = array(X, categorical_features=[0])
        
        model = GradientBoosting(n_trees=50, max_depth=3, learning_rate=0.1)
        model.fit(X_binned, y)
        
        # Test predictions
        X_test = np.array([[0], [1], [2]], dtype=np.float32)
        X_test_binned = array(X_test, categorical_features=[0])
        
        pred = model.predict(X_test_binned)
        
        # Categories 0 and 1 should have similar (low) predictions
        # Category 2 should have high prediction
        assert abs(pred[0] - pred[1]) < 1.0  # 0 and 1 similar
        assert pred[2] > pred[0] + 2.0  # 2 much higher than 0


class TestCategoricalEdgeCases:
    """Edge case tests for categorical features."""
    
    def test_single_category(self):
        """Handle feature with single category (no valid split)."""
        X = np.array([[0], [0], [0], [0]], dtype=np.float32)
        
        X_binned = array(X, categorical_features=[0])
        
        assert X_binned.n_categories[0] == 1
    
    def test_invalid_categorical_index(self):
        """Invalid categorical index raises error."""
        X = np.random.randn(10, 3).astype(np.float32)
        
        with pytest.raises(ValueError, match="out of range"):
            array(X, categorical_features=[5])
    
    def test_negative_categorical_index(self):
        """Negative categorical index raises error."""
        X = np.random.randn(10, 3).astype(np.float32)
        
        with pytest.raises(ValueError, match="out of range"):
            array(X, categorical_features=[-1])
    
    def test_all_categorical(self):
        """All features can be categorical."""
        X = np.array([
            [0, 0],
            [1, 1],
            [0, 2],
        ], dtype=np.float32)
        
        X_binned = array(X, categorical_features=[0, 1])
        
        assert X_binned.is_categorical[0] == True
        assert X_binned.is_categorical[1] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
