"""Tests for Phase 17: Large-Scale Training features.

Tests GOSS (Gradient-based One-Side Sampling) and mini-batch support.
"""

import numpy as np
import pytest

import openboost as ob
from openboost._sampling import (
    goss_sample,
    random_sample,
    apply_sampling,
    MiniBatchIterator,
    GOSSConfig,
    MiniBatchConfig,
    SamplingResult,
    SamplingStrategy,
)


# =============================================================================
# GOSS Sampling Tests
# =============================================================================

class TestGOSS:
    """Tests for Gradient-based One-Side Sampling."""
    
    def test_goss_sample_basic(self):
        """Test basic GOSS sampling."""
        n_samples = 10000
        grad = np.random.randn(n_samples).astype(np.float32)
        hess = np.ones(n_samples, dtype=np.float32)
        
        result = goss_sample(grad, hess, top_rate=0.2, other_rate=0.1)
        
        # Check result type
        assert isinstance(result, SamplingResult)
        
        # Check sample count
        expected_n_top = int(n_samples * 0.2)
        expected_n_other = int((n_samples - expected_n_top) * 0.1)
        expected_total = expected_n_top + expected_n_other
        
        # Allow some tolerance due to edge cases
        assert abs(result.n_selected - expected_total) <= 2
        
        # Check sample rate
        assert 0.2 < result.sample_rate < 0.35  # ~28% expected
        
        # Check indices are valid
        assert result.indices.min() >= 0
        assert result.indices.max() < n_samples
        
        # Check no duplicates
        assert len(set(result.indices)) == len(result.indices)
    
    def test_goss_keeps_high_gradient_samples(self):
        """GOSS should preferentially keep high-gradient samples."""
        n_samples = 10000
        
        # Create gradient with clear high/low groups
        grad = np.zeros(n_samples, dtype=np.float32)
        grad[:1000] = 100.0  # High gradient samples (indices 0-999)
        grad[1000:] = 0.01   # Low gradient samples
        
        result = goss_sample(grad, top_rate=0.2, other_rate=0.1)
        
        # All high-gradient samples should be in the result
        # (top 20% = 2000 samples, but we only have 1000 high-gradient)
        high_grad_in_result = np.sum(result.indices < 1000)
        assert high_grad_in_result >= 900  # Most high-gradient samples should be included
    
    def test_goss_weights(self):
        """GOSS weights should upweight low-gradient samples."""
        n_samples = 10000
        grad = np.random.randn(n_samples).astype(np.float32)
        
        result = goss_sample(grad, top_rate=0.2, other_rate=0.1)
        
        n_top = int(n_samples * 0.2)
        
        # Top samples should have weight 1.0
        assert np.allclose(result.weights[:n_top], 1.0)
        
        # Other samples should have upweight
        expected_upweight = (1 - 0.2) / 0.1
        if len(result.weights) > n_top:
            assert np.allclose(result.weights[n_top:], expected_upweight)
    
    def test_goss_reproducibility(self):
        """GOSS with same seed should give same results."""
        n_samples = 10000
        grad = np.random.randn(n_samples).astype(np.float32)
        
        result1 = goss_sample(grad, seed=42)
        result2 = goss_sample(grad, seed=42)
        
        np.testing.assert_array_equal(result1.indices, result2.indices)
        np.testing.assert_array_equal(result1.weights, result2.weights)
    
    def test_goss_different_seeds(self):
        """GOSS with different seeds should give different results."""
        n_samples = 10000
        grad = np.random.randn(n_samples).astype(np.float32)
        
        result1 = goss_sample(grad, seed=42)
        result2 = goss_sample(grad, seed=123)
        
        # Top samples might be same (deterministic), but other samples should differ
        # Just check they're not identical
        assert not np.array_equal(result1.indices, result2.indices)
    
    def test_goss_multidim_gradient(self):
        """GOSS should handle multi-dimensional gradients (for distributional GBDT)."""
        n_samples = 10000
        n_params = 2
        grad = np.random.randn(n_samples, n_params).astype(np.float32)
        
        result = goss_sample(grad, top_rate=0.2, other_rate=0.1)
        
        assert result.n_selected > 0
        assert result.indices.max() < n_samples


class TestRandomSampling:
    """Tests for random subsampling."""
    
    def test_random_sample_basic(self):
        """Test basic random sampling."""
        n_samples = 10000
        sample_rate = 0.3
        
        result = random_sample(n_samples, sample_rate)
        
        expected_n = int(n_samples * sample_rate)
        assert result.n_selected == expected_n
        assert result.n_original == n_samples
        assert len(result.indices) == expected_n
        assert len(result.weights) == expected_n
        
        # All weights should be 1.0
        assert np.allclose(result.weights, 1.0)
    
    def test_random_sample_no_duplicates(self):
        """Random sampling should not have duplicate indices."""
        n_samples = 10000
        sample_rate = 0.5
        
        result = random_sample(n_samples, sample_rate)
        
        assert len(set(result.indices)) == len(result.indices)
    
    def test_random_sample_full(self):
        """Sample rate >= 1.0 should return all samples."""
        n_samples = 10000
        
        result = random_sample(n_samples, sample_rate=1.0)
        
        assert result.n_selected == n_samples
        np.testing.assert_array_equal(result.indices, np.arange(n_samples))
    
    def test_random_sample_reproducibility(self):
        """Random sampling with same seed should give same results."""
        n_samples = 10000
        
        result1 = random_sample(n_samples, sample_rate=0.3, seed=42)
        result2 = random_sample(n_samples, sample_rate=0.3, seed=42)
        
        np.testing.assert_array_equal(result1.indices, result2.indices)


class TestApplySampling:
    """Tests for apply_sampling convenience function."""
    
    def test_apply_sampling_none(self):
        """apply_sampling with 'none' should return all samples."""
        n_samples = 1000
        grad = np.random.randn(n_samples).astype(np.float32)
        hess = np.ones(n_samples, dtype=np.float32)
        
        indices, weights, grad_s, hess_s = apply_sampling(
            grad, hess, strategy='none'
        )
        
        assert len(indices) == n_samples
        np.testing.assert_array_equal(grad_s, grad)
        np.testing.assert_array_equal(hess_s, hess)
    
    def test_apply_sampling_goss(self):
        """apply_sampling with 'goss' should apply GOSS sampling."""
        n_samples = 10000
        grad = np.random.randn(n_samples).astype(np.float32)
        hess = np.ones(n_samples, dtype=np.float32)
        
        indices, weights, grad_s, hess_s = apply_sampling(
            grad, hess, strategy='goss', top_rate=0.2, other_rate=0.1
        )
        
        # Should have fewer samples
        assert len(indices) < n_samples
        
        # Gradients should be weighted
        assert len(grad_s) == len(indices)
        assert len(hess_s) == len(indices)
    
    def test_apply_sampling_random(self):
        """apply_sampling with 'random' should apply random sampling."""
        n_samples = 10000
        grad = np.random.randn(n_samples).astype(np.float32)
        hess = np.ones(n_samples, dtype=np.float32)
        
        indices, weights, grad_s, hess_s = apply_sampling(
            grad, hess, strategy='random', sample_rate=0.3
        )
        
        expected_n = int(n_samples * 0.3)
        assert len(indices) == expected_n


# =============================================================================
# Mini-Batch Tests
# =============================================================================

class TestMiniBatch:
    """Tests for mini-batch training support."""
    
    def test_minibatch_iterator_basic(self):
        """Test basic mini-batch iteration."""
        n_samples = 10000
        batch_size = 1000
        
        iterator = MiniBatchIterator(n_samples, batch_size)
        
        assert iterator.n_batches == 10
        
        batches = list(iterator)
        assert len(batches) == 10
        
        # Check all indices are covered
        all_indices = np.concatenate(batches)
        np.testing.assert_array_equal(np.sort(all_indices), np.arange(n_samples))
    
    def test_minibatch_iterator_uneven(self):
        """Test mini-batch with uneven split."""
        n_samples = 10000
        batch_size = 3000
        
        iterator = MiniBatchIterator(n_samples, batch_size)
        
        assert iterator.n_batches == 4  # ceil(10000/3000)
        
        batches = list(iterator)
        assert len(batches) == 4
        
        # Last batch should be smaller
        assert len(batches[-1]) == 10000 - 3 * 3000  # 1000
        
        # All indices covered
        all_indices = np.concatenate(batches)
        np.testing.assert_array_equal(np.sort(all_indices), np.arange(n_samples))
    
    def test_minibatch_iterator_shuffle(self):
        """Test mini-batch iteration with shuffling."""
        n_samples = 10000
        batch_size = 1000
        
        iterator = MiniBatchIterator(n_samples, batch_size, shuffle=True, seed=42)
        
        batches = list(iterator)
        
        # All indices still covered (just in different order)
        all_indices = np.concatenate(batches)
        np.testing.assert_array_equal(np.sort(all_indices), np.arange(n_samples))
        
        # First batch should not be 0-999 (shuffled)
        assert not np.array_equal(batches[0], np.arange(1000))
    
    def test_minibatch_config(self):
        """Test MiniBatchConfig validation."""
        config = MiniBatchConfig(batch_size=100_000, shuffle=True, seed=42)
        assert config.batch_size == 100_000
        
        with pytest.raises(ValueError):
            MiniBatchConfig(batch_size=0)
        
        with pytest.raises(ValueError):
            MiniBatchConfig(batch_size=-100)


class TestGOSSConfig:
    """Tests for GOSS configuration."""
    
    def test_goss_config_basic(self):
        """Test basic GOSS config."""
        config = GOSSConfig(top_rate=0.2, other_rate=0.1)
        
        assert config.top_rate == 0.2
        assert config.other_rate == 0.1
        
        # Effective rate = 0.2 + 0.1 * 0.8 = 0.28
        assert abs(config.effective_sample_rate - 0.28) < 0.01
    
    def test_goss_config_validation(self):
        """Test GOSS config validation."""
        with pytest.raises(ValueError):
            GOSSConfig(top_rate=0.0)
        
        with pytest.raises(ValueError):
            GOSSConfig(top_rate=1.0)
        
        with pytest.raises(ValueError):
            GOSSConfig(other_rate=0.0)


# =============================================================================
# Integration Tests with GradientBoosting
# =============================================================================

class TestGOSSIntegration:
    """Integration tests for GOSS with GradientBoosting."""
    
    def test_gradient_boosting_with_goss(self):
        """Test GradientBoosting with GOSS sampling."""
        np.random.seed(42)
        
        # Create synthetic dataset
        n_samples = 10000
        n_features = 10
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples) * 0.1).astype(np.float32)
        
        # Train with GOSS
        model = ob.GradientBoosting(
            n_trees=20,
            max_depth=4,
            learning_rate=0.1,
            subsample_strategy='goss',
            goss_top_rate=0.2,
            goss_other_rate=0.1,
        )
        model.fit(X, y)
        
        # Should have trees
        assert len(model.trees_) == 20
        
        # Should make reasonable predictions
        pred = model.predict(X)
        mse = np.mean((pred - y) ** 2)
        assert mse < 0.5  # Reasonable fit
    
    def test_gradient_boosting_goss_vs_standard(self):
        """Compare GOSS training with standard training."""
        np.random.seed(42)
        
        # Create synthetic dataset
        n_samples = 10000
        n_features = 10
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples) * 0.1).astype(np.float32)
        
        # Split data
        X_train, X_test = X[:8000], X[8000:]
        y_train, y_test = y[:8000], y[8000:]
        
        # Standard training
        model_std = ob.GradientBoosting(
            n_trees=50,
            max_depth=4,
            learning_rate=0.1,
        )
        model_std.fit(X_train, y_train)
        pred_std = model_std.predict(X_test)
        mse_std = np.mean((pred_std - y_test) ** 2)
        
        # GOSS training
        model_goss = ob.GradientBoosting(
            n_trees=50,
            max_depth=4,
            learning_rate=0.1,
            subsample_strategy='goss',
            goss_top_rate=0.2,
            goss_other_rate=0.1,
        )
        model_goss.fit(X_train, y_train)
        pred_goss = model_goss.predict(X_test)
        mse_goss = np.mean((pred_goss - y_test) ** 2)
        
        # GOSS should achieve similar accuracy (within 20% relative)
        assert mse_goss < mse_std * 1.2
    
    def test_sklearn_wrapper_with_goss(self):
        """Test sklearn wrapper with GOSS."""
        np.random.seed(42)
        
        n_samples = 5000
        n_features = 10
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples) * 0.1).astype(np.float32)
        
        # Use sklearn wrapper with GOSS
        model = ob.OpenBoostRegressor(
            n_estimators=20,
            max_depth=4,
            subsample_strategy='goss',
            goss_top_rate=0.2,
            goss_other_rate=0.1,
        )
        model.fit(X, y)
        
        # Should work with sklearn interface
        pred = model.predict(X)
        score = model.score(X, y)  # R² score
        
        assert score > 0.5  # Reasonable fit
    
    def test_multiclass_with_goss(self):
        """Test multi-class classification with GOSS."""
        np.random.seed(42)
        
        n_samples = 5000
        n_features = 10
        n_classes = 3
        
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, n_classes, size=n_samples)
        
        model = ob.MultiClassGradientBoosting(
            n_classes=n_classes,
            n_trees=20,
            max_depth=4,
            subsample_strategy='goss',
            goss_top_rate=0.2,
            goss_other_rate=0.1,
        )
        model.fit(X, y)
        
        # Should have trees
        assert len(model.trees_) == 20
        
        # Should make predictions
        pred = model.predict(X)
        accuracy = np.mean(pred == y)
        assert accuracy > 0.4  # Better than random (1/3)


# =============================================================================
# Memory Efficiency Tests (placeholder for large-scale testing)
# =============================================================================

class TestMemoryEfficiency:
    """Tests for memory-efficient training features."""
    
    def test_memmap_creation(self):
        """Test memory-mapped array creation."""
        import tempfile
        import os
        
        n_samples = 10000
        n_features = 10
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'binned.npy')
            
            # Create memmap
            mmap = ob.create_memmap_binned(path, X)
            
            # Check shape (n_features, n_samples after transpose)
            assert mmap.shape == (n_features, n_samples)
            
            # Load it back
            mmap_loaded = ob.load_memmap_binned(path, n_features, n_samples)
            assert mmap_loaded.shape == (n_features, n_samples)
            
            # Values should match
            np.testing.assert_array_equal(mmap, mmap_loaded)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
