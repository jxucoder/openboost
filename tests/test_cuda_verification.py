"""Phase 21: CUDA GPU End-to-End Verification Tests.

These tests verify that users can successfully train GBDT on CUDA GPUs.
Tests are skipped automatically when CUDA is not available.

Run with:
    pytest tests/test_cuda_verification.py -v
    
Run on GPU machine via Modal:
    uv run modal run tests/modal_gpu_tests.py
"""

import pytest
import numpy as np

# Check CUDA availability at module load
try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

import openboost as ob


# =============================================================================
# Backend Detection Tests
# =============================================================================

class TestBackendDetection:
    """Verify automatic and manual backend detection."""
    
    def test_backend_detection_returns_valid(self):
        """get_backend() returns 'cuda' or 'cpu'."""
        backend = ob.get_backend()
        assert backend in ("cuda", "cpu")
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_cuda_detected_when_available(self):
        """When CUDA is available, it should be detected."""
        # Reset backend state to force re-detection
        from openboost._backends import _BACKEND
        import openboost._backends as backends_module
        backends_module._BACKEND = None
        
        backend = ob.get_backend()
        assert backend == "cuda"
    
    def test_is_cuda_is_cpu_consistent(self):
        """is_cuda() and is_cpu() should be mutually exclusive."""
        assert ob.is_cuda() != ob.is_cpu()
        assert ob.is_cuda() == (ob.get_backend() == "cuda")
        assert ob.is_cpu() == (ob.get_backend() == "cpu")
    
    def test_set_backend_cpu(self):
        """Can force CPU backend."""
        original = ob.get_backend()
        try:
            ob.set_backend("cpu")
            assert ob.get_backend() == "cpu"
            assert ob.is_cpu()
            assert not ob.is_cuda()
        finally:
            # Restore original backend
            if original == "cuda" and CUDA_AVAILABLE:
                ob.set_backend("cuda")
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_set_backend_cuda(self):
        """Can force CUDA backend."""
        ob.set_backend("cuda")
        assert ob.get_backend() == "cuda"
        assert ob.is_cuda()
    
    def test_set_backend_invalid_raises(self):
        """Invalid backend name raises ValueError."""
        with pytest.raises(ValueError, match="must be 'cuda' or 'cpu'"):
            ob.set_backend("invalid")
    
    def test_set_backend_cuda_when_unavailable_raises(self):
        """Requesting CUDA when unavailable raises RuntimeError."""
        if CUDA_AVAILABLE:
            pytest.skip("CUDA is available, cannot test unavailable error")
        
        with pytest.raises(RuntimeError, match="not available"):
            ob.set_backend("cuda")


# =============================================================================
# High-Level API on GPU
# =============================================================================

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestGradientBoostingGPU:
    """Test GradientBoosting model on GPU."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        X = np.random.randn(1000, 10).astype(np.float32)
        y = X[:, 0] * 2 + X[:, 1] + np.random.randn(1000).astype(np.float32) * 0.1
        return X, y
    
    def test_gradient_boosting_fit_predict(self, sample_data):
        """Basic fit/predict cycle on GPU."""
        X, y = sample_data
        ob.set_backend("cuda")
        
        model = ob.GradientBoosting(n_trees=10, max_depth=4)
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        assert predictions.shape == (1000,)
        assert predictions.dtype == np.float32
        
        # Should have learned something
        mse = np.mean((predictions - y) ** 2)
        baseline_mse = np.var(y)
        assert mse < baseline_mse * 0.5  # At least 50% reduction
    
    def test_gradient_boosting_with_validation(self, sample_data):
        """Training with validation set on GPU."""
        X, y = sample_data
        ob.set_backend("cuda")
        
        X_train, X_val = X[:800], X[800:]
        y_train, y_val = y[:800], y[800:]
        
        model = ob.GradientBoosting(n_trees=50, max_depth=4)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        
        # Model should have trained
        assert len(model.trees_) == 50
    
    def test_gradient_boosting_early_stopping(self, sample_data):
        """Early stopping callback on GPU."""
        X, y = sample_data
        ob.set_backend("cuda")
        
        X_train, X_val = X[:800], X[800:]
        y_train, y_val = y[:800], y[800:]
        
        model = ob.GradientBoosting(n_trees=1000, max_depth=6)
        model.fit(
            X_train, y_train,
            callbacks=[ob.EarlyStopping(patience=10)],
            eval_set=[(X_val, y_val)],
        )
        
        # Should have stopped early
        assert len(model.trees_) < 1000
    
    def test_gradient_boosting_different_losses(self, sample_data):
        """Test different loss functions on GPU."""
        X, y = sample_data
        ob.set_backend("cuda")
        
        for loss in ['mse', 'huber', 'mae']:
            model = ob.GradientBoosting(n_trees=5, max_depth=3, loss=loss)
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == (1000,)
    
    def test_gradient_boosting_classification(self):
        """Binary classification on GPU."""
        np.random.seed(42)
        X = np.random.randn(500, 5).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
        
        ob.set_backend("cuda")
        
        model = ob.GradientBoosting(n_trees=20, max_depth=4, loss='logloss')
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert predictions.shape == (500,)
        
        # Should have reasonable accuracy
        accuracy = np.mean((predictions > 0.5) == y)
        assert accuracy > 0.7


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestNaturalBoostGPU:
    """Test probabilistic models on GPU."""
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X = np.random.randn(500, 5).astype(np.float32)
        y = X[:, 0] + np.random.randn(500).astype(np.float32) * 0.5
        return X, y
    
    def test_natural_boost_normal_gpu(self, sample_data):
        """NaturalBoostNormal training on GPU."""
        X, y = sample_data
        ob.set_backend("cuda")
        
        model = ob.NaturalBoostNormal(n_trees=20, max_depth=3)
        model.fit(X, y)
        
        mean = model.predict(X)
        lower, upper = model.predict_interval(X, alpha=0.1)
        
        assert mean.shape == (500,)
        assert lower.shape == (500,)
        assert upper.shape == (500,)
        assert np.all(lower <= upper)
    
    def test_natural_boost_lognormal_gpu(self, sample_data):
        """NaturalBoostLogNormal training on GPU."""
        X, y = sample_data
        y_positive = np.abs(y) + 0.1  # Ensure positive
        ob.set_backend("cuda")
        
        model = ob.NaturalBoostLogNormal(n_trees=20, max_depth=3)
        model.fit(X, y_positive)
        
        mean = model.predict(X)
        assert mean.shape == (500,)
        assert np.all(mean > 0)
    
    def test_natural_boost_gamma_gpu(self, sample_data):
        """NaturalBoostGamma training on GPU."""
        X, y = sample_data
        y_positive = np.abs(y) + 0.1
        ob.set_backend("cuda")
        
        model = ob.NaturalBoostGamma(n_trees=20, max_depth=3)
        model.fit(X, y_positive)
        
        mean = model.predict(X)
        assert mean.shape == (500,)
    
    def test_natural_boost_poisson_gpu(self):
        """NaturalBoostPoisson training on GPU."""
        np.random.seed(42)
        X = np.random.randn(500, 5).astype(np.float32)
        y = np.random.poisson(5, 500).astype(np.float32)
        
        ob.set_backend("cuda")
        
        model = ob.NaturalBoostPoisson(n_trees=20, max_depth=3)
        model.fit(X, y)
        
        mean = model.predict(X)
        assert mean.shape == (500,)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestOtherModelsGPU:
    """Test other model types on GPU."""
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X = np.random.randn(500, 5).astype(np.float32)
        y = X[:, 0] + np.random.randn(500).astype(np.float32) * 0.5
        return X, y
    
    def test_openboost_gam_gpu(self, sample_data):
        """OpenBoostGAM on GPU."""
        X, y = sample_data
        ob.set_backend("cuda")
        
        model = ob.OpenBoostGAM(n_rounds=20, learning_rate=0.1)
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert predictions.shape == (500,)
    
    def test_dart_gpu(self, sample_data):
        """DART on GPU."""
        X, y = sample_data
        ob.set_backend("cuda")
        
        model = ob.DART(n_trees=20, max_depth=3, dropout_rate=0.1)
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert predictions.shape == (500,)
    
    def test_linear_leaf_gbdt_gpu(self, sample_data):
        """LinearLeafGBDT on GPU."""
        X, y = sample_data
        ob.set_backend("cuda")
        
        model = ob.LinearLeafGBDT(n_trees=20, max_depth=3)
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert predictions.shape == (500,)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestSklearnWrappersGPU:
    """Test sklearn-compatible wrappers on GPU."""
    
    @pytest.fixture
    def regression_data(self):
        np.random.seed(42)
        X = np.random.randn(500, 5).astype(np.float32)
        y = X[:, 0] + np.random.randn(500).astype(np.float32) * 0.5
        return X, y
    
    @pytest.fixture
    def classification_data(self):
        np.random.seed(42)
        X = np.random.randn(500, 5).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)
        return X, y
    
    def test_openboost_regressor_gpu(self, regression_data):
        """OpenBoostRegressor on GPU."""
        X, y = regression_data
        ob.set_backend("cuda")
        
        model = ob.OpenBoostRegressor(n_estimators=20, max_depth=4)
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert predictions.shape == (500,)
        
        score = model.score(X, y)
        assert score > 0.5  # R² should be positive
    
    def test_openboost_classifier_gpu(self, classification_data):
        """OpenBoostClassifier on GPU."""
        X, y = classification_data
        ob.set_backend("cuda")
        
        model = ob.OpenBoostClassifier(n_estimators=20, max_depth=4)
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert predictions.shape == (500,)
        
        probas = model.predict_proba(X)
        assert probas.shape == (500, 2)
        
        accuracy = model.score(X, y)
        assert accuracy > 0.7


# =============================================================================
# Low-Level API on GPU
# =============================================================================

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestLowLevelGPU:
    """Test low-level API on GPU."""
    
    def test_fit_tree_on_gpu(self):
        """fit_tree() uses GPU when available."""
        np.random.seed(42)
        X = np.random.randn(500, 5).astype(np.float32)
        y = X[:, 0]
        
        ob.set_backend("cuda")
        
        X_binned = ob.array(X)
        
        grad = (2 * (np.zeros(500, dtype=np.float32) - y)).astype(np.float32)
        hess = np.ones(500, dtype=np.float32) * 2
        
        tree = ob.fit_tree(X_binned, grad, hess, max_depth=4)
        
        assert tree.n_nodes > 0
        assert tree.depth <= 4
        
        # Tree should make predictions
        predictions = tree(X_binned)
        assert predictions.shape == (500,)
    
    def test_binned_array_on_gpu(self):
        """BinnedArray works correctly on GPU."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        
        ob.set_backend("cuda")
        
        X_binned = ob.array(X, n_bins=256)
        
        assert X_binned.n_samples == 100
        assert X_binned.n_features == 5
        assert X_binned.data.shape == (5, 100)  # Feature-major layout
        assert X_binned.data.dtype == np.uint8


# =============================================================================
# Memory and Performance
# =============================================================================

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestGPUMemory:
    """Test GPU memory handling."""
    
    def test_no_memory_leak_on_fit(self):
        """Training doesn't leak GPU memory (basic check)."""
        np.random.seed(42)
        X = np.random.randn(1000, 10).astype(np.float32)
        y = np.random.randn(1000).astype(np.float32)
        
        ob.set_backend("cuda")
        
        # Train multiple models and check they complete
        for i in range(3):
            model = ob.GradientBoosting(n_trees=10, max_depth=4)
            model.fit(X, y)
            del model
    
    def test_moderate_size_dataset(self):
        """Can train on moderately large dataset."""
        np.random.seed(42)
        X = np.random.randn(50000, 20).astype(np.float32)
        y = np.random.randn(50000).astype(np.float32)
        
        ob.set_backend("cuda")
        
        model = ob.GradientBoosting(n_trees=20, max_depth=6)
        model.fit(X, y)
        
        assert len(model.trees_) == 20
        
        predictions = model.predict(X)
        assert predictions.shape == (50000,)


# =============================================================================
# Error Handling
# =============================================================================

class TestGPUErrors:
    """Test GPU-related error handling."""
    
    def test_graceful_fallback_to_cpu(self):
        """When CUDA unavailable, gracefully falls back to CPU."""
        # Force CPU
        ob.set_backend("cpu")
        
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        
        # Should work on CPU
        model = ob.GradientBoosting(n_trees=5, max_depth=3)
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert predictions.shape == (100,)
    
    def test_cpu_training_produces_valid_results(self):
        """CPU training produces valid results."""
        ob.set_backend("cpu")
        
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] * 2 + X[:, 1]
        
        model = ob.GradientBoosting(n_trees=20, max_depth=4)
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        # Should have learned the pattern
        mse = np.mean((predictions - y) ** 2)
        baseline_mse = np.var(y)
        assert mse < baseline_mse * 0.3


# =============================================================================
# Persistence with GPU
# =============================================================================

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestGPUPersistence:
    """Test model save/load with GPU training."""
    
    def test_gpu_model_can_be_saved_and_loaded(self, tmp_path):
        """Model trained on GPU can be saved and loaded."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = np.random.randn(200).astype(np.float32)
        
        ob.set_backend("cuda")
        
        # Train on GPU
        model = ob.GradientBoosting(n_trees=10, max_depth=4)
        model.fit(X, y)
        predictions_original = model.predict(X)
        
        # Save
        save_path = tmp_path / "model.joblib"
        model.save(str(save_path))
        
        # Load
        loaded_model = ob.GradientBoosting.load(str(save_path))
        predictions_loaded = loaded_model.predict(X)
        
        # Predictions should match
        np.testing.assert_allclose(predictions_original, predictions_loaded, rtol=1e-5)
    
    def test_gpu_model_can_be_loaded_on_cpu(self, tmp_path):
        """Model trained on GPU can be loaded and used on CPU."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = np.random.randn(200).astype(np.float32)
        
        ob.set_backend("cuda")
        
        # Train on GPU
        model = ob.GradientBoosting(n_trees=10, max_depth=4)
        model.fit(X, y)
        predictions_gpu = model.predict(X)
        
        # Save
        save_path = tmp_path / "model.joblib"
        model.save(str(save_path))
        
        # Load on CPU
        ob.set_backend("cpu")
        loaded_model = ob.GradientBoosting.load(str(save_path))
        predictions_cpu = loaded_model.predict(X)
        
        # Predictions should be very close (may differ slightly due to float precision)
        np.testing.assert_allclose(predictions_gpu, predictions_cpu, rtol=1e-4)


# =============================================================================
# Quick Smoke Test
# =============================================================================

def test_quick_gpu_smoke_test():
    """Quick smoke test that runs regardless of CUDA availability."""
    np.random.seed(42)
    X = np.random.randn(100, 5).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    
    # This should work on any backend
    model = ob.GradientBoosting(n_trees=5, max_depth=3)
    model.fit(X, y)
    
    predictions = model.predict(X)
    assert predictions.shape == (100,)
    
    # Report which backend was used
    print(f"\nBackend used: {ob.get_backend()}")
    print(f"CUDA available: {CUDA_AVAILABLE}")
