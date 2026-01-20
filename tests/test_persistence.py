"""Tests for model persistence (Phase 20.1).

Tests save/load functionality for all model types.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def regression_data():
    """Generate simple regression data."""
    np.random.seed(42)
    X = np.random.randn(500, 10).astype(np.float32)
    y = (X[:, 0] * 2 + X[:, 1] + np.random.randn(500) * 0.1).astype(np.float32)
    return X, y


@pytest.fixture
def binary_data():
    """Generate simple binary classification data."""
    np.random.seed(42)
    X = np.random.randn(500, 10).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    return X, y


@pytest.fixture
def multiclass_data():
    """Generate simple multiclass classification data."""
    np.random.seed(42)
    X = np.random.randn(500, 10).astype(np.float32)
    y = np.argmax(X[:, :5], axis=1).astype(np.int32)
    return X, y


# =============================================================================
# GradientBoosting Tests
# =============================================================================


class TestGradientBoostingPersistence:
    """Test save/load for GradientBoosting."""

    def test_save_load_basic(self, regression_data, tmp_path):
        """Test basic save/load functionality."""
        import openboost as ob

        X, y = regression_data
        model = ob.GradientBoosting(n_trees=10, max_depth=3, learning_rate=0.1)
        model.fit(X[:400], y[:400])

        # Get predictions before save
        pred_before = model.predict(X[400:])

        # Save
        save_path = tmp_path / "model.joblib"
        model.save(save_path)

        # Load
        loaded = ob.GradientBoosting.load(save_path)

        # Get predictions after load
        pred_after = loaded.predict(X[400:])

        # Predictions should match
        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5)

    def test_save_load_with_different_losses(self, regression_data, tmp_path):
        """Test save/load with various loss functions."""
        import openboost as ob

        X, y = regression_data

        for loss in ["mse", "mae", "huber"]:
            model = ob.GradientBoosting(n_trees=5, max_depth=3, loss=loss)
            model.fit(X[:400], y[:400])

            save_path = tmp_path / f"model_{loss}.joblib"
            model.save(save_path)

            loaded = ob.GradientBoosting.load(save_path)
            pred_before = model.predict(X[400:])
            pred_after = loaded.predict(X[400:])

            np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5)

    def test_pickle_compatibility(self, regression_data, tmp_path):
        """Test that models are pickle-compatible."""
        import openboost as ob
        import pickle

        X, y = regression_data
        model = ob.GradientBoosting(n_trees=10, max_depth=3)
        model.fit(X[:400], y[:400])

        # Pickle
        save_path = tmp_path / "model.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(model, f)

        # Unpickle
        with open(save_path, "rb") as f:
            loaded = pickle.load(f)

        pred_before = model.predict(X[400:])
        pred_after = loaded.predict(X[400:])

        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5)

    def test_joblib_compatibility(self, regression_data, tmp_path):
        """Test that models work with joblib directly."""
        import openboost as ob
        import joblib

        X, y = regression_data
        model = ob.GradientBoosting(n_trees=10, max_depth=3)
        model.fit(X[:400], y[:400])

        # Save with joblib directly
        save_path = tmp_path / "model_joblib.joblib"
        joblib.dump(model, save_path)

        # Load with joblib directly
        loaded = joblib.load(save_path)

        pred_before = model.predict(X[400:])
        pred_after = loaded.predict(X[400:])

        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5)


# =============================================================================
# MultiClassGradientBoosting Tests
# =============================================================================


class TestMultiClassGradientBoostingPersistence:
    """Test save/load for MultiClassGradientBoosting."""

    def test_save_load_basic(self, multiclass_data, tmp_path):
        """Test basic save/load functionality."""
        import openboost as ob

        X, y = multiclass_data
        n_classes = len(np.unique(y))

        model = ob.MultiClassGradientBoosting(
            n_classes=n_classes, n_trees=10, max_depth=3
        )
        model.fit(X[:400], y[:400])

        # Get predictions before save
        pred_before = model.predict(X[400:])
        proba_before = model.predict_proba(X[400:])

        # Save
        save_path = tmp_path / "model.joblib"
        model.save(save_path)

        # Load
        loaded = ob.MultiClassGradientBoosting.load(save_path)

        # Get predictions after load
        pred_after = loaded.predict(X[400:])
        proba_after = loaded.predict_proba(X[400:])

        # Predictions should match
        np.testing.assert_array_equal(pred_before, pred_after)
        np.testing.assert_allclose(proba_before, proba_after, rtol=1e-5)


# =============================================================================
# DART Tests
# =============================================================================


class TestDARTPersistence:
    """Test save/load for DART."""

    def test_save_load_basic(self, regression_data, tmp_path):
        """Test basic save/load functionality."""
        import openboost as ob

        X, y = regression_data
        model = ob.DART(n_trees=10, max_depth=3, dropout_rate=0.1, seed=42)
        model.fit(X[:400], y[:400])

        # Get predictions before save
        pred_before = model.predict(X[400:])

        # Save
        save_path = tmp_path / "model.joblib"
        model.save(save_path)

        # Load
        loaded = ob.DART.load(save_path)

        # Get predictions after load
        pred_after = loaded.predict(X[400:])

        # Predictions should match
        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5)


# =============================================================================
# OpenBoostGAM Tests
# =============================================================================


class TestOpenBoostGAMPersistence:
    """Test save/load for OpenBoostGAM."""

    def test_save_load_basic(self, regression_data, tmp_path):
        """Test basic save/load functionality."""
        import openboost as ob

        X, y = regression_data
        model = ob.OpenBoostGAM(n_rounds=50, learning_rate=0.1)
        model.fit(X[:400], y[:400])

        # Get predictions before save
        pred_before = model.predict(X[400:])

        # Save
        save_path = tmp_path / "model.joblib"
        model.save(save_path)

        # Load
        loaded = ob.OpenBoostGAM.load(save_path)

        # Get predictions after load
        pred_after = loaded.predict(X[400:])

        # Predictions should match
        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5)


# =============================================================================
# NaturalBoost (Distributional) Tests
# =============================================================================


class TestNaturalBoostPersistence:
    """Test save/load for NaturalBoost/DistributionalGBDT."""

    def test_save_load_normal(self, regression_data, tmp_path):
        """Test save/load for NaturalBoost with Normal distribution."""
        import openboost as ob

        X, y = regression_data
        # Use NaturalBoostNormal factory function which returns NaturalBoost
        model = ob.NaturalBoostNormal(n_trees=10, max_depth=3)
        model.fit(X[:400], y[:400])

        # Get predictions before save
        pred_before = model.predict(X[400:])
        interval_before = model.predict_interval(X[400:], alpha=0.1)

        # Save
        save_path = tmp_path / "model.joblib"
        model.save(save_path)

        # Load using NaturalBoost.load() (the actual class)
        loaded = ob.NaturalBoost.load(save_path)

        # Get predictions after load
        pred_after = loaded.predict(X[400:])
        interval_after = loaded.predict_interval(X[400:], alpha=0.1)

        # Predictions should match
        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5)
        np.testing.assert_allclose(interval_before[0], interval_after[0], rtol=1e-5)
        np.testing.assert_allclose(interval_before[1], interval_after[1], rtol=1e-5)

    def test_save_load_poisson(self, tmp_path):
        """Test save/load for NaturalBoost with Poisson distribution."""
        import openboost as ob

        np.random.seed(42)
        X = np.random.randn(500, 10).astype(np.float32)
        y = np.random.poisson(5, 500).astype(np.float32)

        model = ob.NaturalBoostPoisson(n_trees=10, max_depth=3)
        model.fit(X[:400], y[:400])

        pred_before = model.predict(X[400:])

        save_path = tmp_path / "model.joblib"
        model.save(save_path)

        # Load using NaturalBoost.load() (the actual class)
        loaded = ob.NaturalBoost.load(save_path)
        pred_after = loaded.predict(X[400:])

        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5)


# =============================================================================
# LinearLeafGBDT Tests
# =============================================================================


class TestLinearLeafGBDTPersistence:
    """Test save/load for LinearLeafGBDT."""

    def test_save_load_basic(self, regression_data, tmp_path):
        """Test basic save/load functionality."""
        import openboost as ob

        X, y = regression_data
        model = ob.LinearLeafGBDT(n_trees=10, max_depth=3, min_samples_leaf=10)
        model.fit(X[:400], y[:400])

        # Get predictions before save
        pred_before = model.predict(X[400:])

        # Save
        save_path = tmp_path / "model.joblib"
        model.save(save_path)

        # Load
        loaded = ob.LinearLeafGBDT.load(save_path)

        # Get predictions after load
        pred_after = loaded.predict(X[400:])

        # Predictions should match
        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5)


# =============================================================================
# sklearn Wrapper Tests
# =============================================================================


class TestSklearnWrappersPersistence:
    """Test that sklearn wrappers are picklable (through booster_)."""

    def test_regressor_pickle(self, regression_data, tmp_path):
        """Test OpenBoostRegressor pickle."""
        import openboost as ob
        import pickle

        X, y = regression_data
        model = ob.OpenBoostRegressor(n_estimators=10, max_depth=3)
        model.fit(X[:400], y[:400])

        pred_before = model.predict(X[400:])

        # Pickle
        save_path = tmp_path / "regressor.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(model, f)

        with open(save_path, "rb") as f:
            loaded = pickle.load(f)

        pred_after = loaded.predict(X[400:])
        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5)

    def test_classifier_pickle(self, binary_data, tmp_path):
        """Test OpenBoostClassifier pickle."""
        import openboost as ob
        import pickle

        X, y = binary_data
        model = ob.OpenBoostClassifier(n_estimators=10, max_depth=3)
        model.fit(X[:400], y[:400])

        pred_before = model.predict(X[400:])
        proba_before = model.predict_proba(X[400:])

        # Pickle
        save_path = tmp_path / "classifier.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(model, f)

        with open(save_path, "rb") as f:
            loaded = pickle.load(f)

        pred_after = loaded.predict(X[400:])
        proba_after = loaded.predict_proba(X[400:])

        np.testing.assert_array_equal(pred_before, pred_after)
        np.testing.assert_allclose(proba_before, proba_after, rtol=1e-5)


# =============================================================================
# Edge Cases
# =============================================================================


class TestPersistenceEdgeCases:
    """Test edge cases for persistence."""

    def test_load_wrong_class_raises(self, regression_data, tmp_path):
        """Test that loading with wrong class raises error."""
        import openboost as ob

        X, y = regression_data
        model = ob.GradientBoosting(n_trees=5, max_depth=3)
        model.fit(X[:400], y[:400])

        save_path = tmp_path / "model.joblib"
        model.save(save_path)

        # Try to load with wrong class
        with pytest.raises(ValueError, match="saved as GradientBoosting"):
            ob.DART.load(save_path)

    def test_file_extensions(self, regression_data, tmp_path):
        """Test various file extensions work."""
        import openboost as ob

        X, y = regression_data
        model = ob.GradientBoosting(n_trees=5, max_depth=3)
        model.fit(X[:400], y[:400])

        pred_expected = model.predict(X[400:])

        # Test various extensions
        for ext in [".joblib", ".pkl", ".pickle"]:
            save_path = tmp_path / f"model{ext}"
            model.save(save_path)
            loaded = ob.GradientBoosting.load(save_path)
            pred_loaded = loaded.predict(X[400:])
            np.testing.assert_allclose(pred_expected, pred_loaded, rtol=1e-5)

    def test_path_object_support(self, regression_data, tmp_path):
        """Test that Path objects work."""
        import openboost as ob

        X, y = regression_data
        model = ob.GradientBoosting(n_trees=5, max_depth=3)
        model.fit(X[:400], y[:400])

        # Use Path object
        save_path = Path(tmp_path) / "model.joblib"
        model.save(save_path)

        loaded = ob.GradientBoosting.load(save_path)
        pred_before = model.predict(X[400:])
        pred_after = loaded.predict(X[400:])

        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5)

    def test_model_params_preserved(self, regression_data, tmp_path):
        """Test that model parameters are preserved after load."""
        import openboost as ob

        X, y = regression_data
        model = ob.GradientBoosting(
            n_trees=15,
            max_depth=5,
            learning_rate=0.05,
            min_child_weight=3.0,
            reg_lambda=2.0,
        )
        model.fit(X[:400], y[:400])

        save_path = tmp_path / "model.joblib"
        model.save(save_path)

        loaded = ob.GradientBoosting.load(save_path)

        assert loaded.n_trees == 15
        assert loaded.max_depth == 5
        assert loaded.learning_rate == 0.05
        assert loaded.min_child_weight == 3.0
        assert loaded.reg_lambda == 2.0
