"""Tests for LinearLeafGBDT model.

Verifies that gradient boosting with linear models in leaves works correctly
on CPU. These are the first CPU tests for this model variant.
"""

import numpy as np
import pytest

import openboost as ob


def _reference_predict(model, X):
    """Golden reference: per-sample Python loop predict.

    This mirrors, operation for operation (including float32 accumulation
    order), the original O(n_samples) implementation of
    ``LinearLeafTree.predict`` / ``LinearLeafGBDT.predict`` that predated
    vectorization. The vectorized implementation must match it exactly.
    """
    X = np.asarray(X, dtype=np.float32)
    n_samples = X.shape[0]

    base = getattr(model, "base_score_", np.float32(0.0))
    pred = np.full(n_samples, base, dtype=np.float32)

    for tree in model.trees_:
        if tree.training_binned is not None:
            binned = tree.training_binned.transform(X).data
        else:
            binned = ob.array(X).data
        binned = np.asarray(binned)

        ts = tree.tree_structure
        tree_pred = np.zeros(n_samples, dtype=np.float32)
        for i in range(n_samples):
            node = 0
            while ts.left_children[node] != -1:
                if binned[ts.features[node], i] <= ts.thresholds[node]:
                    node = ts.left_children[node]
                else:
                    node = ts.right_children[node]

            leaf_idx = tree.leaf_ids.get(int(node), 0)
            weights = tree.leaf_weights[leaf_idx]
            feat_indices = tree.leaf_features[leaf_idx]

            p = weights[0]
            for j, feat_idx in enumerate(feat_indices):
                if j + 1 < len(weights):
                    p = p + weights[j + 1] * X[i, feat_idx]
            tree_pred[i] = p

        pred = pred + model.learning_rate * tree_pred

    return pred


class TestLinearLeafVectorizedPredict:
    """Vectorized predict must match the original per-sample loop exactly."""

    @pytest.mark.parametrize("max_features_linear", ["sqrt", None, 2])
    def test_predict_matches_reference_loop(self, max_features_linear):
        """Vectorized predictions equal the per-sample reference within 1e-10."""
        rng = np.random.RandomState(7)
        X = rng.randn(300, 8).astype(np.float32)
        y = (2 * X[:, 0] - X[:, 3] + 0.5 * rng.randn(300)).astype(np.float32)

        model = ob.LinearLeafGBDT(
            n_trees=12, max_depth=3, max_features_linear=max_features_linear
        )
        model.fit(X, y)

        # Includes values outside the training range (extrapolation bins)
        X_new = (rng.randn(500, 8) * 2.0).astype(np.float32)
        np.testing.assert_allclose(
            model.predict(X_new), _reference_predict(model, X_new),
            rtol=0, atol=1e-10,
        )

    def test_predict_matches_reference_with_missing(self):
        """Equivalence must hold when prediction data contains NaN."""
        rng = np.random.RandomState(11)
        X = rng.randn(200, 5).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float32)

        model = ob.LinearLeafGBDT(n_trees=8, max_depth=3)
        model.fit(X, y)

        X_new = rng.randn(300, 5).astype(np.float32)
        X_new[::7, 2] = np.nan  # NaN routes through MISSING_BIN
        np.testing.assert_allclose(
            model.predict(X_new), _reference_predict(model, X_new),
            rtol=0, atol=1e-10,
        )


class TestLinearLeafBasic:
    """Basic functionality tests."""

    def test_basic_fit_predict(self, regression_200x10):
        """Fit and predict should produce correct shapes and dtypes."""
        X, y = regression_200x10

        model = ob.LinearLeafGBDT(n_trees=10, max_depth=3, learning_rate=0.1)
        model.fit(X, y)
        pred = model.predict(X)

        assert pred.shape == y.shape, f"Expected shape {y.shape}, got {pred.shape}"
        assert pred.dtype == np.float32
        assert np.all(np.isfinite(pred)), "Predictions should be finite"

    def test_training_reduces_loss(self, regression_200x10):
        """More trees should reduce training loss."""
        X, y = regression_200x10

        model_few = ob.LinearLeafGBDT(n_trees=5, max_depth=3)
        model_few.fit(X, y)
        mse_few = np.mean((model_few.predict(X) - y) ** 2)

        model_many = ob.LinearLeafGBDT(n_trees=30, max_depth=3)
        model_many.fit(X, y)
        mse_many = np.mean((model_many.predict(X) - y) ** 2)

        assert mse_many < mse_few, (
            f"More trees should reduce MSE: {mse_many} >= {mse_few}"
        )

    def test_deterministic(self, regression_100x5):
        """Same input should produce identical output."""
        X, y = regression_100x5

        model1 = ob.LinearLeafGBDT(n_trees=5, max_depth=2)
        model1.fit(X, y)
        pred1 = model1.predict(X)

        model2 = ob.LinearLeafGBDT(n_trees=5, max_depth=2)
        model2.fit(X, y)
        pred2 = model2.predict(X)

        np.testing.assert_array_equal(pred1, pred2)

    def test_predict_before_fit_raises(self):
        """Predict on unfitted model should raise."""
        model = ob.LinearLeafGBDT(n_trees=5)
        rng = np.random.RandomState(42)
        X = rng.randn(10, 3).astype(np.float32)

        with pytest.raises((RuntimeError, AttributeError)):
            model.predict(X)


class TestLinearLeafExtrapolation:
    """Verify that linear leaves improve extrapolation."""

    def test_extrapolation_on_linear_target(self):
        """LinearLeaf must beat standard GBDT decisively out of the training range.

        Standard GBDT predicts piecewise constants, so beyond the training
        range it saturates at the boundary value. Linear leaves extrapolate
        the local trend, so on y = 3x their out-of-range RMSE should be a
        small fraction of standard GBDT's.
        """
        rng = np.random.RandomState(42)
        # Training: x in [-2, 2], y = 3x + small noise
        X_train = rng.uniform(-2, 2, (400, 1)).astype(np.float32)
        y_train = (3 * X_train[:, 0] + 0.1 * rng.randn(400)).astype(np.float32)

        # Test: x in [3, 5], well outside the training range
        X_test = rng.uniform(3, 5, (100, 1)).astype(np.float32)
        y_test = (3 * X_test[:, 0]).astype(np.float32)

        standard = ob.GradientBoosting(n_trees=50, max_depth=4, learning_rate=0.1)
        standard.fit(X_train, y_train)
        std_rmse = float(np.sqrt(np.mean((standard.predict(X_test) - y_test) ** 2)))

        linear = ob.LinearLeafGBDT(n_trees=50, max_depth=3, learning_rate=0.1)
        linear.fit(X_train, y_train)
        lin_pred = linear.predict(X_test)
        lin_rmse = float(np.sqrt(np.mean((lin_pred - y_test) ** 2)))

        assert np.all(np.isfinite(lin_pred)), "Linear leaf predictions should be finite"
        # Standard GBDT saturates near y = 6 (boundary), giving RMSE ~6 here;
        # linear leaves should track y = 3x closely. Factor 2 leaves margin.
        assert lin_rmse <= 0.5 * std_rmse, (
            f"LinearLeafGBDT should extrapolate at least 2x better: "
            f"linear RMSE {lin_rmse:.4f} vs standard RMSE {std_rmse:.4f}"
        )


class TestLinearLeafEdgeCases:
    """Edge cases for LinearLeafGBDT."""

    def test_with_constant_features(self):
        """Should handle constant features gracefully."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3).astype(np.float32)
        X[:, 1] = 5.0  # Constant feature
        y = X[:, 0].copy()

        model = ob.LinearLeafGBDT(n_trees=10, max_depth=2)
        model.fit(X, y)
        pred = model.predict(X)

        assert np.all(np.isfinite(pred))

    def test_with_missing_values(self):
        """Should handle NaN in features."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3).astype(np.float32)
        X[:5, 0] = np.nan
        y = rng.randn(100).astype(np.float32)

        model = ob.LinearLeafGBDT(n_trees=10, max_depth=2)
        model.fit(X, y)
        pred = model.predict(X)

        assert pred.shape == y.shape
        assert np.all(np.isfinite(pred))

    def test_shallow_trees_with_linear_leaves(self):
        """Shallow trees (depth 1-2) should still work with linear leaves."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float32)

        model = ob.LinearLeafGBDT(n_trees=20, max_depth=1, learning_rate=0.1)
        model.fit(X, y)
        pred = model.predict(X)

        mse = np.mean((pred - y) ** 2)
        baseline_mse = np.var(y)
        assert mse < baseline_mse, "Model should fit better than mean prediction"

    def test_single_tree(self):
        """Should work with a single tree."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3).astype(np.float32)
        y = rng.randn(50).astype(np.float32)

        model = ob.LinearLeafGBDT(n_trees=1, max_depth=3)
        model.fit(X, y)
        pred = model.predict(X)

        assert pred.shape == y.shape
        assert np.all(np.isfinite(pred))


class _RecordingCallback(ob.Callback):
    """Records every hook invocation for assertions."""

    def __init__(self):
        self.train_begin_calls = 0
        self.train_end_calls = 0
        self.begin_rounds = []
        self.end_rounds = []
        self.train_losses = []
        self.val_losses = []

    def on_train_begin(self, state):
        self.train_begin_calls += 1

    def on_round_begin(self, state):
        self.begin_rounds.append(state.round_idx)

    def on_round_end(self, state):
        self.end_rounds.append(state.round_idx)
        self.train_losses.append(state.train_loss)
        self.val_losses.append(state.val_loss)
        return True

    def on_train_end(self, state):
        self.train_end_calls += 1


class _NoOpCallback(ob.Callback):
    """Callback that does nothing (inherits all default hooks)."""


def _es_dataset():
    """Train/val split prone to overfitting, for early-stopping tests."""
    rng = np.random.RandomState(123)
    X = rng.randn(250, 6).astype(np.float32)
    y = (1.5 * X[:, 0] - X[:, 2] + 0.3 * rng.randn(250)).astype(np.float32)
    X_val = rng.randn(80, 6).astype(np.float32)
    y_val = (1.5 * X_val[:, 0] - X_val[:, 2] + 0.3 * rng.randn(80)).astype(np.float32)
    return X, y, X_val, y_val


class TestLinearLeafObservability:
    """fit() observability: callbacks, eval_set, early stopping."""

    def test_noop_callback_predictions_identical(self, regression_200x10):
        """Fit with a no-op callback must be byte-identical to plain fit.

        This proves the observability additions (loss computation, callback
        dispatch) do not perturb the training path at all — i.e. the default
        no-args behavior is unchanged from before the feature existed.
        """
        X, y = regression_200x10

        plain = ob.LinearLeafGBDT(n_trees=10, max_depth=3)
        plain.fit(X, y)

        instrumented = ob.LinearLeafGBDT(n_trees=10, max_depth=3)
        instrumented.fit(X, y, callbacks=[_NoOpCallback()])

        np.testing.assert_array_equal(plain.predict(X), instrumented.predict(X))

    def test_callback_rounds_and_losses(self, regression_200x10):
        """Callbacks see correct round indices and correct train/val MSE."""
        X, y = regression_200x10
        X_train, y_train = X[:150], y[:150]
        X_val, y_val = X[150:], y[150:]
        n_trees = 8

        cb = _RecordingCallback()
        model = ob.LinearLeafGBDT(n_trees=n_trees, max_depth=3)
        model.fit(X_train, y_train, callbacks=[cb], eval_set=[(X_val, y_val)])

        # Hook cadence and round numbering
        assert cb.train_begin_calls == 1
        assert cb.train_end_calls == 1
        assert cb.begin_rounds == list(range(n_trees))
        assert cb.end_rounds == list(range(n_trees))

        # Losses are populated and finite every round
        assert all(
            loss is not None and np.isfinite(loss) for loss in cb.train_losses
        )
        assert all(
            loss is not None and np.isfinite(loss) for loss in cb.val_losses
        )

        # Final round's losses must equal MSE of the full model's predictions
        # (incremental tracking follows the same accumulation order as
        # predict(), so this is exact)
        train_mse = float(np.mean((model.predict(X_train) - y_train) ** 2))
        val_mse = float(np.mean((model.predict(X_val) - y_val) ** 2))
        assert cb.train_losses[-1] == train_mse
        assert cb.val_losses[-1] == val_mse

        # val_loss reported to callbacks is the recorded eval history
        assert cb.val_losses == model.evals_result_['eval_0']['mse']

    def test_evals_result_history_lengths(self, regression_200x10):
        """evals_result_ records one MSE per eval set per round trained."""
        X, y = regression_200x10
        X_train, y_train = X[:120], y[:120]
        n_trees = 7

        model = ob.LinearLeafGBDT(n_trees=n_trees, max_depth=3)
        model.fit(
            X_train, y_train,
            eval_set=[(X[120:160], y[120:160]), (X[160:], y[160:])],
        )

        assert set(model.evals_result_.keys()) == {'eval_0', 'eval_1'}
        assert len(model.evals_result_['eval_0']['mse']) == n_trees
        assert len(model.evals_result_['eval_1']['mse']) == n_trees

        # No eval_set -> empty history dict
        plain = ob.LinearLeafGBDT(n_trees=3, max_depth=2)
        plain.fit(X_train, y_train)
        assert plain.evals_result_ == {}

    def test_early_stopping_truncates_on_plateau(self):
        """Early stopping halts before n_trees and truncates to best round."""
        X, y, X_val, y_val = _es_dataset()
        n_trees, patience = 200, 5

        model = ob.LinearLeafGBDT(n_trees=n_trees, max_depth=3)
        model.fit(
            X, y,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=patience,
        )

        history = model.evals_result_['eval_0']['mse']
        rounds_trained = len(history)

        # Stopped early, well short of n_trees
        assert rounds_trained < n_trees

        # Trees truncated to the best round; leaf models live inside each
        # LinearLeafTree so they are truncated with it
        assert model.best_iteration_ == int(np.argmin(history))
        assert len(model.trees_) == model.best_iteration_ + 1
        assert model.best_score_ == min(history)

        # Stop happened exactly patience rounds after the best round
        assert rounds_trained == model.best_iteration_ + patience + 1

        # History covers every round trained, not just up to the best round
        assert rounds_trained > model.best_iteration_

        pred = model.predict(X_val)
        assert np.all(np.isfinite(pred))

    def test_truncated_model_matches_short_refit(self):
        """Restored-best model predicts identically to a fresh fit with
        n_trees == best_iteration_ + 1 (training is deterministic)."""
        X, y, X_val, y_val = _es_dataset()

        es_model = ob.LinearLeafGBDT(n_trees=200, max_depth=3)
        es_model.fit(X, y, eval_set=[(X_val, y_val)], early_stopping_rounds=5)

        short = ob.LinearLeafGBDT(n_trees=es_model.best_iteration_ + 1, max_depth=3)
        short.fit(X, y)

        np.testing.assert_array_equal(es_model.predict(X_val), short.predict(X_val))

    def test_early_stopping_callback_rounds(self):
        """Under early stopping, callbacks see rounds 0..rounds_trained-1."""
        X, y, X_val, y_val = _es_dataset()

        cb = _RecordingCallback()
        model = ob.LinearLeafGBDT(n_trees=200, max_depth=3)
        model.fit(
            X, y,
            callbacks=[cb],
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=5,
        )

        rounds_trained = len(model.evals_result_['eval_0']['mse'])
        assert cb.end_rounds == list(range(rounds_trained))
        assert cb.train_end_calls == 1

    def test_early_stopping_without_eval_set_warns(self, regression_100x5):
        """early_stopping_rounds with no eval_set warns and has no effect."""
        X, y = regression_100x5

        model = ob.LinearLeafGBDT(n_trees=5, max_depth=2)
        with pytest.warns(UserWarning, match="eval_set"):
            model.fit(X, y, early_stopping_rounds=3)

        assert len(model.trees_) == 5  # trained to completion


class TestLinearLeafPersistence:
    """Save/load functionality."""

    def test_save_load_roundtrip(self, regression_100x5, tmp_path):
        """Predictions should match after save/load."""
        X, y = regression_100x5

        model = ob.LinearLeafGBDT(n_trees=5, max_depth=2)
        model.fit(X, y)
        pred_before = model.predict(X)

        path = str(tmp_path / "linear_leaf_model.json")
        model.save(path)

        loaded = ob.LinearLeafGBDT.load(path)
        pred_after = loaded.predict(X)

        np.testing.assert_array_equal(pred_before, pred_after)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
