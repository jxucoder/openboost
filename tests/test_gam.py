"""Tests for OpenBoostGAM model.

Verifies that the GPU-accelerated Generalized Additive Model works
correctly on CPU. These are the first CPU tests for this model variant.
"""

import numpy as np
import pytest

import openboost as ob


class TestGAMBasic:
    """Basic functionality tests."""

    def test_basic_fit_predict(self, regression_200x10):
        """Fit and predict should produce correct shapes."""
        X, y = regression_200x10

        gam = ob.OpenBoostGAM(n_rounds=50, learning_rate=0.05, reg_lambda=1.0)
        gam.fit(X, y)
        pred = gam.predict(X)

        assert pred.shape == y.shape
        assert pred.dtype == np.float32
        assert np.all(np.isfinite(pred))

    def test_shape_values_shape(self, regression_200x10):
        """shape_values_ should be (n_features, 256)."""
        X, y = regression_200x10

        gam = ob.OpenBoostGAM(n_rounds=20, learning_rate=0.05)
        gam.fit(X, y)

        assert gam.shape_values_ is not None
        assert gam.shape_values_.shape == (10, 256), (
            f"Expected shape (10, 256), got {gam.shape_values_.shape}"
        )

    def test_training_reduces_loss(self, regression_200x10):
        """Training should reduce loss compared to baseline."""
        X, y = regression_200x10

        gam = ob.OpenBoostGAM(n_rounds=100, learning_rate=0.05)
        gam.fit(X, y)
        pred = gam.predict(X)

        mse = np.mean((pred - y) ** 2)
        baseline_mse = np.var(y)

        assert mse < baseline_mse * 0.5, (
            f"GAM MSE ({mse:.4f}) should be well below baseline ({baseline_mse:.4f})"
        )

    def test_deterministic(self, regression_100x5):
        """Same input should produce identical output."""
        X, y = regression_100x5

        gam1 = ob.OpenBoostGAM(n_rounds=20, learning_rate=0.05)
        gam1.fit(X, y)
        pred1 = gam1.predict(X)

        gam2 = ob.OpenBoostGAM(n_rounds=20, learning_rate=0.05)
        gam2.fit(X, y)
        pred2 = gam2.predict(X)

        np.testing.assert_array_equal(pred1, pred2)


class TestGAMInterpretability:
    """Verify GAM interpretability properties."""

    def test_shape_functions_capture_correct_features(self):
        """When y = f(X[:,0]), feature 0's shape function should be most active."""
        rng = np.random.RandomState(42)
        X = rng.randn(300, 5).astype(np.float32)
        y = np.sin(X[:, 0]).astype(np.float32)

        gam = ob.OpenBoostGAM(n_rounds=500, learning_rate=0.03)
        gam.fit(X, y)

        # Feature 0 should have the largest shape function range
        ranges = [np.ptp(gam.shape_values_[f]) for f in range(5)]
        assert np.argmax(ranges) == 0, (
            f"Feature 0 should have largest range but ranges are: {ranges}"
        )

    def test_additive_prediction_structure(self, regression_100x5):
        """Predictions should be sum of shape functions + base score."""
        X, y = regression_100x5

        gam = ob.OpenBoostGAM(n_rounds=30, learning_rate=0.05)
        gam.fit(X, y)

        # Get predictions the normal way
        pred_normal = gam.predict(X)

        # Manually compute from shape functions
        binned = gam.X_binned_
        binned_data = binned.data
        if hasattr(binned_data, 'copy_to_host'):
            binned_data = binned_data.copy_to_host()
        binned_data = np.asarray(binned_data)

        base = getattr(gam, 'base_score_', np.float32(0.0))
        pred_manual = np.full(len(y), base, dtype=np.float32)
        for f in range(X.shape[1]):
            pred_manual += gam.shape_values_[f, binned_data[f, :]]

        np.testing.assert_allclose(pred_normal, pred_manual, atol=1e-5)


class TestGAMClassification:
    """GAM with classification loss."""

    def test_logloss(self, binary_500x10):
        """GAM should work with logloss for binary classification."""
        X, y = binary_500x10

        gam = ob.OpenBoostGAM(n_rounds=100, learning_rate=0.05, loss='logloss')
        gam.fit(X, y)
        pred_raw = gam.predict(X)

        # Convert to probabilities
        prob = 1.0 / (1.0 + np.exp(-pred_raw))
        labels = (prob > 0.5).astype(float)
        accuracy = np.mean(labels == y)

        assert accuracy > 0.70, f"GAM classification accuracy {accuracy:.3f} < 0.70"


class TestGAMEdgeCases:
    """Edge cases for OpenBoostGAM."""

    def test_predict_before_fit_raises(self):
        """Predict on unfitted model should raise."""
        gam = ob.OpenBoostGAM(n_rounds=10)
        rng = np.random.RandomState(42)
        X = rng.randn(10, 3).astype(np.float32)

        with pytest.raises(RuntimeError, match="not fitted"):
            gam.predict(X)

    def test_single_round(self):
        """Should work with a single boosting round."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3).astype(np.float32)
        y = rng.randn(50).astype(np.float32)

        gam = ob.OpenBoostGAM(n_rounds=1, learning_rate=0.1)
        gam.fit(X, y)
        pred = gam.predict(X)

        assert pred.shape == y.shape
        assert np.all(np.isfinite(pred))

    def test_constant_target(self):
        """GAM with constant target should predict that constant."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3).astype(np.float32)
        y = np.full(100, 2.5, dtype=np.float32)

        gam = ob.OpenBoostGAM(n_rounds=50, learning_rate=0.1)
        gam.fit(X, y)
        pred = gam.predict(X)

        np.testing.assert_allclose(pred, 2.5, atol=0.2,
                                   err_msg="Should converge to constant target")


class TestGAMNBins:
    """n_bins validation and non-default bin counts."""

    def test_n_bins_300_raises(self):
        """n_bins above the uint8 contract should raise at construction."""
        with pytest.raises(ValueError, match="n_bins"):
            ob.OpenBoostGAM(n_bins=300)

    def test_n_bins_1_raises(self):
        """n_bins below 2 is meaningless and should raise."""
        with pytest.raises(ValueError, match="n_bins"):
            ob.OpenBoostGAM(n_bins=1)

    def test_default_n_bins_valid(self):
        """The default configuration must pass its own validation."""
        gam = ob.OpenBoostGAM()
        assert 2 <= gam.n_bins <= 256

    def test_n_bins_64_trains_and_predicts(self, regression_200x10):
        """A reduced bin count should train and predict correctly."""
        X, y = regression_200x10

        gam = ob.OpenBoostGAM(n_rounds=100, learning_rate=0.1, n_bins=64)
        gam.fit(X, y)
        pred = gam.predict(X)

        assert pred.shape == y.shape
        assert np.all(np.isfinite(pred))
        mse = np.mean((pred - y) ** 2)
        assert mse < 0.5 * np.var(y), f"MSE {mse:.4f} should beat baseline {np.var(y):.4f}"

        # 64-bin request yields at most 63 edges (occupied bins 0..len(edges)-1)
        assert all(len(e) <= 63 for e in gam.X_binned_.bin_edges)
        # Storage stays 256-wide so uint8 bins (incl. MISSING_BIN) are in bounds
        assert gam.shape_values_.shape == (10, 256)


class TestGAMPlotting:
    """plot_shape_function (skipped when matplotlib is not installed)."""

    @pytest.fixture(autouse=True)
    def _agg_backend(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg", force=True)
        yield
        import matplotlib.pyplot as plt

        plt.close("all")

    @pytest.fixture
    def fitted(self, regression_100x5):
        X, y = regression_100x5
        gam = ob.OpenBoostGAM(n_rounds=10, learning_rate=0.05)
        gam.fit(X, y)
        return gam, X

    @staticmethod
    def _shape_line(ax):
        """The step line for the shape function (axhline only has 2 points)."""
        return max(ax.lines, key=lambda ln: len(ln.get_xdata()))

    def test_plot_returns_new_axes(self, fitted):
        """Without ax, a new axes is created, drawn on, and returned."""
        gam, _ = fitted
        ax = gam.plot_shape_function(0)

        assert ax is not None
        assert len(ax.lines) >= 1
        assert ax.get_ylabel() == "Contribution to prediction"

    def test_plot_draws_on_provided_ax(self, fitted):
        """A provided ax is drawn on and returned unchanged."""
        import matplotlib.pyplot as plt

        gam, _ = fitted
        fig, ax = plt.subplots()
        returned = gam.plot_shape_function(1, feature_name="feat_1", ax=ax)

        assert returned is ax
        assert len(ax.lines) >= 1
        assert "feat_1" in ax.get_xlabel()

    def test_xaxis_uses_original_feature_scale(self, fitted):
        """x values must span the feature's data range, not bin indices 0-255."""
        gam, X = fitted
        ax = gam.plot_shape_function(0)

        xdata = np.asarray(self._shape_line(ax).get_xdata(), dtype=np.float64)
        fmin, fmax = float(X[:, 0].min()), float(X[:, 0].max())
        span = fmax - fmin

        # Within the data range (bin edges are interior quantiles)...
        assert xdata.min() >= fmin - 0.05 * span
        assert xdata.max() <= fmax + 0.05 * span
        # ...and covering most of it (rules out bin indices 0-255)
        assert xdata.min() <= fmin + 0.25 * span
        assert xdata.max() >= fmax - 0.25 * span

    def test_plot_before_fit_raises(self):
        """Plotting an unfitted model should raise a clear error."""
        gam = ob.OpenBoostGAM(n_rounds=5)
        with pytest.raises(RuntimeError, match="not fitted"):
            gam.plot_shape_function(0)

    def test_plot_bad_feature_idx_raises(self, fitted):
        """Out-of-range feature_idx should raise ValueError."""
        gam, _ = fitted
        with pytest.raises(ValueError, match="feature_idx"):
            gam.plot_shape_function(99)
        with pytest.raises(ValueError, match="feature_idx"):
            gam.plot_shape_function(-1)


class TestGAMPersistence:
    """Save/load functionality."""

    def test_save_load_roundtrip(self, regression_100x5, tmp_path):
        """Predictions should match after save/load."""
        X, y = regression_100x5

        gam = ob.OpenBoostGAM(n_rounds=10, learning_rate=0.05)
        gam.fit(X, y)
        pred_before = gam.predict(X)

        path = str(tmp_path / "gam_model.json")
        gam.save(path)

        loaded = ob.OpenBoostGAM.load(path)
        pred_after = loaded.predict(X)

        np.testing.assert_array_equal(pred_before, pred_after)

    def test_save_load_roundtrip_interactions(self, tmp_path):
        """Interaction tables and pairs must survive save/load."""
        rng = np.random.RandomState(5)
        X = rng.uniform(-2, 2, (400, 3)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] + 2.0 * X[:, 0] * X[:, 1]).astype(np.float32)

        gam = ob.OpenBoostGAM(
            n_rounds=30, learning_rate=0.1, n_bins=16,
            interactions=1, interaction_rounds=20,
        )
        gam.fit(X, y)
        assert gam.interaction_pairs_, "expected at least one selected pair"
        pred_before = gam.predict(X)

        path = str(tmp_path / "gam_interactions.joblib")
        gam.save(path)

        loaded = ob.OpenBoostGAM.load(path)
        assert loaded.interaction_pairs_ == gam.interaction_pairs_
        assert set(loaded.pair_shape_values_) == set(gam.pair_shape_values_)
        np.testing.assert_array_equal(loaded.predict(X), pred_before)

        # Generic auto-detecting loader must also work
        loaded_generic = ob.load(path)
        np.testing.assert_array_equal(loaded_generic.predict(X), pred_before)


class _NoOpCallback(ob.Callback):
    """Callback that observes but never modifies training."""


class TestGAMBackwardCompat:
    """New parameters must default to exactly the pre-change behavior."""

    def test_noop_callback_identical_predictions(self, regression_100x5):
        """The callback-enabled code path must not perturb the numerics."""
        X, y = regression_100x5

        gam_plain = ob.OpenBoostGAM(n_rounds=30, learning_rate=0.05)
        gam_plain.fit(X, y)

        gam_cb = ob.OpenBoostGAM(n_rounds=30, learning_rate=0.05)
        gam_cb.fit(X, y, callbacks=[_NoOpCallback()])

        np.testing.assert_array_equal(gam_plain.predict(X), gam_cb.predict(X))

    def test_explicit_defaults_identical_predictions(self, regression_100x5):
        """Explicitly passing the new defaults must change nothing."""
        X, y = regression_100x5

        gam_plain = ob.OpenBoostGAM(n_rounds=30, learning_rate=0.05)
        gam_plain.fit(X, y)

        gam_explicit = ob.OpenBoostGAM(
            n_rounds=30, learning_rate=0.05,
            interactions=0, interaction_rounds=None,
            smoothing=0.0, monotone=None,
        )
        gam_explicit.fit(X, y)

        np.testing.assert_array_equal(
            gam_plain.predict(X), gam_explicit.predict(X)
        )
        assert gam_explicit.interaction_pairs_ == []
        assert gam_explicit.pair_shape_values_ == {}

    def test_param_validation(self):
        """New constructor params validate at construction time."""
        with pytest.raises(ValueError, match="interactions"):
            ob.OpenBoostGAM(interactions=-1)
        with pytest.raises(ValueError, match="interaction_rounds"):
            ob.OpenBoostGAM(interaction_rounds=-5)
        with pytest.raises(ValueError, match="smoothing"):
            ob.OpenBoostGAM(smoothing=-0.1)
        with pytest.raises(ValueError, match="monotone"):
            ob.OpenBoostGAM(monotone={0: 2})


class TestGAMInteractions:
    """GA2M-style pairwise interaction terms."""

    @staticmethod
    def _interaction_data(n=2400, seed=7):
        rng = np.random.RandomState(seed)
        X = rng.uniform(-2, 2, (n, 4)).astype(np.float32)
        y = (
            np.sin(X[:, 0])
            + 0.5 * X[:, 1]
            + 3.0 * np.sin(X[:, 0] * X[:, 1])
            + 0.05 * rng.randn(n)
        ).astype(np.float32)
        n_train = int(0.75 * n)
        return X[:n_train], y[:n_train], X[n_train:], y[n_train:]

    def test_interactions_improve_oos_mse(self):
        """interactions>0 must beat interactions=0 on y with a strong pair term."""
        X_tr, y_tr, X_te, y_te = self._interaction_data()

        common = dict(n_rounds=150, learning_rate=0.1, n_bins=16)
        gam0 = ob.OpenBoostGAM(**common, interactions=0)
        gam0.fit(X_tr, y_tr)
        mse0 = float(np.mean((gam0.predict(X_te) - y_te) ** 2))

        gam1 = ob.OpenBoostGAM(**common, interactions=1, interaction_rounds=150)
        gam1.fit(X_tr, y_tr)
        mse1 = float(np.mean((gam1.predict(X_te) - y_te) ** 2))

        assert mse1 < 0.75 * mse0, (
            f"Interactions should improve OOS MSE: {mse1:.4f} vs {mse0:.4f}"
        )

    def test_fast_ranking_selects_true_pair(self):
        """FAST-style ranking should pick (0, 1) as the top pair."""
        X_tr, y_tr, _, _ = self._interaction_data()

        gam = ob.OpenBoostGAM(
            n_rounds=100, learning_rate=0.1, n_bins=16,
            interactions=2, interaction_rounds=20,
        )
        gam.fit(X_tr, y_tr)

        assert len(gam.interaction_pairs_) == 2
        assert gam.interaction_pairs_[0] == (0, 1), (
            f"Expected (0, 1) as top pair, got {gam.interaction_pairs_}"
        )
        table = gam.get_pair_shape_function(0, 1)
        assert table.shape == (256, 256)
        assert np.any(table != 0.0)

    def test_additive_structure_with_pairs(self):
        """Predictions must equal base + 1D lookups + 2D lookups."""
        X_tr, y_tr, X_te, _ = self._interaction_data(n=1200)

        gam = ob.OpenBoostGAM(
            n_rounds=40, learning_rate=0.1, n_bins=16,
            interactions=1, interaction_rounds=30,
        )
        gam.fit(X_tr, y_tr)

        X_binned = gam.X_binned_.transform(X_te)
        binned = np.asarray(X_binned.data)

        pred_manual = np.full(binned.shape[1], gam.base_score_, dtype=np.float32)
        for f in range(binned.shape[0]):
            pred_manual += gam.shape_values_[f, binned[f, :]]
        for (i, j), table in gam.pair_shape_values_.items():
            pred_manual += table[binned[i, :], binned[j, :]]

        np.testing.assert_allclose(gam.predict(X_te), pred_manual, atol=1e-5)


class TestGAMSmoothing:
    """Fused-ridge smoothing of 1D shape functions."""

    @staticmethod
    def _sparse_noisy_data(n_train=150, n_test=400, seed=3):
        rng = np.random.RandomState(seed)
        X = rng.uniform(-2, 2, (n_train + n_test, 2)).astype(np.float32)
        y = (np.sin(X[:, 0]) + 0.3 * rng.randn(n_train + n_test)).astype(np.float32)
        return X[:n_train], y[:n_train], X[n_train:], y[n_train:]

    @staticmethod
    def _total_variation(gam, feature_idx):
        n_used = len(gam.X_binned_.bin_edges[feature_idx])
        vals = gam.shape_values_[feature_idx, :n_used]
        return float(np.abs(np.diff(vals)).sum())

    def test_smoothing_reduces_total_variation(self):
        """Smoothing must damp sparse-bin noise without large OOS MSE loss."""
        X_tr, y_tr, X_te, y_te = self._sparse_noisy_data()

        common = dict(n_rounds=200, learning_rate=0.1)
        gam_rough = ob.OpenBoostGAM(**common, smoothing=0.0)
        gam_rough.fit(X_tr, y_tr)
        tv_rough = self._total_variation(gam_rough, 0)
        mse_rough = float(np.mean((gam_rough.predict(X_te) - y_te) ** 2))

        gam_smooth = ob.OpenBoostGAM(**common, smoothing=2.0)
        gam_smooth.fit(X_tr, y_tr)
        tv_smooth = self._total_variation(gam_smooth, 0)
        mse_smooth = float(np.mean((gam_smooth.predict(X_te) - y_te) ** 2))

        assert tv_smooth < 0.7 * tv_rough, (
            f"Smoothing should reduce total variation: {tv_smooth:.3f} "
            f"vs {tv_rough:.3f}"
        )
        assert mse_smooth < 1.2 * mse_rough, (
            f"Smoothing must not cost much OOS accuracy: {mse_smooth:.4f} "
            f"vs {mse_rough:.4f}"
        )

    def test_smoothing_keeps_missing_bin_raw(self):
        """The missing-value bin (255) must not be coupled to ordinal bins."""
        rng = np.random.RandomState(9)
        X = rng.uniform(-2, 2, (300, 2)).astype(np.float64)
        X[rng.rand(300) < 0.3, 0] = np.nan
        y = np.where(np.isnan(X[:, 0]), 2.0, -1.0).astype(np.float32)
        y += 0.01 * rng.randn(300).astype(np.float32)

        gam = ob.OpenBoostGAM(n_rounds=150, learning_rate=0.1, smoothing=5.0)
        gam.fit(X, y)

        # Missing rows carry a strongly positive contribution in bin 255 that
        # smoothing must not have dragged toward the (negative) ordinal bins.
        assert gam.shape_values_[0, 255] > 1.0


class TestGAMMonotone:
    """Monotone shape constraints via PAVA projection."""

    @staticmethod
    def _monotone_data(slope, n=500, seed=11):
        rng = np.random.RandomState(seed)
        X = rng.uniform(-2, 2, (n, 3)).astype(np.float32)
        y = (
            slope * X[:, 0] + np.sin(2 * X[:, 1]) + 0.2 * rng.randn(n)
        ).astype(np.float32)
        return X, y

    def test_monotone_increasing(self):
        X, y = self._monotone_data(slope=1.5)

        gam = ob.OpenBoostGAM(n_rounds=200, learning_rate=0.1, monotone={0: 1})
        gam.fit(X, y)

        n_used = len(gam.X_binned_.bin_edges[0])
        diffs = np.diff(gam.shape_values_[0, :n_used])
        assert np.all(diffs >= -1e-7), (
            f"Shape must be non-decreasing; min diff = {diffs.min()}"
        )
        # The constrained model must still fit the monotone effect
        mse = float(np.mean((gam.predict(X) - y) ** 2))
        assert mse < 0.5 * np.var(y)

    def test_monotone_decreasing(self):
        X, y = self._monotone_data(slope=-1.5)

        gam = ob.OpenBoostGAM(n_rounds=200, learning_rate=0.1, monotone={0: -1})
        gam.fit(X, y)

        n_used = len(gam.X_binned_.bin_edges[0])
        diffs = np.diff(gam.shape_values_[0, :n_used])
        assert np.all(diffs <= 1e-7), (
            f"Shape must be non-increasing; max diff = {diffs.max()}"
        )

    def test_monotone_bad_index_raises(self):
        rng = np.random.RandomState(0)
        X = rng.randn(50, 3).astype(np.float32)
        y = rng.randn(50).astype(np.float32)

        gam = ob.OpenBoostGAM(n_rounds=5, monotone={10: 1})
        with pytest.raises(ValueError, match="out of range"):
            gam.fit(X, y)


class TestGAMEvalSetCallbacks:
    """eval_set / callbacks / early_stopping_rounds on fit()."""

    @staticmethod
    def _split_data(n=300, seed=21):
        rng = np.random.RandomState(seed)
        X = rng.uniform(-2, 2, (n, 3)).astype(np.float32)
        y = (X[:, 0] + 0.5 * np.sin(2 * X[:, 1]) + 0.5 * rng.randn(n)).astype(
            np.float32
        )
        n_train = 2 * n // 3
        return X[:n_train], y[:n_train], X[n_train:], y[n_train:]

    def test_evals_result_history(self):
        """Every eval set gets a per-round metric history."""
        X_tr, y_tr, X_va, y_va = self._split_data()

        gam = ob.OpenBoostGAM(n_rounds=25, learning_rate=0.1)
        gam.fit(X_tr, y_tr, eval_set=[(X_va, y_va), (X_tr, y_tr)])

        assert set(gam.evals_result_) == {"eval_0", "eval_1"}
        assert len(gam.evals_result_["eval_0"]["mse"]) == 25
        assert len(gam.evals_result_["eval_1"]["mse"]) == 25
        # Training-set loss should be decreasing overall
        hist = gam.evals_result_["eval_1"]["mse"]
        assert hist[-1] < hist[0]

    def test_evals_result_includes_interaction_rounds(self):
        """With interactions the history spans main + interaction rounds."""
        rng = np.random.RandomState(2)
        X = rng.uniform(-2, 2, (600, 3)).astype(np.float32)
        y = (X[:, 0] * X[:, 1]).astype(np.float32)

        gam = ob.OpenBoostGAM(
            n_rounds=20, learning_rate=0.1, n_bins=16,
            interactions=1, interaction_rounds=10,
        )
        gam.fit(X[:400], y[:400], eval_set=[(X[400:], y[400:])])

        assert len(gam.evals_result_["eval_0"]["mse"]) == 30

    def test_history_callback(self):
        X_tr, y_tr, X_va, y_va = self._split_data()

        hist = ob.HistoryCallback()
        gam = ob.OpenBoostGAM(n_rounds=15, learning_rate=0.1)
        gam.fit(X_tr, y_tr, callbacks=[hist], eval_set=[(X_va, y_va)])

        assert len(hist.history["train_loss"]) == 15
        assert len(hist.history["val_loss"]) == 15

    def test_early_stopping_truncates(self):
        """Early stopping must stop training well before n_rounds."""
        X_tr, y_tr, X_va, y_va = self._split_data(n=240)

        gam = ob.OpenBoostGAM(n_rounds=400, learning_rate=0.5)
        gam.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            early_stopping_rounds=15,
        )

        n_recorded = len(gam.evals_result_["eval_0"]["mse"])
        assert n_recorded < 400, "training should have stopped early"
        assert hasattr(gam, "best_iteration_")
        assert gam.best_iteration_ < n_recorded

    def test_early_stopping_restores_best(self):
        """After restore, the val loss must equal the recorded best score."""
        X_tr, y_tr, X_va, y_va = self._split_data(n=240)

        gam = ob.OpenBoostGAM(n_rounds=400, learning_rate=0.5)
        gam.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            early_stopping_rounds=15,
        )

        history = gam.evals_result_["eval_0"]["mse"]
        assert np.isclose(gam.best_score_, min(history), rtol=1e-6)

        val_mse = float(
            np.mean(
                (
                    np.asarray(gam.predict(X_va), dtype=np.float64)
                    - np.asarray(y_va, dtype=np.float64)
                ) ** 2
            )
        )
        assert np.isclose(val_mse, gam.best_score_, rtol=1e-5), (
            f"Restored model val MSE {val_mse:.6f} != best_score_ "
            f"{gam.best_score_:.6f}"
        )

    def test_early_stopping_without_eval_warns(self, regression_100x5):
        X, y = regression_100x5

        gam = ob.OpenBoostGAM(n_rounds=5, learning_rate=0.1)
        with pytest.warns(UserWarning, match="eval_set"):
            gam.fit(X, y, early_stopping_rounds=3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
