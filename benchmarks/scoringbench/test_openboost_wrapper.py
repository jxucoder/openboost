"""Upstream-style smoke test for the OpenBoost ScoringBench wrapper."""

import numpy as np
from scoringbench.wrappers.base import DistributionPrediction

from benchmarks.scoringbench.openboost_wrapper import (
    OpenBoostHistogramWrapper,
    OpenBoostWrapper,
)


def test_openboost_wrapper_distribution_contract():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(160, 4)).astype(np.float32)
    sigma = 0.25 + np.abs(X[:, 1])
    y = (2 * X[:, 0] - X[:, 2] + rng.normal(scale=sigma)).astype(np.float32)

    model = OpenBoostWrapper(
        backend="cpu",
        n_trees=12,
        learning_rate=0.05,
        max_depth=2,
        n_quantiles=15,
    )
    returned = model.fit(X[:120], y[:120])
    distribution = model.predict_distribution(X[120:])

    assert returned is model
    assert isinstance(distribution, DistributionPrediction)
    assert distribution.probas.shape == (40, 14)
    assert distribution.bin_edges.shape == (40, 15)
    assert distribution.mean.shape == (40,)
    assert np.all(np.isfinite(distribution.probas))
    assert np.all(np.isfinite(distribution.bin_edges))
    assert np.allclose(distribution.probas.sum(axis=1), 1.0)
    assert np.all(np.diff(distribution.bin_edges, axis=1) > 0)


def test_openboost_wrapper_forwards_crps_training_objective():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(100, 3)).astype(np.float32)
    y = (X[:, 0] + rng.normal(scale=0.4, size=100)).astype(np.float32)
    model = OpenBoostWrapper(
        backend="cpu",
        n_trees=5,
        learning_rate=0.05,
        max_depth=2,
        n_quantiles=9,
        model_params={"training_objective": "crps"},
    )

    model.fit(X[:80], y[:80])

    assert model._model.training_objective == "crps"
    assert np.all(np.isfinite(model.predict(X[80:])))


def test_openboost_histogram_wrapper_preserves_native_distribution_grid():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(100, 3)).astype(np.float32)
    y = (X[:, 0] + rng.normal(scale=0.5, size=100)).astype(np.float32)
    model = OpenBoostHistogramWrapper(
        n_distribution_bins=8,
        n_trees=3,
        max_depth=2,
        n_feature_bins=12,
    ).fit(X[:80], y[:80])

    distribution = model.predict_distribution(X[80:])

    assert isinstance(distribution, DistributionPrediction)
    assert distribution.is_natively_gridded_model is True
    assert distribution.probas.shape == (20, 8)
    assert distribution.bin_edges.shape == (9,)
    np.testing.assert_allclose(distribution.probas.sum(axis=1), 1.0)
    np.testing.assert_allclose(distribution.mean, model.predict(X[80:]))


def test_openboost_histogram_wrapper_selects_temperature_on_inner_validation():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(80, 3)).astype(np.float32)
    y = (X[:, 0] + rng.normal(scale=0.3, size=80)).astype(np.float32)
    model = OpenBoostHistogramWrapper(
        n_distribution_bins=6,
        n_trees=2,
        max_depth=1,
        n_feature_bins=10,
        temperature_grid=(0.7, 1.0, 1.2),
        calibration_fraction=0.2,
        calibration_seed=3,
    ).fit(X[:60], y[:60])

    assert model._selected_temperature in model.temperature_grid
    assert set(model._temperature_scores) == set(model.temperature_grid)
    assert all(np.isfinite(list(model._temperature_scores.values())))
    distribution = model.predict_distribution(X[60:])
    np.testing.assert_allclose(distribution.mean, model.predict(X[60:]))
