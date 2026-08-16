"""Upstream-style smoke test for the OpenBoost ScoringBench wrapper."""

import numpy as np
from scoringbench.wrappers.base import DistributionPrediction

from benchmarks.scoringbench.openboost_wrapper import OpenBoostWrapper


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
