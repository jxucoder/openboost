"""Correctness tests for shared-vector histogram distribution boosting."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import openboost as ob
from openboost._core._vector_tree import _find_best_vector_split, fit_vector_tree
from openboost._models._histogram_boost import (
    _continuous_crps_loss,
    _continuous_crps_terms,
    _crps_grad_gn,
)


def test_continuous_crps_logit_gradient_matches_finite_difference():
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(3, 5))
    y = np.array([-0.2, 1.3, 4.8])
    bin_edges = np.array([-1.0, 0.0, 0.7, 2.0, 3.5, 5.0])
    grad, _ = _crps_grad_gn(logits, y, bin_edges)

    eps = 1e-6
    numerical = np.empty_like(logits)
    for row in range(logits.shape[0]):
        for output in range(logits.shape[1]):
            plus = logits.copy()
            minus = logits.copy()
            plus[row, output] += eps
            minus[row, output] -= eps
            numerical[row, output] = (
                _continuous_crps_loss(plus, y, bin_edges)[row]
                - _continuous_crps_loss(minus, y, bin_edges)[row]
            ) / (2 * eps)

    np.testing.assert_allclose(grad, numerical, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.sum(grad, axis=1), 0.0, atol=1e-7)


def test_continuous_crps_includes_uniform_within_bin_distance():
    # A Uniform(0, 1) forecast observed at 0.5 has CRPS 1/12. The second
    # far-away bin has negligible softmax mass at these logits.
    loss = _continuous_crps_loss(
        np.array([[50.0, -50.0]]),
        np.array([0.5]),
        np.array([0.0, 1.0, 2.0]),
    )
    assert loss[0] == pytest.approx(1.0 / 12.0)


def test_continuous_crps_psd_diagonal_matches_explicit_jacobian():
    logits = np.array([[0.4, -0.2, 0.1, 0.7]])
    y = np.array([1.2])
    bin_edges = np.array([-0.5, 0.0, 1.5, 2.25, 4.0])
    scale = 1.7
    _, hess = _crps_grad_gn(
        logits,
        y,
        bin_edges,
        curvature_scale=scale,
        curvature_floor=0.0,
    )

    p = np.exp(logits[0] - np.max(logits[0]))
    p /= p.sum()
    _, pairwise_distance = _continuous_crps_terms(y, bin_edges)
    jacobian = np.diag(p) - np.outer(p, p)
    expected = scale * np.diag(jacobian.T @ (-pairwise_distance) @ jacobian)

    np.testing.assert_allclose(hess[0], expected, rtol=2e-6, atol=1e-8)
    assert np.all(hess >= 0.0)


def test_crps_sample_weight_scales_gradient_and_curvature():
    logits = np.zeros((3, 4))
    y = np.array([-0.5, 1.2, 4.0])
    bin_edges = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    weights = np.array([0.0, 2.0, 5.0])
    grad, hess = _crps_grad_gn(logits, y, bin_edges)
    weighted_grad, weighted_hess = _crps_grad_gn(
        logits,
        y,
        bin_edges,
        sample_weight=weights,
    )
    np.testing.assert_allclose(weighted_grad, grad * weights[:, None])
    np.testing.assert_allclose(weighted_hess, hess * weights[:, None])


def _brute_split(binned, grad, hess, reg_lambda):
    total_grad = grad.sum(axis=0)
    total_hess = hess.sum(axis=0)

    def score(g, h):
        return np.mean(g * g / (h + reg_lambda))

    parent = score(total_grad, total_hess)
    best = None
    for feature in range(binned.shape[0]):
        for threshold in range(254):
            for missing_left in (True, False):
                left = binned[feature] <= threshold
                if missing_left:
                    left |= binned[feature] == ob.MISSING_BIN
                else:
                    left &= binned[feature] != ob.MISSING_BIN
                if not np.any(left) or np.all(left):
                    continue
                gain = (
                    score(grad[left].sum(axis=0), hess[left].sum(axis=0))
                    + score(grad[~left].sum(axis=0), hess[~left].sum(axis=0))
                    - parent
                )
                candidate = (gain, feature, threshold, missing_left)
                if best is None or candidate[0] > best[0]:
                    best = candidate
    return best


def test_vector_split_and_leaf_values_match_brute_force():
    binned = np.array(
        [
            [0, 0, 1, 1, ob.MISSING_BIN],
            [0, 1, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    grad = np.array([[-2.0, -1.0], [-1.5, -0.5], [1.0, 2.0], [1.5, 2.5], [-0.5, 0.5]])
    hess = np.ones_like(grad)
    reg_lambda = 1.0
    expected = _brute_split(binned, grad, hess, reg_lambda)
    split = _find_best_vector_split(
        binned,
        grad,
        hess,
        min_child_weight=0.0,
        reg_lambda=reg_lambda,
        reg_alpha=0.0,
        min_gain=0.0,
    )
    assert (split.feature, split.threshold, split.missing_go_left) == expected[1:]
    assert split.gain == pytest.approx(expected[0])

    tree = fit_vector_tree(
        binned,
        grad,
        hess,
        max_depth=1,
        min_child_weight=0.0,
        reg_lambda=reg_lambda,
    )
    prediction = tree.predict(binned)
    assert prediction.shape == grad.shape
    left = binned[split.feature] <= split.threshold
    if split.missing_go_left:
        left |= binned[split.feature] == ob.MISSING_BIN
    else:
        left &= binned[split.feature] != ob.MISSING_BIN
    np.testing.assert_allclose(
        prediction[left][0],
        -grad[left].sum(axis=0) / (hess[left].sum(axis=0) + reg_lambda),
    )


def test_histogram_boost_defaults_are_frozen_and_sklearn_cloneable():
    model = ob.HistogramBoost()
    assert model.n_distribution_bins == 50
    assert model.n_trees == 100
    assert model.learning_rate == 0.05
    assert model.max_depth == 6
    assert model.curvature_scale == 1.0

    sklearn = pytest.importorskip("sklearn.base")
    cloned = sklearn.clone(model)
    assert cloned.get_params() == model.get_params()
    assert not cloned.__sklearn_is_fitted__()


def test_histogram_boost_fit_predict_distribution_and_no_crossing():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(80, 3)).astype(np.float32)
    X[3, 1] = np.nan
    y = (X[:, 0] + rng.normal(scale=0.4, size=80)).astype(np.float32)
    model = ob.HistogramBoost(
        n_distribution_bins=12,
        n_trees=4,
        learning_rate=0.05,
        max_depth=2,
        n_feature_bins=16,
    ).fit(X, y)

    edges_before = model.target_bin_edges_.copy()
    dist = model.predict_distribution(np.array([[100.0, 0.0, 0.0], X[0]]))
    assert dist.probas.shape == (2, 12)
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1.0)
    assert np.all(np.diff(np.cumsum(dist.probas, axis=1), axis=1) >= -1e-12)
    assert np.all(np.diff(np.column_stack([dist.quantile(q) for q in [0.1, 0.5, 0.9]])) >= 0)
    np.testing.assert_array_equal(model.target_bin_edges_, edges_before)
    assert model.predict(X[:5]).shape == (5,)


def test_histogram_distribution_output_moments_quantiles_and_sampling():
    dist = ob.HistogramDistributionOutput(
        probas=np.array([[0.25, 0.75], [1.0, 0.0]]),
        bin_edges=np.array([0.0, 1.0, 2.0]),
    )
    np.testing.assert_allclose(dist.mean(), [1.25, 0.5])
    np.testing.assert_allclose(dist.quantile(0.5), [4.0 / 3.0, 0.5])
    lower, upper = dist.interval(0.2)
    assert np.all(lower <= upper)
    samples1 = dist.sample(20, seed=7)
    samples2 = dist.sample(20, seed=7)
    np.testing.assert_array_equal(samples1, samples2)
    assert samples1.shape == (2, 20)
    assert np.all((samples1 >= 0.0) & (samples1 <= 2.0))

    with pytest.raises(ValueError, match="bin_edges must contain only finite"):
        ob.HistogramDistributionOutput(
            probas=np.array([[0.5, 0.5]]),
            bin_edges=np.array([0.0, np.nan, 2.0]),
        )


def test_histogram_boost_sample_weight_controls_base_distribution():
    X = np.arange(8, dtype=np.float32).reshape(-1, 1)
    y = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    unweighted = ob.HistogramBoost(
        n_distribution_bins=3,
        n_trees=0,
        base_smoothing=0.0,
    ).fit(X, y)
    weighted = ob.HistogramBoost(
        n_distribution_bins=3,
        n_trees=0,
        base_smoothing=0.0,
    ).fit(X, y, sample_weight=np.array([10, 10, 10, 10, 1, 1, 1, 1]))

    p_unweighted = unweighted.predict_distribution(X[:1]).probas[0]
    p_weighted = weighted.predict_distribution(X[:1]).probas[0]
    assert p_unweighted[0] == pytest.approx(p_unweighted[-1])
    assert p_weighted[0] / p_weighted[-1] == pytest.approx(10.0)
    with pytest.raises(ValueError, match="positive total weight"):
        weighted.fit(X, y, sample_weight=np.zeros(len(y)))
    with pytest.raises(ValueError, match="only finite"):
        weighted.fit(X, y, sample_weight=np.full(len(y), np.inf))


def test_histogram_boost_base_smoothing_is_total_prior_weight():
    X = np.arange(4, dtype=np.float32).reshape(-1, 1)
    y = np.zeros(4, dtype=np.float32)
    model = ob.HistogramBoost(
        n_distribution_bins=5,
        n_trees=0,
        base_smoothing=1.0,
    ).fit(X, y)

    probabilities = model.predict_distribution(X[:1]).probas[0]
    labels = np.searchsorted(model.target_bin_edges_[1:-1], y, side="right")
    counts = np.bincount(labels, minlength=5)
    expected = (counts + 1.0 / 5.0) / (len(y) + 1.0)
    np.testing.assert_allclose(probabilities, expected)


def test_histogram_boost_persistence_preserves_vector_predictions(tmp_path):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 2)).astype(np.float32)
    y = (X[:, 0] - 0.5 * X[:, 1]).astype(np.float32)
    model = ob.HistogramBoost(
        n_distribution_bins=8,
        n_trees=3,
        max_depth=2,
        n_feature_bins=10,
    ).fit(X, y)
    expected = model.predict_distribution(X[:7]).probas
    path = tmp_path / "histogram.joblib"
    model.save(path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        restored = ob.HistogramBoost.load(path)
        auto_restored = ob.load(path)
    np.testing.assert_allclose(restored.predict_distribution(X[:7]).probas, expected)
    np.testing.assert_allclose(auto_restored.predict_distribution(X[:7]).probas, expected)
    assert restored.trees_[0].leaf_values_array.ndim == 2
