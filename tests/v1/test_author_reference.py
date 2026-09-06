"""D1/D3/D4 mathematical and mutation counterexamples."""

import numpy as np
import pytest

from .reference.author import (
    expectile,
    expectile_base,
    ordered_normal,
    penalized_quantile,
    penalized_tree,
)
from .reference.coupled import normal
from .reference.scalar import squared_error
from .reference.tree import fit_tree


def test_expectile_hand_signs_weights_and_symmetric_limit():
    loss, g, h = expectile([0, 2, 1], [1, 1, 1], tau=0.8, weight=[1, 3, 0])
    assert loss == pytest.approx(0.35)
    np.testing.assert_allclose(g, [-1.6, 0.4, 0])
    np.testing.assert_allclose(h, [1.6, 0.4, 1.6])  # declared r=0 branch, no smooth-Hessian claim
    assert expectile_base([0, 2], tau=0.8) == pytest.approx(1.6)
    assert expectile_base([0, 2], tau=0.8, weight=[3, 1]) == pytest.approx(8 / 7)
    for actual, expected in zip(
        expectile([0.3, 2], [1, 1], tau=0.5), squared_error([0.3, 2], [1, 1]), strict=True
    ):
        np.testing.assert_allclose(actual, expected)


def test_expectile_calculus_base_and_two_rounds():
    raw = np.array([-0.3, 2.2])
    y = np.array([0.0, 2.0])
    eps = 1e-5
    _, g, h = expectile(raw, y, tau=0.8)
    for i in range(2):
        delta = np.eye(2)[i] * eps
        plus, minus = expectile(raw + delta, y, tau=0.8), expectile(raw - delta, y, tau=0.8)
        assert (plus[0] - minus[0]) / eps == pytest.approx(g[i], abs=1e-9)
        assert (plus[1][i] - minus[1][i]) / (2 * eps) == pytest.approx(h[i], abs=1e-9)
    base = expectile_base(y, tau=0.8)
    assert sum(expectile([base, base], y, tau=0.8)[1]) == pytest.approx(0, abs=1e-14)
    raw = np.full(2, base)
    predictions = raw.copy()
    for iteration in range(2):
        _, g, h = expectile(raw, y, tau=0.8)
        tree = fit_tree([[0], [1]], g, h, max_depth=1)
        expected = (
            [-16 / 35, 16 / 65]
            if iteration == 0
            else [-(0.4 * (1.6 - 8 / 175)) / 1.4, 1.6 * (0.4 - 8 / 325) / 2.6]
        )
        np.testing.assert_allclose(tree.predict([[0], [1]]), expected)
        raw += 0.1 * tree.predict([[0], [1]])
        predictions += 0.1 * np.array(expected)
    np.testing.assert_allclose(raw, predictions)


def objective(value, residual, weight, q, penalty, anchor):
    r = np.asarray(residual) - value
    return float(
        np.dot(weight, np.maximum(q * r, (q - 1) * r)) + penalty * (value - anchor) ** 2 / 2
    )


def test_penalized_quantile_stationary_point_is_not_default_quantile():
    value = penalized_quantile([0, 2, 10], 0.5, [1, 3, 1], penalty=1, anchor=0)
    assert value == 1.5  # lies strictly between residual breakpoints
    assert penalized_quantile([0, 2, 10], 0.5, [1, 3, 1], penalty=10, anchor=0) == 0.15
    assert penalized_quantile([0, 2, 10], 0.5, [1, 3, 1], penalty=0.1, anchor=0) == 2


@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("anchor", [-3.0, 1.0, 8.0])
def test_penalized_quantile_subgradient_and_global_minimum(q, anchor):
    residual = np.array([0.0, 2.0, 2.0, 10.0])
    weight = np.array([1.0, 0.0, 3.0, 1.0])
    penalty = 0.7
    value = penalized_quantile(residual, q, weight, penalty=penalty, anchor=anchor)
    left = sum(weight[residual < value]) - q * sum(weight) + penalty * (value - anchor)
    right = sum(weight[residual <= value]) - q * sum(weight) + penalty * (value - anchor)
    assert left <= 1e-12 and right >= -1e-12
    best = objective(value, residual, weight, q, penalty, anchor)
    for candidate in np.linspace(-5, 12, 201):
        assert objective(candidate, residual, weight, q, penalty, anchor) >= best - 1e-12


def test_penalized_tree_two_rounds_uses_current_routed_residual():
    y = np.array([0.0, 2.0, 10.0])
    raw = np.full(3, 2.0)
    for iteration in range(2):
        tree = penalized_tree(
            [[0], [0], [1]], raw, y, 0.5, weight=[2, 1, 2], penalty=0.1, anchor=0, max_depth=1
        )
        assert tree.predict([[1]])[0] == pytest.approx(8 if iteration == 0 else 7.2)
        assert tree.predict([[0]])[0] == pytest.approx(-2 if iteration == 0 else -1.8)
        raw += 0.1 * tree.predict([[0], [0], [1]])
    np.testing.assert_allclose(raw, [1.62, 1.62, 3.52])


def test_d4_exact_rates_reversed_direction_and_nan_candidates():
    raw = np.zeros((2, 2))
    rates = tuple(0.1 * 0.5**j for j in range(6))

    def reversed_gradient(r, y, weight=None):
        loss, g, metric = normal(r, y, weight=weight)
        return loss, -g, metric

    rejected = ordered_normal([[0], [0]], raw, [1, 3], objective=reversed_gradient)
    for result in rejected:
        assert not result.accepted and result.terms == ()
        assert tuple(t[0] for t in result.trials) == rates
        np.testing.assert_array_equal(result.raw_after, raw)

    def invalid_candidate(r, y, weight=None):
        loss, g, metric = normal(r, y, weight=weight)
        return (loss if np.array_equal(r, raw) else float("nan")), g, metric

    for result in ordered_normal([[0], [0]], raw, [1, 3], objective=invalid_candidate):
        assert not result.accepted and len(result.trials) == 6
        assert all(t[1] is None for t in result.trials)


def test_d4_two_round_ordered_commit_and_first_rejection():
    raw = np.zeros((2, 2))
    for _ in range(2):
        first, second = ordered_normal([[0], [0]], raw, [1, 3])
        assert first.accepted and second.accepted
        assert second.raw_before == first.raw_after
        assert first.loss_after < first.loss_before and second.loss_after < second.loss_before
        raw = np.array(second.raw_after)

    def reject_mean(r, y, weight=None):
        loss, g, metric = normal(r, y, weight=weight)
        g[:, 0] *= -1
        return loss, g, metric

    first, second = ordered_normal([[0], [0]], np.zeros((2, 2)), [1, 3], objective=reject_mean)
    assert not first.accepted and second.accepted
    assert second.raw_before == first.raw_before


def test_penalty_weight_mass_is_not_silently_normalized():
    r = [0, 2, 10]
    original = penalized_quantile(r, 0.5, [1, 3, 1], penalty=1, anchor=0)
    replicated = penalized_quantile([0, 2, 2, 2, 10], 0.5, penalty=1, anchor=0)
    assert original == replicated == 1.5
    assert penalized_quantile(r, 0.5, [2, 6, 2], penalty=1, anchor=0) == 2
    assert penalized_quantile(r, 0.5, [2, 6, 2], penalty=2, anchor=0) == original
    assert penalized_quantile([-100, 0, 2, 10], 0.5, [0, 1, 3, 1], penalty=1, anchor=0) == original


@pytest.mark.parametrize("tau", [0, 1, -1, np.nan])
def test_expectile_rejects_invalid_asymmetry(tau):
    with pytest.raises(ValueError):
        expectile_base([0, 2], tau=tau)
    with pytest.raises(ValueError):
        expectile([0, 0], [0, 2], tau=tau)


@pytest.mark.parametrize("penalty", [0, -1, np.nan, np.inf])
def test_penalized_leaf_requires_finite_positive_penalty(penalty):
    with pytest.raises(ValueError):
        penalized_quantile([0, 2], 0.5, penalty=penalty, anchor=0)


def test_expectile_base_replication_and_zero_weight_outlier():
    value = expectile_base([0, 2], tau=0.8, weight=[3, 1])
    assert expectile_base([0, 0, 0, 2], tau=0.8) == pytest.approx(value)
    assert expectile_base([-100, 0, 2], tau=0.8, weight=[0, 3, 1]) == pytest.approx(value)
    assert expectile_base([2, 2], tau=0.8) == 2
    with pytest.raises(ValueError):
        expectile_base([0, 2], weight=[0, 0])


def test_ordered_rejection_does_not_mutate_input_or_global_rng():
    raw = np.zeros((2, 2))
    original = raw.copy()
    rng_before = np.random.get_state()

    def reversed_gradient(r, y, weight=None):
        loss, g, metric = normal(r, y, weight=weight)
        return loss, -g, metric

    records = ordered_normal([[0], [0]], raw, [1, 3], objective=reversed_gradient)
    rng_after = np.random.get_state()
    assert rng_before[0] == rng_after[0] and rng_before[2:] == rng_after[2:]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    np.testing.assert_array_equal(raw, original)
    raw[:] = 100
    for record in records:
        np.testing.assert_array_equal(record.raw_after, original)
        assert record.loss_after == record.loss_before and record.terms == ()
