"""A7-A10 checks against hand calculations and independent calculus."""

import math

import numpy as np
import pytest

from .reference.positive import (
    gamma,
    gamma_base,
    poisson,
    poisson_base,
    poisson_predict,
    policy_losses,
    tweedie,
)
from .reference.survival import aft, aft_predict, normal_tail
from .reference.tree import fit_tree


def test_poisson_exposure_base_and_hand_geometry():
    base = poisson_base([1, 3], [1, 2], weight=[1, 2], minimum_rate=1e-8)
    assert base == pytest.approx(math.log(7 / 5))
    loss, g, h = poisson([0, 0], [0, 2], [1, 2])
    assert loss == pytest.approx((3 - math.log(2)) / 2)
    np.testing.assert_allclose(g, [1, 0])
    np.testing.assert_allclose(h, [1, 2])
    _, _, doubled = poisson([0, 0], [0, 2], [2, 4])
    np.testing.assert_allclose(doubled, 2 * h)
    assert poisson_base([0, 0], [1, 2], minimum_rate=0.01) == pytest.approx(math.log(0.01))


def test_gamma_tweedie_hand_values():
    loss, g, h = gamma([0, 0], [1, 3])
    assert loss == 2
    np.testing.assert_allclose(g, [0, -2])
    np.testing.assert_allclose(h, [1, 3])
    loss, g, h = tweedie([0, 0], [0, 2], power=1.5)
    assert loss == 4
    np.testing.assert_allclose(g, [1, -1])
    np.testing.assert_allclose(h, [0.5, 1.5])


@pytest.mark.parametrize("kind", ["poisson", "gamma", "tweedie", "event", "censored"])
def test_positive_and_aft_finite_differences(kind):
    def objective(raw):
        if kind == "poisson":
            return poisson(raw, [2, 1], [1.5, 0.7])
        if kind == "gamma":
            return gamma(raw, [2.0, 0.7])
        if kind == "tweedie":
            return tweedie(raw, [0.0, 2.0], power=1.3)
        return aft(raw, [1.2, 3.0], [1.2, 3.0] if kind == "event" else [np.inf, np.inf], sigma=0.8)

    raw = np.array([0.3, -0.2])
    _, g, h = objective(raw)
    for i in range(2):
        delta = np.eye(2)[i] * 1e-5
        plus, minus = objective(raw + delta), objective(raw - delta)
        assert (plus[0] - minus[0]) / 2e-5 * 2 == pytest.approx(g[i], abs=1e-8)
        assert (plus[1][i] - minus[1][i]) / 2e-5 == pytest.approx(h[i], abs=1e-8)


def test_aft_event_censoring_and_units():
    event = aft([0], [1], [1])
    censored = aft([0], [1], [np.inf])
    assert event[0] == pytest.approx(0.5 * math.log(2 * math.pi))
    assert censored[0] == pytest.approx(math.log(2))
    assert event[1][0] == 0
    assert censored[1][0] == pytest.approx(-math.sqrt(2 / math.pi))
    assert censored[2][0] == pytest.approx(2 / math.pi)
    output = aft_predict([math.log(2)], sigma=1, times=[1, 2, 4], probabilities=[0.1, 0.5, 0.9])
    assert output["median"][0] == pytest.approx(2)
    assert output["mean"][0] == pytest.approx(2 * math.exp(0.5))
    assert output["survival"][0, 1] == pytest.approx(0.5)
    assert np.all(np.diff(output["survival"][0]) < 0)
    assert np.all(np.diff(output["quantile"][0]) > 0)
    assert output["quantile"][0, 1] == pytest.approx(2)


@pytest.mark.parametrize("z", [-10.0, 0.0, 8.0, 10.0, 20.0, 30.0, 40.0, 100.0])
def test_normal_tail_against_independent_integral(z):
    logsf, mills, curvature = normal_tail(z)
    if z <= 30:
        sf = math.erfc(z / math.sqrt(2)) / 2
        assert logsf == pytest.approx(math.log(sf), abs=1e-11)
    else:
        # SF(z)/phi(z) = integral_0^inf exp(-u-u²/(2z²))du / z.
        # Gauss-Laguerre integration is independent of the continued fraction.
        nodes, weights = np.polynomial.laguerre.laggauss(64)
        ratio = sum(weights * np.exp(-(nodes**2) / (2 * z * z))) / z
        assert logsf == pytest.approx(
            -z * z / 2 - 0.5 * math.log(2 * math.pi) + math.log(ratio), abs=1e-11
        )
        assert mills == pytest.approx(1 / ratio, rel=1e-12)
    assert np.isfinite(logsf) and mills >= 0 and 0 <= curvature <= 1.000001
    if z >= 40:
        assert curvature > 0.999


@pytest.mark.parametrize(
    "objective,target,extra",
    [
        (poisson, [0, 2], {"exposure": [1, 1]}),
        (gamma, [1, 3], {}),
        (tweedie, [0, 2], {"power": 1.5}),
    ],
)
def test_positive_two_round_root_and_integer_weights(objective, target, extra):
    raw = np.zeros(2)
    for _ in range(2):
        _, g, h = objective(raw, target, weight=[1, 3], **extra)
        tree = fit_tree([[0], [0]], g, h, weight=[1, 3], max_depth=0)
        mu = math.exp(raw[0])
        if objective is poisson:
            expected = (6 - 4 * mu) / (1 + 4 * mu)
        elif objective is gamma:
            expected = (10 / mu - 4) / (1 + 10 / mu)
        else:
            expected = (6 / math.sqrt(mu) - 4 * math.sqrt(mu)) / (
                1 + 2 * math.sqrt(mu) + 3 / math.sqrt(mu)
            )
        assert tree.predict([[0]])[0] == pytest.approx(expected)
        raw += 0.1 * tree.predict([[0], [0]])
    repeated_extra = {"exposure": [1] * 4} if objective is poisson else extra
    repeated = objective(raw[[0, 1, 1, 1]], np.array(target)[[0, 1, 1, 1]], **repeated_extra)
    assert repeated[0] == pytest.approx(objective(raw, target, weight=[1, 3], **extra)[0])


def test_aft_two_round_mixed_event_censoring():
    raw = np.zeros(2)
    for _ in range(2):
        loss, g, h = aft(raw, [1, 1], [1, np.inf])
        z = -raw[0]
        sf = math.erfc(z / math.sqrt(2)) / 2
        m = math.exp(-z * z / 2) / math.sqrt(2 * math.pi) / sf
        expected = (z + m) / (2 + m * (m - z))
        tree = fit_tree([[0], [0]], g, h, max_depth=0)
        assert tree.predict([[0]])[0] == pytest.approx(expected)
        raw += 0.1 * tree.predict([[0], [0]])
        assert aft(raw, [1, 1], [1, np.inf])[0] < loss


def test_policy_join_paid_count_and_annualized_units():
    records, excluded = policy_losses(
        [("a", 2, 0.5), ("b", 0, 1.0), ("c", 1, 1.0), ("d", 0, 1.0)],
        [("a", 10.0), ("a", 20.0), ("a", 0.0), ("d", 5.0), ("orphan", 8.0)],
    )
    assert records == (("a", 0.5, 30.0, 60.0, 2, 15.0), ("b", 1.0, 0.0, 0.0, 0, 0.0))
    assert set(excluded) == {
        ("a", "nonpositive_payment"),
        ("orphan", "orphan_payment"),
        ("c", "count_without_payment"),
        ("d", "payment_without_count"),
    }
    a = records[0]
    assert (a[4] / a[1]) * a[5] == a[3]  # paid-count rate * mean paid severity
    reverse, _ = policy_losses([("b", 0, 1.0), ("a", 2, 0.5)], [("a", 20.0), ("a", 10.0)])
    assert reverse == records
    raw = np.log([60.0, 1.0])
    loss = tweedie(raw, [60, 0], weight=[0.5, 1], power=1.5)[0]
    assert loss == pytest.approx((0.5 * 4 * math.sqrt(60) + 2) / 1.5)


def test_poisson_prediction_requires_exposure_and_gamma_base():
    original = poisson_predict([math.log(3)], [2])
    doubled = poisson_predict([math.log(3)], [4])
    np.testing.assert_allclose(original["rate"], doubled["rate"])
    np.testing.assert_allclose(doubled["count_mean"], 2 * original["count_mean"])
    assert gamma_base([1, 3], weight=[1, 3]) == pytest.approx(math.log(2.5))


@pytest.mark.parametrize(
    "lower,upper",
    [
        ([0], [np.inf]),
        ([-1], [1]),
        ([1], [0.5]),
        ([1], [2]),
        ([1], [np.nan]),
        ([np.nan], [np.inf]),
        ([np.inf], [np.inf]),
        ([1], []),
    ],
)
def test_aft_rejects_unsupported_intervals(lower, upper):
    with pytest.raises(ValueError):
        aft([0], lower, upper)


@pytest.mark.parametrize("sigma", [0, -1, np.nan, np.inf, 1e-200, 1e200])
def test_aft_rejects_invalid_scale(sigma):
    with pytest.raises(ValueError):
        aft([0], [1], [1], sigma=sigma)


@pytest.mark.parametrize("exposure", [[0], [-1], [np.nan], [np.inf], []])
def test_poisson_rejects_invalid_exposure(exposure):
    with pytest.raises(ValueError):
        poisson([0], [1], exposure)


@pytest.mark.parametrize(
    "objective,target,extra",
    [
        (poisson, [-1], {"exposure": [1]}),
        (poisson, [0.5], {"exposure": [1]}),
        (gamma, [0], {}),
        (gamma, [-1], {}),
        (tweedie, [-1], {}),
        (tweedie, [1], {"power": 1}),
        (tweedie, [1], {"power": 2}),
    ],
)
def test_invalid_positive_target_support(objective, target, extra):
    with pytest.raises(ValueError):
        objective([0], target, **extra)


def test_out_of_range_geometry_rejected_without_clipping():
    with pytest.raises(ValueError):
        poisson([1000], [1], [1])
    with pytest.raises(ValueError):
        gamma([-1000], [1])
    with pytest.raises(ValueError):
        tweedie([2000], [0])
    with pytest.raises(ValueError):
        aft_predict([1000])


def test_aft_integer_weights_and_zero_weight_geometry_boundary():
    raw, lo, hi = np.array([0.2, -0.3]), np.array([1.0, 2.0]), np.array([1.0, np.inf])
    loss, g, h = aft(raw, lo, hi, weight=[1, 3])
    plain = aft(raw, lo, hi)
    np.testing.assert_array_equal(g, plain[1])
    np.testing.assert_array_equal(h, plain[2])
    ids = [0, 1, 1, 1]
    assert loss == pytest.approx(aft(raw[ids], lo[ids], hi[ids])[0])
    assert aft(raw, lo, hi, weight=[1, 0])[0] == pytest.approx(aft(raw[:1], lo[:1], hi[:1])[0])


def test_policy_metadata_validation_and_paid_count_differs_from_claim_count():
    with pytest.raises(ValueError):
        policy_losses([("a", 0, 1), ("a", 0, 1)], [])
    rows, _ = policy_losses([("a", 3, 2)], [("a", 10), ("a", 20)])
    assert rows[0][4] == 2  # not raw ClaimNb=3
    assert rows[0][3] == (rows[0][4] / 2) * rows[0][5] == 15


def test_censored_curvature_far_tail_and_output_time_rescaling():
    # log(lower)=40, location=0: ordinary survival CDF subtraction would be zero.
    result = aft([0], [math.exp(40)], [np.inf])
    assert result[0] > 800 and result[2][0] > 0.999
    factor = 7
    event = aft([0.3], [2], [2])
    scaled_event = aft([0.3 + math.log(factor)], [2 * factor], [2 * factor])
    assert scaled_event[0] == pytest.approx(event[0] + math.log(factor))  # time-density Jacobian
    np.testing.assert_allclose(scaled_event[1:], event[1:])
    censored = aft([0.3], [2], [np.inf])
    scaled_censored = aft([0.3 + math.log(factor)], [2 * factor], [np.inf])
    for actual, expected in zip(scaled_censored, censored, strict=True):
        np.testing.assert_allclose(actual, expected)


def test_gamma_large_but_representable_ratio_and_tail_transition():
    loss, g, h = gamma([math.log(1e308)], [1e308])
    assert loss == pytest.approx(1 + math.log(1e308))
    assert g[0] == pytest.approx(0, abs=1e-13)
    assert h[0] == pytest.approx(1)
    np.testing.assert_allclose(normal_tail(8), normal_tail(np.nextafter(8.0, np.inf)), rtol=1e-12)


@pytest.mark.parametrize("minimum_rate", [0, -1, np.nan, np.inf])
def test_all_zero_rate_policy_must_be_explicit_and_valid(minimum_rate):
    with pytest.raises(ValueError):
        poisson_base([0], [1], minimum_rate=minimum_rate)


@pytest.mark.parametrize("weight", [[0, 0], [1, -1], [np.nan, 1], [1]])
def test_aft_and_positive_reject_invalid_weight(weight):
    with pytest.raises(ValueError):
        aft([0, 0], [1, 1], [1, np.inf], weight=weight)
    with pytest.raises(ValueError):
        poisson([0, 0], [1, 1], [1, 1], weight=weight)
