import numpy as np
import pytest
from normal_fisher import ChannelDecay, NormalFisher

from openboost.experimental import ExecutionContext


def ctx():
    return ExecutionContext("cpu", np, np.random.default_rng(7), 0)


def test_gradient_finite_difference_and_fisher():
    obj = NormalFisher()
    y = np.array([-1, 2, 4], np.float32)
    w = np.array([0, 0.5, 2], np.float32)
    raw = {
        "mu": np.array([0.2, 0.4, 0.1], np.float32),
        "log_sigma": np.array([-0.3, 0.2, 0.5], np.float32),
    }
    stats = obj.step(raw, y, w, context=ctx())

    def independent_nll(mu, logs):
        return (
            logs + 0.5 * ((y.astype(float) - mu) * np.exp(-logs)) ** 2 + 0.5 * np.log(2 * np.pi)
        ) * w

    for channel in raw:
        plus = {k: v.astype(float) for k, v in raw.items()}
        minus = {k: v.astype(float) for k, v in raw.items()}
        plus[channel] += 1e-5
        minus[channel] -= 1e-5
        fd = (
            independent_nll(plus["mu"], plus["log_sigma"])
            - independent_nll(minus["mu"], minus["log_sigma"])
        ) / 2e-5
        np.testing.assert_allclose(stats[channel][0], fd, rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(stats["mu"][1], w / np.exp(2 * raw["log_sigma"]), rtol=1e-6)
    np.testing.assert_array_equal(stats["log_sigma"][1], 2 * w)
    assert obj.loss_value(raw, y, w, context=ctx()) == pytest.approx(
        independent_nll(raw["mu"].astype(float), raw["log_sigma"].astype(float)).sum() / w.sum()
    )
    base = obj.init_raw(y, w)
    mean = np.average(y.astype(float), weights=w)
    assert base["mu"] == pytest.approx(mean)
    assert base["log_sigma"] == pytest.approx(0.5 * np.log(np.average((y - mean) ** 2, weights=w)))


def test_schedule_and_invalid_inputs():
    schedule = ChannelDecay(tau=1)
    assert schedule.coefficients(0, ("mu", "log_sigma"), 0.2) == {"mu": 0.2, "log_sigma": 0.1}
    assert schedule.coefficients(1, ("mu", "log_sigma"), 0.2) == {"mu": 0.1, "log_sigma": 0.05}
    for value in (0, -1, np.inf, np.nan):
        with pytest.raises(ValueError):
            ChannelDecay(tau=value)
    obj = NormalFisher()
    for weights in (np.zeros(2), np.array([-1, 2]), np.array([1, np.nan])):
        with pytest.raises(ValueError):
            obj.init_raw(np.ones(2), weights)
    with pytest.raises(ValueError):
        obj.init_raw(np.ones(2), extra={"exposure": np.ones(2)})
    with pytest.raises(ValueError):
        obj.step({"mu": np.ones(2), "log_sigma": np.full(2, -1000)}, np.ones(2), context=ctx())
    with pytest.raises(ValueError):
        obj.step({"mu": np.ones(2), "log_sigma": np.full(2, 1000)}, np.ones(2), context=ctx())
    assert obj.init_raw(np.ones(2))["log_sigma"] == pytest.approx(0.5 * np.log(1e-6))


def test_declared_devices():
    assert NormalFisher.supported_devices == frozenset({"cpu", "cuda"})
