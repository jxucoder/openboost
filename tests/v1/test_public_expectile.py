"""D1 development objective against independent mathematical references."""

import numpy as np

from openboost import NumericData, Problem
from tests.v1.reference.author import expectile_base
from tests.v1.test_public_extensions import ROOT, load


def extension():
    return load(ROOT / "expectile/src/ob_expectile/__init__.py", "expectile_extension")


def test_weighted_base_matches_stationary_intervals():
    plugin = extension()
    data = NumericData(np.arange(5)[:, None], np.arange(5), ("x",))
    p = Problem(
        data,
        [[-3], [0], [2], [2], [90]],
        data.row_ids,
        weight=[2, 1, 3, 1, 0],
        offset=np.ones((5, 1)),
    )
    np.testing.assert_allclose(
        plugin.Expectile().base(p),
        [expectile_base(p.target[:, 0] - 1, weight=p.weight)],
        atol=1e-12,
    )


def test_two_round_exhaustive_reference(tmp_path):
    import json

    from examples.v1_extensions.expectile_checks import check
    from examples.v1_extensions.expectile_oracle import evidence

    (tmp_path / "expectile-expected.json").write_text(json.dumps(evidence()))
    check(extension(), tmp_path)


def test_derivatives_sign_zero_weight_and_finite_difference():
    from tests.v1.reference.author import expectile

    plugin = extension()
    data = NumericData(np.arange(4)[:, None], np.arange(4), ("x",))
    p = Problem(data, [[-2], [0], [3], [5]], data.row_ids, weight=[2, 1, 3, 0])
    raw = np.zeros((4, 1))
    loss, g, h = plugin.Expectile().geometry(p, raw)
    expected = expectile(raw[:, 0], p.target[:, 0], weight=p.weight)
    np.testing.assert_allclose(loss, expected[0])
    np.testing.assert_allclose(g, expected[1])
    np.testing.assert_allclose(h, expected[2])
    assert g[1] == 0 and h[1] == 1.6
    for i in (0, 2, 3):
        delta = np.zeros_like(raw)
        delta[i] = 1e-5
        plus = plugin.Expectile().loss(p, raw + delta)
        minus = plugin.Expectile().loss(p, raw - delta)
        np.testing.assert_allclose(
            (plus - minus) / 2e-5, g[i] * p.weight[i] / p.weight.sum(), atol=1e-9
        )


def test_invalid_options_and_zero_rounds():
    import pytest

    from openboost import RunContext

    plugin = extension()
    data = NumericData([[0], [1]], [0, 1], ("x",))
    p = Problem(data, [[1], [1]], data.row_ids)
    for tau in (0, 1, float("nan"), True, "0.8"):
        with pytest.raises(ValueError):
            plugin.Expectile(tau)
    for rate in (0, -1, float("inf"), True):
        with pytest.raises(ValueError):
            plugin.fit(p, p, context=RunContext("bad", 0), learning_rate=rate)
    result = plugin.fit(p, p, context=RunContext("zero", 0), rounds=0)
    assert result.steps == () and result.stop.reason == "budget"
    np.testing.assert_array_equal(result.state.train_raw, [[1], [1]])


def test_initialization_across_weighted_asymmetric_fixtures():
    rng = np.random.default_rng(43)
    plugin = extension()
    data = NumericData(np.arange(12)[:, None], np.arange(12), ("x",))
    for tau in (0.05, 0.5, 0.8, 0.95):
        for _ in range(8):
            y = rng.integers(-10, 11, 12).astype(float)
            weight = rng.integers(0, 5, 12).astype(float)
            weight[0] = 1
            p = Problem(data, y[:, None], data.row_ids, weight=weight)
            objective = plugin.Expectile(tau)
            base = objective.base(p)
            np.testing.assert_allclose(
                base, [expectile_base(y, tau=tau, weight=weight)], atol=1e-12
            )
            _, gradient, _ = objective.geometry(p, np.broadcast_to(base, (12, 1)))
            assert abs(np.dot(weight, gradient)) < 1e-10
