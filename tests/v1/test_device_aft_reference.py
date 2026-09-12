"""Freeze AFT tail/domain, offset, censoring and prescribed-round mathematics."""

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

from openboost import NumericData, Problem
from openboost.survival import LogNormalAFT, normal_tail

from .reference.device_aft import DOMAIN_CASES, base, geometry, rounds, tail
from .reference.device_splits import enumerate_candidates


def fixture(*, validation=False, all_censored=False):
    x = np.array([0, 1, 2, 3, 4, np.nan, 6, 7])[:, None]
    lower = np.array([1, 2, 1, 8, 16, 4, 32, 8], dtype=float)
    events = np.array([True, False, True, True, False, False, True, False])
    weight = np.array([1, 2, 1, 3, 1, 2, 2, 1])
    offset = (np.arange(8) % 3 - 1) / 8
    ids = np.arange(8) + (100 if validation else 0)
    if validation:
        order = [7, 2, 5, 0, 6, 3, 1, 4]
        x, lower, events, offset = x[order], lower[order] * 2, events[order], offset[order] / 2
        weight = np.arange(1, 9)
    if all_censored:
        events[:] = False
    data = NumericData(x, ids, ("x",))
    return Problem(data, np.column_stack((lower, np.where(events, lower, np.inf))), ids,
                   target_kind="event_right", weight=weight, offset=offset[:, None])


def original_rows(problem):
    return dict(x=problem.data.values, lower=problem.target[:, 0],
                event=problem.target[:, 0] == problem.target[:, 1],
                offset=problem.offset[:, 0], weight=problem.weight)


@pytest.mark.parametrize("z", [-14, -12, -8.01, -8, -2, 0, 2, 8, 8.01, 12, 40, 1000, 1e8])
def test_cpu_tail_matches_independent_decimal_math(z):
    first, second = tail(z, precision=90), tail(z, precision=120)
    for a, b in zip(first, second, strict=True):
        assert abs(a - b) <= abs(b) * Decimal("1e-65")
    np.testing.assert_allclose(normal_tail(z), [float(v) for v in second], rtol=2e-12, atol=0)


@pytest.mark.parametrize("zero_weight", [False, True])
@pytest.mark.parametrize("name,raw,lower,event,sigma,valid", DOMAIN_CASES, ids=[r[0] for r in DOMAIN_CASES])
def test_stored_support_includes_zero_weight_rows(name, raw, lower, event, sigma, valid, zero_weight):
    args = [raw, 0], [lower, 1], [event, True], [0, 0], [0 if zero_weight else 1, 1], sigma
    if not valid:
        with pytest.raises(ValueError):
            geometry(*args)
    else:
        _, g, h = geometry(*args)
        assert np.isfinite(g).all() and np.all(h > 0)
        if not event:
            assert g[0] < 0
        if name == "subnormal_tail":
            assert 0 < abs(g[0]) < np.finfo(np.float32).tiny


@pytest.mark.parametrize("sigma", [0.5, 1, 2])
def test_cpu_geometry_base_and_finite_differences(sigma):
    p, obj = fixture(), LogNormalAFT(sigma)
    initial = base(p.target[:, 0], p.offset[:, 0], p.weight)
    assert initial == pytest.approx(obj.base(p)[0], rel=1e-6)
    raw = np.full((8, 1), initial)
    expected = geometry(raw[:, 0], p.target[:, 0], p.target[:, 0] == p.target[:, 1], p.offset[:, 0], p.weight, sigma)
    actual = obj.geometry(p, raw)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8)
    for row in (0, 1, 5):
        delta = np.zeros_like(raw)
        delta[row] = 1e-4
        plus, minus = obj.loss(p, raw + delta), obj.loss(p, raw - delta)
        q = p.weight[row] / p.weight.sum()
        assert (plus - minus) / 2e-4 == pytest.approx(actual[1][row] * q, rel=1e-5, abs=1e-8)
        assert (plus + minus - 2 * actual[0]) / 1e-8 == pytest.approx(actual[2][row] * q, rel=2e-5, abs=2e-7)


@pytest.mark.parametrize("sigma", [0.5, 0.7, 1, 2])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_two_round_fixture_has_independent_geometry_and_unique_splits(sigma, depth):
    train, val = fixture(), fixture(validation=True)
    _, steps = rounds(original_rows(train), original_rows(val), sigma=sigma, depth=depth)
    assert len(steps) == 2
    for step in steps:
        for p, raw, value in ((train, step["raw"], step["loss"]), (val, step["validation_raw"], step["score"])):
            assert value == pytest.approx(LogNormalAFT(sigma).loss(p, raw[:, None]), rel=1e-6, abs=1e-8)
        if depth:
            candidates, _ = enumerate_candidates(train.data.values, step["fields"], range(8))
            gains = sorted((c["gain"] for c in candidates if c["legal"]), reverse=True)
            assert gains[0] - gains[1] > 1e-3
    assert not np.array_equal(steps[0]["gradient"], steps[1]["gradient"])


def test_censoring_weights_and_offset_enter_once():
    p = fixture(all_censored=True)
    obj = LogNormalAFT(0.7)
    value = base(p.target[:, 0], p.offset[:, 0], p.weight)
    assert value == pytest.approx(obj.base(p)[0], rel=1e-6)
    shifted = replace(p, offset=p.offset + 2)
    assert obj.base(shifted)[0] == pytest.approx(obj.base(p)[0] - 2)
    raw = np.full((8, 1), value)
    loss, g, h = obj.geometry(p, raw)
    alternate = obj.geometry(replace(p, weight=np.arange(1, 9)), raw)
    np.testing.assert_array_equal(alternate[1], g)
    np.testing.assert_array_equal(alternate[2], h)
    assert alternate[0] != loss
    events = replace(p, target=np.column_stack((p.target[:, 0], p.target[:, 0])))
    assert obj.loss(events, raw) != loss
