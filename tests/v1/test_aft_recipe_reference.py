"""Freeze distinguishing AFT recipe trajectories before real device consumers."""

import numpy as np
import pytest

from .reference.aft_comparison import compare, direct_difference
from .reference.aft_recipe import SETTINGS, fit
from .test_device_aft_reference import fixture, original_rows


@pytest.mark.parametrize("sigma", [.5, .7, 1, 2])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("step,rate", SETTINGS)
def test_reference_trajectories_distinguish_retries_rejection_and_decrease(sigma, depth, step, rate):
    train, validation = original_rows(fixture()), original_rows(fixture(validation=True))
    result = fit(train, validation, sigma=sigma, depth=depth, step=step, rate=rate)
    before = np.full(len(train["lower"]), result["initial"])
    for item in result["history"]:
        if step == "backtracking" and any(t["accepted"] for t in item["trials"]):
            assert direct_difference(before, item["raw"], train["lower"], train["event"], train["offset"], train["weight"], sigma) < 0
        before = item["raw"]
    if rate == 0:
        assert result["reason"] == "patience" and result["terms"] == 0
        assert all(len(s["trials"]) == 6 for s in result["history"])
    if rate == 8:
        assert any(len(s["trials"]) > 1 for s in result["history"])
        assert result["terms"] > 0


def test_distinct_best_and_patience_anchors_for_fixed_events():
    sigma, best, anchor, stale = .7, 2., 2., 0
    selected, counts = [], []
    for value in (3, 1.95, 1.97, 1.9, 1.89):
        raw = float(np.float32(value))
        if compare([best], [raw], [1], [True], [0], [1], sigma).improves():
            best = raw
        if compare([anchor], [raw], [1], [True], [0], [1], sigma).improves(.21/sigma**2):
            anchor, stale = raw, 0
        else:
            stale += 1
        selected.append(best)
        counts.append(stale)
    assert counts == [1, 2, 3, 4, 0]
    np.testing.assert_allclose(selected, [2, 1.95, 1.95, 1.9, 1.89], rtol=1e-6)


@pytest.mark.parametrize("sigma", [.7, 2])
def test_all_censored_recipe_keeps_censoring_semantics(sigma):
    train = original_rows(fixture(all_censored=True))
    validation = original_rows(fixture(validation=True, all_censored=True))
    result = fit(train, validation, sigma=sigma, depth=1, step="backtracking", rate=.25)
    assert result["terms"] == 3 and result["best_terms"] == 3
    assert all(s["trials"][0]["accepted"] for s in result["history"])
