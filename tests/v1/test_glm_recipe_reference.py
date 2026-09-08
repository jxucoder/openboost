"""Freeze reference trajectories and distinguishing recipe conditions before CUDA."""

import numpy as np
import pytest

from .reference.glm_comparison import compare, direct_difference
from .reference.glm_recipe import SETTINGS, fit
from .test_device_glm_reference import original_rows, paired_fixture


@pytest.mark.parametrize("family", ["binary", "poisson"])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("step,rate", SETTINGS)
def test_reference_trajectories_include_rejection_and_real_decrease(family, depth, step, rate):
    train, validation = (original_rows(p) for p in paired_fixture(family))
    result = fit(family, train, validation, depth=depth, step=step, rate=rate)
    before = np.full(len(train["target"]), result["initial"])
    for item in result["history"]:
        if step == "backtracking" and any(t["accepted"] for t in item["trials"]):
            assert (
                direct_difference(
                    family,
                    before,
                    item["raw"],
                    train["target"],
                    train["offset"],
                    train["weight"],
                    train.get("exposure", np.ones(len(before))),
                )
                < 0
            )
        before = item["raw"]
    if rate == 0:
        assert result["reason"] == "patience" and result["terms"] == 0
        assert all(len(s["trials"]) == 6 for s in result["history"])
    if rate == 8:
        assert any(len(s["trials"]) > 1 for s in result["history"])
        assert result["terms"] > 0


@pytest.mark.parametrize("family", ["binary", "poisson"])
def test_three_independent_anchor_sequence(family):
    raw, best, anchor = 2.0, 2.0, 2.0
    stale, counts, selected = 0, [], []
    for value in (3, 1.95, 1.97, 1.9, 1.89):
        raw = float(np.float32(value))
        change = compare(family, [best] * 2, [raw] * 2, [0, 1], [0, 0], [1, 1], [1, 1])
        if change.improves():
            best = raw
        change = compare(family, [anchor] * 2, [raw] * 2, [0, 1], [0, 0], [1, 1], [1, 1])
        if change.improves(0.04 if family == "binary" else 0.7):
            anchor, stale = raw, 0
        else:
            stale += 1
        counts.append(stale)
        selected.append(best)
    assert counts == [1, 2, 3, 4, 0]
    np.testing.assert_allclose(selected, [2, 1.95, 1.95, 1.9, 1.89], rtol=1e-6)
