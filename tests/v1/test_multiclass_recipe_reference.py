"""Freeze joint multiclass recipe decisions before constructing CUDA consumers."""

import numpy as np
import pytest

from .reference.multiclass_comparison import direct_difference
from .reference.multiclass_recipe import SETTINGS, comparison, fit
from .test_device_multiclass_reference import fixture
from .test_multiclass_composition import original_rows


@pytest.mark.parametrize("width", [2, 3, 5])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("step,rate", SETTINGS)
def test_joint_reference_rejection_and_direct_loss_changes(width, depth, step, rate):
    train, val = original_rows(fixture(width)), original_rows(fixture(width, validation=True))
    result = fit(train, val, depth=depth, step=step, rate=rate)
    before = np.broadcast_to(result["initial"], train["offset"].shape)
    for item in result["history"]:
        if step == "backtracking" and any(t["accepted"] for t in item["trials"]):
            assert direct_difference(
                before, item["raw"], train["target"], train["offset"], train["weight"]
            ) < 0
        assert item["terms"] == width * item["version"]
        before = item["raw"]
    if rate == 0:
        assert result["reason"] == "patience" and result["terms"] == 0
        assert all(len(s["trials"]) == 6 for s in result["history"])
    if rate == 8:
        assert any(len(s["trials"]) > 1 for s in result["history"])
        assert result["terms"] > 0


def test_independent_best_and_patience_anchor_sequence():
    data = dict(target=[0, 1, 2], offset=np.zeros((3, 3)), weight=[1, 1, 1])
    def raw(value):
        return np.tile([np.float32(value), 0, 0], (3, 1)).astype(np.float32)
    best, anchor, stale, counts, selected = raw(2), raw(2), 0, [], []
    for value in (3, 1.95, 1.97, 1.9, 1.89):
        current = raw(value)
        if comparison(data, best, current).improves():
            best = current
        if comparison(data, anchor, current).improves(0.047):
            anchor, stale = current, 0
        else:
            stale += 1
        counts.append(stale)
        selected.append(best[0, 0])
    assert counts == [1, 2, 3, 4, 0]
    np.testing.assert_allclose(selected, [2, 1.95, 1.95, 1.9, 1.89], rtol=1e-6)
