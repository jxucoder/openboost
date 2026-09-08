"""Freeze exhaustive split expectations against current public CPU behavior."""

from functools import partial

import numpy as np
import pytest

from openboost.binning import Binning
from openboost.data import NumericData, Problem
from openboost.ops import candidates, choose, feasible, histogram, newton_leaf, partition, score
from openboost.stats import newton

from .reference.device_splits import CASES, enumerate_candidates, fixture, winner


def prepared_fixture(case):
    x, g, h, weight, info, rows = fixture(case)
    data = NumericData(x, 100 + 7 * np.arange(len(x)), tuple(f"f{i}" for i in range(x.shape[1])))
    p = Problem(data, np.zeros((len(x), 1)), data.row_ids, weight=weight)
    cuts = tuple(
        np.arange(int(np.nanmax(col))) + 0.5 if np.any(~np.isnan(col)) else np.array([])
        for col in x.T
    )
    binned = Binning(data.feature_names, cuts).transform(data)
    fields = newton(p, g, h).add_independent("a", info[:, 0]).add_independent("b", info[:, 1])
    return binned, p, fields, rows


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("minimum", [0, 1, 2])
def test_frozen_split_rows_gains_masks_and_leaves(case, minimum):
    binned, _, fields, rows = prepared_fixture(case)
    x = binned.data.values
    expected, total = enumerate_candidates(x, fields.values, rows, minimum=minimum)
    actual = candidates(histogram(binned, fields, rows))
    assert [c.key for c in actual] == [c["key"] for c in expected]
    for c, ref in zip(actual, expected, strict=True):
        np.testing.assert_array_equal([c.left, c.right], ref["sums"])
        np.testing.assert_array_equal([c.left_count, c.right_count], ref["counts"])
        assert feasible(c) == ref["legal"]
        if ref["legal"]:
            assert score(c) == pytest.approx(ref["gain"], abs=1e-12)
        for a, e in zip(partition(binned, rows, c), ref["rows"], strict=True):
            np.testing.assert_array_equal(a, e)
    for constrained in (False, True):
        legality = (
            partial(feasible, min_information={"a": minimum, "b": minimum})
            if constrained
            else feasible
        )
        chosen, ref = choose(actual, legality=legality), winner(expected, constrained)
        assert (chosen.key if chosen else None) == (ref["key"] if ref else None)
    assert newton_leaf(total, fields.names) == pytest.approx(-total[0] / (total[1] + 1))


def test_hand_checked_d2_and_exact_tie_winners():
    binned, _, fields, rows = prepared_fixture("d2")
    expected, _ = enumerate_candidates(binned.data.values, fields.values, rows)
    assert winner(expected)["key"] == (0, 0, False)
    assert winner(expected)["gain"] == 12
    assert winner(expected, True)["key"] == (0, 1, False)
    assert winner(expected, True)["gain"] == pytest.approx(20 / 3)
    binned, _, fields, rows = prepared_fixture("ties")
    expected, _ = enumerate_candidates(binned.data.values, fields.values, rows)
    assert winner(expected)["key"] == (0, 0, False)
