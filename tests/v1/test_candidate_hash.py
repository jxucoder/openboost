"""Candidate row identity is invariant across split enumeration."""

import numpy as np
import pytest

from openboost import MixedData, Problem, ops
from openboost.binning import Binning
from openboost.stats import newton


@pytest.mark.parametrize("rows", [None, [0, 2, 3, 5]])
def test_candidate_digest_computed_once_with_exact_identity(monkeypatch, rows):
    data = MixedData(
        [[0, "a"], [1, "b"], [2, None], [3, "a"], [None, "b"], [5, "c"]],
        np.arange(6),
        ("numeric", "category"),
        ("numeric", "categorical"),
    )
    p = Problem(data, np.zeros((6, 1)), data.row_ids, weight=[1, 0, 2, 1, 3, 1])
    hist = ops.histogram(
        Binning.fit(data, bins=6).transform(data), newton(p, np.arange(6.0) - 2, np.ones(6)), rows
    )
    original = ops._identity
    expected = original(hist.rows)
    calls = []

    def counted(*parts):
        calls.append(parts)
        return original(*parts)

    monkeypatch.setattr(ops, "_identity", counted)
    candidates = ops.candidates(hist)
    assert len(candidates) > 2
    assert all(c.rows_identity == expected for c in candidates)
    assert all(c.data_identity == hist.data.identity for c in candidates)
    for candidate in candidates:
        np.testing.assert_allclose(candidate.left + candidate.right, hist.total, atol=1e-14)
        assert candidate.left_count + candidate.right_count == len(hist.rows)
    assert len(calls) == 1
    assert calls[0][0] is hist.rows
