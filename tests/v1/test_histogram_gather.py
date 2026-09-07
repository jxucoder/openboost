"""Exact histogram semantics while reusing feature-independent row gathers."""

import numpy as np
import pytest

from openboost import MixedData, Problem
from openboost.binning import Binning
from openboost.ops import histogram
from openboost.stats import vector_newton


@pytest.mark.parametrize("rows", [None, [5, 2, 0, 3], []])
def test_histogram_reuses_contiguous_columns_with_exact_statistics(monkeypatch, rows):
    data = MixedData(
        [[0, "a"], [1, "b"], [2, None], [3, "a"], [None, "b"], [5, "c"]],
        np.arange(6),
        ("number", "category"),
        ("numeric", "categorical"),
    )
    problem = Problem(data, np.zeros((6, 3)), data.row_ids, raw_width=3, weight=[1, 0, 2, 1, 3, 1])
    gradient = np.array(
        [[1e10, -1, 3], [2, 0, 4], [-1e10, 4, -2], [1e-8, -2, 8], [5, 8, 0], [-7, 3, 1.0]]
    )
    fields = vector_newton(problem, gradient, np.ones((6, 3))).add_independent(
        "cohort", np.arange(6.0) % 2
    )
    binned = Binning.fit(data, bins=6).transform(data)
    selected = np.arange(6) if rows is None else np.asarray(rows, dtype=int)
    original = np.bincount
    buffers = []

    def counted(codes, weights=None, **kwargs):
        if weights is not None:
            buffers.append(weights)
        return original(codes, weights=weights, **kwargs)

    monkeypatch.setattr(np, "bincount", counted)
    actual = histogram(binned, fields, rows)
    for f, bins in enumerate(binned.binning.bin_counts):
        codes = np.where(binned.missing[f, selected], bins, binned.codes[f, selected])
        expected = np.column_stack(
            [original(codes, weights=c[selected], minlength=bins + 1) for c in fields.values.T]
        )
        np.testing.assert_array_equal(actual.sums[f], expected)
        np.testing.assert_array_equal(actual.counts[f], original(codes, minlength=bins + 1))
    np.testing.assert_array_equal(actual.total, fields.values[selected].sum(axis=0))
    assert all(a.flags.c_contiguous for a in buffers)
    width = fields.values.shape[1]
    assert len(buffers) == 2 * width
    # Every feature reuses the same memory for each selected statistic column.
    assert all(
        a.ctypes.data == b.ctypes.data
        for a, b in zip(buffers[:width], buffers[width:], strict=True)
    )
