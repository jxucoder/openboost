"""Check preregistered aggregation oracles against CPU public contracts."""

import numpy as np
import pytest

from openboost.binning import Binning
from openboost.data import NumericData, Problem
from openboost.ops import histogram
from openboost.stats import RowFields, apply_weight

from .reference.device_histogram import NAMES, ROLES, aggregate, fixture, weighted_fields


def prepared_fixture(large=False):
    codes, missing, bins, values, weight = fixture(large)
    x = codes.T.astype(float)
    x[missing.T] = np.nan
    ids = 10 + 3 * np.arange(len(x))
    data = NumericData(x, ids, tuple(f"x{i}" for i in range(x.shape[1])))
    problem = Problem(data, np.zeros((len(x), 1)), ids, weight=weight)
    binned = Binning(data.feature_names, tuple(np.arange(b - 1) + 0.5 for b in bins)).transform(data)
    return binned, problem, values


@pytest.mark.parametrize("rows", [None, [6, 0, 3, 7], [], [0, 4], [1, 6]])
def test_frozen_small_original_row_sums(rows):
    codes, missing, bins, values, weight = fixture()
    selected = np.arange(8) if rows is None else np.asarray(rows, dtype=np.int32)
    expected = aggregate(codes, missing, bins, weighted_fields(values, weight), selected)
    binned, problem, values = prepared_fixture()
    fields = apply_weight(RowFields(problem.identity, problem.data.identity, NAMES, values, ROLES), problem)
    result = histogram(binned, fields, rows)
    for actual, reference in zip((result.sums, result.counts, result.total), expected, strict=True):
        np.testing.assert_array_equal(actual, reference)
    if rows is None:
        np.testing.assert_array_equal(expected[2], [10, 9, 4, 4])
        np.testing.assert_array_equal(expected[0][0, 4], [-1, 2, 1, 1])
    if rows == [0, 4]:
        np.testing.assert_array_equal(expected[2], [0, 0, 2, 0])


@pytest.mark.parametrize("stride", [1, 3])
def test_frozen_large_original_row_sums(stride):
    codes, missing, bins, values, weight = fixture(True)
    selected = np.arange(0, 8192, stride)
    expected = aggregate(codes, missing, bins, weighted_fields(values, weight), selected)
    binned, problem, values = prepared_fixture(True)
    fields = apply_weight(RowFields(problem.identity, problem.data.identity, NAMES, values, ROLES), problem)
    result = histogram(binned, fields, selected)
    for actual, reference in zip((result.sums, result.counts, result.total), expected, strict=True):
        np.testing.assert_array_equal(actual, reference)
