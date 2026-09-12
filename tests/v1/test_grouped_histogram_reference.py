"""Hand-derived controls of the independent stored-row grouping oracle."""

import numpy as np
import pytest

from .reference.grouped_histogram import finite, histogram, stored_sum


@pytest.mark.parametrize('rows,expected', [([0, 1, 2], 0), ([0, 2, 1], 1), ([], 0)])
def test_row_sequence_preserves_stored_float32_rounding(rows, expected):
    values = np.array([2**24, 1, -2**24], np.float32)
    assert stored_sum(values[rows]) == expected


def test_original_row_buckets_missingness_padding_and_totals():
    codes = np.array([[0, 1, 0, 0], [0, 2, 1, 0]], np.int32)
    missing = np.array([[False, False, True, False], [False, False, False, True]])
    values = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], np.float32)
    sums, counts, total = histogram(codes, missing, (2, 3), values, [3, 1, 2])
    np.testing.assert_array_equal(counts, [[1, 1, 1, 0], [0, 1, 1, 1]])
    np.testing.assert_array_equal(sums[..., 0], [[4, 2, 3, 0], [0, 3, 2, 4]])
    np.testing.assert_array_equal(sums[..., 1], [[40, 20, 30, 0], [0, 30, 20, 40]])
    np.testing.assert_array_equal(total, [9, 90])
    assert finite((sums, counts, total))


@pytest.mark.parametrize('kind', ['bucket', 'total'])
def test_finite_inputs_can_overflow_one_independent_output(kind):
    codes = np.array([[0, 0, 1, 1]], np.int32)
    values = (np.array([3e38, 3e38, -3e38, -3e38], np.float32) if kind == 'bucket'
              else np.array([2e38, -1e38, 2e38, -1e38], np.float32))
    rows = [0, 2, 1, 3]
    result = histogram(codes, np.zeros_like(codes, bool), (2,), values[:, None], rows)
    assert not finite(result)
    assert np.isfinite(result[2]).all() == (kind == 'bucket')
    assert np.isfinite(result[0]).all() == (kind == 'total')
