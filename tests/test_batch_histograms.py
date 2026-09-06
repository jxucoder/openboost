"""Independent sample-sum oracle for the experimental histogram contract."""

import numpy as np
import pytest

from openboost.experimental import build_histograms


def oracle(bins, grad, hess, ids, active):
    shape = (len(active), bins.shape[0], 256)
    g, h = np.zeros(shape), np.zeros(shape)
    counts = np.zeros(len(active), dtype=np.int32)
    for i, node in enumerate(ids):
        if node == -1 or not active[node]:
            continue
        counts[node] += 1
        for f in range(bins.shape[0]):
            g[node, f, bins[f, i]] += float(grad[i])
            h[node, f, bins[f, i]] += float(hess[i])
    return g, h, counts


def fixture():
    bins = np.array([[0, 1, 255, 1, 2, 0], [3, 3, 3, 3, 3, 3]], dtype=np.uint8)
    weights = np.array([1, 0, 2, .5, 3, 1], dtype=np.float32)
    grad = np.array([2, 9, -1, 4, 5, 8], dtype=np.float32) * weights
    hess = np.array([1, 3, 2, 1, 4, 2], dtype=np.float32) * weights
    ids = np.array([0, 0, 2, 2, 1, -1], dtype=np.int32)
    active = np.array([True, False, True, True])
    return bins, grad, hess, ids, active


def test_sample_sum_oracle():
    args = fixture()
    expected = oracle(*args)
    batch = build_histograms(*args)
    for actual, wanted in zip((batch.grad, batch.hess, batch.counts), expected, strict=True):
        np.testing.assert_array_equal(actual, wanted)
    assert batch.grad.dtype == batch.hess.dtype == np.float32
    assert batch.counts.tolist() == [2, 0, 2, 0]  # zero weight still counts
    assert batch.grad[2, 0, 255] == -2
    assert not np.shares_memory(batch.active, args[-1])


def test_empty_samples_and_exact_budget():
    args = (np.zeros((2, 0), np.uint8), np.zeros(0, np.float32), np.zeros(0, np.float32),
            np.zeros(0, np.int32), np.ones(3, bool))
    required = 3 * 2 * 256 * 8 + 3 * 5
    result = build_histograms(*args, memory_budget_bytes=required)
    assert result.nbytes == required
    assert not result.counts.any() and not result.grad.any()
    with pytest.raises(MemoryError):
        build_histograms(*args, memory_budget_bytes=required - 1)


@pytest.mark.parametrize('index,value', [
    (1, np.ones(6, np.float64)), (2, np.full(6, -1, np.float32)),
    (1, np.full(6, np.nan, np.float32)), (3, np.full(6, 4, np.int32)),
    (3, np.full(6, -2, np.int32)), (4, np.ones(4, np.int32)),
    (1, np.ones(12, np.float32)[::2]),
])
def test_invalid_input(index, value):
    args = list(fixture())
    args[index] = value
    with pytest.raises((ValueError, TypeError)):
        build_histograms(*args)


def test_zero_curvature_and_overflow():
    args = list(fixture())
    args[1] = np.zeros(6, np.float32)
    args[2] = np.zeros(6, np.float32)
    batch = build_histograms(*args)
    assert batch.counts.sum() == 4
    assert not batch.grad.any() and not batch.hess.any()
    args[1] = np.full(6, np.finfo(np.float32).max, np.float32)
    with pytest.raises(ValueError, match='overflow'):
        build_histograms(*args)
