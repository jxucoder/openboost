"""Exhaustive row-mask split oracle, independent of production histograms."""

import numpy as np
import pytest

from openboost.experimental import build_histograms, find_splits, partition


def example():
    bins = np.array([[0, 0, 1, 1, 2, 2, 3, 3], [0, 0, 1, 1, 2, 2, 3, 3]], np.uint8)
    weights = np.array([1, 0, 2, 1, 0.5, 1, 3, 1], np.float32)
    g = np.array([4, 9, 2, 2, -1, -1, -5, -5], np.float32) * weights
    return bins, g, weights, np.zeros(8, np.int32), np.array([True] + [False] * 6)


def exhaustive(bins, g, h, ids, active, reg_lambda=1.0, min_child_weight=1.0, min_gain=0.0):
    answer = []
    for node in range(len(active)):
        best = (-1, -1, 0.0)
        rows = ids == node
        if active[node] and 2 * node + 2 < len(active):
            G, H = sum(float(x) for x in g[rows]), sum(float(x) for x in h[rows])
            if H > 0:
                for f in range(len(bins)):
                    for threshold in range(255):
                        left = rows & (bins[f] <= threshold)
                        right = rows & ~left
                        GL, HL = sum(float(x) for x in g[left]), sum(float(x) for x in h[left])
                        GR, HR = sum(float(x) for x in g[right]), sum(float(x) for x in h[right])
                        if HL <= 0 or HR <= 0 or min(HL, HR) < min_child_weight:
                            continue
                        gain = (
                            GL**2 / (HL + reg_lambda)
                            + GR**2 / (HR + reg_lambda)
                            - G**2 / (H + reg_lambda)
                        )
                        if gain > best[2] and gain >= min_gain:
                            best = (f, threshold, gain)
        answer.append(best)
    return answer


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"reg_lambda": 0.0, "min_child_weight": 0.0},
        {"min_child_weight": 20.0},
        {"min_gain": 1e6},
    ],
)
def test_exhaustive_split_oracle(kwargs):
    args = example()
    splits = find_splits(build_histograms(*args), **kwargs)
    expected = exhaustive(*args, **kwargs)
    np.testing.assert_array_equal(splits.feature, [x[0] for x in expected])
    np.testing.assert_array_equal(splits.threshold, [x[1] for x in expected])
    np.testing.assert_allclose(splits.gain, [x[2] for x in expected], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(splits.valid, [x[0] >= 0 for x in expected])
    assert splits.feature[0] in (-1, 0)  # duplicate feature: first feature wins


def test_exact_gain_boundary_and_threshold_tie():
    bins = np.array([[0, 0, 2, 2]], np.uint8)
    g = np.array([2, 2, -2, -2], np.float32)
    hist = build_histograms(
        bins, g, np.ones(4, np.float32), np.zeros(4, np.int32), np.array([True, False, False])
    )
    result = find_splits(hist, reg_lambda=0.0, min_gain=16.0, min_child_weight=2.0)
    assert result.gain[0] == 16.0 and result.threshold[0] == 0 and result.valid[0]
    assert not find_splits(hist, reg_lambda=0.0, min_gain=np.nextafter(16.0, np.inf)).valid.any()


def test_routed_rows_make_real_child_histograms():
    bins, g, h, ids, active = example()
    ids[-1] = -1
    hist = build_histograms(bins, g, h, ids, active)
    splits = find_splits(hist)
    original = ids.copy()
    routed = partition(bins, ids, splits)
    expected = ids.copy()
    for i, node in enumerate(ids):
        if node >= 0 and splits.valid[node]:
            expected[i] = 2 * node + (
                1 if bins[splits.feature[node], i] <= splits.threshold[node] else 2
            )
    np.testing.assert_array_equal(routed, expected)
    np.testing.assert_array_equal(ids, original)
    assert not np.shares_memory(routed, ids)
    next_active = np.array([False, True, True, False, False, False, False])
    children = build_histograms(bins, g, h, routed, next_active)
    for node in (1, 2):
        assert children.counts[node] == (expected == node).sum()
        assert children.grad[node, 0].sum() == g[expected == node].sum()
        assert children.hess[node, 0].sum() == h[expected == node].sum()
    second = find_splits(children)
    expected_second = exhaustive(bins, g, h, expected, next_active)
    np.testing.assert_array_equal(second.feature, [x[0] for x in expected_second])
    np.testing.assert_array_equal(second.threshold, [x[1] for x in expected_second])


@pytest.mark.parametrize("kind", ["constant", "zero", "terminal", "inactive"])
def test_no_legal_split(kind):
    bins, g, h, ids, active = example()
    if kind == "constant":
        bins[:] = 2
    if kind == "zero":
        h[:] = 0
    if kind == "terminal":
        ids[:] = 6
        active[:] = False
        active[6] = True
    if kind == "inactive":
        active[:] = False
    result = find_splits(build_histograms(bins, g, h, ids, active), min_child_weight=0.0)
    assert not result.valid.any()
    assert (result.feature == -1).all() and (result.left_child == -1).all()
    np.testing.assert_array_equal(partition(bins, ids, result), ids)


def test_missing_and_invalid_routing_rejected():
    args = list(example())
    hist = build_histograms(*args)
    splits = find_splits(hist)
    bad_ids = np.full(8, 99, np.int32)
    with pytest.raises(ValueError):
        partition(args[0], bad_ids, splits)
    splits.left_child[0] = 4
    with pytest.raises(ValueError):
        partition(args[0], args[3], splits)
    args[0][0, 0] = 255
    with pytest.raises(ValueError, match="missing"):
        find_splits(build_histograms(*args))
    with pytest.raises(ValueError, match="missing"):
        partition(args[0], args[3], find_splits(hist))


@pytest.mark.parametrize(
    "kwargs", [{"reg_lambda": -1}, {"min_gain": np.inf}, {"min_child_weight": np.nan}]
)
def test_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        find_splits(build_histograms(*example()), **kwargs)


def test_empty_partition_and_negative_gain():
    args = example()
    hist = build_histograms(args[0][:, :0].copy(), args[1][:0], args[2][:0], args[3][:0], args[4])
    empty = find_splits(hist)
    assert partition(args[0][:, :0].copy(), args[3][:0], empty).shape == (0,)
    # Equal gradients in every row with L2: splitting has negative gain.
    hist = build_histograms(
        args[0], np.ones(8, np.float32), np.ones(8, np.float32), args[3], args[4]
    )
    assert not find_splits(hist).valid.any()
