"""Frozen 086 aggregation inputs and original-row float64 loops; no device code."""

import numpy as np

NAMES = ("gradient", "curvature", "cohort:a", "cohort:b")
ROLES = ("unweighted", "unweighted", "independent", "independent")


def fixture(large=False):
    if large:
        r, f = np.arange(8192), np.arange(32)[:, None]
        codes = ((r + 3 * f) % 32).astype(np.int32)
        missing = (r + f) % 17 == 0
        gradient, curvature, weight = (r % 13 - 6) / 8, (r % 7 + 1) / 8, (r % 5) / 4
        bins = (32,) * 32
    else:
        r = np.arange(8)
        codes = np.array([[0, 0, 1, 1, 2, 2, 3, 3], [3, 2, 1, 0, 3, 2, 1, 0]], np.int32)
        missing = np.zeros_like(codes, dtype=bool)
        missing[0, [1, 6]] = True
        missing[1, 4] = True
        gradient = [-6, 1, 1, 1, 1, 2, -2, 2]
        curvature, weight, bins = np.ones(8), [0, 1, 2, 1, 0, 3, 1, 1], (4, 4)
    values = np.column_stack((gradient, curvature, r % 2 == 0, r % 2 == 1)).astype(np.float32)
    return codes, missing, bins, values, np.asarray(weight, dtype=np.float32)


def weighted_fields(values, weight):
    result = np.zeros(values.shape, dtype=np.float64)
    for r in range(len(values)):
        for q, role in enumerate(ROLES):
            result[r, q] = float(values[r, q]) * (float(weight[r]) if role == "unweighted" else 1)
    return result


def aggregate(codes, missing, bins, values, rows):
    """Count each selected original row, separately from every additive field."""
    sums = np.zeros((len(bins), max(bins) + 1, values.shape[1]), dtype=np.float64)
    counts = np.zeros(sums.shape[:2], dtype=np.int64)
    total = np.zeros(values.shape[1], dtype=np.float64)
    for r in rows:
        total += values[r]
        for f, nbin in enumerate(bins):
            b = nbin if missing[f, r] else codes[f, r]
            counts[f, b] += 1
            sums[f, b] += values[r]
    return sums, counts, total
