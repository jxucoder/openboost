"""Independent bucket-list oracle; no production imports or CUDA emulation."""

import numpy as np


def stored_sum(values):
    # A double exactly represents the sum of two finite stored float32 values.
    # Round after each selected row, as declared by the resident reduction API.
    result = np.float32(0)
    with np.errstate(over='ignore', invalid='ignore'):
        for value in values:
            result = np.float32(float(result) + float(np.float32(value)))
    return result


def histogram(codes, missing, bins, values, rows):
    values = np.asarray(values, np.float32)
    rows = list(rows)
    sums = np.zeros((len(bins), max(bins) + 1, values.shape[1]), np.float32)
    counts = np.zeros(sums.shape[:2], np.int64)
    for feature, size in enumerate(bins):
        buckets = [[] for _ in range(size + 1)]
        for row in rows:
            buckets[size if missing[feature, row] else codes[feature, row]].append(row)
        for code, members in enumerate(buckets):
            counts[feature, code] = len(members)
            for column in range(values.shape[1]):
                sums[feature, code, column] = stored_sum(values[members, column])
    total = np.array([stored_sum(values[rows, q]) for q in range(values.shape[1])], np.float32)
    return sums, counts, total


def finite(result):
    return all(np.isfinite(x).all() for x in (result[0], result[2]))
