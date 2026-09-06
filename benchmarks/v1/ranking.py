"""Explicit query grouping for baseline rankers; no row-to-pair weight coercion."""

import numpy as np


def groups(query, rows, weight=None):
    q = np.asarray(query)
    if q.ndim != 1 or len(q) != rows or rows == 0 or q.dtype.kind not in "iuUS":
        raise ValueError("aligned integer/string query IDs required")
    starts = np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1]
    ids = q[starts]
    if len(np.unique(ids)) != len(ids):
        raise ValueError("queries must be contiguous, not fragmented")
    sizes = np.diff(np.r_[starts, rows])
    w = np.ones(len(ids)) if weight is None else np.asarray(weight, dtype=float)
    if w.shape != (len(ids),) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("one nonnegative weight per query with positive total required")
    return ids, sizes, w


def validate(arrays, patience):
    if "weight_train" in arrays or "weight_validation" in arrays:
        raise ValueError("ranking rejects row weights; supply explicit query weights")
    train = groups(arrays["query_train"], len(arrays["x_train"]), arrays.get("query_weight_train"))
    valid = groups(
        arrays["query_validation"],
        len(arrays["x_validation"]),
        arrays.get("query_weight_validation"),
    )
    if train[0].dtype.kind != valid[0].dtype.kind or np.intersect1d(train[0], valid[0]).size:
        raise ValueError("ranking query types differ or train/validation queries overlap")
    for key in ["y_train"] + (["y_validation"] if patience is not None else []):
        y = arrays.get(key)
        if y is None or y.ndim != 1 or not np.isin(y, [0, 1, 2, 3, 4]).all():
            raise ValueError("ranking requires relevance 0..4")
    return train, valid
