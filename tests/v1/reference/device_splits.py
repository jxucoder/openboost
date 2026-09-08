"""Frozen 087 cases; exhaustive original-row math, independent of device kernels."""

import numpy as np

from .device_histogram import fixture as histogram_fixture

CASES = (
    "d2",
    "no_feasible",
    "missing_weighted",
    "ties",
    "zero_gradient",
    "zero_curvature",
    "missing_only",
    "inactive",
    "empty",
    "one_bin",
)


def fixture(case):
    if case == "missing_weighted":
        codes, missing, _, values, weight = histogram_fixture()
        x = codes.T.astype(float)
        x[missing.T] = np.nan
        return x, values[:, 0], values[:, 1], weight, values[:, 2:], np.array([6, 0, 3, 7])
    if case in ("d2", "no_feasible", "empty"):
        x = np.arange(6, dtype=float)[:, None]
        g = np.array([-6, 1, 1, 1, 1, 2.0])
        info = np.eye(2)[np.arange(6) % 2]
        if case == "no_feasible":
            x = np.array([0, 0, 0, 1, 1, 1.0])[:, None]
            info = np.eye(2)[np.arange(6) // 3]
    else:
        x = np.array([[0, 0], [0, 0], [1, 1], [1, 1.0]])
        g = np.array([-2, -2, 2, 2.0])
        info = np.eye(2)[np.arange(4) % 2]
        if case == "missing_only":
            x = np.array([0, 0, np.nan, np.nan])[:, None]
        elif case == "inactive":
            x = np.array([[0, np.nan], [0, np.nan], [2, np.nan], [2, np.nan]])
        elif case == "one_bin":
            x = np.zeros((4, 1))
        elif case == "zero_gradient":
            g[:] = 0
    h = np.zeros(len(g)) if case == "zero_curvature" else np.ones(len(g))
    return (
        x,
        g,
        h,
        np.ones(len(g)),
        info,
        np.array([], dtype=int) if case == "empty" else np.arange(len(g)),
    )


def enumerate_candidates(
    x, fields, rows, *, reg_lambda=1.0, split_penalty=0.0, minimum=1.0, min_h=0.0
):
    """Enumerate actual conditions and row partitions, without histogram reductions."""
    result = []
    parent = np.zeros(fields.shape[1], dtype=float)
    for r in rows:
        parent += fields[r]
    for f in range(x.shape[1]):
        thresholds = sorted(set(x[~np.isnan(x[:, f]), f]))
        for threshold in thresholds:
            for missing_left in (False, True):
                left, right = [], []
                for r in rows:
                    goes_left = missing_left if np.isnan(x[r, f]) else x[r, f] <= threshold
                    (left if goes_left else right).append(int(r))
                sums = np.zeros((2, fields.shape[1]), dtype=float)
                for side, positions in enumerate((left, right)):
                    for r in positions:
                        sums[side] += fields[r]
                legal = bool(left and right and min(sums[:, 1]) > 0 and min(sums[:, 1]) >= min_h)
                gain = 0.0
                if legal:
                    gain = (
                        sums[0, 0] ** 2 / (2 * (sums[0, 1] + reg_lambda))
                        + sums[1, 0] ** 2 / (2 * (sums[1, 1] + reg_lambda))
                        - parent[0] ** 2 / (2 * (parent[1] + reg_lambda))
                        - split_penalty
                    )
                result.append(
                    dict(
                        key=(f, int(threshold), missing_left),
                        sums=sums,
                        counts=np.array([len(left), len(right)]),
                        rows=(left, right),
                        gain=gain,
                        legal=legal,
                        information=bool(np.all(sums[:, 2:] >= minimum)),
                    )
                )
    return result, parent


def winner(options, constrained=False):
    allowed = [
        c for c in options if c["legal"] and (not constrained or c["information"]) and c["gain"] > 0
    ]
    return min(allowed, key=lambda c: (-c["gain"], c["key"])) if allowed else None
