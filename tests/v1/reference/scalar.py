"""Float64 scalar mathematics for tiny v1 fixtures, not a production backend."""

from numbers import Integral

import numpy as np


def finite_vector(values, name, length=None):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector")
    if length is not None and len(result) != length:
        raise ValueError(f"{name} must have length {length}")
    return result


def training_weights(weight, length):
    values = (
        np.ones(length, dtype=np.float64)
        if weight is None
        else finite_vector(weight, "weight", length)
    )
    total = float(np.sum(values))
    if np.any(values < 0) or not np.isfinite(total) or total <= 0:
        raise ValueError("weight must be nonnegative with finite positive total")
    return values


def row_ids(rows, length):
    result = tuple(rows)
    if any(
        not isinstance(i, Integral) or isinstance(i, bool) or i < 0 or i >= length for i in result
    ) or len(set(result)) != len(result):
        raise ValueError("rows must contain unique in-range integer indices")
    return tuple(sorted(int(i) for i in result))


def sum_rows(fields, rows):
    """Explicit row-wise addition in original row order, including empty sets."""
    values = np.asarray(fields, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("fields must be a finite matrix")
    total = np.zeros(values.shape[1], dtype=np.float64)
    for row in row_ids(rows, len(values)):
        total += values[row]
    if not np.all(np.isfinite(total)):
        raise ValueError("non-finite row reduction")
    return total


def weighted_mean(y, weight=None):
    target = finite_vector(y, "target")
    w = training_weights(weight, len(target))
    weighted_y, total_w = sum_rows(np.column_stack((w * target, w)), range(len(target)))
    return float(weighted_y / total_w)


def squared_error(raw, y, weight=None):
    """Weighted mean half-square loss; derivatives remain UNWEIGHTED per row."""
    target = finite_vector(y, "target")
    prediction = finite_vector(raw, "raw", len(target))
    w = training_weights(weight, len(target))
    gradient = prediction - target
    loss = float(sum_rows((w * gradient * gradient / 2)[:, None], range(len(w)))[0] / sum(w))
    return loss, gradient.copy(), np.ones(len(target), dtype=np.float64)


def row_statistics(gradient, curvature, weight=None):
    """Multiply raw derivatives by train weight exactly once at this boundary."""
    g = finite_vector(gradient, "gradient")
    h = finite_vector(curvature, "curvature", len(g))
    if np.any(h < 0):
        raise ValueError("curvature must be nonnegative")
    w = training_weights(weight, len(g))
    result = np.column_stack((w * g, w * h))
    if not np.all(np.isfinite(result)):
        raise ValueError("non-finite weighted statistics")
    return result


def nonnegative(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def newton_leaf(gradient_sum, curvature_sum, *, reg_lambda=1.0):
    regularization = nonnegative(reg_lambda, "reg_lambda")
    h = nonnegative(curvature_sum, "curvature_sum")
    denominator = h + regularization
    if not np.isfinite(gradient_sum) or not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("invalid Newton gradient or denominator")
    value = -float(gradient_sum) / denominator
    if not np.isfinite(value):
        raise ValueError("non-finite Newton leaf")
    return value


def node_score(gradient_sum, curvature_sum, *, reg_lambda=1.0):
    """Reduction in the regularized quadratic: G² / (2*(H+lambda))."""
    value = newton_leaf(gradient_sum, curvature_sum, reg_lambda=reg_lambda)
    score = -0.5 * float(gradient_sum) * value
    if not np.isfinite(score):
        raise ValueError("non-finite node score")
    return score
