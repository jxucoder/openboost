"""Tiny Normal/Formula geometry and immutable directional-update probes."""

import math
from dataclasses import dataclass

import numpy as np

from .positive import _exp, _positive, _result
from .scalar import finite_vector, nonnegative, training_weights, weighted_mean
from .tree import fit_tree, numeric_bins


def _raw(values):
    raw = np.asarray(values, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 2 or len(raw) == 0 or not np.all(np.isfinite(raw)):
        raise ValueError("raw must be a nonempty finite [N,2] matrix")
    return raw


def normal(raw, target, *, weight=None):
    raw = _raw(raw)
    y = finite_vector(target, "target", len(raw))
    mu, ell = raw.T
    precision = _exp(-2 * ell, "Normal precision")
    residual = mu - y
    square = residual**2 * precision
    loss = ell + square / 2 + 0.5 * math.log(2 * math.pi)
    g = np.column_stack((residual * precision, 1 - square))
    fisher = np.zeros((len(raw), 2, 2))
    fisher[:, 0, 0], fisher[:, 1, 1] = precision, 2
    return _result(loss, g, fisher, weight)


def normal_base(target, *, minimum_scale, weight=None):
    floor = _positive([minimum_scale], "minimum_scale")[0]
    y = finite_vector(target, "target")
    mu = weighted_mean(y, weight)
    w = training_weights(weight, len(y))
    scale = math.sqrt(sum((w / sum(w)) * (y - mu) ** 2))
    if not math.isfinite(scale):
        raise ValueError("Normal scale exceeds float64")
    return (mu, math.log(max(scale, floor)))


def normal_scores(raw, target, *, weight=None):
    """Separate evaluator from the objective: direct density and closed-form CRPS."""
    raw = _raw(raw)
    y = finite_vector(target, "target", len(raw))
    w = training_weights(weight, len(y))
    nll, crps = [], []
    for (mu, ell), value in zip(raw, y, strict=True):
        sigma = float(_exp(ell, "Normal scale"))
        z = (value - mu) / sigma
        phi = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
        cdf = 0.5 * math.erfc(-z / math.sqrt(2))
        nll.append(math.log(sigma * math.sqrt(2 * math.pi)) + z * z / 2)
        crps.append(sigma * (z * (2 * cdf - 1) + 2 * phi - 1 / math.sqrt(math.pi)))
    if not np.all(np.isfinite(nll)) or not np.all(np.isfinite(crps)):
        raise ValueError("non-finite Normal score")
    return float(np.dot(w / sum(w), nll)), float(np.dot(w / sum(w), crps))


def _softplus_inverse(value):
    return value + np.log(-np.expm1(-value))


def formula_base(target, *, weight=None):
    a = max(weighted_mean(target, weight), 1e-6)
    return (float(_softplus_inverse(a)), float(_softplus_inverse(1.0)))


def formula_predict(raw, x):
    raw = _raw(raw)
    x = _positive(x, "structure x", len(raw))
    a, b = np.logaddexp(0, raw).T
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("formula parameters underflowed outside positive support")
    prediction = a * (-np.expm1(-b * x))
    if not np.all(np.isfinite(prediction)):
        raise ValueError("non-finite formula prediction")
    return prediction, a, b


def formula(raw, target, x, *, weight=None):
    raw = _raw(raw)
    y = finite_vector(target, "target", len(raw))
    x = _positive(x, "structure x", len(raw))
    prediction, a, b = formula_predict(raw, x)
    tail = np.exp(-np.abs(raw))
    sigmoid = np.where(raw >= 0, 1 / (1 + tail), tail / (1 + tail))
    jacobian = np.column_stack(
        (sigmoid[:, 0] * (-np.expm1(-b * x)), a * x * np.exp(-b * x) * sigmoid[:, 1])
    )
    residual = prediction - y
    g = jacobian * residual[:, None]
    ggn = np.array([np.outer(row, row) for row in jacobian])
    return _result(residual**2 / 2, g, ggn, weight)


def directions(gradient, metric, *, mode="full", damping=0.0):
    g = _raw(gradient)
    metric = np.asarray(metric, dtype=float)
    damping = nonnegative(damping, "damping")
    if metric.shape != (len(g), 2, 2) or not np.all(np.isfinite(metric)):
        raise ValueError("metric must be finite [N,2,2]")
    if mode == "ordinary":
        return -g.copy()
    if mode == "diagonal":
        diagonal = np.diagonal(metric, axis1=1, axis2=2) + damping
        if np.any(diagonal <= 0):
            raise ValueError("nonpositive diagonal solve")
        result = -g / diagonal
    elif mode == "full":
        result = []
        # Explicit 2x2 inversion, independent of production linear algebra.
        for row, m in zip(g, metric, strict=True):
            a, b, c, d = m[0, 0] + damping, m[0, 1], m[1, 0], m[1, 1] + damping
            determinant = a * d - b * c
            if (
                not np.isfinite(determinant)
                or b != c
                or a <= 0
                or determinant <= np.finfo(float).eps * max(abs(a * d), abs(b * c))
            ):
                raise ValueError("metric is not numerically positive definite; specify damping")
            result.append(
                [(-d * row[0] + b * row[1]) / determinant, (c * row[0] - a * row[1]) / determinant]
            )
        result = np.array(result)
    else:
        raise ValueError("unknown direction mode")
    if not np.all(np.isfinite(result)):
        raise ValueError("non-finite solve")
    return result


@dataclass(frozen=True)
class Update:
    raw_before: tuple
    raw_after: tuple
    loss_before: float
    loss_after: float
    accepted: bool
    terms: tuple
    trials: tuple


def step(
    bins,
    raw,
    target,
    objective,
    *,
    mode="full",
    damping=0.0,
    weight=None,
    channels=(0, 1),
    rates=(1.0, 0.5, 0.1),
    require_decrease=True,
):
    """Fit directions once, try finite coefficients, commit one immutable snapshot.

    Call with one channel then the other for ordered updates. This is a small
    reference probe, not the planned public transaction/runtime implementation.
    """
    raw = _raw(raw).copy()
    bins = numeric_bins(bins)
    if len(raw) != len(bins):
        raise ValueError("bins/raw alignment mismatch")
    channels, rates = tuple(channels), tuple(rates)
    if (
        not channels
        or len(set(channels)) != len(channels)
        or any(c not in (0, 1) for c in channels)
    ):
        raise ValueError("channels must be a unique nonempty subset of (0,1)")
    if not rates or any(not np.isscalar(r) or not np.isfinite(r) or r <= 0 for r in rates):
        raise ValueError("rates must be finite positive candidates")
    before = tuple(tuple(row) for row in raw)
    loss, g, metric = objective(raw, target, weight=weight)
    direction = directions(g, metric, mode=mode, damping=damping)
    trees = [
        (k, fit_tree(bins, -direction[:, k], np.ones(len(raw)), weight=weight)) for k in channels
    ]
    delta = np.zeros_like(raw)
    for k, tree in trees:
        delta[:, k] = tree.predict(bins)
    trials = []
    for coefficient in rates:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                candidate = raw + coefficient * delta
                candidate_loss = objective(candidate, target, weight=weight)[0]
            if not np.isfinite(candidate_loss):
                raise ValueError("non-finite candidate loss")
        except (ValueError, FloatingPointError, OverflowError) as error:
            trials.append((float(coefficient), None, type(error).__name__))
            continue
        trials.append((float(coefficient), float(candidate_loss), "evaluated"))
        if not require_decrease or candidate_loss < loss:
            return Update(
                before,
                tuple(tuple(row) for row in candidate),
                loss,
                candidate_loss,
                True,
                tuple((k, tree, float(coefficient)) for k, tree in trees),
                tuple(trials),
            )
    return Update(before, before, loss, loss, False, (), tuple(trials))
