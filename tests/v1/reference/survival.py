"""Log-normal event/right-censoring oracle using stdlib tails and a continued fraction."""

import math
from statistics import NormalDist

import numpy as np

from .positive import _exp, _positive, _result
from .scalar import finite_vector


def _scale(sigma):
    if not np.isscalar(sigma) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    sigma = float(sigma)
    square = sigma * sigma
    if not math.isfinite(square) or square == 0 or not math.isfinite(1 / square):
        raise ValueError("sigma geometry exceeds float64")
    return sigma


def normal_tail(z):
    """Return log survival, inverse Mills ratio, and Mills*(Mills-z).

    For z>8, Laplace's continued fraction computes the small correction to z
    directly, preserving curvature when subtracting Mills-z would cancel.
    This slow scalar oracle is checked against erfc and independent quadrature.
    """
    if not np.isscalar(z) or not np.isfinite(z):
        raise ValueError("z must be finite")
    z = float(z)
    log_phi = -z * z / 2 - 0.5 * math.log(2 * math.pi)
    if not math.isfinite(log_phi):
        raise ValueError("normal tail exceeds float64")
    if z > 8:
        correction = 0.0
        for n in range(300, 0, -1):
            correction = n / (z + correction)
        mills = z + correction
        return log_phi - math.log(mills), mills, mills * correction
    logsf = (
        math.log(math.erfc(z / math.sqrt(2)) / 2)
        if z >= 0
        else math.log1p(-math.erfc(-z / math.sqrt(2)) / 2)
    )
    mills = math.exp(log_phi - logsf)
    return logsf, mills, mills * (mills - z)


def aft(raw, lower, upper, *, sigma=1.0, weight=None):
    sigma = _scale(sigma)
    lo = _positive(lower, "lower")
    f = finite_vector(raw, "raw", len(lo))
    hi = np.asarray(upper, dtype=float)
    if hi.shape != lo.shape or np.any(np.isnan(hi)) or np.any(~((hi == lo) | np.isposinf(hi))):
        raise ValueError("only exact events and right-censored intervals are supported")
    loss, g, h = [], [], []
    for location, lower_time, upper_time in zip(f, lo, hi, strict=True):
        z = (math.log(lower_time) - location) / sigma
        if lower_time == upper_time:
            loss.append(
                math.log(lower_time) + math.log(sigma) + z * z / 2 + 0.5 * math.log(2 * math.pi)
            )
            g.append(-z / sigma)
            h.append(1 / sigma**2)
        else:
            logsf, mills, curvature = normal_tail(z)
            loss.append(-logsf)
            g.append(-mills / sigma)
            h.append(curvature / sigma**2)
    return _result(np.array(loss), np.array(g), np.array(h), weight)


def aft_predict(raw, *, sigma=1.0, times=(), probabilities=()):
    sigma = _scale(sigma)
    f = finite_vector(raw, "raw")
    times = _positive(times, "times")
    probabilities = finite_vector(probabilities, "probabilities")
    if np.any(probabilities <= 0) or np.any(probabilities >= 1):
        raise ValueError("probabilities must be strictly between zero and one")
    survival = np.array(
        [
            [math.exp(normal_tail((math.log(t) - location) / sigma)[0]) for t in times]
            for location in f
        ]
    ).reshape(len(f), len(times))
    quantile_z = np.array([NormalDist().inv_cdf(p) for p in probabilities])
    return {
        "median": _exp(f, "AFT median"),
        "mean": _exp(f + sigma**2 / 2, "AFT mean"),
        "survival": survival,
        "quantile": _exp(f[:, None] + sigma * quantile_z, "AFT quantile"),
    }
