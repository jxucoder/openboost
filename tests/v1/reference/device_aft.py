"""Independent Decimal AFT mathematics at explicit float32 storage boundaries."""

import math
from decimal import Decimal, localcontext
from functools import cache

import numpy as np

from .device_glm import stored
from .device_rounds import predict, tree

RTOL, ATOL = 1e-4, 1e-5
GEOMETRY_RTOL, GEOMETRY_ATOL, METRIC_SCALE = 1e-6, 1e-8, 1e-3


@cache
def pi(precision):
    # Machin's identity, independent of float64 Normal constants in production.
    with localcontext() as ctx:
        ctx.prec = precision + 15
        def atan_inverse(n):
            x = Decimal(1) / n
            term, total, k = x, x, 1
            while True:
                term *= -x * x
                updated = total + term / (2 * k + 1)
                if updated == total:
                    return total
                total, k = updated, k + 1
        return 16 * atan_inverse(5) - 4 * atan_inverse(239)


def tail(z, *, precision=100):
    """Log survival, Mills ratio and curvature from central integration / fraction."""
    with localcontext() as ctx:
        # Central integration and log(1-small_tail) lose significant digits.
        # Keep guard digits rather than relaxing the cross-precision check.
        ctx.prec = precision + 80
        z = z if isinstance(z, Decimal) else Decimal(float(z))
        normalizer = (2 * pi(ctx.prec)).sqrt()
        log_phi = -z * z / 2 - normalizer.ln()
        if abs(z) > 8:
            x, correction = abs(z), Decimal(0)
            for k in range(500, 0, -1):
                correction = Decimal(k) / (x + correction)
            mills = x + correction
            positive_logsf = log_phi - mills.ln()
            if z > 0:
                return positive_logsf, mills, mills * correction
            logsf = (1 - positive_logsf.exp()).ln()
        else:
            term, integral, k = z, z, 0
            while True:
                term *= -z * z * (2 * k + 1) / (2 * (k + 1) * (2 * k + 3))
                updated = integral + term
                if updated == integral:
                    break
                integral, k = updated, k + 1
            logsf = (Decimal("0.5") - integral / normalizer).ln()
        mills = (log_phi - logsf).exp()
        return logsf, mills, mills * (mills - z)


def row(location, lower, event, sigma):
    with localcontext() as ctx:
        ctx.prec = 100
        f, t, s = (Decimal(float(v)) for v in (location, lower, sigma))
        log_time = t.ln()
        z = (log_time - f) / s
        if event:
            return (log_time + s.ln() + z * z / 2 + (2 * pi(100)).ln() / 2,
                    -z / s, 1 / (s * s))
        logsf, mills, curvature = tail(z)
        return -logsf, -mills / s, curvature / (s * s)


def geometry(raw, lower, event, offset, weight, sigma):
    raw, lower, offset, weight = (stored(v) for v in (raw, lower, offset, weight))
    if np.any(lower <= 0):
        raise ValueError("positive stored lower time required")
    losses, gradient, curvature = [], [], []
    for f, t, e, o in zip(raw, lower, event, offset, strict=True):
        loss, g, h = row(f + o, t, e, sigma)
        g32, h32 = float(stored(float(g))), float(stored(float(h)))
        if h32 <= 0 or (not e and g32 == 0) or not math.isfinite(float(loss)):
            raise ValueError("representable float32 AFT geometry required")
        losses.append(loss)
        gradient.append(g32)
        curvature.append(h32)
    with localcontext() as ctx:
        ctx.prec = 100
        weights = [Decimal(float(w)) for w in weight]
        total = sum(w * loss for w, loss in zip(weights, losses, strict=True)) / sum(weights)
    return float(total), np.array(gradient), np.array(curvature)


def base(lower, offset, weight):
    lower, offset, weight = (stored(v) for v in (lower, offset, weight))
    with localcontext() as ctx:
        ctx.prec = 100
        total, mass = Decimal(0), Decimal(0)
        for t, o, w in zip(lower, offset, weight, strict=True):
            t, o, w = (Decimal(float(v)) for v in (t, o, w))
            total += w * (t.ln() - o)
            mass += w
        return float(stored(float(total / mass)))


def rounds(train, validation, *, sigma, depth=1, count=2, rate=0.25):
    def value(data, raw):
        return geometry(raw, data["lower"], data["event"], data["offset"], data["weight"], sigma)
    initial = base(train["lower"], train["offset"], train["weight"])
    raw, valid_raw = np.full(len(train["lower"]), initial), np.full(len(validation["lower"]), initial)
    steps = []
    for _ in range(count):
        before, loss = raw.copy(), value(train, raw)
        _, g, h = loss
        fields = stored(np.column_stack((g, h)) * stored(train["weight"])[:, None])
        nodes = tree(train["x"], fields, depth)
        train_delta = stored(rate * stored(predict(nodes, train["x"])))
        valid_delta = stored(rate * stored(predict(nodes, validation["x"])))
        raw, valid_raw = stored(raw + train_delta), stored(valid_raw + valid_delta)
        steps.append(dict(before=before, gradient=g, curvature=h, fields=fields, nodes=nodes,
                          raw=raw.copy(), validation_raw=valid_raw.copy(),
                          loss=value(train, raw)[0], score=value(validation, valid_raw)[0]))
    return initial, steps


# name, location, lower time, exact event, sigma, supported
DOMAIN_CASES = (
    ("stationary_event", 0, 1, True, 1, True),
    ("central_censor", 0, 1, False, 1, True),
    ("positive_tail", -1000, 1, False, 1, True),
    ("large_positive_tail", -1e8, 1, False, 1, True),
    ("negative_tail", 12, 1, False, 1, True),
    ("subnormal_tail", 14, 1, False, 1, True),
    ("lost_gradient", 14.4, 1, False, 1, False),
    ("lost_curvature", 15, 1, False, 1, False),
    ("gradient_overflow", -1e38, 1, False, 0.25, False),
    ("curvature_overflow", 0, 1, True, 1e-20, False),
    ("subnormal_curvature", 0, 1, True, 1e20, True),
    ("curvature_underflow", 0, 1, True, 1e23, False),
    ("lost_time", 0, 1e-60, True, 1, False),
    ("time_overflow", 0, 1e40, False, 1, False),
)
