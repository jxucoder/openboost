"""Independent 107 convex-bound prototype and direct Decimal likelihood oracle."""

from decimal import Decimal, localcontext

import numpy as np

from .device_glm import geometry, stored
from .normal_comparison import ONE, TWO, ZERO, Interval, UnresolvedArithmetic, _result, exponential


def exp_bound(value):
    if value.lower < -256 or value.upper > 256:
        raise UnresolvedArithmetic("exponent_range")
    reduced = exponential(value / TWO / TWO / TWO)
    squared = reduced * reduced
    fourth = squared * squared
    return fourth * fourth


def sigmoid(point):
    if point == 0:
        return Interval.point(0.5)
    tail = exp_bound(Interval.point(-abs(point)))
    return (ONE if point > 0 else tail) / (ONE + tail)


def curvature(distance):
    if distance == 0:
        return Interval.point(0.25)
    tail = exp_bound(Interval.point(-distance))
    denominator = ONE + tail
    return tail / (denominator * denominator)


def row_change(family, old, new, y, offset, exposure):
    if old == new:
        return ZERO
    old_raw, new_raw = (
        Interval.point(old) + Interval.point(offset),
        Interval.point(new) + Interval.point(offset),
    )
    delta = Interval.point(new) - Interval.point(old)
    if family == "binary":
        sign = Interval.point(1 - 2 * y)
        old_raw, new_raw, delta = old_raw * sign, new_raw * sign, delta * sign
    left, right = min(old_raw.lower, new_raw.lower), max(old_raw.upper, new_raw.upper)
    if family == "binary":
        gradient = Interval(sigmoid(old_raw.lower).lower, sigmoid(old_raw.upper).upper)
        far = max(abs(left), abs(right))
        near = 0 if left <= 0 <= right else min(abs(left), abs(right))
        h = Interval(curvature(far).lower, min(0.25, curvature(near).upper))
    else:
        e = Interval.point(exposure)
        gradient = e * exp_bound(old_raw) - Interval.point(y)
        h = e * exp_bound(Interval(left, right))
    return gradient * delta + h * (delta * delta) / TWO


def compare(family, before, after, target, offset, weight, exposure):
    arrays = tuple(stored(a) for a in (before, after, target, offset, weight, exposure))
    old, new, y, o, w, e = arrays
    if any(a.shape != old.shape or a.ndim != 1 for a in arrays) or old.size == 0:
        raise ValueError("aligned scalar rows required")
    for raw in (old, new):
        geometry(family, raw, y, o, w, e)
    try:
        total, mass = ZERO, ZERO
        for row in zip(old, new, y, o, w, e, strict=True):
            a, b, label, off, mass_value, exp = row
            total += row_change(family, a, b, label, off, exp) * Interval.point(mass_value)
            mass += Interval.point(mass_value)
        return _result(total / mass, unchanged=np.array_equal(old, new))
    except UnresolvedArithmetic as error:
        from .normal_comparison import Comparison

        return Comparison("unresolved", str(error), None, None, None, None)


def direct_difference(family, before, after, target, offset, weight, exposure, *, precision=220):
    """Subtract independent arbitrary-precision likelihoods, with constants cancelling."""
    with localcontext() as context:
        context.prec = precision
        old_total, new_total, mass = Decimal(0), Decimal(0), Decimal(0)
        for row in zip(before, after, target, offset, weight, exposure, strict=True):
            old, new, y, o, w, e = (Decimal(float(np.float32(v))) for v in row)
            values = []
            for raw in (old, new):
                effective = raw + o
                if family == "binary":
                    margin = (1 - 2 * y) * effective
                    value = (1 + margin.exp()).ln()
                else:
                    # Fixed log-factorial is omitted from both full likelihoods.
                    value = e * effective.exp() - y * (effective + e.ln())
                values.append(value)
            old_total += w * values[0]
            new_total += w * values[1]
            mass += w
        return (new_total - old_total) / mass


def cases():
    result = []

    def add(name, family, old, new, y, *, offset=None, weight=None, exposure=None, status=None):
        n = len(y)
        arrays = tuple(
            np.asarray(v, np.float32).tolist()
            for v in (
                old,
                new,
                y,
                [0] * n if offset is None else offset,
                [1] * n if weight is None else weight,
                [1] * n if exposure is None else exposure,
            )
        )
        result.append(dict(id=family + "/" + name, family=family, arrays=arrays, status=status))

    for family in ("binary", "poisson"):
        add("unchanged", family, [0, 1], [0, 1], [0, 1], status="unchanged")
        add(
            "tiny-improvement",
            family,
            [0],
            [1e-20],
            [1 if family == "binary" else 2],
            status="improvement",
        )
        add("worsening", family, [0], [-0.1], [1 if family == "binary" else 2], status="worsening")
        add(
            "zero-weight-change",
            family,
            [0, 0],
            [0, 0.5],
            [0, 1],
            weight=[1, 0],
            status="unresolved",
        )
        add(
            "offset-exposure",
            family,
            [-1, 2, -0.5],
            [-0.9, 1.8, -0.75],
            [0, 1, 1],
            offset=[0.5, -0.25, 0.125],
            exposure=[0.5, 2, 3],
            weight=[2, 0, 3],
        )
        add("large-step", family, [-2, 1], [2, -1], [0, 1])
    add("positive-tail", "binary", [80], [80.01], [1], status="improvement")
    add("negative-tail", "binary", [-80], [-80.01], [0], status="improvement")
    add("equal-loss-different-raw", "binary", [-1, 1], [1, -1], [0, 0], status="unresolved")
    add("balanced-stationary", "binary", [0, 0], [1e-20, 1e-20], [0, 1], status="unresolved")
    add("stationary-rounded-tie", "poisson", [0], [1e-10], [1], status="worsening")
    add("stationary-tiny", "poisson", [0], [1e-30], [1], status="worsening")
    add("zero-count", "poisson", [0], [-0.25], [0], exposure=[2], status="improvement")
    for family in ("binary", "poisson"):
        rng = np.random.default_rng(107)
        for i in range(20):
            old = rng.uniform(-3, 3, 7)
            add(
                "seed107-" + str(i),
                family,
                old,
                old + rng.uniform(-1, 1, 7),
                rng.integers(0, 2 if family == "binary" else 10, 7),
                offset=rng.uniform(-0.5, 0.5, 7),
                weight=[0, 1, 2, 3, 4, 1, 2],
                exposure=rng.uniform(0.5, 3, 7),
            )
    return result


CASES = cases()
