"""Independent 111 convex softmax prototype and direct high-precision likelihoods."""

from decimal import Decimal, localcontext

import numpy as np

from .device_multiclass import geometry
from .normal_comparison import (
    TWO,
    ZERO,
    Comparison,
    Interval,
    UnresolvedArithmetic,
    _result,
    exponential,
)


def add(left, right):
    if left.lower == left.upper == -right.lower == -right.upper:
        return ZERO
    return left + right


def difference(left, right):
    return ZERO if left == right else Interval.point(left) - Interval.point(right)


def square(value):
    product = value * value
    return Interval(max(0, product.lower), product.upper)


def exp_bound(value):
    if value.lower < -512 or value.upper > 512:
        raise UnresolvedArithmetic("exponent_range")
    result = exponential(value / TWO / TWO / TWO / TWO)
    for _ in range(4):
        result = result * result
    return result


def row_change(before, after, target, offset):
    if np.array_equal(before, after):
        return ZERO
    y = int(target)
    weights, step = [], []
    for old, new, off in zip(before, after, offset, strict=True):
        weights.append(exp_bound(add(difference(old, before[y]), difference(off, offset[y]))))
        step.append(add(difference(new, after[y]), -difference(old, before[y])))
    mass, linear, variance = ZERO, ZERO, ZERO
    for a, delta in zip(weights, step, strict=True):
        mass += a
        linear = add(linear, a * delta)
    for i, a in enumerate(weights):
        for j in range(i):
            gap = add(step[i], -step[j])
            variance += a * weights[j] * square(gap)
    linear, variance = linear / mass, variance / square(mass)
    # An interval upper bound on the exact step range; the target step is zero.
    extent = (
        Interval.point(max(s.upper for s in step)) - Interval.point(min(s.lower for s in step))
    ).upper
    radius = Interval.point(max(0, extent))
    low = exp_bound(-TWO * radius) * variance
    high = exp_bound(TWO * radius) * variance
    global_upper = (square(radius) / Interval.point(4)).upper
    curvature = Interval(max(0, low.lower), min(high.upper, global_upper))
    return linear + curvature / TWO


def inputs(before, after, target, offset, weight):
    with np.errstate(over="raise", invalid="raise"):
        arrays = tuple(np.asarray(a, np.float32) for a in (before, after, target, offset, weight))
    old, new, y, o, w = arrays
    if old.shape != new.shape:
        raise ValueError("aligned before/after matrices required")
    for raw in (old, new):
        geometry(raw, y, o, w)
    return arrays


def compare(before, after, target, offset, weight):
    old, new, y, o, w = inputs(before, after, target, offset, weight)
    try:
        total, mass = ZERO, ZERO
        for a, b, code, off, weight in zip(old, new, y, o, w, strict=True):
            total += row_change(a, b, code, off) * Interval.point(weight)
            mass += Interval.point(weight)
        return _result(total / mass, unchanged=np.array_equal(old, new))
    except UnresolvedArithmetic as error:
        return Comparison("unresolved", str(error), None, None, None, None)


def direct_difference(before, after, target, offset, weight, *, precision=260):
    """Independent full softmax likelihood subtraction at exact float32 inputs."""
    with localcontext() as ctx:
        ctx.prec = precision
        totals, mass = [Decimal(0), Decimal(0)], Decimal(0)
        for a, b, code, off, w in zip(before, after, target, offset, weight, strict=True):
            w = Decimal(float(np.float32(w)))
            for index, raw in enumerate((a, b)):
                values = [
                    Decimal(float(np.float32(v))) + Decimal(float(np.float32(o)))
                    for v, o in zip(raw, off, strict=True)
                ]
                maximum = max(values)
                loss = sum((v - maximum).exp() for v in values).ln() - (values[int(code)] - maximum)
                totals[index] += w * loss
            mass += w
        return (totals[1] - totals[0]) / mass


def cases():
    result = []

    def append(name, old, new, target, *, offset=None, weight=None, status=None):
        old = np.asarray(old, np.float32)
        arrays = [
            old,
            np.asarray(new, np.float32),
            np.asarray(target, np.float32),
            np.zeros_like(old) if offset is None else np.asarray(offset, np.float32),
            np.ones(len(old), np.float32) if weight is None else np.asarray(weight, np.float32),
        ]
        result.append(dict(id=name, arrays=[a.tolist() for a in arrays], status=status))

    for width in (2, 3, 5):
        zeros = np.zeros((1, width))
        improved = zeros.copy()
        improved[0, 0] = 1e-20
        append(f"k{width}/unchanged", zeros, zeros, [0], status="unchanged")
        append(f"k{width}/tiny-improvement", zeros, improved, [0], status="improvement")
        append(f"k{width}/tiny-worsening", zeros, -improved, [0], status="worsening")
        append(f"k{width}/common-shift", zeros, zeros + 16, [0], status="unresolved")
        old = zeros.copy()
        old[0, 0] = 80
        new = old.copy()
        new[0, 0] += 0.01
        append(f"k{width}/dominant-tail", old, new, [0], status="improvement")
        append(f"k{width}/wrong-class-tail", old, new, [1], status="worsening")
        append(f"k{width}/large-step", -old / 40, old / 40, [0])
    for value in (1e-10, 1e-30):
        append(f"stationary/{value:g}", [[0, 0, 0]], [[0, value, -value]], [0], status="worsening")
    append("balanced-rows", [[0, 0], [0, 0]], [[1e-20, 0], [1e-20, 0]], [0, 1], status="unresolved")
    append(
        "zero-weight-change",
        [[0, 0, 0]] * 2,
        [[0, 0, 0], [0.5, 0, 0]],
        [0, 1],
        weight=[1, 0],
        status="unresolved",
    )
    append("equal-loss-permutation", [[1, -1, 0]], [[-1, 1, 0]], [2], status="unresolved")
    append(
        "large-common-offset",
        [[0, 0, 0]],
        [[0.125, -0.125, 0]],
        [0],
        offset=[[800, 800, 800]],
        status="improvement",
    )
    append(
        "weighted-offsets",
        [[0, 1, -1], [1, -1, 0], [0, 0, 0]],
        [[0.1, 1, -1], [1, -1.25, 0], [0, 0, 0.125]],
        [0, 1, 2],
        offset=[[0.25, -0.125, 0], [0.125, 0, -0.25], [0, 0.25, -0.125]],
        weight=[2, 0, 3],
    )
    rng = np.random.default_rng(111)
    for width in (2, 3, 5):
        for index in range(12):
            old = rng.uniform(-3, 3, (4, width))
            append(
                f"k{width}/seed111-{index}",
                old,
                old + rng.uniform(-0.5, 0.5, old.shape),
                rng.integers(0, width, 4),
                offset=rng.uniform(-0.5, 0.5, old.shape),
                weight=[0, 1, 3, 2],
            )
    return result


CASES = cases()
