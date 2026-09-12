"""Independent AFT interval prototype and direct Decimal likelihood differences."""

import math
from decimal import Decimal, localcontext

import numpy as np

from .device_aft import geometry, pi, tail
from .device_glm import stored
from .glm_comparison import exp_bound
from .normal_comparison import ONE, TWO, ZERO, Comparison, Interval, UnresolvedArithmetic, _result

HALF = Interval.point(0.5)
LOG_TWO = Interval(0.6931471805599453, 0.6931471805599454)
DENSITY = Interval(0.39894228040143265, 0.3989422804014327)


def logarithm(value):
    if value == 1:
        return ZERO
    mantissa, exponent = math.frexp(value)
    mantissa, exponent = 2 * mantissa, exponent - 1
    t = (Interval.point(mantissa) - ONE) / (Interval.point(mantissa) + ONE)
    square, power, total = t * t, t, ZERO
    for k in range(32):
        total += power / Interval.point(2 * k + 1)
        power *= square
    remainder = (TWO * power / (Interval.point(65) * (ONE - square))).upper
    return LOG_TWO * Interval.point(exponent) + TWO * total + Interval(0, remainder)


def mills(point):
    x = Interval.point(abs(point))
    if abs(point) > 2:
        correction = Interval(0, (Interval.point(129) / x).upper)
        for k in range(128, 0, -1):
            correction = Interval.point(k) / (x + correction)
        positive = x + correction
        if point > 0:
            return positive
        density = DENSITY * exp_bound(-(x * x) / TWO)
        return density / (ONE - density / positive)
    z = Interval.point(point)
    power, integral = z, ZERO
    square = -(z * z) / TWO
    for k in range(32):
        integral += power / Interval.point(2 * k + 1)
        power = power * square / Interval.point(k + 1)
    remainder = max(abs(power.lower), abs(power.upper)) / 65
    remainder = math.nextafter(remainder, math.inf)
    integral += Interval(-remainder, remainder)
    density = DENSITY * exp_bound(square)
    return density / (HALF - DENSITY * integral)


def row_change(old, new, lower, event, offset, sigma):
    if old == new:
        return ZERO
    scale = Interval.point(sigma)
    delta = (Interval.point(new) - Interval.point(old)) / scale
    before = (logarithm(lower) - Interval.point(old) - Interval.point(offset)) / scale
    if event:
        return delta * (-before + delta / TWO)
    after = (logarithm(lower) - Interval.point(new) - Interval.point(offset)) / scale
    left, right = min(before.lower, after.lower), max(before.upper, after.upper)
    if left < -16 or right > 1e12:
        raise UnresolvedArithmetic("tail_range")
    rectangle = -delta * Interval(mills(left).lower, mills(right).upper)
    gradient = -Interval(mills(before.lower).lower, mills(before.upper).upper)
    remainder = (delta * delta / TWO).upper
    taylor = gradient * delta + Interval(0, remainder)
    return Interval(max(rectangle.lower, taylor.lower), min(rectangle.upper, taylor.upper))


def compare(before, after, lower, event, offset, weight, sigma):
    old, new, lower, offset, weight = (stored(a) for a in (before, after, lower, offset, weight))
    event = np.asarray(event, bool)
    arrays = (old, new, lower, event, offset, weight)
    if not old.size or any(a.shape != old.shape or a.ndim != 1 for a in arrays):
        raise ValueError("aligned scalar AFT rows required")
    if np.any(weight < 0) or not np.any(weight > 0):
        raise ValueError("nonnegative positive-total weights required")
    for raw in (old, new):
        geometry(raw, lower, event, offset, weight, sigma)
    try:
        total, mass = ZERO, ZERO
        for a, b, t, e, o, w in zip(*arrays, strict=True):
            total += row_change(a, b, t, e, o, sigma) * Interval.point(w)
            mass += Interval.point(w)
        return _result(total / mass, unchanged=np.array_equal(old, new))
    except UnresolvedArithmetic as error:
        return Comparison("unresolved", str(error), None, None, None, None)


def direct_difference(before, after, lower, event, offset, weight, sigma, *, precision=220):
    """Independent full-likelihood subtraction, retaining exact stored inputs."""
    with localcontext() as ctx:
        ctx.prec = precision
        old_total, new_total, mass = Decimal(0), Decimal(0), Decimal(0)
        scale = Decimal(float(sigma))
        for a, b, t, e, o, w in zip(before, after, lower, event, offset, weight, strict=True):
            a, b, t, o, w = (Decimal(float(np.float32(v))) for v in (a, b, t, o, w))
            values = []
            for raw in (a, b):
                z = (t.ln() - raw - o) / scale
                value = (t.ln() + scale.ln() + z*z/2 + (2*pi(precision)).ln()/2 if e
                         else -tail(z, precision=precision)[0])
                values.append(value)
            old_total += w * values[0]
            new_total += w * values[1]
            mass += w
        return (new_total - old_total) / mass


def cases():
    result = []

    def add(name, old, new, lower, event, *, sigma=1, offset=None, weight=None, status=None):
        n = len(old)
        arrays = [stored(v).tolist() for v in (old, new, lower)]
        arrays += [[bool(e) for e in event], stored([0]*n if offset is None else offset).tolist(),
                   stored([1]*n if weight is None else weight).tolist(), sigma]
        result.append(dict(id=name, arrays=arrays, status=status))

    for sigma in (0.5, 0.7, 1, 2):
        prefix = str(sigma) + "/"
        add(prefix+"unchanged", [0, 1], [0, 1], [1, 2], [True, False], sigma=sigma, status="unchanged")
        add(prefix+"stationary-event", [0], [1e-20], [1], [True], sigma=sigma, status="worsening")
        add(prefix+"tiny-censor", [0], [1e-20], [1], [False], sigma=sigma, status="improvement")
        add(prefix+"mixed", [-1, 0, 2], [-.8, .1, 1.9], [1, 2, 4], [True, False, True],
            sigma=sigma, offset=[.5, -.25, .125], weight=[2, 0, 3])
        add(prefix+"event-improvement", [-1], [-.5], [1], [True], sigma=sigma, status="improvement")
        add(prefix+"all-censored", [-2, 1], [1, 4], [1, 2], [False, False], sigma=sigma, status="improvement")
    for z in (-14, -12, -2, -1e-10, 0, 1e-10, 2, 8, 40, 1000, 1e8):
        old = np.float32(-z)
        new = np.nextafter(old, np.float32(math.inf), dtype=np.float32)
        if new == 0:
            new = np.float32(1e-20)
        add("tail/"+str(z), [old], [new], [1], [False], status="improvement")
    add("zero-weight-change", [0, 0], [0, .5], [1, 1], [True, False], weight=[1, 0], status="unresolved")
    add("equal-event-loss", [-1, 1], [1, -1], [1, 1], [True, True], status="unresolved")
    add("balanced-event-tiny", [-1, 1], [-1+2**-23, 1+2**-23], [1, 1], [True, True], status="worsening")
    add("tiny-time", [0], [.25], [1e-40], [False], sigma=16, status="improvement")
    add("large-time", [0], [.25], [1e38], [False], status="improvement")
    add("tiny-scale-event", [0], [1e-30], [1], [True], sigma=1e-10, status="worsening")
    add("large-scale-event", [0], [1e-20], [1], [True], sigma=1e20, status="worsening")
    rng = np.random.default_rng(115)
    for k in range(20):
        old = rng.uniform(-2, 2, 5)
        add("random/"+str(k), old, old+rng.uniform(-.3, .3, 5), np.exp(rng.uniform(-1, 2, 5)),
            rng.integers(0, 2, 5).astype(bool), sigma=.7 if k % 2 else 2,
            offset=rng.uniform(-.5, .5, 5), weight=rng.integers(0, 4, 5)+.25)
    return tuple(result)


CASES = cases()
