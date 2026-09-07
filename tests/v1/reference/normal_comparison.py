"""Independent bounded binary64 experiment for Sprint 092, not a backend.

The enclosure uses basic IEEE arithmetic, nextafter and a Taylor remainder;
neither platform exp/expm1 accuracy nor agreement with Decimal proves the bound.
See v1-sprints/092-normal-comparison-mathematics.md for assumptions and derivation.
"""

import math
import sys
from dataclasses import dataclass

import numpy as np


class UnresolvedArithmetic(ArithmeticError):
    """A finite enclosure is unavailable in the deliberately bounded prototype."""


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float

    def __post_init__(self):
        if not (math.isfinite(self.lower) and math.isfinite(self.upper)):
            raise UnresolvedArithmetic("arithmetic_range")
        if self.lower > self.upper:
            raise ValueError("ordered interval required")

    @classmethod
    def point(cls, value):
        return cls(float(value), float(value))

    @classmethod
    def outward(cls, lower, upper):
        return cls(math.nextafter(lower, -math.inf), math.nextafter(upper, math.inf))

    def __neg__(self):
        return Interval(-self.upper, -self.lower)

    def __add__(self, other):
        if self == ZERO:
            return other
        if other == ZERO:
            return self
        return Interval.outward(self.lower + other.lower, self.upper + other.upper)

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        if self == ZERO or other == ZERO:
            return ZERO
        if self == ONE:
            return other
        if other == ONE:
            return self
        products = [a * b for a in (self.lower, self.upper) for b in (other.lower, other.upper)]
        return Interval.outward(min(products), max(products))

    def __truediv__(self, other):
        if other.lower <= 0 <= other.upper:
            raise UnresolvedArithmetic("denominator_contains_zero")
        if self == ZERO:
            return ZERO
        if other == ONE:
            return self
        quotients = [a / b for a in (self.lower, self.upper) for b in (other.lower, other.upper)]
        return Interval.outward(min(quotients), max(quotients))


ZERO, ONE, TWO = (Interval.point(v) for v in (0, 1, 2))


def _exponential_point(value, *, minus_one):
    if not math.isfinite(value) or abs(value) > 64:
        raise UnresolvedArithmetic("exponent_range")
    if value == 0:
        return ZERO if minus_one else ONE
    reduced, halvings = Interval.point(value), 0
    # Outward halving includes even subnormal rounding. No logarithm chooses k.
    while max(abs(reduced.lower), abs(reduced.upper)) > 1 / 16:
        reduced = reduced / TWO
        halvings += 1
    term, total = reduced, reduced
    for degree in range(2, 19):
        term = term * reduced / Interval.point(degree)
        total = total + term
    # The sum from degree 19 onwards is < 2 * |t|^19/19! for |t| <= 1/16.
    absolute_term = Interval.point(max(abs(term.lower), abs(term.upper)))
    absolute_t = Interval.point(max(abs(reduced.lower), abs(reduced.upper)))
    remainder = (TWO * absolute_term * absolute_t / Interval.point(19)).upper
    result = total + Interval(-remainder, remainder)
    if not minus_one:
        result = ONE + result
    for _ in range(halvings):
        result = result * (result + TWO) if minus_one else result * result
    return result


def exponential(argument, *, minus_one=False):
    """Monotone endpoint enclosures; both exp and expm1 avoid platform libm."""
    lower = _exponential_point(argument.lower, minus_one=minus_one).lower
    upper = _exponential_point(argument.upper, minus_one=minus_one).upper
    return Interval(lower, upper)


@dataclass(frozen=True)
class Comparison:
    status: str
    reason: str
    lower: float | None
    upper: float | None
    estimate: float | None
    uncertainty: float | None

    def improves(self, min_delta=0.0):
        """Strict improvement by more than the caller's stated minimum change."""
        if isinstance(min_delta, bool) or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("finite nonnegative min_delta required")
        return self.upper is not None and self.upper < -min_delta


def _result(interval, *, unchanged):
    if unchanged:
        return Comparison("unchanged", "identical_stored_raw", 0.0, 0.0, 0.0, 0.0)
    lower, upper = interval.lower, interval.upper
    status = "improvement" if upper < 0 else "worsening" if lower > 0 else "unresolved"
    midpoint = min(upper, max(lower, lower / 2 + upper / 2))
    radius = max(midpoint - lower, upper - midpoint)
    radius = math.nextafter(radius, math.inf) if radius else 0.0
    return Comparison(
        status,
        "contains_zero" if status == "unresolved" else "bounded_sign",
        lower,
        upper,
        midpoint,
        radius,
    )


def compare(before, after, target, offset, weight):
    """Bound exact-input mean NLL change; validate every row before zero shortcuts.

    Supports binary64 arithmetic with gradual underflow and separately rounded
    operations. Finite, aligned inputs are required. Unsupported exponent or
    intermediate ranges return unresolved, including on a zero-weight row.
    This is an original-row CPU experiment, not CUDA simulation or a trainer.
    """
    if (sys.float_info.radix, sys.float_info.mant_dig, sys.float_info.rounds) != (2, 53, 1):
        raise RuntimeError("round-to-nearest binary64 required")
    before, after, target, offset, weight = (
        np.asarray(a, dtype=np.float64) for a in (before, after, target, offset, weight)
    )
    n = target.size
    if (
        target.shape != (n,)
        or n == 0
        or before.shape != (n, 2)
        or after.shape != (n, 2)
        or offset.shape != (n, 2)
        or weight.shape != (n,)
        or not all(np.isfinite(a).all() for a in (before, after, target, offset, weight))
        or np.any(weight < 0)
        or not np.any(weight > 0)
    ):
        raise ValueError(
            "finite aligned Normal rows with nonnegative positive-total weights required"
        )
    try:
        rows = []
        for old, new, y, off in zip(before, after, target, offset, strict=True):
            residuals, precisions = [], []
            for raw in (old, new):
                residual = Interval.point(raw[0]) + Interval.point(off[0]) - Interval.point(y)
                ell = Interval.point(raw[1]) + Interval.point(off[1])
                precision = exponential(-TWO * ell)
                # Check the finite quadratic row domain even when its weight is zero.
                _ = residual * residual * precision / TWO + ell
                residuals.append(residual)
                precisions.append(precision)
            dm = ZERO if old[0] == new[0] else Interval.point(new[0]) - Interval.point(old[0])
            dl = ZERO if old[1] == new[1] else Interval.point(new[1]) - Interval.point(old[1])
            ratio = exponential(-TWO * dl)
            ratio_change = exponential(-TWO * dl, minus_one=True)
            r, p = residuals[0], precisions[0]
            change = dl + p / TWO * ((TWO * r * dm + dm * dm) * ratio + r * r * ratio_change)
            rows.append(change)
        total, mass = ZERO, ZERO
        for row, w in zip(rows, weight, strict=True):
            total = total + row * Interval.point(w)
            mass = mass + Interval.point(w)
        return _result(total / mass, unchanged=np.array_equal(before, after))
    except UnresolvedArithmetic as error:
        return Comparison("unresolved", str(error), None, None, None, None)
