"""Objective-owned numerical evidence, separate from absolute reporting metrics."""

import math
from dataclasses import dataclass
from numbers import Real


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("finite real comparison bounds and thresholds required")
    return float(value)


@dataclass(frozen=True)
class LossChange:
    """An objective's enclosure of loss(after) - loss(before) at stored inputs.

    Bounds are authoritative; estimate/radius are derived diagnostics. None/None
    means no enclosure is available. An author supplying an operation owns the
    mathematical validity of its bounds and names the method used to obtain them.
    Snapshot identity is enforced by the operation/transaction, not this record.
    """

    lower: float | None
    upper: float | None
    method: str
    reason: str
    unchanged: bool = False

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in (self.method, self.reason)):
            raise ValueError("nonempty comparison method and reason required")
        if type(self.unchanged) is not bool:
            raise ValueError("explicit unchanged boolean required")
        if (self.lower is None) != (self.upper is None):
            raise ValueError("both comparison bounds or neither required")
        if self.lower is not None:
            for name in ("lower", "upper"):
                object.__setattr__(self, name, _finite(getattr(self, name)))
            if self.lower > self.upper:
                raise ValueError("ordered comparison bounds required")
        if self.unchanged and (self.lower != 0 or self.upper != 0):
            raise ValueError("unchanged raw requires exactly zero change")

    @property
    def status(self):
        if self.unchanged:
            return "unchanged"
        if self.lower is None:
            return "unresolved"
        if self.upper < 0:
            return "improvement"
        return "worsening" if self.lower > 0 else "unresolved"

    @property
    def estimate(self):
        if self.lower is None:
            return None
        return min(self.upper, max(self.lower, self.lower / 2 + self.upper / 2))

    @property
    def uncertainty(self):
        if self.lower is None:
            return None
        radius = max(self.estimate - self.lower, self.upper - self.estimate)
        return math.nextafter(radius, math.inf) if radius else 0.0

    def improves(self, min_delta=0.0):
        """Prove a strict decrease exceeding an explicitly supplied threshold."""
        threshold = _finite(min_delta)
        if threshold < 0:
            raise ValueError("nonnegative improvement threshold required")
        return self.upper is not None and self.upper < -threshold


def _normal_result(lower, upper, code, unchanged):
    method = "normal-taylor18-interval-v1"
    if code:
        return LossChange(None, None, method, {1: "exponent_range", 2: "arithmetic_range"}[code])
    if unchanged:
        return LossChange(0.0, 0.0, method, "identical_stored_raw", unchanged=True)
    reason = "contains_zero" if lower <= 0 <= upper else "bounded_sign"
    return LossChange(lower, upper, method, reason)


def _glm_result(family, lower, upper, code, unchanged):
    method = family + "-convex-taylor18-interval-v1"
    if code:
        return LossChange(None, None, method, {1: "exponent_range", 2: "arithmetic_range"}[code])
    if unchanged:
        return LossChange(0.0, 0.0, method, "identical_stored_raw", unchanged=True)
    return LossChange(lower, upper, method,
                      "contains_zero" if lower <= 0 <= upper else "bounded_sign")
