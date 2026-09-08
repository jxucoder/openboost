"""106 stored-input GLM mathematics; independent of production and CUDA kernels."""

from decimal import Decimal, localcontext

import numpy as np

from .classification import binary as binary_reference
from .classification import binary_base
from .positive import poisson as poisson_reference

RTOL, ATOL = 1e-4, 1e-5
GEOMETRY_RTOL, GEOMETRY_ATOL = 1e-6, 1e-8
METRIC_SCALE = 1e-3
REQUIRED_CELLS = (
    "R1/squared",
    "R1/binary",
    "R1/multiclass",
    "R4/poisson",
    "R5/aft",
    "R6/ordinary/fixed",
    "R6/ordinary/backtracking",
    "R6/fisher/fixed",
    "R6/fisher/backtracking",
    "R8/independent",
    "R8/shared",
    "R9/batched/1",
    "R9/batched/8",
    "R9/batched/32",
)
CHECKS = {
    "base",
    "geometry",
    "statistics",
    "split",
    "route",
    "leaf",
    "rounds",
    "prediction",
    "metric",
    "persistence",
    "ownership",
}


def complete_cells(records):
    """A test of accounting only, never a substitute for actual device evidence."""
    return set(records) == set(REQUIRED_CELLS) and all(
        r.get("backend") == "cuda"
        and r.get("passed") is True
        and r.get("skipped") == 0
        and r.get("cpu_fallback") is False
        and set(r.get("checks", [])) == CHECKS
        for r in records.values()
    )


def stored(values):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.asarray(values, np.float32).astype(float)
    if not np.isfinite(result).all():
        raise ValueError("finite float32 storage required")
    return result


def geometry(family, raw, target, offset, weight, exposure=None):
    r, y, o, w = (stored(v) for v in (raw, target, offset, weight))
    if family == "binary":
        value, g, h = binary_reference(r + o, y, weight=w)
    elif family == "poisson":
        if not np.array_equal(y, target):
            raise ValueError("counts must survive float32 storage exactly")
        value, g, h = poisson_reference(r + o, y, stored(exposure), weight=w)
    else:
        raise ValueError("explicit GLM family required")
    g, h = stored(g), stored(h)
    if np.any(h <= 0):
        raise ValueError("positive representable float32 curvature required")
    return value, g, h


def base(family, target, offset, weight, exposure=None, *, parameter=1e-6):
    y, o, w = (stored(v) for v in (target, offset, weight))
    parameter = float(stored(parameter))
    if family == "binary":
        return float(stored(binary_base(y, weight=w, clip=parameter) - np.dot(w / w.sum(), o)))
    if family != "poisson" or not np.array_equal(y, target):
        raise ValueError("exact Poisson counts required")
    e = stored(exposure)
    if np.any(e <= 0) or parameter <= 0:
        raise ValueError("positive exposure and initial rate required")
    # Decimal summation/exponentials are structurally different from the device
    # implementation's float64 log-sum-exp and also cover offsets beyond exp64.
    with localcontext() as context:
        context.prec = 70
        terms = [
            (Decimal(float(a)), Decimal(float(b)), Decimal(float(c)), Decimal(float(d)))
            for a, b, c, d in zip(y, o, w, e, strict=True)
        ]
        numerator = sum(a * c for a, b, c, d in terms if c > 0)
        if not numerator:
            return float(stored(np.log(parameter)))
        denominator = sum(c * d * b.exp() for a, b, c, d in terms if c > 0)
        return float(stored(float(numerator.ln() - denominator.ln())))


DOMAIN_CASES = (
    ("binary", "positive_tail", 80, 1, 0, 1, True),
    ("binary", "negative_tail", -80, 0, 0, 1, True),
    ("binary", "wrong_class_tail", 80, 0, 0, 1, True),
    ("binary", "tail_extinction", 120, 1, 0, 1, False),
    ("binary", "offset_once", 1, 1, 0.5, 1, True),
    ("poisson", "zero_count", 0, 0, 0, 2, True),
    ("poisson", "offset_exposure", -1, 3, 0.5, 2, True),
    ("poisson", "mean_underflow", -120, 0, 0, 1, False),
    ("poisson", "mean_overflow", 90, 1, 0, 1, False),
    ("poisson", "lost_count", 0, 16777217, 0, 1, False),
    ("poisson", "exact_large_count", 16, 16777218, 0, 1, True),
    ("poisson", "lost_exposure", 0, 0, 0, 1e-60, False),
)
