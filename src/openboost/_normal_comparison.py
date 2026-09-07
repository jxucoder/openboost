"""CPU Normal comparison; domain validation is independent of reported metrics."""

import math
import sys

import numpy as np

from ._comparison_math import cpu_add, cpu_div, cpu_mul, cpu_row_change
from .comparison import _normal_result
from .data import _owned


def _raw(problem, raw):
    raw = _owned(raw, ndim=2)
    values = problem.with_offset(raw)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        scale, precision = np.exp(values[:, 1]), np.exp(-2 * values[:, 1])
        residual = values[:, 0] - problem.target[:, 0]
        square = residual**2 * precision
        arrays = (
            scale,
            precision,
            residual * precision,
            1 - square,
            values[:, 1] + square / 2 + np.log(2 * np.pi) / 2,
        )
    if (
        any(not np.isfinite(a).all() for a in arrays)
        or np.any(scale <= 0)
        or np.any(precision <= 0)
    ):
        raise ValueError("Normal comparison requires positive finite scale and geometry")
    return raw


def compare(problem, before, after):
    if (sys.float_info.radix, sys.float_info.mant_dig, sys.float_info.rounds) != (2, 53, 1):
        raise RuntimeError("round-to-nearest binary64 required for comparison")
    before, after = _raw(problem, before), _raw(problem, after)
    total, mass, code = (0.0, 0.0), (0.0, 0.0), 0
    for old, new, target, offset, weight in zip(
        before, after, problem.target, problem.offset, problem.weight, strict=True
    ):
        lower, upper, row_code = cpu_row_change(
            float(old[0]),
            float(old[1]),
            float(new[0]),
            float(new[1]),
            float(target[0]),
            float(offset[0]),
            float(offset[1]),
        )
        code = max(code, row_code)
        w = float(weight), float(weight)
        total, mass = cpu_add(total, cpu_mul((lower, upper), w)), cpu_add(mass, w)
    bounds = cpu_div(total, mass)
    if not all(math.isfinite(v) for v in bounds):
        code = max(code, 2)
    return _normal_result(*bounds, code, bool(np.array_equal(before, after)))
