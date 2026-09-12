"""Cancelled squared-error polynomial with explicit directed-rounding dependencies."""

import math

from ._comparison_math import cpu_add, cpu_mul


def make_squared_math(compile_function, add, mul):
    @compile_function
    def channel_change(old, new, target, offset):
        # Complete geometry validation belongs to the caller, before identity shortcuts.
        if old == new:
            return 0.0, 0.0, 0
        residual = add(add((old, old), (offset, offset)), (-target, -target))
        delta = add((new, new), (-old, -old))
        change = mul(delta, add(residual, mul((0.5, 0.5), delta)))
        if not math.isfinite(change[0]) or not math.isfinite(change[1]):
            return 0.0, 0.0, 2
        return change[0], change[1], 0

    return channel_change


cpu_channel_change = make_squared_math(lambda function: function, cpu_add, cpu_mul)
