"""High-precision original-row Normal differences; no production/device imports.

This is a numerical diagnostic oracle, not a replacement acceptance policy.
Binary input values are converted exactly; exp and arithmetic use the requested
Decimal precision. Comparing two precisions is not an interval error bound.
"""

from decimal import Decimal, localcontext

import numpy as np


def loss_difference(before, after, target, offset, weight, *, precision=80):
    """Weighted mean NLL(after) - NLL(before); the common constant cancels."""
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
    if type(precision) is not int or precision < 32:
        raise ValueError("at least 32 decimal digits required")

    def decimal(value):
        return Decimal.from_float(float(value))

    with localcontext() as context:
        context.prec = precision
        total, mass = Decimal(0), Decimal(0)
        for old, new, y, off, w in zip(before, after, target, offset, weight, strict=True):
            y, w = decimal(y), decimal(w)
            m0, l0 = (decimal(old[k]) + decimal(off[k]) for k in range(2))
            m1, l1 = (decimal(new[k]) + decimal(off[k]) for k in range(2))
            old_square = (m0 - y) ** 2 * (-2 * l0).exp()
            new_square = (m1 - y) ** 2 * (-2 * l1).exp()
            total += w * (l1 - l0 + (new_square - old_square) / 2)
            mass += w
        return total / mass


def compare_precisions(before, after, target, offset, weight):
    """Retain both numerical estimates and their signs, without rounding to float."""
    values = [
        loss_difference(before, after, target, offset, weight, precision=p) for p in (60, 100)
    ]
    signs = [int(v > 0) - int(v < 0) for v in values]
    with localcontext() as context:
        context.prec = 110
        difference = abs(values[0] - values[1])
        scale = max(abs(v) for v in values)
        agreement = difference == 0 if scale == 0 else difference <= scale * Decimal("1e-40")
    return dict(
        decimal60=str(values[0]),
        decimal100=str(values[1]),
        signs=signs,
        estimates_agree=bool(agreement and signs[0] == signs[1]),
        interpretation="Precision agreement is numerical evidence, not an interval error bound.",
    )
