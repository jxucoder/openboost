"""090-B representability cases frozen before Normal device kernels.

Float64 row mathematics with explicit float32 storage checks, not kernel emulation.
"""

import numpy as np

from .device_normal import geometry

DOMAIN_CASES = (
    ("origin", (0, 0), 1, (0, 0), True),
    ("positive_log_scale", (0, 40), 1, (0, 0), True),
    ("negative_log_scale", (0, -40), 0, (0, 0), True),
    ("offset_once", (1, 0.25), 2, (0.5, -0.125), True),
    ("precision_overflow", (0, -50), 0, (0, 0), False),
    ("precision_underflow", (0, 60), 0, (0, 0), False),
    ("scale_overflow", (0, 100), 0, (0, 0), False),
    ("gradient_overflow", (1e25, 0), 0, (0, 0), False),
    ("offset_precision_overflow", (0, -25), 0, (0, -25), False),
)

# Fixed before hardware measurements; old 212-case tolerances are unchanged.
RTOL, ATOL = 2e-4, 2e-5
LOSS_RTOL, LOSS_ATOL = 2e-5, 2e-6


def stored_geometry(raw, target, offset, weight):
    raw, target, offset = (np.asarray(a, np.float32) for a in (raw, target, offset))
    value, gradient, fisher = geometry(raw, target, offset, weight)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scale = np.exp(raw[:, 1].astype(float) + offset[:, 1].astype(float)).astype(np.float32)
        gradient, fisher = gradient.astype(np.float32), fisher.astype(np.float32)
    if (
        not np.isfinite(gradient).all()
        or not np.isfinite(fisher).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
        or np.any(fisher <= 0)
    ):
        raise ValueError("Normal geometry is not representable in float32")
    return value, gradient, fisher
