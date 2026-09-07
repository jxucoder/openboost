"""Exact run-4 scoring function for device diagnostics; not an independent oracle."""

import math

from numba import cuda, float32


@cuda.jit
def scalar_scores(values, counts, active, parent, g, h, regularization, penalty, output):
    c = cuda.grid(1)
    if c < output.size:
        gain = float32(0)
        hl, hr = values[c, 0, h], values[c, 1, h]
        if active[c] and counts[c, 0] > 0 and counts[c, 1] > 0 and hl > 0 and hr > 0:
            dl, dr, dp = hl + regularization, hr + regularization, parent[h] + regularization
            if not math.isfinite(dl) or not math.isfinite(dr) or not math.isfinite(dp) or dp <= 0:
                gain = float32(math.nan)
            else:
                gl, gr = values[c, 0, g], values[c, 1, g]
                gain = (
                    float32(0.5) * gl * (gl / dl)
                    + float32(0.5) * gr * (gr / dr)
                    - float32(0.5) * parent[g] * (parent[g] / dp)
                    - penalty
                )
        output[c] = gain
