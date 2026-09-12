"""Resident softmax row math and class fields; no host objective fallback."""

import math

from numba import cuda, float32, float64


@cuda.jit(device=True)
def partition(raw, offset, row):
    maximum, index = float64(-math.inf), 0
    for j in range(raw.shape[1]):
        value = float64(raw[row, j]) + float64(offset[row, j])
        if value > maximum:
            maximum, index = value, j
    tail = float64(0)
    for j in range(raw.shape[1]):
        if j != index:
            tail += math.exp(float64(raw[row, j]) + float64(offset[row, j]) - maximum)
    return maximum, index, tail


@cuda.jit(device=True)
def channel(raw, offset, row, j, target, maximum, index, tail):
    denominator = 1 + tail
    p = (
        1 / denominator
        if j == index
        else math.exp(float64(raw[row, j]) + float64(offset[row, j]) - maximum) / denominator
    )
    complement = tail / denominator if j == index else 1 - p
    g = float32(-complement if j == target else p)
    h = float32(2 * p * complement)
    if not math.isfinite(g) or not math.isfinite(h) or h <= 0:
        return float32(math.nan), float32(math.nan)
    return g, h


@cuda.jit
def multiclass_base(target, output):
    j = cuda.grid(1)
    if j < output.size:
        observed = False
        for r in range(target.shape[0]):
            observed = observed or target[r, 0] == j
        output[j] = 0 if observed else math.nan


@cuda.jit
def multiclass_geometry(target, offset, raw, gradient, bound):
    r = cuda.grid(1)
    if r < raw.shape[0]:
        maximum, index, tail = partition(raw, offset, r)
        for j in range(raw.shape[1]):
            g, h = channel(raw, offset, r, j, target[r, 0], maximum, index, tail)
            gradient[r, j], bound[r, j] = g, h


@cuda.jit
def multiclass_fields(gradient, bound, selected, output):
    r = cuda.grid(1)
    if r < gradient.shape[0]:
        # Reject an unsupported matrix even if the selected class is well behaved.
        valid = True
        for j in range(bound.shape[1]):
            valid = valid and bound[r, j] > 0
        output[r, 0] = gradient[r, selected] if valid else math.nan
        output[r, 1] = bound[r, selected] if valid else math.nan


@cuda.jit
def multiclass_loss(target, offset, weight, raw, output):
    if cuda.grid(1) == 0:
        total, mass = float64(0), float64(0)
        for r in range(raw.shape[0]):
            maximum, index, tail = partition(raw, offset, r)
            # Validate every channel on every row before any weighting shortcut.
            for j in range(raw.shape[1]):
                g, h = channel(raw, offset, r, j, target[r, 0], maximum, index, tail)
                if not math.isfinite(g) or not math.isfinite(h):
                    output[0] = math.nan
                    return
            code = int(target[r, 0])
            shifted = float64(raw[r, code]) + float64(offset[r, code]) - maximum
            value = math.log1p(tail) - shifted
            total += float64(weight[r]) * value
            mass += float64(weight[r])
        output[0] = total / mass
