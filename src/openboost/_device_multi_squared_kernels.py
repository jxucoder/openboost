"""Multi-output squared and generic diagonal-field kernels; real CUDA only."""

import math

from numba import cuda, float32, float64

from ._device_glm_comparison import _add, _mul
from ._multi_squared_math import make_squared_math

_change = make_squared_math(cuda.jit(device=True), _add, _mul)


@cuda.jit(device=True)
def residual(raw, offset, target):
    return float64(raw) + float64(offset) - float64(target)


@cuda.jit
def multi_squared_base(target, offset, weight, output):
    k = cuda.grid(1)
    if k < output.size:
        total, mass = float64(0), float64(0)
        for r in range(weight.size):
            total += float64(weight[r]) * (float64(target[r, k]) - float64(offset[r, k]))
            mass += float64(weight[r])
        output[k] = total / mass


@cuda.jit
def multi_squared_geometry(target, offset, raw, gradient, curvature):
    r = cuda.grid(1)
    if r < raw.shape[0]:
        for k in range(raw.shape[1]):
            gradient[r, k] = residual(raw[r, k], offset[r, k], target[r, k])
            curvature[r, k] = 1


@cuda.jit
def multi_squared_loss(target, offset, weight, raw, output):
    if cuda.grid(1) == 0:
        total, mass = float64(0), float64(0)
        for r in range(raw.shape[0]):
            row = float64(0)
            for k in range(raw.shape[1]):
                error = residual(raw[r, k], offset[r, k], target[r, k])
                if not math.isfinite(float32(error)):
                    output[0] = math.nan
                    return
                row += error * error / 2
            total += float64(weight[r]) * row
            mass += float64(weight[r])
        output[0] = total / mass


@cuda.jit
def multi_squared_compare_rows(target, offset, before, after, output):
    r = cuda.grid(1)
    if r < before.shape[0]:
        total, code, unchanged = (float64(0), float64(0)), 0, 1
        for k in range(before.shape[1]):
            old, new = float64(before[r, k]), float64(after[r, k])
            y, off = float64(target[r, k]), float64(offset[r, k])
            if not math.isfinite(float32(residual(old, off, y))) or not math.isfinite(
                float32(residual(new, off, y))
            ):
                code = 3
            else:
                lower, upper, channel_code = _change(old, new, y, off)
                total = _add(total, (lower, upper))
                code = max(code, channel_code)
            if old != new:
                unchanged = 0
        output[r, 0], output[r, 1], output[r, 2], output[r, 3] = total[0], total[1], code, unchanged


@cuda.jit
def diagonal_fields(gradient, curvature, output):
    r = cuda.grid(1)
    if r < gradient.shape[0]:
        width = gradient.shape[1]
        for k in range(width):
            output[r, k], output[r, width + k] = gradient[r, k], curvature[r, k]


@cuda.jit
def projected_diagonal_fields(gradient, curvature, projection, output):
    r = cuda.grid(1)
    if r < gradient.shape[0]:
        width = output.shape[1] // 2
        for j in range(width):
            g, h = float64(0), float64(0)
            for k in range(gradient.shape[1]):
                p = float64(projection[k * width + j])
                g += float64(gradient[r, k]) * p
                h += float64(curvature[r, k]) * p * p
            output[r, j], output[r, width + j] = g, h
