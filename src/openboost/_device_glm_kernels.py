"""Resident scalar likelihood kernels; execute only on the owning CUDA stream."""

import math

from numba import cuda, float32, float64


@cuda.jit(device=True)
def supported(value, gradient, curvature):
    g, h = float32(gradient), float32(curvature)
    if not math.isfinite(value) or not math.isfinite(g) or not math.isfinite(h) or h <= 0:
        return float64(math.nan), float32(math.nan), float32(math.nan)
    return value, g, h


@cuda.jit(device=True)
def binary_row(raw, target):
    tail = math.exp(-abs(raw))
    positive = 1 / (1 + tail) if raw >= 0 else tail / (1 + tail)
    negative = tail / (1 + tail) if raw >= 0 else 1 / (1 + tail)
    margin = -raw if target == 1 else raw
    value = max(margin, float64(0)) + math.log1p(math.exp(-abs(margin)))
    return supported(value, -negative if target == 1 else positive, tail / (1 + tail) ** 2)


@cuda.jit(device=True)
def poisson_row(raw, target, exposure):
    log_mean = raw + math.log(exposure)
    mean = math.exp(log_mean)
    value = mean - target * log_mean + math.lgamma(target + 1)
    return supported(value, mean - target, mean)


@cuda.jit
def binary_base(target, offset, weight, clip, output):
    if cuda.grid(1) == 0:
        mass, probability, center = float64(0), float64(0), float64(0)
        zero, one = False, False
        for r in range(weight.size):
            mass += float64(weight[r])
            zero = zero or target[r, 0] == 0
            one = one or target[r, 0] == 1
        if not zero or not one:
            output[0] = math.nan
            return
        for r in range(weight.size):
            q = float64(weight[r]) / mass
            probability += q * float64(target[r, 0])
            center += q * float64(offset[r, 0])
        probability = min(max(probability, float64(clip)), 1 - float64(clip))
        output[0] = math.log(probability) - math.log1p(-probability) - center


@cuda.jit
def poisson_base(target, offset, exposure, weight, minimum_rate, output):
    if cuda.grid(1) == 0:
        count_max, exposure_max = float64(-math.inf), float64(-math.inf)
        for r in range(weight.size):
            if weight[r] > 0:
                log_weight = math.log(float64(weight[r]))
                exposure_max = max(
                    exposure_max,
                    log_weight + math.log(float64(exposure[r, 0])) + float64(offset[r, 0]),
                )
                if target[r, 0] > 0:
                    count_max = max(count_max, log_weight + math.log(float64(target[r, 0])))
        if count_max == -math.inf:
            output[0] = math.log(float64(minimum_rate))
            return
        counts, exposures = float64(0), float64(0)
        for r in range(weight.size):
            if weight[r] > 0:
                log_weight = math.log(float64(weight[r]))
                exposures += math.exp(
                    log_weight
                    + math.log(float64(exposure[r, 0]))
                    + float64(offset[r, 0])
                    - exposure_max
                )
                if target[r, 0] > 0:
                    counts += math.exp(log_weight + math.log(float64(target[r, 0])) - count_max)
        output[0] = count_max + math.log(counts) - exposure_max - math.log(exposures)


@cuda.jit
def binary_geometry(target, offset, raw, output):
    r = cuda.grid(1)
    if r < raw.shape[0]:
        _, g, h = binary_row(float64(raw[r, 0]) + float64(offset[r, 0]), float64(target[r, 0]))
        output[r, 0], output[r, 1] = g, h


@cuda.jit
def poisson_geometry(target, offset, exposure, raw, output):
    r = cuda.grid(1)
    if r < raw.shape[0]:
        _, g, h = poisson_row(
            float64(raw[r, 0]) + float64(offset[r, 0]),
            float64(target[r, 0]),
            float64(exposure[r, 0]),
        )
        output[r, 0], output[r, 1] = g, h


@cuda.jit
def glm_gradient(geometry, output):
    r = cuda.grid(1)
    if r < output.size:
        output[r] = geometry[r, 0]


@cuda.jit
def binary_loss(target, offset, weight, raw, output):
    if cuda.grid(1) == 0:
        total, mass = float64(0), float64(0)
        for r in range(weight.size):
            value, _, _ = binary_row(
                float64(raw[r, 0]) + float64(offset[r, 0]), float64(target[r, 0])
            )
            total += float64(weight[r]) * value
            mass += float64(weight[r])
        output[0] = total / mass


@cuda.jit
def poisson_loss(target, offset, exposure, weight, raw, output):
    if cuda.grid(1) == 0:
        total, mass = float64(0), float64(0)
        for r in range(weight.size):
            value, _, _ = poisson_row(
                float64(raw[r, 0]) + float64(offset[r, 0]),
                float64(target[r, 0]),
                float64(exposure[r, 0]),
            )
            total += float64(weight[r]) * value
            mass += float64(weight[r])
        output[0] = total / mass
