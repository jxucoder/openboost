"""Resident AFT erfc/quadrature kernels; no host loss or derivative evaluation."""

import math

from numba import cuda, float32, float64

from .survival import _NODES, _WEIGHTS


@cuda.jit(device=True)
def aft_row(raw, lower, event, sigma):
    log_time = math.log(lower)
    z = (log_time - raw) / sigma
    log_phi = -float64(0.5) * z * z - float64(0.5) * math.log(2 * math.pi)
    if not math.isfinite(z) or not math.isfinite(log_phi):
        return float64(math.nan), float32(math.nan), float32(math.nan)
    if event:
        value = log_time + math.log(sigma) - log_phi
        g, h = -z / sigma, 1 / (sigma * sigma)
    else:
        if z > 8:
            integral, moment = float64(0), float64(0)
            for k in range(32):
                kernel = math.exp(-float64(0.5) * (_NODES[k] / z) ** 2)
                integral += _WEIGHTS[k] * kernel
                moment += _WEIGHTS[k] * _NODES[k] * kernel
            logsf, mills = log_phi + math.log(integral) - math.log(z), z / integral
            curvature = moment / (integral * integral)
        else:
            logsf = (math.log(math.erfc(z / math.sqrt(2)) / 2) if z >= 0
                     else math.log1p(-math.erfc(-z / math.sqrt(2)) / 2))
            mills = math.exp(log_phi - logsf)
            curvature = mills * (mills - z)
        value, g, h = -logsf, -mills / sigma, curvature / (sigma * sigma)
    gradient, curvature = float32(g), float32(h)
    if (not math.isfinite(value) or not math.isfinite(gradient) or not math.isfinite(curvature)
            or curvature <= 0 or (not event and gradient == 0)):
        return float64(math.nan), float32(math.nan), float32(math.nan)
    return value, gradient, curvature


@cuda.jit
def aft_base(lower, offset, weight, output):
    if cuda.grid(1) == 0:
        mass, total = float64(0), float64(0)
        for r in range(weight.size):
            mass += float64(weight[r])
        for r in range(weight.size):
            total += float64(weight[r]) / mass * (math.log(float64(lower[r, 0])) - float64(offset[r, 0]))
        output[0] = total


@cuda.jit
def aft_geometry(lower, offset, event, sigma, raw, output):
    r = cuda.grid(1)
    if r < raw.shape[0]:
        _, g, h = aft_row(float64(raw[r, 0]) + float64(offset[r, 0]), float64(lower[r, 0]), event[r], sigma)
        output[r, 0], output[r, 1] = g, h


@cuda.jit
def aft_loss(lower, offset, event, sigma, weight, raw, output):
    if cuda.grid(1) == 0:
        total, mass = float64(0), float64(0)
        for r in range(weight.size):
            value, _, _ = aft_row(float64(raw[r, 0]) + float64(offset[r, 0]), float64(lower[r, 0]), event[r], sigma)
            total += float64(weight[r]) * value
            mass += float64(weight[r])
        output[0] = total / mass
