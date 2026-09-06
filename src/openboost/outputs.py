"""Inference-only output transforms, independent of training objectives."""

import numpy as np

from .data import _owned


def binary_probabilities(raw):
    values = _owned(raw, ndim=2)
    if values.shape[1] != 1:
        raise ValueError("binary probabilities require one raw logit column")
    r = values[:, 0]
    tail = np.exp(-np.abs(r))
    small = tail / (1 + tail)
    positive = np.where(r >= 0, 1 / (1 + tail), small)
    negative = np.where(r >= 0, small, 1 / (1 + tail))
    return _owned(np.column_stack((negative, positive)), ndim=2)


def softmax_probabilities(raw):
    values = _owned(raw, ndim=2)
    if values.shape[1] < 2:
        raise ValueError("softmax requires at least two raw columns")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        shifted = values - values.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return _owned(exp / exp.sum(axis=1, keepdims=True), ndim=2)


def poisson_mean(raw, exposure):
    """Explicit unit-exposure rate and period count mean from scalar log rates."""
    values = _owned(raw, ndim=2)
    e = _owned(exposure, ndim=1)
    if values.shape != (len(e), 1) or np.any(e <= 0):
        raise ValueError("scalar log rates and aligned positive exposure required")
    with np.errstate(over="raise", invalid="raise"):
        rate = np.exp(values[:, 0])
        count = np.exp(values[:, 0] + np.log(e))
    if np.any(rate <= 0) or np.any(count <= 0):
        raise ValueError("Poisson means must remain positive in float64")
    return {"rate": _owned(rate, ndim=1), "count_mean": _owned(count, ndim=1)}
