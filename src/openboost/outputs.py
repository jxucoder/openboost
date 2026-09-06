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
