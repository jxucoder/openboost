"""Float64 class mappings and unweighted classification geometry for tiny fixtures."""

from dataclasses import dataclass

import numpy as np

from .data import CategoryMap
from .scalar import finite_vector, training_weights


def _codes(target, length, classes):
    y = finite_vector(target, "target", length)
    if np.any(y != np.floor(y)) or np.any(y < 0) or np.any(y >= classes):
        raise ValueError("target must contain in-range integer class codes")
    return y.astype(np.int64)


@dataclass(frozen=True)
class ClassMap:
    values: tuple[str | int, ...]

    @classmethod
    def fit(cls, labels):
        labels = tuple(labels)
        mapping = CategoryMap.fit(labels)
        _, missing = mapping.transform(labels)
        if len(mapping.values) < 2 or any(missing):
            raise ValueError("training requires at least two classes and no missing labels")
        return cls(mapping.values)

    def encode(self, labels):
        codes, missing = CategoryMap(self.values).transform(labels)
        if any(missing):
            raise ValueError("unknown or missing class label")
        return codes

    def decode(self, codes):
        codes = tuple(codes)
        return tuple(self.values[i] for i in _codes(codes, len(codes), len(self.values)))


def binary_base(target, *, weight=None, clip):
    """Explicit probability clipping; single observed training class is illegal."""
    target = tuple(target)
    y = _codes(target, len(target), 2)
    if len(set(y)) != 2:
        raise ValueError("binary training requires both classes")
    if not np.isscalar(clip) or not np.isfinite(clip) or not 0 < clip < 0.5:
        raise ValueError("clip must be finite and strictly between zero and one half")
    if 1 - clip == 1:
        raise ValueError("clip is too small to represent an upper probability below one")
    w = training_weights(weight, len(y))
    p = float(np.clip(sum(w * y) / sum(w), clip, 1 - clip))
    return float(np.log(p) - np.log1p(-p))


def binary(raw, target, *, weight=None):
    raw = finite_vector(raw, "raw")
    y = _codes(target, len(raw), 2)
    w = training_weights(weight, len(raw))
    # Use signed-margin loss to avoid subtracting two nearly equal positive terms.
    losses = np.logaddexp(0, np.where(y == 1, -raw, raw))
    tail = np.exp(-np.abs(raw))
    p = np.where(raw >= 0, 1 / (1 + tail), tail / (1 + tail))
    gradient = np.where(y == 1, -np.where(raw >= 0, tail / (1 + tail), 1 / (1 + tail)), p)
    curvature = tail / (1 + tail) ** 2
    return float(sum((w / sum(w)) * losses)), gradient, curvature


def softmax(raw, target, *, weight=None):
    """Return loss, probability, gradient, exact Hessian, diagonal UPPER BOUND.

    The bound 2*p*(1-p) is for separate-channel tree fitting, not an exact Hessian.
    Every channel is derived from the same raw snapshot; weighting happens later.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 2 or not np.all(np.isfinite(raw)):
        raise ValueError("raw must be a finite [N,K] matrix with K >= 2")
    y = _codes(target, len(raw), raw.shape[1])
    w = training_weights(weight, len(raw))
    with np.errstate(over="ignore"):
        shifted = raw - np.max(raw, axis=1, keepdims=True)
    if not np.all(np.isfinite(shifted)):
        raise ValueError("raw dynamic range exceeds float64")
    exp = np.exp(shifted)
    normalizer = np.sum(exp, axis=1, keepdims=True)
    p = exp / normalizer
    losses = np.log(normalizer[:, 0]) - shifted[np.arange(len(raw)), y]
    gradient = p.copy()
    gradient[np.arange(len(raw)), y] -= 1
    exact = np.array([np.diag(row) - np.outer(row, row) for row in p])
    bound = 2 * p * (1 - p)
    return float(sum((w / sum(w)) * losses)), p, gradient, exact, bound
