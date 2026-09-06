"""Small shared-stump oracle with separate split projection and complete vector leaves."""

from dataclasses import dataclass

import numpy as np

from .scalar import newton_leaf, training_weights
from .tree import Condition, _route, enumerate_splits, numeric_bins


def _matrix(values, name):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return result


@dataclass(frozen=True)
class TargetScale:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    constant: tuple[bool, ...]

    @classmethod
    def fit(cls, target):
        y = _matrix(target, "target")
        mean, scale = np.mean(y, axis=0), np.std(y, axis=0)
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("target moments exceed float64")
        constant = scale == 0
        return cls(
            tuple(mean), tuple(np.where(constant, 1.0, scale)), tuple(bool(c) for c in constant)
        )

    def _validate(self, values):
        values = _matrix(values, "values")
        if values.shape[1] != len(self.mean):
            raise ValueError("target output schema mismatch")
        return values

    def transform(self, values):
        return (self._validate(values) - self.mean) / self.scale

    def inverse(self, values):
        return self._validate(values) * self.scale + self.mean


@dataclass(frozen=True)
class VectorStump:
    n_features: int
    condition: Condition | None
    left: tuple[float, ...]
    right: tuple[float, ...]

    def predict(self, bins):
        x = numeric_bins(bins)
        if x.shape[1] != self.n_features:
            raise ValueError("feature schema mismatch")
        if self.condition is None:
            return np.tile(self.left, (len(x), 1))
        left, _ = _route(x, tuple(range(len(x))), self.condition)
        prediction = np.tile(self.right, (len(x), 1))
        prediction[list(left)] = self.left
        return prediction


def fit_vector_stump(bins, raw, target, *, weight=None, projection=None, reg_lambda=1.0):
    x, raw, target = numeric_bins(bins), _matrix(raw, "raw"), _matrix(target, "target")
    if raw.shape != target.shape or len(raw) != len(x):
        raise ValueError("raw, target and bins must align")
    w = training_weights(weight, len(x))
    gradient = raw - target
    projection = np.eye(raw.shape[1]) if projection is None else _matrix(projection, "projection")
    if projection.shape[0] != raw.shape[1] or np.any(np.sum(projection**2, axis=0) == 0):
        raise ValueError("projection must have K rows and nonzero columns")
    # Squared-target sketch: projected residual has unit curvature scaled by
    # column norm squared. This only ranks splits; it never changes leaf outputs.
    split_g = gradient @ projection
    split_h = np.broadcast_to(np.sum(projection**2, axis=0), split_g.shape)
    candidates = [
        {
            c.condition: c
            for c in enumerate_splits(
                x, split_g[:, k], split_h[:, k], weight=w, reg_lambda=reg_lambda
            )
        }
        for k in range(split_g.shape[1])
    ]
    common = set.intersection(*(set(c) for c in candidates))
    gains = {c: sum(channel[c].gain for channel in candidates) for c in common}
    condition = min(common, key=lambda c: (-gains[c], c)) if common else None
    if condition is not None and gains[condition] <= 0:
        condition = None
    rows = tuple(range(len(x)))
    left, right = (rows, rows) if condition is None else _route(x, rows, condition)

    def leaf(selected):
        return tuple(
            newton_leaf(
                sum(w[i] * gradient[i, k] for i in selected),
                sum(w[i] for i in selected),
                reg_lambda=reg_lambda,
            )
            for k in range(raw.shape[1])
        )

    return VectorStump(x.shape[1], condition, leaf(left), leaf(right))
