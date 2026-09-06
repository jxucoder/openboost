"""Owned routed residuals and scalar pinball leaf solvers."""

from dataclasses import dataclass

import numpy as np

from .binning import _array
from .data import Problem, _owned


@dataclass(frozen=True, eq=False)
class ResidualView:
    row_ids: np.ndarray
    residual: np.ndarray
    weight: np.ndarray

    def __post_init__(self):
        residual, weight = _owned(self.residual, ndim=1), _owned(self.weight, ndim=1)
        ids = np.asarray(self.row_ids)
        if (
            ids.shape != residual.shape
            or weight.shape != residual.shape
            or ids.dtype.kind not in "iu"
            or len(np.unique(ids)) != len(ids)
            or np.any(weight < 0)
        ):
            raise ValueError("aligned unique row IDs, residuals and nonnegative weights required")
        object.__setattr__(self, "row_ids", _array(ids, ids.dtype))
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True, eq=False)
class ResidualContext:
    problem: Problem
    residual: np.ndarray

    def __post_init__(self):
        residual = _owned(self.residual, ndim=1)
        if not isinstance(self.problem, Problem) or len(residual) != len(self.problem.target):
            raise ValueError("residuals must align with a problem")
        object.__setattr__(self, "residual", residual)

    def view(self, positions):
        rows = np.asarray(positions)
        if (
            rows.ndim != 1
            or not rows.size
            or rows.dtype.kind not in "iu"
            or np.any(rows < 0)
            or np.any(rows >= len(self.residual))
            or len(np.unique(rows)) != len(rows)
        ):
            raise ValueError("nonempty unique routed row positions required")
        return ResidualView(
            self.problem.row_ids[rows], self.residual[rows], self.problem.weight[rows]
        )


def quantile_leaf(view, total=None, names=None, *, q=0.5, penalty=0.0, anchor=0.0):
    """Minimize summed weighted pinball plus penalty*(value-anchor)**2/2.

    Zero penalty chooses the leftmost weighted quantile. Positive penalty gives
    the unique optimum by scanning the monotone subgradient at breakpoints.
    Additive total/names are accepted for the public routed solver callback.
    """
    if (
        not np.isscalar(q)
        or not np.isfinite(q)
        or not 0 < q < 1
        or not np.isscalar(penalty)
        or not np.isfinite(penalty)
        or penalty < 0
        or not np.isscalar(anchor)
        or not np.isfinite(anchor)
        or (penalty == 0 and anchor != 0)
    ):
        raise ValueError("q in (0,1), nonnegative penalty and a finite applicable anchor required")
    positive = view.weight > 0
    if not np.any(positive):
        if penalty > 0:
            return float(anchor)
        raise ValueError("unpenalized quantile requires positive leaf mass")
    residual, weight = view.residual[positive], view.weight[positive]
    order = np.argsort(residual, kind="stable")
    residual, weight = residual[order], weight[order]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if penalty == 0:
            mass = np.cumsum(weight / weight.max())
            return float(residual[min(np.searchsorted(mass, q * mass[-1]), len(mass) - 1)])
        values, indices = np.unique(residual, return_index=True)
        masses = np.add.reduceat(weight, indices)
        target = q * masses.sum()
        left = 0.0
        for value, mass in zip(values, masses, strict=True):
            root = anchor + (target - left) / penalty
            if root <= value:
                return float(root)
            left += mass
            if left - target + penalty * (value - anchor) >= 0:
                return float(value)
        result = anchor + (target - left) / penalty
    if not np.isfinite(result):
        raise ValueError("nonfinite penalized leaf")
    return float(result)
