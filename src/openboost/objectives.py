"""Objective geometry over raw state, with explicit weight and offset semantics."""

import numpy as np

from .data import Problem, _owned
from .stats import newton


class Squared:
    """Scalar weighted mean half-square loss; derivatives are weighted once."""

    @staticmethod
    def validate(problem):
        if (
            not isinstance(problem, Problem)
            or problem.target.shape[1] != 1
            or problem.raw_width != 1
        ):
            raise ValueError("squared recipe requires scalar [N, 1] targets")

    @classmethod
    def base(cls, problem):
        cls.validate(problem)
        with np.errstate(over="raise", invalid="raise"):
            return _owned(
                np.sum(
                    (problem.target - problem.offset)
                    * (problem.weight / problem.weight.sum())[:, None],
                    axis=0,
                ),
                ndim=1,
            )

    @classmethod
    def gradient(cls, problem, raw):
        cls.validate(problem)
        with np.errstate(over="raise", invalid="raise"):
            return _owned((problem.with_offset(raw) - problem.target)[:, 0], ndim=1)

    @classmethod
    def fields(cls, problem, raw):
        return newton(problem, cls.gradient(problem, raw), np.ones(len(problem.target)))

    @classmethod
    def loss(cls, problem, raw):
        gradient = cls.gradient(problem, raw)
        with np.errstate(over="raise", invalid="raise"):
            result = float(np.dot(problem.weight / problem.weight.sum(), gradient**2 / 2))
        if not np.isfinite(result):
            raise ValueError("nonfinite squared loss")
        return result


class Normal:
    """Scalar observed targets, raw (mean, log-scale), diagonal Fisher geometry.

    Minimum scale regularizes initialization only. Runtime scale/precision must
    remain strictly positive and finite; invalid trials are never clipped.
    """

    @staticmethod
    def validate(problem):
        if (
            not isinstance(problem, Problem)
            or problem.target.shape[1] != 1
            or problem.raw_width != 2
        ):
            raise ValueError("Normal requires scalar targets and raw_width=2")

    @staticmethod
    def parameters(raw):
        values = _owned(raw, ndim=2)
        if values.shape[1] != 2:
            raise ValueError("Normal raw values require mean/log-scale columns")
        with np.errstate(over="raise", invalid="raise"):
            scale = np.exp(values[:, 1])
        if np.any(scale <= 0) or not np.isfinite(scale).all():
            raise ValueError("Normal scale must be positive and finite")
        return _owned(np.column_stack((values[:, 0], scale)), ndim=2)

    @classmethod
    def base(cls, problem, *, minimum_scale=1e-6):
        cls.validate(problem)
        if not np.isscalar(minimum_scale) or not np.isfinite(minimum_scale) or minimum_scale <= 0:
            raise ValueError("positive finite minimum_scale required")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            relative_precision = np.exp(-2 * problem.offset[:, 1])
            if np.any(relative_precision <= 0):
                raise ValueError("Normal offset precision underflow")
            mass = problem.weight / problem.weight.sum()
            mean_mass = mass * relative_precision
            centered = problem.target[:, 0] - problem.offset[:, 0]
            mean = np.dot(mean_mass / mean_mass.sum(), centered)
            variance = np.dot(mean_mass, (centered - mean) ** 2)
            base = _owned([mean, np.log(max(np.sqrt(variance), minimum_scale))], ndim=1)
        # Reject an unrepresentable initial predictive distribution.
        cls.loss(problem, np.broadcast_to(base, problem.offset.shape))
        return base

    @classmethod
    def geometry(cls, problem, raw):
        """Return loss, unweighted gradient and Fisher diagonal [N, 2]."""
        cls.validate(problem)
        values = problem.with_offset(raw)
        cls.parameters(values)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            precision = np.exp(-2 * values[:, 1])
            if np.any(precision <= 0):
                raise ValueError("Normal precision must be positive")
            residual = values[:, 0] - problem.target[:, 0]
            square = residual**2 * precision
            losses = values[:, 1] + square / 2 + 0.5 * np.log(2 * np.pi)
            gradient = _owned(np.column_stack((residual * precision, 1 - square)), ndim=2)
            fisher = _owned(np.column_stack((precision, np.full(len(values), 2.0))), ndim=2)
            loss = float(np.dot(problem.weight / problem.weight.sum(), losses))
        if not np.isfinite(loss):
            raise ValueError("nonfinite Normal loss")
        return loss, gradient, fisher

    @classmethod
    def loss(cls, problem, raw):
        return cls.geometry(problem, raw)[0]


def diagonal_direction(gradient, metric, *, mode="natural", damping=0.0):
    """Unweighted ordinary/Fisher-diagonal direction, before regression weights."""
    g, h = _owned(gradient, ndim=2), _owned(metric, ndim=2)
    if g.shape != h.shape or np.any(h <= 0):
        raise ValueError("aligned positive metric diagonal required")
    if not np.isscalar(damping) or not np.isfinite(damping) or damping < 0:
        raise ValueError("finite nonnegative damping required")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if mode == "ordinary":
            if damping != 0:
                raise ValueError("ordinary direction does not use damping")
            return _owned(-g, ndim=2)
        if mode != "natural":
            raise ValueError("ordinary/natural direction required")
        return _owned(-g / (h + damping), ndim=2)
