"""Objective geometry over raw state, with explicit weight and offset semantics."""

import numpy as np

from .data import Problem, _owned
from .stats import newton


class Squared:
    """Scalar weighted mean half-square loss; derivatives are weighted once."""

    @staticmethod
    def validate(problem):
        if not isinstance(problem, Problem) or problem.target.shape[1] != 1:
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
