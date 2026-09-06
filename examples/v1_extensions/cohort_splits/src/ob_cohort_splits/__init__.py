"""D2: independent cohort information constrains ordinary Newton splits."""

from functools import partial

import numpy as np

from openboost.ops import feasible, newton_leaf, score
from openboost.tree import depthwise


class CohortLearner:
    """Bind information columns to a problem, without adding feature columns.

    Each child needs at least one unit from every information column. Information
    is independent of objective weights, which have already been applied to G/H.
    """

    def __init__(self, problem, information, *, grower=depthwise, max_depth=2, reg_lambda=1.0):
        values = np.asarray(information, dtype=float)
        if (
            values.ndim != 2
            or values.shape[0] != len(problem.target)
            or values.shape[1] == 0
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
        ):
            raise ValueError("aligned finite nonnegative cohort information required")
        self.problem_identity = problem.identity
        self.information = np.frombuffer(values.tobytes(), dtype=float).reshape(values.shape)
        self.names = tuple(f"cohort:{i}" for i in range(values.shape[1]))
        self.grower, self.max_depth, self.reg_lambda = grower, max_depth, reg_lambda

    def legal(self, candidate):
        if not feasible(candidate):
            return False
        for name in self.names:
            i = candidate.names.index(name)
            if candidate.roles[i] != "independent":
                raise ValueError("cohort information must be independent of training weights")
            if candidate.left[i] < 1 or candidate.right[i] < 1:
                return False
        return True

    def __call__(self, binned, fields):
        if fields.problem_identity != self.problem_identity:
            raise ValueError("cohort information belongs to a different problem")
        for i, name in enumerate(self.names):
            fields = fields.add_independent(name, self.information[:, i])
        return self.grower(
            binned,
            fields,
            max_depth=self.max_depth,
            legality=self.legal,
            scoring=partial(score, reg_lambda=self.reg_lambda),
            leaf=partial(newton_leaf, reg_lambda=self.reg_lambda),
        )
