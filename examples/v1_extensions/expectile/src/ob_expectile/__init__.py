"""External D1 expectile objective and a public-operation CPU recipe."""

from dataclasses import dataclass
from numbers import Real

import numpy as np

from openboost.artifacts import TreeTerm
from openboost.binning import prepare_training
from openboost.objectives import Squared
from openboost.runtime import initialize, propose_terms, resolve
from openboost.stats import newton
from openboost.stopping import StopState
from openboost.tree import depthwise


@dataclass(frozen=True)
class Expectile:
    tau: float = 0.8

    def __post_init__(self):
        if isinstance(self.tau, bool) or not isinstance(self.tau, Real) or not 0 < self.tau < 1:
            raise ValueError("tau must be strictly between zero and one")

    def geometry(self, problem, raw):
        Squared.validate(problem)
        with np.errstate(over="raise", invalid="raise"):
            residual = (problem.target - problem.with_offset(raw))[:, 0]
            asymmetry = np.where(residual < 0, 1 - self.tau, self.tau)
            loss = float(np.dot(problem.weight / problem.weight.sum(), asymmetry * residual**2))
            gradient = -2 * asymmetry * residual
            curvature = 2 * asymmetry
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            raise ValueError("nonfinite expectile geometry")
        return loss, gradient, curvature

    def loss(self, problem, raw):
        return self.geometry(problem, raw)[0]

    def base(self, problem):
        """Bisection of the weighted derivative, unlike the interval oracle."""
        Squared.validate(problem)
        with np.errstate(over="raise", invalid="raise"):
            target = (problem.target - problem.offset)[:, 0]
            active = problem.weight > 0
            y = target[active]
            weight = problem.weight[active] / problem.weight.sum()
            low, high = float(y.min()), float(y.max())
            for _ in range(100):
                middle = low / 2 + high / 2
                residual = y - middle
                derivative = np.dot(
                    weight, np.where(residual < 0, 1 - self.tau, self.tau) * residual
                )
                if derivative > 0:
                    low = middle
                else:
                    high = middle
            base = np.array([low / 2 + high / 2])
        self.loss(problem, np.broadcast_to(base, (len(target), 1)))
        return base


@dataclass(frozen=True)
class Result:
    state: object
    steps: tuple
    stop: StopState


def fit(
    train,
    validation,
    *,
    context,
    tau=0.8,
    rounds=2,
    learning_rate=0.1,
    bins=254,
    prepared=None,
    patience=None,
    min_delta=0.0,
):
    """Fixed depth-two Newton updates; unsupported options raise TypeError.

    Derivatives are unweighted; newton applies training weights once. Raw models
    exclude offsets, which callers must supply separately for final predictions.
    """
    objective = Expectile(tau)
    Squared.validate(train)
    Squared.validate(validation)
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, Real)
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be positive and finite")
    binned = prepare_training(train.data, bins=bins, prepared=prepared)
    state = initialize(context, train, validation, objective.base(train), score=objective.loss)
    stop = StopState.start(state.best_score, rounds=rounds, patience=patience, min_delta=min_delta)
    steps = []
    for _ in range(rounds):
        before = state
        _, g, h = objective.geometry(train, state.train_raw)
        tree = depthwise(binned, newton(train, g, h))
        proposal = propose_terms(state, (TreeTerm(tree, [[1]], learning_rate),))
        state = resolve(state, proposal, accept=True, score=objective.loss)
        steps.append((before.train_raw, state.train_raw))
        stop = stop.observe(objective.loss(validation, state.validation_raw))
        if stop.reason is not None:
            break
    return Result(state, tuple(steps), stop)
