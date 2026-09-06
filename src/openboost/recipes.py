"""Readable CPU recipes composed from public objective, learner and state operations."""

from dataclasses import dataclass
from functools import partial

import numpy as np

from .artifacts import TreeTerm
from .binning import NumericBinning
from .objectives import Squared
from .ops import _nonnegative, feasible, newton_leaf, score
from .runtime import AcceptedState, initialize, preview, propose_terms, resolve
from .tree import depthwise


@dataclass(frozen=True, eq=False)
class SquaredStep:
    gradient: np.ndarray
    raw_before: np.ndarray
    raw_after: np.ndarray
    loss_before: float
    loss_after: float
    coefficients: tuple[float, ...]
    accepted: bool


@dataclass(frozen=True)
class FitResult:
    state: AcceptedState
    steps: tuple[SquaredStep, ...]


def squared(
    train,
    validation,
    *,
    context,
    rounds=2,
    learning_rate=0.1,
    bins=254,
    max_depth=2,
    max_leaves=None,
    reg_lambda=1.0,
    min_child_h=0.0,
    split_penalty=0.0,
    step="fixed",
    max_trials=6,
    learner=None,
):
    """Scalar squared boosting, returning final state and per-round evidence.

    Fixed steps always commit finite candidates. Backtracking tries alpha/2**j
    up to max_trials and requires strict training loss improvement. The learner
    is fitted once each round. Validation only selects best_model. Custom learner
    (binned_data, weighted_fields) replaces growth; tree options then are forbidden.
    """
    Squared.validate(train)
    Squared.validate(validation)
    if type(rounds) is not int or rounds < 0:
        raise ValueError("nonnegative integer rounds required")
    if type(max_depth) is not int or max_depth < 0:
        raise ValueError("nonnegative integer max_depth required")
    if max_leaves is not None and (type(max_leaves) is not int or not 1 <= max_leaves <= 2**30):
        raise ValueError("positive bounded integer max_leaves required")
    if (
        step not in ("fixed", "backtracking")
        or type(max_trials) is not int
        or not 1 <= max_trials <= 6
    ):
        raise ValueError("fixed/backtracking step and 1..6 trials required")
    rate = _nonnegative(learning_rate)
    regularizer = _nonnegative(reg_lambda)
    minimum = _nonnegative(min_child_h)
    penalty = _nonnegative(split_penalty)
    if learner is not None:
        if not callable(learner) or (max_depth, max_leaves, regularizer, minimum, penalty) != (
            2,
            None,
            1.0,
            0.0,
            0.0,
        ):
            raise ValueError("custom learner owns growth options")
    else:
        learner = partial(
            depthwise,
            max_depth=max_depth,
            max_leaves=max_leaves,
            scoring=partial(score, reg_lambda=regularizer, split_penalty=penalty),
            legality=partial(feasible, min_child_h=minimum),
            leaf=partial(newton_leaf, reg_lambda=regularizer),
        )
    binned = NumericBinning.fit(train.data, bins=bins).transform(train.data)
    state = initialize(context, train, validation, Squared.base(train), score=Squared.loss)
    steps = []
    for _ in range(rounds):
        before = state.train_raw
        gradient = Squared.gradient(train, before)
        loss_before = Squared.loss(train, before)
        tree = learner(binned, Squared.fields(train, before))
        coefficients, accepted = [], False
        for trial in range(1 if step == "fixed" else max_trials):
            alpha = rate * 0.5**trial
            coefficients.append(alpha)
            proposal = propose_terms(state, (TreeTerm(tree, [[1]], alpha),))
            candidate = preview(state, proposal)
            loss_after = Squared.loss(train, candidate.predict(train.data))
            accepted = step == "fixed" or loss_after < loss_before
            state = resolve(state, proposal, accept=accepted, score=Squared.loss)
            if accepted:
                break
        steps.append(
            SquaredStep(
                gradient,
                before,
                state.train_raw,
                loss_before,
                Squared.loss(train, state.train_raw),
                tuple(coefficients),
                accepted,
            )
        )
    return FitResult(state, tuple(steps))
