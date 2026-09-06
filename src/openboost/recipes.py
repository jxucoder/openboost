"""Readable CPU recipes composed from public objective, learner and state operations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

import numpy as np

from .artifacts import Model, TreeTerm
from .binning import Binning
from .objectives import Formula, Normal, Squared, diagonal_direction, full_direction
from .ops import _nonnegative, feasible, newton_leaf, score
from .runtime import AcceptedState, initialize, preview, propose_terms, resolve
from .stats import least_squares
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
    steps: tuple[SquaredStep | NormalStep | FormulaStep, ...]


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
    rate, learner = _configuration(
        rounds,
        learning_rate,
        max_depth,
        max_leaves,
        reg_lambda,
        min_child_h,
        split_penalty,
        step,
        max_trials,
        learner,
    )
    binned = Binning.fit(train.data, bins=bins).transform(train.data)
    state = initialize(context, train, validation, Squared.base(train), score=Squared.loss)
    steps = []
    for _ in range(rounds):
        before = state.train_raw
        gradient = Squared.gradient(train, before)
        loss_before = Squared.loss(train, before)
        tree = learner(binned, Squared.fields(train, before))
        state, coefficients, accepted, _failures = _trials(
            state, (TreeTerm(tree, [[1]]),), Squared.loss, loss_before, rate, step, max_trials
        )
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


def _configuration(
    rounds,
    learning_rate,
    max_depth,
    max_leaves,
    reg_lambda,
    min_child_h,
    split_penalty,
    step,
    max_trials,
    learner,
):
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
    return rate, learner


def _trials(state, terms, loss, loss_before, rate, policy, max_trials):
    # Structural errors are configuration failures, never rejected search trials.
    Model(
        state.model.feature_names,
        state.model.base,
        tuple(replace(t, coefficient=0.0) for t in terms),
    )
    coefficients, failures = [], []
    for trial in range(1 if policy == "fixed" else max_trials):
        alpha = rate * 0.5**trial
        coefficients.append(alpha)
        try:
            proposal = propose_terms(state, tuple(replace(t, coefficient=alpha) for t in terms))
            candidate = preview(state, proposal)
            candidate_loss = loss(state.train, candidate.predict(state.train.data))
            accepted = policy == "fixed" or candidate_loss < loss_before
            updated = resolve(state, proposal, accept=accepted, score=loss)
        except (ValueError, FloatingPointError, OverflowError) as error:
            if policy == "fixed":
                raise
            failures.append(type(error).__name__)
            continue
        failures.append(None)
        if accepted:
            return updated, tuple(coefficients), True, tuple(failures)
    return state, tuple(coefficients), False, tuple(failures)


@dataclass(frozen=True, eq=False)
class NormalStep:
    gradient: np.ndarray
    fisher_diagonal: np.ndarray
    direction: np.ndarray
    raw_before: np.ndarray
    raw_after: np.ndarray
    loss_before: float
    loss_after: float
    coefficients: tuple[float, ...]
    accepted: bool
    failures: tuple[str | None, ...]


def normal(
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
    step="backtracking",
    max_trials=6,
    learner=None,
    mode="natural",
    damping=0.0,
    minimum_scale=1e-6,
):
    """Joint mean/log-scale Normal updates using shared scalar learners and state.

    Geometry is computed from the accepted snapshot; both channel trees are fit
    once and committed/rejected together. Natural mode uses the diagonal Fisher,
    ordinary mode uses negative likelihood gradients. No ordered update is implied.
    """
    Normal.validate(train)
    Normal.validate(validation)
    rate, learner = _configuration(
        rounds,
        learning_rate,
        max_depth,
        max_leaves,
        reg_lambda,
        min_child_h,
        split_penalty,
        step,
        max_trials,
        learner,
    )
    # Validate mode/damping even for zero rounds.
    diagonal_direction([[0, 0]], [[1, 2]], mode=mode, damping=damping)
    base = Normal.base(train, minimum_scale=minimum_scale)
    binned = Binning.fit(train.data, bins=bins).transform(train.data)
    state = initialize(context, train, validation, base, score=Normal.loss)
    steps = []
    for _ in range(rounds):
        before = state.train_raw
        loss_before, gradient, metric = Normal.geometry(train, before)
        direction = diagonal_direction(gradient, metric, mode=mode, damping=damping)
        terms = tuple(
            TreeTerm(learner(binned, least_squares(train, direction[:, k])), np.eye(2)[k : k + 1])
            for k in range(2)
        )
        state, coefficients, accepted, failures = _trials(
            state, terms, Normal.loss, loss_before, rate, step, max_trials
        )
        steps.append(
            NormalStep(
                gradient,
                metric,
                direction,
                before,
                state.train_raw,
                loss_before,
                Normal.loss(train, state.train_raw),
                coefficients,
                accepted,
                failures,
            )
        )
    return FitResult(state, tuple(steps))


@dataclass(frozen=True, eq=False)
class FormulaStep:
    gradient: np.ndarray
    metric: np.ndarray
    direction: np.ndarray
    raw_before: np.ndarray
    raw_after: np.ndarray
    loss_before: float
    loss_after: float
    coefficients: tuple[float, ...]
    accepted: bool
    failures: tuple[str | None, ...]


def formula(
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
    step="backtracking",
    max_trials=6,
    learner=None,
    damping=0.1,
):
    """Joint Formula updates through full GGN directions and shared scalar trees."""
    Formula.validate(train)
    Formula.validate(validation)
    rate, learner = _configuration(
        rounds,
        learning_rate,
        max_depth,
        max_leaves,
        reg_lambda,
        min_child_h,
        split_penalty,
        step,
        max_trials,
        learner,
    )
    full_direction([[0, 0]], [[[1, 0], [0, 1]]], damping=damping)
    base = Formula.base(train)
    binned = Binning.fit(train.data, bins=bins).transform(train.data)
    state = initialize(context, train, validation, base, score=Formula.loss)
    steps = []
    for _ in range(rounds):
        before = state.train_raw
        loss_before, gradient, metric = Formula.geometry(train, before)
        direction = full_direction(gradient, metric, damping=damping)
        terms = tuple(
            TreeTerm(learner(binned, least_squares(train, direction[:, k])), np.eye(2)[k : k + 1])
            for k in range(2)
        )
        state, coefficients, accepted, failures = _trials(
            state, terms, Formula.loss, loss_before, rate, step, max_trials
        )
        steps.append(
            FormulaStep(
                gradient,
                metric,
                direction,
                before,
                state.train_raw,
                loss_before,
                Formula.loss(train, state.train_raw),
                coefficients,
                accepted,
                failures,
            )
        )
    return FitResult(state, tuple(steps))
