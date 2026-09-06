"""Readable CPU recipes composed from public objective, learner and state operations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

import numpy as np

from .artifacts import Model, TreeTerm
from .binning import Binning
from .objectives import (
    Binary,
    Formula,
    Multiclass,
    Normal,
    Squared,
    diagonal_direction,
    full_direction,
)
from .ops import (
    _nonnegative,
    feasible,
    newton_leaf,
    score,
    vector_feasible,
    vector_leaf,
    vector_score,
)
from .runtime import AcceptedState, initialize, preview, propose_terms, resolve
from .stats import least_squares, newton, vector_newton
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
    steps: tuple[
        SquaredStep
        | NormalStep
        | FormulaStep
        | BinaryStep
        | MulticlassStep
        | RankingStep
        | QuantileStep
        | PoissonStep
        | GammaStep
        | TweedieStep
        | AFTStep
        | MultiSquaredStep,
        ...,
    ]


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
        state.model.classes,
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


@dataclass(frozen=True, eq=False)
class BinaryStep:
    gradient: np.ndarray
    curvature: np.ndarray
    raw_before: np.ndarray
    raw_after: np.ndarray
    loss_before: float
    loss_after: float
    coefficients: tuple[float, ...]
    accepted: bool
    failures: tuple[str | None, ...]


def binary(
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
    clip=1e-6,
):
    """Binary logistic boosting with persisted class order and shared transactions."""
    Binary.validate(train)
    Binary.validate(validation)
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
    base = Binary.base(train, clip=clip)
    binned = Binning.fit(train.data, bins=bins).transform(train.data)
    state = initialize(context, train, validation, base, score=Binary.loss)
    steps = []
    for _ in range(rounds):
        before = state.train_raw
        loss_before, gradient, curvature = Binary.geometry(train, before)
        tree = learner(binned, newton(train, gradient, curvature))
        state, coefficients, accepted, failures = _trials(
            state, (TreeTerm(tree, [[1]]),), Binary.loss, loss_before, rate, step, max_trials
        )
        steps.append(
            BinaryStep(
                gradient,
                curvature,
                before,
                state.train_raw,
                loss_before,
                Binary.loss(train, state.train_raw),
                coefficients,
                accepted,
                failures,
            )
        )
    return FitResult(state, tuple(steps))


@dataclass(frozen=True, eq=False)
class MulticlassStep:
    gradient: np.ndarray
    diagonal_bound: np.ndarray
    raw_before: np.ndarray
    raw_after: np.ndarray
    loss_before: float
    loss_after: float
    coefficients: tuple[float, ...]
    accepted: bool
    failures: tuple[str | None, ...]


def multiclass(
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
    """One joint vector tree per round from a common softmax raw snapshot."""
    Multiclass.validate(train)
    Multiclass.validate(validation)
    custom = learner is not None
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
    if not custom:
        learner = partial(
            depthwise,
            max_depth=max_depth,
            max_leaves=max_leaves,
            scoring=partial(vector_score, reg_lambda=reg_lambda, split_penalty=split_penalty),
            legality=partial(vector_feasible, min_child_h=min_child_h),
            leaf=partial(vector_leaf, reg_lambda=reg_lambda),
        )
    base = Multiclass.base(train)
    binned = Binning.fit(train.data, bins=bins).transform(train.data)
    state = initialize(context, train, validation, base, score=Multiclass.loss)
    steps = []
    for _ in range(rounds):
        before = state.train_raw
        loss_before, gradient, bound = Multiclass.geometry(train, before)
        tree = learner(binned, vector_newton(train, gradient, bound))
        state, coefficients, accepted, failures = _trials(
            state,
            (TreeTerm(tree, np.eye(train.raw_width)),),
            Multiclass.loss,
            loss_before,
            rate,
            step,
            max_trials,
        )
        steps.append(
            MulticlassStep(
                gradient,
                bound,
                before,
                state.train_raw,
                loss_before,
                Multiclass.loss(train, state.train_raw),
                coefficients,
                accepted,
                failures,
            )
        )
    return FitResult(state, tuple(steps))


@dataclass(frozen=True, eq=False)
class RankingStep:
    gradient: np.ndarray
    curvature: np.ndarray
    raw_before: np.ndarray
    raw_after: np.ndarray
    pair_loss: float


def ranking(
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
    lambdas=False,
    k=10,
    learner=None,
):
    """Fixed-step ranking; validation query-weighted NDCG selects the best model."""
    from .ranking import Ranking

    objective = Ranking(lambdas=lambdas, k=k)
    objective.validate(train)
    objective.validate(validation)
    rate, learner = _configuration(
        rounds,
        learning_rate,
        max_depth,
        max_leaves,
        reg_lambda,
        min_child_h,
        split_penalty,
        "fixed",
        1,
        learner,
    )
    binned = Binning.fit(train.data, bins=bins).transform(train.data)
    state = initialize(context, train, validation, [0.0], score=objective.score)
    steps = []
    for _ in range(rounds):
        before = state.train_raw
        geometry = objective.geometry(train, before)
        tree = learner(binned, newton(train, geometry.gradient, geometry.curvature))
        state = resolve(
            state,
            propose_terms(state, (TreeTerm(tree, [[1]], rate),)),
            accept=True,
            score=objective.score,
        )
        steps.append(
            RankingStep(
                geometry.gradient,
                geometry.curvature,
                before,
                state.train_raw,
                geometry.loss,
            )
        )
    return FitResult(state, tuple(steps))


@dataclass(frozen=True, eq=False)
class QuantileStep:
    raw_before: np.ndarray
    raw_after: np.ndarray
    loss_before: float
    loss_after: float
    coefficients: tuple[float, ...]
    accepted: bool
    failures: tuple[str | None, ...]


def quantile(
    train,
    validation,
    *,
    context,
    q=0.5,
    rounds=2,
    learning_rate=0.1,
    bins=254,
    max_depth=2,
    max_leaves=None,
    reg_lambda=1.0,
    min_child_h=0.0,
    split_penalty=0.0,
    penalty=0.0,
    anchor=0.0,
    step="fixed",
    max_trials=6,
    grower=depthwise,
):
    """Pinball splits with exact routed quantile or penalized residual leaves.

    reg_lambda affects split scoring; penalty is the distinct leaf penalty.
    Acceptance/validation use unpenalized prediction pinball loss.
    """
    from .leaves import ResidualContext, quantile_leaf
    from .objectives import Quantile

    objective = Quantile(q)
    objective.validate(train)
    objective.validate(validation)
    rate, _ = _configuration(
        rounds,
        learning_rate,
        max_depth,
        max_leaves,
        reg_lambda,
        min_child_h,
        split_penalty,
        step,
        max_trials,
        None,
    )
    base = objective.base(train)
    solver = partial(quantile_leaf, q=q, penalty=penalty, anchor=anchor)
    # Validate leaf configuration even for zero-round runs.
    solver(
        ResidualContext(train, objective.residuals(train, np.zeros_like(train.target))).view(
            np.arange(len(train.target))
        )
    )
    binned = Binning.fit(train.data, bins=bins).transform(train.data)
    state = initialize(context, train, validation, base, score=objective.loss)
    steps = []
    for _ in range(rounds):
        before = state.train_raw
        loss_before = objective.loss(train, before)
        tree = grower(
            binned,
            objective.fields(train, before),
            max_depth=max_depth,
            max_leaves=max_leaves,
            scoring=partial(score, reg_lambda=reg_lambda, split_penalty=split_penalty),
            legality=partial(feasible, min_child_h=min_child_h),
            row_leaf=solver,
            leaf_context=ResidualContext(train, objective.residuals(train, before)),
        )
        state, coefficients, accepted, failures = _trials(
            state,
            (TreeTerm(tree, [[1]]),),
            objective.loss,
            loss_before,
            rate,
            step,
            max_trials,
        )
        steps.append(
            QuantileStep(
                before,
                state.train_raw,
                loss_before,
                objective.loss(train, state.train_raw),
                coefficients,
                accepted,
                failures,
            )
        )
    return FitResult(state, tuple(steps))


@dataclass(frozen=True, eq=False)
class PoissonStep:
    gradient: np.ndarray
    curvature: np.ndarray
    raw_before: np.ndarray
    raw_after: np.ndarray
    loss_before: float
    loss_after: float
    coefficients: tuple[float, ...]
    accepted: bool
    failures: tuple[str | None, ...]


def poisson(
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
    minimum_rate=1e-6,
    step="fixed",
    max_trials=6,
    learner=None,
):
    """Scalar count boosting; exposure and offsets are applied once in geometry."""
    from .objectives import Poisson

    objective = Poisson(minimum_rate)
    objective.validate(train)
    objective.validate(validation)
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
    state = initialize(context, train, validation, objective.base(train), score=objective.loss)
    steps = []
    for _ in range(rounds):
        before = state.train_raw
        loss_before, gradient, curvature = objective.geometry(train, before)
        tree = learner(binned, newton(train, gradient, curvature))
        state, coefficients, accepted, failures = _trials(
            state,
            (TreeTerm(tree, [[1]]),),
            objective.loss,
            loss_before,
            rate,
            step,
            max_trials,
        )
        steps.append(
            PoissonStep(
                gradient,
                curvature,
                before,
                state.train_raw,
                loss_before,
                objective.loss(train, state.train_raw),
                coefficients,
                accepted,
                failures,
            )
        )
    return FitResult(state, tuple(steps))


@dataclass(frozen=True, eq=False)
class GammaStep:
    gradient: np.ndarray
    curvature: np.ndarray
    raw_before: np.ndarray
    raw_after: np.ndarray
    loss_before: float
    loss_after: float
    coefficients: tuple[float, ...]
    accepted: bool
    failures: tuple[str | None, ...]


def gamma(
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
    """Positive Gamma mean boosting with original weights and additive log-mean offsets."""
    from .objectives import Gamma

    objective = Gamma()
    objective.validate(train)
    objective.validate(validation)
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
    state = initialize(context, train, validation, objective.base(train), score=objective.loss)
    steps = []
    for _ in range(rounds):
        before = state.train_raw
        loss_before, gradient, curvature = objective.geometry(train, before)
        tree = learner(binned, newton(train, gradient, curvature))
        state, coefficients, accepted, failures = _trials(
            state,
            (TreeTerm(tree, [[1]]),),
            objective.loss,
            loss_before,
            rate,
            step,
            max_trials,
        )
        steps.append(
            GammaStep(
                gradient,
                curvature,
                before,
                state.train_raw,
                loss_before,
                objective.loss(train, state.train_raw),
                coefficients,
                accepted,
                failures,
            )
        )
    return FitResult(state, tuple(steps))


@dataclass(frozen=True, eq=False)
class TweedieStep:
    gradient: np.ndarray
    curvature: np.ndarray
    raw_before: np.ndarray
    raw_after: np.ndarray
    loss_before: float
    loss_after: float
    coefficients: tuple[float, ...]
    accepted: bool
    failures: tuple[str | None, ...]


def tweedie(
    train,
    validation,
    *,
    context,
    power=1.5,
    minimum_mean=1e-6,
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
    """Nonnegative fixed-power Tweedie mean boosting with original weights and additive log-mean offsets."""
    from .objectives import Tweedie

    objective = Tweedie(power, minimum_mean)
    objective.validate(train)
    objective.validate(validation)
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
    state = initialize(context, train, validation, objective.base(train), score=objective.loss)
    steps = []
    for _ in range(rounds):
        before = state.train_raw
        loss_before, gradient, curvature = objective.geometry(train, before)
        tree = learner(binned, newton(train, gradient, curvature))
        state, coefficients, accepted, failures = _trials(
            state,
            (TreeTerm(tree, [[1]]),),
            objective.loss,
            loss_before,
            rate,
            step,
            max_trials,
        )
        steps.append(
            TweedieStep(
                gradient,
                curvature,
                before,
                state.train_raw,
                loss_before,
                objective.loss(train, state.train_raw),
                coefficients,
                accepted,
                failures,
            )
        )
    return FitResult(state, tuple(steps))


@dataclass(frozen=True, eq=False)
class AFTStep:
    gradient: np.ndarray
    curvature: np.ndarray
    raw_before: np.ndarray
    raw_after: np.ndarray
    loss_before: float
    loss_after: float
    coefficients: tuple[float, ...]
    accepted: bool
    failures: tuple[str | None, ...]


def aft(
    train,
    validation,
    *,
    context,
    sigma=1.0,
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
    """Fixed-scale log-normal AFT with event/right-censored likelihood."""
    from .survival import LogNormalAFT

    objective = LogNormalAFT(sigma)
    objective.validate(train)
    objective.validate(validation)
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
    state = initialize(context, train, validation, objective.base(train), score=objective.loss)
    steps = []
    for _ in range(rounds):
        before = state.train_raw
        loss_before, gradient, curvature = objective.geometry(train, before)
        tree = learner(binned, newton(train, gradient, curvature))
        state, coefficients, accepted, failures = _trials(
            state,
            (TreeTerm(tree, [[1]]),),
            objective.loss,
            loss_before,
            rate,
            step,
            max_trials,
        )
        steps.append(
            AFTStep(
                gradient,
                curvature,
                before,
                state.train_raw,
                loss_before,
                objective.loss(train, state.train_raw),
                coefficients,
                accepted,
                failures,
            )
        )
    return FitResult(state, tuple(steps))


@dataclass(frozen=True, eq=False)
class MultiSquaredStep:
    gradient: np.ndarray
    raw_before: np.ndarray
    raw_after: np.ndarray
    mse_before: np.ndarray
    mse_after: np.ndarray
    coefficients: tuple[float, ...]
    accepted: bool
    failures: tuple[str | None, ...]


def multi_squared(
    train,
    validation,
    *,
    context,
    mode="shared",
    projection=None,
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
    grower=depthwise,
):
    """Independent or shared topology; all output updates commit atomically.

    projection [K,S] is a caller-declared split sketch; full K-dimensional
    statistics still solve leaves. It is supported only in shared mode.
    """
    from .data import _owned
    from .objectives import MultiSquared

    objective = MultiSquared
    objective.validate(train)
    objective.validate(validation)
    if mode not in ("shared", "independent") or not callable(grower):
        raise ValueError("shared/independent mode and callable grower required")
    if projection is not None:
        projection = _owned(projection, ndim=2)
        if (
            mode != "shared"
            or projection.shape[0] != train.raw_width
            or np.any(np.max(np.abs(projection), axis=0) == 0)
        ):
            raise ValueError("nonzero shared split projection columns must match output width")
    rate, _ = _configuration(
        rounds,
        learning_rate,
        max_depth,
        max_leaves,
        reg_lambda,
        min_child_h,
        split_penalty,
        step,
        max_trials,
        None,
    )
    binned = Binning.fit(train.data, bins=bins).transform(train.data)
    state = initialize(context, train, validation, objective.base(train), score=objective.loss)
    mapping = np.eye(train.raw_width)
    steps = []
    for _ in range(rounds):
        before = state.train_raw
        gradient = objective.gradient(train, before)
        curvature = np.ones_like(gradient)
        options = dict(max_depth=max_depth, max_leaves=max_leaves)
        if mode == "shared":
            leaves = vector_newton(train, gradient, curvature)
            with np.errstate(over="raise", invalid="raise"):
                fields = (
                    leaves
                    if projection is None
                    else vector_newton(train, gradient @ projection, curvature @ (projection**2))
                )
            tree = grower(
                binned,
                fields,
                **options,
                leaf_fields=leaves,
                scoring=partial(vector_score, reg_lambda=reg_lambda, split_penalty=split_penalty),
                legality=partial(vector_feasible, min_child_h=min_child_h),
                leaf=partial(vector_leaf, reg_lambda=reg_lambda),
            )
            terms = (TreeTerm(tree, mapping),)
        else:
            terms = tuple(
                TreeTerm(
                    grower(
                        binned,
                        newton(train, gradient[:, k], curvature[:, k]),
                        **options,
                        scoring=partial(score, reg_lambda=reg_lambda, split_penalty=split_penalty),
                        legality=partial(feasible, min_child_h=min_child_h),
                        leaf=partial(newton_leaf, reg_lambda=reg_lambda),
                    ),
                    mapping[k : k + 1],
                )
                for k in range(train.raw_width)
            )
        state, coefficients, accepted, failures = _trials(
            state,
            terms,
            objective.loss,
            objective.loss(train, before),
            rate,
            step,
            max_trials,
        )
        steps.append(
            MultiSquaredStep(
                gradient,
                before,
                state.train_raw,
                objective.mse(train, before),
                objective.mse(train, state.train_raw),
                coefficients,
                accepted,
                failures,
            )
        )
    return FitResult(state, tuple(steps))
