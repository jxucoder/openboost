"""D4 ordered updates composed exclusively from public CPU operations."""

from dataclasses import dataclass
from functools import partial

import numpy as np

from openboost.artifacts import TreeTerm
from openboost.binning import prepare_training
from openboost.diagnostics import TraceSummary, validate_retention
from openboost.objectives import Formula, Normal, diagonal_direction, full_direction
from openboost.runtime import initialize, preview, propose_terms, resolve
from openboost.stats import least_squares
from openboost.stopping import StopState
from openboost.tree import depthwise


@dataclass(frozen=True)
class Substep:
    channel: int
    before: object
    after: object
    loss_before: float
    loss_after: float
    coefficients: tuple
    failures: tuple


def validate_order(order):
    order = tuple(order)
    if len(order) != 2 or any(type(k) is not int for k in order) or set(order) != {0, 1}:
        raise ValueError("order must be a permutation of (0,1)")
    return order


def sweep(state, binned, *, geometry, direction, loss, order=(0, 1), learner=None):
    """One outer round; later parameters read only the latest accepted state.

    A learner is fitted once per parameter. Invalid learner construction fails;
    numerical candidate failures are recorded and rejected without state mutation.
    """
    order = validate_order(order)
    if state.train.raw_width != 2 or binned.data.identity != state.train.data.identity:
        raise ValueError("two-channel state and matching prepared data required")
    learner = partial(depthwise, max_depth=2) if learner is None else learner
    steps = []
    for channel in order:
        before = state
        initial_loss, gradient, metric = geometry(state.train, state.train_raw)
        if not np.isfinite(initial_loss):
            raise ValueError("finite accepted training loss required")
        values = np.asarray(direction(gradient, metric), dtype=float)
        if values.shape != state.train_raw.shape or not np.all(np.isfinite(values)):
            raise ValueError("finite aligned two-channel direction required")
        tree = learner(binned, least_squares(state.train, values[:, channel]))
        mapping = np.eye(2)[channel : channel + 1]
        TreeTerm(tree, mapping)  # Validate structural learner errors outside search.
        coefficients, failures = [], []
        for attempt in range(6):
            alpha = 0.1 * 0.5**attempt
            coefficients.append(alpha)
            try:
                proposal = propose_terms(state, (TreeTerm(tree, mapping, alpha),))
                candidate = preview(state, proposal)
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    value = float(loss(state.train, candidate.predict(state.train.data)))
                if not np.isfinite(value):
                    raise ValueError("nonfinite candidate loss")
                accepted = value < initial_loss
                state = resolve(state, proposal, accept=accepted, score=loss)
            except (ValueError, FloatingPointError, OverflowError) as error:
                failures.append(type(error).__name__)
                continue
            failures.append(None)
            if accepted:
                break
        steps.append(
            Substep(
                channel,
                before,
                state,
                float(initial_loss),
                float(loss(state.train, state.train_raw)),
                tuple(coefficients),
                tuple(failures),
            )
        )
    return state, tuple(steps)


@dataclass(frozen=True)
class OrderedResult:
    state: object
    steps: tuple
    stop: StopState


def fit(
    train,
    validation,
    *,
    context,
    objective,
    retention="full",
    direction,
    base,
    order=(0, 1),
    rounds=2,
    bins=254,
    prepared=None,
    patience=None,
    min_delta=0.0,
    learner=None,
):
    """Generic two-parameter ordered loop, with one stop observation per sweep."""
    retention = validate_retention(retention)
    order = validate_order(order)
    objective.validate(train)
    objective.validate(validation)
    state = initialize(context, train, validation, base, score=objective.loss)
    stop = StopState.start(state.best_score, rounds=rounds, patience=patience, min_delta=min_delta)
    binned = prepare_training(train.data, bins=bins, prepared=prepared)
    steps = []
    for _ in range(rounds):
        state, substeps = sweep(
            state,
            binned,
            geometry=objective.geometry,
            direction=direction,
            loss=objective.loss,
            order=order,
            learner=learner,
        )
        if retention == "summary":
            substeps = tuple(
                TraceSummary(
                    "OrderedSubstep",
                    (
                        ("channel", item.channel),
                        ("before_version", item.before.version),
                        ("after_version", item.after.version),
                        ("accepted", item.after.version > item.before.version),
                        ("loss_before", item.loss_before),
                        ("loss_after", item.loss_after),
                        ("coefficients", item.coefficients),
                        ("failures", item.failures),
                    ),
                    ("before", "after"),
                )
                for item in substeps
            )
        steps.append(substeps)
        stop = stop.observe(objective.loss(validation, state.validation_raw))
        if stop.reason is not None:
            break
    return OrderedResult(state, tuple(steps), stop)


def normal(train, validation, *, context, mode="natural", damping=0.0, **options):
    direction = partial(diagonal_direction, mode=mode, damping=damping)
    direction([[0, 0]], [[1, 2]])
    return fit(
        train,
        validation,
        context=context,
        objective=Normal,
        direction=direction,
        base=Normal.base(train),
        **options,
    )


def formula(train, validation, *, context, damping=0.1, **options):
    direction = partial(full_direction, damping=damping)
    direction([[0, 0]], [[[1, 0], [0, 1]]])
    return fit(
        train,
        validation,
        context=context,
        objective=Formula,
        direction=direction,
        base=Formula.base(train),
        **options,
    )
