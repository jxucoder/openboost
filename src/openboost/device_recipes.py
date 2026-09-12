"""Experimental resident scalar recipe assembled from public device operations."""

from dataclasses import dataclass, replace

import numpy as np

from . import device_aft as aft_operations
from . import device_glm as glm_operations
from . import device_multi_squared as multi_squared_operations
from . import device_multiclass as multiclass_operations
from . import device_newton_order as exact_newton
from . import device_normal as normal_operations
from . import device_objectives as objectives
from . import device_tree as trees
from .comparison import LossChange
from .device import _parameter, _workspace
from .device_newton_leaf import leaf as exact_newton_leaf
from .device_runtime import DeviceRun, DeviceState, DeviceTerm
from .stopping import StopState


@dataclass(frozen=True)
class DeviceStep:
    round_index: int
    coefficients: tuple[float, ...]
    accepted: bool
    failures: tuple[str, ...]
    loss: float
    validation_score: float
    best_score: float


@dataclass(frozen=True)
class DeviceTrial:
    coefficient: float
    loss: float | None
    validation_score: float | None
    accepted: bool
    failure: str | None
    comparison: LossChange | None = None


@dataclass(frozen=True)
class DeviceNormalStep:
    round_index: int
    channels: tuple[int, ...]
    before_version: int
    after_version: int
    trials: tuple[DeviceTrial, ...]
    loss: float
    validation_score: float
    best_score: float
    validation_change: LossChange | None = None

    @property
    def accepted(self):
        return self.after_version > self.before_version

    @property
    def coefficients(self):
        return tuple(t.coefficient for t in self.trials)

    @property
    def failures(self):
        return tuple(t.failure for t in self.trials)


@dataclass(frozen=True)
class DeviceMulticlassStep(DeviceNormalStep):
    """One joint K-tree round, including its independent patience observation."""


@dataclass(frozen=True)
class DeviceFitResult:
    """Owned run/state plus scalar history. Release the run after exporting artifacts."""

    run: DeviceRun
    state: DeviceState
    steps: "tuple[DeviceStep | DeviceNormalStep | DeviceScalarStep | DeviceMulticlassStep, ...]"
    stop: StopState


@dataclass(frozen=True)
class DeviceScalarStep:
    """One scalar outer round with every trial and its separate patience comparison."""

    round_index: int
    before_version: int
    after_version: int
    trials: tuple[DeviceTrial, ...]
    loss: float
    validation_score: float
    best_score: float
    validation_change: LossChange

    @property
    def accepted(self):
        return self.after_version > self.before_version


def squared(
    ops,
    train,
    validation,
    *,
    run_id,
    seed,
    rounds=2,
    learning_rate=0.1,
    max_depth=2,
    reg_lambda=1.0,
    min_child_h=0.0,
    split_penalty=0.0,
    binning=None,
    bins=254,
    learner=None,
    step="fixed",
    max_trials=6,
    patience=None,
    min_delta=0.0,
    prepared=None,
):
    """Scalar squared rounds, with explicit fixed/backtracking acceptance and stopping.

    learner(ops, data, fields) may compose public device operations. Its returned
    tree is snapshotted; new callback scratch is temporary. Each proposal snapshots
    the learner again to separate proposal lifetime from caller workspace. History
    stores scalars, never past [N, 1] raw arrays. Retained final/best trees share
    immutable owned terms. No CPU fallback or extension of CPU run_many is implied.
    prepared may borrow explicit training/validation feature encodings; keep them
    live through fitting. Run-specific bindings and all mutable state stay owned.
    """
    policy = StopState.start(0, rounds=rounds, patience=patience, min_delta=min_delta)
    rate = _parameter(learning_rate)
    regularization, minimum, penalty = map(_parameter, (reg_lambda, min_child_h, split_penalty))
    if type(max_depth) is not int or max_depth < 0:
        raise ValueError("nonnegative integer max_depth required")
    if (
        step not in ("fixed", "backtracking")
        or type(max_trials) is not int
        or not 1 <= max_trials <= 6
    ):
        raise ValueError("fixed/backtracking step and 1..6 trials required")
    if learner is not None and (
        not callable(learner) or (max_depth, reg_lambda, min_child_h, split_penalty) != (2, 1, 0, 0)
    ):
        raise ValueError("supplied learner owns its tree configuration")
    run = DeviceRun(ops, train, validation, run_id=run_id, seed=seed, binning=binning, bins=bins,
                    prepared=prepared)
    try:
        state = run.initialize()
        policy = StopState.start(
            state.validation_score, rounds=rounds, patience=patience, min_delta=min_delta
        )
        history = []
        while policy.reason is None:
            with _workspace(ops) as retained:
                fields = run.fields(state)
                tree = (
                    learner(ops, run.data, fields)
                    if learner is not None
                    else trees.depthwise(
                        ops,
                        run.data,
                        fields,
                        binning=run.binning,
                        max_depth=max_depth,
                        reg_lambda=regularization,
                        min_child_h=minimum,
                        split_penalty=penalty,
                    )
                )
                tree = trees.copy(ops, tree)
                retained.add(tree)
            coefficients, failures, accepted = [], [], False
            try:
                for trial in range(1 if step == "fixed" else max_trials):
                    coefficient = float(rate) * 0.5**trial
                    coefficients.append(coefficient)
                    proposal = None
                    try:
                        proposal = run.propose(state, tree, coefficient=coefficient)
                        accepted = step == "fixed" or proposal.loss < state.loss
                        updated = run.resolve(state, proposal, accept=accepted)
                    except (ValueError, FloatingPointError, OverflowError) as error:
                        if step == "fixed":
                            raise
                        accepted = False
                        failures.append(type(error).__name__ + ": " + str(error))
                        continue
                    finally:
                        if proposal is not None:
                            run.release(proposal)
                    if accepted:
                        run.release(state)
                        state = updated
                        break
            finally:
                ops.release(tree)
            history.append(
                DeviceStep(
                    policy.completed_rounds,
                    tuple(coefficients),
                    accepted,
                    tuple(failures),
                    state.loss,
                    state.validation_score,
                    state.best_score,
                )
            )
            policy = policy.observe(state.validation_score)
        return DeviceFitResult(run, state, tuple(history), policy)
    except BaseException:
        run.close()
        raise


def _search_configuration(step, max_trials, learning_rate):
    if (
        step not in ("fixed", "backtracking")
        or type(max_trials) is not int
        or not 1 <= max_trials <= 6
    ):
        raise ValueError("fixed/backtracking step and 1..6 trials required")
    return float(_parameter(learning_rate))


def try_terms(run, state, terms, *, learning_rate=0.1, step="backtracking", max_trials=6):
    """Search one atomic tuple; caller owns terms, prior state and returned state.

    Structural errors fail before search. Numerical failures are retained as
    scalar trial diagnostics. Infrastructure errors propagate. Every proposal is
    released; an accepted state's immutable tree snapshots remain independently owned.
    """
    rate = _search_configuration(step, max_trials, learning_rate)
    run.validate_state(state)
    terms = run.validate_terms(terms)
    trials = []
    for j in range(1 if step == "fixed" else max_trials):
        coefficient = rate * 0.5**j
        proposal = None
        change = None
        try:
            proposal = run.propose_terms(state, terms, coefficient=coefficient)
            if run.comparison == "objective":
                change = run.compare(state, proposal)
            improved = proposal.loss < state.loss if change is None else change.improves()
            accepted = step == "fixed" or improved
            updated = run.resolve(state, proposal, accept=accepted)
            item = DeviceTrial(
                coefficient, proposal.loss, proposal.validation_score, accepted, None, change
            )
        except (ValueError, FloatingPointError, OverflowError) as error:
            if step == "fixed":
                raise
            trials.append(
                DeviceTrial(
                    coefficient,
                    None if proposal is None else proposal.loss,
                    None if proposal is None else proposal.validation_score,
                    False,
                    type(error).__name__ + ": " + str(error),
                    change,
                )
            )
            continue
        finally:
            if proposal is not None:
                run.release(proposal)
        trials.append(item)
        if accepted:
            return updated, tuple(trials)
    return state, tuple(trials)


def normal(
    ops,
    train,
    validation,
    *,
    run_id,
    seed,
    rounds=2,
    mode="natural",
    damping=0.0,
    update="joint",
    minimum_scale=1e-6,
    learning_rate=0.1,
    max_depth=2,
    reg_lambda=1.0,
    min_child_h=0.0,
    split_penalty=0.0,
    binning=None,
    bins=254,
    learner=None,
    step="backtracking",
    max_trials=6,
    patience=None,
    min_delta=0.0,
):
    """Compose Normal geometry, scalar direction fits and joint/ordered transactions.

    update is joint, forward (mean then scale), or reverse. Ordered geometry reads
    only the latest accepted state; StopState observes once per outer sweep.
    Optional learner(ops, data, fields) owns its tree configuration. Summary
    diagnostics retain every trial and substep, never sample arrays or old states.
    Backtracking, best and patience use distinct objective-comparison anchors.
    Default trees use exact stored-field ordering and original-row Newton leaves.
    """
    policy = StopState.start(0, rounds=rounds, patience=patience, min_delta=min_delta)
    mode, damping = objectives.direction_configuration(mode, damping)
    objective = normal_operations.objective(minimum_scale=minimum_scale)
    rate = _search_configuration(step, max_trials, learning_rate)
    regularization, minimum, penalty = map(_parameter, (reg_lambda, min_child_h, split_penalty))
    if type(max_depth) is not int or max_depth < 0:
        raise ValueError("nonnegative integer max_depth required")
    if update not in ("joint", "forward", "reverse"):
        raise ValueError("joint/forward/reverse update required")
    if learner is not None and (
        not callable(learner) or (max_depth, reg_lambda, min_child_h, split_penalty) != (2, 1, 0, 0)
    ):
        raise ValueError("supplied learner owns its tree configuration")
    groups = {"joint": ((0, 1),), "forward": ((0,), (1,)), "reverse": ((1,), (0,))}[update]
    def ordering(o, batch):
        ranked = exact_newton.rank(
            o, batch, reg_lambda=regularization, min_child_h=minimum, split_penalty=penalty
        )
        return exact_newton.choose(o, ranked)

    def field_leaf(o, fields, rows):
        return exact_newton_leaf(o, fields, rows, reg_lambda=regularization)

    run = DeviceRun(
        ops,
        train,
        validation,
        run_id=run_id,
        seed=seed,
        binning=binning,
        bins=bins,
        objective=objective,
        comparison="objective",
    )
    patience_raw = None
    try:
        state = run.initialize()
        policy = StopState.start(
            state.validation_score, rounds=rounds, patience=patience, min_delta=min_delta
        )
        patience_raw = run.raw(state, validation=True)
        history = []
        while policy.reason is None:
            for channels in groups:
                before_version = state.version
                with _workspace(ops) as retained:
                    raw = run.raw(state)
                    gradient, fisher = normal_operations.geometry(ops, run.problem, raw)
                    direction = objectives.diagonal_direction(
                        ops, gradient, fisher, mode=mode, damping=damping
                    )
                    terms = []
                    for k in channels:
                        fields = objectives.least_squares(ops, run.data, direction, k)
                        tree = (
                            learner(ops, run.data, fields)
                            if learner is not None
                            else trees.depthwise(
                                ops,
                                run.data,
                                fields,
                                binning=run.binning,
                                max_depth=max_depth,
                                ordering=ordering,
                                field_leaf=field_leaf,
                            )
                        )
                        snapshot = trees.copy(ops, tree)
                        terms.append(DeviceTerm(snapshot, np.eye(2)[k : k + 1]))
                        retained.add(snapshot)
                try:
                    updated, trials = try_terms(
                        run, state, terms, learning_rate=rate, step=step, max_trials=max_trials
                    )
                    if updated is not state:
                        run.release(state)
                        state = updated
                finally:
                    for term in terms:
                        ops.release(term.tree)
                history.append(
                    DeviceNormalStep(
                        policy.completed_rounds,
                        channels,
                        before_version,
                        state.version,
                        trials,
                        state.loss,
                        state.validation_score,
                        state.best_score,
                    )
                )
            with _workspace(ops) as retained:
                current = run.raw(state, validation=True)
                change = run.objective.loss_change(
                    ops, run.validation_problem, patience_raw, current
                )
                observed = policy.observe_change(state.validation_score, change)
                improved = change.improves(policy.min_delta)
                if improved:
                    retained.add(current)
            if improved:
                previous, patience_raw = patience_raw, current
                run.execution.release(previous)
            policy = observed
            # One observation per complete outer sweep, attached to its last substep.
            history[-1] = replace(history[-1], validation_change=change)
        run.execution.release(patience_raw)
        patience_raw = None
        return DeviceFitResult(run, state, tuple(history), policy)
    except BaseException:
        run.close()
        raise
    finally:
        if patience_raw is not None:
            run.execution.release(patience_raw)


def scalar(
    ops,
    train,
    validation,
    *,
    objective,
    run_id,
    seed,
    rounds=2,
    learning_rate=0.1,
    max_depth=2,
    reg_lambda=1.0,
    min_child_h=0.0,
    split_penalty=0.0,
    binning=None,
    bins=254,
    learner=None,
    step="backtracking",
    max_trials=6,
    patience=None,
    min_delta=0.0,
):
    """Compose scalar fields, tree operations and three objective-comparison anchors.

    ObjectiveOperations owns preparation, fields, loss and comparison. Optional
    learner(ops, data, fields) owns its tree policy. Final run/state are caller-owned;
    all working fields, trees, old states, proposals and patience snapshots are freed.
    New binary/Poisson device recipe validation is pending.
    """
    policy = StopState.start(0, rounds=rounds, patience=patience, min_delta=min_delta)
    rate = _search_configuration(step, max_trials, learning_rate)
    regularization, minimum, penalty = map(_parameter, (reg_lambda, min_child_h, split_penalty))
    if type(max_depth) is not int or max_depth < 0:
        raise ValueError("nonnegative integer max_depth required")
    if learner is not None and (
        not callable(learner) or (max_depth, reg_lambda, min_child_h, split_penalty) != (2, 1, 0, 0)
    ):
        raise ValueError("supplied learner owns its tree configuration")
    if not isinstance(objective, objectives.ObjectiveOperations):
        raise ValueError("explicit ObjectiveOperations required")
    if objective.compare is None or objective.fields is None:
        raise NotImplementedError("scalar recipe requires objective fields and loss-change")
    objective.validate(train)
    objective.validate(validation)
    if train.raw_width != 1 or validation.raw_width != 1:
        raise ValueError("scalar recipe requires raw_width=1")
    run = DeviceRun(
        ops,
        train,
        validation,
        run_id=run_id,
        seed=seed,
        binning=binning,
        bins=bins,
        objective=objective,
        comparison="objective",
    )
    patience_raw = None
    try:
        state = run.initialize()
        policy = StopState.start(
            state.validation_score, rounds=rounds, patience=patience, min_delta=min_delta
        )
        patience_raw = run.raw(state, validation=True)
        history = []
        while policy.reason is None:
            before_version = state.version
            with _workspace(ops) as retained:
                fields = run.fields(state)
                tree = (
                    learner(ops, run.data, fields)
                    if learner is not None
                    else trees.depthwise(
                        ops,
                        run.data,
                        fields,
                        binning=run.binning,
                        max_depth=max_depth,
                        reg_lambda=regularization,
                        min_child_h=minimum,
                        split_penalty=penalty,
                    )
                )
                tree = trees.copy(ops, tree)
                retained.add(tree)
            try:
                updated, trials = try_terms(
                    run,
                    state,
                    (DeviceTerm(tree, [[1]]),),
                    learning_rate=rate,
                    step=step,
                    max_trials=max_trials,
                )
                if updated is not state:
                    run.release(state)
                    state = updated
            finally:
                ops.release(tree)
            with _workspace(ops) as retained:
                current = run.raw(state, validation=True)
                change = run.objective.loss_change(
                    ops, run.validation_problem, patience_raw, current
                )
                observed = policy.observe_change(state.validation_score, change)
                improved = change.improves(policy.min_delta)
                if improved:
                    retained.add(current)
            if improved:
                previous, patience_raw = patience_raw, current
                run.execution.release(previous)
            history.append(
                DeviceScalarStep(
                    policy.completed_rounds,
                    before_version,
                    state.version,
                    trials,
                    state.loss,
                    state.validation_score,
                    state.best_score,
                    change,
                )
            )
            policy = observed
        return DeviceFitResult(run, state, tuple(history), policy)
    except BaseException:
        run.close()
        raise
    finally:
        if patience_raw is not None:
            run.execution.release(patience_raw)


def binary(ops, train, validation, *, clip=1e-6, **configuration):
    """Experimental binary scalar recipe; see scalar() for execution configuration."""
    return scalar(
        ops, train, validation, objective=glm_operations.binary(clip=clip), **configuration
    )


def poisson(ops, train, validation, *, minimum_rate=1e-6, **configuration):
    """Experimental exposure-aware count recipe, exporting a raw log-rate Model."""
    return scalar(
        ops,
        train,
        validation,
        objective=glm_operations.poisson(minimum_rate=minimum_rate),
        **configuration,
    )


def aft(ops, train, validation, *, sigma=1.0, **configuration):
    """Fixed-scale event/right-censored scalar recipe with three comparison anchors.

    Return the caller-owned run/state; use device_aft.export to retain the fitting
    scale for CPU inference. Configuration and ownership follow scalar().
    AFT CUDA lowering and recipe validation remain pending.
    """
    return scalar(ops, train, validation, objective=aft_operations.objective(sigma), **configuration)


def multiclass(
    ops,
    train,
    validation,
    *,
    run_id,
    seed,
    rounds=2,
    learning_rate=0.1,
    max_depth=2,
    reg_lambda=1.0,
    min_child_h=0.0,
    split_penalty=0.0,
    binning=None,
    bins=254,
    learner=None,
    step="backtracking",
    max_trials=6,
    patience=None,
    min_delta=0.0,
):
    """Fit K scalar trees from one geometry snapshot and propose them atomically.

    Acceptance, best validation and patience use independent comparison anchors.
    Optional learner(ops, data, fields) owns its tree policy; each returned tree is
    snapshotted before the next class. Final run/state are caller-owned. This is
    separate class topology; the CPU convenience recipe retains shared topology.
    Multiclass CUDA recipe validation is pending.
    """
    policy = StopState.start(0, rounds=rounds, patience=patience, min_delta=min_delta)
    rate = _search_configuration(step, max_trials, learning_rate)
    regularization, minimum, penalty = map(_parameter, (reg_lambda, min_child_h, split_penalty))
    if type(max_depth) is not int or max_depth < 0:
        raise ValueError("nonnegative integer max_depth required")
    if learner is not None and (
        not callable(learner) or (max_depth, reg_lambda, min_child_h, split_penalty) != (2, 1, 0, 0)
    ):
        raise ValueError("supplied learner owns its tree configuration")
    objective = multiclass_operations.objective()
    objective.validate(train)
    objective.validate(validation)
    channels = tuple(range(train.raw_width))
    mapping = np.eye(train.raw_width)
    run = DeviceRun(
        ops,
        train,
        validation,
        run_id=run_id,
        seed=seed,
        binning=binning,
        bins=bins,
        objective=objective,
        comparison="objective",
    )
    patience_raw = None
    try:
        state = run.initialize()
        policy = StopState.start(
            state.validation_score, rounds=rounds, patience=patience, min_delta=min_delta
        )
        patience_raw = run.raw(state, validation=True)
        history = []
        while policy.reason is None:
            before_version = state.version
            with _workspace(ops) as retained:
                raw = run.raw(state)
                gradient, bound = multiclass_operations.geometry(ops, run.problem, raw)
                terms = []
                for channel in channels:
                    fields = multiclass_operations.channel_fields(
                        ops, run.data, gradient, bound, channel=channel
                    )
                    tree = (
                        learner(ops, run.data, fields)
                        if learner is not None
                        else trees.depthwise(
                            ops, run.data, fields, binning=run.binning, max_depth=max_depth,
                            reg_lambda=regularization, min_child_h=minimum, split_penalty=penalty,
                        )
                    )
                    snapshot = trees.copy(ops, tree)
                    terms.append(DeviceTerm(snapshot, mapping[channel : channel + 1]))
                    retained.add(snapshot)
            try:
                updated, trials = try_terms(
                    run,
                    state,
                    terms,
                    learning_rate=rate,
                    step=step,
                    max_trials=max_trials,
                )
                if updated is not state:
                    run.release(state)
                    state = updated
            finally:
                for term in terms:
                    ops.release(term.tree)
            with _workspace(ops) as retained:
                current = run.raw(state, validation=True)
                change = run.objective.loss_change(
                    ops, run.validation_problem, patience_raw, current
                )
                observed = policy.observe_change(state.validation_score, change)
                improved = change.improves(policy.min_delta)
                if improved:
                    retained.add(current)
            if improved:
                previous, patience_raw = patience_raw, current
                run.execution.release(previous)
            history.append(
                DeviceMulticlassStep(
                    policy.completed_rounds,
                    channels,
                    before_version,
                    state.version,
                    trials,
                    state.loss,
                    state.validation_score,
                    state.best_score,
                    change,
                )
            )
            policy = observed
        return DeviceFitResult(run, state, tuple(history), policy)
    except BaseException:
        run.close()
        raise
    finally:
        if patience_raw is not None:
            run.execution.release(patience_raw)


def mapped(
    ops, train, validation, *, objective, learner, run_id, seed, rounds=2,
    learning_rate=0.1, binning=None, bins=254, step="backtracking", max_trials=6,
    patience=None, min_delta=0.0,
):
    """Public joint-term recipe with independent acceptance, best and patience anchors.

    learner(run, raw) returns DeviceTerms. Raw and new learner workspace are
    temporary; supplied preexisting trees are borrowed and every term is copied.
    The returned run owns its accepted state. Geometry and topology stay in the learner.
    """
    StopState.start(0, rounds=rounds, patience=patience, min_delta=min_delta)
    rate = _search_configuration(step, max_trials, learning_rate)
    if not callable(learner):
        raise ValueError("callable joint term learner required")
    if not isinstance(objective, objectives.ObjectiveOperations) or objective.compare is None:
        raise ValueError("objective with explicit comparison required")
    run = DeviceRun(ops, train, validation, run_id=run_id, seed=seed, binning=binning,
                    bins=bins, objective=objective, comparison="objective")
    patience_raw = None
    try:
        state = run.initialize()
        policy = StopState.start(state.validation_score, rounds=rounds, patience=patience, min_delta=min_delta)
        patience_raw = run.raw(state, validation=True)
        history = []
        while policy.reason is None:
            before_version = state.version
            with _workspace(ops) as retained:
                raw = run.raw(state)
                supplied = run.validate_terms(learner(run, raw))
                terms = tuple(DeviceTerm(trees.copy(ops, term.tree), term.mapping) for term in supplied)
                retained.update(term.tree for term in terms)
            try:
                updated, trials = try_terms(run, state, terms, learning_rate=rate, step=step, max_trials=max_trials)
                if updated is not state:
                    run.release(state)
                    state = updated
            finally:
                for term in terms:
                    ops.release(term.tree)
            with _workspace(ops) as retained:
                current = run.raw(state, validation=True)
                change = run.objective.loss_change(ops, run.validation_problem, patience_raw, current)
                observed = policy.observe_change(state.validation_score, change)
                improved = change.improves(policy.min_delta)
                if improved:
                    retained.add(current)
            if improved:
                previous, patience_raw = patience_raw, current
                run.execution.release(previous)
            history.append(DeviceScalarStep(policy.completed_rounds, before_version, state.version,
                                            trials, state.loss, state.validation_score, state.best_score, change))
            policy = observed
        return DeviceFitResult(run, state, tuple(history), policy)
    except BaseException:
        run.close()
        raise
    finally:
        if patience_raw is not None:
            run.execution.release(patience_raw)


def multi_squared(
    ops, train, validation, *, mode="shared", projection=None, max_depth=2,
    reg_lambda=1.0, min_child_h=0.0, split_penalty=0.0, learner=None, **configuration,
):
    """Required independent/shared vector squared recipe, using public mapped rounds.

    Optional learner(ops, data, split_fields, leaf_fields, output_width) owns its
    tree parameters. A [K,S] projection changes shared split fields only. Scaling
    is explicit through TargetScale/MultiOutputModel. CUDA validation is pending.
    """
    if mode not in ("independent", "shared"):
        raise ValueError("independent or shared topology required")
    if mode == "independent" and projection is not None:
        raise ValueError("projection requires shared topology")
    objective = multi_squared_operations.objective()
    objective.validate(train)
    objective.validate(validation)
    projection = objectives._projection(projection, train.raw_width)
    regularization, minimum, penalty = map(_parameter, (reg_lambda, min_child_h, split_penalty))
    if type(max_depth) is not int or max_depth < 0:
        raise ValueError("nonnegative integer max_depth required")
    if learner is not None and (
        not callable(learner) or (max_depth, reg_lambda, min_child_h, split_penalty) != (2, 1, 0, 0)
    ):
        raise ValueError("supplied learner owns its tree configuration")
    mapping = np.eye(train.raw_width, dtype=np.float32)

    def learn(run, raw):
        gradient, curvature = multi_squared_operations.geometry(ops, run.problem, raw)
        terms = []
        for channel in (range(run.raw_width) if mode == "independent" else (None,)):
            if channel is None:
                leaves = objectives.vector_fields(ops, run.data, gradient, curvature)
                fields = leaves if projection is None else objectives.vector_fields(
                    ops, run.data, gradient, curvature, projection=projection)
                width, output_map = run.raw_width, mapping
            else:
                fields = objectives.channel_fields(ops, run.data, gradient, curvature, channel=channel)
                leaves, width, output_map = fields, 1, mapping[channel:channel+1]
            if learner is not None:
                tree = learner(ops, run.data, fields, leaves, width)
            elif channel is not None:
                tree = trees.depthwise(ops, run.data, fields, binning=run.binning, max_depth=max_depth,
                                       reg_lambda=regularization, min_child_h=minimum, split_penalty=penalty)
            else:
                tree = trees.depthwise(
                    ops, run.data, fields, binning=run.binning, max_depth=max_depth,
                    output_width=width, leaf_fields=leaves,
                    scoring=lambda o, b: o.vector_scores(b, reg_lambda=regularization, split_penalty=penalty),
                    legality=lambda o, b: o.vector_feasible(b, min_child_h=minimum),
                    leaf=lambda o, h: o.vector_leaf(h, reg_lambda=regularization),
                )
            # Snapshot before another user callback can release or replace its workspace.
            terms.append(DeviceTerm(trees.copy(ops, tree), output_map))
        return tuple(terms)

    return mapped(ops, train, validation, objective=objective, learner=learn, **configuration)
