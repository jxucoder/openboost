"""Experimental resident scalar recipe assembled from public device operations."""

from dataclasses import dataclass

from . import device_tree as trees
from .device import _parameter, _workspace
from .device_runtime import DeviceRun, DeviceState
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
class DeviceFitResult:
    """Owned run/state plus scalar history. Release the run after exporting artifacts."""

    run: DeviceRun
    state: DeviceState
    steps: tuple[DeviceStep, ...]
    stop: StopState


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
):
    """Scalar squared rounds, with explicit fixed/backtracking acceptance and stopping.

    learner(ops, data, fields) may compose public device operations. Its returned
    tree is snapshotted; new callback scratch is temporary. Each proposal snapshots
    the learner again to separate proposal lifetime from caller workspace. History
    stores scalars, never past [N, 1] raw arrays. Retained final/best trees share
    immutable owned terms. No CPU fallback or extension of CPU run_many is implied.
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
    run = DeviceRun(ops, train, validation, run_id=run_id, seed=seed, binning=binning, bins=bins)
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
