"""Sequential device recipe execution with detached CPU outcomes; no batching."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .artifacts import Model
from .binning import Binning
from .data import Problem
from .device import DeviceOperations, _workspace
from .device_inputs import DeviceFeatures
from .device_recipes import (
    DeviceFitResult,
    DeviceNormalStep,
    DeviceScalarStep,
    DeviceStep,
)
from .device_runtime import DeviceRun, DeviceState
from .objectives import Squared
from .runtime import RunContext
from .stopping import StopState


@dataclass(frozen=True, eq=False)
class RunSpec:
    """One scalar squared-compatible recipe with explicit cuts or resident features.

    Recipes receive (ops, train, validation, run_id=..., seed=..., **options)
    and must return an owned DeviceFitResult. Options are immutable scalars;
    custom learners/policies belong in an explicit recipe callable. Supplying
    binning reuses cuts. prepared borrows a training/validation DeviceFeatures
    pair; the recipe must accept it and preserve that pair in its returned run.
    Targets, offsets, weights and all mutable run storage stay independent.
    """

    run_id: str
    seed: int
    train: Problem
    validation: Problem
    recipe: Callable
    options: Mapping = field(default_factory=dict)
    binning: Binning | None = None
    prepared: tuple[DeviceFeatures, DeviceFeatures] | None = None

    def __post_init__(self):
        RunContext(self.run_id, self.seed)  # Shared identity/key validation only.
        if (
            not isinstance(self.train, Problem)
            or not isinstance(self.validation, Problem)
            or not callable(self.recipe)
            or not isinstance(self.options, Mapping)
            or (self.binning is not None and not isinstance(self.binning, Binning))
            or (self.prepared is not None and (
                not isinstance(self.prepared, tuple) or len(self.prepared) != 2
                or any(not isinstance(f, DeviceFeatures) for f in self.prepared)
            ))
        ):
            raise ValueError("explicit problems, recipe, options and fitted binning required")
        options = dict(self.options)
        reserved = {"ops", "train", "validation", "run_id", "seed", "binning", "prepared"}
        if any(not isinstance(k, str) or k in reserved for k in options):
            raise ValueError("options cannot replace run identity, problems or binning")
        if any(type(v) not in (str, int, float, bool, type(None)) for v in options.values()):
            raise ValueError("run options require immutable scalar values")
        object.__setattr__(self, "options", MappingProxyType(options))


@dataclass(frozen=True)
class RunResult:
    """Detached final/best inference and diagnostics; no live device run or raw arrays.

    state is diagnostic metadata, not a record that can be used with another run.
    steps retain native recipe observations, including ordered channel substeps.
    """

    model: Model
    best_model: Model
    state: DeviceState
    steps: tuple
    stop: StopState


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    result: RunResult | None
    error_type: str | None = None
    error_message: str | None = None


def _completed(state, steps, stop, run_id):
    if not isinstance(state, DeviceState) or state.run_id != run_id:
        raise ValueError("recipe returned foreign state")
    if not isinstance(stop, StopState) or stop.reason is None:
        raise ValueError("recipe requires terminal StopState")
    if not isinstance(steps, tuple) or any(
        not isinstance(s, (DeviceStep, DeviceNormalStep, DeviceScalarStep)) for s in steps
    ):
        raise ValueError("recipe requires native immutable device steps")
    indices = tuple(s.round_index for s in steps)
    if (
        any(type(i) is not int or not 0 <= i < stop.completed_rounds for i in indices)
        or tuple(sorted(indices)) != indices
        or set(indices) != set(range(stop.completed_rounds))
    ):
        raise ValueError("recipe steps must cover completed outer rounds in order")


def _one(ops, spec):
    # The initial scheduling contract exports raw scalar Models. Specialized
    # target schemas need their own export contract before they enter this path.
    Squared.validate(spec.train)
    Squared.validate(spec.validation)
    # The enclosing workspace owns every allocation made by this invocation,
    # including scratch left behind by an exception or malformed result.
    previous = set(ops._records)
    with _workspace(ops):
        preparation = {} if spec.binning is None else {"binning": spec.binning}
        if spec.prepared is not None:
            preparation["prepared"] = spec.prepared
        fit = spec.recipe(
            ops, spec.train, spec.validation, run_id=spec.run_id, seed=spec.seed,
            **spec.options, **preparation,
        )
        if not isinstance(fit, DeviceFitResult) or not isinstance(fit.run, DeviceRun):
            raise ValueError("recipe requires an owned DeviceFitResult")
        run = fit.run
        if (
            run.ops is not ops
            or run.run_id != spec.run_id
            or run.seed != spec.seed
            or run.data.problem_identity != spec.train.identity
            or run.validation_data.problem_identity != spec.validation.identity
            or run.problem in previous
            or (spec.binning is not None and run.binning.identity != spec.binning.identity)
            or (spec.prepared is not None and (
                run.prepared_features != spec.prepared
                or any(data.codes is not features.codes or data.missing is not features.missing
                       for data, features in zip((run.data, run.validation_data), spec.prepared, strict=True))
            ))
        ):
            raise ValueError("recipe returned foreign or caller-owned run")
        _completed(fit.state, fit.steps, fit.stop, spec.run_id)
        if "rounds" in spec.options and fit.stop.rounds != spec.options["rounds"]:
            raise ValueError("recipe changed the requested round budget")
        # export validates actual state ownership; a copied diagnostic is not a
        # valid accepted state. Both exports precede release of any run storage.
        result = RunResult(
            run.export(fit.state), run.export(fit.state, best=True),
            fit.state, fit.steps, fit.stop,
        )
        run.close()
        return result


def run_many(ops, specs, *, execution="sequential"):
    """Execute in requested order and retain each result/error under its stable ID.

    Each fit releases its new allocations before the next fit; completed results
    can predict after the caller closes ops.execution. Existing caller resources
    remain borrowed. The caller's context cap applies to all live work; this is
    cooperative ownership, not process isolation against arbitrary callbacks.

    Only scalar squared-compatible target schemas are accepted in this first
    reference; specialized objective/inference schemas are rejected explicitly.
    Supplied fitted cuts avoid refitting. An explicit prepared feature pair also
    reuses encoded codes/missingness; weights and objective preparation remain
    per fit. Keep feature records alive until all borrowers finish. No batching,
    concurrent execution, CPU fallback or performance improvement is implied.
    KeyboardInterrupt/SystemExit propagate; ordinary errors retain outcomes.
    """
    specs = tuple(specs)
    if execution != "sequential" or any(not isinstance(s, RunSpec) for s in specs):
        raise ValueError("sequential device RunSpec execution required")
    if len({s.run_id for s in specs}) != len(specs):
        raise ValueError("run IDs must be unique before execution")
    if not isinstance(ops, DeviceOperations):
        raise ValueError("DeviceOperations required; CPU fallback is unavailable")
    ops.execution._check()
    outcomes = []
    for spec in specs:
        try:
            result = _one(ops, spec)
            outcomes.append(RunOutcome(spec.run_id, result))
        except Exception as error:
            outcomes.append(RunOutcome(spec.run_id, None, type(error).__name__, str(error)))
    return tuple(outcomes)
