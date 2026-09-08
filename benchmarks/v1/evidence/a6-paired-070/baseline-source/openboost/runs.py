"""Explicit sequential execution of heterogeneous CPU recipes; no batching claim."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .binning import PreparedData
from .data import Problem
from .results import RecipeResult, validate_result
from .runtime import RunContext


@dataclass(frozen=True, eq=False)
class RunSpec:
    context: RunContext
    train: Problem
    validation: Problem
    recipe: Callable
    options: Mapping = field(default_factory=dict)
    prepared: PreparedData | None = None

    def __post_init__(self):
        if (
            not isinstance(self.context, RunContext)
            or not isinstance(self.train, Problem)
            or not isinstance(self.validation, Problem)
            or not callable(self.recipe)
            or not isinstance(self.options, Mapping)
            or (self.prepared is not None and not isinstance(self.prepared, PreparedData))
        ):
            raise ValueError("explicit context, problems, recipe and options required")
        options = dict(self.options)
        if any(
            not isinstance(k, str) or k in {"context", "train", "validation", "prepared"}
            for k in options
        ):
            raise ValueError("recipe options cannot replace run identity or problems")
        if any(type(v) not in (str, int, float, bool, type(None)) for v in options.values()):
            raise ValueError("run options currently require immutable scalar values")
        object.__setattr__(self, "options", MappingProxyType(options))


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    result: RecipeResult | None
    error_type: str | None = None
    error_message: str | None = None


def run_many(specs, *, execution="sequential"):
    """Return every run's result/error in requested order, with unique logical IDs.

    Problems may share immutable NumericData. Independent same-ID execution is the
    equivalence baseline. This is not process isolation: callbacks must honor the
    immutable input contract. KeyboardInterrupt/SystemExit propagate.
    """
    specs = tuple(specs)
    if execution != "sequential" or any(not isinstance(s, RunSpec) for s in specs):
        raise ValueError("sequential RunSpec execution required")
    ids = [s.context.run_id for s in specs]
    if len(set(ids)) != len(ids):
        raise ValueError("run IDs must be unique before execution")
    outcomes = []
    for spec in specs:
        try:
            preparation = {} if spec.prepared is None else {"prepared": spec.prepared}
            result = spec.recipe(
                spec.train, spec.validation, context=spec.context, **spec.options, **preparation
            )
            validate_result(
                result, context=spec.context, train=spec.train, validation=spec.validation
            )
            outcomes.append(RunOutcome(spec.context.run_id, result))
        except Exception as error:
            outcomes.append(RunOutcome(spec.context.run_id, None, type(error).__name__, str(error)))
    return tuple(outcomes)
