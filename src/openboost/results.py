"""Structural recipe results shared by built-in and external algorithms."""

from typing import Protocol, runtime_checkable

from .runtime import AcceptedState
from .stopping import StopState


@runtime_checkable
class RecipeResult(Protocol):
    """Completed recipe result; per-round diagnostic payloads belong to authors.

    No inheritance or conversion is required. Implementations must preserve the
    immutable state/input contract; this protocol is not process isolation.
    """

    @property
    def state(self) -> AcceptedState: ...

    @property
    def steps(self) -> tuple[object, ...]: ...

    @property
    def stop(self) -> StopState: ...


def validate_result(result, *, context, train, validation):
    """Reject missing/malformed/incomplete/foreign results without inspecting payloads."""
    if not isinstance(result, RecipeResult):
        raise ValueError("recipe result requires state, steps and stop")
    state, steps, stop = result.state, result.steps, result.stop
    if not isinstance(state, AcceptedState) or not isinstance(stop, StopState):
        raise ValueError("recipe result requires AcceptedState and StopState")
    if not isinstance(steps, tuple) or len(steps) != stop.completed_rounds:
        raise ValueError("recipe steps must be a tuple with one entry per completed outer round")
    if stop.reason is None:
        raise ValueError("recipe returned unfinished stopping state")
    if (
        state.context != context
        or state.train.identity != train.identity
        or state.validation.identity != validation.identity
    ):
        raise ValueError("recipe returned foreign run state")
    return result
