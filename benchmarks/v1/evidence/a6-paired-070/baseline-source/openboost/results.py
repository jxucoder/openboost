"""Structural recipe results shared by built-in and external algorithms."""

from typing import Protocol, runtime_checkable

from .runtime import AcceptedState
from .stopping import StoppingStatus


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
    def stop(self) -> StoppingStatus: ...


def validate_result(result, *, context, train, validation):
    """Reject missing/malformed/incomplete/foreign results without inspecting payloads."""
    if not isinstance(result, RecipeResult):
        raise ValueError("recipe result requires state, steps and stop")
    state, steps, stop = result.state, result.steps, result.stop
    if not isinstance(state, AcceptedState):
        raise ValueError("recipe result requires AcceptedState")
    if not isinstance(stop, StoppingStatus):
        raise ValueError("recipe stop requires rounds, completed_rounds and reason")
    rounds, completed, reason = stop.rounds, stop.completed_rounds, stop.reason
    if type(rounds) is not int or rounds < 0:
        raise ValueError("recipe stop requires a nonnegative integer round budget")
    if type(completed) is not int or not 0 <= completed <= rounds:
        raise ValueError("recipe stop requires an integer completed count within its budget")
    if not isinstance(reason, str) or not reason:
        raise ValueError("recipe stop requires a nonempty terminal reason")
    if reason == "budget" and completed != rounds:
        raise ValueError("budget termination requires all rounds completed")
    if not isinstance(steps, tuple) or len(steps) != completed:
        raise ValueError("recipe steps must be a tuple with one entry per completed outer round")
    if (
        state.context != context
        or state.train.identity != train.identity
        or state.validation.identity != validation.identity
    ):
        raise ValueError("recipe returned foreign run state")
    return result
