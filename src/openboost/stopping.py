"""Public immutable validation stopping, independent of model transactions."""

from dataclasses import dataclass, replace
from math import isfinite
from numbers import Real
from typing import Protocol, runtime_checkable


@runtime_checkable
class StoppingStatus(Protocol):
    """Public completion metadata, independent of an author's stopping policy.

    None denotes an active policy; completed results require a nonempty reason.
    Read-only properties describe the contract, not runtime mutation isolation.
    """

    @property
    def rounds(self) -> int: ...

    @property
    def completed_rounds(self) -> int: ...

    @property
    def reason(self) -> str | None: ...


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError("finite real validation score and min_delta required")
    return float(value)


@dataclass(frozen=True)
class StopState:
    rounds: int
    patience: int | None
    min_delta: float
    reference_score: float
    last_score: float
    completed_rounds: int = 0
    stale_rounds: int = 0

    def __post_init__(self):
        if type(self.rounds) is not int or self.rounds < 0:
            raise ValueError("nonnegative integer round budget required")
        if self.patience is not None and (type(self.patience) is not int or self.patience <= 0):
            raise ValueError("positive integer patience or None required")
        for name in ("min_delta", "reference_score", "last_score"):
            object.__setattr__(self, name, _finite(getattr(self, name)))
        if self.min_delta < 0 or (self.patience is None and self.min_delta != 0):
            raise ValueError("nonnegative min_delta requires enabled patience")
        if (
            type(self.completed_rounds) is not int
            or not 0 <= self.completed_rounds <= self.rounds
            or type(self.stale_rounds) is not int
            or not 0 <= self.stale_rounds <= self.completed_rounds
            or (self.patience is not None and self.stale_rounds > self.patience)
        ):
            raise ValueError("valid completed and stale round counts required")

    @classmethod
    def start(cls, initial_score, *, rounds, patience=None, min_delta=0.0):
        """Initial validation consumes no round; None disables patience stopping."""
        return cls(rounds, patience, min_delta, initial_score, initial_score)

    @property
    def reason(self):
        """None while running; patience takes precedence over a coincident budget."""
        if self.patience is not None and self.stale_rounds >= self.patience:
            return "patience"
        if self.completed_rounds >= self.rounds:
            return "budget"
        return None

    def observe(self, score):
        """Observe once per completed outer round, including full rejection.

        Smaller is better. Improvement must strictly exceed min_delta relative
        to the last qualifying improvement. Trials/substeps are not observations.
        This state never chooses or mutates an accepted/best model.
        """
        if self.reason is not None:
            raise ValueError("cannot observe a finished stop state")
        score = _finite(score)
        improved = self.reference_score - score > self.min_delta
        return replace(
            self,
            reference_score=score if improved else self.reference_score,
            last_score=score,
            completed_rounds=self.completed_rounds + 1,
            stale_rounds=0 if improved else self.stale_rounds + 1,
        )
