"""Explicit immutable scalar diagnostics; no automatic conversion of author payloads."""

from dataclasses import dataclass

import numpy as np

from .comparison import LossChange


def validate_retention(retention):
    if not isinstance(retention, str) or retention not in ("full", "summary"):
        raise ValueError("retention must be 'full' or 'summary'")
    return retention


def _scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) in (str, int, float, bool, type(None)):
        return value
    if type(value) is tuple:
        return tuple(_scalar(v) for v in value)
    if type(value) is LossChange:
        return LossChange(
            value.lower, value.upper, str(value.method), str(value.reason), value.unchanged
        )
    raise ValueError("summary values must be scalars or tuples of scalars")


@dataclass(frozen=True)
class TraceSummary:
    """Named scalar evidence for an author-declared step kind.

    Arrays, models and states are rejected; authors explicitly choose scalar
    observations. LossChange is an explicitly supported immutable scalar record;
    arbitrary author records are not traversed. This record does not alter the
    structural RecipeResult contract.
    """

    kind: str
    values: tuple[tuple[str, object], ...]
    omitted: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("nonempty summary kind required")
        values = tuple((name, _scalar(value)) for name, value in self.values)
        names = tuple(name for name, _ in values)
        omitted = tuple(self.omitted)
        if any(not isinstance(n, str) or not n for n in (*names, *omitted)) or len(
            set((*names, *omitted))
        ) != len(names) + len(omitted):
            raise ValueError("unique nonempty retained and omitted field names required")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "omitted", omitted)
