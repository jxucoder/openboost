"""Owned numeric CPU inputs and explicit problem roles (initial B03 contract)."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np


def _owned(value, *, ndim, finite=True):
    a = np.array(value, dtype=np.float64, copy=True, order="C")
    if a.ndim != ndim or not a.size or (finite and not np.isfinite(a).all()):
        raise ValueError("nonempty finite array with declared dimensions required")
    # A bytes owner prevents callers from re-enabling writeability on the array.
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _identity(*parts):
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            payload = json.dumps([part.dtype.str, part.shape]).encode() + part.tobytes()
        else:
            payload = json.dumps(part, sort_keys=True, allow_nan=False).encode()
        h.update(len(payload).to_bytes(8, "big"))
        h.update(payload)
    return h.hexdigest()


@dataclass(frozen=True, eq=False)
class NumericData:
    """Unbinned numeric features; missing values allowed, infinity rejected.

    Owns immutable storage. Row IDs are unique integers; feature names define order.
    This does not yet fit a transformer or provide categorical/CUDA data.
    """

    values: np.ndarray
    row_ids: np.ndarray
    feature_names: tuple[str, ...]
    device: str = "cpu"
    identity: str = field(init=False)

    def __post_init__(self):
        if self.device != "cpu":
            raise ValueError("NumericData currently supports CPU only")
        x = np.array(self.values, dtype=np.float64, copy=True)
        if x.ndim != 2 or not x.size or np.isinf(x).any():
            raise ValueError("nonempty numeric matrix without infinity required")
        x[np.isnan(x)] = np.nan
        x = _owned(x, ndim=2, finite=False)
        ids = np.asarray(self.row_ids)
        if ids.shape != (len(x),) or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
            raise ValueError("aligned unique integer row IDs required")
        if np.any(ids > np.iinfo(np.int64).max):
            raise ValueError("row ID exceeds int64")
        ids = np.frombuffer(ids.astype("<i8").tobytes(), dtype="<i8")
        names = tuple(self.feature_names)
        if (
            len(names) != x.shape[1]
            or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("unique nonempty feature names must match columns")
        object.__setattr__(self, "values", x)
        object.__setattr__(self, "row_ids", ids)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "identity", _identity("numeric-cpu-v1", x, ids, names))


@dataclass(frozen=True, eq=False)
class Problem:
    """Numeric targets, original row weights and raw offsets in the given row order.

    Targets are [N, T]; offsets are [N, raw_width]. Weight remains [N] and is not applied
    here. The caller supplies one explicit row order for all aligned role arrays.
    Objective-specific target support and additional roles arrive in later slices.
    """

    data: NumericData
    target: np.ndarray
    row_ids: np.ndarray
    weight: np.ndarray | None = None
    offset: np.ndarray | None = None
    raw_width: int | None = None
    structure: Mapping | None = None
    identity: str = field(init=False)

    def __post_init__(self):
        ids = np.asarray(self.row_ids)
        if ids.dtype.kind not in "iu":
            raise ValueError("integer role row IDs required")
        if not isinstance(self.data, NumericData) or not np.array_equal(
            self.row_ids, self.data.row_ids
        ):
            raise ValueError("problem roles must match prepared row order")
        y = _owned(self.target, ndim=2)
        if len(y) != len(self.data.values):
            raise ValueError("target rows differ from data")
        w = _owned(np.ones(len(y)) if self.weight is None else self.weight, ndim=1)
        if w.shape != (len(y),) or np.any(w < 0) or not np.isfinite(w.sum()) or w.sum() <= 0:
            raise ValueError("nonnegative aligned weights with positive finite mass required")
        width = y.shape[1] if self.raw_width is None else self.raw_width
        if type(width) is not int or width < 1:
            raise ValueError("positive integer raw_width required")
        offset = _owned(np.zeros((len(y), width)) if self.offset is None else self.offset, ndim=2)
        if offset.shape != (len(y), width):
            raise ValueError("offset must match raw parameter shape")
        object.__setattr__(self, "raw_width", width)
        roles = {} if self.structure is None else self.structure
        if not isinstance(roles, Mapping) or any(not isinstance(k, str) or not k for k in roles):
            raise ValueError("named structural roles required")
        roles = {k: _owned(v, ndim=2) for k, v in roles.items()}
        if any(len(v) != len(y) for v in roles.values()):
            raise ValueError("structure must match problem row order")
        object.__setattr__(self, "structure", MappingProxyType(roles))
        object.__setattr__(self, "target", y)
        object.__setattr__(self, "weight", w)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "row_ids", self.data.row_ids)
        object.__setattr__(
            self,
            "identity",
            _identity(
                "problem-cpu-v1",
                self.data.identity,
                y,
                w,
                offset,
                tuple((k, _identity(roles[k])) for k in sorted(roles)),
            ),
        )

    def with_offset(self, raw):
        """Return prediction-space raw values; never mutate the cached raw input."""
        raw = np.asarray(raw, dtype=float)
        if raw.shape != self.offset.shape or not np.isfinite(raw).all():
            raise ValueError("raw values must match the problem output shape")
        with np.errstate(over="raise", invalid="raise"):
            return raw + self.offset
