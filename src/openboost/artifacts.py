"""Versioned constant-term inference for B03 state checks; no tree trainer yet."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import NumericData, _identity, _owned


@dataclass(frozen=True, eq=False)
class ConstantTerm:
    value: np.ndarray
    coefficient: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "value", _owned(self.value, ndim=1))
        if (
            isinstance(self.coefficient, bool)
            or not np.isscalar(self.coefficient)
            or not np.isfinite(self.coefficient)
        ):
            raise ValueError("finite scalar coefficient required")
        object.__setattr__(self, "coefficient", float(self.coefficient))


@dataclass(frozen=True, eq=False)
class ConstantModel:
    """Immutable numeric raw predictor, with offset supplied at prediction time.

    Coefficients apply once. This minimal artifact deliberately supports constant
    terms only; later tree/mapping artifacts must declare their own state schema.
    """

    feature_names: tuple[str, ...]
    base: np.ndarray
    terms: tuple[ConstantTerm, ...] = ()

    def __post_init__(self):
        names = tuple(self.feature_names)
        if (
            not names
            or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("unique nonempty feature names required")
        base = _owned(self.base, ndim=1)
        terms = tuple(self.terms)
        if any(not isinstance(t, ConstantTerm) or t.value.shape != base.shape for t in terms):
            raise ValueError("constant term output width differs from base")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "terms", terms)
        self._value()  # Reject overflow before accepting or serializing a model.

    def _value(self):
        result = self.base.copy()
        with np.errstate(over="raise", invalid="raise"):
            for term in self.terms:
                result += term.coefficient * term.value
        return result

    def predict(self, data, *, offset=None):
        if not isinstance(data, NumericData) or data.feature_names != self.feature_names:
            raise ValueError("inference feature schema differs from model")
        raw = np.broadcast_to(self._value(), (len(data.values), len(self.base))).copy()
        if offset is not None:
            a = np.asarray(offset, dtype=float)
            if a.shape != raw.shape or not np.isfinite(a).all():
                raise ValueError("finite aligned inference offset required")
            with np.errstate(over="raise", invalid="raise"):
                raw += a
        return raw

    def record(self):
        return dict(
            format="openboost-constant-v1",
            feature_names=list(self.feature_names),
            base=self.base.tolist(),
            terms=[dict(value=t.value.tolist(), coefficient=t.coefficient) for t in self.terms],
        )

    @property
    def identity(self):
        return _identity(self.record())

    def save(self, path):
        Path(path).write_text(json.dumps(self.record(), allow_nan=False) + "\n")

    @classmethod
    def load(cls, path):
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate artifact field")
                result[key] = value
            return result

        record = json.loads(Path(path).read_text(), object_pairs_hook=pairs)
        if (
            not isinstance(record, dict)
            or set(record) != {"format", "feature_names", "base", "terms"}
            or record["format"] != "openboost-constant-v1"
        ):
            raise ValueError("unsupported or corrupt artifact schema")
        if not isinstance(record["feature_names"], list) or not isinstance(record["base"], list):
            raise ValueError("invalid artifact vector fields")
        if not isinstance(record["terms"], list) or any(
            not isinstance(t, dict) or set(t) != {"value", "coefficient"} for t in record["terms"]
        ):
            raise ValueError("invalid artifact terms")
        return cls(
            record["feature_names"],
            record["base"],
            tuple(ConstantTerm(**t) for t in record["terms"]),
        )
