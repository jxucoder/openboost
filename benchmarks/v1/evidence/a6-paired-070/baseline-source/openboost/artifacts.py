"""Immutable numeric ensembles with explicit term coefficients and output maps."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import ClassSchema, MixedData, NumericData, _identity, _owned
from .tree import Tree


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
class TreeTerm:
    learner: Tree
    mapping: np.ndarray
    coefficient: float = 1.0

    def __post_init__(self):
        if not isinstance(self.learner, Tree):
            raise ValueError("tree learner required")
        mapping = _owned(self.mapping, ndim=2)
        if mapping.shape[0] != self.learner.output_width:
            raise ValueError("learner requires an [L, K] output mapping")
        coefficient = ConstantTerm([0], self.coefficient).coefficient
        object.__setattr__(self, "mapping", mapping)
        object.__setattr__(self, "coefficient", coefficient)


@dataclass(frozen=True, eq=False)
class Model:
    """Numeric raw ensemble; observation offsets are supplied at inference time.

    This replaces the B03 constant-only format. Explicit [L, K] maps let vector
    learners update multiple raw columns; constants update the entire base width.
    """

    feature_names: tuple[str, ...]
    base: np.ndarray
    terms: tuple[ConstantTerm | TreeTerm, ...] = ()
    classes: ClassSchema | None = None

    def __post_init__(self):
        names = tuple(self.feature_names)
        if (
            not names
            or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("unique nonempty feature names required")
        base = _owned(self.base, ndim=1)
        if self.classes is not None and (
            not isinstance(self.classes, ClassSchema)
            or not (
                (len(self.classes.values) == 2 and len(base) == 1)
                or len(base) == len(self.classes.values)
            )
        ):
            raise ValueError("classifier raw width must be one binary logit or one logit per class")
        terms = tuple(self.terms)
        bound = np.abs(base).copy()
        with np.errstate(over="raise", invalid="raise"):
            for term in terms:
                if isinstance(term, ConstantTerm):
                    if term.value.shape != base.shape:
                        raise ValueError("constant term output width differs from base")
                    bound += abs(term.coefficient) * np.abs(term.value)
                elif isinstance(term, TreeTerm):
                    if term.learner.binning.feature_names != names or term.mapping.shape != (
                        term.learner.output_width,
                        len(base),
                    ):
                        raise ValueError("tree schema or output mapping differs from model")
                    # Conservative finite envelope over every leaf and every input.
                    magnitude = np.max(
                        np.abs(term.learner.value[term.learner.feature == -1]), axis=0
                    )
                    bound += abs(term.coefficient) * (magnitude @ np.abs(term.mapping))
                else:
                    raise ValueError("unsupported model term")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "terms", terms)

    def predict(self, data, *, offset=None):
        if (
            not isinstance(data, (NumericData, MixedData))
            or data.feature_names != self.feature_names
        ):
            raise ValueError("inference feature schema differs from model")
        raw = np.broadcast_to(self.base, (len(data.values), len(self.base))).copy()
        with np.errstate(over="raise", invalid="raise"):
            for term in self.terms:
                delta = (
                    term.value
                    if isinstance(term, ConstantTerm)
                    else term.learner.predict(data) @ term.mapping
                )
                raw += term.coefficient * delta
            if offset is not None:
                a = np.asarray(offset, dtype=float)
                if a.shape != raw.shape or not np.isfinite(a).all():
                    raise ValueError("finite aligned inference offset required")
                raw += a
        return raw

    def predict_proba(self, data, *, offset=None):
        if self.classes is None:
            raise ValueError("probabilities require a classification schema")
        from .outputs import binary_probabilities, softmax_probabilities

        raw = self.predict(data, offset=offset)
        return binary_probabilities(raw) if len(self.base) == 1 else softmax_probabilities(raw)

    def predict_label(self, data, *, offset=None):
        probabilities = self.predict_proba(data, offset=offset)
        return self.classes.decode(np.argmax(probabilities, axis=1))

    def record(self):
        terms = []
        for term in self.terms:
            if isinstance(term, ConstantTerm):
                terms.append(
                    dict(kind="constant", value=term.value.tolist(), coefficient=term.coefficient)
                )
            else:
                terms.append(
                    dict(
                        kind="tree",
                        learner=term.learner.record(),
                        mapping=term.mapping.tolist(),
                        coefficient=term.coefficient,
                    )
                )
        return dict(
            format="openboost-ensemble-v2",
            feature_names=list(self.feature_names),
            base=self.base.tolist(),
            terms=terms,
            classes=None if self.classes is None else list(self.classes.values),
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
        return cls.from_record(record)

    @classmethod
    def from_record(cls, record):
        """Validate a nested raw model without temporary files."""
        if (
            not isinstance(record, dict)
            or set(record) != {"format", "feature_names", "base", "terms", "classes"}
            or record["format"] != "openboost-ensemble-v2"
            or any(
                not isinstance(record[name], list) for name in ("feature_names", "base", "terms")
            )
        ):
            raise ValueError("unsupported or corrupt artifact schema")
        if record["classes"] is not None and not isinstance(record["classes"], list):
            raise ValueError("invalid class schema record")
        classes = None if record["classes"] is None else ClassSchema(record["classes"])
        terms = []
        for term in record["terms"]:
            if not isinstance(term, dict):
                raise ValueError("invalid artifact term")
            if term.get("kind") == "constant" and set(term) == {"kind", "value", "coefficient"}:
                terms.append(ConstantTerm(term["value"], term["coefficient"]))
            elif term.get("kind") == "tree" and set(term) == {
                "kind",
                "learner",
                "mapping",
                "coefficient",
            }:
                terms.append(
                    TreeTerm(
                        Tree.from_record(term["learner"]),
                        term["mapping"],
                        term["coefficient"],
                    )
                )
            else:
                raise ValueError("invalid artifact term schema")
        return cls(record["feature_names"], record["base"], tuple(terms), classes)
