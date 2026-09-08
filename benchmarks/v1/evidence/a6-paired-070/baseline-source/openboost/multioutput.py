"""Train-only target scaling and original-unit multi-output inference."""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .artifacts import Model
from .data import _identity, _owned
from .objectives import MultiSquared


@dataclass(frozen=True, eq=False)
class TargetScale:
    mean: np.ndarray
    scale: np.ndarray
    constant: tuple[bool, ...]

    def __post_init__(self):
        mean, scale = _owned(self.mean, ndim=1), _owned(self.scale, ndim=1)
        constant = tuple(self.constant)
        if (
            mean.shape != scale.shape
            or np.any(scale <= 0)
            or len(constant) != len(mean)
            or any(type(v) is not bool for v in constant)
            or any(flag and value != 1 for flag, value in zip(constant, scale, strict=True))
        ):
            raise ValueError("aligned target means, positive scales and constant flags required")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "constant", constant)

    @classmethod
    def fit(cls, train):
        MultiSquared.validate(train)
        with np.errstate(over="raise", invalid="raise"):
            weight = (train.weight / train.weight.sum())[:, None]
            supported = train.target[train.weight > 0]
            flags = np.all(supported == supported[0], axis=0)
            mean = np.where(flags, supported[0], np.sum(weight * train.target, axis=0))
            variance = np.sum(weight * (train.target - mean) ** 2, axis=0)
            constant = tuple(bool(v) for v in flags)
            scale = np.where(flags, 1.0, np.sqrt(variance))
        return cls(mean, scale, constant)

    def transform(self, problem):
        MultiSquared.validate(problem)
        if problem.raw_width != len(self.mean):
            raise ValueError("target width differs from fitted scaling")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return replace(
                problem,
                target=(problem.target - self.mean) / self.scale,
                offset=problem.offset / self.scale,
            )


@dataclass(frozen=True, eq=False)
class MultiOutputModel:
    model: Model
    target_scale: TargetScale

    def __post_init__(self):
        if (
            not isinstance(self.model, Model)
            or self.model.classes is not None
            or not isinstance(self.target_scale, TargetScale)
            or len(self.model.base) != len(self.target_scale.mean)
        ):
            raise ValueError("raw regression model must match target scaling")

    def predict(self, data, *, offset=None):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if offset is not None:
                offset = np.asarray(offset, dtype=float)
                if offset.shape != (len(data.row_ids), len(self.model.base)):
                    raise ValueError("aligned original-unit offsets required")
                offset = offset / self.target_scale.scale
            result = self.model.predict(data, offset=offset)
            return _owned(result * self.target_scale.scale + self.target_scale.mean, ndim=2)

    def record(self):
        return dict(
            format="openboost-multioutput-v1",
            model=self.model.record(),
            mean=self.target_scale.mean.tolist(),
            scale=self.target_scale.scale.tolist(),
            constant=list(self.target_scale.constant),
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
                    raise ValueError("duplicate multi-output artifact field")
                result[key] = value
            return result

        record = json.loads(Path(path).read_text(), object_pairs_hook=pairs)
        if (
            not isinstance(record, dict)
            or set(record) != {"format", "model", "mean", "scale", "constant"}
            or record["format"] != "openboost-multioutput-v1"
        ):
            raise ValueError("unsupported multi-output artifact")
        return cls(
            Model.from_record(record["model"]),
            TargetScale(record["mean"], record["scale"], record["constant"]),
        )
