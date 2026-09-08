"""Event/right-censored fixed-scale log-normal geometry and inference."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np

from .artifacts import Model
from .data import Problem, _identity, _owned
from .outputs import positive_mean

# Tail quadrature differs structurally from the continued-fraction test oracle.
_NODES, _WEIGHTS = np.polynomial.laguerre.laggauss(32)


def _scale(sigma):
    if not np.isscalar(sigma) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("positive finite AFT scale required")
    square = float(sigma) * float(sigma)
    if not math.isfinite(square) or square == 0 or not math.isfinite(1 / square):
        raise ValueError("AFT scale geometry exceeds float64")
    return float(sigma)


def normal_tail(z):
    """Log survival, inverse Mills ratio and curvature using tail quadrature."""
    z = float(z)
    log_phi = -0.5 * z * z - 0.5 * math.log(2 * math.pi)
    if not math.isfinite(z) or not math.isfinite(log_phi):
        raise ValueError("normal tail argument exceeds float64")
    if z > 8:
        kernel = np.exp(-0.5 * (_NODES / z) ** 2)
        integral = float(np.dot(_WEIGHTS, kernel))
        moment = float(np.dot(_WEIGHTS, _NODES * kernel))
        return log_phi + math.log(integral) - math.log(z), z / integral, moment / integral**2
    logsf = (
        math.log(math.erfc(z / math.sqrt(2)) / 2)
        if z >= 0
        else math.log1p(-math.erfc(-z / math.sqrt(2)) / 2)
    )
    mills = math.exp(log_phi - logsf)
    return logsf, mills, mills * (mills - z)


@dataclass(frozen=True)
class LogNormalAFT:
    sigma: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "sigma", _scale(self.sigma))

    @staticmethod
    def validate(problem):
        if (
            not isinstance(problem, Problem)
            or problem.target_kind != "event_right"
            or problem.raw_width != 1
            or problem.structure
            or problem.classes is not None
        ):
            raise ValueError("AFT requires event_right bounds, one raw output and no structure")

    def base(self, problem):
        self.validate(problem)
        # A finite starting location, not a censoring-adjusted intercept optimum.
        with np.errstate(over="raise", invalid="raise"):
            value = np.dot(
                problem.weight / problem.weight.sum(),
                np.log(problem.target[:, 0]) - problem.offset[:, 0],
            )
        return _owned([value], ndim=1)

    def geometry(self, problem, raw):
        self.validate(problem)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            log_time = np.log(problem.target[:, 0])
            z = (log_time - problem.with_offset(raw)[:, 0]) / self.sigma
            event = problem.target[:, 0] == problem.target[:, 1]
            loss = log_time + math.log(self.sigma) + z**2 / 2 + math.log(2 * math.pi) / 2
            g = -z / self.sigma
            h = np.full(len(z), 1 / self.sigma**2)
            for i in np.flatnonzero(~event):
                logsf, mills, curvature = normal_tail(z[i])
                loss[i], g[i], h[i] = -logsf, -mills / self.sigma, curvature / self.sigma**2
            aggregate = float(np.dot(problem.weight / problem.weight.sum(), loss))
        if not np.isfinite(aggregate):
            raise ValueError("nonfinite AFT likelihood")
        return aggregate, _owned(g, ndim=1), _owned(h, ndim=1)

    def loss(self, problem, raw):
        return self.geometry(problem, raw)[0]


@dataclass(frozen=True, eq=False)
class AFTModel:
    model: Model
    sigma: float = 1.0

    def __post_init__(self):
        if (
            not isinstance(self.model, Model)
            or len(self.model.base) != 1
            or self.model.classes is not None
        ):
            raise ValueError("scalar AFT raw model required")
        object.__setattr__(self, "sigma", _scale(self.sigma))

    def predict(self, data, *, times, probabilities=(0.5,), offset=None):
        times = _owned(times, ndim=1)
        probabilities = _owned(probabilities, ndim=1)
        if np.any(times <= 0) or np.any(probabilities <= 0) or np.any(probabilities >= 1):
            raise ValueError("positive times and probabilities in (0,1) required")
        raw = self.model.predict(data, offset=offset)
        with np.errstate(over="raise", invalid="raise"):
            z = (np.log(times)[None, :] - raw) / self.sigma
            survival = np.array([math.exp(normal_tail(v)[0]) for v in z.flat]).reshape(z.shape)
            quantile_z = np.array([NormalDist().inv_cdf(float(p)) for p in probabilities])
            qraw = raw + self.sigma * quantile_z[None, :]
            quantiles = positive_mean(qraw.reshape(-1, 1)).reshape(qraw.shape)
            mean = positive_mean(raw + self.sigma**2 / 2)
        return dict(
            median=positive_mean(raw),
            mean=mean,
            survival=_owned(survival, ndim=2),
            quantile=_owned(quantiles, ndim=2),
        )

    def record(self):
        return dict(
            format="openboost-aft-lognormal-v1", sigma=self.sigma, model=self.model.record()
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
                    raise ValueError("duplicate AFT artifact field")
                result[key] = value
            return result

        record = json.loads(Path(path).read_text(), object_pairs_hook=pairs)
        if (
            not isinstance(record, dict)
            or set(record) != {"format", "sigma", "model"}
            or record["format"] != "openboost-aft-lognormal-v1"
        ):
            raise ValueError("unsupported AFT artifact")
        return cls(Model.from_record(record["model"]), record["sigma"])
