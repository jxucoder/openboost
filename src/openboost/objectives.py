"""Objective geometry over raw state, with explicit weight and offset semantics."""

import numpy as np

from .data import Problem, _owned
from .stats import newton


class Squared:
    """Scalar weighted mean half-square loss; derivatives are weighted once."""

    @staticmethod
    def validate(problem):
        if (
            not isinstance(problem, Problem)
            or problem.classes is not None
            or problem.target.shape[1] != 1
            or problem.raw_width != 1
            or bool(problem.structure)
        ):
            raise ValueError("squared recipe requires scalar [N, 1] targets")

    @classmethod
    def base(cls, problem):
        cls.validate(problem)
        with np.errstate(over="raise", invalid="raise"):
            return _owned(
                np.sum(
                    (problem.target - problem.offset)
                    * (problem.weight / problem.weight.sum())[:, None],
                    axis=0,
                ),
                ndim=1,
            )

    @classmethod
    def gradient(cls, problem, raw):
        cls.validate(problem)
        with np.errstate(over="raise", invalid="raise"):
            return _owned((problem.with_offset(raw) - problem.target)[:, 0], ndim=1)

    @classmethod
    def fields(cls, problem, raw):
        return newton(problem, cls.gradient(problem, raw), np.ones(len(problem.target)))

    @classmethod
    def loss(cls, problem, raw):
        gradient = cls.gradient(problem, raw)
        with np.errstate(over="raise", invalid="raise"):
            result = float(np.dot(problem.weight / problem.weight.sum(), gradient**2 / 2))
        if not np.isfinite(result):
            raise ValueError("nonfinite squared loss")
        return result


class Normal:
    """Scalar observed targets, raw (mean, log-scale), diagonal Fisher geometry.

    Minimum scale regularizes initialization only. Runtime scale/precision must
    remain strictly positive and finite; invalid trials are never clipped.
    """

    @staticmethod
    def validate(problem):
        if (
            not isinstance(problem, Problem)
            or problem.classes is not None
            or problem.target.shape[1] != 1
            or problem.raw_width != 2
            or bool(problem.structure)
        ):
            raise ValueError("Normal requires scalar targets and raw_width=2")

    @staticmethod
    def parameters(raw):
        values = _owned(raw, ndim=2)
        if values.shape[1] != 2:
            raise ValueError("Normal raw values require mean/log-scale columns")
        with np.errstate(over="raise", invalid="raise"):
            scale = np.exp(values[:, 1])
        if np.any(scale <= 0) or not np.isfinite(scale).all():
            raise ValueError("Normal scale must be positive and finite")
        return _owned(np.column_stack((values[:, 0], scale)), ndim=2)

    @classmethod
    def base(cls, problem, *, minimum_scale=1e-6):
        cls.validate(problem)
        if not np.isscalar(minimum_scale) or not np.isfinite(minimum_scale) or minimum_scale <= 0:
            raise ValueError("positive finite minimum_scale required")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            relative_precision = np.exp(-2 * problem.offset[:, 1])
            if np.any(relative_precision <= 0):
                raise ValueError("Normal offset precision underflow")
            mass = problem.weight / problem.weight.sum()
            mean_mass = mass * relative_precision
            centered = problem.target[:, 0] - problem.offset[:, 0]
            mean = np.dot(mean_mass / mean_mass.sum(), centered)
            variance = np.dot(mean_mass, (centered - mean) ** 2)
            base = _owned([mean, np.log(max(np.sqrt(variance), minimum_scale))], ndim=1)
        # Reject an unrepresentable initial predictive distribution.
        cls.loss(problem, np.broadcast_to(base, problem.offset.shape))
        return base

    @classmethod
    def geometry(cls, problem, raw):
        """Return loss, unweighted gradient and Fisher diagonal [N, 2]."""
        cls.validate(problem)
        values = problem.with_offset(raw)
        cls.parameters(values)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            precision = np.exp(-2 * values[:, 1])
            if np.any(precision <= 0):
                raise ValueError("Normal precision must be positive")
            residual = values[:, 0] - problem.target[:, 0]
            square = residual**2 * precision
            losses = values[:, 1] + square / 2 + 0.5 * np.log(2 * np.pi)
            gradient = _owned(np.column_stack((residual * precision, 1 - square)), ndim=2)
            fisher = _owned(np.column_stack((precision, np.full(len(values), 2.0))), ndim=2)
            loss = float(np.dot(problem.weight / problem.weight.sum(), losses))
        if not np.isfinite(loss):
            raise ValueError("nonfinite Normal loss")
        return loss, gradient, fisher

    @classmethod
    def loss(cls, problem, raw):
        return cls.geometry(problem, raw)[0]

    @classmethod
    def compare(cls, problem, before, after):
        """Bound the mean NLL change at stored raw snapshots in problem row order.

        Absolute loss and geometry remain separate reporting/learning operations.
        Unsupported comparison ranges return unresolved; invalid Normal rows raise.
        """
        from ._normal_comparison import compare

        cls.validate(problem)
        return compare(problem, before, after)


def diagonal_direction(gradient, metric, *, mode="natural", damping=0.0):
    """Unweighted ordinary/Fisher-diagonal direction, before regression weights."""
    g, h = _owned(gradient, ndim=2), _owned(metric, ndim=2)
    if g.shape != h.shape or np.any(h <= 0):
        raise ValueError("aligned positive metric diagonal required")
    if not np.isscalar(damping) or not np.isfinite(damping) or damping < 0:
        raise ValueError("finite nonnegative damping required")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if mode == "ordinary":
            if damping != 0:
                raise ValueError("ordinary direction does not use damping")
            return _owned(-g, ndim=2)
        if mode != "natural":
            raise ValueError("ordinary/natural direction required")
        return _owned(-g / (h + damping), ndim=2)


class Formula:
    """Saturation a*(1-exp(-b*x)); softplus parameters and explicit structure x."""

    @staticmethod
    def validate(problem):
        if (
            not isinstance(problem, Problem)
            or problem.classes is not None
            or problem.target.shape[1] != 1
            or problem.raw_width != 2
            or set(problem.structure) != {"x"}
            or problem.structure["x"].shape != problem.target.shape
            or np.any(problem.structure["x"] <= 0)
        ):
            raise ValueError(
                "Formula requires scalar target, raw_width=2 and positive structure x [N,1]"
            )

    @staticmethod
    def predict(raw, structure):
        values, x = _owned(raw, ndim=2), _owned(structure, ndim=2)
        if values.shape != (len(x), 2) or x.shape[1] != 1 or np.any(x <= 0):
            raise ValueError("Formula requires aligned raw [N,2] and positive structure [N,1]")
        with np.errstate(over="raise", invalid="raise"):
            a, b = np.logaddexp(0, values).T
            if np.any(a <= 0) or np.any(b <= 0):
                raise ValueError("Formula parameters underflowed")
            return _owned((a * -np.expm1(-b * x[:, 0]))[:, None], ndim=2)

    @classmethod
    def base(cls, problem):
        cls.validate(problem)
        # A deterministic initializer, not an optimum of the coupled objective.
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            a = max(
                float(np.dot(problem.weight / problem.weight.sum(), problem.target[:, 0])), 1e-6
            )
            positive = np.array([a, 1.0])
            base = _owned(positive + np.log(-np.expm1(-positive)), ndim=1)
        cls.loss(problem, np.broadcast_to(base, problem.offset.shape))
        return base

    @classmethod
    def geometry(cls, problem, raw):
        """Weighted loss, unweighted gradient and rank-one per-row GGN."""
        cls.validate(problem)
        values = problem.with_offset(raw)
        x = problem.structure["x"][:, 0]
        prediction = cls.predict(values, problem.structure["x"])[:, 0]
        with np.errstate(over="raise", invalid="raise"):
            a, b = np.logaddexp(0, values).T
            sigmoid = np.exp(-np.logaddexp(0, -values))
            jacobian = np.column_stack(
                (sigmoid[:, 0] * -np.expm1(-b * x), sigmoid[:, 1] * a * x * np.exp(-b * x))
            )
            residual = prediction - problem.target[:, 0]
            gradient = _owned(residual[:, None] * jacobian, ndim=2)
            metric = _owned(np.einsum("ni,nj->nij", jacobian, jacobian), ndim=3)
            loss = float(np.dot(problem.weight / problem.weight.sum(), residual**2 / 2))
        if not np.isfinite(loss):
            raise ValueError("nonfinite Formula loss")
        return loss, gradient, metric

    @classmethod
    def loss(cls, problem, raw):
        return cls.geometry(problem, raw)[0]


def full_direction(gradient, metric, *, damping):
    """Damped SPD solve; no silent diagonal approximation or pseudoinverse.

    A relative eigenvalue check rejects numerically singular matrices even if a
    floating point Cholesky implementation happens to accept them.
    """
    g, h = _owned(gradient, ndim=2), _owned(metric, ndim=3)
    if h.shape != (len(g), g.shape[1], g.shape[1]) or not np.array_equal(h, h.swapaxes(1, 2)):
        raise ValueError("aligned symmetric full metric required")
    if not np.isscalar(damping) or not np.isfinite(damping) or damping < 0:
        raise ValueError("nonnegative finite damping required")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        system = h + damping * np.eye(g.shape[1])
        eigenvalues = np.linalg.eigvalsh(system)
        if np.any(eigenvalues[:, 0] <= np.finfo(float).eps * eigenvalues[:, -1]):
            raise ValueError("metric is not numerically positive definite; specify damping")
        factor = np.linalg.cholesky(system)
        rhs = np.linalg.solve(factor, -g[..., None])
        return _owned(np.linalg.solve(factor.swapaxes(1, 2), rhs)[..., 0], ndim=2)


class Binary:
    """Signed-margin logistic loss with unweighted stable first/second derivatives."""

    @staticmethod
    def validate(problem):
        if (
            not isinstance(problem, Problem)
            or problem.raw_width != 1
            or problem.classes is None
            or len(problem.classes.values) != 2
            or problem.structure
        ):
            raise ValueError(
                "binary requires two-class encoded targets, raw_width=1 and no structure"
            )

    @classmethod
    def base(cls, problem, *, clip=1e-6):
        cls.validate(problem)
        if not np.isscalar(clip) or not np.isfinite(clip) or not 0 < clip < 0.5 or 1 - clip == 1:
            raise ValueError("representable probability clip in (0, 0.5) required")
        if len(np.unique(problem.target[:, 0])) != 2:
            raise ValueError("binary training requires both classes")
        with np.errstate(over="raise", invalid="raise"):
            mass = problem.weight / problem.weight.sum()
            p = np.clip(np.dot(mass, problem.target[:, 0]), clip, 1 - clip)
            # Offset-centred prior; a deterministic initializer, not an offset MLE.
            return _owned([np.log(p) - np.log1p(-p) - np.dot(mass, problem.offset[:, 0])], ndim=1)

    @classmethod
    def geometry(cls, problem, raw):
        from .outputs import binary_probabilities

        cls.validate(problem)
        values = problem.with_offset(raw)
        r, y = values[:, 0], problem.target[:, 0]
        probabilities = binary_probabilities(values)
        tail = np.exp(-np.abs(r))
        gradient = _owned(np.where(y == 1, -probabilities[:, 0], probabilities[:, 1]), ndim=1)
        curvature = _owned(tail / (1 + tail) ** 2, ndim=1)
        with np.errstate(over="raise", invalid="raise"):
            losses = np.logaddexp(0, np.where(y == 1, -r, r))
            loss = float(np.dot(problem.weight / problem.weight.sum(), losses))
        if not np.isfinite(loss):
            raise ValueError("nonfinite binary loss")
        return loss, gradient, curvature

    @classmethod
    def loss(cls, problem, raw):
        return cls.geometry(problem, raw)[0]


class Multiclass:
    """Softmax likelihood with 2*p*(1-p) diagonal upper bound, not exact Hessian."""

    @staticmethod
    def validate(problem):
        if (
            not isinstance(problem, Problem)
            or problem.classes is None
            or problem.raw_width != len(problem.classes.values)
            or problem.structure
        ):
            raise ValueError(
                "multiclass requires class schema, one raw column per class and no structure"
            )

    @classmethod
    def base(cls, problem):
        cls.validate(problem)
        if len(np.unique(problem.target[:, 0])) != problem.raw_width:
            raise ValueError("multiclass training requires every declared class")
        return _owned(np.zeros(problem.raw_width), ndim=1)

    @classmethod
    def geometry(cls, problem, raw):
        from .outputs import softmax_probabilities

        cls.validate(problem)
        values = problem.with_offset(raw)
        probability = softmax_probabilities(values)
        codes = problem.target[:, 0].astype(int)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            shifted = values - values.max(axis=1, keepdims=True)
            losses = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(values)), codes]
            gradient = probability.copy()
            gradient[np.arange(len(values)), codes] -= 1
            bound = 2 * probability * (1 - probability)
            loss = float(np.dot(problem.weight / problem.weight.sum(), losses))
        if not np.isfinite(loss):
            raise ValueError("nonfinite multiclass loss")
        return loss, _owned(gradient, ndim=2), _owned(bound, ndim=2)

    @classmethod
    def loss(cls, problem, raw):
        return cls.geometry(problem, raw)[0]


class Quantile:
    """Pinball loss and unit pseudo-curvature for split construction."""

    def __init__(self, q=0.5):
        if not np.isscalar(q) or not np.isfinite(q) or not 0 < q < 1:
            raise ValueError("q must lie strictly between zero and one")
        self.q = float(q)

    validate = staticmethod(Squared.validate)

    def residuals(self, problem, raw):
        self.validate(problem)
        with np.errstate(over="raise", invalid="raise"):
            return _owned((problem.target - problem.with_offset(raw))[:, 0], ndim=1)

    def base(self, problem):
        from .leaves import ResidualContext, quantile_leaf

        residual = self.residuals(problem, np.zeros_like(problem.target))
        view = ResidualContext(problem, residual).view(np.arange(len(residual)))
        return [quantile_leaf(view, q=self.q)]

    def loss(self, problem, raw):
        residual = self.residuals(problem, raw)
        with np.errstate(over="raise", invalid="raise"):
            value = np.maximum(self.q * residual, (self.q - 1) * residual)
            return float(np.dot(problem.weight / problem.weight.sum(), value))

    def fields(self, problem, raw):
        residual = self.residuals(problem, raw)
        return newton(problem, (residual < 0).astype(float) - self.q, np.ones(len(residual)))


class Poisson:
    """Count likelihood with raw log rate, explicit exposure and additive offset."""

    def __init__(self, minimum_rate=1e-6):
        if not np.isscalar(minimum_rate) or not np.isfinite(minimum_rate) or minimum_rate <= 0:
            raise ValueError("positive finite minimum_rate required")
        self.minimum_rate = float(minimum_rate)

    @staticmethod
    def validate(problem):
        if (
            not isinstance(problem, Problem)
            or problem.classes is not None
            or problem.target.shape[1] != 1
            or problem.raw_width != 1
            or set(problem.structure) != {"exposure"}
        ):
            raise ValueError("Poisson requires scalar counts and an explicit exposure role")
        y, e = problem.target[:, 0], problem.structure["exposure"]
        if e.shape != problem.target.shape or np.any(e <= 0):
            raise ValueError("aligned positive exposure required")
        if np.any(y < 0) or np.any(y != np.floor(y)):
            raise ValueError("nonnegative integer counts required")

    def base(self, problem):
        self.validate(problem)
        positive = problem.weight > 0
        y = problem.target[:, 0]
        counted = positive & (y > 0)
        if not np.any(counted):
            return [float(np.log(self.minimum_rate))]

        def log_sum(values):
            maximum = np.max(values)
            return maximum + np.log(np.exp(values - maximum).sum())

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            numerator = log_sum(np.log(problem.weight[counted]) + np.log(y[counted]))
            denominator = log_sum(
                np.log(problem.weight[positive])
                + np.log(problem.structure["exposure"][positive, 0])
                + problem.offset[positive, 0]
            )
            value = numerator - denominator
        if not np.isfinite(value):
            raise ValueError("nonfinite Poisson intercept")
        return [float(value)]

    def geometry(self, problem, raw):
        import math

        self.validate(problem)
        with np.errstate(over="raise", invalid="raise"):
            log_mean = problem.with_offset(raw)[:, 0] + np.log(problem.structure["exposure"][:, 0])
            mean = np.exp(log_mean)
            if np.any(mean <= 0):
                raise ValueError("Poisson mean underflows float64")
            y = problem.target[:, 0]
            losses = mean - y * log_mean + np.array([math.lgamma(v + 1) for v in y])
            gradient = mean - y
            loss = float(np.dot(problem.weight / problem.weight.sum(), losses))
        if not np.isfinite(loss):
            raise ValueError("nonfinite Poisson likelihood")
        return loss, _owned(gradient, ndim=1), _owned(mean, ndim=1)

    def loss(self, problem, raw):
        return self.geometry(problem, raw)[0]


class Gamma:
    """Positive mean regression: loss y/exp(f)+f, with fixed unit dispersion."""

    @staticmethod
    def validate(problem):
        Squared.validate(problem)
        if np.any(problem.target <= 0):
            raise ValueError("Gamma targets must be strictly positive")

    @classmethod
    def base(cls, problem):
        cls.validate(problem)
        positive = problem.weight > 0
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            weights = problem.weight[positive]
            weights = weights / weights.max()
            terms = (
                np.log(weights) + np.log(problem.target[positive, 0]) - problem.offset[positive, 0]
            )
            maximum = np.max(terms)
            value = maximum + np.log(np.exp(terms - maximum).sum()) - np.log(weights.sum())
        return _owned([value], ndim=1)

    @classmethod
    def geometry(cls, problem, raw):
        cls.validate(problem)
        with np.errstate(over="raise", invalid="raise"):
            values = problem.with_offset(raw)[:, 0]
            ratio = np.exp(np.log(problem.target[:, 0]) - values)
            if np.any(ratio <= 0):
                raise ValueError("Gamma target/mean ratio underflows float64")
            loss = float(np.dot(problem.weight / problem.weight.sum(), ratio + values))
        if not np.isfinite(loss):
            raise ValueError("nonfinite Gamma objective")
        return loss, _owned(1 - ratio, ndim=1), _owned(ratio, ndim=1)

    @classmethod
    def loss(cls, problem, raw):
        return cls.geometry(problem, raw)[0]


class Tweedie:
    """Nonnegative mean objective with fixed variance power strictly between 1 and 2."""

    def __init__(self, power=1.5, minimum_mean=1e-6):
        if (
            not np.isscalar(power)
            or not np.isfinite(power)
            or not 1 < power < 2
            or not np.isscalar(minimum_mean)
            or not np.isfinite(minimum_mean)
            or minimum_mean <= 0
        ):
            raise ValueError("power in (1,2) and positive finite minimum_mean required")
        self.power, self.minimum_mean = float(power), float(minimum_mean)

    @staticmethod
    def validate(problem):
        Squared.validate(problem)
        if np.any(problem.target < 0):
            raise ValueError("Tweedie targets must be nonnegative")

    def base(self, problem):
        self.validate(problem)
        positive = problem.weight > 0
        counted = positive & (problem.target[:, 0] > 0)
        if not np.any(counted):
            return _owned([np.log(self.minimum_mean)], ndim=1)

        def log_sum(values):
            maximum = values.max()
            return maximum + np.log(np.exp(values - maximum).sum())

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            numerator = log_sum(
                np.log(problem.weight[counted])
                + np.log(problem.target[counted, 0])
                + (1 - self.power) * problem.offset[counted, 0]
            )
            denominator = log_sum(
                np.log(problem.weight[positive]) + (2 - self.power) * problem.offset[positive, 0]
            )
            value = numerator - denominator
        return _owned([value], ndim=1)

    def geometry(self, problem, raw):
        self.validate(problem)
        with np.errstate(over="raise", invalid="raise"):
            values = problem.with_offset(raw)[:, 0]
            y = problem.target[:, 0]
            a = np.exp((2 - self.power) * values)
            b = np.zeros_like(y)
            positive = y > 0
            b[positive] = np.exp(np.log(y[positive]) + (1 - self.power) * values[positive])
            if np.any(a <= 0) or np.any(b[positive] <= 0):
                raise ValueError("Tweedie positive terms underflow float64")
            losses = b / (self.power - 1) + a / (2 - self.power)
            loss = float(np.dot(problem.weight / problem.weight.sum(), losses))
            gradient = a - b
            curvature = (2 - self.power) * a + (self.power - 1) * b
        if not np.isfinite(loss):
            raise ValueError("nonfinite Tweedie objective")
        return loss, _owned(gradient, ndim=1), _owned(curvature, ndim=1)

    def loss(self, problem, raw):
        return self.geometry(problem, raw)[0]


class MultiSquared:
    """Sum of output half-squared errors, averaged with original row weights."""

    @staticmethod
    def validate(problem):
        if (
            not isinstance(problem, Problem)
            or problem.target_kind != "numeric"
            or problem.classes is not None
            or problem.structure
            or problem.target.shape[1] < 1
            or problem.raw_width != problem.target.shape[1]
        ):
            raise ValueError("numeric multi-output targets and matching raw width required")

    @classmethod
    def base(cls, problem):
        cls.validate(problem)
        with np.errstate(over="raise", invalid="raise"):
            return _owned(
                np.sum(
                    (problem.target - problem.offset)
                    * (problem.weight / problem.weight.sum())[:, None],
                    axis=0,
                ),
                ndim=1,
            )

    @classmethod
    def gradient(cls, problem, raw):
        cls.validate(problem)
        with np.errstate(over="raise", invalid="raise"):
            return _owned(problem.with_offset(raw) - problem.target, ndim=2)

    @classmethod
    def mse(cls, problem, raw):
        error = cls.gradient(problem, raw)
        with np.errstate(over="raise", invalid="raise"):
            return _owned(
                np.sum((problem.weight / problem.weight.sum())[:, None] * error**2, axis=0), ndim=1
            )

    @classmethod
    def loss(cls, problem, raw):
        with np.errstate(over="raise", invalid="raise"):
            value = float(cls.mse(problem, raw).sum() / 2)
        if not np.isfinite(value):
            raise ValueError("nonfinite multi-output squared loss")
        return value
