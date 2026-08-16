"""Shared-tree histogram distribution boosting with a CRPS objective."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._array import BinnedArray, array
from .._core._growth import TreeStructure
from .._core._vector_tree import fit_vector_tree
from .._persistence import PersistenceMixin
from .._validation import validate_sample_weight, validate_X, validate_y


def _softmax(logits: NDArray) -> NDArray:
    shifted = np.asarray(logits, dtype=np.float64)
    if not np.all(np.isfinite(shifted)):
        raise FloatingPointError("histogram logits contain non-finite values")
    shifted = shifted - np.max(shifted, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _continuous_crps_terms(
    y: NDArray,
    bin_edges: NDArray,
) -> tuple[NDArray, NDArray]:
    """Return normalized terms for exact piecewise-uniform histogram CRPS.

    CRPS has the energy representation
    ``E|X-y| - 0.5 E|X-X'|``.  Each histogram bin represents a uniform
    conditional density, not a point mass at its midpoint.  The returned first
    term has shape ``(n_samples, n_bins)`` and the pairwise-distance matrix has
    shape ``(n_bins, n_bins)``.  Both are divided by mean bin width so tree
    regularization remains invariant to target units.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    bin_edges = np.asarray(bin_edges, dtype=np.float64).reshape(-1)
    if bin_edges.size < 3:
        raise ValueError("bin_edges must describe at least two bins")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(bin_edges)):
        raise ValueError("y and bin_edges must contain only finite values")
    widths = np.diff(bin_edges)
    if np.any(widths <= 0.0):
        raise ValueError("bin_edges must be strictly increasing")
    midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    y_column = y[:, None]
    lower = bin_edges[:-1][None, :]
    upper = bin_edges[1:][None, :]
    distance_to_target = np.where(
        y_column < lower,
        midpoints[None, :] - y_column,
        np.where(
            y_column > upper,
            y_column - midpoints[None, :],
            ((y_column - lower) ** 2 + (upper - y_column) ** 2) / (2.0 * widths[None, :]),
        ),
    )

    pairwise_distance = np.abs(midpoints[:, None] - midpoints[None, :])
    np.fill_diagonal(pairwise_distance, widths / 3.0)
    normalization = float(np.mean(widths))
    return distance_to_target / normalization, pairwise_distance / normalization


def _continuous_crps_loss(
    logits: NDArray,
    y: NDArray,
    bin_edges: NDArray,
) -> NDArray:
    """Exact per-row CRPS for the represented piecewise-uniform histogram."""
    probabilities = _softmax(logits)
    distance_to_target, pairwise_distance = _continuous_crps_terms(y, bin_edges)
    if distance_to_target.shape != probabilities.shape:
        raise ValueError("y and bin_edges must match the logit rows and outputs")
    first = np.sum(probabilities * distance_to_target, axis=1)
    second = 0.5 * np.einsum(
        "ni,ij,nj->n",
        probabilities,
        pairwise_distance,
        probabilities,
    )
    return first - second


def _crps_grad_gn(
    logits: NDArray,
    y: NDArray,
    bin_edges: NDArray,
    *,
    curvature_scale: float = 1.0,
    sample_weight: NDArray | None = None,
    curvature_floor: float = 1e-6,
) -> tuple[NDArray, NDArray]:
    """Gradient and PSD diagonal curvature for exact continuous CRPS.

    If ``a_j = E|U_j-y|`` and ``D_jk = E|U_j-U_k|`` for uniform histogram
    bins, CRPS is ``a.T @ p - 0.5 * p.T @ D @ p``.  The gradient is exact.
    Curvature is the diagonal of ``J.T @ (-D) @ J`` for the softmax Jacobian
    ``J``; ``-D`` is positive semidefinite on the probability-simplex tangent
    space.  The indefinite residual term from differentiating ``J`` is
    deliberately excluded.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError("logits must have shape (n_samples, n_distribution_bins)")
    n_samples, n_outputs = logits.shape
    if curvature_scale <= 0.0:
        raise ValueError("curvature_scale must be strictly positive")

    probabilities = _softmax(logits)
    distance_to_target, pairwise_distance = _continuous_crps_terms(y, bin_edges)
    if distance_to_target.shape != (n_samples, n_outputs):
        raise ValueError("y and bin_edges must match the logit rows and outputs")

    probability_distance = probabilities @ pairwise_distance
    grad_probability = distance_to_target - probability_distance
    centered = grad_probability - np.sum(probabilities * grad_probability, axis=1, keepdims=True)
    gradient = probabilities * centered

    probability_quadratic = np.sum(
        probabilities * probability_distance,
        axis=1,
        keepdims=True,
    )
    distance_diagonal = np.diag(pairwise_distance)[None, :]
    curvature = (
        curvature_scale
        * probabilities
        * probabilities
        * (2.0 * probability_distance - distance_diagonal - probability_quadratic)
    )
    curvature = np.maximum(curvature, curvature_floor)

    if sample_weight is not None:
        weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1, 1)
        if weight.shape[0] != n_samples:
            raise ValueError("sample_weight must have one entry per logit row")
        gradient *= weight
        curvature *= weight

    return gradient.astype(np.float32), curvature.astype(np.float32)


@dataclass
class HistogramDistributionOutput:
    """Piecewise-uniform predictive distributions on one shared bin grid."""

    probas: NDArray
    bin_edges: NDArray

    def __post_init__(self) -> None:
        self.probas = np.asarray(self.probas, dtype=np.float64)
        self.bin_edges = np.asarray(self.bin_edges, dtype=np.float64).reshape(-1)
        if self.probas.ndim != 2:
            raise ValueError("probas must have shape (n_samples, n_bins)")
        if self.bin_edges.shape != (self.probas.shape[1] + 1,):
            raise ValueError("bin_edges must have n_bins + 1 entries")
        if not np.all(np.isfinite(self.bin_edges)):
            raise ValueError("bin_edges must contain only finite values")
        if np.any(np.diff(self.bin_edges) <= 0.0):
            raise ValueError("bin_edges must be strictly increasing")
        if not np.all(np.isfinite(self.probas)) or np.any(self.probas < 0.0):
            raise ValueError("probas must be finite and non-negative")
        totals = np.sum(self.probas, axis=1, keepdims=True)
        if np.any(totals <= 0.0):
            raise ValueError("each probability row must have positive mass")
        self.probas = self.probas / totals

    @property
    def bin_midpoints(self) -> NDArray:
        return 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

    def mean(self) -> NDArray:
        return self.probas @ self.bin_midpoints

    def variance(self) -> NDArray:
        mean = self.mean()
        widths = np.diff(self.bin_edges)
        second_centered = (self.bin_midpoints[None, :] - mean[:, None]) ** 2 + widths[
            None, :
        ] ** 2 / 12.0
        return np.sum(self.probas * second_centered, axis=1)

    def std(self) -> NDArray:
        return np.sqrt(self.variance())

    def quantile(self, q: float) -> NDArray:
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must lie in [0, 1]")
        if q == 0.0:
            return np.full(self.probas.shape[0], self.bin_edges[0])
        if q == 1.0:
            return np.full(self.probas.shape[0], self.bin_edges[-1])
        cdf = np.cumsum(self.probas, axis=1)
        indices = np.argmax(cdf >= q, axis=1)
        rows = np.arange(self.probas.shape[0])
        previous = np.where(indices == 0, 0.0, cdf[rows, np.maximum(indices - 1, 0)])
        mass = self.probas[rows, indices]
        fraction = np.divide(
            q - previous,
            mass,
            out=np.zeros_like(mass),
            where=mass > 0.0,
        )
        widths = np.diff(self.bin_edges)
        return self.bin_edges[indices] + np.clip(fraction, 0.0, 1.0) * widths[indices]

    def interval(self, alpha: float = 0.1) -> tuple[NDArray, NDArray]:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        return self.quantile(alpha / 2.0), self.quantile(1.0 - alpha / 2.0)

    def sample(self, n_samples: int = 1, seed: int | None = None) -> NDArray:
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        rng = np.random.default_rng(seed)
        uniforms = rng.random((self.probas.shape[0], n_samples))
        cdf = np.cumsum(self.probas, axis=1)
        samples = np.empty_like(uniforms)
        widths = np.diff(self.bin_edges)
        for row in range(self.probas.shape[0]):
            indices = np.searchsorted(cdf[row], uniforms[row], side="left")
            previous = np.where(indices == 0, 0.0, cdf[row, np.maximum(indices - 1, 0)])
            mass = self.probas[row, indices]
            fraction = np.divide(
                uniforms[row] - previous,
                mass,
                out=np.zeros(n_samples, dtype=np.float64),
                where=mass > 0.0,
            )
            samples[row] = self.bin_edges[indices] + np.clip(fraction, 0.0, 1.0) * widths[indices]
        return samples


@dataclass
class HistogramBoost(PersistenceMixin):
    """CPU shared-tree boosting for a flexible histogram distribution.

    Each boosting round fits one tree structure with a vector of logit updates
    in every leaf.  The objective is the exact continuous CRPS of the
    represented piecewise-uniform histogram and therefore produces a monotone
    CDF by construction without independent-quantile crossing.

    This first implementation supports numeric CPU input (including NaNs).
    Categorical splits, CUDA, callbacks, and evaluation sets intentionally raise
    or remain outside this API until their vector paths are implemented.
    """

    n_distribution_bins: int = 50
    n_trees: int = 100
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: float = 1e-3
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    min_gain: float = 0.0
    n_feature_bins: int = 254
    curvature_scale: float = 1.0
    base_smoothing: float = 1.0

    trees_: list[TreeStructure] = field(default_factory=list, init=False, repr=False)
    X_binned_: BinnedArray | None = field(default=None, init=False, repr=False)
    base_logits_: NDArray | None = field(default=None, init=False, repr=False)
    target_bin_edges_: NDArray | None = field(default=None, init=False, repr=False)
    target_bin_midpoints_: NDArray | None = field(default=None, init=False, repr=False)
    n_features_in_: int | None = field(default=None, init=False)

    _PARAM_NAMES = (
        "n_distribution_bins",
        "n_trees",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "reg_lambda",
        "reg_alpha",
        "min_gain",
        "n_feature_bins",
        "curvature_scale",
        "base_smoothing",
    )

    def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002
        """Return constructor parameters for sklearn cloning."""
        return {name: getattr(self, name) for name in self._PARAM_NAMES}

    def set_params(self, **params: Any) -> HistogramBoost:
        unknown = sorted(set(params) - set(self._PARAM_NAMES))
        if unknown:
            raise ValueError(f"Unknown parameter(s): {unknown}")
        for name, value in params.items():
            setattr(self, name, value)
        return self

    def _validate_params(self) -> None:
        if self.n_distribution_bins < 2:
            raise ValueError("n_distribution_bins must be at least 2")
        if self.n_trees < 0:
            raise ValueError("n_trees must be non-negative")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be strictly positive")
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if self.min_child_weight < 0.0:
            raise ValueError("min_child_weight must be non-negative")
        if self.reg_lambda <= 0.0:
            raise ValueError("reg_lambda must be strictly positive")
        if self.reg_alpha < 0.0 or self.min_gain < 0.0:
            raise ValueError("reg_alpha and min_gain must be non-negative")
        if not 2 <= self.n_feature_bins <= 254:
            raise ValueError("n_feature_bins must lie in [2, 254]")
        if self.curvature_scale <= 0.0:
            raise ValueError("curvature_scale must be strictly positive")
        if self.base_smoothing < 0.0:
            raise ValueError("base_smoothing must be non-negative")

    def _make_target_grid(self, y: NDArray) -> tuple[NDArray, NDArray]:
        lower = float(np.min(y))
        upper = float(np.max(y))
        if lower == upper:
            half_span = max(abs(lower) * 1e-6, 1e-6)
            edges = np.linspace(
                lower - half_span,
                upper + half_span,
                self.n_distribution_bins + 1,
            )
        else:
            spacing = (upper - lower) / (self.n_distribution_bins - 1)
            edges = np.linspace(
                lower - 0.5 * spacing,
                upper + 0.5 * spacing,
                self.n_distribution_bins + 1,
            )
        return edges, 0.5 * (edges[:-1] + edges[1:])

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any | None = None,
    ) -> HistogramBoost:
        """Fit on numeric CPU data; target grid state is learned from train y only."""
        self._validate_params()
        if isinstance(X, BinnedArray):
            raise TypeError(
                "HistogramBoost.fit expects raw numeric X so it can own and persist "
                "the training bin transform"
            )
        X_valid = validate_X(X, allow_binned=False, allow_nan=True, context="fit")
        y_valid = validate_y(y, n_samples=X_valid.shape[0], task="regression")
        weights = validate_sample_weight(sample_weight, X_valid.shape[0])
        if weights is not None and float(np.sum(weights)) <= 0.0:
            raise ValueError("sample_weight must contain positive total weight")

        self.X_binned_ = array(X_valid, n_bins=self.n_feature_bins, device="cpu")
        if self.X_binned_.any_categorical:
            raise NotImplementedError("HistogramBoost currently supports numeric features only")
        self.n_features_in_ = X_valid.shape[1]
        self.target_bin_edges_, self.target_bin_midpoints_ = self._make_target_grid(y_valid)
        labels = np.searchsorted(self.target_bin_edges_[1:-1], y_valid, side="right").astype(
            np.int64
        )
        labels = np.clip(labels, 0, self.n_distribution_bins - 1)

        count_weights = (
            np.ones_like(y_valid, dtype=np.float64)
            if weights is None
            else weights.astype(np.float64)
        )
        counts = np.bincount(
            labels,
            weights=count_weights,
            minlength=self.n_distribution_bins,
        ).astype(np.float64)
        # ``base_smoothing`` is a total Dirichlet concentration, distributed
        # evenly so changing the number of bins does not change prior strength.
        counts += self.base_smoothing / self.n_distribution_bins
        probabilities = np.maximum(counts, np.finfo(np.float64).tiny)
        probabilities /= np.sum(probabilities)
        self.base_logits_ = np.log(probabilities)
        self.base_logits_ -= np.mean(self.base_logits_)

        self.trees_ = []
        logits = np.broadcast_to(
            self.base_logits_, (X_valid.shape[0], self.n_distribution_bins)
        ).copy()
        for _ in range(self.n_trees):
            grad, hess = _crps_grad_gn(
                logits,
                y_valid,
                self.target_bin_edges_,
                curvature_scale=self.curvature_scale,
                sample_weight=weights,
            )
            tree = fit_vector_tree(
                self.X_binned_,
                grad,
                hess,
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                min_gain=self.min_gain,
            )
            self.trees_.append(tree)
            update = self.learning_rate * np.asarray(tree(self.X_binned_))
            if not np.all(np.isfinite(update)):
                raise FloatingPointError("HistogramBoost produced a non-finite tree update")
            logits += update
            if not np.all(np.isfinite(logits)):
                raise FloatingPointError("HistogramBoost produced non-finite training logits")
        return self

    def _predict_logits(self, X: Any) -> NDArray:
        if self.X_binned_ is None or self.base_logits_ is None or self.n_features_in_ is None:
            raise ValueError("HistogramBoost is not fitted. Call fit before predict.")
        if isinstance(X, BinnedArray):
            raise TypeError("HistogramBoost.predict expects raw X")
        X_valid = validate_X(X, allow_binned=False, allow_nan=True, context="predict")
        if X_valid.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_valid.shape[1]} features, expected {self.n_features_in_}")
        X_binned = self.X_binned_.transform(X_valid)
        logits = np.broadcast_to(
            self.base_logits_, (X_valid.shape[0], self.n_distribution_bins)
        ).copy()
        for tree in self.trees_:
            update = self.learning_rate * np.asarray(tree(X_binned))
            if not np.all(np.isfinite(update)):
                raise FloatingPointError("HistogramBoost produced a non-finite tree update")
            logits += update
        if not np.all(np.isfinite(logits)):
            raise FloatingPointError("HistogramBoost produced non-finite prediction logits")
        return logits

    def predict_distribution(self, X: Any) -> HistogramDistributionOutput:
        if self.target_bin_edges_ is None:
            raise ValueError("HistogramBoost is not fitted. Call fit before predict.")
        return HistogramDistributionOutput(
            probas=_softmax(self._predict_logits(X)),
            bin_edges=self.target_bin_edges_,
        )

    def predict(self, X: Any) -> NDArray:
        return self.predict_distribution(X).mean()

    def score(self, X: Any, y: Any) -> float:
        y_true = np.asarray(y, dtype=np.float64).reshape(-1)
        prediction = self.predict(X)
        residual = np.sum((y_true - prediction) ** 2)
        total = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1.0 - residual / total) if total > 0.0 else 0.0

    def __sklearn_is_fitted__(self) -> bool:
        return self.base_logits_ is not None and self.X_binned_ is not None
