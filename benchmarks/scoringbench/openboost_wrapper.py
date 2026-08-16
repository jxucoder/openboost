"""ScoringBench wrappers for OpenBoost distributional models.

This module intentionally lives in OpenBoost's repository while the integration
is being validated.  It is also shaped as an upstream-ready ScoringBench wrapper:
copy it to ``scoringbench/wrappers/openboost_wrapper.py`` and update the upstream
registry when submitting benchmark results.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np

try:
    from scoringbench.wrappers.base import DistributionPrediction, ProbabilisticWrapper
    from scoringbench.wrappers.quantile_based import quantiles_to_distribution
except ImportError as exc:  # pragma: no cover - depends on the external checkout
    raise ImportError(
        "OpenBoostWrapper requires a ScoringBench checkout on PYTHONPATH. "
        "See benchmarks/scoringbench/README.md."
    ) from exc


class OpenBoostWrapper(ProbabilisticWrapper):
    """OpenBoost NaturalBoost with a Gaussian predictive distribution.

    Parameters mirror ScoringBench's NGBoost Gaussian entry by default: 500
    boosting rounds, learning rate 0.01, depth-3 trees and 99 quantile levels.
    The backend is explicit so CPU and CUDA results cannot be accidentally
    conflated on a leaderboard.

    Parameters
    ----------
    backend:
        ``"cpu"``, ``"cuda"`` or ``"auto"``.  ``"auto"`` uses OpenBoost's
        normal backend detection; reproducible benchmark runs should use an
        explicit backend.
    n_trees:
        Number of NaturalBoost rounds.
    learning_rate:
        Boosting shrinkage.
    max_depth:
        Maximum depth of each parameter tree.
    n_bins:
        Histogram bins. OpenBoost reserves bin 255 for missing values, so 254
        is the largest non-warning value.
    n_quantiles:
        Number of probability levels used to convert the analytic Normal
        distribution into ScoringBench's common PMF representation.
    model_params:
        Additional keyword arguments forwarded to ``NaturalBoostNormal``.
    """

    _VALID_BACKENDS = {"auto", "cpu", "cuda"}

    def __init__(
        self,
        *,
        backend: str = "cpu",
        n_trees: int = 500,
        learning_rate: float = 0.01,
        max_depth: int = 3,
        n_bins: int = 254,
        n_quantiles: int = 99,
        model_params: dict | None = None,
    ) -> None:
        backend = backend.lower()
        if backend not in self._VALID_BACKENDS:
            raise ValueError(
                f"backend must be one of {sorted(self._VALID_BACKENDS)}, got {backend!r}"
            )
        if n_quantiles < 2:
            raise ValueError("n_quantiles must be at least 2")

        self.backend = backend
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.n_quantiles = n_quantiles
        self.model_params = dict(model_params or {})

        self._alphas = np.linspace(
            1 / (n_quantiles + 1),
            n_quantiles / (n_quantiles + 1),
            n_quantiles,
            dtype=np.float64,
        )
        self._model = None
        self._resolved_backend: str | None = None
        self._y_range = (0.0, 1.0)

    @staticmethod
    def _sanitize_X(X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        return np.nan_to_num(X, nan=0.0, posinf=1e7, neginf=-1e7)

    def _backend_context(self):
        import openboost as ob

        if self._resolved_backend is None:
            return nullcontext()
        return ob.backend_context(self._resolved_backend)

    def _require_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

    def fit(self, X, y) -> OpenBoostWrapper:
        import openboost as ob

        X = self._sanitize_X(X)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        valid = np.isfinite(y)
        X, y = X[valid], y[valid]
        if len(y) == 0:
            raise ValueError("No valid finite training samples")

        lo, hi = float(y.min()), float(y.max())
        if lo == hi:
            pad = max(abs(lo) * 1e-6, 1e-7)
            lo, hi = lo - pad, hi + pad
        self._y_range = (lo, hi)

        self._resolved_backend = ob.get_backend() if self.backend == "auto" else self.backend
        params = {
            "n_trees": self.n_trees,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "n_bins": self.n_bins,
            **self.model_params,
        }
        self._model = ob.NaturalBoostNormal(**params)
        with self._backend_context():
            self._model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        self._require_fitted()
        X = self._sanitize_X(X)
        with self._backend_context():
            pred = self._model.predict(X)
        return np.asarray(pred, dtype=np.float64).reshape(-1)

    def predict_distribution(self, X) -> DistributionPrediction:
        self._require_fitted()
        X = self._sanitize_X(X)
        with self._backend_context():
            output = self._model.predict_distribution(X)
            mean = np.asarray(output.mean(), dtype=np.float64).reshape(-1)
            quantiles = np.column_stack(
                [
                    np.asarray(output.quantile(float(alpha)), dtype=np.float64)
                    for alpha in self._alphas
                ]
            )

        return quantiles_to_distribution(
            quantiles,
            self._alphas,
            mean=mean,
            y_range=self._y_range,
        )


class OpenBoostHistogramWrapper(ProbabilisticWrapper):
    """OpenBoost shared-tree histogram distribution trained by CRPS.

    The defaults are the candidate frozen before the
    ``crps_distribution_v1`` development run: 50 target bins, 100 trees,
    learning rate 0.05, depth 6, and Gauss--Newton curvature scale 1.
    """

    def __init__(
        self,
        *,
        n_distribution_bins: int = 50,
        n_trees: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        n_feature_bins: int = 254,
        curvature_scale: float = 1.0,
        temperature_grid: tuple[float, ...] = (1.0,),
        calibration_fraction: float = 0.2,
        calibration_seed: int = 42,
        evaluation_subdivisions: int = 1,
        model_params: dict | None = None,
    ) -> None:
        self.n_distribution_bins = n_distribution_bins
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_feature_bins = n_feature_bins
        self.curvature_scale = curvature_scale
        self.temperature_grid = tuple(float(value) for value in temperature_grid)
        self.calibration_fraction = calibration_fraction
        self.calibration_seed = calibration_seed
        self.evaluation_subdivisions = evaluation_subdivisions
        self.model_params = dict(model_params or {})
        self._model = None
        self._selected_temperature = 1.0
        self._temperature_scores: dict[float, float] = {}

    @staticmethod
    def _sanitize_X(X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        # HistogramBoost handles NaN explicitly; only infinities need a finite
        # sentinel to match the other ScoringBench wrappers.
        return np.nan_to_num(X, nan=np.nan, posinf=1e7, neginf=-1e7)

    def _require_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

    def fit(self, X, y) -> OpenBoostHistogramWrapper:
        import openboost as ob

        X = self._sanitize_X(X)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        valid = np.isfinite(y)
        X, y = X[valid], y[valid]
        if len(y) == 0:
            raise ValueError("No valid finite training samples")
        if not self.temperature_grid or any(
            not np.isfinite(value) or value <= 0.0 for value in self.temperature_grid
        ):
            raise ValueError("temperature_grid must contain positive finite values")
        if not 0.0 < self.calibration_fraction < 1.0:
            raise ValueError("calibration_fraction must lie in (0, 1)")
        if (
            isinstance(self.evaluation_subdivisions, bool)
            or not isinstance(self.evaluation_subdivisions, (int, np.integer))
            or self.evaluation_subdivisions < 1
        ):
            raise ValueError("evaluation_subdivisions must be a positive integer")

        params = {
            "n_distribution_bins": self.n_distribution_bins,
            "n_trees": self.n_trees,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "n_feature_bins": self.n_feature_bins,
            "curvature_scale": self.curvature_scale,
            **self.model_params,
        }

        self._selected_temperature = 1.0
        self._temperature_scores = {}
        if len(self.temperature_grid) > 1 and len(y) >= 4:
            rng = np.random.default_rng(self.calibration_seed)
            indices = rng.permutation(len(y))
            n_calibration = min(
                max(1, int(round(self.calibration_fraction * len(y)))),
                len(y) - 2,
            )
            calibration_idx = indices[:n_calibration]
            inner_train_idx = indices[n_calibration:]
            calibration_model = ob.HistogramBoost(**params).fit(
                X[inner_train_idx],
                y[inner_train_idx],
            )
            calibration_output = calibration_model.predict_distribution(X[calibration_idx])
            for temperature in self.temperature_grid:
                score = np.mean(calibration_output.tempered(temperature).crps(y[calibration_idx]))
                self._temperature_scores[temperature] = float(score)
            self._selected_temperature = min(
                self.temperature_grid,
                key=lambda value: (
                    self._temperature_scores[value],
                    abs(value - 1.0),
                    value,
                ),
            )

        self._model = ob.HistogramBoost(**params).fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.predict_distribution(X).mean, dtype=np.float64).reshape(-1)

    def predict_distribution(self, X) -> DistributionPrediction:
        self._require_fitted()
        output = self._model.predict_distribution(self._sanitize_X(X)).tempered(
            self._selected_temperature
        )
        output = output.subdivide(self.evaluation_subdivisions)
        return DistributionPrediction(
            probas=np.asarray(output.probas, dtype=np.float64),
            bin_edges=np.asarray(output.bin_edges, dtype=np.float64),
            bin_midpoints=np.asarray(output.bin_midpoints, dtype=np.float64),
            mean=np.asarray(output.mean(), dtype=np.float64),
            is_natively_gridded_model=True,
        )
