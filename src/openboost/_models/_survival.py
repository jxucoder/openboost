"""WeibullAFT: boosted Weibull survival regression with censoring.

Both the scale ``lambda(z)`` and the shape ``k(z)`` are boosting ensembles over
the features ``z`` and trained on a right-censored negative log-likelihood.
XGBoost's ``survival:aft`` learns only the location and holds the distribution
scale as a single global hyperparameter, so it cannot vary the Weibull shape
with covariates; NaturalBoost's built-in distributions have no censored
likelihood at all. WeibullAFT expresses both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._array import BinnedArray
from .._callbacks import Callback
from .._core._growth import TreeStructure
from .._objectives import WeibullAFTObjective
from .._persistence import PersistenceMixin
from .._trainer import TrainerConfig, fit_boosting, predict_raw


def _as_1d(a, n: int, name: str) -> NDArray:
    arr = np.asarray(a, dtype=np.float64).ravel()
    if arr.shape[0] != n:
        raise ValueError(
            f"{name} has length {arr.shape[0]}, expected {n} (matching y)."
        )
    return arr


@dataclass
class WeibullAFT(PersistenceMixin):
    """Boosted Weibull accelerated-failure-time model with right censoring.

    Args:
        damp: Levenberg-Marquardt damping on the per-sample 2x2 information.
        n_trees, max_depth, learning_rate, ...: standard tree knobs.

    Example:
        ```python
        m = WeibullAFT(n_trees=300, max_depth=3, learning_rate=0.1)
        m.fit(Z, time, event=observed)          # event: 1 seen, 0 censored
        params = m.predict_params(Z)            # per-sample {scale, shape}
        t_hat = m.predict(Z)                    # predicted median time
        s = m.predict_survival(Z, t=5.0)        # S(5.0 | z) per sample
        ```
    """

    damp: float = 1.0
    n_trees: int = 100
    max_depth: int = 3
    learning_rate: float = 0.1
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    n_bins: int = 254

    trees_: dict[str, list[TreeStructure]] = field(
        default_factory=dict, init=False, repr=False
    )
    evals_result_: dict[str, dict[str, list[float]]] = field(
        default_factory=dict, init=False, repr=False
    )
    X_binned_: BinnedArray | None = field(default=None, init=False, repr=False)
    _base_scores: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    n_features_in_: int = field(default=0, init=False, repr=False)
    _objective: WeibullAFTObjective | None = field(
        default=None, init=False, repr=False
    )

    def _make_objective(self) -> WeibullAFTObjective:
        return WeibullAFTObjective(damp=self.damp)

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        event: NDArray | None = None,
        sample_weight: NDArray | None = None,
        callbacks: list[Callback] | None = None,
        eval_set: list[tuple] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> WeibullAFT:
        """Fit ``lambda(z)``, ``k(z)`` from ``(X, time, event)``.

        ``y`` is the observed time (>0). ``event`` is 1 for an observed event
        and 0 for right-censored (default: all observed). ``eval_set`` entries
        are ``(X_val, y_val, event_val)``.
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        if np.any(y <= 0):
            raise ValueError("WeibullAFT requires strictly positive times y.")
        ev = None if event is None else _as_1d(event, len(y), "event")
        self._objective = self._make_objective()

        eval_sets: list[dict[str, Any]] | None = None
        if eval_set is not None:
            if (
                isinstance(eval_set, tuple)
                and len(eval_set) in (2, 3)
                and not isinstance(eval_set[0], tuple)
            ):
                eval_set = [eval_set]
            eval_sets = []
            for item in eval_set:
                if not (isinstance(item, tuple) and len(item) in (2, 3)):
                    raise ValueError(
                        "WeibullAFT eval_set entries must be "
                        "(X_val, y_val[, event_val])."
                    )
                X_e, y_e = item[0], np.asarray(item[1], dtype=np.float64).ravel()
                ev_e = (_as_1d(item[2], len(y_e), "event")
                        if len(item) == 3 else None)
                eval_sets.append(
                    {"X": X_e, "y": y_e, "extra": {"event": ev_e}}
                )

        fit_boosting(
            self,
            self._objective,
            X,
            y,
            config=TrainerConfig(
                n_trees=self.n_trees,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                n_bins=self.n_bins,
            ),
            sample_weight=sample_weight,
            extra={"event": ev},
            callbacks=callbacks,
            early_stopping_rounds=early_stopping_rounds,
            eval_sets=eval_sets,
            eval_metric_name="nll",
        )
        return self

    def predict_params(self, X: NDArray | BinnedArray) -> dict[str, NDArray]:
        """Per-sample constrained Weibull parameters ``{scale, shape}``."""
        if self._objective is None:
            self._objective = self._make_objective()
        return self._objective.constrain(predict_raw(self, X))

    def predict_quantile(self, X: NDArray | BinnedArray, q: float = 0.5) -> NDArray:
        """Predict the time ``t`` at which ``P(T <= t) = q`` for each row."""
        if not 0.0 < q < 1.0:
            raise ValueError("q must be in (0, 1).")
        params = self.predict_params(X)
        lam, k = params["scale"], params["shape"]
        return lam * (-np.log(1.0 - q)) ** (1.0 / k)

    def predict_median(self, X: NDArray | BinnedArray) -> NDArray:
        return self.predict_quantile(X, 0.5)

    def predict(self, X: NDArray | BinnedArray) -> NDArray:
        """Point prediction = predicted median survival time."""
        return self.predict_median(X)

    def predict_survival(self, X: NDArray | BinnedArray, t: float | NDArray) -> NDArray:
        """Survival probability ``S(t | z) = exp(-(t / lambda)^k)`` per row."""
        params = self.predict_params(X)
        lam, k = params["scale"], params["shape"]
        t = np.asarray(t, dtype=np.float64)
        return np.exp(-((t / lam) ** k))

    def nll(
        self,
        X: NDArray | BinnedArray,
        y: NDArray,
        event: NDArray | None = None,
    ) -> float:
        """Mean censored negative log-likelihood on ``(X, y, event)``."""
        if self._objective is None:
            self._objective = self._make_objective()
        y = np.asarray(y, dtype=np.float64).ravel()
        ev = None if event is None else _as_1d(event, len(y), "event")
        raw = predict_raw(self, X)
        return self._objective.loss_value(raw, y, None, {"event": ev})
