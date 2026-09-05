"""FormulaBoost: boost every parameter of a user formula.

Varying-coefficient model. Features ``Z`` determine
parameter surfaces ``theta(Z)`` via trees; a user formula
``y ≈ f(theta, x)`` consumes those parameters and a structural input ``x``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._array import BinnedArray
from .._callbacks import Callback
from .._core._growth import TreeStructure
from .._objectives import FormulaObjective
from .._persistence import PersistenceMixin
from .._trainer import TrainerConfig, fit_boosting, predict_raw
from .._validation import validate_1d


@dataclass
class FormulaBoost(PersistenceMixin):
    """Boost the parameters of an arbitrary differentiable formula.

    Args:
        formula: ``formula(theta, x) -> yhat``. ``theta`` is a tuple of K
            arrays in constrained parameter space; ``x`` is the structural
            input (e.g. spend, dose).
        n_params: Number of formula parameters K.
        links: Per-parameter link (``identity``, ``log``, ``softplus``,
            ``sigmoid``), length K.
        loss: Training loss. Currently ``mse`` only.
        precond: GGN preconditioner: ``full`` (default), ``diag``, or
            ``plain`` (raw gradient, usually a bad idea).
        damp: Levenberg–Marquardt damping added to the GGN matrix.
        param_names: Optional names for the K parameters. Defaults to
            ``theta_0``, ``theta_1``, ...

    Tree knobs (``n_trees``, ``max_depth``, ``learning_rate``, ...) match
    ``GradientBoosting``.

    Example:
        ```python
        def curve(theta, x):
            a, b = theta
            return a * x ** (1.0 / (1.0 + np.exp(-b * x)))

        m = FormulaBoost(
            formula=curve, n_params=2, links=("log", "identity"),
            param_names=("a", "b"),
        )
        m.fit(Z, y, model_input=x)
        params = m.predict_params(Z)          # per-sample (a, b)
        yhat = m.predict(Z, model_input=x_new)
        ```
    """

    formula: Callable
    n_params: int
    links: tuple[str, ...]
    loss: str = "mse"
    precond: str = "full"
    damp: float = 1.0
    param_names: tuple[str, ...] | None = None
    n_trees: int = 100
    max_depth: int = 3
    learning_rate: float = 0.1
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    n_bins: int = 254
    random_state: int | None = None

    trees_: dict[str, list[TreeStructure]] = field(
        default_factory=dict, init=False, repr=False
    )
    evals_result_: dict[str, dict[str, list[float]]] = field(
        default_factory=dict, init=False, repr=False
    )
    X_binned_: BinnedArray | None = field(default=None, init=False, repr=False)
    _base_scores: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    n_features_in_: int = field(default=0, init=False, repr=False)
    _objective: FormulaObjective | None = field(default=None, init=False, repr=False)

    def _make_objective(self) -> FormulaObjective:
        return FormulaObjective(
            self.formula,
            self.n_params,
            self.links,
            loss=self.loss,
            precond=self.precond,
            damp=self.damp,
            param_names=self.param_names,
        )

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        model_input: NDArray,
        sample_weight: NDArray | None = None,
        callbacks: list[Callback] | None = None,
        eval_set: list[tuple] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> FormulaBoost:
        """Fit parameter surfaces ``theta(Z)`` from ``(X, y, model_input)``.

        ``eval_set`` entries are ``(X_val, y_val, model_input_val)``.
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        x = validate_1d(model_input, len(y), "model_input")
        self._objective = self._make_objective()

        eval_sets: list[dict[str, Any]] | None = None
        if eval_set is not None:
            if (
                isinstance(eval_set, tuple)
                and len(eval_set) == 3
                and not isinstance(eval_set[0], tuple)
            ):
                eval_set = [eval_set]
            eval_sets = []
            for item in eval_set:
                if not (isinstance(item, tuple) and len(item) == 3):
                    raise ValueError(
                        "FormulaBoost eval_set entries must be "
                        "(X_val, y_val, model_input_val)."
                    )
                X_e, y_e, x_e = item
                y_e = np.asarray(y_e, dtype=np.float64).ravel()
                eval_sets.append(
                    {
                        "X": X_e,
                        "y": y_e,
                        "extra": {"model_input": validate_1d(x_e, len(y_e), "model_input")},
                    }
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
                random_state=self.random_state,
            ),
            sample_weight=sample_weight,
            extra={"model_input": x},
            callbacks=callbacks,
            early_stopping_rounds=early_stopping_rounds,
            eval_sets=eval_sets,
            eval_metric_name="mse",
        )
        return self

    def predict_params(self, X: NDArray | BinnedArray) -> dict[str, NDArray]:
        """Predict constrained formula parameters for each row of ``X``."""
        if self._objective is None:
            self._objective = self._make_objective()
        raw = predict_raw(self, X)
        return self._objective.constrain(raw)

    def predict(self, X: NDArray | BinnedArray, model_input: NDArray) -> NDArray:
        """Evaluate the formula at ``predict_params(X)`` and ``model_input``."""
        params = self.predict_params(X)
        names = self._objective.channel_names
        x = validate_1d(model_input, next(iter(params.values())).shape[0], "model_input")
        theta = tuple(params[name] for name in names)
        return np.asarray(self.formula(theta, x), dtype=np.float64).ravel()
