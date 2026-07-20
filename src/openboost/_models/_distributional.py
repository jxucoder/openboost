"""Distributional Gradient Boosting and NGBoost.

Phase 15.2-15.3: Probabilistic prediction via distributional regression.

Predicts full probability distributions instead of point estimates.
Trains separate tree ensembles for each distribution parameter.

Classes:
- DistributionalGBDT: Uses ordinary gradient descent
- NGBoost: Uses natural gradient descent (faster convergence)

Example:
    ```python
    import openboost as ob
    
    # Standard distributional GBDT
    model = ob.DistributionalGBDT(distribution='normal', n_trees=100)
    model.fit(X_train, y_train)
    
    # Get distribution parameters
    output = model.predict_distribution(X_test)
    mu, sigma = output.params['loc'], output.params['scale']
    
    # Get prediction intervals
    lower, upper = output.interval(alpha=0.1)  # 90% interval
    
    # Sample from predicted distribution
    samples = output.sample(n_samples=100)
    
    # NaturalBoost (recommended)
    model = ob.NaturalBoostNormal(n_trees=500)
    model.fit(X_train, y_train)
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from .._array import BinnedArray, array
from .._callbacks import (
    Callback,
    CallbackManager,
    EarlyStopping,
    TrainingState,
    warn_if_early_stopping_without_eval_set,
)
from .._core._growth import TreeStructure
from .._core._tree import fit_tree
from .._distributions import (
    Distribution,
    DistributionOutput,
    Normal,
    get_distribution,
)
from .._persistence import PersistenceMixin
from .._utils import crps_empirical, crps_gaussian, interval_score, pinball_loss
from .._validation import validate_eval_set, validate_sample_weight

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: Metrics accepted by ``fit(eval_metric=...)``.
EVAL_METRICS = ('nll', 'crps', 'pinball', 'interval_score')


def _validate_exposure(exposure, n_samples: int, context: str = "fit") -> NDArray:
    """Validate an exposure vector: positive, finite, shape (n_samples,).

    Scalars are broadcast to all samples.
    """
    exposure = np.asarray(exposure, dtype=np.float64)
    if exposure.ndim == 0:
        exposure = np.full(n_samples, float(exposure), dtype=np.float64)
    if exposure.ndim != 1 or len(exposure) != n_samples:
        raise ValueError(
            f"exposure must be a scalar or 1D array of length {n_samples} "
            f"(matching X in {context}), got shape {exposure.shape}."
        )
    if not np.all(np.isfinite(exposure)) or np.any(exposure <= 0):
        raise ValueError("exposure must contain strictly positive finite values.")
    return exposure.astype(np.float32)


def _split_eval_exposures(eval_set):
    """Split optional 3-tuple ``(X, y, exposure)`` eval entries.

    Returns ``(pairs, exposures)`` where ``pairs`` is a list of ``(X, y)``
    tuples (ready for ``validate_eval_set``) and ``exposures`` the aligned
    list of per-set exposure vectors (None where not given). Mirrors
    ``validate_eval_set``'s auto-wrap of a single bare tuple.
    """
    if eval_set is None:
        return None, None
    if (
        isinstance(eval_set, tuple)
        and len(eval_set) in (2, 3)
        and (hasattr(eval_set[0], 'shape') or hasattr(eval_set[0], '__len__'))
        and not isinstance(eval_set[0], tuple)
    ):
        eval_set = [eval_set]

    pairs, exposures = [], []
    for item in eval_set:
        if isinstance(item, tuple) and len(item) == 3:
            pairs.append((item[0], item[1]))
            exposures.append(item[2])
        else:
            # 2-tuples (and malformed items, which validate_eval_set rejects
            # with its standard message) pass through unchanged
            pairs.append(item)
            exposures.append(None)
    return pairs, exposures


@dataclass
class DistributionalGBDT(PersistenceMixin):
    """Distributional Gradient Boosting for probabilistic prediction.
    
    Trains K tree ensembles, where K = number of distribution parameters.
    Each ensemble predicts one parameter (e.g., mean, variance).
    Uses ordinary gradient descent.
    
    For faster convergence, consider using NGBoost (natural gradient).
    
    Args:
        distribution: Distribution name ('normal', 'gamma', 'poisson', etc.)
                     or Distribution instance
        n_trees: Number of boosting rounds
        max_depth: Maximum depth of each tree
        learning_rate: Shrinkage factor applied to each tree
        min_child_weight: Minimum sum of hessian in a leaf
        reg_lambda: L2 regularization on leaf values
        reg_alpha: L1 regularization on leaf values
        subsample: Row sampling ratio (0.0-1.0)
        colsample_bytree: Column sampling ratio (0.0-1.0)
        n_bins: Number of bins for histogram building
        
    Attributes:
        trees_: Dict mapping param_name -> list of trees
        distribution_: Fitted Distribution instance
        evals_result_: Per-round eval-set metric history recorded during
            fit(), e.g. ``{'eval_0': {'nll': [...]}, 'eval_1': {'nll': [...]}}``
        best_iteration_: Best round index (set when early stopping is used)
        best_score_: Best monitored metric value (set with best_iteration_)

    Example:
        ```python
        model = DistributionalGBDT(distribution='normal', n_trees=100)
        model.fit(X_train, y_train)
        
        # Point prediction (mean)
        y_pred = model.predict(X_test)
        
        # Full distribution
        output = model.predict_distribution(X_test)
        lower, upper = output.interval(alpha=0.1)
        ```
    """
    
    distribution: (
        Literal['normal', 'lognormal', 'gamma', 'poisson', 'studentt', 'tweedie', 'negbin']
        | Distribution
    ) = 'normal'
    n_trees: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    n_bins: int = 256
    
    # Fitted attributes (not init)
    trees_: dict[str, list[TreeStructure]] = field(default_factory=dict, init=False, repr=False)
    distribution_: Distribution | None = field(default=None, init=False, repr=False)
    evals_result_: dict[str, dict[str, list[float]]] = field(
        default_factory=dict, init=False, repr=False
    )
    X_binned_: BinnedArray | None = field(default=None, init=False, repr=False)
    _base_scores: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    n_features_in_: int = field(default=0, init=False, repr=False)
    
    def fit(
        self,
        X: NDArray,
        y: NDArray,
        sample_weight: NDArray | None = None,
        exposure: NDArray | None = None,
        callbacks: list[Callback] | None = None,
        eval_set: list[tuple] | None = None,
        eval_metric: Literal['nll', 'crps', 'pinball', 'interval_score'] = 'nll',
        quantiles: list[float] | None = None,
        interval_alpha: float = 0.1,
        early_stopping_rounds: int | None = None,
    ) -> DistributionalGBDT:
        """Fit the distributional gradient boosting model.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training targets, shape (n_samples,)
            sample_weight: Optional non-negative per-sample weights ``w_i``,
                shape (n_samples,). The training objective becomes
                ``sum_i w_i * NLL_i`` and the reported train loss is the
                weighted mean NLL.
            exposure: Optional strictly positive per-sample exposure ``e_i``
                (scalar or shape (n_samples,)) for families with a log-link
                mean (Poisson, NegativeBinomial, Gamma, Tweedie): the mean is
                modeled as ``mu_i = e_i * exp(raw_mu_i)``, i.e. ``log(e_i)``
                is added as an offset to the mean parameter's raw score.
                Raises ValueError for families without a log-link mean
                (Normal, LogNormal, StudentT).
            callbacks: List of Callback instances (e.g. EarlyStopping, Logger).
            eval_set: Validation set(s) as a list of ``(X_val, y_val)``
                tuples. Entries may also be 3-tuples
                ``(X_val, y_val, exposure_val)`` for exposure-aware
                validation of log-link-mean families. Every eval set is
                evaluated each round and the per-round history is stored in
                ``evals_result_``.
            eval_metric: Metric computed on each eval set every round:
                'nll' (default), 'crps' (closed form for Normal, otherwise
                estimated from 100 fixed-seed Monte Carlo samples per round),
                'pinball' (mean pinball loss over ``quantiles``), or
                'interval_score' (central interval with miscoverage
                ``interval_alpha``).
            quantiles: Quantile levels used by eval_metric='pinball'.
                Defaults to ``[0.05, 0.5, 0.95]``.
            interval_alpha: Miscoverage rate used by
                eval_metric='interval_score' (0.1 scores the central 90%
                interval).
            early_stopping_rounds: Stop training when the LAST eval set's
                metric has not improved for this many consecutive rounds.
                The model is restored to (truncated at) the best iteration
                and ``best_iteration_`` / ``best_score_`` are set.

        Returns:
            self: Fitted model
        """
        # Get distribution instance
        self.distribution_ = get_distribution(self.distribution)

        y = np.asarray(y, dtype=np.float32).ravel()
        n_samples = len(y)

        sample_weight = validate_sample_weight(sample_weight, n_samples)

        if eval_metric not in EVAL_METRICS:
            raise ValueError(
                f"Unknown eval_metric '{eval_metric}'. "
                f"Available: {', '.join(EVAL_METRICS)}."
            )

        # Split optional (X, y, exposure) eval entries before validation
        eval_pairs, eval_exposures = _split_eval_exposures(eval_set)

        # Resolve which raw parameter carries the log-exposure offset
        # (raises ValueError for families without a log-link mean)
        exposure_param: str | None = None
        exposure_sign = 0.0
        needs_exposure = exposure is not None or any(
            e is not None for e in (eval_exposures or [])
        )
        if needs_exposure:
            exposure_param, exposure_sign = self._resolve_exposure_offset()

        train_log_offset = None
        if exposure is not None:
            exposure = _validate_exposure(exposure, n_samples, context="fit")
            train_log_offset = (exposure_sign * np.log(exposure)).astype(np.float32)

        # Bin features
        if isinstance(X, BinnedArray):
            self.X_binned_ = X
        else:
            self.X_binned_ = array(X, n_bins=self.n_bins)

        self.n_features_in_ = self.X_binned_.n_features

        # Initialize tree storage
        self.trees_ = {}
        for param_name in self.distribution_.param_names:
            self.trees_[param_name] = []

        # Initialize raw predictions (in link space) using data statistics
        init_params = self.distribution_.init_params(y)
        raw_preds = {}

        for param_name in self.distribution_.param_names:
            raw_init = init_params[param_name]
            self._base_scores[param_name] = float(raw_init)
            raw_preds[param_name] = np.full(n_samples, raw_init, dtype=np.float32)

        # Setup callbacks (early_stopping_rounds is sugar for EarlyStopping)
        cb_list = list(callbacks) if callbacks else []
        if early_stopping_rounds is not None:
            cb_list.append(
                EarlyStopping(patience=early_stopping_rounds, restore_best=True)
            )
        cb_manager = CallbackManager(cb_list)
        state = TrainingState(model=self, n_rounds=self.n_trees)
        cb_manager.on_train_begin(state)

        eval_pairs = validate_eval_set(eval_pairs, self.X_binned_.n_features)
        warn_if_early_stopping_without_eval_set(cb_list, eval_pairs)

        # Per-eval-set state: binned features, targets, incrementally
        # maintained raw scores, and optional log-exposure offset
        eval_data = []
        if eval_pairs:
            for (X_e, y_e), exp_e in zip(eval_pairs, eval_exposures, strict=True):
                X_e_binned = (
                    X_e if isinstance(X_e, BinnedArray)
                    else self.X_binned_.transform(X_e)
                )
                raw_e = {
                    p: np.full(
                        X_e_binned.n_samples, self._base_scores[p], dtype=np.float32
                    )
                    for p in self.distribution_.param_names
                }
                log_off_e = None
                if exp_e is not None:
                    exp_e = _validate_exposure(
                        exp_e, X_e_binned.n_samples, context="eval"
                    )
                    log_off_e = (exposure_sign * np.log(exp_e)).astype(np.float32)
                eval_data.append((X_e_binned, y_e, raw_e, log_off_e))

        self.evals_result_ = {
            f'eval_{i}': {eval_metric: []} for i in range(len(eval_data))
        }

        # Training loop
        for _round_idx in range(self.n_trees):
            # Constrained params from raw scores (+ exposure offset)
            params = self._constrained_params(
                raw_preds, exposure_param, train_log_offset
            )

            # Get gradients for each parameter
            grads_dict = self._compute_gradients(y, params)

            if sample_weight is not None:
                # Weighted likelihood: the objective is sum_i w_i * NLL_i.
                #
                # Ordinary-gradient path (DistributionalGBDT): grad_i/hess_i
                # are per-sample derivatives of NLL_i, so both scale linearly
                # in w_i; the Newton leaf value -Σ w_i g_i / (Σ w_i h_i + λ)
                # is then the correct weighted step.
                #
                # Natural-gradient path (NaturalBoost): under the weighted
                # likelihood the per-sample Fisher information also scales by
                # w_i, so the per-sample natural gradient
                # (w_i F_i)^{-1} (w_i g_i) = F_i^{-1} g_i is weight-INVARIANT.
                # The correct weighted natural-gradient aggregate is obtained
                # by scaling the per-sample natural gradient (returned by
                # _compute_gradients with unit hessians) by w_i, and scaling
                # its unit hessian by w_i so leaf aggregation becomes the
                # weighted mean -Σ w_i g̃_i / (Σ w_i + λ). Post-scaling both
                # grad and hess by w_i therefore covers both paths.
                grads_dict = {
                    p: (
                        (g * sample_weight).astype(np.float32),
                        (h * sample_weight).astype(np.float32),
                    )
                    for p, (g, h) in grads_dict.items()
                }

            # Train one tree per parameter
            for param_name in self.distribution_.param_names:
                grad, hess = grads_dict[param_name]

                tree = fit_tree(
                    self.X_binned_,
                    grad,
                    hess,
                    max_depth=self.max_depth,
                    min_child_weight=self.min_child_weight,
                    reg_lambda=self.reg_lambda,
                    reg_alpha=self.reg_alpha,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                )

                self.trees_[param_name].append(tree)

                # Update raw predictions (add tree prediction, as in standard GBDT)
                # Tree is trained on gradients, so it outputs the negative gradient direction
                tree_pred = tree(self.X_binned_)
                if hasattr(tree_pred, 'copy_to_host'):
                    tree_pred = tree_pred.copy_to_host()

                raw_preds[param_name] += self.learning_rate * tree_pred

                # Keep eval-set raw scores in sync (incremental, avoids
                # re-running all trees each round)
                for X_e_binned, _y_e, raw_e, _log_off_e in eval_data:
                    pred_e = tree(X_e_binned)
                    if hasattr(pred_e, 'copy_to_host'):
                        pred_e = pred_e.copy_to_host()
                    raw_e[param_name] += self.learning_rate * pred_e

            # Evaluate ALL eval sets and record per-round history
            last_metric = None
            for i, (_X_e_binned, y_e, raw_e, log_off_e) in enumerate(eval_data):
                params_e = self._constrained_params(
                    raw_e, exposure_param, log_off_e
                )
                last_metric = self._eval_metric_value(
                    y_e, params_e, eval_metric, quantiles, interval_alpha
                )
                self.evals_result_[f'eval_{i}'][eval_metric].append(last_metric)

            # Callbacks
            state.round_idx = _round_idx
            if cb_manager.callbacks:
                # Train loss: (weighted) mean NLL of the post-round params,
                # matching the timing of the eval-set metrics. Recomputed from
                # raw_preds because identity-link params alias raw_preds and
                # were mutated in place by the tree updates above.
                params_report = self._constrained_params(
                    raw_preds, exposure_param, train_log_offset
                )
                state.train_loss = float(
                    np.average(
                        self.distribution_.nll(y, params_report),
                        weights=sample_weight,
                    )
                )
                if last_metric is not None:
                    # Early stopping monitors the LAST eval set's metric
                    state.val_loss = last_metric
                if not cb_manager.on_round_end(state):
                    break

        cb_manager.on_train_end(state)
        return self

    def _resolve_exposure_offset(self) -> tuple[str, float]:
        """Resolve (and verify) the raw parameter carrying the log-exposure offset.

        Returns:
            (param_name, sign) such that adding ``sign * log(e)`` to that
            parameter's raw score multiplies the distribution mean by ``e``.

        Raises:
            ValueError: If the family has no log-link mean (exposure unsupported).
        """
        dist = self.distribution_
        info = dist.exposure_offset
        if info is None:
            raise ValueError(
                f"exposure is not supported for {type(dist).__name__}"
            )
        name, sign = info

        # Verify (rather than trust) the declared offset against the actual
        # Distribution object: the offset parameter must have a multiplicative
        # (log) link, and shifting its raw score by sign*log(k) must multiply
        # the mean by k. Two baselines guard against coincidental matches.
        k = 2.0
        c = float(np.log(k))
        for base in (0.25, -0.4):
            raws = {p: np.array([base]) for p in dist.param_names}
            params0 = {p: dist.link(p, raws[p]) for p in dist.param_names}
            raws_off = dict(raws)
            raws_off[name] = raws[name] + sign * c
            params1 = {p: dist.link(p, raws_off[p]) for p in dist.param_names}
            link_multiplicative = np.allclose(
                params1[name], params0[name] * k ** sign, rtol=1e-5
            )
            mean_scales = np.allclose(
                dist.mean(params1), k * dist.mean(params0), rtol=1e-5
            )
            if not (link_multiplicative and mean_scales):
                raise ValueError(
                    f"exposure is not supported for {type(dist).__name__}: "
                    f"declared exposure_offset {info} failed verification "
                    "(offset does not scale the mean multiplicatively)."
                )
        return name, float(sign)

    def _constrained_params(
        self,
        raw_preds: dict[str, NDArray],
        exposure_param: str | None = None,
        log_offset: NDArray | None = None,
    ) -> dict[str, NDArray]:
        """Apply link functions (plus optional log-exposure offset) to raw scores."""
        params = {}
        for p in self.distribution_.param_names:
            raw = raw_preds[p]
            if log_offset is not None and p == exposure_param:
                raw = raw + log_offset
            params[p] = self.distribution_.link(p, raw)
        return params

    def _eval_metric_value(
        self,
        y: NDArray,
        params: dict[str, NDArray],
        metric: str,
        quantiles: list[float] | None,
        interval_alpha: float,
    ) -> float:
        """Compute one eval-set metric value from constrained parameters."""
        if metric == 'nll':
            return float(np.mean(self.distribution_.nll(y, params)))
        if metric == 'crps':
            if isinstance(self.distribution_, Normal):
                return crps_gaussian(y, params['loc'], params['scale'])
            # Non-Gaussian families: empirical CRPS from Monte Carlo samples.
            # Fixed seed keeps per-round values comparable across rounds.
            samples = self.distribution_.sample(params, n_samples=100, seed=0)
            return crps_empirical(y, samples)
        if metric == 'pinball':
            qs = list(quantiles) if quantiles is not None else [0.05, 0.5, 0.95]
            losses = [
                pinball_loss(y, self.distribution_.quantile(params, q), quantile=q)
                for q in qs
            ]
            return float(np.mean(losses))
        if metric == 'interval_score':
            lower = self.distribution_.quantile(params, interval_alpha / 2)
            upper = self.distribution_.quantile(params, 1 - interval_alpha / 2)
            return interval_score(y, lower, upper, alpha=interval_alpha)
        raise ValueError(f"Unknown eval_metric '{metric}'.")  # pragma: no cover
    
    def _compute_gradients(
        self,
        y: NDArray,
        params: dict[str, NDArray],
    ) -> dict[str, tuple[NDArray, NDArray]]:
        """Compute gradients (ordinary gradient descent).
        
        Subclasses can override for different gradient computation.
        """
        return self.distribution_.nll_gradient(y, params)
    
    def _predict_raw(self, X: NDArray | BinnedArray) -> dict[str, NDArray]:
        """Predict raw (link-space) parameters.
        
        Args:
            X: Features to predict on
            
        Returns:
            Dictionary mapping param_name -> raw predictions
        """
        if not self.trees_:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Bin the data if needed, using training bin edges for consistency
        if isinstance(X, BinnedArray):
            X_binned = X
        elif self.X_binned_ is not None:
            # Use transform to apply training bin edges to new data
            X_binned = self.X_binned_.transform(X)
        else:
            X_binned = array(X, n_bins=self.n_bins)
        
        n_samples = X_binned.n_samples
        raw_preds = {}
        
        for param_name in self.distribution_.param_names:
            # Start with base score
            pred = np.full(n_samples, self._base_scores[param_name], dtype=np.float32)
            
            # Accumulate tree predictions
            for tree in self.trees_[param_name]:
                tree_pred = tree(X_binned)
                if hasattr(tree_pred, 'copy_to_host'):
                    tree_pred = tree_pred.copy_to_host()
                pred += self.learning_rate * tree_pred
            
            raw_preds[param_name] = pred
        
        return raw_preds
    
    def predict_params(
        self,
        X: NDArray | BinnedArray,
        exposure: NDArray | None = None,
    ) -> dict[str, NDArray]:
        """Predict distribution parameters.

        Args:
            X: Features to predict on
            exposure: Optional strictly positive per-sample exposure (scalar
                or shape (n_samples,)) for log-link-mean families: the mean
                parameter becomes ``e_i * exp(raw_mu_i)``. Default None
                (exposure 1). Raises ValueError for families without a
                log-link mean.

        Returns:
            Dictionary mapping param_name -> predicted values
            (in constrained parameter space)
        """
        raw_preds = self._predict_raw(X)

        if exposure is not None:
            name, sign = self._resolve_exposure_offset()
            n = next(iter(raw_preds.values())).shape[0]
            exposure = _validate_exposure(exposure, n, context="predict")
            log_offset = (sign * np.log(exposure)).astype(np.float32)
            return self._constrained_params(raw_preds, name, log_offset)

        return self._constrained_params(raw_preds)

    def predict_distribution(
        self,
        X: NDArray | BinnedArray,
        exposure: NDArray | None = None,
    ) -> DistributionOutput:
        """Predict full distribution.

        Args:
            X: Features to predict on
            exposure: Optional per-sample exposure (see ``predict_params``)

        Returns:
            DistributionOutput with params, mean(), variance(),
            quantile(), interval(), sample() methods
        """
        params = self.predict_params(X, exposure=exposure)
        return DistributionOutput(params=params, distribution=self.distribution_)

    def predict(
        self,
        X: NDArray | BinnedArray,
        exposure: NDArray | None = None,
    ) -> NDArray:
        """Predict mean (expected value).

        This provides a point prediction for compatibility with standard GBDT.

        Args:
            X: Features to predict on
            exposure: Optional per-sample exposure (see ``predict_params``)

        Returns:
            Predicted mean values
        """
        params = self.predict_params(X, exposure=exposure)
        return self.distribution_.mean(params)

    def predict_interval(
        self,
        X: NDArray | BinnedArray,
        alpha: float = 0.1,
        exposure: NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Predict (1-alpha) prediction interval.

        Args:
            X: Features to predict on
            alpha: Significance level (0.1 = 90% interval)
            exposure: Optional per-sample exposure (see ``predict_params``)

        Returns:
            (lower, upper) bounds
        """
        output = self.predict_distribution(X, exposure=exposure)
        return output.interval(alpha)

    def predict_quantile(
        self,
        X: NDArray | BinnedArray,
        q: float,
        exposure: NDArray | None = None,
    ) -> NDArray:
        """Predict q-th quantile.

        Args:
            X: Features to predict on
            q: Quantile level (0 < q < 1)
            exposure: Optional per-sample exposure (see ``predict_params``)

        Returns:
            Predicted quantiles
        """
        output = self.predict_distribution(X, exposure=exposure)
        return output.quantile(q)

    def sample(
        self,
        X: NDArray | BinnedArray,
        n_samples: int = 1,
        seed: int | None = None,
        exposure: NDArray | None = None,
    ) -> NDArray:
        """Sample from predicted distribution.

        Args:
            X: Features, shape (n, n_features)
            n_samples: Number of samples per observation
            seed: Random seed for reproducibility
            exposure: Optional per-sample exposure (see ``predict_params``)

        Returns:
            samples: Shape (n, n_samples)
        """
        output = self.predict_distribution(X, exposure=exposure)
        return output.sample(n_samples, seed)

    def score(
        self,
        X: NDArray | BinnedArray,
        y: NDArray,
        exposure: NDArray | None = None,
    ) -> float:
        """Compute negative log-likelihood (lower is better).

        Args:
            X: Features
            y: True target values
            exposure: Optional per-sample exposure (see ``predict_params``)

        Returns:
            Mean negative log-likelihood
        """
        output = self.predict_distribution(X, exposure=exposure)
        nll = output.nll(np.asarray(y, dtype=np.float32))
        return float(np.mean(nll))

    def nll(
        self,
        X: NDArray | BinnedArray,
        y: NDArray,
        exposure: NDArray | None = None,
    ) -> float:
        """Alias for score() - compute mean NLL."""
        return self.score(X, y, exposure=exposure)

    def _post_load(self) -> None:
        """Recreate distribution instance after loading from file."""
        if self.distribution_ is None and self.distribution is not None:
            self.distribution_ = get_distribution(self.distribution)


@dataclass
class NaturalBoost(DistributionalGBDT):
    """Natural Gradient Boosting for probabilistic prediction.
    
    OpenBoost's implementation of natural gradient boosting, inspired by NGBoost.
    Uses natural gradient instead of ordinary gradient, leading to faster
    convergence by accounting for the geometry of the parameter space.
    
    Natural gradient: F^{-1} @ ordinary_gradient
    where F is the Fisher information matrix.
    
    Key advantages over standard GBDT:
    - Full probability distributions, not just point estimates
    - Prediction intervals and uncertainty quantification
    - Faster convergence than ordinary gradient descent
    
    Key advantages over official NGBoost:
    - GPU acceleration via histogram-based trees
    - Faster on large datasets (>10k samples)
    - Custom distributions with autodiff support
    
    Reference:
        Duan et al. "NGBoost: Natural Gradient Boosting for Probabilistic
        Prediction." ICML 2020.
    
    Args:
        distribution: Distribution name or instance
        n_trees: Number of boosting rounds (often needs fewer than ordinary)
        max_depth: Maximum depth of each tree (default 4, often smaller is better)
        learning_rate: Shrinkage factor
        min_child_weight: Minimum sum of hessian in a leaf
        reg_lambda: L2 regularization
        n_bins: Number of bins for histogram building
        
    Example:
        ```python
        model = NaturalBoost(distribution='normal', n_trees=500)
        model.fit(X_train, y_train)
        
        # Get prediction intervals
        lower, upper = model.predict_interval(X_test, alpha=0.1)
        
        # Get full distribution
        output = model.predict_distribution(X_test)
        samples = output.sample(n_samples=1000)
        ```
    """
    
    # Override defaults for NaturalBoost
    max_depth: int = 4  # Shallower trees often work better
    learning_rate: float = 0.1
    
    def _compute_gradients(
        self,
        y: NDArray,
        params: dict[str, NDArray],
    ) -> dict[str, tuple[NDArray, NDArray]]:
        """Compute natural gradients.
        
        Natural gradient = F^{-1} @ ordinary_gradient
        where F is the Fisher information matrix.
        """
        return self.distribution_.natural_gradient(y, params)


# =============================================================================
# Convenience aliases
# =============================================================================

# NaturalBoost with specific distributions
def NaturalBoostNormal(**kwargs) -> NaturalBoost:
    """NaturalBoost with Normal distribution."""
    return NaturalBoost(distribution='normal', **kwargs)


def NaturalBoostLogNormal(**kwargs) -> NaturalBoost:
    """NaturalBoost with LogNormal distribution (for positive data)."""
    return NaturalBoost(distribution='lognormal', **kwargs)


def NaturalBoostGamma(**kwargs) -> NaturalBoost:
    """NaturalBoost with Gamma distribution (for positive data)."""
    return NaturalBoost(distribution='gamma', **kwargs)


def NaturalBoostPoisson(**kwargs) -> NaturalBoost:
    """NaturalBoost with Poisson distribution (for count data)."""
    return NaturalBoost(distribution='poisson', **kwargs)


def NaturalBoostStudentT(**kwargs) -> NaturalBoost:
    """NaturalBoost with Student-t distribution (for heavy-tailed data)."""
    return NaturalBoost(distribution='studentt', **kwargs)


# =============================================================================
# Kaggle Competition Favorites
# =============================================================================

def NaturalBoostTweedie(power: float = 1.5, **kwargs) -> NaturalBoost:
    """NaturalBoost with Tweedie distribution (for insurance claims, zero-inflated data).
    
    **Kaggle Use Cases**:
    - Porto Seguro Safe Driver Prediction
    - Allstate Claims Severity
    - Any zero-inflated positive target
    
    Args:
        power: Tweedie power parameter (1 < power < 2).
               1.5 is the default for insurance claims.
        **kwargs: Other NaturalBoost parameters (n_trees, learning_rate, etc.)
        
    Example:
        ```python
        model = NaturalBoostTweedie(power=1.5, n_trees=500)
        model.fit(X_train, y_train)  # y has zeros and positive values
        
        # Get prediction intervals (XGBoost can't do this!)
        lower, upper = model.predict_interval(X_test, alpha=0.1)
        ```
    """
    from .._distributions import Tweedie
    return NaturalBoost(distribution=Tweedie(power=power), **kwargs)


def NaturalBoostNegBin(**kwargs) -> NaturalBoost:
    """NaturalBoost with Negative Binomial distribution (for overdispersed count data).
    
    **Kaggle Use Cases**:
    - Rossmann Store Sales
    - Bike Sharing Demand
    - Grupo Bimbo Inventory Demand
    - Any count prediction where variance > mean
    
    Args:
        **kwargs: NaturalBoost parameters (n_trees, learning_rate, etc.)
        
    Example:
        ```python
        model = NaturalBoostNegBin(n_trees=500)
        model.fit(X_train, y_train)  # y is count data
        
        # Probability of exceeding threshold (demand planning!)
        output = model.predict_distribution(X_test)
        prob_high_demand = output.distribution.prob_exceed(output.params, 100)
        ```
    """
    return NaturalBoost(distribution='negativebinomial', **kwargs)


# =============================================================================
# Backward compatibility aliases (deprecated)
# =============================================================================

# Keep old names working but mark as deprecated
NGBoost = NaturalBoost  # Alias for backward compatibility
NGBoostNormal = NaturalBoostNormal
NGBoostLogNormal = NaturalBoostLogNormal
NGBoostGamma = NaturalBoostGamma
NGBoostPoisson = NaturalBoostPoisson
NGBoostStudentT = NaturalBoostStudentT
NGBoostTweedie = NaturalBoostTweedie
NGBoostNegBin = NaturalBoostNegBin
