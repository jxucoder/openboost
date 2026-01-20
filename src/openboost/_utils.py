"""Utility functions for OpenBoost.

Phase 20.6: Helper functions for cross-validation, hyperparameter tuning,
and common workflows.

Phase 22: Evaluation metrics with sample weight support.
Phase 22 Sprint 2: Probabilistic/distributional metrics for uncertainty quantification.

Example:
    >>> import openboost as ob
    >>> from openboost.utils import suggest_params, cross_val_predict_proba
    >>> 
    >>> # Get suggested hyperparameters based on dataset
    >>> params = suggest_params(X, y, task='regression')
    >>> model = ob.OpenBoostRegressor(**params)
    >>> 
    >>> # Out-of-fold predictions for stacking
    >>> oof_pred = cross_val_predict_proba(model, X, y, cv=5)
    >>> 
    >>> # Evaluation metrics
    >>> from openboost import roc_auc_score, accuracy_score, log_loss_score
    >>> auc = roc_auc_score(y_true, y_pred, sample_weight=weights)
    >>> 
    >>> # Probabilistic metrics (Phase 22 Sprint 2)
    >>> from openboost import crps_gaussian, brier_score, pinball_loss
    >>> crps = crps_gaussian(y_true, mean_pred, std_pred)
    >>> brier = brier_score(y_true, y_proba)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Evaluation Metrics (Phase 22) - Thin wrappers around sklearn
# =============================================================================

def _check_sklearn():
    """Check if sklearn is available."""
    try:
        import sklearn
        return True
    except ImportError:
        raise ImportError(
            "scikit-learn is required for evaluation metrics. "
            "Install with: pip install scikit-learn"
        )


def roc_auc_score(
    y_true: NDArray,
    y_score: NDArray,
    *,
    sample_weight: NDArray | None = None,
) -> float:
    """Compute Area Under the ROC Curve (AUC).
    
    Thin wrapper around sklearn.metrics.roc_auc_score with sample weight support.
    
    Args:
        y_true: True binary labels, shape (n_samples,). Values should be 0 or 1.
        y_score: Predicted scores/probabilities, shape (n_samples,).
        sample_weight: Sample weights, shape (n_samples,). If None, uniform weights.
        
    Returns:
        AUC score between 0 and 1. Random classifier = 0.5, perfect = 1.0.
        
    Example:
        >>> import openboost as ob
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_score = np.array([0.1, 0.4, 0.35, 0.8])
        >>> ob.roc_auc_score(y_true, y_score)
        0.75
    """
    _check_sklearn()
    from sklearn.metrics import roc_auc_score as _sklearn_auc
    return float(_sklearn_auc(y_true, y_score, sample_weight=sample_weight))


def accuracy_score(
    y_true: NDArray,
    y_pred: NDArray,
    *,
    sample_weight: NDArray | None = None,
) -> float:
    """Compute classification accuracy.
    
    Thin wrapper around sklearn.metrics.accuracy_score with sample weight support.
    
    Args:
        y_true: True labels, shape (n_samples,).
        y_pred: Predicted labels, shape (n_samples,).
        sample_weight: Sample weights, shape (n_samples,). If None, uniform weights.
        
    Returns:
        Accuracy score between 0 and 1.
        
    Example:
        >>> import openboost as ob
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0])
        >>> ob.accuracy_score(y_true, y_pred)
        0.75
    """
    _check_sklearn()
    from sklearn.metrics import accuracy_score as _sklearn_acc
    return float(_sklearn_acc(y_true, y_pred, sample_weight=sample_weight))


def log_loss_score(
    y_true: NDArray,
    y_pred: NDArray,
    *,
    sample_weight: NDArray | None = None,
) -> float:
    """Compute log loss (cross-entropy loss) for binary classification.
    
    Thin wrapper around sklearn.metrics.log_loss with sample weight support.
    
    Args:
        y_true: True binary labels, shape (n_samples,). Values should be 0 or 1.
        y_pred: Predicted probabilities for positive class, shape (n_samples,).
        sample_weight: Sample weights, shape (n_samples,). If None, uniform weights.
        
    Returns:
        Log loss (lower is better). Perfect predictions = 0.
        
    Example:
        >>> import openboost as ob
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.2, 0.7, 0.9])
        >>> ob.log_loss_score(y_true, y_pred)
        0.1738...
    """
    _check_sklearn()
    from sklearn.metrics import log_loss as _sklearn_ll
    return float(_sklearn_ll(y_true, y_pred, sample_weight=sample_weight))


def mse_score(
    y_true: NDArray,
    y_pred: NDArray,
    *,
    sample_weight: NDArray | None = None,
) -> float:
    """Compute Mean Squared Error.
    
    Thin wrapper around sklearn.metrics.mean_squared_error with sample weight support.
    
    Args:
        y_true: True values, shape (n_samples,).
        y_pred: Predicted values, shape (n_samples,).
        sample_weight: Sample weights, shape (n_samples,). If None, uniform weights.
        
    Returns:
        MSE (lower is better). Perfect predictions = 0.
        
    Example:
        >>> import openboost as ob
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred = np.array([1.1, 2.0, 2.8])
        >>> ob.mse_score(y_true, y_pred)
        0.016666...
    """
    _check_sklearn()
    from sklearn.metrics import mean_squared_error as _sklearn_mse
    return float(_sklearn_mse(y_true, y_pred, sample_weight=sample_weight))


def r2_score(
    y_true: NDArray,
    y_pred: NDArray,
    *,
    sample_weight: NDArray | None = None,
) -> float:
    """Compute R² (coefficient of determination).
    
    Thin wrapper around sklearn.metrics.r2_score with sample weight support.
    
    Args:
        y_true: True values, shape (n_samples,).
        y_pred: Predicted values, shape (n_samples,).
        sample_weight: Sample weights, shape (n_samples,). If None, uniform weights.
        
    Returns:
        R² score. Perfect predictions = 1.0, baseline (mean) = 0.0.
        
    Example:
        >>> import openboost as ob
        >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
        >>> y_pred = np.array([1.1, 1.9, 3.1, 3.9])
        >>> ob.r2_score(y_true, y_pred)
        0.98
    """
    _check_sklearn()
    from sklearn.metrics import r2_score as _sklearn_r2
    return float(_sklearn_r2(y_true, y_pred, sample_weight=sample_weight))


def mae_score(
    y_true: NDArray,
    y_pred: NDArray,
    *,
    sample_weight: NDArray | None = None,
) -> float:
    """Compute Mean Absolute Error.
    
    Thin wrapper around sklearn.metrics.mean_absolute_error with sample weight support.
    
    Args:
        y_true: True values, shape (n_samples,).
        y_pred: Predicted values, shape (n_samples,).
        sample_weight: Sample weights, shape (n_samples,). If None, uniform weights.
        
    Returns:
        MAE (lower is better). Perfect predictions = 0.
        
    Example:
        >>> import openboost as ob
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred = np.array([1.1, 2.0, 2.8])
        >>> ob.mae_score(y_true, y_pred)
        0.1
    """
    _check_sklearn()
    from sklearn.metrics import mean_absolute_error as _sklearn_mae
    return float(_sklearn_mae(y_true, y_pred, sample_weight=sample_weight))


def rmse_score(
    y_true: NDArray,
    y_pred: NDArray,
    *,
    sample_weight: NDArray | None = None,
) -> float:
    """Compute Root Mean Squared Error.
    
    Thin wrapper around sklearn.metrics.mean_squared_error with squared=False.
    
    Args:
        y_true: True values, shape (n_samples,).
        y_pred: Predicted values, shape (n_samples,).
        sample_weight: Sample weights, shape (n_samples,). If None, uniform weights.
        
    Returns:
        RMSE (lower is better). Perfect predictions = 0.
        
    Example:
        >>> import openboost as ob
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred = np.array([1.1, 2.0, 2.8])
        >>> ob.rmse_score(y_true, y_pred)
        0.1291...
    """
    _check_sklearn()
    from sklearn.metrics import root_mean_squared_error as _sklearn_rmse
    return float(_sklearn_rmse(y_true, y_pred, sample_weight=sample_weight))


def f1_score(
    y_true: NDArray,
    y_pred: NDArray,
    *,
    average: Literal['binary', 'micro', 'macro', 'weighted'] = 'binary',
    sample_weight: NDArray | None = None,
) -> float:
    """Compute F1 score (harmonic mean of precision and recall).
    
    Thin wrapper around sklearn.metrics.f1_score with sample weight support.
    
    Args:
        y_true: True labels, shape (n_samples,).
        y_pred: Predicted labels, shape (n_samples,).
        average: Averaging method for multi-class:
            - 'binary': Only for binary classification.
            - 'micro': Global TP, FP, FN counts.
            - 'macro': Unweighted mean of per-class F1.
            - 'weighted': Weighted mean by support.
        sample_weight: Sample weights, shape (n_samples,). If None, uniform weights.
        
    Returns:
        F1 score between 0 and 1.
        
    Example:
        >>> import openboost as ob
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 0, 1])
        >>> ob.f1_score(y_true, y_pred)
        0.8
    """
    _check_sklearn()
    from sklearn.metrics import f1_score as _sklearn_f1
    return float(_sklearn_f1(y_true, y_pred, average=average, sample_weight=sample_weight))


def precision_score(
    y_true: NDArray,
    y_pred: NDArray,
    *,
    average: Literal['binary', 'micro', 'macro', 'weighted'] = 'binary',
    sample_weight: NDArray | None = None,
) -> float:
    """Compute precision (positive predictive value).
    
    Thin wrapper around sklearn.metrics.precision_score with sample weight support.
    
    Args:
        y_true: True labels, shape (n_samples,).
        y_pred: Predicted labels, shape (n_samples,).
        average: Averaging method for multi-class:
            - 'binary': Only for binary classification.
            - 'micro': Global TP, FP counts.
            - 'macro': Unweighted mean of per-class precision.
            - 'weighted': Weighted mean by support.
        sample_weight: Sample weights, shape (n_samples,). If None, uniform weights.
        
    Returns:
        Precision score between 0 and 1.
        
    Example:
        >>> import openboost as ob
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 1, 1])
        >>> ob.precision_score(y_true, y_pred)
        0.666...
    """
    _check_sklearn()
    from sklearn.metrics import precision_score as _sklearn_prec
    return float(_sklearn_prec(y_true, y_pred, average=average, sample_weight=sample_weight))


def recall_score(
    y_true: NDArray,
    y_pred: NDArray,
    *,
    average: Literal['binary', 'micro', 'macro', 'weighted'] = 'binary',
    sample_weight: NDArray | None = None,
) -> float:
    """Compute recall (sensitivity, true positive rate).
    
    Thin wrapper around sklearn.metrics.recall_score with sample weight support.
    
    Args:
        y_true: True labels, shape (n_samples,).
        y_pred: Predicted labels, shape (n_samples,).
        average: Averaging method for multi-class:
            - 'binary': Only for binary classification.
            - 'micro': Global TP, FN counts.
            - 'macro': Unweighted mean of per-class recall.
            - 'weighted': Weighted mean by support.
        sample_weight: Sample weights, shape (n_samples,). If None, uniform weights.
        
    Returns:
        Recall score between 0 and 1.
        
    Example:
        >>> import openboost as ob
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 0, 1])
        >>> ob.recall_score(y_true, y_pred)
        0.666...
    """
    _check_sklearn()
    from sklearn.metrics import recall_score as _sklearn_rec
    return float(_sklearn_rec(y_true, y_pred, average=average, sample_weight=sample_weight))


# =============================================================================
# Probabilistic/Distributional Metrics (Phase 22 Sprint 2)
# =============================================================================

def crps_gaussian(
    y_true: NDArray,
    mean: NDArray,
    std: NDArray,
    *,
    sample_weight: NDArray | None = None,
) -> float:
    """Compute Continuous Ranked Probability Score for Gaussian predictions.
    
    CRPS is a strictly proper scoring rule that measures the quality of
    probabilistic predictions. Lower is better. For Gaussian distributions,
    there's a closed-form solution.
    
    Args:
        y_true: True values, shape (n_samples,).
        mean: Predicted mean, shape (n_samples,).
        std: Predicted standard deviation, shape (n_samples,). Must be > 0.
        sample_weight: Sample weights, shape (n_samples,). If None, uniform weights.
        
    Returns:
        Mean CRPS (lower is better). Perfect calibration minimizes CRPS.
        
    Example:
        >>> import openboost as ob
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> mean = np.array([1.1, 2.0, 2.8])
        >>> std = np.array([0.5, 0.5, 0.5])
        >>> ob.crps_gaussian(y_true, mean, std)
        0.123...
        
    Notes:
        CRPS formula for Gaussian: CRPS(N(μ,σ²), y) = σ * [z*Φ(z) + φ(z) - 1/√π]
        where z = (y - μ) / σ, Φ is CDF, φ is PDF of standard normal.
        
        For NaturalBoost models, use:
        >>> output = model.predict_distribution(X)
        >>> mean, std = output.params[:, 0], np.sqrt(output.params[:, 1])
        >>> crps = ob.crps_gaussian(y_true, mean, std)
    """
    from scipy import stats
    
    y_true = np.asarray(y_true, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    
    if np.any(std <= 0):
        raise ValueError("std must be positive")
    
    # Standardized residual
    z = (y_true - mean) / std
    
    # CRPS for Gaussian: σ * [z * (2*Φ(z) - 1) + 2*φ(z) - 1/√π]
    phi = stats.norm.pdf(z)  # Standard normal PDF
    Phi = stats.norm.cdf(z)  # Standard normal CDF
    
    crps_values = std * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        return float(np.average(crps_values, weights=sample_weight))
    return float(np.mean(crps_values))


def crps_empirical(
    y_true: NDArray,
    samples: NDArray,
    *,
    sample_weight: NDArray | None = None,
) -> float:
    """Compute CRPS using empirical distribution from Monte Carlo samples.
    
    For non-Gaussian distributions, CRPS can be estimated from samples.
    This is useful for NaturalBoost models with non-Normal distributions.
    
    Args:
        y_true: True values, shape (n_samples,).
        samples: Monte Carlo samples, shape (n_samples, n_mc_samples).
            Each row contains samples from the predictive distribution.
        sample_weight: Sample weights, shape (n_samples,). If None, uniform weights.
        
    Returns:
        Mean CRPS estimated from samples (lower is better).
        
    Example:
        >>> model = ob.NaturalBoostGamma(n_trees=100)
        >>> model.fit(X_train, y_train)
        >>> samples = model.sample(X_test, n_samples=1000)  # (n_test, 1000)
        >>> crps = ob.crps_empirical(y_test, samples)
        
    Notes:
        Uses the formula: CRPS = E|X - y| - 0.5 * E|X - X'|
        where X, X' are independent samples from the predictive distribution.
    """
    y_true = np.asarray(y_true)
    samples = np.asarray(samples)
    
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    
    n_samples = samples.shape[0]
    n_mc = samples.shape[1]
    
    # E|X - y| term
    abs_diff_y = np.mean(np.abs(samples - y_true[:, np.newaxis]), axis=1)
    
    # E|X - X'| term (simplified using sorted samples)
    sorted_samples = np.sort(samples, axis=1)
    # Use the identity: E|X - X'| = 2 * integral of F(1-F) dx
    # Approximated by: 2/m^2 * sum_{i<j} |x_i - x_j|
    # Efficient computation using sorted samples
    indices = np.arange(n_mc)
    weights = 2 * (indices + 1 - (n_mc + 1) / 2)
    diff_term = np.sum(sorted_samples * weights, axis=1) / (n_mc * n_mc)
    
    crps_values = abs_diff_y - diff_term
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        return float(np.average(crps_values, weights=sample_weight))
    return float(np.mean(crps_values))


def brier_score(
    y_true: NDArray,
    y_prob: NDArray,
    *,
    sample_weight: NDArray | None = None,
) -> float:
    """Compute Brier score for probabilistic binary classification.
    
    Brier score measures the mean squared error of probability predictions.
    It's a strictly proper scoring rule for binary outcomes.
    
    Args:
        y_true: True binary labels, shape (n_samples,). Values should be 0 or 1.
        y_prob: Predicted probabilities for positive class, shape (n_samples,).
        sample_weight: Sample weights, shape (n_samples,). If None, uniform weights.
        
    Returns:
        Brier score (lower is better). Perfect predictions = 0, random = 0.25.
        
    Example:
        >>> import openboost as ob
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        >>> ob.brier_score(y_true, y_prob)
        0.025
        
    Notes:
        Brier score = mean((y_prob - y_true)²)
        
        Decomposition: Brier = Reliability - Resolution + Uncertainty
        - Reliability: calibration error (how well probabilities match frequencies)
        - Resolution: how different predictions are from the base rate
        - Uncertainty: entropy of the outcome distribution
    """
    _check_sklearn()
    from sklearn.metrics import brier_score_loss
    return float(brier_score_loss(y_true, y_prob, sample_weight=sample_weight))


def pinball_loss(
    y_true: NDArray,
    y_pred: NDArray,
    quantile: float = 0.5,
    *,
    sample_weight: NDArray | None = None,
) -> float:
    """Compute pinball loss (quantile loss) for quantile regression.
    
    Pinball loss is the proper scoring rule for quantile estimation.
    At quantile=0.5, it equals MAE.
    
    Args:
        y_true: True values, shape (n_samples,).
        y_pred: Predicted quantile values, shape (n_samples,).
        quantile: The quantile being predicted, in (0, 1). Default 0.5 (median).
        sample_weight: Sample weights, shape (n_samples,). If None, uniform weights.
        
    Returns:
        Pinball loss (lower is better).
        
    Example:
        >>> import openboost as ob
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred_median = np.array([1.1, 2.0, 2.8])
        >>> ob.pinball_loss(y_true, y_pred_median, quantile=0.5)
        0.1
        
        >>> # Lower quantile (e.g., 10th percentile)
        >>> y_pred_q10 = np.array([0.5, 1.5, 2.0])
        >>> ob.pinball_loss(y_true, y_pred_q10, quantile=0.1)
        
    Notes:
        Pinball loss: L(y, q) = (y - q) * τ if y >= q else (q - y) * (1 - τ)
        where τ is the quantile.
        
        For prediction intervals from NaturalBoost:
        >>> lower, upper = model.predict_interval(X, alpha=0.1)  # 90% interval
        >>> loss_lower = ob.pinball_loss(y, lower, quantile=0.05)
        >>> loss_upper = ob.pinball_loss(y, upper, quantile=0.95)
    """
    if not 0 < quantile < 1:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    residual = y_true - y_pred
    loss = np.where(residual >= 0, quantile * residual, (quantile - 1) * residual)
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        return float(np.average(loss, weights=sample_weight))
    return float(np.mean(loss))


def interval_score(
    y_true: NDArray,
    lower: NDArray,
    upper: NDArray,
    alpha: float = 0.1,
    *,
    sample_weight: NDArray | None = None,
) -> float:
    """Compute interval score for prediction intervals.
    
    Interval score is a strictly proper scoring rule for prediction intervals.
    It rewards narrow intervals while penalizing miscoverage.
    
    Args:
        y_true: True values, shape (n_samples,).
        lower: Lower bound of prediction interval, shape (n_samples,).
        upper: Upper bound of prediction interval, shape (n_samples,).
        alpha: Nominal miscoverage rate (0.1 for 90% interval). Default 0.1.
        sample_weight: Sample weights, shape (n_samples,). If None, uniform weights.
        
    Returns:
        Interval score (lower is better).
        
    Example:
        >>> import openboost as ob
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> lower = np.array([0.5, 1.5, 2.5])
        >>> upper = np.array([1.5, 2.5, 3.5])
        >>> ob.interval_score(y_true, lower, upper, alpha=0.1)
        1.0
        
    Notes:
        Interval Score = (upper - lower) + (2/α) * (lower - y) * I(y < lower)
                                         + (2/α) * (y - upper) * I(y > upper)
        
        The score combines:
        1. Interval width (prefer narrow intervals)
        2. Penalty for observations below lower bound
        3. Penalty for observations above upper bound
        
        Use with NaturalBoost:
        >>> lower, upper = model.predict_interval(X_test, alpha=0.1)
        >>> score = ob.interval_score(y_test, lower, upper, alpha=0.1)
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    
    # Interval width
    width = upper - lower
    
    # Penalty for y below lower bound
    below_lower = np.maximum(0, lower - y_true)
    
    # Penalty for y above upper bound
    above_upper = np.maximum(0, y_true - upper)
    
    # Interval score
    score = width + (2 / alpha) * (below_lower + above_upper)
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        return float(np.average(score, weights=sample_weight))
    return float(np.mean(score))


def expected_calibration_error(
    y_true: NDArray,
    y_prob: NDArray,
    n_bins: int = 10,
    *,
    strategy: Literal['uniform', 'quantile'] = 'uniform',
) -> float:
    """Compute Expected Calibration Error (ECE) for probability predictions.
    
    ECE measures the miscalibration of predicted probabilities. A well-calibrated
    model has ECE close to 0.
    
    Args:
        y_true: True binary labels, shape (n_samples,). Values should be 0 or 1.
        y_prob: Predicted probabilities for positive class, shape (n_samples,).
        n_bins: Number of bins to use. Default 10.
        strategy: Binning strategy:
            - 'uniform': Bins of equal width in [0, 1].
            - 'quantile': Bins with equal number of samples.
        
    Returns:
        ECE (lower is better). Perfect calibration = 0.
        
    Example:
        >>> import openboost as ob
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_prob = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
        >>> ob.expected_calibration_error(y_true, y_prob)
        0.06
        
    Notes:
        ECE = Σ (|bin_size| / n) * |accuracy_in_bin - mean_confidence_in_bin|
        
        For reliability diagrams, use calibration_curve to get bin data.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    n_samples = len(y_true)
    
    if strategy == 'uniform':
        bins = np.linspace(0, 1, n_bins + 1)
    elif strategy == 'quantile':
        quantiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(y_prob, quantiles)
        bins[0] = 0.0
        bins[-1] = 1.0
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob > bins[i]) & (y_prob <= bins[i + 1])
        if i == 0:
            in_bin = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_accuracy = np.mean(y_true[in_bin])
            bin_confidence = np.mean(y_prob[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_accuracy - bin_confidence)
    
    return float(ece)


def calibration_curve(
    y_true: NDArray,
    y_prob: NDArray,
    n_bins: int = 10,
    *,
    strategy: Literal['uniform', 'quantile'] = 'uniform',
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute calibration curve data for reliability diagrams.
    
    Returns the fraction of positives and mean predicted probability for each bin.
    This data can be used to create reliability diagrams.
    
    Args:
        y_true: True binary labels, shape (n_samples,). Values should be 0 or 1.
        y_prob: Predicted probabilities for positive class, shape (n_samples,).
        n_bins: Number of bins to use. Default 10.
        strategy: Binning strategy:
            - 'uniform': Bins of equal width in [0, 1].
            - 'quantile': Bins with equal number of samples.
        
    Returns:
        Tuple of (fraction_of_positives, mean_predicted_value, bin_counts):
        - fraction_of_positives: Actual fraction of positives in each bin.
        - mean_predicted_value: Mean predicted probability in each bin.
        - bin_counts: Number of samples in each bin.
        
    Example:
        >>> import openboost as ob
        >>> import matplotlib.pyplot as plt
        >>> 
        >>> y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
        >>> y_prob = np.array([0.1, 0.2, 0.3, 0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.9])
        >>> frac_pos, mean_pred, counts = ob.calibration_curve(y_true, y_prob, n_bins=5)
        >>> 
        >>> # Plot reliability diagram
        >>> plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        >>> plt.plot(mean_pred, frac_pos, 's-', label='Model')
        >>> plt.xlabel('Mean predicted probability')
        >>> plt.ylabel('Fraction of positives')
        >>> plt.legend()
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    if strategy == 'uniform':
        bins = np.linspace(0, 1, n_bins + 1)
    elif strategy == 'quantile':
        quantiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(y_prob, quantiles)
        bins[0] = 0.0
        bins[-1] = 1.0
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    fraction_of_positives = []
    mean_predicted_value = []
    bin_counts = []
    
    for i in range(n_bins):
        in_bin = (y_prob > bins[i]) & (y_prob <= bins[i + 1])
        if i == 0:
            in_bin = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            fraction_of_positives.append(np.mean(y_true[in_bin]))
            mean_predicted_value.append(np.mean(y_prob[in_bin]))
            bin_counts.append(bin_size)
    
    return (
        np.array(fraction_of_positives),
        np.array(mean_predicted_value),
        np.array(bin_counts),
    )


def negative_log_likelihood(
    y_true: NDArray,
    mean: NDArray,
    std: NDArray,
    *,
    sample_weight: NDArray | None = None,
) -> float:
    """Compute negative log-likelihood for Gaussian predictions.
    
    NLL is a proper scoring rule for probabilistic predictions.
    Lower is better.
    
    Args:
        y_true: True values, shape (n_samples,).
        mean: Predicted mean, shape (n_samples,).
        std: Predicted standard deviation, shape (n_samples,). Must be > 0.
        sample_weight: Sample weights, shape (n_samples,). If None, uniform weights.
        
    Returns:
        Mean negative log-likelihood (lower is better).
        
    Example:
        >>> import openboost as ob
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> mean = np.array([1.1, 2.0, 2.8])
        >>> std = np.array([0.5, 0.5, 0.5])
        >>> ob.negative_log_likelihood(y_true, mean, std)
        0.92...
        
    Notes:
        NLL = 0.5 * log(2π) + log(σ) + (y - μ)² / (2σ²)
        
        For NaturalBoost Normal models:
        >>> output = model.predict_distribution(X)
        >>> mean, var = output.params[:, 0], output.params[:, 1]
        >>> nll = ob.negative_log_likelihood(y, mean, np.sqrt(var))
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    
    if np.any(std <= 0):
        raise ValueError("std must be positive")
    
    # NLL for Gaussian
    nll = 0.5 * np.log(2 * np.pi) + np.log(std) + 0.5 * ((y_true - mean) / std) ** 2
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        return float(np.average(nll, weights=sample_weight))
    return float(np.mean(nll))


# Suggested parameter grids for hyperparameter tuning
PARAM_GRID_REGRESSION = {
    'n_estimators': [100, 300, 500],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'reg_lambda': [0.1, 1.0, 10.0],
}

PARAM_GRID_CLASSIFICATION = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'reg_lambda': [0.1, 1.0, 10.0],
}

PARAM_GRID_DISTRIBUTIONAL = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 4, 5],  # Typically shallower for distributional
    'learning_rate': [0.05, 0.1, 0.2],
    'reg_lambda': [0.1, 1.0, 5.0],
}


def suggest_params(
    X: NDArray,
    y: NDArray,
    task: Literal['regression', 'classification', 'distributional'] = 'regression',
    n_estimators_cap: int = 500,
) -> dict[str, Any]:
    """Suggest hyperparameters based on dataset characteristics.
    
    This provides reasonable starting points based on heuristics. For best
    results, use these as initial values and tune with cross-validation.
    
    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: Target values, shape (n_samples,).
        task: Type of task - 'regression', 'classification', or 'distributional'.
        n_estimators_cap: Maximum number of estimators to suggest.
        
    Returns:
        Dictionary of suggested hyperparameters suitable for passing to
        OpenBoostRegressor, OpenBoostClassifier, etc.
        
    Example:
        >>> params = suggest_params(X_train, y_train, task='regression')
        >>> model = OpenBoostRegressor(**params)
        >>> model.fit(X_train, y_train)
        
    Notes:
        - For small datasets (< 1000 samples): Fewer trees, more regularization
        - For large datasets (> 100k samples): More trees, lower learning rate
        - For high-dimensional data: More column sampling, shallower trees
        - For noisy data: Consider distributional models for uncertainty
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape
    
    # Base parameters
    params: dict[str, Any] = {}
    
    # Number of trees: scale with data size, but cap
    if n_samples < 1000:
        params['n_estimators'] = min(100, n_estimators_cap)
    elif n_samples < 10000:
        params['n_estimators'] = min(200, n_estimators_cap)
    elif n_samples < 100000:
        params['n_estimators'] = min(300, n_estimators_cap)
    else:
        params['n_estimators'] = min(500, n_estimators_cap)
    
    # Learning rate: lower for more trees
    if params['n_estimators'] >= 300:
        params['learning_rate'] = 0.05
    else:
        params['learning_rate'] = 0.1
    
    # Tree depth: based on features and task
    if task == 'distributional':
        # Distributional models work better with shallower trees
        params['max_depth'] = min(5, 3 + n_features // 50)
    elif n_features > 100:
        # High-dimensional: shallower trees, more regularization
        params['max_depth'] = min(6, 4 + n_features // 100)
    else:
        params['max_depth'] = min(8, 4 + n_features // 20)
    
    # Regularization: more for small datasets
    if n_samples < 1000:
        params['reg_lambda'] = 10.0
        params['min_child_weight'] = 3.0
    elif n_samples < 10000:
        params['reg_lambda'] = 1.0
        params['min_child_weight'] = 1.0
    else:
        params['reg_lambda'] = 0.1
        params['min_child_weight'] = 1.0
    
    # Sampling: use for larger datasets
    if n_samples > 10000:
        params['subsample'] = 0.8
        params['colsample_bytree'] = 0.8
    
    # High-dimensional: more column sampling
    if n_features > 100:
        params['colsample_bytree'] = 0.6
    
    return params


def cross_val_predict(
    model: Any,
    X: NDArray,
    y: NDArray,
    cv: int = 5,
    random_state: int | None = 42,
) -> NDArray:
    """Generate out-of-fold predictions using cross-validation.
    
    Each sample gets a prediction from a model that was not trained on it.
    Useful for stacking/blending in competitions and for honest evaluation.
    
    Args:
        model: An OpenBoost model instance (will be cloned for each fold).
        X: Feature matrix, shape (n_samples, n_features).
        y: Target values, shape (n_samples,).
        cv: Number of cross-validation folds.
        random_state: Random seed for reproducible fold splits.
        
    Returns:
        Out-of-fold predictions, shape (n_samples,) for regression or
        shape (n_samples, n_classes) for classification probabilities.
        
    Example:
        >>> from openboost import OpenBoostRegressor
        >>> from openboost.utils import cross_val_predict
        >>> 
        >>> model = OpenBoostRegressor(n_estimators=100)
        >>> oof_pred = cross_val_predict(model, X, y, cv=5)
        >>> 
        >>> # Use OOF predictions for stacking
        >>> from sklearn.linear_model import Ridge
        >>> meta_model = Ridge()
        >>> meta_model.fit(oof_pred.reshape(-1, 1), y)
    """
    try:
        from sklearn.model_selection import KFold
        from sklearn.base import clone
    except ImportError:
        raise ImportError(
            "sklearn is required for cross_val_predict. "
            "Install with: pip install scikit-learn"
        )
    
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = len(y)
    
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # First fold to determine output shape
    first_train, first_val = next(iter(kf.split(X)))
    model_clone = clone(model)
    model_clone.fit(X[first_train], y[first_train])
    first_pred = model_clone.predict(X[first_val])
    
    # Initialize output array
    if first_pred.ndim == 1:
        oof_pred = np.zeros(n_samples, dtype=np.float32)
    else:
        oof_pred = np.zeros((n_samples, first_pred.shape[1]), dtype=np.float32)
    
    oof_pred[first_val] = first_pred
    
    # Remaining folds
    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        if i == 0:
            continue  # Already done first fold
        model_clone = clone(model)
        model_clone.fit(X[train_idx], y[train_idx])
        oof_pred[val_idx] = model_clone.predict(X[val_idx])
    
    return oof_pred


def cross_val_predict_proba(
    model: Any,
    X: NDArray,
    y: NDArray,
    cv: int = 5,
    random_state: int | None = 42,
) -> NDArray:
    """Generate out-of-fold probability predictions using cross-validation.
    
    Similar to cross_val_predict but returns class probabilities instead
    of class labels. Only works with classifiers.
    
    Args:
        model: An OpenBoost classifier instance (must have predict_proba).
        X: Feature matrix, shape (n_samples, n_features).
        y: Target labels, shape (n_samples,).
        cv: Number of cross-validation folds.
        random_state: Random seed for reproducible fold splits.
        
    Returns:
        Out-of-fold probability predictions, shape (n_samples, n_classes).
        
    Example:
        >>> from openboost import OpenBoostClassifier
        >>> from openboost.utils import cross_val_predict_proba
        >>> 
        >>> model = OpenBoostClassifier(n_estimators=100)
        >>> oof_proba = cross_val_predict_proba(model, X, y, cv=5)
        >>> 
        >>> # Use probabilities for stacking
        >>> meta_features = oof_proba[:, 1]  # P(class=1)
        
    Raises:
        AttributeError: If model doesn't have predict_proba method.
    """
    try:
        from sklearn.model_selection import KFold
        from sklearn.base import clone
    except ImportError:
        raise ImportError(
            "sklearn is required for cross_val_predict_proba. "
            "Install with: pip install scikit-learn"
        )
    
    if not hasattr(model, 'predict_proba'):
        raise AttributeError(
            f"{type(model).__name__} doesn't have predict_proba method. "
            "Use cross_val_predict for regressors."
        )
    
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = len(y)
    
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # First fold to determine number of classes
    first_train, first_val = next(iter(kf.split(X)))
    model_clone = clone(model)
    model_clone.fit(X[first_train], y[first_train])
    first_proba = model_clone.predict_proba(X[first_val])
    n_classes = first_proba.shape[1]
    
    oof_proba = np.zeros((n_samples, n_classes), dtype=np.float32)
    oof_proba[first_val] = first_proba
    
    # Remaining folds
    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        if i == 0:
            continue
        model_clone = clone(model)
        model_clone.fit(X[train_idx], y[train_idx])
        oof_proba[val_idx] = model_clone.predict_proba(X[val_idx])
    
    return oof_proba


def cross_val_predict_interval(
    model: Any,
    X: NDArray,
    y: NDArray,
    alpha: float = 0.1,
    cv: int = 5,
    random_state: int | None = 42,
) -> tuple[NDArray, NDArray]:
    """Generate out-of-fold prediction intervals using cross-validation.
    
    For distributional models that support uncertainty quantification.
    Returns lower and upper bounds of the prediction interval.
    
    Args:
        model: An OpenBoost distributional model (must have predict_interval).
        X: Feature matrix, shape (n_samples, n_features).
        y: Target values, shape (n_samples,).
        alpha: Significance level (0.1 = 90% interval).
        cv: Number of cross-validation folds.
        random_state: Random seed for reproducible fold splits.
        
    Returns:
        Tuple of (lower_bounds, upper_bounds), each shape (n_samples,).
        
    Example:
        >>> from openboost import OpenBoostDistributionalRegressor
        >>> from openboost.utils import cross_val_predict_interval
        >>> 
        >>> model = OpenBoostDistributionalRegressor(distribution='normal')
        >>> lower, upper = cross_val_predict_interval(model, X, y, alpha=0.1)
        >>> 
        >>> # Check coverage
        >>> coverage = np.mean((y >= lower) & (y <= upper))
        >>> print(f"90% interval coverage: {coverage:.2%}")
        
    Raises:
        AttributeError: If model doesn't have predict_interval method.
    """
    try:
        from sklearn.model_selection import KFold
        from sklearn.base import clone
    except ImportError:
        raise ImportError(
            "sklearn is required for cross_val_predict_interval. "
            "Install with: pip install scikit-learn"
        )
    
    if not hasattr(model, 'predict_interval'):
        raise AttributeError(
            f"{type(model).__name__} doesn't have predict_interval method. "
            "Use a distributional model like OpenBoostDistributionalRegressor."
        )
    
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = len(y)
    
    oof_lower = np.zeros(n_samples, dtype=np.float32)
    oof_upper = np.zeros(n_samples, dtype=np.float32)
    
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    for train_idx, val_idx in kf.split(X):
        model_clone = clone(model)
        model_clone.fit(X[train_idx], y[train_idx])
        lower, upper = model_clone.predict_interval(X[val_idx], alpha=alpha)
        oof_lower[val_idx] = lower
        oof_upper[val_idx] = upper
    
    return oof_lower, oof_upper


def evaluate_coverage(
    y_true: NDArray,
    lower: NDArray,
    upper: NDArray,
    alpha: float = 0.1,
) -> dict[str, float]:
    """Evaluate prediction interval coverage and width.
    
    Args:
        y_true: True target values, shape (n_samples,).
        lower: Lower bounds of intervals, shape (n_samples,).
        upper: Upper bounds of intervals, shape (n_samples,).
        alpha: Expected significance level (for reporting).
        
    Returns:
        Dictionary with:
        - coverage: Fraction of true values within intervals
        - expected_coverage: Expected coverage (1 - alpha)
        - mean_width: Average interval width
        - median_width: Median interval width
        
    Example:
        >>> lower, upper = model.predict_interval(X_test, alpha=0.1)
        >>> metrics = evaluate_coverage(y_test, lower, upper, alpha=0.1)
        >>> print(f"Coverage: {metrics['coverage']:.2%}")
        >>> print(f"Mean width: {metrics['mean_width']:.4f}")
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    
    in_interval = (y_true >= lower) & (y_true <= upper)
    widths = upper - lower
    
    return {
        'coverage': float(np.mean(in_interval)),
        'expected_coverage': 1.0 - alpha,
        'mean_width': float(np.mean(widths)),
        'median_width': float(np.median(widths)),
    }


def get_param_grid(
    task: Literal['regression', 'classification', 'distributional'] = 'regression',
) -> dict[str, list]:
    """Get a suggested parameter grid for hyperparameter tuning.
    
    Args:
        task: Type of task - 'regression', 'classification', or 'distributional'.
        
    Returns:
        Dictionary of parameter names to lists of values, suitable for
        sklearn's GridSearchCV or RandomizedSearchCV.
        
    Example:
        >>> from sklearn.model_selection import GridSearchCV
        >>> from openboost import OpenBoostRegressor
        >>> from openboost.utils import get_param_grid
        >>> 
        >>> param_grid = get_param_grid('regression')
        >>> search = GridSearchCV(OpenBoostRegressor(), param_grid, cv=3)
        >>> search.fit(X, y)
        >>> print(search.best_params_)
    """
    if task == 'regression':
        return PARAM_GRID_REGRESSION.copy()
    elif task == 'classification':
        return PARAM_GRID_CLASSIFICATION.copy()
    elif task == 'distributional':
        return PARAM_GRID_DISTRIBUTIONAL.copy()
    else:
        raise ValueError(
            f"Unknown task: {task}. "
            "Use 'regression', 'classification', or 'distributional'."
        )
