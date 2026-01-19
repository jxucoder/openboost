"""Split finding for gradient boosting trees.

Phase 14: Added missing value handling - learns optimal direction for NaN values.
Phase 14.3: Added native categorical feature support with Fisher-based splits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from .._backends import is_cuda
from .._array import MISSING_BIN

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SplitInfo(NamedTuple):
    """Information about a split.
    
    For ordinal (numeric) splits:
        - threshold: bin index, samples with bin <= threshold go left
        
    For categorical splits (Phase 14.3):
        - is_categorical: True
        - cat_bitset: uint64 bitmask, bit i=1 means category i goes left
    """
    feature: int      # Feature index (-1 if no valid split)
    threshold: int    # Bin threshold (ordinal) or split_point (categorical)
    gain: float       # Split gain
    missing_go_left: bool = True  # Direction for missing values (Phase 14)
    is_categorical: bool = False  # Phase 14.3: True if categorical split
    cat_bitset: int = 0           # Phase 14.3: Bitmask for categories going left
    cat_threshold: int = -1       # Phase 14.3: Split point in sorted category order
    
    @property
    def is_valid(self) -> bool:
        """Check if this is a valid split."""
        return self.feature >= 0 and self.gain > 0


def find_best_split(
    hist_grad: NDArray,
    hist_hess: NDArray,
    total_grad: float | None = None,
    total_hess: float | None = None,
    *,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    min_gain: float = 0.0,
) -> SplitInfo:
    """Find the best split across all features.
    
    Args:
        hist_grad: Gradient histogram, shape (n_features, 256)
        hist_hess: Hessian histogram, shape (n_features, 256)
        total_grad: Sum of gradients (computed if None)
        total_hess: Sum of hessians (computed if None)
        reg_lambda: L2 regularization term
        min_child_weight: Minimum sum of hessian in each child
        min_gain: Minimum gain to make a split
        
    Returns:
        SplitInfo with best feature, threshold, and gain
    """
    # Compute totals if not provided
    if total_grad is None:
        total_grad = float(_sum_histogram(hist_grad))
    if total_hess is None:
        total_hess = float(_sum_histogram(hist_hess))
    
    # Dispatch to backend
    if is_cuda() and hasattr(hist_grad, '__cuda_array_interface__'):
        from .._backends._cuda import find_best_split_cuda
        feature, threshold, gain = find_best_split_cuda(
            hist_grad, hist_hess,
            total_grad, total_hess,
            reg_lambda, min_child_weight,
        )
    else:
        from .._backends._cpu import find_best_split_cpu
        # Ensure numpy for CPU
        hist_grad_np = np.asarray(hist_grad.copy_to_host() if hasattr(hist_grad, 'copy_to_host') else hist_grad)
        hist_hess_np = np.asarray(hist_hess.copy_to_host() if hasattr(hist_hess, 'copy_to_host') else hist_hess)
        feature, threshold, gain = find_best_split_cpu(
            hist_grad_np, hist_hess_np,
            total_grad, total_hess,
            reg_lambda, min_child_weight,
        )
    
    # Apply minimum gain threshold
    if gain < min_gain:
        return SplitInfo(feature=-1, threshold=-1, gain=0.0)
    
    return SplitInfo(feature=feature, threshold=threshold, gain=gain)


def compute_leaf_value(
    sum_grad: float,
    sum_hess: float,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
) -> float:
    """Compute optimal leaf value with L1/L2 regularization.
    
    Without L1 (reg_alpha=0):
        leaf_value = -sum_grad / (sum_hess + lambda)
    
    With L1 (reg_alpha > 0), uses soft-thresholding:
        if |sum_grad| <= reg_alpha: return 0
        else: return -(sum_grad - sign(sum_grad)*reg_alpha) / (sum_hess + lambda)
    
    Args:
        sum_grad: Sum of gradients in the leaf
        sum_hess: Sum of hessians in the leaf
        reg_lambda: L2 regularization
        reg_alpha: L1 regularization (Phase 11)
        
    Returns:
        Optimal leaf value
    """
    # L1 soft-thresholding
    if reg_alpha > 0.0:
        if abs(sum_grad) <= reg_alpha:
            return 0.0
        elif sum_grad > 0:
            return -(sum_grad - reg_alpha) / (sum_hess + reg_lambda)
        else:
            return -(sum_grad + reg_alpha) / (sum_hess + reg_lambda)
    else:
        return -sum_grad / (sum_hess + reg_lambda)


def _sum_histogram(hist: NDArray) -> float:
    """Sum all values in a histogram."""
    if hasattr(hist, 'copy_to_host'):
        hist = hist.copy_to_host()
    return float(np.sum(hist))


def find_best_split_with_missing(
    hist_grad: NDArray,
    hist_hess: NDArray,
    total_grad: float | None = None,
    total_hess: float | None = None,
    *,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    min_gain: float = 0.0,
    has_missing: NDArray | None = None,
) -> SplitInfo:
    """Find the best split considering missing values.
    
    For each split candidate, tries both directions for missing values:
    1. Missing goes LEFT
    2. Missing goes RIGHT
    
    Picks whichever gives higher gain.
    
    Args:
        hist_grad: Gradient histogram, shape (n_features, 256)
                   Bin 255 contains stats for missing values
        hist_hess: Hessian histogram, shape (n_features, 256)
        total_grad: Sum of gradients (computed if None)
        total_hess: Sum of hessians (computed if None)
        reg_lambda: L2 regularization term
        min_child_weight: Minimum sum of hessian in each child
        min_gain: Minimum gain to make a split
        has_missing: Boolean array (n_features,) indicating which features have missing
        
    Returns:
        SplitInfo with best feature, threshold, gain, and missing direction
    """
    # Compute totals if not provided
    if total_grad is None:
        total_grad = float(_sum_histogram(hist_grad))
    if total_hess is None:
        total_hess = float(_sum_histogram(hist_hess))
    
    # Check if any feature has missing values
    n_features = hist_grad.shape[0]
    any_missing = has_missing is not None and np.any(has_missing)
    
    # If no missing values, use standard split finding
    if not any_missing:
        split = find_best_split(
            hist_grad, hist_hess, total_grad, total_hess,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            min_gain=min_gain,
        )
        return SplitInfo(split.feature, split.threshold, split.gain, True)
    
    # Dispatch to backend
    if is_cuda() and hasattr(hist_grad, '__cuda_array_interface__'):
        # Phase 14.2: Use GPU kernel for missing-aware split finding
        from .._backends._cuda import find_best_split_with_missing_cuda
        feature, threshold, gain, missing_go_left = find_best_split_with_missing_cuda(
            hist_grad, hist_hess,
            total_grad, total_hess,
            reg_lambda, min_child_weight,
            has_missing,
        )
    else:
        hist_grad_np = np.asarray(hist_grad.copy_to_host() if hasattr(hist_grad, 'copy_to_host') else hist_grad)
        hist_hess_np = np.asarray(hist_hess.copy_to_host() if hasattr(hist_hess, 'copy_to_host') else hist_hess)
        
        from .._backends._cpu import find_best_split_with_missing_cpu
        feature, threshold, gain, missing_go_left = find_best_split_with_missing_cpu(
            hist_grad_np, hist_hess_np,
            total_grad, total_hess,
            reg_lambda, min_child_weight,
            has_missing,
        )
    
    # Apply minimum gain threshold
    if gain < min_gain:
        return SplitInfo(feature=-1, threshold=-1, gain=0.0, missing_go_left=True)
    
    return SplitInfo(feature=feature, threshold=threshold, gain=gain, missing_go_left=missing_go_left)


# =============================================================================
# Phase 14.3: Categorical Split Finding
# =============================================================================

def find_best_split_with_categorical(
    hist_grad: NDArray,
    hist_hess: NDArray,
    total_grad: float | None = None,
    total_hess: float | None = None,
    *,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    min_gain: float = 0.0,
    has_missing: NDArray | None = None,
    is_categorical: NDArray | None = None,
    n_categories: NDArray | None = None,
) -> SplitInfo:
    """Find the best split considering both categorical and missing values.
    
    For numeric features: ordinal splits (value <= threshold)
    For categorical features: set membership splits via Fisher's optimal ordering
    
    Args:
        hist_grad: Gradient histogram, shape (n_features, 256)
        hist_hess: Hessian histogram, shape (n_features, 256)
        total_grad: Sum of gradients (computed if None)
        total_hess: Sum of hessians (computed if None)
        reg_lambda: L2 regularization term
        min_child_weight: Minimum sum of hessian in each child
        min_gain: Minimum gain to make a split
        has_missing: Boolean array (n_features,) for features with NaN
        is_categorical: Boolean array (n_features,) for categorical features
        n_categories: Number of categories per feature (0 for numeric)
        
    Returns:
        SplitInfo with best feature, threshold/bitset, gain, etc.
    """
    # Compute totals if not provided
    if total_grad is None:
        total_grad = float(_sum_histogram(hist_grad))
    if total_hess is None:
        total_hess = float(_sum_histogram(hist_hess))
    
    n_features = hist_grad.shape[0]
    
    # Check if we have categorical features
    any_categorical = is_categorical is not None and np.any(is_categorical)
    any_missing = has_missing is not None and np.any(has_missing)
    
    # If no categorical and no missing, use standard split
    if not any_categorical and not any_missing:
        return find_best_split(
            hist_grad, hist_hess, total_grad, total_hess,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            min_gain=min_gain,
        )
    
    # If only missing (no categorical), use missing-aware split
    if not any_categorical:
        return find_best_split_with_missing(
            hist_grad, hist_hess, total_grad, total_hess,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            min_gain=min_gain,
            has_missing=has_missing,
        )
    
    # Handle mixed categorical and numeric features
    # Dispatch to GPU or CPU backend
    if is_cuda() and hasattr(hist_grad, '__cuda_array_interface__'):
        # Phase 14.4: Use GPU kernel for categorical split finding
        from .._backends._cuda import find_best_split_categorical_cuda
        feature, threshold, gain, missing_left, is_cat, cat_bitset, cat_threshold = find_best_split_categorical_cuda(
            hist_grad, hist_hess,
            total_grad, total_hess,
            reg_lambda, min_child_weight,
            is_categorical if is_categorical is not None else np.zeros(n_features, dtype=np.bool_),
            n_categories if n_categories is not None else np.zeros(n_features, dtype=np.int32),
            has_missing if has_missing is not None else np.zeros(n_features, dtype=np.bool_),
        )
    else:
        hist_grad_np = np.asarray(hist_grad.copy_to_host() if hasattr(hist_grad, 'copy_to_host') else hist_grad)
        hist_hess_np = np.asarray(hist_hess.copy_to_host() if hasattr(hist_hess, 'copy_to_host') else hist_hess)
        
        from .._backends._cpu import find_best_split_categorical_cpu
        feature, threshold, gain, missing_left, is_cat, cat_bitset, cat_threshold = find_best_split_categorical_cpu(
            hist_grad_np, hist_hess_np,
            total_grad, total_hess,
            reg_lambda, min_child_weight,
            has_missing if has_missing is not None else np.zeros(n_features, dtype=np.bool_),
            is_categorical if is_categorical is not None else np.zeros(n_features, dtype=np.bool_),
            n_categories if n_categories is not None else np.zeros(n_features, dtype=np.int32),
        )
    
    # Apply minimum gain threshold
    if gain < min_gain:
        return SplitInfo(feature=-1, threshold=-1, gain=0.0, missing_go_left=True)
    
    return SplitInfo(
        feature=feature,
        threshold=threshold,
        gain=gain,
        missing_go_left=missing_left,
        is_categorical=is_cat,
        cat_bitset=cat_bitset,
        cat_threshold=cat_threshold,
    )

