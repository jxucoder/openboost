"""Split finding for gradient boosting trees."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from .._backends import is_cuda

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SplitInfo(NamedTuple):
    """Information about a split."""
    feature: int      # Feature index (-1 if no valid split)
    threshold: int    # Bin threshold (go left if bin <= threshold)
    gain: float       # Split gain
    
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

