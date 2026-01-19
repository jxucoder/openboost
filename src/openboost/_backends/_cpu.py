"""CPU backend implementations using Numba JIT."""

from __future__ import annotations

import numpy as np
from numba import jit, prange


# =============================================================================
# Histogram Functions
# =============================================================================

@jit(nopython=True, parallel=True, cache=True)
def _build_histogram_cpu(
    binned: np.ndarray,  # (n_features, n_samples) uint8
    grad: np.ndarray,    # (n_samples,) float32
    hess: np.ndarray,    # (n_samples,) float32
    hist_grad: np.ndarray,  # (n_features, 256) float32
    hist_hess: np.ndarray,  # (n_features, 256) float32
):
    """Build gradient and hessian histograms for all features (CPU).
    
    Phase 3.3: Use float32 to match GPU performance characteristics.
    """
    n_features = binned.shape[0]
    n_samples = binned.shape[1]
    
    # Process features in parallel
    for f in prange(n_features):
        # Local histograms (float32 to match GPU)
        local_grad = np.zeros(256, dtype=np.float32)
        local_hess = np.zeros(256, dtype=np.float32)
        
        for i in range(n_samples):
            bin_idx = binned[f, i]
            local_grad[bin_idx] += grad[i]
            local_hess[bin_idx] += hess[i]
        
        hist_grad[f, :] = local_grad
        hist_hess[f, :] = local_hess


def build_histogram_cpu(
    binned: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build histograms on CPU.
    
    Args:
        binned: Binned feature matrix, shape (n_features, n_samples), uint8
        grad: Gradient vector, shape (n_samples,), float32
        hess: Hessian vector, shape (n_samples,), float32
        
    Returns:
        hist_grad: Gradient histogram, shape (n_features, 256), float32
        hist_hess: Hessian histogram, shape (n_features, 256), float32
    """
    n_features = binned.shape[0]
    
    # Phase 3.3: float32 to match GPU
    hist_grad = np.zeros((n_features, 256), dtype=np.float32)
    hist_hess = np.zeros((n_features, 256), dtype=np.float32)
    
    _build_histogram_cpu(binned, grad, hess, hist_grad, hist_hess)
    
    return hist_grad, hist_hess


# =============================================================================
# Split Finding
# =============================================================================

@jit(nopython=True, parallel=True, cache=True)
def _find_best_split_all_features(
    hist_grad: np.ndarray,   # (n_features, 256) float64
    hist_hess: np.ndarray,   # (n_features, 256) float64
    total_grad: float,
    total_hess: float,
    reg_lambda: float,
    min_child_weight: float,
    best_gains: np.ndarray,  # (n_features,) float64
    best_bins: np.ndarray,   # (n_features,) int32
):
    """Find best split for each feature in parallel."""
    n_features = hist_grad.shape[0]
    
    parent_gain = (total_grad * total_grad) / (total_hess + reg_lambda)
    
    for f in prange(n_features):
        best_gain = -1e10
        best_bin = -1
        
        left_grad = 0.0
        left_hess = 0.0
        
        for bin_idx in range(255):
            left_grad += hist_grad[f, bin_idx]
            left_hess += hist_hess[f, bin_idx]
            
            right_grad = total_grad - left_grad
            right_hess = total_hess - left_hess
            
            if left_hess < min_child_weight or right_hess < min_child_weight:
                continue
            
            left_score = (left_grad * left_grad) / (left_hess + reg_lambda)
            right_score = (right_grad * right_grad) / (right_hess + reg_lambda)
            gain = left_score + right_score - parent_gain
            
            if gain > best_gain:
                best_gain = gain
                best_bin = bin_idx
        
        best_gains[f] = best_gain
        best_bins[f] = best_bin


def find_best_split_cpu(
    hist_grad: np.ndarray,
    hist_hess: np.ndarray,
    total_grad: float,
    total_hess: float,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
) -> tuple[int, int, float]:
    """Find the best split across all features (CPU).
    
    Returns:
        best_feature: Index of best feature (-1 if no valid split)
        best_bin: Bin threshold for split
        best_gain: Gain from the split
    """
    n_features = hist_grad.shape[0]
    
    # Phase 3.3: Keep float64 for gain comparison precision on CPU
    # (CPU has no float64 penalty, and this is small data)
    best_gains = np.full(n_features, -1e10, dtype=np.float64)
    best_bins = np.full(n_features, -1, dtype=np.int32)
    
    _find_best_split_all_features(
        hist_grad, hist_hess,
        total_grad, total_hess,
        reg_lambda, min_child_weight,
        best_gains, best_bins,
    )
    
    best_feature = int(np.argmax(best_gains))
    best_gain = float(best_gains[best_feature])
    best_bin = int(best_bins[best_feature])
    
    if best_gain <= 0 or best_bin < 0:
        return -1, -1, 0.0
    
    return best_feature, best_bin, best_gain


# =============================================================================
# Phase 14: Missing Value Support
# =============================================================================

MISSING_BIN = 255  # Reserved bin for missing values


@jit(nopython=True, cache=True)
def _compute_gain(
    left_grad: float,
    left_hess: float,
    right_grad: float,
    right_hess: float,
    reg_lambda: float,
) -> float:
    """Compute split gain."""
    total_grad = left_grad + right_grad
    total_hess = left_hess + right_hess
    parent_gain = (total_grad * total_grad) / (total_hess + reg_lambda)
    left_gain = (left_grad * left_grad) / (left_hess + reg_lambda)
    right_gain = (right_grad * right_grad) / (right_hess + reg_lambda)
    return left_gain + right_gain - parent_gain


@jit(nopython=True, parallel=True, cache=True)
def _find_best_split_with_missing_all_features(
    hist_grad: np.ndarray,     # (n_features, 256) float64
    hist_hess: np.ndarray,     # (n_features, 256) float64
    total_grad: float,
    total_hess: float,
    reg_lambda: float,
    min_child_weight: float,
    has_missing: np.ndarray,   # (n_features,) bool
    best_gains: np.ndarray,    # (n_features,) float64
    best_bins: np.ndarray,     # (n_features,) int32
    best_missing_left: np.ndarray,  # (n_features,) bool
):
    """Find best split for each feature considering missing values."""
    n_features = hist_grad.shape[0]
    
    for f in prange(n_features):
        best_gain = -1e10
        best_bin = -1
        best_miss_left = True
        
        # Extract missing statistics for this feature
        miss_grad = hist_grad[f, MISSING_BIN]
        miss_hess = hist_hess[f, MISSING_BIN]
        feature_has_missing = has_missing[f]
        
        # Total for non-missing values
        nonmiss_total_grad = total_grad - miss_grad if feature_has_missing else total_grad
        nonmiss_total_hess = total_hess - miss_hess if feature_has_missing else total_hess
        
        left_grad = 0.0
        left_hess = 0.0
        
        # Iterate through bins 0-254 (not 255 which is missing)
        for bin_idx in range(255):
            left_grad += hist_grad[f, bin_idx]
            left_hess += hist_hess[f, bin_idx]
            
            right_grad = nonmiss_total_grad - left_grad
            right_hess = nonmiss_total_hess - left_hess
            
            if feature_has_missing and (miss_grad != 0.0 or miss_hess != 0.0):
                # Try missing goes LEFT
                left_g_miss = left_grad + miss_grad
                left_h_miss = left_hess + miss_hess
                
                if left_h_miss >= min_child_weight and right_hess >= min_child_weight:
                    gain_miss_left = _compute_gain(
                        left_g_miss, left_h_miss,
                        right_grad, right_hess,
                        reg_lambda
                    )
                    if gain_miss_left > best_gain:
                        best_gain = gain_miss_left
                        best_bin = bin_idx
                        best_miss_left = True
                
                # Try missing goes RIGHT
                right_g_miss = right_grad + miss_grad
                right_h_miss = right_hess + miss_hess
                
                if left_hess >= min_child_weight and right_h_miss >= min_child_weight:
                    gain_miss_right = _compute_gain(
                        left_grad, left_hess,
                        right_g_miss, right_h_miss,
                        reg_lambda
                    )
                    if gain_miss_right > best_gain:
                        best_gain = gain_miss_right
                        best_bin = bin_idx
                        best_miss_left = False
            else:
                # No missing values - standard split
                if left_hess >= min_child_weight and right_hess >= min_child_weight:
                    gain = _compute_gain(left_grad, left_hess, right_grad, right_hess, reg_lambda)
                    if gain > best_gain:
                        best_gain = gain
                        best_bin = bin_idx
                        best_miss_left = True  # Default direction
        
        best_gains[f] = best_gain
        best_bins[f] = best_bin
        best_missing_left[f] = best_miss_left


def find_best_split_with_missing_cpu(
    hist_grad: np.ndarray,
    hist_hess: np.ndarray,
    total_grad: float,
    total_hess: float,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    has_missing: np.ndarray | None = None,
) -> tuple[int, int, float, bool]:
    """Find the best split considering missing values (CPU).
    
    Returns:
        best_feature: Index of best feature (-1 if no valid split)
        best_bin: Bin threshold for split
        best_gain: Gain from the split
        missing_go_left: Whether missing values should go left
    """
    n_features = hist_grad.shape[0]
    
    # If has_missing not provided, check for non-zero bin 255
    if has_missing is None:
        has_missing = np.zeros(n_features, dtype=np.bool_)
        for f in range(n_features):
            if hist_grad[f, MISSING_BIN] != 0 or hist_hess[f, MISSING_BIN] != 0:
                has_missing[f] = True
    
    best_gains = np.full(n_features, -1e10, dtype=np.float64)
    best_bins = np.full(n_features, -1, dtype=np.int32)
    best_missing_left = np.ones(n_features, dtype=np.bool_)
    
    _find_best_split_with_missing_all_features(
        hist_grad.astype(np.float64),
        hist_hess.astype(np.float64),
        total_grad, total_hess,
        reg_lambda, min_child_weight,
        has_missing,
        best_gains, best_bins, best_missing_left,
    )
    
    best_feature = int(np.argmax(best_gains))
    best_gain = float(best_gains[best_feature])
    best_bin = int(best_bins[best_feature])
    best_miss_left = bool(best_missing_left[best_feature])
    
    if best_gain <= 0 or best_bin < 0:
        return -1, -1, 0.0, True
    
    return best_feature, best_bin, best_gain, best_miss_left


# =============================================================================
# Phase 14.3: Categorical Split Finding
# =============================================================================

def find_best_split_categorical_cpu(
    hist_grad: np.ndarray,
    hist_hess: np.ndarray,
    total_grad: float,
    total_hess: float,
    reg_lambda: float,
    min_child_weight: float,
    has_missing: np.ndarray,
    is_categorical: np.ndarray,
    n_categories: np.ndarray,
) -> tuple[int, int, float, bool, bool, int, int]:
    """Find best split with categorical and missing value support (CPU).
    
    For numeric features: ordinal splits (value <= threshold)
    For categorical features: Fisher-based optimal category ordering
    
    Returns:
        best_feature: Feature index (-1 if no valid split)
        best_threshold: Bin threshold (ordinal) or category split point
        best_gain: Gain from the split
        missing_go_left: Direction for missing values
        is_cat: Whether the best split is categorical
        cat_bitset: Bitmask for categories going left (for categorical)
        cat_threshold: Split point in sorted category order
    """
    n_features = hist_grad.shape[0]
    
    best_feature = -1
    best_threshold = -1
    best_gain = -1e10
    best_missing_left = True
    best_is_cat = False
    best_cat_bitset = 0
    best_cat_threshold = -1
    
    for f in range(n_features):
        miss_grad = hist_grad[f, MISSING_BIN]
        miss_hess = hist_hess[f, MISSING_BIN]
        has_miss = has_missing[f]
        
        if is_categorical[f]:
            # Categorical split using Fisher's optimal ordering
            n_cats = int(n_categories[f])
            if n_cats < 2:
                continue
            
            gain, cat_threshold, cat_bitset, miss_left = _find_best_categorical_split(
                hist_grad[f], hist_hess[f],
                n_cats, miss_grad, miss_hess, has_miss,
                reg_lambda, min_child_weight,
            )
            
            if gain > best_gain:
                best_gain = gain
                best_feature = f
                best_threshold = cat_threshold
                best_missing_left = miss_left
                best_is_cat = True
                best_cat_bitset = cat_bitset
                best_cat_threshold = cat_threshold
        else:
            # Numeric split (ordinal)
            gain, threshold, miss_left = _find_best_numeric_split(
                hist_grad[f], hist_hess[f],
                total_grad, total_hess,
                miss_grad, miss_hess, has_miss,
                reg_lambda, min_child_weight,
            )
            
            if gain > best_gain:
                best_gain = gain
                best_feature = f
                best_threshold = threshold
                best_missing_left = miss_left
                best_is_cat = False
                best_cat_bitset = 0
                best_cat_threshold = -1
    
    if best_gain <= 0 or best_feature < 0:
        return -1, -1, 0.0, True, False, 0, -1
    
    return (best_feature, best_threshold, best_gain, best_missing_left,
            best_is_cat, best_cat_bitset, best_cat_threshold)


def _find_best_numeric_split(
    hist_grad_f: np.ndarray,  # (256,)
    hist_hess_f: np.ndarray,
    total_grad: float,
    total_hess: float,
    miss_grad: float,
    miss_hess: float,
    has_miss: bool,
    reg_lambda: float,
    min_child_weight: float,
) -> tuple[float, int, bool]:
    """Find best numeric (ordinal) split for a single feature.
    
    Returns: (gain, threshold, missing_go_left)
    """
    best_gain = -1e10
    best_threshold = -1
    best_miss_left = True
    
    # Non-missing totals
    nonmiss_total_grad = total_grad - miss_grad if has_miss else total_grad
    nonmiss_total_hess = total_hess - miss_hess if has_miss else total_hess
    
    left_grad = 0.0
    left_hess = 0.0
    
    for bin_idx in range(255):
        left_grad += hist_grad_f[bin_idx]
        left_hess += hist_hess_f[bin_idx]
        
        right_grad = nonmiss_total_grad - left_grad
        right_hess = nonmiss_total_hess - left_hess
        
        if has_miss and (miss_grad != 0.0 or miss_hess != 0.0):
            # Try missing LEFT
            left_g_miss = left_grad + miss_grad
            left_h_miss = left_hess + miss_hess
            
            if left_h_miss >= min_child_weight and right_hess >= min_child_weight:
                gain = _compute_gain(left_g_miss, left_h_miss, right_grad, right_hess, reg_lambda)
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = bin_idx
                    best_miss_left = True
            
            # Try missing RIGHT
            right_g_miss = right_grad + miss_grad
            right_h_miss = right_hess + miss_hess
            
            if left_hess >= min_child_weight and right_h_miss >= min_child_weight:
                gain = _compute_gain(left_grad, left_hess, right_g_miss, right_h_miss, reg_lambda)
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = bin_idx
                    best_miss_left = False
        else:
            if left_hess >= min_child_weight and right_hess >= min_child_weight:
                gain = _compute_gain(left_grad, left_hess, right_grad, right_hess, reg_lambda)
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = bin_idx
                    best_miss_left = True
    
    return best_gain, best_threshold, best_miss_left


def _find_best_categorical_split(
    hist_grad_f: np.ndarray,  # (256,)
    hist_hess_f: np.ndarray,
    n_categories: int,
    miss_grad: float,
    miss_hess: float,
    has_miss: bool,
    reg_lambda: float,
    min_child_weight: float,
) -> tuple[float, int, int, bool]:
    """Find best categorical split using Fisher's optimal ordering.
    
    Fisher's method: Sort categories by gradient/hessian ratio,
    then find best split point in sorted order.
    
    Returns: (gain, cat_threshold, cat_bitset, missing_go_left)
    """
    # Compute ordering score for each category: -G / (H + lambda)
    eps = 1e-10
    scores = np.zeros(n_categories, dtype=np.float64)
    cat_grads = np.zeros(n_categories, dtype=np.float64)
    cat_hess = np.zeros(n_categories, dtype=np.float64)
    
    for cat in range(n_categories):
        cat_grads[cat] = hist_grad_f[cat]
        cat_hess[cat] = hist_hess_f[cat]
        if cat_hess[cat] > eps:
            scores[cat] = -cat_grads[cat] / (cat_hess[cat] + reg_lambda)
        else:
            scores[cat] = 0.0
    
    # Sort categories by score
    sorted_cats = np.argsort(scores)
    
    # Total gradient/hessian for this feature (excluding missing)
    total_g = float(np.sum(cat_grads))
    total_h = float(np.sum(cat_hess))
    
    best_gain = -1e10
    best_split = 0
    best_miss_left = True
    
    # Find best split point in sorted order
    left_g = 0.0
    left_h = 0.0
    
    for i in range(n_categories - 1):
        cat = sorted_cats[i]
        left_g += cat_grads[cat]
        left_h += cat_hess[cat]
        
        right_g = total_g - left_g
        right_h = total_h - left_h
        
        if has_miss and (miss_grad != 0.0 or miss_hess != 0.0):
            # Try missing LEFT
            if left_h + miss_hess >= min_child_weight and right_h >= min_child_weight:
                gain = _compute_gain(left_g + miss_grad, left_h + miss_hess,
                                    right_g, right_h, reg_lambda)
                if gain > best_gain:
                    best_gain = gain
                    best_split = i + 1
                    best_miss_left = True
            
            # Try missing RIGHT
            if left_h >= min_child_weight and right_h + miss_hess >= min_child_weight:
                gain = _compute_gain(left_g, left_h,
                                    right_g + miss_grad, right_h + miss_hess, reg_lambda)
                if gain > best_gain:
                    best_gain = gain
                    best_split = i + 1
                    best_miss_left = False
        else:
            if left_h >= min_child_weight and right_h >= min_child_weight:
                gain = _compute_gain(left_g, left_h, right_g, right_h, reg_lambda)
                if gain > best_gain:
                    best_gain = gain
                    best_split = i + 1
                    best_miss_left = True
    
    # Build bitmask: categories in sorted order before split_point go left
    cat_bitset = 0
    for i in range(best_split):
        cat = sorted_cats[i]
        cat_bitset |= (1 << cat)
    
    return best_gain, best_split, cat_bitset, best_miss_left


# =============================================================================
# Prediction
# =============================================================================

@jit(nopython=True, parallel=True, cache=True)
def _predict_cpu(
    binned: np.ndarray,          # (n_features, n_samples) uint8
    tree_features: np.ndarray,   # (n_nodes,) int32
    tree_thresholds: np.ndarray, # (n_nodes,) uint8
    tree_values: np.ndarray,     # (n_nodes,) float32
    tree_left: np.ndarray,       # (n_nodes,) int32
    tree_right: np.ndarray,      # (n_nodes,) int32
    predictions: np.ndarray,     # (n_samples,) float32
):
    """Predict using tree structure (CPU)."""
    n_samples = binned.shape[1]
    
    for i in prange(n_samples):
        node = 0
        while tree_left[node] != -1:
            feature = tree_features[node]
            threshold = tree_thresholds[node]
            bin_value = binned[feature, i]
            
            if bin_value <= threshold:
                node = tree_left[node]
            else:
                node = tree_right[node]
        
        predictions[i] = tree_values[node]


def predict_cpu(
    binned: np.ndarray,
    tree_features: np.ndarray,
    tree_thresholds: np.ndarray,
    tree_values: np.ndarray,
    tree_left: np.ndarray,
    tree_right: np.ndarray,
    tree_missing_left: np.ndarray | None = None,
) -> np.ndarray:
    """Predict using a tree on CPU.
    
    Phase 14: Added support for missing value routing.
    
    Args:
        binned: Binned features (n_features, n_samples)
        tree_features: Feature index for each node
        tree_thresholds: Threshold for each node
        tree_values: Leaf values
        tree_left: Left child indices
        tree_right: Right child indices
        tree_missing_left: Whether missing goes left for each node (Phase 14)
    
    Returns:
        predictions: Shape (n_samples,), float32
    """
    n_samples = binned.shape[1]
    predictions = np.empty(n_samples, dtype=np.float32)
    
    if tree_missing_left is not None:
        _predict_cpu_with_missing(
            binned, tree_features, tree_thresholds, tree_values,
            tree_left, tree_right, tree_missing_left, predictions
        )
    else:
        _predict_cpu(
            binned, tree_features, tree_thresholds, tree_values,
            tree_left, tree_right, predictions
        )
    
    return predictions


# =============================================================================
# Phase 14: Prediction with Missing Value Support
# =============================================================================

@jit(nopython=True, parallel=True, cache=True)
def _predict_cpu_with_missing(
    binned: np.ndarray,          # (n_features, n_samples) uint8
    tree_features: np.ndarray,   # (n_nodes,) int32
    tree_thresholds: np.ndarray, # (n_nodes,) uint8
    tree_values: np.ndarray,     # (n_nodes,) float32
    tree_left: np.ndarray,       # (n_nodes,) int32
    tree_right: np.ndarray,      # (n_nodes,) int32
    tree_missing_left: np.ndarray,  # (n_nodes,) bool
    predictions: np.ndarray,     # (n_samples,) float32
):
    """Predict using tree structure with missing value handling (CPU).
    
    Missing values (bin 255) are routed according to the learned direction
    stored in tree_missing_left.
    """
    n_samples = binned.shape[1]
    
    for i in prange(n_samples):
        node = 0
        while tree_left[node] != -1:
            feature = tree_features[node]
            threshold = tree_thresholds[node]
            bin_value = binned[feature, i]
            
            # Check for missing value
            if bin_value == MISSING_BIN:
                # Use learned direction
                if tree_missing_left[node]:
                    node = tree_left[node]
                else:
                    node = tree_right[node]
            elif bin_value <= threshold:
                node = tree_left[node]
            else:
                node = tree_right[node]
        
        predictions[i] = tree_values[node]

