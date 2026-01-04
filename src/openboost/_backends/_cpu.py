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
) -> np.ndarray:
    """Predict using a tree on CPU.
    
    Returns:
        predictions: Shape (n_samples,), float32
    """
    n_samples = binned.shape[1]
    predictions = np.empty(n_samples, dtype=np.float32)
    
    _predict_cpu(
        binned, tree_features, tree_thresholds, tree_values,
        tree_left, tree_right, predictions
    )
    
    return predictions

