"""Prediction utilities for OpenBoost.

This module provides prediction for ensembles of trees.
Single-tree prediction is in _tree.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._array import BinnedArray
from .._backends import is_cuda

if TYPE_CHECKING:
    from ._tree import Tree
    from numpy.typing import NDArray


def predict_ensemble(
    trees: list[Tree],
    X: BinnedArray | NDArray,
    learning_rate: float = 1.0,
    init_score: float = 0.0,
) -> NDArray:
    """Predict using an ensemble of trees.
    
    Args:
        trees: List of fitted Tree objects
        X: BinnedArray or binned data
        learning_rate: Learning rate to apply to each tree
        init_score: Initial prediction value
        
    Returns:
        predictions: Shape (n_samples,)
    """
    if isinstance(X, BinnedArray):
        n_samples = X.n_samples
        device = X.device
    else:
        n_samples = X.shape[1]
        device = "cuda" if is_cuda() and hasattr(X, '__cuda_array_interface__') else "cpu"
    
    # Initialize predictions
    if device == "cuda" and is_cuda():
        from numba import cuda
        pred = cuda.device_array(n_samples, dtype=np.float32)
        # Initialize to init_score
        _fill_cuda(pred, init_score)
    else:
        pred = np.full(n_samples, init_score, dtype=np.float32)
    
    # Accumulate tree predictions
    for tree in trees:
        tree_pred = tree(X)
        if device == "cuda" and is_cuda():
            _add_inplace_cuda(pred, tree_pred, learning_rate)
        else:
            pred += learning_rate * tree_pred
    
    return pred


def _fill_cuda(arr, value: float):
    """Fill CUDA array with a value."""
    from numba import cuda
    
    @cuda.jit
    def _kernel(arr, val):
        idx = cuda.grid(1)
        if idx < arr.shape[0]:
            arr[idx] = val
    
    threads = 256
    blocks = (len(arr) + threads - 1) // threads
    _kernel[blocks, threads](arr, value)


def _add_inplace_cuda(arr, other, scale: float):
    """arr += scale * other (in-place on GPU)."""
    from numba import cuda
    
    @cuda.jit
    def _kernel(arr, other, scale):
        idx = cuda.grid(1)
        if idx < arr.shape[0]:
            arr[idx] += scale * other[idx]
    
    threads = 256
    blocks = (len(arr) + threads - 1) // threads
    _kernel[blocks, threads](arr, other, scale)


# =============================================================================
# Efficient In-Place Tree Prediction (Phase 5)
# =============================================================================

def predict_tree_add_gpu(
    tree: Tree,
    X: BinnedArray,
    pred_gpu,
    learning_rate: float = 1.0,
):
    """Add tree predictions to pred_gpu in-place (no intermediate allocation).
    
    This is more efficient than tree(X) + add because it:
    1. Doesn't allocate a new array for tree predictions
    2. Fuses traversal and addition into a single kernel
    3. Uses GPU-resident tree arrays if available (Phase 5.1 - zero copy!)
    
    Args:
        tree: Fitted Tree object
        X: BinnedArray with binned features
        pred_gpu: Device array to update in-place
        learning_rate: Scale factor for predictions
    """
    # Phase 5.1: Use to_gpu_arrays() which returns GPU arrays directly
    # if tree was built with fit_tree_gpu_native() (zero copy!)
    # Note: to_gpu_arrays() returns (features, thresholds, values, left, right)
    node_features, node_thresholds, node_values, node_left, node_right = tree.to_gpu_arrays()
    
    # Get binned data
    X_data = X.data if isinstance(X, BinnedArray) else X
    n_samples = X.n_samples if isinstance(X, BinnedArray) else X.shape[1]
    
    threads = 256
    blocks = (n_samples + threads - 1) // threads
    
    # Kernel expects: features, thresholds, left, right, values (match signature!)
    _predict_tree_add_kernel[blocks, threads](
        X_data, node_features, node_thresholds, node_left, node_right,
        node_values, pred_gpu, learning_rate, n_samples
    )


# Module-level kernel to avoid recompilation
_predict_tree_add_kernel = None


def _get_predict_tree_add_kernel():
    """Get or compile the predict-add kernel."""
    global _predict_tree_add_kernel
    if _predict_tree_add_kernel is not None:
        return _predict_tree_add_kernel
    
    from numba import cuda
    
    @cuda.jit
    def kernel(X_binned, node_features, node_thresholds, node_left, node_right,
               node_values, pred, learning_rate, n_samples):
        idx = cuda.grid(1)
        if idx < n_samples:
            # Tree traversal
            node = 0
            while node_features[node] >= 0:  # Not a leaf
                feat = node_features[node]
                val = X_binned[feat, idx]  # Feature-major layout
                if val <= node_thresholds[node]:
                    node = node_left[node]
                else:
                    node = node_right[node]
            
            # Add leaf value to prediction
            pred[idx] += learning_rate * node_values[node]
    
    _predict_tree_add_kernel = kernel
    return kernel


# Initialize kernel at module load if CUDA available
if is_cuda():
    try:
        _predict_tree_add_kernel = _get_predict_tree_add_kernel()
    except Exception:
        pass

