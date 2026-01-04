"""Split finding for gradient boosting."""

import numpy as np
from numba import cuda

from ._kernels import find_best_split_kernel, HIST_BLOCK_SIZE


def find_best_splits(
    histograms: np.ndarray,  # (n_nodes, n_features, MAX_BINS, 2)
    min_samples_leaf: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find best split for each node.
    
    Returns:
        best_gain: (n_nodes,)
        best_feature: (n_nodes,)
        best_bin: (n_nodes,)
    """
    n_nodes = histograms.shape[0]
    
    # Compute node totals from histograms
    node_sum_grad = histograms[:, :, :, 0].sum(axis=(1, 2))  # Sum over features and bins
    node_count = histograms[:, :, :, 1].sum(axis=(1, 2))
    # Fix: sum only over bins (axis 1 of the feature dimension gives wrong result)
    # Actually need sum over all bins for one feature (they should be same across features)
    node_sum_grad = histograms[:, 0, :, 0].sum(axis=1)  # Use feature 0
    node_count = histograms[:, 0, :, 1].sum(axis=1)
    
    # Allocate outputs
    d_histograms = cuda.to_device(histograms)
    d_node_sum_grad = cuda.to_device(node_sum_grad.astype(np.float32))
    d_node_count = cuda.to_device(node_count.astype(np.float32))
    d_best_gain = cuda.device_array(n_nodes, dtype=np.float32)
    d_best_feature = cuda.device_array(n_nodes, dtype=np.int32)
    d_best_bin = cuda.device_array(n_nodes, dtype=np.int32)
    
    # One block per node
    find_best_split_kernel[n_nodes, HIST_BLOCK_SIZE](
        d_histograms, d_node_sum_grad, d_node_count,
        d_best_gain, d_best_feature, d_best_bin,
        min_samples_leaf
    )
    
    return (
        d_best_gain.copy_to_host(),
        d_best_feature.copy_to_host(),
        d_best_bin.copy_to_host(),
    )


def compute_leaf_values(
    histograms: np.ndarray,  # (n_nodes, n_features, MAX_BINS, 2)
) -> np.ndarray:
    """Compute leaf values (mean of gradients) for each node."""
    # Sum gradients and counts across all bins (use feature 0 since totals are same)
    sum_grad = histograms[:, 0, :, 0].sum(axis=1)
    count = histograms[:, 0, :, 1].sum(axis=1)
    
    # Leaf value = -sum_grad / count (negative because we minimize)
    with np.errstate(divide='ignore', invalid='ignore'):
        values = np.where(count > 0, -sum_grad / count, 0.0)
    
    return values.astype(np.float32)

