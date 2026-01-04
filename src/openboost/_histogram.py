"""Histogram building for gradient boosting."""

import numpy as np
from numba import cuda

from ._kernels import histogram_kernel, HIST_BLOCK_SIZE, MAX_BINS


def build_histograms(
    d_X_binned,      # Device array (n_features, n_samples)
    d_gradients,     # Device array (n_samples,)
    d_sample_nodes,  # Device array (n_samples,)
    node_start: int,
    n_nodes: int,
) -> np.ndarray:
    """
    Build histograms for all nodes at a level.
    
    Returns:
        histograms: (n_nodes, n_features, MAX_BINS, 2) on host
    """
    n_features, n_samples = d_X_binned.shape
    
    # Allocate histogram array on device
    d_histograms = cuda.device_array(
        (n_nodes, n_features, MAX_BINS, 2), dtype=np.float32
    )
    # Zero initialize
    cuda.to_device(
        np.zeros((n_nodes, n_features, MAX_BINS, 2), dtype=np.float32),
        to=d_histograms
    )
    
    # Launch: grid = (sample_blocks, n_features, n_nodes)
    threads = HIST_BLOCK_SIZE
    sample_blocks = min(256, (n_samples + threads - 1) // threads)
    blocks = (sample_blocks, n_features, n_nodes)
    
    histogram_kernel[blocks, threads](
        d_X_binned, d_gradients, d_sample_nodes,
        node_start, n_nodes, d_histograms
    )
    
    return d_histograms.copy_to_host()

