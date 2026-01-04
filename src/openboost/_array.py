"""Array utilities for quantile binning."""

import numpy as np


def quantile_bin(X: np.ndarray, max_bins: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin features into uint8 using quantile binning.
    
    Args:
        X: Input array (n_samples, n_features), float32
        max_bins: Maximum number of bins (default 256 for uint8)
    
    Returns:
        X_binned: (n_features, n_samples) uint8, feature-major for coalesced GPU access
        bin_edges: List of bin edges per feature for later use
    """
    n_samples, n_features = X.shape
    X_binned = np.empty((n_features, n_samples), dtype=np.uint8)
    bin_edges = []
    
    for f in range(n_features):
        col = X[:, f]
        # Compute quantile edges
        percentiles = np.linspace(0, 100, max_bins + 1)
        edges = np.percentile(col, percentiles)
        edges = np.unique(edges)  # Remove duplicates
        
        # Digitize: bin index for each sample
        binned = np.digitize(col, edges[1:-1], right=False).astype(np.uint8)
        X_binned[f] = binned
        bin_edges.append(edges)
    
    return X_binned, bin_edges

