"""Array handling and binning for OpenBoost.

Provides `ob.array()` for converting data to the internal binned format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._backends import get_backend, is_cuda

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


@dataclass
class BinnedArray:
    """Binned feature matrix ready for tree building.
    
    Attributes:
        data: Binned data, shape (n_features, n_samples), dtype uint8
        bin_edges: List of bin edges per feature, for inverse transform
        n_features: Number of features
        n_samples: Number of samples
        device: "cuda" or "cpu"
    """
    data: NDArray[np.uint8]  # Or DeviceNDArray for CUDA
    bin_edges: list[NDArray[np.float64]]
    n_features: int
    n_samples: int
    device: str
    
    def __repr__(self) -> str:
        return (
            f"BinnedArray(n_features={self.n_features}, n_samples={self.n_samples}, "
            f"device={self.device!r})"
        )


def array(
    X: ArrayLike,
    n_bins: int = 256,
    *,
    device: str | None = None,
) -> BinnedArray:
    """Convert input data to binned format for tree building.
    
    This is the primary entry point for data. Binning is done once,
    then the binned data can be used for training many models.
    
    Args:
        X: Input features, shape (n_samples, n_features)
           Accepts numpy arrays, PyTorch tensors, JAX arrays, CuPy arrays.
        n_bins: Maximum number of bins (max 256 for uint8 storage).
        device: Target device ("cuda" or "cpu"). Auto-detected if None.
        
    Returns:
        BinnedArray with binned data in feature-major layout (n_features, n_samples).
        
    Example:
        >>> import openboost as ob
        >>> X_binned = ob.array(X_train)  # Bin once
        >>> for config in configs:
        ...     tree = ob.fit_tree(X_binned, grad, hess)  # Reuse binned data
    """
    if n_bins > 256:
        raise ValueError(f"n_bins must be <= 256 (uint8 storage), got {n_bins}")
    
    # Convert to numpy for binning computation
    X_np = _to_numpy(X)
    
    if X_np.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features), got shape {X_np.shape}")
    
    n_samples, n_features = X_np.shape
    
    # Compute bin edges and bin the data
    binned, bin_edges = _quantile_bin(X_np, n_bins)
    
    # Transpose to feature-major layout: (n_features, n_samples)
    binned = np.ascontiguousarray(binned.T)
    
    # Determine device
    if device is None:
        device = get_backend()
    
    # Transfer to GPU if needed
    if device == "cuda" and is_cuda():
        from ._backends._cuda import to_device
        binned = to_device(binned)
    
    return BinnedArray(
        data=binned,
        bin_edges=bin_edges,
        n_features=n_features,
        n_samples=n_samples,
        device=device,
    )


def _to_numpy(arr: ArrayLike) -> NDArray:
    """Convert various array types to numpy.
    
    Handles: numpy, PyTorch, JAX, CuPy
    """
    # Already numpy
    if isinstance(arr, np.ndarray):
        return arr
    
    # PyTorch
    if hasattr(arr, 'cpu') and hasattr(arr, 'numpy'):
        return arr.cpu().numpy()
    
    # JAX (has __array__ protocol)
    if hasattr(arr, '__array__'):
        return np.asarray(arr)
    
    # CuPy
    if hasattr(arr, 'get'):
        return arr.get()
    
    # Fallback
    return np.asarray(arr)


def _quantile_bin(
    X: NDArray[np.floating],
    n_bins: int,
) -> tuple[NDArray[np.uint8], list[NDArray[np.float64]]]:
    """Bin features using quantiles (parallelized across features).
    
    Args:
        X: Input data, shape (n_samples, n_features)
        n_bins: Number of bins
        
    Returns:
        binned: Binned data, shape (n_samples, n_features), uint8
        bin_edges: List of bin edges per feature
    """
    from joblib import Parallel, delayed
    
    n_samples, n_features = X.shape
    
    # Pre-compute percentiles (shared across all features)
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    
    def bin_single_feature(f: int) -> tuple[NDArray[np.uint8], NDArray[np.float64]]:
        """Bin a single feature column."""
        col = X[:, f].astype(np.float64)
        
        # Compute quantile-based bin edges
        edges = np.percentile(col, percentiles)
        
        # Remove duplicate edges (constant features)
        edges = np.unique(edges)
        
        # Digitize: maps values to bin indices 0..len(edges)
        binned_col = np.digitize(col, edges).astype(np.uint8)
        
        return binned_col, edges
    
    # Parallel processing across features
    # Use threads (not processes) to avoid data copying overhead
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(bin_single_feature)(f) for f in range(n_features)
    )
    
    # Combine results
    binned = np.column_stack([r[0] for r in results])
    bin_edges = [r[1] for r in results]
    
    return binned, bin_edges


def as_numba_array(arr):
    """Convert GPU array to Numba device array (zero-copy where possible).
    
    Supports PyTorch, JAX, CuPy arrays via __cuda_array_interface__.
    
    Args:
        arr: Array with __cuda_array_interface__ or numpy array
        
    Returns:
        Numba device array (CUDA) or numpy array (CPU)
    """
    # CUDA arrays (PyTorch .cuda(), JAX GPU, CuPy)
    if hasattr(arr, '__cuda_array_interface__'):
        if is_cuda():
            from ._backends._cuda import as_cuda_array
            return as_cuda_array(arr)
        else:
            raise RuntimeError(
                "Received CUDA array but CUDA backend not available. "
                "Call arr.cpu() first or set OPENBOOST_BACKEND=cpu"
            )
    
    # CPU arrays
    if hasattr(arr, '__array_interface__'):
        return np.asarray(arr)
    
    # Already numpy
    if isinstance(arr, np.ndarray):
        return arr
    
    raise TypeError(
        f"Cannot convert {type(arr).__name__} to Numba array. "
        "Expected numpy, PyTorch, JAX, or CuPy array."
    )


def ensure_contiguous_float32(arr) -> np.ndarray:
    """Ensure array is contiguous float32 (for grad/hess)."""
    arr = _to_numpy(arr) if not isinstance(arr, np.ndarray) else arr
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    return arr
