"""Histogram building for gradient boosting.

Dispatches to CUDA or CPU backend based on configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._backends import is_cuda

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from .._array import BinnedArray


def build_histogram(
    binned: BinnedArray | NDArray,
    grad: NDArray,
    hess: NDArray,
    sample_indices: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Build gradient and hessian histograms.
    
    Args:
        binned: Binned feature data (BinnedArray or raw array)
        grad: Gradient vector, shape (n_samples,) or (n_subset,)
        hess: Hessian vector, shape (n_samples,) or (n_subset,)
        sample_indices: Optional subset of samples to use
        
    Returns:
        hist_grad: Shape (n_features, 256), float64
        hist_hess: Shape (n_features, 256), float64
    """
    # Extract raw data if BinnedArray
    from .._array import BinnedArray
    if isinstance(binned, BinnedArray):
        binned_data = binned.data
        device = binned.device
    else:
        binned_data = binned
        device = "cuda" if is_cuda() and hasattr(binned_data, '__cuda_array_interface__') else "cpu"
    
    # Handle sample subsetting
    if sample_indices is not None:
        # Subset the data (needed for node-specific histograms)
        if device == "cuda":
            from .._backends._cuda import gather_cuda
            binned_data = gather_cuda(binned_data, sample_indices)
            grad = gather_cuda(grad, sample_indices)
            hess = gather_cuda(hess, sample_indices)
        else:
            binned_data = binned_data[:, sample_indices]
            grad = grad[sample_indices]
            hess = hess[sample_indices]
    
    # Dispatch to backend
    if device == "cuda" and is_cuda():
        from .._backends._cuda import build_histogram_cuda
        return build_histogram_cuda(binned_data, grad, hess)
    else:
        from .._backends._cpu import build_histogram_cpu
        # Ensure numpy arrays for CPU backend
        if hasattr(binned_data, 'copy_to_host'):
            binned_data = binned_data.copy_to_host()
        if hasattr(grad, 'copy_to_host'):
            grad = grad.copy_to_host()
        if hasattr(hess, 'copy_to_host'):
            hess = hess.copy_to_host()
        return build_histogram_cpu(
            np.asarray(binned_data),
            np.asarray(grad, dtype=np.float32),
            np.asarray(hess, dtype=np.float32),
        )


def subtract_histogram(
    parent_grad: NDArray,
    parent_hess: NDArray,
    child_grad: NDArray,
    child_hess: NDArray,
) -> tuple[NDArray, NDArray]:
    """Compute sibling histogram via subtraction.
    
    sibling = parent - child
    
    This is O(n_features * 256) instead of O(n_features * n_samples),
    giving ~2x speedup on histogram building.
    
    Args:
        parent_grad, parent_hess: Parent node histograms
        child_grad, child_hess: One child's histograms
        
    Returns:
        sibling_grad, sibling_hess: Other child's histograms
    """
    # Check if we're dealing with CUDA arrays
    if hasattr(parent_grad, '__cuda_array_interface__'):
        # Use CUDA subtraction kernel
        from .._backends._cuda import subtract_histograms_cuda
        return subtract_histograms_cuda(parent_grad, parent_hess, child_grad, child_hess)
    else:
        # NumPy arrays - direct subtraction
        sibling_grad = parent_grad - child_grad
        sibling_hess = parent_hess - child_hess
        return sibling_grad, sibling_hess


# Note: _subset_cuda removed in Phase 2, replaced by gather_cuda in _backends/_cuda.py

