"""Tree building primitives for OpenBoost.

Phase 8.1: Extract reusable primitives that can be composed into different
tree growth strategies (level-wise, leaf-wise, symmetric).

Phase 14: Added missing value handling - samples with bin 255 (NaN) are
routed according to the learned direction.

These primitives operate on the `sample_node_ids` paradigm:
- Each sample is assigned to a node ID
- Histograms are built by aggregating samples per node
- Partitioning updates node IDs based on split decisions

This design enables:
- Level-wise growth (XGBoost default): process all nodes at a depth
- Leaf-wise growth (LightGBM): process best leaf across all depths
- Symmetric growth (CatBoost): single split per depth level
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .._backends import is_cuda
from .._array import MISSING_BIN
from ._split import SplitInfo

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class NodeHistogram:
    """Histogram data for a single node."""
    node_id: int
    hist_grad: NDArray  # (n_features, 256) float32
    hist_hess: NDArray  # (n_features, 256) float32
    sum_grad: float
    sum_hess: float
    n_samples: int


@dataclass 
class NodeSplit:
    """Split decision for a single node.
    
    Phase 14: Added missing_go_left for NaN handling.
    Phase 14.3: Added categorical split info.
    """
    node_id: int
    split: SplitInfo
    left_child: int   # Node ID for left child (2 * node_id in binary tree)
    right_child: int  # Node ID for right child (2 * node_id + 1)
    missing_go_left: bool = True     # Direction for missing values (Phase 14)
    is_categorical: bool = False     # Phase 14.3: True if categorical split
    cat_bitset: int = 0              # Phase 14.3: Bitmask for categories going left


# =============================================================================
# Primitive: Build Histograms for Nodes
# =============================================================================

def build_node_histograms(
    binned: NDArray,
    grad: NDArray,
    hess: NDArray,
    sample_node_ids: NDArray,
    node_ids: list[int],
) -> dict[int, NodeHistogram]:
    """Build gradient/hessian histograms for specified nodes.
    
    This is the core primitive for tree building. It aggregates gradients
    and hessians into 256-bin histograms for each feature, grouped by node.
    
    Args:
        binned: Binned feature data, shape (n_features, n_samples), uint8
        grad: Gradients, shape (n_samples,), float32
        hess: Hessians, shape (n_samples,), float32
        sample_node_ids: Node assignment for each sample, shape (n_samples,), int32
        node_ids: List of node IDs to build histograms for
        
    Returns:
        Dictionary mapping node_id -> NodeHistogram
        
    Example:
        >>> # Build histograms for nodes at depth 2 (nodes 3, 4, 5, 6)
        >>> histograms = build_node_histograms(
        ...     binned, grad, hess, sample_node_ids,
        ...     node_ids=[3, 4, 5, 6]
        ... )
        >>> histograms[3].sum_grad  # Total gradient in node 3
    """
    if is_cuda() and hasattr(binned, '__cuda_array_interface__'):
        return _build_node_histograms_gpu(binned, grad, hess, sample_node_ids, node_ids)
    return _build_node_histograms_cpu(binned, grad, hess, sample_node_ids, node_ids)


def _build_node_histograms_cpu(
    binned: NDArray,
    grad: NDArray,
    hess: NDArray,
    sample_node_ids: NDArray,
    node_ids: list[int],
) -> dict[int, NodeHistogram]:
    """CPU implementation of node histogram building."""
    from .._backends._cpu import build_histogram_cpu
    
    # Ensure numpy arrays
    binned = np.asarray(binned)
    grad = np.asarray(grad, dtype=np.float32)
    hess = np.asarray(hess, dtype=np.float32)
    sample_node_ids = np.asarray(sample_node_ids, dtype=np.int32)
    
    n_features = binned.shape[0]
    result = {}
    
    for node_id in node_ids:
        # Get samples belonging to this node
        mask = sample_node_ids == node_id
        n_samples = int(np.sum(mask))
        
        if n_samples == 0:
            # Empty node - create zero histogram
            result[node_id] = NodeHistogram(
                node_id=node_id,
                hist_grad=np.zeros((n_features, 256), dtype=np.float32),
                hist_hess=np.zeros((n_features, 256), dtype=np.float32),
                sum_grad=0.0,
                sum_hess=0.0,
                n_samples=0,
            )
            continue
        
        # Subset data for this node
        node_binned = binned[:, mask]
        node_grad = grad[mask]
        node_hess = hess[mask]
        
        # Build histogram
        hist_grad, hist_hess = build_histogram_cpu(node_binned, node_grad, node_hess)
        
        # Compute sums from histogram (sum across all bins of any feature)
        sum_grad = float(np.sum(hist_grad[0]))
        sum_hess = float(np.sum(hist_hess[0]))
        
        result[node_id] = NodeHistogram(
            node_id=node_id,
            hist_grad=hist_grad,
            hist_hess=hist_hess,
            sum_grad=sum_grad,
            sum_hess=sum_hess,
            n_samples=n_samples,
        )
    
    return result


def _build_node_histograms_gpu(
    binned: NDArray,
    grad: NDArray,
    hess: NDArray,
    sample_node_ids: NDArray,
    node_ids: list[int],
) -> dict[int, NodeHistogram]:
    """GPU implementation of node histogram building.
    
    Uses the optimized shared memory histogram kernel from Phase 6.3.
    """
    from numba import cuda
    import math
    from .._backends._cuda import (
        _build_histogram_shared_kernel,
        _zero_level_histograms_kernel,
    )
    
    n_features, n_samples = binned.shape
    
    # Convert node_ids to contiguous range for kernel
    # The kernel expects nodes in range [level_start, level_start + n_nodes)
    # For arbitrary node_ids, we need to map them
    
    if not node_ids:
        return {}
    
    # For efficiency, check if nodes are contiguous (common case: level-wise)
    min_node = min(node_ids)
    max_node = max(node_ids)
    is_contiguous = (max_node - min_node + 1 == len(node_ids))
    
    if is_contiguous and len(node_ids) <= 32:
        # Fast path: use existing kernel directly
        return _build_node_histograms_gpu_contiguous(
            binned, grad, hess, sample_node_ids, min_node, len(node_ids)
        )
    
    # Slow path: build each node separately (for leaf-wise or sparse node sets)
    return _build_node_histograms_gpu_sparse(
        binned, grad, hess, sample_node_ids, node_ids
    )


def _build_node_histograms_gpu_contiguous(
    binned,
    grad,
    hess, 
    sample_node_ids,
    level_start: int,
    n_nodes: int,
) -> dict[int, NodeHistogram]:
    """GPU histogram building for contiguous node range."""
    from numba import cuda
    import math
    from .._backends._cuda import (
        _build_histogram_shared_kernel,
        _zero_level_histograms_kernel,
    )
    
    n_features, n_samples = binned.shape
    max_nodes = level_start + n_nodes
    
    # Allocate histogram storage
    # Shape: (max_nodes, n_features, 256, 2) where [:,:,:,0]=grad, [:,:,:,1]=hess
    histograms = cuda.device_array((max_nodes, n_features, 256, 2), dtype=np.float32)
    
    # Zero histograms
    level_end = level_start + n_nodes
    zero_grid = (n_nodes, n_features)
    _zero_level_histograms_kernel[zero_grid, 256](
        histograms, level_start, level_end, n_features
    )
    
    # Build histograms using shared memory kernel
    CHUNK_SIZE = 4096
    n_chunks = math.ceil(n_samples / CHUNK_SIZE)
    hist_grid = (n_features, n_chunks)
    
    if n_nodes <= 16:
        _build_histogram_shared_kernel[hist_grid, 256](
            binned, grad, hess, sample_node_ids,
            level_start, n_nodes, 0,
            histograms
        )
    else:
        # Two passes for >16 nodes
        _build_histogram_shared_kernel[hist_grid, 256](
            binned, grad, hess, sample_node_ids,
            level_start, 16, 0,
            histograms
        )
        remaining = n_nodes - 16
        _build_histogram_shared_kernel[hist_grid, 256](
            binned, grad, hess, sample_node_ids,
            level_start, remaining, 16,
            histograms
        )
    
    # Copy histograms to host and create NodeHistogram objects
    histograms_cpu = histograms.copy_to_host()
    sample_node_ids_cpu = sample_node_ids.copy_to_host()
    
    result = {}
    for i, node_id in enumerate(range(level_start, level_end)):
        node_hist = histograms_cpu[node_id]
        hist_grad = node_hist[:, :, 0]  # (n_features, 256)
        hist_hess = node_hist[:, :, 1]
        
        sum_grad = float(np.sum(hist_grad[0]))
        sum_hess = float(np.sum(hist_hess[0]))
        n_samples_node = int(np.sum(sample_node_ids_cpu == node_id))
        
        result[node_id] = NodeHistogram(
            node_id=node_id,
            hist_grad=hist_grad.copy(),
            hist_hess=hist_hess.copy(),
            sum_grad=sum_grad,
            sum_hess=sum_hess,
            n_samples=n_samples_node,
        )
    
    return result


def _build_node_histograms_gpu_sparse(
    binned,
    grad,
    hess,
    sample_node_ids,
    node_ids: list[int],
) -> dict[int, NodeHistogram]:
    """GPU histogram building for non-contiguous nodes (leaf-wise)."""
    from numba import cuda
    from .._backends._cuda import build_histogram_cuda, gather_cuda
    
    # For sparse node sets, build each node separately
    # This is less efficient but works for any node configuration
    
    sample_node_ids_cpu = sample_node_ids.copy_to_host()
    
    result = {}
    for node_id in node_ids:
        mask = sample_node_ids_cpu == node_id
        n_samples_node = int(np.sum(mask))
        
        if n_samples_node == 0:
            n_features = binned.shape[0]
            result[node_id] = NodeHistogram(
                node_id=node_id,
                hist_grad=np.zeros((n_features, 256), dtype=np.float32),
                hist_hess=np.zeros((n_features, 256), dtype=np.float32),
                sum_grad=0.0,
                sum_hess=0.0,
                n_samples=0,
            )
            continue
        
        # Get indices for this node
        indices = np.where(mask)[0].astype(np.int32)
        indices_gpu = cuda.to_device(indices)
        
        # Gather data for this node
        node_binned = gather_cuda(binned, indices_gpu)
        node_grad = gather_cuda(grad, indices_gpu)
        node_hess = gather_cuda(hess, indices_gpu)
        
        # Build histogram
        hist_grad, hist_hess = build_histogram_cuda(node_binned, node_grad, node_hess)
        
        # Copy to CPU
        hist_grad_cpu = hist_grad.copy_to_host()
        hist_hess_cpu = hist_hess.copy_to_host()
        
        sum_grad = float(np.sum(hist_grad_cpu[0]))
        sum_hess = float(np.sum(hist_hess_cpu[0]))
        
        result[node_id] = NodeHistogram(
            node_id=node_id,
            hist_grad=hist_grad_cpu,
            hist_hess=hist_hess_cpu,
            sum_grad=sum_grad,
            sum_hess=sum_hess,
            n_samples=n_samples_node,
        )
    
    return result


# =============================================================================
# Primitive: Histogram Subtraction
# =============================================================================

def subtract_histogram(
    parent: NodeHistogram,
    child: NodeHistogram,
    sibling_node_id: int,
) -> NodeHistogram:
    """Compute sibling histogram via subtraction: sibling = parent - child.
    
    This is O(n_features * 256) instead of O(n_features * n_samples),
    giving ~2x speedup on histogram building when used with the
    "build smaller child, subtract for larger" strategy.
    
    Args:
        parent: Parent node histogram
        child: One child's histogram
        sibling_node_id: Node ID for the sibling
        
    Returns:
        NodeHistogram for the sibling node
    """
    sibling_grad = parent.hist_grad - child.hist_grad
    sibling_hess = parent.hist_hess - child.hist_hess
    
    return NodeHistogram(
        node_id=sibling_node_id,
        hist_grad=sibling_grad,
        hist_hess=sibling_hess,
        sum_grad=parent.sum_grad - child.sum_grad,
        sum_hess=parent.sum_hess - child.sum_hess,
        n_samples=parent.n_samples - child.n_samples,
    )


# =============================================================================
# Primitive: Find Splits for Nodes
# =============================================================================

def find_node_splits(
    histograms: dict[int, NodeHistogram],
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    min_gain: float = 0.0,
    has_missing: NDArray | None = None,
    is_categorical: NDArray | None = None,
    n_categories: NDArray | None = None,
) -> dict[int, NodeSplit]:
    """Find the best split for each node given histograms.
    
    Phase 14: Added has_missing parameter for NaN handling.
    Phase 14.3: Added categorical feature support.
    
    Args:
        histograms: Dictionary mapping node_id -> NodeHistogram
        reg_lambda: L2 regularization term
        min_child_weight: Minimum sum of hessian in each child
        min_gain: Minimum gain required to split
        has_missing: Boolean array (n_features,) indicating which features have NaN.
                     If None, standard split finding is used.
        is_categorical: Boolean array (n_features,) indicating categorical features
        n_categories: Number of categories per feature (0 for numeric)
        
    Returns:
        Dictionary mapping node_id -> NodeSplit
        Only includes nodes with valid splits (gain > min_gain).
        
    Example:
        >>> histograms = build_node_histograms(...)
        >>> splits = find_node_splits(histograms, reg_lambda=1.0)
        >>> for node_id, split in splits.items():
        ...     print(f"Node {node_id}: split on feature {split.split.feature}")
    """
    from ._split import find_best_split, find_best_split_with_missing, find_best_split_with_categorical
    
    result = {}
    
    # Check if we should use special split finding
    use_missing = has_missing is not None and np.any(has_missing)
    use_categorical = is_categorical is not None and np.any(is_categorical)
    
    for node_id, hist in histograms.items():
        # Skip empty nodes
        if hist.n_samples == 0:
            continue
        
        # Skip nodes that don't meet min_child_weight
        if hist.sum_hess < min_child_weight:
            continue
        
        # Find best split
        if use_categorical:
            # Phase 14.3: Use categorical-aware split finding
            split = find_best_split_with_categorical(
                hist.hist_grad,
                hist.hist_hess,
                hist.sum_grad,
                hist.sum_hess,
                reg_lambda=reg_lambda,
                min_child_weight=min_child_weight,
                min_gain=min_gain,
                has_missing=has_missing,
                is_categorical=is_categorical,
                n_categories=n_categories,
            )
        elif use_missing:
            split = find_best_split_with_missing(
                hist.hist_grad,
                hist.hist_hess,
                hist.sum_grad,
                hist.sum_hess,
                reg_lambda=reg_lambda,
                min_child_weight=min_child_weight,
                min_gain=min_gain,
                has_missing=has_missing,
            )
        else:
            split = find_best_split(
                hist.hist_grad,
                hist.hist_hess,
                hist.sum_grad,
                hist.sum_hess,
                reg_lambda=reg_lambda,
                min_child_weight=min_child_weight,
                min_gain=min_gain,
            )
        
        if split.is_valid:
            result[node_id] = NodeSplit(
                node_id=node_id,
                split=split,
                left_child=2 * node_id + 1,   # Binary tree indexing
                right_child=2 * node_id + 2,
                missing_go_left=split.missing_go_left,  # Phase 14
                is_categorical=split.is_categorical,    # Phase 14.3
                cat_bitset=split.cat_bitset,            # Phase 14.3
            )
    
    return result


# =============================================================================
# Primitive: Partition Samples
# =============================================================================

def partition_samples(
    binned: NDArray,
    sample_node_ids: NDArray,
    splits: dict[int, NodeSplit],
    missing_go_left: NDArray | None = None,
) -> NDArray:
    """Update sample node assignments based on split decisions.
    
    For each sample in a node that was split:
    - If feature[split.feature] <= split.threshold: go to left child
    - Otherwise: go to right child
    - Phase 14: Missing values (bin 255) go according to learned direction
    
    Samples in nodes without splits remain unchanged.
    
    Args:
        binned: Binned feature data, shape (n_features, n_samples), uint8
        sample_node_ids: Current node assignment, shape (n_samples,), int32
        splits: Dictionary of splits from find_node_splits()
        missing_go_left: Boolean array (n_nodes,) for missing direction (Phase 14)
        
    Returns:
        Updated sample_node_ids array (new array, original unchanged)
        
    Note:
        This function returns a new array. For GPU, the returned array
        is on GPU. For CPU, it's a numpy array.
    """
    if is_cuda() and hasattr(binned, '__cuda_array_interface__'):
        return _partition_samples_gpu(binned, sample_node_ids, splits, missing_go_left)
    return _partition_samples_cpu(binned, sample_node_ids, splits, missing_go_left)


def _partition_samples_cpu(
    binned: NDArray,
    sample_node_ids: NDArray,
    splits: dict[int, NodeSplit],
    missing_go_left: NDArray | None = None,
) -> NDArray:
    """CPU implementation of sample partitioning.
    
    Phase 14: Handles missing values (bin 255) using learned direction.
    Phase 14.3: Handles categorical splits using bitmask membership.
    """
    binned = np.asarray(binned)
    sample_node_ids = np.asarray(sample_node_ids, dtype=np.int32)
    
    # Create output array
    new_node_ids = sample_node_ids.copy()
    
    for node_id, node_split in splits.items():
        mask = sample_node_ids == node_id
        if not np.any(mask):
            continue
        
        feature = node_split.split.feature
        threshold = node_split.split.threshold
        
        # Get feature values for samples in this node
        feature_values = binned[feature, mask]
        
        # Phase 14: Handle missing values
        is_missing = feature_values == MISSING_BIN
        
        # Phase 14.3: Determine routing based on split type
        if node_split.is_categorical:
            # Categorical split: use bitmask membership
            bitset = node_split.cat_bitset
            goes_left = np.array([(bitset >> fv) & 1 for fv in feature_values], dtype=np.bool_)
        else:
            # Ordinal split: use threshold
            goes_left = feature_values <= threshold
        
        # Override for missing values using learned direction
        if np.any(is_missing):
            # Get the missing direction for this split
            miss_left = node_split.missing_go_left if hasattr(node_split, 'missing_go_left') else True
            if missing_go_left is not None and node_id < len(missing_go_left):
                miss_left = missing_go_left[node_id]
            goes_left[is_missing] = miss_left
        
        # Update node IDs
        sample_indices = np.where(mask)[0]
        new_node_ids[sample_indices[goes_left]] = node_split.left_child
        new_node_ids[sample_indices[~goes_left]] = node_split.right_child
    
    return new_node_ids


def _partition_samples_gpu(
    binned,
    sample_node_ids,
    splits: dict[int, NodeSplit],
    missing_go_left: NDArray | None = None,
) -> "DeviceNDArray":
    """GPU implementation of sample partitioning.
    
    Phase 14: Handles missing values (bin 255) using learned direction.
    """
    from numba import cuda
    import math
    
    n_samples = sample_node_ids.shape[0]
    
    # Create output array on GPU
    new_node_ids = cuda.device_array(n_samples, dtype=np.int32)
    
    # Copy current node IDs
    cuda.to_device(sample_node_ids.copy_to_host(), to=new_node_ids)
    
    # Build arrays for kernel
    if not splits:
        return new_node_ids
    
    # Create split lookup arrays
    max_node_id = max(splits.keys()) + 1
    split_features = np.full(max_node_id, -1, dtype=np.int32)
    split_thresholds = np.full(max_node_id, 0, dtype=np.int32)
    left_children = np.full(max_node_id, -1, dtype=np.int32)
    right_children = np.full(max_node_id, -1, dtype=np.int32)
    split_missing_left = np.ones(max_node_id, dtype=np.uint8)  # Phase 14
    
    for node_id, node_split in splits.items():
        split_features[node_id] = node_split.split.feature
        split_thresholds[node_id] = node_split.split.threshold
        left_children[node_id] = node_split.left_child
        right_children[node_id] = node_split.right_child
        split_missing_left[node_id] = 1 if node_split.missing_go_left else 0  # Phase 14
    
    # Transfer to GPU
    split_features_gpu = cuda.to_device(split_features)
    split_thresholds_gpu = cuda.to_device(split_thresholds)
    left_children_gpu = cuda.to_device(left_children)
    right_children_gpu = cuda.to_device(right_children)
    split_missing_left_gpu = cuda.to_device(split_missing_left)
    
    # Launch kernel
    threads = 256
    blocks = math.ceil(n_samples / threads)
    _partition_kernel_with_missing[blocks, threads](
        binned, sample_node_ids, new_node_ids,
        split_features_gpu, split_thresholds_gpu,
        left_children_gpu, right_children_gpu,
        split_missing_left_gpu,
        max_node_id, n_samples
    )
    
    return new_node_ids


# Partition kernel with missing value support (Phase 14)
_partition_kernel_with_missing = None

def _init_partition_kernel_with_missing():
    global _partition_kernel_with_missing
    if _partition_kernel_with_missing is not None:
        return
    
    from numba import cuda, int32, uint8
    
    @cuda.jit
    def kernel(binned, old_node_ids, new_node_ids, 
               split_features, split_thresholds, left_children, right_children,
               split_missing_left, max_node_id, n_samples):
        """Partition kernel with missing value handling (Phase 14)."""
        idx = cuda.grid(1)
        if idx >= n_samples:
            return
        
        node_id = old_node_ids[idx]
        
        # Check if this node was split
        if node_id >= max_node_id or split_features[node_id] < 0:
            new_node_ids[idx] = node_id
            return
        
        feature = split_features[node_id]
        threshold = split_thresholds[node_id]
        bin_value = int32(binned[feature, idx])
        
        # Phase 14: Handle missing values (bin 255)
        if bin_value == 255:
            # Use learned direction for missing
            if split_missing_left[node_id] == 1:
                new_node_ids[idx] = left_children[node_id]
            else:
                new_node_ids[idx] = right_children[node_id]
        elif bin_value <= threshold:
            new_node_ids[idx] = left_children[node_id]
        else:
            new_node_ids[idx] = right_children[node_id]
    
    _partition_kernel_with_missing = kernel


# Initialize kernel if CUDA available
if is_cuda():
    try:
        _init_partition_kernel_with_missing()
    except Exception:
        pass


# =============================================================================
# Primitive: Compute Leaf Values
# =============================================================================

def compute_leaf_values(
    grad: NDArray,
    hess: NDArray,
    sample_node_ids: NDArray,
    leaf_node_ids: list[int],
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
) -> dict[int, float]:
    """Compute optimal leaf values for specified nodes.
    
    Uses the Newton-Raphson optimal value with L1/L2 regularization.
    
    Without L1: -sum(grad) / (sum(hess) + lambda)
    With L1: soft-thresholding applied to gradients
    
    Args:
        grad: Gradients, shape (n_samples,), float32
        hess: Hessians, shape (n_samples,), float32
        sample_node_ids: Node assignment for each sample, shape (n_samples,), int32
        leaf_node_ids: List of node IDs to compute values for
        reg_lambda: L2 regularization term
        reg_alpha: L1 regularization term (Phase 11)
        
    Returns:
        Dictionary mapping node_id -> leaf_value
    """
    if is_cuda() and hasattr(grad, '__cuda_array_interface__'):
        return _compute_leaf_values_gpu(grad, hess, sample_node_ids, leaf_node_ids, reg_lambda, reg_alpha)
    return _compute_leaf_values_cpu(grad, hess, sample_node_ids, leaf_node_ids, reg_lambda, reg_alpha)


def _compute_leaf_values_cpu(
    grad: NDArray,
    hess: NDArray,
    sample_node_ids: NDArray,
    leaf_node_ids: list[int],
    reg_lambda: float,
    reg_alpha: float = 0.0,
) -> dict[int, float]:
    """CPU implementation of leaf value computation."""
    from ._split import compute_leaf_value
    
    grad = np.asarray(grad, dtype=np.float32)
    hess = np.asarray(hess, dtype=np.float32)
    sample_node_ids = np.asarray(sample_node_ids, dtype=np.int32)
    
    result = {}
    for node_id in leaf_node_ids:
        mask = sample_node_ids == node_id
        if not np.any(mask):
            result[node_id] = 0.0
            continue
        
        sum_grad = float(np.sum(grad[mask]))
        sum_hess = float(np.sum(hess[mask]))
        
        # Use shared leaf value computation (supports L1/L2)
        result[node_id] = compute_leaf_value(sum_grad, sum_hess, reg_lambda, reg_alpha)
    
    return result


def _compute_leaf_values_gpu(
    grad,
    hess,
    sample_node_ids,
    leaf_node_ids: list[int],
    reg_lambda: float,
    reg_alpha: float = 0.0,
) -> dict[int, float]:
    """GPU implementation of leaf value computation."""
    # For simplicity, copy to CPU and compute there
    # Future optimization: use GPU reduction kernel
    grad_cpu = grad.copy_to_host()
    hess_cpu = hess.copy_to_host()
    sample_node_ids_cpu = sample_node_ids.copy_to_host()
    
    return _compute_leaf_values_cpu(
        grad_cpu, hess_cpu, sample_node_ids_cpu, leaf_node_ids, reg_lambda, reg_alpha
    )


# =============================================================================
# Primitive: Initialize Sample Node IDs
# =============================================================================

def init_sample_node_ids(n_samples: int, device: str = "auto") -> NDArray:
    """Initialize sample node IDs to root node (node 0).
    
    Args:
        n_samples: Number of samples
        device: "cpu", "cuda", or "auto" (detect from backend)
        
    Returns:
        Array of zeros with shape (n_samples,), dtype int32
    """
    if device == "auto":
        device = "cuda" if is_cuda() else "cpu"
    
    if device == "cuda":
        from numba import cuda
        arr = cuda.device_array(n_samples, dtype=np.int32)
        # Zero-fill using kernel
        threads = 256
        blocks = (n_samples + threads - 1) // threads
        _zero_int_kernel[blocks, threads](arr, n_samples)
        return arr
    else:
        return np.zeros(n_samples, dtype=np.int32)


_zero_int_kernel = None

def _init_zero_kernel():
    global _zero_int_kernel
    if _zero_int_kernel is not None:
        return
    
    from numba import cuda
    
    @cuda.jit
    def kernel(arr, n):
        idx = cuda.grid(1)
        if idx < n:
            arr[idx] = 0
    
    _zero_int_kernel = kernel


if is_cuda():
    try:
        _init_zero_kernel()
    except Exception:
        pass


# =============================================================================
# Utility: Get Active Nodes
# =============================================================================

def get_nodes_at_depth(depth: int) -> list[int]:
    """Get node IDs at a given depth (for level-wise growth).
    
    Uses binary tree indexing where:
    - Root is node 0 (depth 0)
    - Depth d has nodes [2^d - 1, 2^(d+1) - 1)
    
    Args:
        depth: Tree depth (0 = root)
        
    Returns:
        List of node IDs at that depth
    """
    start = 2**depth - 1
    end = 2**(depth + 1) - 1
    return list(range(start, end))


def get_children(node_id: int) -> tuple[int, int]:
    """Get child node IDs for a given node.
    
    Uses binary tree indexing:
    - Left child: 2 * node_id + 1
    - Right child: 2 * node_id + 2
    
    Args:
        node_id: Parent node ID
        
    Returns:
        (left_child_id, right_child_id)
    """
    return (2 * node_id + 1, 2 * node_id + 2)


def get_parent(node_id: int) -> int:
    """Get parent node ID.
    
    Args:
        node_id: Child node ID (must be > 0)
        
    Returns:
        Parent node ID
    """
    if node_id <= 0:
        raise ValueError("Root node has no parent")
    return (node_id - 1) // 2
