"""Tree structure and fitting for OpenBoost.

Phase 8.3+8.4: Refactored to use growth strategies from _growth.py.
The main `fit_tree()` function now uses composable primitives and strategies.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .._array import BinnedArray, as_numba_array
from .._backends import is_cuda
from ._growth import (
    GrowthConfig,
    GrowthStrategy,
    SymmetricGrowth,
    TreeStructure,
    get_growth_strategy,
)

# Legacy imports for backward compatibility with internal code
from ._histogram import build_histogram, subtract_histogram
from ._split import compute_leaf_value, find_best_split

if TYPE_CHECKING:
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    from numpy.typing import NDArray

    from .._batch import ConfigBatch
    from .._loss import LossFunction


# =============================================================================
# Legacy Tree Classes (kept for backward compatibility)
# =============================================================================

@dataclass
class TreeNode:
    """A node in the decision tree (legacy)."""
    feature: int = -1
    threshold: int = -1
    value: float = 0.0
    left: int = -1
    right: int = -1
    n_samples: int = 0
    sum_grad: float = 0.0
    sum_hess: float = 0.0
    
    @property
    def is_leaf(self) -> bool:
        return self.left == -1


@dataclass
class Tree:
    """A decision tree for gradient boosting.
    
    Uses array-of-structs layout for simplicity.
    Can be converted to struct-of-arrays for prediction kernels.
    
    Supports both CPU and GPU array storage for efficient training:
    - GPU arrays (_*_gpu) are used during GPU training to avoid transfers
    - CPU arrays (_*) are lazily populated when needed (serialization, CPU prediction)
    """
    nodes: list[TreeNode] = field(default_factory=list)
    n_features: int = 0
    
    # Cached CPU arrays for prediction/serialization
    _features: NDArray | None = field(default=None, repr=False)
    _thresholds: NDArray | None = field(default=None, repr=False)
    _values: NDArray | None = field(default=None, repr=False)
    _left: NDArray | None = field(default=None, repr=False)
    _right: NDArray | None = field(default=None, repr=False)
    
    # GPU arrays for fast GPU training (Phase 5.1)
    _features_gpu: DeviceNDArray | None = field(default=None, repr=False)
    _thresholds_gpu: DeviceNDArray | None = field(default=None, repr=False)
    _values_gpu: DeviceNDArray | None = field(default=None, repr=False)
    _left_gpu: DeviceNDArray | None = field(default=None, repr=False)
    _right_gpu: DeviceNDArray | None = field(default=None, repr=False)
    
    @property
    def on_gpu(self) -> bool:
        """Check if tree arrays are stored on GPU."""
        return self._features_gpu is not None
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    @property
    def depth(self) -> int:
        """Compute tree depth (number of splits from root to deepest leaf).
        
        A tree with just a root leaf has depth 0.
        A tree with one split (root + 2 leaves) has depth 1.
        """
        if not self.nodes:
            return 0
        return self._node_depth(0)
    
    def _node_depth(self, idx: int) -> int:
        node = self.nodes[idx]
        if node.is_leaf:
            return 0  # Leaf contributes 0 to depth
        return 1 + max(
            self._node_depth(node.left),
            self._node_depth(node.right)
        )
    
    @property
    def n_leaves(self) -> int:
        return sum(1 for n in self.nodes if n.is_leaf)
    
    def to_arrays(self) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """Convert to struct-of-arrays for prediction kernels (CPU).
        
        If arrays are on GPU, lazily copies them to CPU.
        
        Returns:
            features: (n_nodes,) int32
            thresholds: (n_nodes,) uint8 (leaf nodes use 0, doesn't matter)
            values: (n_nodes,) float32
            left: (n_nodes,) int32
            right: (n_nodes,) int32
        """
        if self._features is not None:
            return self._features, self._thresholds, self._values, self._left, self._right
        
        # If we have GPU arrays, copy them to CPU (lazy transfer)
        if self.on_gpu:
            self._features = self._features_gpu.copy_to_host()
            self._thresholds = self._thresholds_gpu.copy_to_host()
            self._values = self._values_gpu.copy_to_host()
            self._left = self._left_gpu.copy_to_host()
            self._right = self._right_gpu.copy_to_host()
            return self._features, self._thresholds, self._values, self._left, self._right
        
        # Build from nodes (CPU path)
        features = np.array([node.feature for node in self.nodes], dtype=np.int32)
        # For leaf nodes (threshold=-1), use 0 since we won't use it anyway
        thresholds = np.array([max(0, node.threshold) for node in self.nodes], dtype=np.uint8)
        values = np.array([node.value for node in self.nodes], dtype=np.float32)
        left = np.array([node.left for node in self.nodes], dtype=np.int32)
        right = np.array([node.right for node in self.nodes], dtype=np.int32)
        
        # Cache for reuse
        self._features = features
        self._thresholds = thresholds
        self._values = values
        self._left = left
        self._right = right
        
        return features, thresholds, values, left, right
    
    def __getstate__(self) -> dict:
        # DeviceNDArray handles cannot be deep-copied or pickled (Cython objects
        # with non-trivial __cinit__), which breaks EarlyStopping's restore_best
        # snapshot and pickling of GPU-trained models. Materialize the CPU
        # arrays first, then drop the device handles; they re-upload lazily.
        if self.on_gpu:
            self.to_arrays()
        state = self.__dict__.copy()
        for key in ('_features_gpu', '_thresholds_gpu', '_values_gpu',
                    '_left_gpu', '_right_gpu'):
            state[key] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def to_gpu_arrays(self):
        """Get GPU arrays for fast GPU prediction.

        Returns arrays already on GPU if available, otherwise transfers from CPU.

        Returns:
            features_gpu, thresholds_gpu, values_gpu, left_gpu, right_gpu
        """
        if self.on_gpu:
            return (self._features_gpu, self._thresholds_gpu, self._values_gpu,
                    self._left_gpu, self._right_gpu)
        
        # Transfer CPU arrays to GPU
        from numba import cuda
        
        # Ensure CPU arrays exist
        self.to_arrays()
        
        # Transfer to GPU and cache
        self._features_gpu = cuda.to_device(self._features)
        self._thresholds_gpu = cuda.to_device(self._thresholds)
        self._values_gpu = cuda.to_device(self._values)
        self._left_gpu = cuda.to_device(self._left)
        self._right_gpu = cuda.to_device(self._right)
        
        return (self._features_gpu, self._thresholds_gpu, self._values_gpu,
                self._left_gpu, self._right_gpu)
    
    def __call__(self, X: BinnedArray | NDArray) -> NDArray:
        """Predict using this tree.
        
        Args:
            X: BinnedArray or binned data array (n_features, n_samples)
            
        Returns:
            predictions: Shape (n_samples,), float32
        """
        return predict_tree(self, X)


# =============================================================================
# GPU-Native Tree Building (Phase 3.2+)
# =============================================================================

def fit_tree_gpu_native(
    X: BinnedArray | NDArray,
    grad: NDArray,
    hess: NDArray,
    *,
    max_depth: int = 6,
    min_child_weight: float = 1.0,
    reg_lambda: float = 1.0,
    min_gain: float = 0.0,
    pred_gpu=None,
    learning_rate: float = 0.0,
    const_hess: float = 0.0,
) -> Tree:
    """Fit a tree using GPU-native building (Phase 3.2).
    
    This is the fastest tree building method. It:
    - Builds the entire tree on GPU with O(depth) kernel launches
    - Has ZERO copy_to_host() during building
    - Uses level-wise parallel histogram building and split finding
    
    Args:
        X: Binned feature data (BinnedArray from ob.array(), or raw binned array)
        grad: Gradient vector, shape (n_samples,), float32
        hess: Hessian vector, shape (n_samples,), float32
        max_depth: Maximum tree depth
        min_child_weight: Minimum sum of hessian in a leaf
        reg_lambda: L2 regularization
        min_gain: Minimum gain to make a split
        
    Returns:
        Fitted Tree object (legacy Tree, not TreeStructure).

    Note:
        This intentionally returns a ``Tree`` (legacy class) rather than
        ``TreeStructure`` (returned by ``fit_tree``).  ``Tree`` keeps GPU
        arrays directly for zero-copy training, while ``TreeStructure`` is
        a struct-of-arrays representation used by the growth strategies.
        If you need a ``TreeStructure``, use ``fit_tree()`` instead.
    """
    if not is_cuda():
        # Fall back to CPU recursive implementation
        return _fit_tree_cpu(X, grad, hess, max_depth=max_depth,
                            min_child_weight=min_child_weight,
                            reg_lambda=reg_lambda, min_gain=min_gain)
    
    # Handle BinnedArray
    if isinstance(X, BinnedArray):
        binned = X.data
        n_features = X.n_features
    else:
        binned = X
        n_features = binned.shape[0]
    
    # Ensure data is on GPU
    binned = as_numba_array(binned)
    grad = as_numba_array(grad)
    hess = as_numba_array(hess)
    
    # Build tree on GPU
    from .._backends._cuda import build_tree_gpu_native
    
    node_features, node_thresholds, node_values, node_left, node_right = build_tree_gpu_native(
        binned, grad, hess,
        max_depth=max_depth,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        min_gain=min_gain,
        pred_gpu=pred_gpu,
        learning_rate=learning_rate,
        const_hess=const_hess,
    )
    
    # Phase 5.1: Keep arrays on GPU for fast training
    # CPU arrays and TreeNode objects are lazily created in to_arrays() if needed
    tree = Tree(n_features=n_features)
    
    # Store GPU arrays directly (NO copy to host!)
    tree._features_gpu = node_features
    tree._thresholds_gpu = node_thresholds
    tree._values_gpu = node_values
    tree._left_gpu = node_left
    tree._right_gpu = node_right
    
    return tree


def fit_tree(
    X: BinnedArray | NDArray,
    grad: NDArray,
    hess: NDArray,
    *,
    max_depth: int = 6,
    min_child_weight: float = 1.0,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    min_gain: float = 0.0,
    gamma: float | None = None,  # Alias for min_gain (XGBoost compat)
    growth: str | GrowthStrategy = "levelwise",
    max_leaves: int | None = None,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
) -> TreeStructure:
    """Fit a single gradient boosting tree.
    
    This is the core function of OpenBoost. It builds a tree using the
    specified growth strategy and returns a TreeStructure that can be
    used for prediction.
    
    Phase 8: Uses composable growth strategies from _growth.py.
    Phase 11: Added reg_alpha, subsample, colsample_bytree.
    Phase 14: Handles missing values automatically via BinnedArray.has_missing.
    
    Args:
        X: Binned feature data (BinnedArray from ob.array(), or raw binned array)
           Missing values (NaN in original data) are encoded as bin 255.
        grad: Gradient vector, shape (n_samples,), float32
        hess: Hessian vector, shape (n_samples,), float32
        max_depth: Maximum tree depth
        min_child_weight: Minimum sum of hessian in a leaf
        reg_lambda: L2 regularization on leaf values
        reg_alpha: L1 regularization on leaf values (Phase 11)
        min_gain: Minimum gain to make a split
        gamma: Alias for min_gain (XGBoost compatibility)
        growth: Growth strategy - "levelwise", "leafwise", "symmetric", 
                or a GrowthStrategy instance
        max_leaves: Maximum leaves (for leafwise growth)
        subsample: Row sampling ratio (0.0-1.0), 1.0 = no sampling (Phase 11)
        colsample_bytree: Column sampling ratio (0.0-1.0), 1.0 = no sampling (Phase 11)
        
    Returns:
        TreeStructure that can predict via tree.predict(X) or tree(X)
        
    Example:
        >>> import openboost as ob
        >>> import numpy as np
        >>> 
        >>> # Missing values handled automatically
        >>> X_train = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])
        >>> X_binned = ob.array(X_train)
        >>> pred = np.zeros(3, dtype=np.float32)
        >>> 
        >>> for round in range(100):
        ...     grad = 2 * (pred - y)  # MSE gradient
        ...     hess = np.ones_like(grad) * 2
        ...     tree = ob.fit_tree(X_binned, grad, hess)
        ...     pred = pred + 0.1 * tree.predict(X_binned)
        
        >>> # Use leaf-wise growth (LightGBM style)
        >>> tree = ob.fit_tree(X_binned, grad, hess, growth="leafwise", max_leaves=32)
        
        >>> # Use symmetric growth (CatBoost style)  
        >>> tree = ob.fit_tree(X_binned, grad, hess, growth="symmetric")
        
        >>> # Stochastic gradient boosting (Phase 11)
        >>> tree = ob.fit_tree(X_binned, grad, hess, subsample=0.8, colsample_bytree=0.8)
    """
    # Handle gamma alias
    if gamma is not None:
        min_gain = gamma
    
    # Extract binned data and missing/categorical info
    has_missing = None
    is_categorical = None
    n_categories = None
    if isinstance(X, BinnedArray):
        binned = X.data
        n_features = X.n_features
        n_samples = X.n_samples
        # Phase 14: Get missing value info if available
        if hasattr(X, 'has_missing') and len(X.has_missing) > 0:
            has_missing = X.has_missing
        # Phase 14.3: Get categorical info if available
        if hasattr(X, 'is_categorical') and len(X.is_categorical) > 0:
            is_categorical = X.is_categorical
        if hasattr(X, 'n_categories') and len(X.n_categories) > 0:
            n_categories = X.n_categories
    else:
        binned = X
        n_features, n_samples = binned.shape
    
    # Convert grad/hess to appropriate format
    grad = as_numba_array(grad)
    hess = as_numba_array(hess)
    
    # Validate shapes
    if grad.shape[0] != n_samples:
        raise ValueError(f"grad has {grad.shape[0]} samples, expected {n_samples}")
    if hess.shape[0] != n_samples:
        raise ValueError(f"hess has {hess.shape[0]} samples, expected {n_samples}")
    
    # Apply row subsampling (Phase 11)
    if subsample < 1.0:
        n_subsample = int(n_samples * subsample)
        if n_subsample < 1:
            n_subsample = 1
        subsample_indices = np.random.choice(n_samples, n_subsample, replace=False)
        subsample_indices = np.sort(subsample_indices)  # Keep order for cache efficiency
        # Create mask for sampling
        subsample_mask = np.zeros(n_samples, dtype=np.bool_)
        subsample_mask[subsample_indices] = True
    else:
        subsample_mask = None
    
    # Get growth strategy
    strategy = get_growth_strategy(growth) if isinstance(growth, str) else growth
    
    # Build config
    config = GrowthConfig(
        max_depth=max_depth,
        max_leaves=max_leaves,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        min_gain=min_gain,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
    )
    
    # Apply subsampling to gradients if needed
    if subsample_mask is not None:
        # Zero out gradients for non-sampled rows.
        # Design choice: we zero the grad/hess for non-sampled rows rather than
        # physically removing them. This is correct (zero-weight samples don't
        # affect the Newton step) but doesn't provide the performance benefit of
        # true subsampling since histogram building still iterates over all samples.
        # TODO: For a performance improvement, consider physically filtering to
        # only the sampled rows before histogram building.
        # Handle both CPU (numpy) and GPU (DeviceNDArray) arrays
        if hasattr(grad, '__cuda_array_interface__'):
            # GPU path: copy to host, modify, copy back
            from numba import cuda
            grad_host = grad.copy_to_host()
            hess_host = hess.copy_to_host()
            grad_host[~subsample_mask] = 0.0
            hess_host[~subsample_mask] = 0.0
            grad_sampled = cuda.to_device(grad_host)
            hess_sampled = cuda.to_device(hess_host)
        else:
            # CPU path
            grad_sampled = grad.copy()
            hess_sampled = hess.copy()
            grad_sampled[~subsample_mask] = 0.0
            hess_sampled[~subsample_mask] = 0.0
        # Phase 14/14.3: Pass has_missing and categorical info to growth strategy
        return strategy.grow(
            binned, grad_sampled, hess_sampled, config, 
            has_missing=has_missing,
            is_categorical=is_categorical,
            n_categories=n_categories,
        )
    else:
        # Phase 14/14.3: Pass has_missing and categorical info to growth strategy
        return strategy.grow(
            binned, grad, hess, config,
            has_missing=has_missing,
            is_categorical=is_categorical,
            n_categories=n_categories,
        )


def fit_tree_legacy(
    X: BinnedArray | NDArray,
    grad: NDArray,
    hess: NDArray,
    *,
    max_depth: int = 6,
    min_child_weight: float = 1.0,
    reg_lambda: float = 1.0,
    min_gain: float = 0.0,
) -> Tree:
    """Legacy fit_tree that returns the old Tree class.
    
    Kept for backward compatibility with code that depends on Tree internals.
    For new code, use fit_tree() which returns TreeStructure.
    """
    # Extract binned data
    if isinstance(X, BinnedArray):
        binned = X.data
        n_features = X.n_features
        n_samples = X.n_samples
    else:
        binned = X
        n_features, n_samples = binned.shape
    
    # Convert grad/hess to appropriate format
    grad = as_numba_array(grad)
    hess = as_numba_array(hess)
    
    # Validate shapes
    if grad.shape[0] != n_samples:
        raise ValueError(f"grad has {grad.shape[0]} samples, expected {n_samples}")
    if hess.shape[0] != n_samples:
        raise ValueError(f"hess has {hess.shape[0]} samples, expected {n_samples}")
    
    # Phase 4: Auto-dispatch to GPU-native when data is on GPU
    if is_cuda() and hasattr(binned, '__cuda_array_interface__'):
        return fit_tree_gpu_native(
            X, grad, hess,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            min_gain=min_gain,
        )
    
    # CPU path: use recursive implementation
    tree = Tree(n_features=n_features)
    sample_indices = np.arange(n_samples, dtype=np.int32)
    
    _build_tree_recursive(
        tree=tree,
        binned=binned,
        grad=grad,
        hess=hess,
        sample_indices=sample_indices,
        depth=0,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        min_gain=min_gain,
    )
    
    return tree


def _fit_tree_cpu(
    X: BinnedArray | NDArray,
    grad: NDArray,
    hess: NDArray,
    *,
    max_depth: int = 6,
    min_child_weight: float = 1.0,
    reg_lambda: float = 1.0,
    min_gain: float = 0.0,
) -> Tree:
    """CPU-only tree fitting using recursive implementation.
    
    This is the fallback when GPU is not available.
    Phase 4: Extracted from fit_tree for clarity.
    """
    # Extract binned data
    if isinstance(X, BinnedArray):
        binned = X.data
        n_features = X.n_features
        n_samples = X.n_samples
    else:
        binned = X
        n_features, n_samples = binned.shape
    
    # Ensure CPU arrays
    binned = np.asarray(binned)
    grad = np.asarray(grad, dtype=np.float32)
    hess = np.asarray(hess, dtype=np.float32)
    
    tree = Tree(n_features=n_features)
    sample_indices = np.arange(n_samples, dtype=np.int32)
    
    _build_tree_recursive(
        tree=tree,
        binned=binned,
        grad=grad,
        hess=hess,
        sample_indices=sample_indices,
        depth=0,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        min_gain=min_gain,
    )
    
    return tree


def _build_tree_recursive(
    tree: Tree,
    binned: NDArray,
    grad: NDArray,
    hess: NDArray,
    sample_indices: NDArray,
    depth: int,
    max_depth: int,
    min_child_weight: float,
    reg_lambda: float,
    min_gain: float,
    parent_hist_grad: NDArray | None = None,
    parent_hist_hess: NDArray | None = None,
    sibling_hist_grad: NDArray | None = None,
    sibling_hist_hess: NDArray | None = None,
) -> int:
    """Recursively build tree nodes.
    
    Returns the index of the created node.
    
    Phase 3: Uses histogram subtraction for ~2x faster histogram building.
    - If sibling_hist provided: compute this node's histogram via subtraction
    - Otherwise: build histogram directly
    - Pass histogram to children for subtraction trick
    """
    n_samples = sample_indices.shape[0]
    
    # Early exit for trivial cases (before building histogram)
    if depth >= max_depth or n_samples < 2:
        # Need to compute sums for leaf value
        if sibling_hist_grad is not None and parent_hist_grad is not None:
            # Use subtraction to get sums
            hist_grad, hist_hess = subtract_histogram(
                parent_hist_grad, parent_hist_hess,
                sibling_hist_grad, sibling_hist_hess
            )
            if is_cuda() and hasattr(hist_grad, '__cuda_array_interface__'):
                sum_grad = float(np.sum(hist_grad[0].copy_to_host()))
                sum_hess = float(np.sum(hist_hess[0].copy_to_host()))
            else:
                sum_grad = float(np.sum(hist_grad[0]))
                sum_hess = float(np.sum(hist_hess[0]))
        elif is_cuda() and hasattr(grad, '__cuda_array_interface__'):
            from .._backends._cuda import reduce_sum_indexed_cuda
            sum_grad = float(reduce_sum_indexed_cuda(grad, sample_indices).copy_to_host()[0])
            sum_hess = float(reduce_sum_indexed_cuda(hess, sample_indices).copy_to_host()[0])
        else:
            sample_indices_cpu = np.asarray(sample_indices)
            sum_grad = float(np.sum(grad[sample_indices_cpu]))
            sum_hess = float(np.sum(hess[sample_indices_cpu]))
        
        node_idx = len(tree.nodes)
        node = TreeNode(n_samples=n_samples, sum_grad=sum_grad, sum_hess=sum_hess)
        node.value = compute_leaf_value(sum_grad, sum_hess, reg_lambda)
        tree.nodes.append(node)
        return node_idx
    
    # Build or compute histogram
    if sibling_hist_grad is not None and parent_hist_grad is not None:
        # Phase 3: Use subtraction trick (O(n_features * 256) instead of O(n_features * n_samples))
        hist_grad, hist_hess = subtract_histogram(
            parent_hist_grad, parent_hist_hess,
            sibling_hist_grad, sibling_hist_hess
        )
    else:
        # Build histogram directly (root node, or fallback)
        hist_grad, hist_hess = build_histogram(binned, grad, hess, sample_indices)
    
    # Get sum_grad/sum_hess from histogram (sum across all bins for any feature)
    if is_cuda() and hasattr(hist_grad, '__cuda_array_interface__'):
        hist_grad_cpu = hist_grad[0].copy_to_host()  # Shape (256,)
        hist_hess_cpu = hist_hess[0].copy_to_host()  # Shape (256,)
        sum_grad = float(np.sum(hist_grad_cpu))
        sum_hess = float(np.sum(hist_hess_cpu))
    else:
        sum_grad = float(np.sum(hist_grad[0]))
        sum_hess = float(np.sum(hist_hess[0]))
    
    # Create node
    node_idx = len(tree.nodes)
    node = TreeNode(
        n_samples=n_samples,
        sum_grad=sum_grad,
        sum_hess=sum_hess,
    )
    tree.nodes.append(node)
    
    # Check min_child_weight stopping condition
    if sum_hess < min_child_weight:
        node.value = compute_leaf_value(sum_grad, sum_hess, reg_lambda)
        return node_idx
    
    # Find best split
    split = find_best_split(
        hist_grad, hist_hess,
        sum_grad, sum_hess,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        min_gain=min_gain,
    )
    
    if not split.is_valid:
        # No valid split, make leaf
        node.value = compute_leaf_value(sum_grad, sum_hess, reg_lambda)
        return node_idx
    
    # Split samples
    if is_cuda() and hasattr(binned, '__cuda_array_interface__'):
        from .._backends._cuda import partition_samples_cuda
        
        left_indices, right_indices, n_left, n_right = partition_samples_cuda(
            binned, sample_indices, split.feature, split.threshold
        )
        
        if n_left == 0 or n_right == 0:
            node.value = compute_leaf_value(sum_grad, sum_hess, reg_lambda)
            return node_idx
    else:
        binned_cpu = np.asarray(binned)
        sample_indices_cpu = np.asarray(sample_indices)
        
        feature_values = binned_cpu[split.feature, sample_indices_cpu]
        left_mask = feature_values <= split.threshold
        
        left_indices = sample_indices_cpu[left_mask].astype(np.int32)
        right_indices = sample_indices_cpu[~left_mask].astype(np.int32)
        n_left = len(left_indices)
        n_right = len(right_indices)
        
        if n_left == 0 or n_right == 0:
            node.value = compute_leaf_value(sum_grad, sum_hess, reg_lambda)
            return node_idx
    
    # Set split info
    node.feature = split.feature
    node.threshold = split.threshold
    
    # Phase 3: Histogram subtraction - build only smaller child, subtract for larger
    # This gives ~2x speedup on histogram building
    if n_left <= n_right:
        # Build left (smaller), subtract for right
        left_hist_grad, left_hist_hess = build_histogram(binned, grad, hess, left_indices)
        
        left_idx = _build_tree_recursive(
            tree, binned, grad, hess, left_indices,
            depth + 1, max_depth, min_child_weight, reg_lambda, min_gain,
            parent_hist_grad=hist_grad, parent_hist_hess=hist_hess,
            sibling_hist_grad=None, sibling_hist_hess=None,  # Already built
        )
        # Store left histogram in node for right child's subtraction
        right_idx = _build_tree_recursive(
            tree, binned, grad, hess, right_indices,
            depth + 1, max_depth, min_child_weight, reg_lambda, min_gain,
            parent_hist_grad=hist_grad, parent_hist_hess=hist_hess,
            sibling_hist_grad=left_hist_grad, sibling_hist_hess=left_hist_hess,
        )
    else:
        # Build right (smaller), subtract for left
        right_hist_grad, right_hist_hess = build_histogram(binned, grad, hess, right_indices)
        
        left_idx = _build_tree_recursive(
            tree, binned, grad, hess, left_indices,
            depth + 1, max_depth, min_child_weight, reg_lambda, min_gain,
            parent_hist_grad=hist_grad, parent_hist_hess=hist_hess,
            sibling_hist_grad=right_hist_grad, sibling_hist_hess=right_hist_hess,
        )
        right_idx = _build_tree_recursive(
            tree, binned, grad, hess, right_indices,
            depth + 1, max_depth, min_child_weight, reg_lambda, min_gain,
            parent_hist_grad=hist_grad, parent_hist_hess=hist_hess,
            sibling_hist_grad=None, sibling_hist_hess=None,  # Already built
        )
    
    # Update node with children indices
    tree.nodes[node_idx].left = left_idx
    tree.nodes[node_idx].right = right_idx
    
    return node_idx


def predict_tree(tree: Tree, X: BinnedArray | NDArray) -> NDArray:
    """Predict using a fitted tree.
    
    Args:
        tree: Fitted Tree object
        X: BinnedArray or binned data (n_features, n_samples)
        
    Returns:
        predictions: Shape (n_samples,), float32
    """
    # Get binned data
    if isinstance(X, BinnedArray):
        binned = X.data
        device = X.device
    else:
        binned = X
        device = "cuda" if is_cuda() and hasattr(binned, '__cuda_array_interface__') else "cpu"
    
    # Convert tree to arrays
    features, thresholds, values, left, right = tree.to_arrays()
    
    # Dispatch to backend
    if device == "cuda" and is_cuda():
        from .._backends._cuda import predict_cuda, to_device
        return predict_cuda(
            binned,
            to_device(features),
            to_device(thresholds),
            to_device(values),
            to_device(left),
            to_device(right),
        )
    else:
        from .._backends._cpu import predict_cpu
        binned_cpu = binned.copy_to_host() if hasattr(binned, 'copy_to_host') else np.asarray(binned)
        return predict_cpu(binned_cpu, features, thresholds, values, left, right)


# =============================================================================
# Batch Training (Phase 2 P2)
# =============================================================================

def fit_trees_batch(
    X: BinnedArray | NDArray,
    grad: NDArray | None = None,
    hess: NDArray | None = None,
    configs: ConfigBatch | None = None,
    *,
    y: NDArray | None = None,
    loss: str | LossFunction = "mse",
    min_gain: float = 0.0,
) -> list[list[Tree]]:
    """Fit multiple boosted tree ensembles while sharing binned input data.
    
    This function is the correctness reference for train-many optimization.
    Configurations currently train sequentially, but expensive input binning is
    shared. Future GPU fusion can optimize this path without changing its results.
    
    Args:
        X: Binned feature data (BinnedArray from ob.array())
        grad: Initial gradient vector for the legacy one-round API.
        hess: Initial hessian vector for the legacy one-round API.
        configs: ConfigBatch with hyperparameter configurations
        y: Training targets. Required when ``n_rounds`` is greater than one.
        loss: Built-in loss name or gradient/hessian callable.
        min_gain: Minimum gain to make a split
        
    Returns:
        List of tree lists, one per configuration.
        ``trees[config_idx][round_idx]`` gives the corresponding round's tree.
        
    Example:
        >>> import openboost as ob
        >>> 
        >>> # Create hyperparameter grid
        >>> configs = ob.ConfigBatch.from_grid(
        ...     max_depth=[4, 6, 8],
        ...     reg_lambda=[0.1, 1.0, 10.0],
        ...     learning_rate=[0.1],
        ...     n_rounds=100,
        ... )
        >>> 
        >>> # Bin data once
        >>> X_binned = ob.array(X_train)
        >>> 
        >>> # Train all configs against the same target
        >>> all_trees = ob.fit_trees_batch(X_binned, configs=configs, y=y_train)
        >>> 
        >>> # all_trees[0] contains trees for first config, etc.
    """
    from .._batch import BatchTrainingState, ConfigBatch
    from .._loss import get_loss_function
    
    if not isinstance(configs, ConfigBatch):
        raise TypeError(f"configs must be ConfigBatch, got {type(configs)}")
    
    # Extract binned data
    if isinstance(X, BinnedArray):
        binned = X.data
        n_samples = X.n_samples
    else:
        binned = X
        _, n_samples = binned.shape

    if y is None:
        if grad is None or hess is None:
            raise ValueError("Provide y for train-many fitting, or grad and hess for one round")
        if configs.n_rounds != 1:
            raise ValueError("y is required when configs.n_rounds is greater than one")
        initial_grad = as_numba_array(grad)
        initial_hess = as_numba_array(hess)
        loss_fn = None
    else:
        if hasattr(y, "copy_to_host"):
            y = y.copy_to_host()
        y = np.asarray(y, dtype=np.float32)
        if y.ndim != 1 or len(y) != n_samples:
            raise ValueError(f"y must have shape ({n_samples},), got {y.shape}")
        initial_grad = None
        initial_hess = None
        loss_fn = get_loss_function(loss)
    
    # Initialize training state
    state = BatchTrainingState.create(configs.n_configs, n_samples)
    return _fit_trees_batch_reference(
        binned,
        configs,
        state,
        min_gain,
        n_samples,
        y=y,
        loss_fn=loss_fn,
        initial_grad=initial_grad,
        initial_hess=initial_hess,
    )


def _fit_trees_batch_reference(
    binned: NDArray,
    configs,
    state,
    min_gain: float,
    n_samples: int,
    *,
    y: NDArray | None,
    loss_fn,
    initial_grad: NDArray | None,
    initial_hess: NDArray | None,
) -> list[list[Tree]]:
    """Correct sequential reference used by CPU and CUDA tree builders."""
    n_configs = configs.n_configs
    n_rounds = configs.n_rounds
    
    # Train each config sequentially
    for config_idx in range(n_configs):
        config = configs[config_idx]
        pred = np.zeros(n_samples, dtype=np.float32)
        
        for _round_idx in range(n_rounds):
            if y is None:
                round_grad, round_hess = initial_grad, initial_hess
            else:
                round_grad, round_hess = loss_fn(pred, y)
                round_grad = as_numba_array(round_grad)
                round_hess = as_numba_array(round_hess)

            tree = fit_tree(
                binned,
                round_grad,
                round_hess,
                max_depth=config['max_depth'],
                min_child_weight=config['min_child_weight'],
                reg_lambda=config['reg_lambda'],
                min_gain=min_gain,
            )
            state.trees[config_idx].append(tree)
            
            # Update predictions
            tree_pred = tree(binned)
            if hasattr(tree_pred, 'copy_to_host'):
                tree_pred = tree_pred.copy_to_host()
            pred = pred + config['learning_rate'] * tree_pred
            state.predictions[config_idx] = pred
    
    return state.trees


# =============================================================================
# Phase 3.4: Symmetric (Oblivious) Trees
# =============================================================================

@dataclass
class SymmetricTree:
    """A symmetric (oblivious) decision tree.
    
    All nodes at the same depth use the SAME split (feature + threshold).
    This enables massive GPU parallelization.
    
    Structure:
    - level_features[d]: Feature used at depth d
    - level_thresholds[d]: Threshold used at depth d  
    - leaf_values[i]: Value for leaf i (2^max_depth leaves)
    
    Prediction:
        leaf_idx = 0
        for d in range(max_depth):
            if X[level_features[d]] > level_thresholds[d]:
                leaf_idx = 2 * leaf_idx + 1
            else:
                leaf_idx = 2 * leaf_idx
        return leaf_values[leaf_idx]
    """
    level_features: NDArray    # (max_depth,) int32 - feature at each level
    level_thresholds: NDArray  # (max_depth,) uint8 - threshold at each level
    leaf_values: NDArray       # (2^max_depth,) float32 - leaf values
    max_depth: int
    n_features: int
    
    # Cached GPU arrays
    _level_features_gpu: NDArray | None = field(default=None, repr=False)
    _level_thresholds_gpu: NDArray | None = field(default=None, repr=False)
    _leaf_values_gpu: NDArray | None = field(default=None, repr=False)

    def __getstate__(self) -> dict:
        # Device handles cannot be deep-copied/pickled; host arrays are the
        # source of truth, caches re-upload lazily (see Tree.__getstate__).
        state = self.__dict__.copy()
        for key in ('_level_features_gpu', '_level_thresholds_gpu', '_leaf_values_gpu'):
            state[key] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def __call__(self, X: BinnedArray | NDArray) -> NDArray:
        """Predict using this symmetric tree."""
        return predict_symmetric_tree(self, X)
    
    @property
    def n_leaves(self) -> int:
        return 2 ** self.max_depth


def fit_tree_symmetric(
    binned: BinnedArray | NDArray,
    grad: NDArray,
    hess: NDArray,
    max_depth: int = 6,
    min_child_weight: float = 1.0,
    reg_lambda: float = 1.0,
    min_gain: float = 0.0,
) -> SymmetricTree:
    """Fit a symmetric (oblivious) tree.

    All nodes at the same depth use the same split, chosen by aggregating
    per-leaf gains across all leaves at that depth (CatBoost-style).
    Delegates to ``SymmetricGrowth`` so this entry point and
    ``fit_tree(..., growth='symmetric')`` share one implementation.

    Args:
        binned: BinnedArray or binned data (n_features, n_samples)
        grad: Gradients, shape (n_samples,)
        hess: Hessians, shape (n_samples,)
        max_depth: Maximum tree depth
        min_child_weight: Minimum sum of hessian in child
        reg_lambda: L2 regularization
        min_gain: Minimum gain to make a split

    Returns:
        SymmetricTree with level-wise splits
    """
    # Extract raw data
    if isinstance(binned, BinnedArray):
        binned_data = binned.data
        n_features = binned.n_features
    else:
        binned_data = binned
        n_features = binned.shape[0]

    config = GrowthConfig(
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        min_gain=min_gain,
    )
    # No categorical info here: this API treats all features as ordinal,
    # which keeps SymmetricTree's ordinal-only prediction consistent.
    ts = SymmetricGrowth().grow(binned_data, grad, hess, config)

    n_leaves = 2 ** max_depth
    level_features = np.full(max_depth, -1, dtype=np.int32)
    level_thresholds = np.zeros(max_depth, dtype=np.uint8)
    leaf_values = np.zeros(n_leaves, dtype=np.float32)

    d = ts.depth
    if d > 0:
        level_features[:d] = ts.level_features
        level_thresholds[:d] = ts.level_thresholds.astype(np.uint8)
    # SymmetricTree prediction stops at the first feature < 0, so leaf ids
    # stay in [0, 2^d); map the TreeStructure leaf slots onto that range.
    leaf_start = 2 ** d - 1
    leaf_values[:2 ** d] = ts.leaf_values_array[leaf_start:leaf_start + 2 ** d]

    return SymmetricTree(
        level_features=level_features,
        level_thresholds=level_thresholds,
        leaf_values=leaf_values,
        max_depth=max_depth,
        n_features=n_features,
    )


def predict_symmetric_tree(tree: SymmetricTree, X: BinnedArray | NDArray) -> NDArray:
    """Predict using a symmetric tree.
    
    Prediction is just bit operations - very fast!
    """
    binned = X.data if isinstance(X, BinnedArray) else X
    
    use_gpu = is_cuda() and hasattr(binned, '__cuda_array_interface__')
    
    if use_gpu:
        from .._backends._cuda import predict_symmetric_cuda
        return predict_symmetric_cuda(
            binned,
            tree.level_features,
            tree.level_thresholds,
            tree.leaf_values,
            tree.max_depth,
        )
    else:
        return _predict_symmetric_cpu(
            np.asarray(binned),
            tree.level_features,
            tree.level_thresholds,
            tree.leaf_values,
            tree.max_depth,
        )


def _predict_symmetric_cpu(
    binned: NDArray,
    level_features: NDArray,
    level_thresholds: NDArray,
    leaf_values: NDArray,
    max_depth: int,
) -> NDArray:
    """CPU prediction for symmetric trees."""
    n_samples = binned.shape[1]
    leaf_ids = np.zeros(n_samples, dtype=np.int32)
    
    for depth in range(max_depth):
        feature = level_features[depth]
        if feature < 0:
            break
        threshold = level_thresholds[depth]
        goes_right = binned[feature, :] > threshold
        leaf_ids = 2 * leaf_ids + goes_right.astype(np.int32)
    
    return leaf_values[leaf_ids]


def fit_tree_symmetric_gpu_native(
    binned: BinnedArray | NDArray,
    grad: NDArray,
    hess: NDArray,
    max_depth: int = 6,
    min_child_weight: float = 1.0,
    reg_lambda: float = 1.0,
    min_gain: float = 0.0,
) -> SymmetricTree:
    """Fit symmetric tree using GPU-native implementation.

    Faster than fit_tree_symmetric() as it minimizes CPU-GPU transfers.

    Warning:
        The GPU kernel is currently DISABLED pending a correctness fix:
        ``_build_symmetric_histogram_kernel`` ignores per-leaf sample
        assignments, so every depth rebuilds the identical root histogram
        and the tree repeats one split per level (degenerate). Unless
        ``OPENBOOST_EXPERIMENTAL_SYMMETRIC_GPU=1`` is set, this function
        falls back to the correct (CPU-side) symmetric builder.

    Args:
        binned: BinnedArray or binned data (n_features, n_samples)
        grad: Gradients, shape (n_samples,)
        hess: Hessians, shape (n_samples,)
        max_depth: Maximum tree depth
        min_child_weight: Minimum sum of hessian in child
        reg_lambda: L2 regularization
        min_gain: Minimum gain to make a split

    Returns:
        SymmetricTree
    """
    if not is_cuda():
        return fit_tree_symmetric(
            binned, grad, hess,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            min_gain=min_gain,
        )

    # CRIT-5 gate: the GPU symmetric builder produces degenerate trees because
    # its histogram kernel is not leaf-aware (see _backends/_cuda.py,
    # _build_symmetric_histogram_kernel). Route to the correct CPU builder
    # unless the experimental escape hatch is explicitly enabled.
    if os.environ.get("OPENBOOST_EXPERIMENTAL_SYMMETRIC_GPU") != "1":
        warnings.warn(
            "symmetric GPU builder disabled pending correctness fix: its "
            "histogram kernel ignores per-leaf sample assignments, so every "
            "depth repeats the same split. Using the CPU symmetric builder "
            "instead. Set OPENBOOST_EXPERIMENTAL_SYMMETRIC_GPU=1 to force "
            "the (known-broken) GPU kernel.",
            UserWarning,
            stacklevel=2,
        )
        return fit_tree_symmetric(
            binned, grad, hess,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            min_gain=min_gain,
        )


    # Extract raw data
    if isinstance(binned, BinnedArray):
        binned_data = binned.data
        n_features = binned.n_features
    else:
        binned_data = binned
        n_features = binned.shape[0]
    
    # Ensure data is on GPU
    from numba import cuda
    if not hasattr(binned_data, '__cuda_array_interface__'):
        binned_data = cuda.to_device(np.ascontiguousarray(binned_data))
    if not hasattr(grad, '__cuda_array_interface__'):
        grad = cuda.to_device(np.ascontiguousarray(grad, dtype=np.float32))
    if not hasattr(hess, '__cuda_array_interface__'):
        hess = cuda.to_device(np.ascontiguousarray(hess, dtype=np.float32))
    
    from .._backends._cuda import build_tree_symmetric_gpu_native
    
    level_features_gpu, level_thresholds_gpu, leaf_values_gpu = build_tree_symmetric_gpu_native(
        binned_data, grad, hess,
        max_depth=max_depth,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        min_gain=min_gain,
    )
    
    # Copy results to CPU for tree structure
    level_features = level_features_gpu.copy_to_host().astype(np.int32)
    level_thresholds = level_thresholds_gpu.copy_to_host().astype(np.uint8)
    leaf_values = leaf_values_gpu.copy_to_host().astype(np.float32)
    
    return SymmetricTree(
        level_features=level_features,
        level_thresholds=level_thresholds,
        leaf_values=leaf_values,
        max_depth=max_depth,
        n_features=n_features,
    )
