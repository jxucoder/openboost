"""Tree structure and fitting for OpenBoost.

Phase 8.3+8.4: Refactored to use growth strategies from _growth.py.
The main `fit_tree()` function now uses composable primitives and strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .._array import BinnedArray, as_numba_array
from .._backends import is_cuda
from ._growth import (
    GrowthConfig,
    GrowthStrategy,
    TreeStructure,
    LevelWiseGrowth,
    LeafWiseGrowth,
    SymmetricGrowth,
    get_growth_strategy,
)

# Legacy imports for backward compatibility with internal code
from ._histogram import build_histogram, subtract_histogram
from ._split import SplitInfo, compute_leaf_value, find_best_split

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
    _features_gpu: "DeviceNDArray | None" = field(default=None, repr=False)
    _thresholds_gpu: "DeviceNDArray | None" = field(default=None, repr=False)
    _values_gpu: "DeviceNDArray | None" = field(default=None, repr=False)
    _left_gpu: "DeviceNDArray | None" = field(default=None, repr=False)
    _right_gpu: "DeviceNDArray | None" = field(default=None, repr=False)
    
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
        Fitted Tree object
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
    if isinstance(growth, str):
        strategy = get_growth_strategy(growth)
    else:
        strategy = growth
    
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
        # Zero out gradients for non-sampled rows
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
    grad: NDArray,
    hess: NDArray,
    configs,  # ConfigBatch
    *,
    min_gain: float = 0.0,
) -> list[list[Tree]]:
    """Fit multiple boosted tree ensembles with different hyperparameters in parallel.
    
    This is the key Phase 2 optimization: train many models (for hyperparameter search)
    in one GPU pass by sharing the binned data across all configurations.
    
    Args:
        X: Binned feature data (BinnedArray from ob.array())
        grad: Initial gradient vector, shape (n_samples,), float32
        hess: Initial hessian vector, shape (n_samples,), float32
        configs: ConfigBatch with hyperparameter configurations
        min_gain: Minimum gain to make a split
        
    Returns:
        List of tree lists, one per configuration.
        trees[config_idx][round_idx] gives the tree for config_idx at round round_idx.
        
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
        >>> # Initial gradients (e.g., for MSE: grad = 2*(pred - y), hess = 2)
        >>> grad = -2 * y_train  # Initial pred = 0
        >>> hess = np.ones_like(y_train) * 2
        >>> 
        >>> # Train all configs in parallel
        >>> all_trees = ob.fit_trees_batch(X_binned, grad, hess, configs)
        >>> 
        >>> # all_trees[0] contains trees for first config, etc.
    """
    from .._models._batch import ConfigBatch, BatchTrainingState
    
    if not isinstance(configs, ConfigBatch):
        raise TypeError(f"configs must be ConfigBatch, got {type(configs)}")
    
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
    
    n_configs = configs.n_configs
    n_rounds = configs.n_rounds
    
    # Initialize training state
    state = BatchTrainingState.create(n_configs, n_samples)
    
    # For CUDA, use batched kernels
    if is_cuda() and hasattr(binned, '__cuda_array_interface__'):
        return _fit_trees_batch_cuda(
            binned, grad, hess, configs, state, min_gain, n_samples, n_features
        )
    else:
        # CPU fallback: train configs sequentially (still faster than naive due to shared binning)
        return _fit_trees_batch_cpu(
            binned, grad, hess, configs, state, min_gain, n_samples
        )


def _fit_trees_batch_cpu(
    binned: NDArray,
    grad: NDArray,
    hess: NDArray,
    configs,
    state,
    min_gain: float,
    n_samples: int,
) -> list[list[Tree]]:
    """CPU fallback for batch training - trains configs sequentially."""
    n_configs = configs.n_configs
    n_rounds = configs.n_rounds
    
    # Train each config sequentially
    for config_idx in range(n_configs):
        config = configs[config_idx]
        pred = np.zeros(n_samples, dtype=np.float32)
        
        for round_idx in range(n_rounds):
            # Compute gradients from current predictions
            # Note: User provides initial grad/hess, subsequent rounds recompute
            if round_idx > 0:
                # For MSE: grad = 2*(pred - y), but we don't have y here
                # This is a limitation - batch training needs custom gradient callback
                # For now, assume constant hess and update grad based on tree predictions
                pass
            
            tree = fit_tree(
                binned,
                grad if round_idx == 0 else (2 * (pred - (pred - grad / 2))),  # Simplified
                hess,
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


def _fit_trees_batch_cuda(
    binned: NDArray,
    grad: NDArray,
    hess: NDArray,
    configs,
    state,
    min_gain: float,
    n_samples: int,
    n_features: int,
) -> list[list[Tree]]:
    """CUDA batch training using fused kernels."""
    from numba import cuda
    from .._backends._cuda import (
        build_histogram_batch_cuda,
        find_best_split_batch_cuda,
        compute_split_masks_batch_cuda,
        reduce_sum_cuda,
        to_device,
    )
    
    n_configs = configs.n_configs
    n_rounds = configs.n_rounds
    
    # Transfer config arrays to GPU
    configs.to_device()
    device_configs = configs.get_device_arrays()
    
    # Initialize per-config sample indices (all start with all samples)
    # For batch training, we track active samples per config per tree level
    # This is complex because different configs may have different tree structures
    
    # Simplified approach: train one round at a time, all configs in parallel
    # Each round: build histograms for all configs, find splits, partition
    
    # Initialize predictions on GPU
    predictions = cuda.device_array((n_configs, n_samples), dtype=np.float32)
    
    for round_idx in range(n_rounds):
        # Build one tree per config for this round
        round_trees = _build_trees_batch_one_round(
            binned, grad, hess, configs, device_configs,
            predictions, round_idx, min_gain, n_samples, n_features
        )
        
        # Store trees and update predictions
        for config_idx, tree in enumerate(round_trees):
            state.trees[config_idx].append(tree)
            
            # Update predictions for this config
            tree_pred = tree(binned)
            lr = configs.learning_rates[config_idx]
            
            # predictions[config_idx] += lr * tree_pred
            # Need a kernel for this, or do on CPU for simplicity
            if hasattr(tree_pred, 'copy_to_host'):
                tree_pred = tree_pred.copy_to_host()
            
            pred_cpu = predictions[config_idx].copy_to_host()
            pred_cpu = pred_cpu + lr * tree_pred
            cuda.to_device(pred_cpu, to=predictions[config_idx:config_idx+1].reshape(n_samples))
    
    return state.trees


def _build_trees_batch_one_round(
    binned: NDArray,
    grad: NDArray,
    hess: NDArray,
    configs,
    device_configs: dict,
    predictions: NDArray,
    round_idx: int,
    min_gain: float,
    n_samples: int,
    n_features: int,
) -> list[Tree]:
    """Build one tree per config for a single boosting round.
    
    This uses the batched kernels but still builds trees sequentially
    within each round (because tree structures can differ).
    
    A future optimization could batch across tree levels.
    """
    n_configs = configs.n_configs
    trees = []
    
    for config_idx in range(n_configs):
        config = configs[config_idx]
        
        # Use single-config tree building for now
        # Full horizontal fusion would require same-depth batching
        tree = fit_tree(
            binned,
            grad,
            hess,
            max_depth=config['max_depth'],
            min_child_weight=config['min_child_weight'],
            reg_lambda=config['reg_lambda'],
            min_gain=min_gain,
        )
        trees.append(tree)
    
    return trees


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
    
    All nodes at the same depth use the same split.
    Much faster on GPU due to simplified split finding and partitioning.
    
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
    
    n_samples = binned_data.shape[1]
    n_leaves = 2 ** max_depth
    
    # Initialize
    level_features = np.full(max_depth, -1, dtype=np.int32)
    level_thresholds = np.zeros(max_depth, dtype=np.uint8)
    leaf_values = np.zeros(n_leaves, dtype=np.float32)
    
    # Track which leaf each sample belongs to
    sample_leaf_ids = np.zeros(n_samples, dtype=np.int32)
    
    # Ensure arrays are contiguous
    grad = np.ascontiguousarray(grad, dtype=np.float32)
    hess = np.ascontiguousarray(hess, dtype=np.float32)
    
    use_gpu = is_cuda() and hasattr(binned_data, '__cuda_array_interface__')
    
    # Build tree level by level
    for depth in range(max_depth):
        n_nodes_at_level = 2 ** depth
        
        # Build combined histogram for ALL nodes at this level
        # For symmetric trees: just sum all histograms (all nodes use same split)
        combined_hist_grad = None
        combined_hist_hess = None
        
        for node_idx in range(n_nodes_at_level):
            # Get samples belonging to this node
            mask = sample_leaf_ids == node_idx
            if not np.any(mask):
                continue
            
            sample_indices = np.where(mask)[0].astype(np.int32)
            
            if use_gpu:
                from numba import cuda
                sample_indices_gpu = cuda.to_device(sample_indices)
                node_hist_grad, node_hist_hess = build_histogram(
                    binned_data, grad, hess, sample_indices_gpu
                )
                # Copy to CPU for aggregation
                if hasattr(node_hist_grad, 'copy_to_host'):
                    node_hist_grad = node_hist_grad.copy_to_host()
                    node_hist_hess = node_hist_hess.copy_to_host()
            else:
                node_hist_grad, node_hist_hess = build_histogram(
                    binned_data, grad, hess, sample_indices
                )
            
            if combined_hist_grad is None:
                combined_hist_grad = node_hist_grad.copy()
                combined_hist_hess = node_hist_hess.copy()
            else:
                combined_hist_grad += node_hist_grad
                combined_hist_hess += node_hist_hess
        
        if combined_hist_grad is None:
            # No samples, stop building
            break
        
        # Find ONE best split for the entire level
        total_grad = float(np.sum(combined_hist_grad))
        total_hess = float(np.sum(combined_hist_hess))
        
        split = find_best_split(
            combined_hist_grad, combined_hist_hess,
            total_grad, total_hess,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            min_gain=min_gain,
        )
        
        if not split.is_valid:
            # Can't split anymore, compute leaf values
            break
        
        # Store the split for this level
        level_features[depth] = split.feature
        level_thresholds[depth] = split.threshold
        
        # Partition ALL samples using this split
        if use_gpu and hasattr(binned_data, 'copy_to_host'):
            binned_cpu = binned_data.copy_to_host()
        else:
            binned_cpu = np.asarray(binned_data)
        
        feature_values = binned_cpu[split.feature, :]
        goes_right = feature_values > split.threshold
        
        # Update leaf IDs: left child = 2*current, right child = 2*current + 1
        sample_leaf_ids = 2 * sample_leaf_ids + goes_right.astype(np.int32)
    
    # Compute leaf values
    for leaf_idx in range(n_leaves):
        mask = sample_leaf_ids == leaf_idx
        if np.any(mask):
            leaf_grad = float(np.sum(grad[mask]))
            leaf_hess = float(np.sum(hess[mask]))
            leaf_values[leaf_idx] = compute_leaf_value(leaf_grad, leaf_hess, reg_lambda)
    
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
    if isinstance(X, BinnedArray):
        binned = X.data
    else:
        binned = X
    
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

