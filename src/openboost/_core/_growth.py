"""Tree growth strategies for OpenBoost.

Phase 8.2: Abstraction for different tree growth approaches.
Phase 9.0: Decoupled leaf values for flexibility (multi-output, distributions, etc.)
Phase 14: Added missing value handling - learns optimal direction for NaN values.

Each strategy uses the primitives from `_primitives.py` but orchestrates
them differently:

- LevelWiseGrowth: Process all nodes at each depth (XGBoost default)
- LeafWiseGrowth: Always split the best leaf (LightGBM style)
- SymmetricGrowth: Same split at each depth (CatBoost style)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from .._backends import is_cuda
from .._array import MISSING_BIN
from ._primitives import (
    NodeHistogram,
    NodeSplit,
    build_node_histograms,
    subtract_histogram,
    find_node_splits,
    partition_samples,
    compute_leaf_values,
    init_sample_node_ids,
    get_nodes_at_depth,
    get_children,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Leaf Values Abstraction (Phase 9.0)
# =============================================================================

@runtime_checkable
class LeafValues(Protocol):
    """Protocol for leaf value storage.
    
    This abstraction allows trees to store different types of values:
    - ScalarLeaves: Standard float per leaf (default)
    - VectorLeaves: Multiple floats per leaf (multi-output)
    - DistributionLeaves: Distribution parameters (uncertainty)
    """
    
    def __getitem__(self, indices: NDArray) -> NDArray:
        """Get values for given leaf indices."""
        ...
    
    def __setitem__(self, index: int, value) -> None:
        """Set value for a leaf."""
        ...
    
    @property
    def shape(self) -> tuple:
        """Shape of the storage array."""
        ...


@dataclass
class ScalarLeaves:
    """Standard scalar leaf values (default).
    
    Each leaf stores a single float value.
    Shape: (n_nodes,)
    """
    _values: NDArray  # (n_nodes,) float32
    
    def __getitem__(self, indices: NDArray) -> NDArray:
        """Get scalar values for indices."""
        return self._values[indices]
    
    def __setitem__(self, index: int, value: float) -> None:
        """Set scalar value."""
        self._values[index] = value
    
    @property
    def shape(self) -> tuple:
        return self._values.shape
    
    @property
    def values(self) -> NDArray:
        """Direct access to underlying array (for backward compatibility)."""
        return self._values
    
    @classmethod
    def zeros(cls, n_nodes: int) -> "ScalarLeaves":
        """Create zero-initialized scalar leaves."""
        return cls(_values=np.zeros(n_nodes, dtype=np.float32))


@dataclass 
class VectorLeaves:
    """Multi-output leaf values.
    
    Each leaf stores a vector of values.
    Shape: (n_nodes, n_outputs)
    """
    _values: NDArray  # (n_nodes, n_outputs) float32
    n_outputs: int
    
    def __getitem__(self, indices: NDArray) -> NDArray:
        """Get vector values for indices. Returns (n_indices, n_outputs)."""
        return self._values[indices]
    
    def __setitem__(self, index: int, value: NDArray) -> None:
        """Set vector value."""
        self._values[index] = value
    
    @property
    def shape(self) -> tuple:
        return self._values.shape
    
    @property
    def values(self) -> NDArray:
        """Direct access to underlying array."""
        return self._values
    
    @classmethod
    def zeros(cls, n_nodes: int, n_outputs: int) -> "VectorLeaves":
        """Create zero-initialized vector leaves."""
        return cls(
            _values=np.zeros((n_nodes, n_outputs), dtype=np.float32),
            n_outputs=n_outputs
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GrowthConfig:
    """Configuration for tree growth.
    
    Args:
        max_depth: Maximum tree depth (for level-wise and symmetric)
        max_leaves: Maximum number of leaves (for leaf-wise)
        min_child_weight: Minimum sum of hessian in each child
        reg_lambda: L2 regularization on leaf values
        reg_alpha: L1 regularization on leaf values (Phase 11)
        min_gain: Minimum gain required to split (alias: gamma)
        subsample: Row sampling ratio per tree (Phase 11)
        colsample_bytree: Column sampling ratio per tree (Phase 11)
    """
    max_depth: int = 6
    max_leaves: int | None = None  # For leaf-wise growth
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0  # Phase 11: L1 regularization
    min_gain: float = 0.0
    subsample: float = 1.0  # Phase 11: row sampling
    colsample_bytree: float = 1.0  # Phase 11: column sampling


# =============================================================================
# Tree Structure (Struct-of-Arrays for GPU efficiency)
# =============================================================================

@dataclass
class TreeStructure:
    """Struct-of-arrays tree representation.
    
    This is the output of all growth strategies. Can be used for prediction
    on both CPU and GPU.
    
    For standard trees:
        - Navigate using left/right children
        - Leaf nodes have left_children[i] == -1
        
    For symmetric trees:
        - Use level_features/level_thresholds for navigation
        - leaf_values has 2^depth entries
        
    Phase 9.0: Leaf values are now abstracted via LeafValues protocol.
    This enables multi-output, distributions, embeddings, etc.
    
    Phase 14: Added missing_go_left for handling NaN values.
    Phase 14.3: Added categorical split support.
    """
    # Tree structure (routing)
    features: NDArray        # (n_nodes,) int32 - split feature (-1 for leaf)
    thresholds: NDArray      # (n_nodes,) int32 - split threshold (ordinal) or split_point (cat)
    left_children: NDArray   # (n_nodes,) int32 - left child (-1 for leaf)
    right_children: NDArray  # (n_nodes,) int32 - right child (-1 for leaf)
    
    # Leaf values (flexible - Phase 9.0)
    # Can be NDArray for backward compat, or LeafValues subclass
    values: NDArray | LeafValues  # Default: (n_nodes,) float32
    
    # Metadata
    n_nodes: int
    depth: int
    n_features: int
    
    # For symmetric trees (optional)
    is_symmetric: bool = False
    level_features: NDArray | None = None    # (depth,) int32
    level_thresholds: NDArray | None = None  # (depth,) int32
    
    # Phase 14: Missing value handling
    missing_go_left: NDArray | None = None   # (n_nodes,) bool - direction for NaN
    
    # Phase 14.3: Categorical split support
    is_categorical_split: NDArray | None = None  # (n_nodes,) bool - True if categorical split
    cat_bitsets: NDArray | None = None           # (n_nodes,) uint64 - bitmask for categories going left
    
    def get_leaf_values(self, leaf_ids: NDArray) -> NDArray:
        """Get leaf values for given leaf IDs.
        
        This method abstracts over different leaf value storage types.
        
        Args:
            leaf_ids: Array of leaf node indices
            
        Returns:
            Values for those leaves (shape depends on leaf type)
        """
        if isinstance(self.values, LeafValues):
            return self.values[leaf_ids]
        # Backward compat: plain NDArray
        return self.values[leaf_ids]
    
    def set_leaf_value(self, leaf_id: int, value) -> None:
        """Set a leaf value.
        
        Args:
            leaf_id: Leaf node index
            value: Value to store (scalar or array depending on leaf type)
        """
        if isinstance(self.values, LeafValues):
            self.values[leaf_id] = value
        else:
            self.values[leaf_id] = value
    
    @property
    def leaf_values_array(self) -> NDArray:
        """Get raw values array (for backward compatibility)."""
        if isinstance(self.values, LeafValues):
            return self.values.values
        return self.values
    
    def predict(self, binned: NDArray) -> NDArray:
        """Predict using this tree.
        
        Args:
            binned: Binned features, shape (n_features, n_samples), uint8
            
        Returns:
            predictions: Shape (n_samples,), float32
        """
        if self.is_symmetric:
            return self._predict_symmetric(binned)
        return self._predict_standard(binned)
    
    def __call__(self, X) -> NDArray:
        """Make tree callable for backward compatibility.
        
        Args:
            X: BinnedArray or binned data array
            
        Returns:
            predictions: Shape (n_samples,), float32
        """
        # Handle BinnedArray
        from .._array import BinnedArray
        if isinstance(X, BinnedArray):
            binned = X.data
        else:
            binned = X
        return self.predict(binned)
    
    def _predict_standard(self, binned: NDArray) -> NDArray:
        """Predict using standard tree traversal."""
        if is_cuda() and hasattr(binned, '__cuda_array_interface__'):
            return self._predict_standard_gpu(binned)
        return self._predict_standard_cpu(binned)
    
    def _predict_standard_cpu(self, binned: NDArray) -> NDArray:
        """CPU prediction for standard trees.
        
        Phase 14: Handles missing values (bin 255) using learned direction.
        Phase 14.3: Handles categorical splits using bitmask membership.
        """
        binned = np.asarray(binned)
        n_samples = binned.shape[1]
        
        # Check if we have missing value handling
        has_missing_handling = self.missing_go_left is not None
        # Phase 14.3: Check for categorical splits
        has_categorical = self.is_categorical_split is not None and self.cat_bitsets is not None
        
        # Get leaf indices for all samples
        leaf_ids = np.empty(n_samples, dtype=np.int32)
        for i in range(n_samples):
            node = 0
            while self.left_children[node] != -1:
                feature = self.features[node]
                threshold = self.thresholds[node]
                bin_value = binned[feature, i]
                
                # Phase 14: Handle missing values first
                if bin_value == MISSING_BIN and has_missing_handling:
                    # Use learned direction for missing values
                    if self.missing_go_left[node]:
                        node = self.left_children[node]
                    else:
                        node = self.right_children[node]
                # Phase 14.3: Handle categorical splits
                elif has_categorical and self.is_categorical_split[node]:
                    # Check bitmask membership: bit[bin_value] == 1 means go left
                    bitset = self.cat_bitsets[node]
                    goes_left = (bitset >> bin_value) & 1
                    if goes_left:
                        node = self.left_children[node]
                    else:
                        node = self.right_children[node]
                # Standard ordinal split
                elif bin_value <= threshold:
                    node = self.left_children[node]
                else:
                    node = self.right_children[node]
            leaf_ids[i] = node
        
        # Get leaf values (works with both NDArray and LeafValues)
        return self.get_leaf_values(leaf_ids)
    
    def _predict_standard_gpu(self, binned) -> NDArray:
        """GPU prediction for standard trees."""
        from numba import cuda
        from .._backends._cuda import predict_cuda, to_device
        
        return predict_cuda(
            binned,
            to_device(self.features),
            to_device(self.thresholds.astype(np.uint8)),
            to_device(self.values),
            to_device(self.left_children),
            to_device(self.right_children),
        )
    
    def _predict_symmetric(self, binned: NDArray) -> NDArray:
        """Predict using symmetric tree (bit operations)."""
        if is_cuda() and hasattr(binned, '__cuda_array_interface__'):
            return self._predict_symmetric_gpu(binned)
        return self._predict_symmetric_cpu(binned)
    
    def _predict_symmetric_cpu(self, binned: NDArray) -> NDArray:
        """CPU prediction for symmetric trees."""
        binned = np.asarray(binned)
        n_samples = binned.shape[1]
        leaf_ids = np.zeros(n_samples, dtype=np.int32)
        
        for d in range(self.depth):
            if self.level_features[d] < 0:
                break
            feature = self.level_features[d]
            threshold = self.level_thresholds[d]
            goes_right = binned[feature, :] > threshold
            leaf_ids = 2 * leaf_ids + goes_right.astype(np.int32)
        
        # Map leaf_ids (0 to 2^depth-1) to actual node indices
        # Leaves start at index 2^depth - 1 in the tree array
        leaf_start = 2**self.depth - 1
        return self.get_leaf_values(leaf_start + leaf_ids)
    
    def _predict_symmetric_gpu(self, binned) -> NDArray:
        """GPU prediction for symmetric trees."""
        from .._backends._cuda import predict_symmetric_cuda
        
        return predict_symmetric_cuda(
            binned,
            self.level_features,
            self.level_thresholds.astype(np.uint8),
            self.values,
            self.depth,
        )


# =============================================================================
# Growth Strategy Base Class
# =============================================================================

class GrowthStrategy(ABC):
    """Abstract base for tree growth strategies."""
    
    @abstractmethod
    def grow(
        self,
        binned: NDArray,
        grad: NDArray,
        hess: NDArray,
        config: GrowthConfig,
        has_missing: NDArray | None = None,
        is_categorical: NDArray | None = None,
        n_categories: NDArray | None = None,
    ) -> TreeStructure:
        """Grow a tree using this strategy.
        
        Args:
            binned: Binned features, shape (n_features, n_samples), uint8
            grad: Gradients, shape (n_samples,), float32
            hess: Hessians, shape (n_samples,), float32
            config: Growth configuration
            has_missing: Boolean array (n_features,) indicating which features
                        have missing values (Phase 14). If None, no missing handling.
            is_categorical: Boolean array (n_features,) indicating categorical features
                           (Phase 14.3). If None, all features are numeric.
            n_categories: Array (n_features,) of category counts (0 for numeric)
            
        Returns:
            TreeStructure ready for prediction
        """
        ...


# =============================================================================
# Level-Wise Growth (XGBoost default)
# =============================================================================

class LevelWiseGrowth(GrowthStrategy):
    """Level-wise (depth-first) tree growth.
    
    Processes all nodes at each depth level before moving to the next.
    This is the default strategy used by XGBoost.
    
    Characteristics:
    - Balanced trees (all branches grow equally)
    - O(depth) kernel launches on GPU
    - Good GPU utilization (batch all nodes at a level)
    
    Phase 14: Added missing value handling.
    Phase 14.3: Added categorical feature support.
    """
    
    def grow(
        self,
        binned: NDArray,
        grad: NDArray,
        hess: NDArray,
        config: GrowthConfig,
        has_missing: NDArray | None = None,
        is_categorical: NDArray | None = None,
        n_categories: NDArray | None = None,
    ) -> TreeStructure:
        """Grow tree level by level.
        
        Phase 14: Added has_missing parameter for NaN handling.
        Phase 14.3: Added categorical feature support.
        """
        n_features, n_samples = binned.shape
        max_nodes = 2**(config.max_depth + 1) - 1
        
        # Initialize arrays
        features = np.full(max_nodes, -1, dtype=np.int32)
        thresholds = np.zeros(max_nodes, dtype=np.int32)
        values = np.zeros(max_nodes, dtype=np.float32)
        left_children = np.full(max_nodes, -1, dtype=np.int32)
        right_children = np.full(max_nodes, -1, dtype=np.int32)
        missing_go_left = np.ones(max_nodes, dtype=np.bool_)  # Phase 14: default left
        is_categorical_split = np.zeros(max_nodes, dtype=np.bool_)  # Phase 14.3
        cat_bitsets = np.zeros(max_nodes, dtype=np.uint64)          # Phase 14.3
        
        # Track sample assignments
        sample_node_ids = init_sample_node_ids(n_samples, device="cpu")
        if is_cuda() and hasattr(binned, '__cuda_array_interface__'):
            sample_node_ids = init_sample_node_ids(n_samples, device="cuda")
        
        # Track active nodes and their histograms for subtraction
        parent_histograms: dict[int, NodeHistogram] = {}
        
        actual_depth = 0
        
        # Build level by level
        for depth in range(config.max_depth):
            nodes_at_level = get_nodes_at_depth(depth)
            
            # Filter to nodes that have samples
            active_nodes = self._get_active_nodes(sample_node_ids, nodes_at_level)
            if not active_nodes:
                break
            
            actual_depth = depth + 1
            
            # Build histograms for active nodes
            histograms = build_node_histograms(
                binned, grad, hess, sample_node_ids, active_nodes
            )
            
            # Find splits for all nodes (with missing/categorical handling)
            splits = find_node_splits(
                histograms,
                reg_lambda=config.reg_lambda,
                min_child_weight=config.min_child_weight,
                min_gain=config.min_gain,
                has_missing=has_missing,        # Phase 14
                is_categorical=is_categorical,  # Phase 14.3
                n_categories=n_categories,      # Phase 14.3
            )
            
            # Apply splits to tree structure
            for node_id, node_split in splits.items():
                features[node_id] = node_split.split.feature
                thresholds[node_id] = node_split.split.threshold
                left_children[node_id] = node_split.left_child
                right_children[node_id] = node_split.right_child
                missing_go_left[node_id] = node_split.missing_go_left      # Phase 14
                is_categorical_split[node_id] = node_split.is_categorical  # Phase 14.3
                cat_bitsets[node_id] = node_split.cat_bitset               # Phase 14.3
            
            # Partition samples (handles missing via learned direction)
            if splits:
                sample_node_ids = partition_samples(
                    binned, sample_node_ids, splits, 
                    missing_go_left=missing_go_left  # Phase 14
                )
            
            # Store histograms for potential subtraction (future optimization)
            parent_histograms = histograms
        
        # Compute leaf values for all leaf nodes
        leaf_nodes = self._find_leaf_nodes(features, left_children, max_nodes)
        leaf_values = compute_leaf_values(
            grad, hess, sample_node_ids, leaf_nodes, config.reg_lambda, config.reg_alpha
        )
        
        for node_id, value in leaf_values.items():
            values[node_id] = value
        
        # Trim to actual size
        n_nodes = self._count_nodes(left_children)
        
        # Check if we have any categorical splits
        any_cat = np.any(is_categorical_split[:n_nodes]) if n_nodes > 0 else False
        
        return TreeStructure(
            features=features[:n_nodes] if n_nodes < max_nodes else features,
            thresholds=thresholds[:n_nodes] if n_nodes < max_nodes else thresholds,
            values=values[:n_nodes] if n_nodes < max_nodes else values,
            left_children=left_children[:n_nodes] if n_nodes < max_nodes else left_children,
            right_children=right_children[:n_nodes] if n_nodes < max_nodes else right_children,
            n_nodes=n_nodes,
            depth=actual_depth,
            n_features=n_features,
            missing_go_left=missing_go_left[:n_nodes] if n_nodes < max_nodes else missing_go_left,  # Phase 14
            is_categorical_split=is_categorical_split[:n_nodes] if any_cat else None,  # Phase 14.3
            cat_bitsets=cat_bitsets[:n_nodes] if any_cat else None,                    # Phase 14.3
        )
    
    def _get_active_nodes(self, sample_node_ids, candidate_nodes: list[int]) -> list[int]:
        """Get nodes that have samples assigned to them."""
        if hasattr(sample_node_ids, 'copy_to_host'):
            sample_node_ids = sample_node_ids.copy_to_host()
        unique_nodes = set(np.unique(sample_node_ids))
        return [n for n in candidate_nodes if n in unique_nodes]
    
    def _find_leaf_nodes(self, features, left_children, max_nodes) -> list[int]:
        """Find all leaf nodes (nodes with no children or not split)."""
        leaves = []
        for i in range(max_nodes):
            if left_children[i] == -1 and (i == 0 or self._has_parent(i, features)):
                leaves.append(i)
        return leaves
    
    def _has_parent(self, node_id: int, features) -> bool:
        """Check if node has a valid parent (was created by a split)."""
        if node_id == 0:
            return True
        parent = (node_id - 1) // 2
        return features[parent] >= 0
    
    def _count_nodes(self, left_children) -> int:
        """Count actual nodes in tree."""
        # Find the highest index that's either root or has a valid parent
        for i in range(len(left_children) - 1, -1, -1):
            if i == 0 or left_children[(i - 1) // 2] != -1 or ((i - 1) // 2 == 0):
                if i == 0:
                    return 1
                # Check if this node or its sibling was created
                parent = (i - 1) // 2
                if left_children[parent] != -1:
                    return max(left_children[parent], left_children[parent] + 1, i) + 1
        return 1


# =============================================================================
# Leaf-Wise Growth (LightGBM style)
# =============================================================================

class LeafWiseGrowth(GrowthStrategy):
    """Leaf-wise (best-first) tree growth.
    
    Always splits the leaf with the highest gain, regardless of depth.
    This is the strategy used by LightGBM.
    
    Characteristics:
    - Unbalanced trees (deeper on informative branches)
    - Often achieves lower loss with fewer leaves
    - Can overfit if max_leaves not set properly
    - More kernel launches on GPU (one per split)
    
    Note:
        Requires `config.max_leaves` to be set.
    """
    
    def grow(
        self,
        binned: NDArray,
        grad: NDArray,
        hess: NDArray,
        config: GrowthConfig,
        has_missing: NDArray | None = None,
        is_categorical: NDArray | None = None,
        n_categories: NDArray | None = None,
    ) -> TreeStructure:
        """Grow tree by always splitting best leaf."""
        if config.max_leaves is None:
            config.max_leaves = 2**config.max_depth
        
        n_features, n_samples = binned.shape
        # Use complete binary tree size to accommodate any depth
        # Node IDs follow binary tree convention, so we need 2^(max_depth+1) - 1 slots
        max_nodes = 2**(config.max_depth + 1) - 1
        
        # Initialize arrays
        features = np.full(max_nodes, -1, dtype=np.int32)
        thresholds = np.zeros(max_nodes, dtype=np.int32)
        values = np.zeros(max_nodes, dtype=np.float32)
        left_children = np.full(max_nodes, -1, dtype=np.int32)
        right_children = np.full(max_nodes, -1, dtype=np.int32)
        
        # Track sample assignments (CPU for leaf-wise since we need frequent access)
        sample_node_ids = np.zeros(n_samples, dtype=np.int32)
        
        # Priority queue: (negative_gain, node_id, split_info, histogram)
        # We use negative gain because heapq is min-heap
        import heapq
        candidates: list[tuple[float, int, NodeSplit, NodeHistogram]] = []
        
        # Start with root
        root_hist = build_node_histograms(binned, grad, hess, sample_node_ids, [0])
        if 0 not in root_hist:
            # No samples, return single leaf
            return self._single_leaf_tree(grad, hess, config, n_features)
        
        root_splits = find_node_splits(
            root_hist,
            reg_lambda=config.reg_lambda,
            min_child_weight=config.min_child_weight,
            min_gain=config.min_gain,
        )
        
        if 0 in root_splits:
            heapq.heappush(candidates, (
                -root_splits[0].split.gain,
                0,
                root_splits[0],
                root_hist[0],
            ))
        
        n_leaves = 1
        actual_depth = 0
        
        while candidates and n_leaves < config.max_leaves:
            neg_gain, node_id, node_split, node_hist = heapq.heappop(candidates)
            
            # Check if this node is still a leaf (might have been split)
            if features[node_id] >= 0:
                continue
            
            # Apply split
            features[node_id] = node_split.split.feature
            thresholds[node_id] = node_split.split.threshold
            left_children[node_id] = node_split.left_child
            right_children[node_id] = node_split.right_child
            
            # Update sample assignments
            sample_node_ids = partition_samples(
                binned, sample_node_ids, {node_id: node_split}
            )
            if hasattr(sample_node_ids, 'copy_to_host'):
                sample_node_ids = sample_node_ids.copy_to_host()
            
            n_leaves += 1  # Split creates one new leaf (2 children - 1 parent)
            
            # Track depth
            node_depth = self._get_depth(node_id)
            actual_depth = max(actual_depth, node_depth + 1)
            
            # Don't exceed max_depth
            if node_depth + 1 >= config.max_depth:
                continue
            
            # Find splits for children
            left_id, right_id = node_split.left_child, node_split.right_child
            child_hists = build_node_histograms(
                binned, grad, hess, sample_node_ids, [left_id, right_id]
            )
            
            child_splits = find_node_splits(
                child_hists,
                reg_lambda=config.reg_lambda,
                min_child_weight=config.min_child_weight,
                min_gain=config.min_gain,
            )
            
            # Add valid child splits to candidates
            for child_id in [left_id, right_id]:
                if child_id in child_splits and child_id in child_hists:
                    heapq.heappush(candidates, (
                        -child_splits[child_id].split.gain,
                        child_id,
                        child_splits[child_id],
                        child_hists[child_id],
                    ))
        
        # Compute leaf values
        leaf_nodes = [i for i in range(max_nodes) if left_children[i] == -1 and 
                      (i == 0 or features[(i-1)//2] >= 0)]
        leaf_values = compute_leaf_values(
            grad, hess, sample_node_ids, leaf_nodes, config.reg_lambda, config.reg_alpha
        )
        
        for node_id, value in leaf_values.items():
            values[node_id] = value
        
        return TreeStructure(
            features=features,
            thresholds=thresholds,
            values=values,
            left_children=left_children,
            right_children=right_children,
            n_nodes=max_nodes,
            depth=actual_depth,
            n_features=n_features,
        )
    
    def _get_depth(self, node_id: int) -> int:
        """Get depth of a node."""
        if node_id == 0:
            return 0
        return int(np.floor(np.log2(node_id + 1)))
    
    def _single_leaf_tree(self, grad, hess, config, n_features) -> TreeStructure:
        """Create a tree with just the root as a leaf."""
        if hasattr(grad, 'copy_to_host'):
            grad = grad.copy_to_host()
            hess = hess.copy_to_host()
        
        leaf_value = -float(np.sum(grad)) / (float(np.sum(hess)) + config.reg_lambda)
        
        return TreeStructure(
            features=np.array([-1], dtype=np.int32),
            thresholds=np.array([0], dtype=np.int32),
            values=np.array([leaf_value], dtype=np.float32),
            left_children=np.array([-1], dtype=np.int32),
            right_children=np.array([-1], dtype=np.int32),
            n_nodes=1,
            depth=0,
            n_features=n_features,
        )


# =============================================================================
# Symmetric Growth (CatBoost style)
# =============================================================================

class SymmetricGrowth(GrowthStrategy):
    """Symmetric (oblivious) tree growth.
    
    All nodes at the same depth use the SAME split condition.
    This is the strategy used by CatBoost.
    
    Characteristics:
    - Very fast prediction (just bit operations)
    - Regularization effect (fewer parameters)
    - 2^depth leaves regardless of data
    - Good for categorical features
    """
    
    def grow(
        self,
        binned: NDArray,
        grad: NDArray,
        hess: NDArray,
        config: GrowthConfig,
        has_missing: NDArray | None = None,
        is_categorical: NDArray | None = None,
        n_categories: NDArray | None = None,
    ) -> TreeStructure:
        """Grow symmetric tree with one split per level."""
        n_features, n_samples = binned.shape
        
        level_features = np.full(config.max_depth, -1, dtype=np.int32)
        level_thresholds = np.zeros(config.max_depth, dtype=np.int32)
        
        # Track sample leaf assignments (0 to 2^depth - 1)
        sample_leaf_ids = np.zeros(n_samples, dtype=np.int32)
        
        actual_depth = 0
        
        for depth in range(config.max_depth):
            # Build combined histogram across ALL current leaves
            # For symmetric trees, we sum histograms since all use same split
            n_leaves = 2**depth
            leaf_ids = list(range(n_leaves))
            
            combined_hist = self._build_combined_histogram(
                binned, grad, hess, sample_leaf_ids, leaf_ids
            )
            
            if combined_hist is None:
                break
            
            # Find ONE best split for entire level
            splits = find_node_splits(
                {0: combined_hist},  # Treat as single node
                reg_lambda=config.reg_lambda,
                min_child_weight=config.min_child_weight,
                min_gain=config.min_gain,
            )
            
            if 0 not in splits:
                break
            
            split = splits[0].split
            level_features[depth] = split.feature
            level_thresholds[depth] = split.threshold
            actual_depth = depth + 1
            
            # Partition ALL samples using this single split
            if hasattr(binned, 'copy_to_host'):
                binned_cpu = binned.copy_to_host()
            else:
                binned_cpu = np.asarray(binned)
            
            goes_right = binned_cpu[split.feature, :] > split.threshold
            sample_leaf_ids = 2 * sample_leaf_ids + goes_right.astype(np.int32)
        
        # Compute leaf values for all 2^depth leaves
        n_leaves = 2**actual_depth
        leaf_values = np.zeros(n_leaves, dtype=np.float32)
        
        if hasattr(grad, 'copy_to_host'):
            grad_cpu = grad.copy_to_host()
            hess_cpu = hess.copy_to_host()
        else:
            grad_cpu = np.asarray(grad)
            hess_cpu = np.asarray(hess)
        
        for leaf_id in range(n_leaves):
            mask = sample_leaf_ids == leaf_id
            if np.any(mask):
                sum_grad = float(np.sum(grad_cpu[mask]))
                sum_hess = float(np.sum(hess_cpu[mask]))
                leaf_values[leaf_id] = -sum_grad / (sum_hess + config.reg_lambda)
        
        # Create TreeStructure with symmetric flag
        # For compatibility, also create standard tree arrays
        max_nodes = 2**(actual_depth + 1) - 1
        features = np.full(max_nodes, -1, dtype=np.int32)
        thresholds = np.zeros(max_nodes, dtype=np.int32)
        values = np.zeros(max_nodes, dtype=np.float32)
        left_children = np.full(max_nodes, -1, dtype=np.int32)
        right_children = np.full(max_nodes, -1, dtype=np.int32)
        
        # Fill in symmetric structure
        self._fill_symmetric_structure(
            features, thresholds, left_children, right_children,
            level_features, level_thresholds, actual_depth
        )
        
        # Set leaf values
        leaf_start = 2**actual_depth - 1
        for i, val in enumerate(leaf_values):
            values[leaf_start + i] = val
        
        return TreeStructure(
            features=features,
            thresholds=thresholds,
            values=values,
            left_children=left_children,
            right_children=right_children,
            n_nodes=max_nodes,
            depth=actual_depth,
            n_features=n_features,
            is_symmetric=True,
            level_features=level_features[:actual_depth],
            level_thresholds=level_thresholds[:actual_depth],
        )
    
    def _build_combined_histogram(
        self,
        binned,
        grad,
        hess,
        sample_leaf_ids,
        leaf_ids: list[int],
    ) -> NodeHistogram | None:
        """Build combined histogram summing all leaves."""
        if hasattr(binned, 'copy_to_host'):
            binned = binned.copy_to_host()
            grad = grad.copy_to_host()
            hess = hess.copy_to_host()
        
        n_features = binned.shape[0]
        combined_grad = np.zeros((n_features, 256), dtype=np.float32)
        combined_hess = np.zeros((n_features, 256), dtype=np.float32)
        total_samples = 0
        total_grad = 0.0
        total_hess = 0.0
        
        for leaf_id in leaf_ids:
            mask = sample_leaf_ids == leaf_id
            n_in_leaf = int(np.sum(mask))
            if n_in_leaf == 0:
                continue
            
            total_samples += n_in_leaf
            
            # Build histogram for this leaf
            leaf_binned = binned[:, mask]
            leaf_grad = grad[mask]
            leaf_hess = hess[mask]
            
            from .._backends._cpu import build_histogram_cpu
            hist_grad, hist_hess = build_histogram_cpu(
                leaf_binned.astype(np.uint8),
                leaf_grad.astype(np.float32),
                leaf_hess.astype(np.float32),
            )
            
            combined_grad += hist_grad
            combined_hess += hist_hess
            total_grad += float(np.sum(leaf_grad))
            total_hess += float(np.sum(leaf_hess))
        
        if total_samples == 0:
            return None
        
        return NodeHistogram(
            node_id=0,
            hist_grad=combined_grad,
            hist_hess=combined_hess,
            sum_grad=total_grad,
            sum_hess=total_hess,
            n_samples=total_samples,
        )
    
    def _fill_symmetric_structure(
        self,
        features,
        thresholds,
        left_children,
        right_children,
        level_features,
        level_thresholds,
        depth,
    ):
        """Fill standard tree arrays from symmetric representation."""
        for d in range(depth):
            feat = level_features[d]
            thresh = level_thresholds[d]
            
            # All nodes at this depth get same split
            level_start = 2**d - 1
            level_end = 2**(d+1) - 1
            
            for node in range(level_start, level_end):
                features[node] = feat
                thresholds[node] = thresh
                left_children[node] = 2 * node + 1
                right_children[node] = 2 * node + 2


# =============================================================================
# Factory Function
# =============================================================================

def get_growth_strategy(name: str) -> GrowthStrategy:
    """Get a growth strategy by name.
    
    Args:
        name: Strategy name - "levelwise", "leafwise", or "symmetric"
        
    Returns:
        GrowthStrategy instance
    """
    strategies = {
        "levelwise": LevelWiseGrowth,
        "level_wise": LevelWiseGrowth,
        "level-wise": LevelWiseGrowth,
        "leafwise": LeafWiseGrowth,
        "leaf_wise": LeafWiseGrowth,
        "leaf-wise": LeafWiseGrowth,
        "symmetric": SymmetricGrowth,
        "oblivious": SymmetricGrowth,
    }
    
    name_lower = name.lower()
    if name_lower not in strategies:
        available = ["levelwise", "leafwise", "symmetric"]
        raise ValueError(f"Unknown growth strategy '{name}'. Available: {available}")
    
    return strategies[name_lower]()
