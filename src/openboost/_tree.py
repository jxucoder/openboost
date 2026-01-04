"""Tree data structure with flat arrays."""

import numpy as np


class Tree:
    """Decision tree stored as flat arrays for GPU-friendly access."""
    
    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.max_nodes = 2 ** (max_depth + 1) - 1
        
        # Flat arrays for tree structure
        self.feature = np.full(self.max_nodes, -1, dtype=np.int32)
        self.bin_threshold = np.full(self.max_nodes, -1, dtype=np.int32)
        self.value = np.zeros(self.max_nodes, dtype=np.float32)
        self.is_leaf = np.ones(self.max_nodes, dtype=np.bool_)
    
    def set_split(self, node_idx: int, feature: int, bin_threshold: int):
        """Set a split at the given node."""
        self.feature[node_idx] = feature
        self.bin_threshold[node_idx] = bin_threshold
        self.is_leaf[node_idx] = False
    
    def set_leaf_value(self, node_idx: int, value: float):
        """Set the prediction value for a leaf node."""
        self.value[node_idx] = value
        self.is_leaf[node_idx] = True

