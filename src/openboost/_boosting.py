"""Main GradientBoosting class."""

import numpy as np
from numba import cuda

from ._array import quantile_bin
from ._tree import Tree
from ._histogram import build_histograms
from ._split import find_best_splits, compute_leaf_values
from ._kernels import update_sample_nodes_kernel, predict_kernel


class GradientBoosting:
    """
    GPU-accelerated gradient boosting for regression (MSE loss).
    
    Example:
        >>> model = GradientBoosting(n_trees=100, max_depth=6)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(
        self,
        n_trees: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 1,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        
        self.trees_: list[Tree] = []
        self.initial_prediction_: float = 0.0
        self.bin_edges_: list = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoosting":
        """
        Fit the gradient boosting model.
        
        Args:
            X: Training features (n_samples, n_features), float32
            y: Training targets (n_samples,), float32
        
        Returns:
            self
        """
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)
        n_samples, n_features = X.shape
        
        # Quantile binning
        X_binned, self.bin_edges_ = quantile_bin(X)
        
        # Initial prediction = mean(y)
        self.initial_prediction_ = float(y.mean())
        predictions = np.full(n_samples, self.initial_prediction_, dtype=np.float32)
        
        # Transfer to GPU
        d_X_binned = cuda.to_device(X_binned)
        
        # Build trees
        self.trees_ = []
        for _ in range(self.n_trees):
            # Compute gradients (MSE: gradient = prediction - y)
            gradients = (predictions - y).astype(np.float32)
            
            # Fit one tree
            tree = self._fit_tree(d_X_binned, gradients, n_samples, n_features)
            self.trees_.append(tree)
            
            # Update predictions
            tree_preds = self._predict_tree(d_X_binned, tree, n_samples)
            predictions += self.learning_rate * tree_preds
        
        return self
    
    def _fit_tree(
        self,
        d_X_binned,
        gradients: np.ndarray,
        n_samples: int,
        n_features: int,
    ) -> Tree:
        """Fit a single tree using level-order building."""
        tree = Tree(self.max_depth)
        
        # Sample to node assignment
        sample_nodes = np.zeros(n_samples, dtype=np.int32)  # All start at root (0)
        d_sample_nodes = cuda.to_device(sample_nodes)
        d_gradients = cuda.to_device(gradients)
        
        # Level-order tree building: O(depth) kernel launches
        for depth in range(self.max_depth):
            node_start = 2**depth - 1  # First node at this level
            n_nodes = 2**depth         # Nodes at this level
            
            # Build histograms for all nodes at this level
            histograms = build_histograms(
                d_X_binned, d_gradients, d_sample_nodes,
                node_start, n_nodes
            )
            
            # Find best splits
            best_gain, best_feature, best_bin = find_best_splits(
                histograms, self.min_samples_leaf
            )
            
            # Compute leaf values for nodes that won't split
            leaf_values = compute_leaf_values(histograms)
            
            # Determine which nodes to split
            is_leaf = best_gain <= 0
            
            # Update tree structure
            for i in range(n_nodes):
                node_idx = node_start + i
                if is_leaf[i]:
                    tree.set_leaf_value(node_idx, leaf_values[i])
                else:
                    tree.set_split(node_idx, best_feature[i], best_bin[i])
            
            # Update sample node assignments on GPU
            if depth < self.max_depth - 1:  # No need to update for last level
                d_split_features = cuda.to_device(best_feature)
                d_split_bins = cuda.to_device(best_bin)
                d_is_leaf = cuda.to_device(is_leaf)
                
                threads = 256
                blocks = (n_samples + threads - 1) // threads
                update_sample_nodes_kernel[blocks, threads](
                    d_X_binned, d_sample_nodes,
                    node_start, n_nodes,
                    d_split_features, d_split_bins, d_is_leaf
                )
                cuda.synchronize()
        
        # Set leaf values for deepest level
        node_start = 2**self.max_depth - 1
        n_nodes = 2**self.max_depth
        histograms = build_histograms(
            d_X_binned, d_gradients, d_sample_nodes,
            node_start, n_nodes
        )
        leaf_values = compute_leaf_values(histograms)
        for i in range(n_nodes):
            node_idx = node_start + i
            tree.set_leaf_value(node_idx, leaf_values[i])
        
        return tree
    
    def _predict_tree(
        self,
        d_X_binned,
        tree: Tree,
        n_samples: int,
    ) -> np.ndarray:
        """Get predictions from a single tree."""
        d_predictions = cuda.device_array(n_samples, dtype=np.float32)
        cuda.to_device(np.zeros(n_samples, dtype=np.float32), to=d_predictions)
        
        d_features = cuda.to_device(tree.feature)
        d_bins = cuda.to_device(tree.bin_threshold)
        d_values = cuda.to_device(tree.value)
        d_is_leaf = cuda.to_device(tree.is_leaf)
        
        threads = 256
        blocks = (n_samples + threads - 1) // threads
        predict_kernel[blocks, threads](
            d_X_binned, d_features, d_bins, d_values, d_is_leaf,
            d_predictions, 1.0  # learning_rate=1.0, applied outside
        )
        
        return d_predictions.copy_to_host()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for X.
        
        Args:
            X: Features (n_samples, n_features), float32
        
        Returns:
            predictions: (n_samples,) float32
        """
        X = np.ascontiguousarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        
        # Bin using stored edges
        X_binned = np.empty((len(self.bin_edges_), n_samples), dtype=np.uint8)
        for f, edges in enumerate(self.bin_edges_):
            X_binned[f] = np.digitize(X[:, f], edges[1:-1], right=False).astype(np.uint8)
        
        d_X_binned = cuda.to_device(X_binned)
        d_predictions = cuda.device_array(n_samples, dtype=np.float32)
        cuda.to_device(
            np.full(n_samples, self.initial_prediction_, dtype=np.float32),
            to=d_predictions
        )
        
        threads = 256
        blocks = (n_samples + threads - 1) // threads
        
        # Accumulate predictions from all trees
        for tree in self.trees_:
            d_features = cuda.to_device(tree.feature)
            d_bins = cuda.to_device(tree.bin_threshold)
            d_values = cuda.to_device(tree.value)
            d_is_leaf = cuda.to_device(tree.is_leaf)
            
            predict_kernel[blocks, threads](
                d_X_binned, d_features, d_bins, d_values, d_is_leaf,
                d_predictions, self.learning_rate
            )
        
        return d_predictions.copy_to_host()

