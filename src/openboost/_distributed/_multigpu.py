"""Multi-GPU training support for OpenBoost using Ray.

Phase 18: Implements data-parallel multi-GPU training where each GPU holds
a subset of the data and computes local histograms, which are then aggregated.

Architecture:
    GPU 0: samples 0-N/4      → local histograms → ┐
    GPU 1: samples N/4-N/2    → local histograms → ├→ AllReduce → global histograms
    GPU 2: samples N/2-3N/4   → local histograms → │
    GPU 3: samples 3N/4-N     → local histograms → ┘

Usage:
    # Simple API
    model = ob.GradientBoosting(n_trees=100, n_gpus=4)
    model.fit(X, y)
    
    # Or explicit device list
    model = ob.GradientBoosting(n_trees=100, devices=[0, 1, 2, 3])
    model.fit(X, y)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import numpy as np
from numpy.typing import NDArray

try:
    import ray
except ImportError:
    ray = None

if TYPE_CHECKING:
    from .._core._growth import TreeStructure
    from .._loss import LossFunction


# =============================================================================
# GPUWorker: Ray actor that owns one GPU and a data shard
# =============================================================================

class GPUWorkerBase:
    """Ray actor that owns one GPU and holds a shard of the training data.
    
    Each worker:
    1. Holds a subset of samples (data shard)
    2. Computes local gradients on its GPU
    3. Builds local histograms for tree construction
    4. Updates local predictions with new trees
    
    Note: This class is decorated with @ray.remote(num_gpus=1) when Ray is available.
    """
    
    def __init__(
        self,
        gpu_id: int,
        X_shard: NDArray,
        y_shard: NDArray,
        n_bins: int,
        bin_edges: Optional[NDArray] = None,
    ):
        """Initialize worker with data shard on assigned GPU.
        
        Args:
            gpu_id: GPU device ID to use
            X_shard: Feature data shard, shape (n_samples_shard, n_features)
            y_shard: Target data shard, shape (n_samples_shard,)
            n_bins: Number of bins for histogram building
            bin_edges: Optional pre-computed bin edges for consistent binning
                       Shape (n_features, n_bins + 1)
        """
        try:
            from numba import cuda
            cuda.select_device(gpu_id)
            self.has_cuda = True
        except Exception:
            self.has_cuda = False
        
        self.gpu_id = gpu_id
        self.n_bins = n_bins
        self.n_samples = len(y_shard)
        
        # Import array function
        import openboost as ob
        
        # Bin and store data
        # If bin_edges provided, use them for consistent binning across shards
        if bin_edges is not None:
            self.X_binned = ob.array(X_shard, n_bins=n_bins, bin_edges=bin_edges)
        else:
            self.X_binned = ob.array(X_shard, n_bins=n_bins)
        
        self.y = y_shard.astype(np.float32)
        self.n_features = self.X_binned.n_features
        
        # Initialize predictions (on GPU if available)
        if self.has_cuda:
            from numba import cuda
            self.pred = cuda.device_array(self.n_samples, dtype=np.float32)
            # Zero initialize
            self._zero_predictions()
            self.y_gpu = cuda.to_device(self.y)
        else:
            self.pred = np.zeros(self.n_samples, dtype=np.float32)
            self.y_gpu = self.y
        
        # Track sample node IDs for distributed tree building
        self.sample_node_ids = np.zeros(self.n_samples, dtype=np.int32)
    
    def _zero_predictions(self):
        """Zero out predictions array on GPU."""
        if self.has_cuda:
            from numba import cuda
            
            @cuda.jit
            def _fill_zeros(arr, n):
                idx = cuda.grid(1)
                if idx < n:
                    arr[idx] = 0.0
            
            threads = 256
            blocks = (self.n_samples + threads - 1) // threads
            _fill_zeros[blocks, threads](self.pred, self.n_samples)
    
    def compute_gradients(self, loss_fn: LossFunction) -> tuple[NDArray, NDArray]:
        """Compute local gradients on this GPU shard.
        
        Args:
            loss_fn: Loss function that computes (grad, hess) from (pred, y)
            
        Returns:
            Tuple of (gradients, hessians) arrays, shape (n_samples_shard,)
        """
        if self.has_cuda and hasattr(self.pred, 'copy_to_host'):
            # Try GPU-native gradient computation first
            try:
                grad, hess = loss_fn(self.pred, self.y_gpu)
                # Return as CPU arrays for aggregation
                if hasattr(grad, 'copy_to_host'):
                    return grad.copy_to_host().astype(np.float32), hess.copy_to_host().astype(np.float32)
                return grad.astype(np.float32), hess.astype(np.float32)
            except Exception:
                # Fall back to CPU computation
                pred_cpu = self.pred.copy_to_host()
                grad, hess = loss_fn(pred_cpu, self.y)
                return grad.astype(np.float32), hess.astype(np.float32)
        else:
            grad, hess = loss_fn(self.pred, self.y)
            return grad.astype(np.float32), hess.astype(np.float32)
    
    def build_histogram(
        self,
        grad: NDArray,
        hess: NDArray,
        node_ids: Optional[List[int]] = None,
    ) -> tuple[NDArray, NDArray]:
        """Build local histogram for this shard.
        
        Args:
            grad: Gradient array, shape (n_samples_shard,)
            hess: Hessian array, shape (n_samples_shard,)
            node_ids: Optional list of node IDs to build histograms for.
                     If None, builds histogram for all samples (root node).
        
        Returns:
            Tuple of (hist_grad, hist_hess) arrays
            Shape: (n_features, n_bins) if node_ids is None
                   or dict mapping node_id to histogram
        """
        from .._core._histogram import build_histogram
        from .._array import as_numba_array
        
        # Get binned data
        binned = self.X_binned.data
        
        # Prepare sample indices
        sample_indices = np.arange(self.n_samples, dtype=np.int32)
        
        if self.has_cuda:
            from numba import cuda
            grad_gpu = cuda.to_device(grad)
            hess_gpu = cuda.to_device(hess)
            sample_indices_gpu = cuda.to_device(sample_indices)
            
            hist_grad, hist_hess = build_histogram(binned, grad_gpu, hess_gpu, sample_indices_gpu)
            
            # Return as CPU arrays
            if hasattr(hist_grad, 'copy_to_host'):
                return hist_grad.copy_to_host(), hist_hess.copy_to_host()
        else:
            hist_grad, hist_hess = build_histogram(binned, grad, hess, sample_indices)
        
        return hist_grad, hist_hess
    
    def update_predictions(self, tree: TreeStructure, learning_rate: float):
        """Update local predictions with new tree.
        
        Args:
            tree: Fitted tree structure
            learning_rate: Learning rate to apply
        """
        # Get tree predictions for this shard
        tree_pred = tree(self.X_binned)
        
        if hasattr(tree_pred, 'copy_to_host'):
            tree_pred = tree_pred.copy_to_host()
        
        if self.has_cuda and hasattr(self.pred, 'copy_to_host'):
            # Update on GPU
            pred_cpu = self.pred.copy_to_host()
            pred_cpu += learning_rate * tree_pred
            from numba import cuda
            cuda.to_device(pred_cpu, to=self.pred)
        else:
            self.pred += learning_rate * tree_pred
    
    def get_predictions(self) -> NDArray:
        """Get current predictions (copies from GPU if needed)."""
        if hasattr(self.pred, 'copy_to_host'):
            return self.pred.copy_to_host()
        return self.pred.copy()
    
    def get_n_features(self) -> int:
        """Get number of features."""
        return self.n_features
    
    def get_n_samples(self) -> int:
        """Get number of samples in this shard."""
        return self.n_samples
    
    def get_bin_edges(self) -> Optional[NDArray]:
        """Get bin edges used by this worker (for consistent binning)."""
        if hasattr(self.X_binned, 'bin_edges'):
            return self.X_binned.bin_edges
        return None


# Create Ray remote version if Ray is available
if ray is not None:
    GPUWorker = ray.remote(num_gpus=1)(GPUWorkerBase)
else:
    GPUWorker = GPUWorkerBase


# =============================================================================
# MultiGPUContext: Manages multiple GPU workers
# =============================================================================

@dataclass
class MultiGPUContext:
    """Context manager for multi-GPU training using Ray.
    
    Handles:
    - GPU worker creation and management
    - Data sharding across GPUs
    - Histogram aggregation (AllReduce)
    - Tree broadcasting to workers
    
    Example:
        ctx = MultiGPUContext(n_gpus=4)
        ctx.setup(X, y, n_bins=256)
        
        for round in range(n_trees):
            # Compute gradients on each GPU
            grad_hess_refs = [w.compute_gradients.remote(loss_fn) for w in ctx.workers]
            
            # Build and aggregate histograms
            hist_refs = [
                w.build_histogram.remote(g, h)
                for w, (g, h) in zip(ctx.workers, grads)
            ]
            global_hist = ctx.aggregate_histograms(hist_refs)
            
            # Build tree from global histogram
            tree = build_tree_from_histogram(global_hist, ...)
            
            # Update predictions on all workers
            ctx.update_predictions(tree, learning_rate)
    """
    
    n_gpus: int = None
    devices: List[int] = None
    workers: List[Any] = None
    n_features: int = None
    n_samples: int = None
    shard_sizes: List[int] = None
    bin_edges: NDArray = None
    
    def __post_init__(self):
        """Initialize Ray and detect available GPUs."""
        if ray is None:
            raise ImportError(
                "Multi-GPU training requires Ray. "
                "Install with: pip install 'openboost[distributed]'"
            )
        
        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Determine which GPUs to use
        if self.devices is not None:
            self.n_gpus = len(self.devices)
        else:
            n_available = int(ray.available_resources().get('GPU', 0))
            if n_available == 0:
                raise RuntimeError(
                    "No GPUs available. Multi-GPU training requires at least one GPU. "
                    "Use GradientBoosting without n_gpus for CPU training."
                )
            if self.n_gpus is None:
                self.n_gpus = n_available
            else:
                self.n_gpus = min(self.n_gpus, n_available)
            self.devices = list(range(self.n_gpus))
        
        self.workers = []
        self.shard_sizes = []
    
    def setup(
        self,
        X: NDArray,
        y: NDArray,
        n_bins: int = 256,
        bin_edges: Optional[NDArray] = None,
    ):
        """Shard data and create GPU workers.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training targets, shape (n_samples,)
            n_bins: Number of bins for histogram
            bin_edges: Optional pre-computed bin edges for consistent binning
        """
        self.n_samples = len(y)
        self.n_features = X.shape[1]
        
        # Compute bin edges globally for consistent binning across shards
        if bin_edges is None:
            # Use a subset of data to compute bin edges efficiently
            sample_size = min(100000, self.n_samples)
            if sample_size < self.n_samples:
                indices = np.random.choice(self.n_samples, sample_size, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
            
            # Compute percentile-based bin edges
            self.bin_edges = np.zeros((self.n_features, n_bins + 1), dtype=np.float32)
            for f in range(self.n_features):
                col = X_sample[:, f]
                # Handle NaN values
                valid = col[~np.isnan(col)]
                if len(valid) > 0:
                    percentiles = np.linspace(0, 100, n_bins + 1)
                    self.bin_edges[f] = np.percentile(valid, percentiles)
                else:
                    self.bin_edges[f] = np.linspace(0, 1, n_bins + 1)
        else:
            self.bin_edges = bin_edges
        
        # Split data into shards
        indices = np.array_split(np.arange(self.n_samples), self.n_gpus)
        self.shard_sizes = [len(idx) for idx in indices]
        
        # Create workers
        self.workers = []
        for gpu_id, shard_indices in zip(self.devices, indices):
            X_shard = X[shard_indices]
            y_shard = y[shard_indices]
            
            worker = GPUWorker.remote(
                gpu_id=gpu_id,
                X_shard=X_shard,
                y_shard=y_shard,
                n_bins=n_bins,
                bin_edges=self.bin_edges,
            )
            self.workers.append(worker)
    
    def compute_all_gradients(
        self,
        loss_fn: LossFunction,
    ) -> List[tuple[NDArray, NDArray]]:
        """Compute gradients on all workers in parallel.
        
        Args:
            loss_fn: Loss function for gradient computation
            
        Returns:
            List of (grad, hess) tuples, one per worker
        """
        grad_hess_refs = [
            worker.compute_gradients.remote(loss_fn)
            for worker in self.workers
        ]
        return ray.get(grad_hess_refs)
    
    def build_all_histograms(
        self,
        grads_hess: List[tuple[NDArray, NDArray]],
    ) -> List[tuple[NDArray, NDArray]]:
        """Build local histograms on all workers in parallel.
        
        Args:
            grads_hess: List of (grad, hess) tuples, one per worker
            
        Returns:
            List of (hist_grad, hist_hess) tuples, one per worker
        """
        hist_refs = [
            worker.build_histogram.remote(grad, hess)
            for worker, (grad, hess) in zip(self.workers, grads_hess)
        ]
        return ray.get(hist_refs)
    
    def aggregate_histograms(
        self,
        local_histograms: List[tuple[NDArray, NDArray]],
    ) -> tuple[NDArray, NDArray]:
        """Sum histograms from all workers (AllReduce).
        
        Args:
            local_histograms: List of (hist_grad, hist_hess) from each worker
            
        Returns:
            Tuple of (global_hist_grad, global_hist_hess)
        """
        if not local_histograms:
            raise ValueError("No histograms to aggregate")
        
        # Sum all histograms
        global_hist_grad = local_histograms[0][0].copy()
        global_hist_hess = local_histograms[0][1].copy()
        
        for hist_grad, hist_hess in local_histograms[1:]:
            global_hist_grad += hist_grad
            global_hist_hess += hist_hess
        
        return global_hist_grad, global_hist_hess
    
    def update_all_predictions(
        self,
        tree: TreeStructure,
        learning_rate: float,
    ):
        """Update predictions on all workers with new tree.
        
        Args:
            tree: Fitted tree to add to ensemble
            learning_rate: Learning rate for this tree
        """
        update_refs = [
            worker.update_predictions.remote(tree, learning_rate)
            for worker in self.workers
        ]
        ray.get(update_refs)  # Wait for completion
    
    def get_all_predictions(self) -> NDArray:
        """Collect predictions from all workers and concatenate.
        
        Returns:
            Full prediction array, shape (n_samples,)
        """
        pred_refs = [worker.get_predictions.remote() for worker in self.workers]
        preds = ray.get(pred_refs)
        return np.concatenate(preds)
    
    def shutdown(self):
        """Shutdown workers and cleanup."""
        if self.workers:
            for worker in self.workers:
                ray.kill(worker)
            self.workers = []


# =============================================================================
# High-level distributed tree fitting
# =============================================================================

def fit_tree_multigpu(
    ctx: MultiGPUContext,
    grads_hess: List[tuple[NDArray, NDArray]],
    *,
    max_depth: int = 6,
    min_child_weight: float = 1.0,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    min_gain: float = 0.0,
) -> TreeStructure:
    """Fit a single tree using multi-GPU histogram aggregation.
    
    This is a simplified single-level tree building that:
    1. Builds local histograms on each GPU
    2. Aggregates to global histogram
    3. Uses standard tree building from global histogram
    
    For full distributed tree building with sample partitioning,
    see fit_tree_distributed in _tree.py.
    
    Args:
        ctx: MultiGPUContext with initialized workers
        grads_hess: List of (grad, hess) tuples from each worker
        max_depth: Maximum tree depth
        min_child_weight: Minimum sum of hessian in a leaf
        reg_lambda: L2 regularization
        reg_alpha: L1 regularization
        min_gain: Minimum gain to make a split
        
    Returns:
        Fitted TreeStructure
    """
    from .._core._tree import fit_tree
    from .._array import BinnedArray
    import openboost as ob
    
    # Build local histograms on each GPU
    local_histograms = ctx.build_all_histograms(grads_hess)
    
    # Aggregate histograms
    global_hist_grad, global_hist_hess = ctx.aggregate_histograms(local_histograms)
    
    # For now, we use a simplified approach:
    # Build tree using the first worker's data structure but with global histograms
    # A more sophisticated approach would do distributed tree building
    
    # Get gradients and hessians aggregated
    total_grad = np.zeros(ctx.n_samples, dtype=np.float32)
    total_hess = np.zeros(ctx.n_samples, dtype=np.float32)
    
    offset = 0
    for (grad, hess), size in zip(grads_hess, ctx.shard_sizes):
        total_grad[offset:offset + size] = grad
        total_hess[offset:offset + size] = hess
        offset += size
    
    # Create a dummy BinnedArray for tree fitting
    # In practice, we'd want to do distributed tree building
    # For now, collect data to driver and fit there
    all_preds = ray.get([w.get_predictions.remote() for w in ctx.workers])
    
    # Use the global histogram for tree building
    # This is where we'd integrate with fit_tree_from_histogram
    # For now, fall back to standard fit_tree with aggregated data
    
    # Get binned data from first worker for structure
    # NOTE: This is a simplification - full implementation would do
    # distributed tree building with sample partitioning
    first_worker_bin_edges = ray.get(ctx.workers[0].get_bin_edges.remote())
    
    # Build tree using growth strategy with global histogram
    from .._core._growth import LevelWiseGrowth, GrowthConfig
    from .._core._primitives import NodeHistogram
    
    config = GrowthConfig(
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        min_gain=min_gain,
    )
    
    # Create tree structure using histogram-based building
    return _build_tree_from_global_histogram(
        global_hist_grad,
        global_hist_hess,
        ctx.n_features,
        config,
    )


def _build_tree_from_global_histogram(
    hist_grad: NDArray,
    hist_hess: NDArray,
    n_features: int,
    config: Any,
) -> TreeStructure:
    """Build a tree structure from aggregated global histogram.
    
    This is a simplified version that builds a single-split tree.
    Full implementation would do recursive histogram-based building.
    
    Args:
        hist_grad: Global gradient histogram, shape (n_features, n_bins)
        hist_hess: Global hessian histogram, shape (n_features, n_bins)
        n_features: Number of features
        config: GrowthConfig with tree building parameters
        
    Returns:
        TreeStructure
    """
    from .._core._growth import TreeStructure
    from .._core._split import find_best_split, compute_leaf_value
    
    # Get total gradient and hessian sums
    sum_grad = float(np.sum(hist_grad))
    sum_hess = float(np.sum(hist_hess))
    
    # Find best split
    split = find_best_split(
        hist_grad, hist_hess,
        sum_grad, sum_hess,
        reg_lambda=config.reg_lambda,
        min_child_weight=config.min_child_weight,
        min_gain=config.min_gain,
    )
    
    # Initialize tree arrays
    max_nodes = 2**(config.max_depth + 1) - 1
    features = np.full(max_nodes, -1, dtype=np.int32)
    thresholds = np.zeros(max_nodes, dtype=np.int32)
    values = np.zeros(max_nodes, dtype=np.float32)
    left_children = np.full(max_nodes, -1, dtype=np.int32)
    right_children = np.full(max_nodes, -1, dtype=np.int32)
    
    if not split.is_valid:
        # No valid split, create leaf
        values[0] = compute_leaf_value(sum_grad, sum_hess, config.reg_lambda, config.reg_alpha)
        return TreeStructure(
            features=features[:1],
            thresholds=thresholds[:1],
            values=values[:1],
            left_children=left_children[:1],
            right_children=right_children[:1],
            n_nodes=1,
            depth=0,
            n_features=n_features,
        )
    
    # Build tree level by level using histogram information
    # This is a simplified implementation - full version would recursively
    # split using histogram subtraction
    features[0] = split.feature
    thresholds[0] = split.threshold
    left_children[0] = 1
    right_children[0] = 2
    
    # Compute left and right sums from histogram
    left_grad = float(np.sum(hist_grad[split.feature, :split.threshold + 1]))
    left_hess = float(np.sum(hist_hess[split.feature, :split.threshold + 1]))
    right_grad = sum_grad - left_grad
    right_hess = sum_hess - left_hess
    
    # Set leaf values
    values[1] = compute_leaf_value(left_grad, left_hess, config.reg_lambda, config.reg_alpha)
    values[2] = compute_leaf_value(right_grad, right_hess, config.reg_lambda, config.reg_alpha)
    
    return TreeStructure(
        features=features[:3],
        thresholds=thresholds[:3],
        values=values[:3],
        left_children=left_children[:3],
        right_children=right_children[:3],
        n_nodes=3,
        depth=1,
        n_features=n_features,
    )


__all__ = [
    "GPUWorkerBase",
    "GPUWorker",
    "MultiGPUContext",
    "fit_tree_multigpu",
]
