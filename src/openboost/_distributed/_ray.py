"""Ray-based distributed training backend for OpenBoost.

Phase 12: Implements distributed training using Ray for multi-GPU/multi-node.
"""

from typing import Any, List, Dict
import numpy as np
from numpy.typing import NDArray

try:
    import ray
except ImportError:
    ray = None

import openboost as ob
from openboost._core._primitives import build_node_histograms, partition_samples


class RayWorker:
    """Worker that holds a data shard and computes local histograms."""
    
    def __init__(self, X_shard: NDArray, y_shard: NDArray, n_bins: int):
        self.X_binned = ob.array(X_shard, n_bins=n_bins)
        self.y = y_shard
        self.n_samples = len(y_shard)
        # Initialize sample_node_ids locally
        self.sample_node_ids = np.zeros(self.n_samples, dtype=np.int32)
        # Initialize predictions
        self.pred = np.zeros(self.n_samples, dtype=np.float32)
    
    def compute_histograms(self, grad: NDArray, hess: NDArray, 
                           node_ids: List[int]) -> Dict[int, Any]:
        """Compute local histograms for this shard."""
        histograms = build_node_histograms(
            self.X_binned.data if hasattr(self.X_binned, 'data') else self.X_binned,
            grad, hess, self.sample_node_ids, node_ids
        )
        return histograms

    def compute_gradients(self, loss_fn: Any) -> tuple[NDArray, NDArray]:
        """Compute gradients locally."""
        grad, hess = loss_fn(self.pred, self.y)
        return grad.astype(np.float32), hess.astype(np.float32)

    def update_predictions(self, tree: Any, learning_rate: float):
        """Update local predictions with new tree."""
        tree_pred = tree(self.X_binned)
        self.pred += learning_rate * tree_pred
        
    def partition_samples(self, splits: dict, node_ids: Any = None):
        """Update sample_node_ids based on splits."""
        self.sample_node_ids = partition_samples(
            self.X_binned.data if hasattr(self.X_binned, 'data') else self.X_binned,
            self.sample_node_ids,
            splits
        )
    
    def get_n_features(self) -> int:
        """Get number of features."""
        if hasattr(self.X_binned, 'n_features'):
            return self.X_binned.n_features
        return self.X_binned.shape[0]

    def get_sample_node_ids(self) -> NDArray:
        """Return current sample node IDs."""
        return self.sample_node_ids
        
    def init_node_ids(self):
        """Reset node IDs to 0."""
        self.sample_node_ids[:] = 0
        return self.sample_node_ids


# Decorate RayWorker with @ray.remote if ray is available
if ray:
    RayWorker = ray.remote(RayWorker)


class RayDistributedContext:
    """Ray-based distributed context."""
    
    def __init__(self, n_workers: int = None):
        if not ray:
            raise ImportError(
                "Distributed training requires Ray. "
                "Install with: pip install 'openboost[distributed]'"
            )
            
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            
        self.n_workers = n_workers or int(ray.available_resources().get('GPU', 1))
        if self.n_workers == 0:
             self.n_workers = int(ray.available_resources().get('CPU', 1))
             
        self.workers = []
        self.rank = 0 
    
    def setup(self, X: NDArray, y: NDArray, n_bins: int):
        """Partition data and create workers."""
        shards = np.array_split(X, self.n_workers)
        y_shards = np.array_split(y, self.n_workers)
        
        self.workers = [
            RayWorker.remote(s, ys, n_bins) 
            for s, ys in zip(shards, y_shards)
        ]
    
    def allreduce_histograms(self, local_hists_refs: List[Any]) -> Dict[int, Any]:
        """Sum histograms from all workers."""
        local_hists = ray.get(local_hists_refs)
        
        if not local_hists:
            return {}
            
        result = local_hists[0]
        
        for i in range(1, len(local_hists)):
            other = local_hists[i]
            for node_id, hist in other.items():
                if node_id not in result:
                    result[node_id] = hist
                else:
                    # Aggregate
                    target = result[node_id]
                    target.hist_grad += hist.hist_grad
                    target.hist_hess += hist.hist_hess
                    target.sum_grad += hist.sum_grad
                    target.sum_hess += hist.sum_hess
                    target.n_samples += hist.n_samples
        
        return result

    def broadcast_tree(self, tree: Any) -> Any:
        return tree
    
    def partition_data(self, X: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        raise NotImplementedError("Use setup() for Ray backend")
