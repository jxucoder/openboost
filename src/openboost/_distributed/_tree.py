"""Distributed tree fitting for OpenBoost.

Phase 12: Implements distributed tree building using histogram aggregation.
"""

from typing import Any, List, Optional
import numpy as np

try:
    import ray
except ImportError:
    ray = None

from openboost._core._growth import TreeStructure, GrowthConfig
from openboost._core._primitives import (
    find_node_splits, 
    compute_leaf_values,
    NodeHistogram,
    NodeSplit
)


def fit_tree_distributed(
    ctx: Any,  # DistributedContext
    workers: List[Any],
    grad_refs: List[Any],  # Ray object refs
    hess_refs: List[Any],
    *,
    max_depth: int = 6,
    min_child_weight: float = 1.0,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    min_gain: float = 0.0,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
) -> TreeStructure:
    """Distributed tree fitting (Level-wise)."""
    if ray is None:
        raise ImportError(
            "Distributed training requires Ray. "
            "Install with: pip install 'openboost[distributed]'"
        )
    
    # 1. Initialize
    sample_node_ids_refs = [w.init_node_ids.remote() for w in workers]
    
    n_features = get_worker_n_features(workers[0])
    
    # Initialize tree arrays (similar to LevelWiseGrowth)
    max_nodes = 2**(max_depth + 1) - 1
    features = np.full(max_nodes, -1, dtype=np.int32)
    thresholds = np.zeros(max_nodes, dtype=np.int32)
    values = np.zeros(max_nodes, dtype=np.float32)
    left_children = np.full(max_nodes, -1, dtype=np.int32)
    right_children = np.full(max_nodes, -1, dtype=np.int32)
    
    # 2. Grow tree level-wise
    active_nodes = [0]
    actual_depth = 0
    
    for depth in range(max_depth):
        if not active_nodes:
            break
            
        actual_depth = depth + 1
            
        # 3. Compute local histograms
        local_hists_refs = [
            w.compute_histograms.remote(g, h, active_nodes)
            for w, g, h in zip(workers, grad_refs, hess_refs)
        ]
        
        # 4. Aggregate histograms
        global_histograms = ctx.allreduce_histograms(local_hists_refs)
        
        # 5. Find splits
        splits = find_node_splits(
            global_histograms,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            min_gain=min_gain
        )
        
        # 6. Apply splits
        new_active_nodes = []
        
        for node_id, node_split in splits.items():
            features[node_id] = node_split.split.feature
            thresholds[node_id] = node_split.split.threshold
            left_children[node_id] = node_split.left_child
            right_children[node_id] = node_split.right_child
            
            new_active_nodes.append(node_split.left_child)
            new_active_nodes.append(node_split.right_child)
            
        # 7. Partition samples on workers
        if splits:
            partition_refs = [w.partition_samples.remote(splits) for w in workers]
        
        active_nodes = new_active_nodes
        
    # 8. Compute leaf values
    leaf_nodes = []
    for i in range(max_nodes):
        if left_children[i] == -1:
             if i == 0 or features[(i-1)//2] >= 0:
                 leaf_nodes.append(i)
    
    if leaf_nodes:
        local_hists_refs = [
            w.compute_histograms.remote(g, h, leaf_nodes)
            for w, g, h in zip(workers, grad_refs, hess_refs)
        ]
        leaf_histograms = ctx.allreduce_histograms(local_hists_refs)
        
        leaf_vals = compute_leaf_values_from_histograms(leaf_histograms, reg_lambda, reg_alpha)
        
        for node_id, val in leaf_vals.items():
            values[node_id] = val

    # Trim arrays
    n_nodes = count_nodes(left_children)
    
    return TreeStructure(
        features=features[:n_nodes] if n_nodes < max_nodes else features,
        thresholds=thresholds[:n_nodes] if n_nodes < max_nodes else thresholds,
        values=values[:n_nodes] if n_nodes < max_nodes else values,
        left_children=left_children[:n_nodes] if n_nodes < max_nodes else left_children,
        right_children=right_children[:n_nodes] if n_nodes < max_nodes else right_children,
        n_nodes=n_nodes,
        depth=actual_depth,
        n_features=n_features,
    )


def compute_leaf_values_from_histograms(histograms: dict, reg_lambda: float, reg_alpha: float) -> dict:
    from openboost._core._split import compute_leaf_value
    result = {}
    for node_id, hist in histograms.items():
        result[node_id] = compute_leaf_value(hist.sum_grad, hist.sum_hess, reg_lambda, reg_alpha)
    return result


def get_worker_n_features(worker):
    if ray is None:
        raise ImportError(
            "Distributed training requires Ray. "
            "Install with: pip install 'openboost[distributed]'"
        )
    return ray.get(worker.get_n_features.remote())


def count_nodes(left_children):
    for i in range(len(left_children) - 1, -1, -1):
        if i == 0 or left_children[(i - 1) // 2] != -1 or ((i - 1) // 2 == 0):
            if i == 0: return 1
            parent = (i - 1) // 2
            if left_children[parent] != -1:
                return max(left_children[parent], left_children[parent] + 1, i) + 1
    return 1
