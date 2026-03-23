"""Core tree building infrastructure.

This module contains the foundation for tree building:
- Primitives: histogram building, split finding, partitioning
- Growth strategies: level-wise, leaf-wise, symmetric
- fit_tree: main entry point for building trees
"""

from ._growth import (
    GrowthConfig,
    GrowthStrategy,
    LeafValues,
    LeafWiseGrowth,
    LevelWiseGrowth,
    ScalarLeaves,
    SymmetricGrowth,
    TreeStructure,
    VectorLeaves,
    get_growth_strategy,
)
from ._histogram import build_histogram
from ._predict import predict_ensemble
from ._primitives import (
    NodeHistogram,
    NodeSplit,
    build_node_histograms,
    compute_leaf_values,
    find_node_splits,
    get_children,
    get_nodes_at_depth,
    get_parent,
    init_sample_node_ids,
    partition_samples,
    subtract_histogram,
)
from ._split import SplitInfo, compute_leaf_value, find_best_split
from ._tree import (
    SymmetricTree,
    Tree,
    TreeNode,
    fit_tree,
    fit_tree_gpu_native,
    fit_tree_symmetric,
    fit_tree_symmetric_gpu_native,
    fit_trees_batch,
    predict_symmetric_tree,
    predict_tree,
)

__all__ = [
    # Primitives
    "NodeHistogram",
    "NodeSplit", 
    "build_node_histograms",
    "subtract_histogram",
    "find_node_splits",
    "partition_samples",
    "compute_leaf_values",
    "init_sample_node_ids",
    "get_nodes_at_depth",
    "get_children",
    "get_parent",
    # Growth
    "GrowthConfig",
    "GrowthStrategy",
    "TreeStructure",
    "LevelWiseGrowth",
    "LeafWiseGrowth",
    "SymmetricGrowth",
    "get_growth_strategy",
    "LeafValues",
    "ScalarLeaves",
    "VectorLeaves",
    # Tree
    "fit_tree",
    "fit_trees_batch",
    "Tree",
    "TreeNode",
    "SymmetricTree",
    "fit_tree_symmetric",
    "fit_tree_symmetric_gpu_native",
    "fit_tree_gpu_native",
    "predict_tree",
    "predict_symmetric_tree",
    # Low-level
    "build_histogram",
    "find_best_split",
    "compute_leaf_value",
    "SplitInfo",
    "predict_ensemble",
]
