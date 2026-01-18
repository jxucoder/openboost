"""OpenBoost: The PyTorch of Gradient Boosting.

Train-many optimized, research-friendly, GPU-accelerated gradient boosting.

Quick Start (Batched Training):
    >>> import openboost as ob
    >>>
    >>> # Simple scikit-learn-like API
    >>> model = ob.GradientBoosting(n_trees=100, loss='mse')
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)

Custom Loss Functions:
    >>> def quantile_loss(pred, y, tau=0.5):
    ...     residual = y - pred
    ...     grad = np.where(residual > 0, -tau, 1 - tau)
    ...     hess = np.ones_like(pred)
    ...     return grad, hess
    >>> model = ob.GradientBoosting(n_trees=100, loss=quantile_loss)
    >>> model.fit(X_train, y_train)

Low-Level API (Full Control):
    >>> # Bin data once, reuse everywhere
    >>> X_binned = ob.array(X_train)
    >>>
    >>> # You own the training loop
    >>> pred = np.zeros(len(y_train))
    >>> for round in range(100):
    ...     grad = 2 * (pred - y_train)  # Your loss, your gradients
    ...     hess = np.ones_like(grad) * 2
    ...     tree = ob.fit_tree(X_binned, grad, hess)
    ...     pred = pred + 0.1 * tree(X_binned)
"""

__version__ = "0.5.0"  # Phase 5 - GradientBoosting batch mode

# =============================================================================
# Data Layer
# =============================================================================
from ._array import BinnedArray, array, as_numba_array

# =============================================================================
# Core (Foundation)
# =============================================================================
from ._core import (
    # Growth strategies (Phase 8.2)
    GrowthConfig,
    GrowthStrategy,
    TreeStructure,
    LevelWiseGrowth,
    LeafWiseGrowth,
    SymmetricGrowth,
    get_growth_strategy,
    # Leaf value abstractions (Phase 9.0)
    LeafValues,
    ScalarLeaves,
    VectorLeaves,
    # Tree building
    fit_tree,
    fit_trees_batch,
    Tree as LegacyTree,
    TreeNode,
    fit_tree_gpu_native,
    predict_tree,
    # Symmetric trees
    SymmetricTree,
    fit_tree_symmetric,
    fit_tree_symmetric_gpu_native,
    predict_symmetric_tree,
    # Primitives (Phase 8.1)
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
    get_parent,
    # Prediction
    predict_ensemble,
)

# Phase 8: TreeStructure is the new Tree
Tree = TreeStructure  # Alias for backward compatibility

# =============================================================================
# Models (High-Level)
# =============================================================================
from ._models import (
    GradientBoosting,
    MultiClassGradientBoosting,
    DART,
    OpenBoostGAM,
    ConfigBatch,
    BatchTrainingState,
)

# =============================================================================
# Loss Functions
# =============================================================================
from ._loss import (
    mse_gradient,
    logloss_gradient,
    huber_gradient,
    mae_gradient,        # Phase 9.1
    quantile_gradient,   # Phase 9.1
    poisson_gradient,    # Phase 9.3
    gamma_gradient,      # Phase 9.3
    tweedie_gradient,    # Phase 9.3
    softmax_gradient,    # Phase 9.2
    get_loss_function,
)

# =============================================================================
# Backend Control
# =============================================================================
from ._backends import get_backend, set_backend, is_cuda, is_cpu

__all__ = [
    # Version
    "__version__",
    # Data
    "array",
    "BinnedArray",
    "as_numba_array",
    # High-level API (recommended)
    "GradientBoosting",
    "MultiClassGradientBoosting",
    "OpenBoostGAM",
    "DART",
    # Loss functions
    "mse_gradient",
    "logloss_gradient",
    "huber_gradient",
    "mae_gradient",
    "quantile_gradient",
    "poisson_gradient",
    "gamma_gradient",
    "tweedie_gradient",
    "softmax_gradient",
    "get_loss_function",
    # Training (single tree, low-level)
    "fit_tree",
    "fit_tree_gpu_native",
    "Tree",
    # Training (symmetric/oblivious trees)
    "fit_tree_symmetric",
    "fit_tree_symmetric_gpu_native",
    "SymmetricTree",
    "predict_symmetric_tree",
    # Training (batch, low-level)
    "fit_trees_batch",
    "ConfigBatch",
    "BatchTrainingState",
    # Tree building primitives (Phase 8.1)
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
    # Growth strategies (Phase 8.2)
    "GrowthConfig",
    "GrowthStrategy",
    "TreeStructure",
    "LevelWiseGrowth",
    "LeafWiseGrowth",
    "SymmetricGrowth",
    "get_growth_strategy",
    # Leaf value abstractions (Phase 9.0)
    "LeafValues",
    "ScalarLeaves",
    "VectorLeaves",
    # Prediction
    "predict_tree",
    "predict_ensemble",
    # Backend
    "get_backend",
    "set_backend",
    "is_cuda",
    "is_cpu",
]
