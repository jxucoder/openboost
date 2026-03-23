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

import warnings as _warnings

__version__ = "1.0.0rc1"

# =============================================================================
# Data Layer
# =============================================================================
from ._array import MISSING_BIN, BinnedArray, array, as_numba_array

# =============================================================================
# Core (Foundation)
# =============================================================================
from ._core import (
    # Growth strategies (Phase 8.2)
    GrowthConfig,
    GrowthStrategy,
    # Leaf value abstractions (Phase 9.0)
    LeafValues,
    LeafWiseGrowth,
    LevelWiseGrowth,
    # Primitives (Phase 8.1)
    NodeHistogram,
    NodeSplit,
    ScalarLeaves,
    SymmetricGrowth,
    # Symmetric trees
    SymmetricTree,
    TreeNode,
    TreeStructure,
    VectorLeaves,
    build_node_histograms,
    compute_leaf_values,
    find_node_splits,
    # Tree building
    fit_tree,
    fit_tree_gpu_native,
    fit_tree_symmetric,
    fit_tree_symmetric_gpu_native,
    fit_trees_batch,
    get_children,
    get_growth_strategy,
    get_nodes_at_depth,
    get_parent,
    init_sample_node_ids,
    partition_samples,
    # Prediction
    predict_ensemble,
    predict_symmetric_tree,
    predict_tree,
    subtract_histogram,
)
from ._core import (
    Tree as LegacyTree,
)

# Phase 8: TreeStructure is the new Tree
Tree = TreeStructure  # Alias for backward compatibility

# =============================================================================
# Models (High-Level)
# =============================================================================
# =============================================================================
# Backend Control
# =============================================================================
from ._backends import get_backend, is_cpu, is_cuda, set_backend

# =============================================================================
# Callbacks (Phase 13)
# =============================================================================
from ._callbacks import (
    Callback,
    CallbackManager,
    EarlyStopping,
    HistoryCallback,
    LearningRateScheduler,
    Logger,
    ModelCheckpoint,
    TrainingState,
)

# =============================================================================
# Multi-GPU Training (Phase 18)
# =============================================================================
from ._distributed import (
    GPUWorker,
    GPUWorkerBase,
    MultiGPUContext,
    fit_tree_multigpu,
)

# =============================================================================
# Distributions (Phase 15)
# =============================================================================
from ._distributions import (
    # Custom distributions with autodiff
    CustomDistribution,
    Distribution,
    DistributionOutput,
    Gamma,
    LogNormal,
    NegativeBinomial,
    Normal,
    Poisson,
    StudentT,
    # Kaggle competition favorites
    Tweedie,
    create_custom_distribution,
    get_distribution,
    list_distributions,
)

# =============================================================================
# Feature Importance (Phase 13)
# =============================================================================
from ._importance import (
    compute_feature_importances,
    get_feature_importance_dict,
    plot_feature_importances,
)

# =============================================================================
# Loss Functions
# =============================================================================
from ._loss import (
    gamma_gradient,  # Phase 9.3
    get_loss_function,
    huber_gradient,
    logloss_gradient,
    mae_gradient,  # Phase 9.1
    mse_gradient,
    poisson_gradient,  # Phase 9.3
    quantile_gradient,  # Phase 9.1
    softmax_gradient,  # Phase 9.2
    tweedie_gradient,  # Phase 9.3
)
from ._models import (
    DART,
    BatchTrainingState,
    ConfigBatch,
    # Phase 15/16: Distributional GBDT (NaturalBoost)
    DistributionalGBDT,
    GradientBoosting,
    # Phase 15: Linear Leaf GBDT
    LinearLeafGBDT,
    MultiClassGradientBoosting,
    NaturalBoost,
    NaturalBoostGamma,
    NaturalBoostLogNormal,
    NaturalBoostNegBin,
    NaturalBoostNormal,
    NaturalBoostPoisson,
    NaturalBoostStudentT,
    NaturalBoostTweedie,
    OpenBoostClassifier,
    # Phase 15: sklearn wrappers for new models
    OpenBoostDistributionalRegressor,
    OpenBoostGAM,
    OpenBoostLinearLeafRegressor,
    # Phase 13: sklearn-compatible wrappers
    OpenBoostRegressor,
)
from ._models import (
    # Backward compatibility aliases (deprecated, accessed via __getattr__)
    NGBoost as _NGBoost,
)
from ._models import (
    NGBoostGamma as _NGBoostGamma,
)
from ._models import (
    NGBoostLogNormal as _NGBoostLogNormal,
)
from ._models import (
    NGBoostNegBin as _NGBoostNegBin,
)
from ._models import (
    NGBoostNormal as _NGBoostNormal,
)
from ._models import (
    NGBoostPoisson as _NGBoostPoisson,
)
from ._models import (
    NGBoostStudentT as _NGBoostStudentT,
)
from ._models import (
    NGBoostTweedie as _NGBoostTweedie,
)
from ._profiler import ProfilingCallback

# =============================================================================
# Sampling Strategies (Phase 17)
# =============================================================================
from ._sampling import (
    GOSSConfig,
    MiniBatchConfig,
    MiniBatchIterator,
    SamplingResult,
    SamplingStrategy,
    accumulate_histograms_minibatch,
    apply_sampling,
    create_memmap_binned,
    goss_sample,
    load_memmap_binned,
    random_sample,
)

# =============================================================================
# Utilities (Phase 20.6)
# =============================================================================
# =============================================================================
# Evaluation Metrics (Phase 22)
# =============================================================================
# =============================================================================
# Probabilistic/Distributional Metrics (Phase 22 Sprint 2)
# =============================================================================
from ._utils import (
    PARAM_GRID_CLASSIFICATION,
    PARAM_GRID_DISTRIBUTIONAL,
    PARAM_GRID_REGRESSION,
    accuracy_score,
    brier_score,
    calibration_curve,
    cross_val_predict,
    cross_val_predict_interval,
    cross_val_predict_proba,
    crps_empirical,
    crps_gaussian,
    evaluate_coverage,
    expected_calibration_error,
    f1_score,
    get_param_grid,
    interval_score,
    log_loss_score,
    mae_score,
    mse_score,
    negative_log_likelihood,
    pinball_loss,
    precision_score,
    r2_score,
    recall_score,
    rmse_score,
    roc_auc_score,
    suggest_params,
)

_DEPRECATED_ALIASES = {
    "NGBoost": ("NaturalBoost", _NGBoost),
    "NGBoostNormal": ("NaturalBoostNormal", _NGBoostNormal),
    "NGBoostLogNormal": ("NaturalBoostLogNormal", _NGBoostLogNormal),
    "NGBoostGamma": ("NaturalBoostGamma", _NGBoostGamma),
    "NGBoostPoisson": ("NaturalBoostPoisson", _NGBoostPoisson),
    "NGBoostStudentT": ("NaturalBoostStudentT", _NGBoostStudentT),
    "NGBoostTweedie": ("NaturalBoostTweedie", _NGBoostTweedie),
    "NGBoostNegBin": ("NaturalBoostNegBin", _NGBoostNegBin),
}


def __getattr__(name: str):
    if name in _DEPRECATED_ALIASES:
        new_name, obj = _DEPRECATED_ALIASES[name]
        _warnings.warn(
            f"{name} is deprecated, use {new_name} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return obj
    raise AttributeError(f"module 'openboost' has no attribute {name!r}")


__all__ = [
    # Version
    "__version__",
    # Data
    "array",
    "BinnedArray",
    "as_numba_array",
    "MISSING_BIN",
    # High-level API (recommended)
    "GradientBoosting",
    "MultiClassGradientBoosting",
    "OpenBoostGAM",
    "DART",
    # Phase 15/16: Distributional GBDT (NaturalBoost)
    "DistributionalGBDT",
    "NaturalBoost",
    "NaturalBoostNormal",
    "NaturalBoostLogNormal",
    "NaturalBoostGamma",
    "NaturalBoostPoisson",
    "NaturalBoostStudentT",
    "NaturalBoostTweedie",
    "NaturalBoostNegBin",
    "LegacyTree",
    # Backward compatibility (deprecated)
    "NGBoost",
    "NGBoostNormal",
    "NGBoostLogNormal",
    "NGBoostGamma",
    "NGBoostPoisson",
    "NGBoostStudentT",
    "NGBoostTweedie",
    "NGBoostNegBin",
    # Phase 15: Linear Leaf GBDT
    "LinearLeafGBDT",
    # Phase 15: Distributions
    "Distribution",
    "DistributionOutput",
    "Normal",
    "LogNormal",
    "Gamma",
    "Poisson",
    "StudentT",
    # Kaggle competition favorites
    "Tweedie",
    "NegativeBinomial",
    # Custom distributions
    "CustomDistribution",
    "create_custom_distribution",
    "get_distribution",
    "list_distributions",
    # sklearn-compatible wrappers (Phase 13 + 15)
    "OpenBoostRegressor",
    "OpenBoostClassifier",
    "OpenBoostDistributionalRegressor",
    "OpenBoostLinearLeafRegressor",
    # Callbacks (Phase 13)
    "Callback",
    "EarlyStopping",
    "Logger",
    "ModelCheckpoint",
    "LearningRateScheduler",
    "HistoryCallback",
    "CallbackManager",
    "TrainingState",
    "ProfilingCallback",
    # Feature importance (Phase 13)
    "compute_feature_importances",
    "get_feature_importance_dict",
    "plot_feature_importances",
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
    "TreeNode",
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
    # Sampling (Phase 17)
    "SamplingStrategy",
    "GOSSConfig",
    "MiniBatchConfig",
    "SamplingResult",
    "goss_sample",
    "random_sample",
    "apply_sampling",
    "MiniBatchIterator",
    "accumulate_histograms_minibatch",
    "create_memmap_binned",
    "load_memmap_binned",
    # Multi-GPU (Phase 18)
    "MultiGPUContext",
    "GPUWorkerBase",
    "GPUWorker",
    "fit_tree_multigpu",
    # Utilities (Phase 20.6)
    "suggest_params",
    "cross_val_predict",
    "cross_val_predict_proba",
    "cross_val_predict_interval",
    "evaluate_coverage",
    "get_param_grid",
    "PARAM_GRID_REGRESSION",
    "PARAM_GRID_CLASSIFICATION",
    "PARAM_GRID_DISTRIBUTIONAL",
    # Evaluation Metrics (Phase 22)
    "roc_auc_score",
    "accuracy_score",
    "log_loss_score",
    "mse_score",
    "r2_score",
    "mae_score",
    "rmse_score",
    "f1_score",
    "precision_score",
    "recall_score",
    # Probabilistic/Distributional Metrics (Phase 22 Sprint 2)
    "crps_gaussian",
    "crps_empirical",
    "brier_score",
    "pinball_loss",
    "interval_score",
    "expected_calibration_error",
    "calibration_curve",
    "negative_log_likelihood",
]
