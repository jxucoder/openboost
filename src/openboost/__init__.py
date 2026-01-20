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

__version__ = "1.0.0rc1"

# =============================================================================
# Data Layer
# =============================================================================
from ._array import BinnedArray, array, as_numba_array, MISSING_BIN

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
    # Phase 13: sklearn-compatible wrappers
    OpenBoostRegressor,
    OpenBoostClassifier,
    # Phase 15: sklearn wrappers for new models
    OpenBoostDistributionalRegressor,
    OpenBoostLinearLeafRegressor,
    # Phase 15/16: Distributional GBDT (NaturalBoost)
    DistributionalGBDT,
    NaturalBoost,
    NaturalBoostNormal,
    NaturalBoostLogNormal,
    NaturalBoostGamma,
    NaturalBoostPoisson,
    NaturalBoostStudentT,
    NaturalBoostTweedie,
    NaturalBoostNegBin,
    # Backward compatibility aliases (deprecated)
    NGBoost,
    NGBoostNormal,
    NGBoostLogNormal,
    NGBoostGamma,
    NGBoostPoisson,
    NGBoostStudentT,
    NGBoostTweedie,
    NGBoostNegBin,
    # Phase 15: Linear Leaf GBDT
    LinearLeafGBDT,
)

# =============================================================================
# Distributions (Phase 15)
# =============================================================================
from ._distributions import (
    Distribution,
    DistributionOutput,
    Normal,
    LogNormal,
    Gamma,
    Poisson,
    StudentT,
    # Kaggle competition favorites
    Tweedie,
    NegativeBinomial,
    # Custom distributions with autodiff
    CustomDistribution,
    create_custom_distribution,
    get_distribution,
    list_distributions,
)

# =============================================================================
# Callbacks (Phase 13)
# =============================================================================
from ._callbacks import (
    Callback,
    EarlyStopping,
    Logger,
    ModelCheckpoint,
    LearningRateScheduler,
    HistoryCallback,
    CallbackManager,
    TrainingState,
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

# =============================================================================
# Sampling Strategies (Phase 17)
# =============================================================================
from ._sampling import (
    SamplingStrategy,
    GOSSConfig,
    MiniBatchConfig,
    SamplingResult,
    goss_sample,
    random_sample,
    apply_sampling,
    MiniBatchIterator,
    accumulate_histograms_minibatch,
    create_memmap_binned,
    load_memmap_binned,
)

# =============================================================================
# Multi-GPU Training (Phase 18)
# =============================================================================
from ._distributed import (
    MultiGPUContext,
    GPUWorkerBase,
    GPUWorker,
    fit_tree_multigpu,
)

# =============================================================================
# Utilities (Phase 20.6)
# =============================================================================
from ._utils import (
    suggest_params,
    cross_val_predict,
    cross_val_predict_proba,
    cross_val_predict_interval,
    evaluate_coverage,
    get_param_grid,
    PARAM_GRID_REGRESSION,
    PARAM_GRID_CLASSIFICATION,
    PARAM_GRID_DISTRIBUTIONAL,
)

# =============================================================================
# Evaluation Metrics (Phase 22)
# =============================================================================
from ._utils import (
    roc_auc_score,
    accuracy_score,
    log_loss_score,
    mse_score,
    r2_score,
    mae_score,
    rmse_score,
    f1_score,
    precision_score,
    recall_score,
)

# =============================================================================
# Probabilistic/Distributional Metrics (Phase 22 Sprint 2)
# =============================================================================
from ._utils import (
    crps_gaussian,
    crps_empirical,
    brier_score,
    pinball_loss,
    interval_score,
    expected_calibration_error,
    calibration_curve,
    negative_log_likelihood,
)

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
