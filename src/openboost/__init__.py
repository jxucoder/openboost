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

# Core API
from ._array import BinnedArray, array, as_numba_array
from ._tree import (
    Tree, fit_tree, fit_tree_gpu_native,
    predict_tree, fit_trees_batch,
    # Symmetric trees
    SymmetricTree, fit_tree_symmetric, fit_tree_symmetric_gpu_native, predict_symmetric_tree,
)
from ._predict import predict_ensemble

# High-level API (scikit-learn-like)
from ._boosting import GradientBoosting
from ._gam import OpenBoostGAM

# Loss functions
from ._loss import mse_gradient, logloss_gradient, huber_gradient, get_loss_function

# Batch training (low-level)
from ._batch import ConfigBatch, BatchTrainingState

# Backend control
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
    "OpenBoostGAM",  # GPU-accelerated interpretable GAM (EBM-style)
    # Loss functions
    "mse_gradient",
    "logloss_gradient",
    "huber_gradient",
    "get_loss_function",
    # Training (single tree, low-level)
    "fit_tree",              # Auto-dispatches to GPU-native when on GPU
    "fit_tree_gpu_native",   # Explicit GPU-native (advanced)
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
    # Prediction
    "predict_tree",
    "predict_ensemble",
    # Backend
    "get_backend",
    "set_backend",
    "is_cuda",
    "is_cpu",
]

