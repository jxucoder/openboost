"""Shared training infrastructure for OpenBoost models.

Phase 13: Common training loop logic usable by all boosting models.

This module provides a generic training loop that handles:
- Callback management (early stopping, logging, etc.)
- Sample weights
- Validation set evaluation
- Train/val loss computation

Models only need to provide a `fit_round_fn` that fits one tree given gradients.

Example:
    >>> from openboost._training import run_training_loop, TrainingConfig
    >>> 
    >>> def fit_one_round(model, X, grad, hess):
    ...     return fit_tree(X, grad, hess, max_depth=model.max_depth)
    >>> 
    >>> config = TrainingConfig(n_rounds=100, callbacks=[EarlyStopping()])
    >>> run_training_loop(model, X, y, loss_fn, config, fit_one_round)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ._callbacks import Callback, CallbackManager, Logger, TrainingState

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class TrainingConfig:
    """Configuration for training loop.
    
    Args:
        n_rounds: Number of boosting rounds.
        learning_rate: Learning rate (shrinkage factor).
        callbacks: List of Callback instances.
        eval_set: List of (X, y) tuples for validation.
        sample_weight: Sample weights for training.
        verbose: Logging verbosity (0=silent, N=every N rounds).
        compute_train_loss: Whether to compute training loss each round.
    """
    n_rounds: int = 100
    learning_rate: float = 0.1
    callbacks: list[Callback] | None = None
    eval_set: list[tuple[NDArray, NDArray]] | None = None
    sample_weight: NDArray | None = None
    verbose: int = 0
    compute_train_loss: bool = True


@dataclass
class TrainingResult:
    """Result from training loop.
    
    Attributes:
        n_rounds_trained: Actual number of rounds (may be less if early stopped).
        stopped_early: Whether training was stopped early.
        final_train_loss: Training loss at end of training.
        final_val_loss: Validation loss at end of training (if eval_set provided).
        best_iteration: Best iteration (if early stopping used).
    """
    n_rounds_trained: int
    stopped_early: bool = False
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    best_iteration: int | None = None


# Type alias for fit function
FitRoundFn = Callable[[Any, Any, NDArray, NDArray], Any]


def run_training_loop(
    model: Any,
    X_binned: Any,
    y: NDArray,
    loss_fn: Callable[[NDArray, NDArray], tuple[NDArray, NDArray]],
    config: TrainingConfig,
    fit_round_fn: FitRoundFn,
    predict_fn: Callable[[Any, Any], NDArray] | None = None,
) -> TrainingResult:
    """Run generic training loop with callbacks.
    
    This is the SHARED training infrastructure. All boosting models can use it.
    Only the `fit_round_fn` differs between models.
    
    Args:
        model: The model being trained (modified in place).
            Must have `trees_` attribute (list) for storing trees.
        X_binned: Binned training features (BinnedArray or similar).
        y: Training targets.
        loss_fn: Function (pred, y) -> (grad, hess).
        config: Training configuration.
        fit_round_fn: Model-specific function to fit one round.
            Signature: (model, X_binned, grad, hess) -> tree
        predict_fn: Optional function to predict with model.
            Signature: (model, X) -> predictions
            If None, uses default accumulation of tree predictions.
            
    Returns:
        TrainingResult with training metadata.
        
    Example (in GradientBoosting.fit):
        >>> def fit_one_round(model, X, grad, hess):
        ...     return fit_tree(X, grad, hess, 
        ...         max_depth=model.max_depth,
        ...         min_child_weight=model.min_child_weight,
        ...         reg_lambda=model.reg_lambda,
        ...     )
        >>> 
        >>> config = TrainingConfig(
        ...     n_rounds=self.n_trees,
        ...     learning_rate=self.learning_rate,
        ...     callbacks=callbacks,
        ...     eval_set=eval_set,
        ... )
        >>> 
        >>> result = run_training_loop(
        ...     self, self.X_binned_, y, self._loss_fn, 
        ...     config, fit_one_round
        ... )
    """
    # Setup callbacks
    callbacks = list(config.callbacks or [])
    if config.verbose > 0:
        callbacks.append(Logger(period=config.verbose))
    
    cb_manager = CallbackManager(callbacks)
    
    # Initialize
    n_samples = len(y)
    pred = np.zeros(n_samples, dtype=np.float32)
    
    # Ensure model has trees_ attribute
    if not hasattr(model, 'trees_'):
        model.trees_ = []
    else:
        model.trees_ = []  # Reset for new fit
    
    # Create training state
    state = TrainingState(
        model=model,
        n_rounds=config.n_rounds,
    )
    
    # Call train begin
    cb_manager.on_train_begin(state)
    
    stopped_early = False
    
    # Training loop
    for i in range(config.n_rounds):
        state.round_idx = i
        
        # Round begin callback
        cb_manager.on_round_begin(state)
        
        # Compute gradients
        grad, hess = loss_fn(pred, y)
        
        # Ensure float32
        grad = np.asarray(grad, dtype=np.float32)
        hess = np.asarray(hess, dtype=np.float32)
        
        # Apply sample weights if provided
        if config.sample_weight is not None:
            weights = np.asarray(config.sample_weight, dtype=np.float32)
            grad = grad * weights
            hess = hess * weights
        
        # Store gradient info for research callbacks
        state.extra['grad_norm'] = float(np.linalg.norm(grad))
        state.extra['hess_mean'] = float(np.mean(hess))
        
        # Fit one round (model-specific)
        tree = fit_round_fn(model, X_binned, grad, hess)
        model.trees_.append(tree)
        
        # Update predictions
        tree_pred = tree(X_binned)
        if hasattr(tree_pred, 'copy_to_host'):
            tree_pred = tree_pred.copy_to_host()
        pred = pred + config.learning_rate * tree_pred
        
        # Compute train loss for callbacks
        if config.compute_train_loss:
            state.train_loss = _compute_loss(pred, y, loss_fn)
        
        # Compute validation loss if eval_set provided
        if config.eval_set:
            X_val, y_val = config.eval_set[0]
            if predict_fn is not None:
                val_pred = predict_fn(model, X_val)
            else:
                val_pred = _default_predict(model, X_val, config.learning_rate)
            state.val_loss = _compute_loss(val_pred, y_val, loss_fn)
        
        # Round end callbacks (may stop training)
        if not cb_manager.on_round_end(state):
            stopped_early = True
            break
    
    # Train end callbacks
    cb_manager.on_train_end(state)
    
    # Build result
    result = TrainingResult(
        n_rounds_trained=len(model.trees_),
        stopped_early=stopped_early,
        final_train_loss=state.train_loss,
        final_val_loss=state.val_loss,
    )
    
    # Copy early stopping attributes if present
    if hasattr(model, 'best_iteration_'):
        result.best_iteration = model.best_iteration_
    
    return result


def _compute_loss(pred: NDArray, y: NDArray, loss_fn) -> float:
    """Compute loss value for predictions.
    
    Uses the gradient to approximate loss value.
    For common losses: MSE = mean((pred - y)^2), etc.
    """
    # Simple approximation: use MSE-like loss
    # This is a reasonable default; specific losses can override
    diff = pred - y
    return float(np.mean(diff ** 2))


def _default_predict(model, X, learning_rate: float) -> NDArray:
    """Default prediction using accumulated trees."""
    from ._array import BinnedArray, array
    
    # Bin if needed
    if isinstance(X, BinnedArray):
        X_binned = X
    else:
        X_binned = array(X, n_bins=256)
    
    n_samples = X_binned.n_samples
    pred = np.zeros(n_samples, dtype=np.float32)
    
    for tree in model.trees_:
        tree_pred = tree(X_binned)
        if hasattr(tree_pred, 'copy_to_host'):
            tree_pred = tree_pred.copy_to_host()
        pred = pred + learning_rate * tree_pred
    
    return pred


def compute_eval_loss(
    model: Any,
    X: NDArray,
    y: NDArray,
    loss_fn,
    learning_rate: float = 0.1,
) -> float:
    """Compute loss on a dataset.
    
    Utility function for evaluating model on validation/test sets.
    
    Args:
        model: Fitted model with trees_.
        X: Features to evaluate on.
        y: True targets.
        loss_fn: Loss function (pred, y) -> (grad, hess).
        learning_rate: Model's learning rate.
        
    Returns:
        Loss value.
    """
    pred = _default_predict(model, X, learning_rate)
    return _compute_loss(pred, y, loss_fn)
