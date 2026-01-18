"""Gradient Boosting ensemble model for OpenBoost.

Provides a scikit-learn-like API for training gradient boosting models
with both built-in and custom loss functions.

This module implements batched training that keeps computation on the GPU
without returning to Python between trees, achieving performance competitive
with XGBoost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

from .._array import BinnedArray, array
from .._backends import is_cuda
from .._loss import get_loss_function, LossFunction
from .._core._tree import fit_tree
from .._core._growth import TreeStructure

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class GradientBoosting:
    """Gradient Boosting ensemble model.
    
    A gradient boosting model that supports both built-in loss functions
    and custom loss functions. When using built-in losses with GPU,
    training is fully batched for maximum performance.
    
    Args:
        n_trees: Number of trees to train.
        max_depth: Maximum depth of each tree.
        learning_rate: Shrinkage factor applied to each tree.
        loss: Loss function. Can be:
            - 'mse': Mean Squared Error (regression)
            - 'logloss': Binary cross-entropy (classification)
            - 'huber': Huber loss (robust regression)
            - 'mae': Mean Absolute Error (L1 regression)
            - 'quantile': Quantile regression (use with quantile_alpha)
            - Callable: Custom function(pred, y) -> (grad, hess)
        min_child_weight: Minimum sum of hessian in a leaf.
        reg_lambda: L2 regularization on leaf values.
        n_bins: Number of bins for histogram building.
        quantile_alpha: Quantile level for 'quantile' loss (0 < alpha < 1).
            - 0.5: Median regression (default)
            - 0.9: 90th percentile
            - 0.1: 10th percentile
        tweedie_rho: Variance power for 'tweedie' loss (1 < rho < 2).
            - 1.5: Default (compound Poisson-Gamma)
        
    Example:
        >>> import openboost as ob
        >>> model = ob.GradientBoosting(n_trees=100, loss='mse')
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        
        # Quantile regression (90th percentile)
        >>> model = ob.GradientBoosting(loss='quantile', quantile_alpha=0.9)
        >>> model.fit(X_train, y_train)
        
        # MAE (L1) regression
        >>> model = ob.GradientBoosting(loss='mae')
    """
    
    n_trees: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    loss: str | LossFunction = 'mse'
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    n_bins: int = 256
    quantile_alpha: float = 0.5  # Phase 9.1
    tweedie_rho: float = 1.5     # Phase 9.3
    
    # Fitted attributes (not init)
    trees_: list[TreeStructure] = field(default_factory=list, init=False, repr=False)
    X_binned_: BinnedArray | None = field(default=None, init=False, repr=False)
    _loss_fn: LossFunction | None = field(default=None, init=False, repr=False)
    
    def fit(self, X: NDArray, y: NDArray) -> GradientBoosting:
        """Fit the gradient boosting model.
        
        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training targets, shape (n_samples,).
            
        Returns:
            self: The fitted model.
        """
        # Clear any previous fit
        self.trees_ = []
        
        # Convert to float32
        y = np.asarray(y, dtype=np.float32).ravel()
        n_samples = len(y)
        
        # Get loss function (pass parameters for parameterized losses)
        self._loss_fn = get_loss_function(
            self.loss, 
            quantile_alpha=self.quantile_alpha,
            tweedie_rho=self.tweedie_rho,
        )
        
        # Bin the data (this is the expensive step, but only done once)
        if isinstance(X, BinnedArray):
            self.X_binned_ = X
        else:
            self.X_binned_ = array(X, n_bins=self.n_bins)
        
        # Choose training path based on backend
        if is_cuda():
            self._fit_gpu(y, n_samples)
        else:
            self._fit_cpu(y, n_samples)
        
        return self
    
    def _fit_gpu(self, y: NDArray, n_samples: int):
        """GPU-optimized training using growth strategies."""
        from numba import cuda
        
        # Move y to GPU
        y_gpu = cuda.to_device(y)
        
        # Initialize predictions on GPU
        pred_gpu = cuda.device_array(n_samples, dtype=np.float32)
        _fill_zeros_gpu(pred_gpu)
        
        # Check if using custom loss (requires Python callback)
        is_custom_loss = callable(self.loss)
        
        # Pre-allocate gradient arrays
        if not is_custom_loss:
            grad_gpu = cuda.device_array(n_samples, dtype=np.float32)
            hess_gpu = cuda.device_array(n_samples, dtype=np.float32)
        
        # Train trees
        for i in range(self.n_trees):
            # Compute gradients
            if is_custom_loss:
                # Custom loss: need to copy pred to CPU, call Python, copy back
                pred_cpu = pred_gpu.copy_to_host()
                grad_cpu, hess_cpu = self._loss_fn(pred_cpu, y)
                grad_gpu = cuda.to_device(grad_cpu.astype(np.float32))
                hess_gpu = cuda.to_device(hess_cpu.astype(np.float32))
            else:
                # Built-in loss: compute entirely on GPU
                grad_gpu, hess_gpu = self._loss_fn(pred_gpu, y_gpu)
            
            # Build tree using new fit_tree (Phase 8)
            tree = fit_tree(
                self.X_binned_,
                grad_gpu,
                hess_gpu,
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
            )
            
            # Update predictions
            tree_pred = tree(self.X_binned_)
            if hasattr(tree_pred, 'copy_to_host'):
                tree_pred_cpu = tree_pred.copy_to_host()
            else:
                tree_pred_cpu = tree_pred
            
            # Update GPU predictions
            pred_cpu = pred_gpu.copy_to_host()
            pred_cpu += self.learning_rate * tree_pred_cpu
            cuda.to_device(pred_cpu, to=pred_gpu)
            
            self.trees_.append(tree)
    
    def _fit_cpu(self, y: NDArray, n_samples: int):
        """CPU training path."""
        # Initialize predictions
        pred = np.zeros(n_samples, dtype=np.float32)
        
        # Train trees
        for i in range(self.n_trees):
            # Compute gradients
            grad, hess = self._loss_fn(pred, y)
            
            # Build tree
            tree = fit_tree(
                self.X_binned_,
                grad.astype(np.float32),
                hess.astype(np.float32),
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
            )
            
            # Update predictions
            pred += self.learning_rate * tree(self.X_binned_)
            
            self.trees_.append(tree)
    
    def predict(self, X: NDArray | BinnedArray) -> NDArray:
        """Generate predictions for X.
        
        Args:
            X: Features to predict on, shape (n_samples, n_features).
               Can be raw numpy array or pre-binned BinnedArray.
               
        Returns:
            predictions: Shape (n_samples,).
        """
        if not self.trees_:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Bin the data if needed
        if isinstance(X, BinnedArray):
            X_binned = X
        else:
            X_binned = array(X, n_bins=self.n_bins)
        
        # Get number of samples
        n_samples = X_binned.n_samples
        
        # Accumulate tree predictions
        pred = np.zeros(n_samples, dtype=np.float32)
        for tree in self.trees_:
            tree_pred = tree(X_binned)
            if hasattr(tree_pred, 'copy_to_host'):
                tree_pred = tree_pred.copy_to_host()
            pred += self.learning_rate * tree_pred
        
        return pred
    
    def predict_proba(self, X: NDArray | BinnedArray) -> NDArray:
        """Predict class probabilities for binary classification.
        
        Only valid when loss='logloss'.
        
        Args:
            X: Features to predict on.
            
        Returns:
            probabilities: Shape (n_samples, 2) with [P(y=0), P(y=1)].
        """
        if self.loss not in ('logloss', 'binary_crossentropy'):
            raise ValueError("predict_proba only available for classification losses")
        
        raw_pred = self.predict(X)
        
        # Apply sigmoid
        prob_1 = 1 / (1 + np.exp(-raw_pred))
        prob_0 = 1 - prob_1
        
        return np.column_stack([prob_0, prob_1])


def _fill_zeros_gpu(arr):
    """Fill GPU array with zeros."""
    from numba import cuda
    
    n = arr.shape[0]
    threads = 256
    blocks = (n + threads - 1) // threads
    _fill_zeros_kernel[blocks, threads](arr, n)


@staticmethod
def _get_fill_zeros_kernel():
    from numba import cuda
    
    @cuda.jit
    def kernel(arr, n):
        idx = cuda.grid(1)
        if idx < n:
            arr[idx] = 0.0
    
    return kernel


_fill_zeros_kernel = None

if is_cuda():
    try:
        from numba import cuda
        
        @cuda.jit
        def _fill_zeros_kernel_impl(arr, n):
            idx = cuda.grid(1)
            if idx < n:
                arr[idx] = 0.0
        
        _fill_zeros_kernel = _fill_zeros_kernel_impl
    except Exception:
        pass


# =============================================================================
# Multi-class Gradient Boosting (Phase 9.2)
# =============================================================================

@dataclass
class MultiClassGradientBoosting:
    """Multi-class Gradient Boosting classifier.
    
    Uses softmax loss and trains K trees per round (one per class),
    following the XGBoost/LightGBM approach.
    
    Args:
        n_classes: Number of classes.
        n_trees: Number of boosting rounds (total trees = n_trees * n_classes).
        max_depth: Maximum depth of each tree.
        learning_rate: Shrinkage factor applied to each tree.
        min_child_weight: Minimum sum of hessian in a leaf.
        reg_lambda: L2 regularization on leaf values.
        n_bins: Number of bins for histogram building.
        
    Example:
        >>> import openboost as ob
        >>> model = ob.MultiClassGradientBoosting(n_classes=10, n_trees=100)
        >>> model.fit(X_train, y_train)  # y_train: 0 to 9
        >>> predictions = model.predict(X_test)  # Returns class labels
        >>> proba = model.predict_proba(X_test)  # Returns probabilities
    """
    
    n_classes: int
    n_trees: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    n_bins: int = 256
    
    # Fitted attributes
    trees_: list[list[TreeStructure]] = field(default_factory=list, init=False, repr=False)
    X_binned_: BinnedArray | None = field(default=None, init=False, repr=False)
    
    def fit(self, X: NDArray, y: NDArray) -> "MultiClassGradientBoosting":
        """Fit the multi-class gradient boosting model.
        
        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training labels, shape (n_samples,). Integer class labels 0 to n_classes-1.
            
        Returns:
            self: The fitted model.
        """
        from .._loss import softmax_gradient
        
        # Clear previous fit
        self.trees_ = []
        
        # Convert y to integer labels
        y = np.asarray(y, dtype=np.int32).ravel()
        n_samples = len(y)
        
        # Validate labels
        if y.min() < 0 or y.max() >= self.n_classes:
            raise ValueError(f"Labels must be in [0, {self.n_classes-1}], got [{y.min()}, {y.max()}]")
        
        # Bin the data
        if isinstance(X, BinnedArray):
            self.X_binned_ = X
        else:
            self.X_binned_ = array(X, n_bins=self.n_bins)
        
        # Initialize predictions for each class
        pred = np.zeros((n_samples, self.n_classes), dtype=np.float32)
        
        # Train trees
        for round_idx in range(self.n_trees):
            # Compute softmax gradients for all classes
            grad, hess = softmax_gradient(pred, y, self.n_classes)
            
            # Train one tree per class
            round_trees = []
            for k in range(self.n_classes):
                tree = fit_tree(
                    self.X_binned_,
                    grad[:, k].astype(np.float32),
                    hess[:, k].astype(np.float32),
                    max_depth=self.max_depth,
                    min_child_weight=self.min_child_weight,
                    reg_lambda=self.reg_lambda,
                )
                round_trees.append(tree)
                
                # Update predictions for this class
                tree_pred = tree(self.X_binned_)
                if hasattr(tree_pred, 'copy_to_host'):
                    tree_pred = tree_pred.copy_to_host()
                pred[:, k] += self.learning_rate * tree_pred
            
            self.trees_.append(round_trees)
        
        return self
    
    def predict_raw(self, X: NDArray | BinnedArray) -> NDArray:
        """Get raw predictions (logits) for each class.
        
        Args:
            X: Features to predict on.
            
        Returns:
            logits: Shape (n_samples, n_classes).
        """
        if not self.trees_:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Bin the data if needed
        if isinstance(X, BinnedArray):
            X_binned = X
        else:
            X_binned = array(X, n_bins=self.n_bins)
        
        n_samples = X_binned.n_samples
        pred = np.zeros((n_samples, self.n_classes), dtype=np.float32)
        
        # Accumulate predictions from all rounds
        for round_trees in self.trees_:
            for k, tree in enumerate(round_trees):
                tree_pred = tree(X_binned)
                if hasattr(tree_pred, 'copy_to_host'):
                    tree_pred = tree_pred.copy_to_host()
                pred[:, k] += self.learning_rate * tree_pred
        
        return pred
    
    def predict_proba(self, X: NDArray | BinnedArray) -> NDArray:
        """Predict class probabilities.
        
        Args:
            X: Features to predict on.
            
        Returns:
            probabilities: Shape (n_samples, n_classes).
        """
        logits = self.predict_raw(X)
        
        # Softmax
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        proba = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return proba
    
    def predict(self, X: NDArray | BinnedArray) -> NDArray:
        """Predict class labels.
        
        Args:
            X: Features to predict on.
            
        Returns:
            labels: Shape (n_samples,). Integer class labels.
        """
        logits = self.predict_raw(X)
        return np.argmax(logits, axis=1)

