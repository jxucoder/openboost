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

from ._array import BinnedArray, array
from ._backends import is_cuda
from ._loss import get_loss_function, LossFunction
from ._predict import predict_tree_add_gpu
from ._tree import Tree, fit_tree_gpu_native, fit_tree

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
            - Callable: Custom function(pred, y) -> (grad, hess)
        min_child_weight: Minimum sum of hessian in a leaf.
        reg_lambda: L2 regularization on leaf values.
        n_bins: Number of bins for histogram building.
        
    Example:
        >>> import openboost as ob
        >>> model = ob.GradientBoosting(n_trees=100, loss='mse')
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        
        # With custom loss
        >>> def quantile_loss(pred, y, tau=0.5):
        ...     residual = y - pred
        ...     grad = np.where(residual > 0, -tau, 1 - tau)
        ...     hess = np.ones_like(pred)
        ...     return grad, hess
        >>> model = ob.GradientBoosting(n_trees=100, loss=quantile_loss)
    """
    
    n_trees: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    loss: str | LossFunction = 'mse'
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    n_bins: int = 256
    
    # Fitted attributes (not init)
    trees_: list[Tree] = field(default_factory=list, init=False, repr=False)
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
        
        # Get loss function
        self._loss_fn = get_loss_function(self.loss)
        
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
        """GPU-optimized batched training."""
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
            
            # Build tree
            tree = fit_tree_gpu_native(
                self.X_binned_,
                grad_gpu,
                hess_gpu,
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
            )
            
            # Update predictions in-place
            predict_tree_add_gpu(tree, self.X_binned_, pred_gpu, self.learning_rate)
            
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
        
        # Initialize predictions
        if is_cuda():
            from numba import cuda
            pred_gpu = cuda.device_array(n_samples, dtype=np.float32)
            _fill_zeros_gpu(pred_gpu)
            
            # Accumulate tree predictions
            for tree in self.trees_:
                predict_tree_add_gpu(tree, X_binned, pred_gpu, self.learning_rate)
            
            return pred_gpu.copy_to_host()
        else:
            pred = np.zeros(n_samples, dtype=np.float32)
            for tree in self.trees_:
                pred += self.learning_rate * tree(X_binned)
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

