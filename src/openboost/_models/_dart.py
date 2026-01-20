"""DART: Dropouts meet Multiple Additive Regression Trees.

Phase 8.5: Proof that the new architecture enables easy algorithm variants.

DART is a regularization technique where random trees are dropped during
training, similar to dropout in neural networks. This prevents later trees
from simply fixing errors of earlier trees, leading to better generalization.

Reference:
    Rashmi, K. V., and Ran Gilad-Bachrach. "DART: Dropouts meet Multiple
    Additive Regression Trees." AISTATS, 2015.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .._array import BinnedArray, array
from .._backends import is_cuda
from .._core._growth import TreeStructure, GrowthConfig
from .._loss import get_loss_function, LossFunction
from .._core._tree import fit_tree
from .._persistence import PersistenceMixin

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class DART(PersistenceMixin):
    """DART: Gradient Boosting with Dropout.
    
    Implements DART (Dropouts meet Multiple Additive Regression Trees),
    which randomly drops trees during training to prevent overfitting.
    
    Args:
        n_trees: Number of trees to train.
        max_depth: Maximum depth of each tree.
        learning_rate: Base learning rate (shrinkage factor).
        loss: Loss function ('mse', 'logloss', 'huber', or callable).
        dropout_rate: Fraction of trees to drop each round (0 to 1).
        skip_drop: Probability of skipping dropout for a round.
        normalize: If True, normalize dropped tree contributions.
        sample_type: How to sample dropped trees ('uniform' or 'weighted').
        min_child_weight: Minimum sum of hessian in a leaf.
        reg_lambda: L2 regularization on leaf values.
        n_bins: Number of bins for histogram building.
        seed: Random seed for reproducibility.
        
    Example:
        ```python
        import openboost as ob
        
        # DART with 10% dropout
        model = ob.DART(n_trees=100, dropout_rate=0.1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # DART with higher dropout for more regularization
        model = ob.DART(n_trees=200, dropout_rate=0.3, skip_drop=0.5)
        model.fit(X_train, y_train)
        ```
    """
    
    n_trees: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    loss: str | LossFunction = 'mse'
    dropout_rate: float = 0.1
    skip_drop: float = 0.0  # Probability of skipping dropout
    normalize: bool = True
    sample_type: str = 'uniform'  # 'uniform' or 'weighted'
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    n_bins: int = 256
    seed: int | None = None
    
    # Fitted attributes (not init)
    trees_: list[TreeStructure] = field(default_factory=list, init=False, repr=False)
    tree_weights_: list[float] = field(default_factory=list, init=False, repr=False)
    X_binned_: BinnedArray | None = field(default=None, init=False, repr=False)
    _loss_fn: LossFunction | None = field(default=None, init=False, repr=False)
    _rng: np.random.Generator | None = field(default=None, init=False, repr=False)
    
    def fit(self, X: NDArray, y: NDArray) -> "DART":
        """Fit the DART model.
        
        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training targets, shape (n_samples,).
            
        Returns:
            self: The fitted model.
        """
        # Clear any previous fit
        self.trees_ = []
        self.tree_weights_ = []
        
        # Initialize RNG
        self._rng = np.random.default_rng(self.seed)
        
        # Convert to float32
        y = np.asarray(y, dtype=np.float32).ravel()
        n_samples = len(y)
        
        # Get loss function
        self._loss_fn = get_loss_function(self.loss)
        
        # Bin the data
        if isinstance(X, BinnedArray):
            self.X_binned_ = X
        else:
            self.X_binned_ = array(X, n_bins=self.n_bins)
        
        # Initialize predictions
        pred = np.zeros(n_samples, dtype=np.float32)
        
        # Train trees
        for i in range(self.n_trees):
            # Decide whether to apply dropout this round
            apply_dropout = (
                len(self.trees_) > 0 and
                self._rng.random() >= self.skip_drop
            )
            
            if apply_dropout:
                # Select trees to drop
                dropped_indices = self._select_dropped_trees()
                
                # Compute predictions without dropped trees
                pred_without_dropped = self._predict_without_trees(
                    self.X_binned_, dropped_indices
                )
                
                # Compute gradients against predictions without dropped trees
                grad, hess = self._loss_fn(pred_without_dropped, y)
            else:
                # No dropout, use full predictions
                dropped_indices = []
                grad, hess = self._loss_fn(pred, y)
            
            # Ensure float32
            grad = np.asarray(grad, dtype=np.float32)
            hess = np.asarray(hess, dtype=np.float32)
            
            # Build tree using standard fit_tree (the whole point of Phase 8!)
            tree = fit_tree(
                self.X_binned_,
                grad,
                hess,
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
            )
            
            # Determine tree weight
            if apply_dropout and self.normalize and dropped_indices:
                # Normalize: new tree weight depends on number of dropped trees
                k = len(dropped_indices)
                tree_weight = 1.0 / (k + 1)
                
                # Also rescale dropped trees
                for idx in dropped_indices:
                    self.tree_weights_[idx] *= k / (k + 1)
            else:
                tree_weight = 1.0
            
            # Store tree
            self.trees_.append(tree)
            self.tree_weights_.append(tree_weight * self.learning_rate)
            
            # Update predictions (full model)
            pred = self._predict_internal(self.X_binned_)
        
        return self
    
    def _select_dropped_trees(self) -> list[int]:
        """Select which trees to drop for this round."""
        n_trees = len(self.trees_)
        if n_trees == 0:
            return []
        
        n_drop = max(1, int(n_trees * self.dropout_rate))
        
        if self.sample_type == 'uniform':
            # Uniform random selection
            dropped = self._rng.choice(n_trees, size=n_drop, replace=False)
        elif self.sample_type == 'weighted':
            # Weight by tree contribution (not implemented, fall back to uniform)
            dropped = self._rng.choice(n_trees, size=n_drop, replace=False)
        else:
            raise ValueError(f"Unknown sample_type: {self.sample_type}")
        
        return dropped.tolist()
    
    def _predict_without_trees(
        self,
        X: BinnedArray,
        excluded_indices: list[int],
    ) -> NDArray:
        """Predict using all trees except those in excluded_indices."""
        n_samples = X.n_samples
        pred = np.zeros(n_samples, dtype=np.float32)
        
        excluded_set = set(excluded_indices)
        
        for i, (tree, weight) in enumerate(zip(self.trees_, self.tree_weights_)):
            if i in excluded_set:
                continue
            tree_pred = tree(X)
            if hasattr(tree_pred, 'copy_to_host'):
                tree_pred = tree_pred.copy_to_host()
            pred += weight * tree_pred
        
        return pred
    
    def _predict_internal(self, X: BinnedArray) -> NDArray:
        """Internal prediction using all trees (for training)."""
        n_samples = X.n_samples
        pred = np.zeros(n_samples, dtype=np.float32)
        
        for tree, weight in zip(self.trees_, self.tree_weights_):
            tree_pred = tree(X)
            if hasattr(tree_pred, 'copy_to_host'):
                tree_pred = tree_pred.copy_to_host()
            pred += weight * tree_pred
        
        return pred
    
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
        
        # Bin the data if needed, using training bin edges for consistency
        if isinstance(X, BinnedArray):
            X_binned = X
        elif self.X_binned_ is not None:
            X_binned = self.X_binned_.transform(X)
        else:
            X_binned = array(X, n_bins=self.n_bins)
        
        return self._predict_internal(X_binned)
    
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
