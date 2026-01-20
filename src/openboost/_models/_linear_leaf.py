"""Linear Leaf Gradient Boosting.

Phase 15.4: Trees with linear models in leaves for better extrapolation.

Each leaf fits: y = w0 + w1*x1 + w2*x2 + ... 
instead of a constant value.

Benefits:
- Better extrapolation beyond training data range
- Smoother predictions at decision boundaries
- Can use shallower trees (linear models add flexibility)
- Better performance on data with linear trends

Reference:
    Similar to LightGBM's linear tree feature.

Example:
    ```python
    import openboost as ob
    
    model = ob.LinearLeafGBDT(n_trees=100, max_depth=4)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)  # Better extrapolation!
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .._array import BinnedArray, array
from .._backends import is_cuda
from .._loss import get_loss_function, LossFunction
from .._core._tree import fit_tree
from .._core._growth import TreeStructure
from .._persistence import PersistenceMixin

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class LinearLeafTree:
    """A tree with linear models in leaves.
    
    Instead of constant leaf values, each leaf has a linear model:
        prediction = w0 + w1*x[f1] + w2*x[f2] + ...
    
    Attributes:
        tree_structure: Base tree for routing samples to leaves
        leaf_weights: (n_leaves, max_features + 1) linear model weights
        leaf_features: List of feature indices used in each leaf
        leaf_ids: Mapping from tree leaf indices to our leaf indices
        n_features: Total number of features in the dataset
        training_binned: Reference to training BinnedArray for transform
    """
    tree_structure: TreeStructure
    leaf_weights: NDArray  # (n_leaves, max_features_linear + 1)
    leaf_features: list[list[int]]  # Features used per leaf
    leaf_ids: dict[int, int]  # Map tree leaf -> our leaf index
    n_features: int
    training_binned: BinnedArray | None = None  # For transform
    
    def __call__(self, X: NDArray) -> NDArray:
        """Predict using linear leaf tree."""
        return self.predict(X)
    
    def predict(self, X: NDArray) -> NDArray:
        """Generate predictions using linear models in leaves.
        
        Args:
            X: Features, shape (n_samples, n_features)
            
        Returns:
            predictions: Shape (n_samples,)
        """
        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        
        # Get leaf assignments from tree structure
        # We need to route samples through the tree to find their leaf
        leaf_preds = self._get_leaf_predictions(X)
        
        predictions = np.zeros(n_samples, dtype=np.float32)
        
        for sample_idx in range(n_samples):
            leaf_pred = leaf_preds[sample_idx]
            
            # Find which of our leaves this corresponds to
            if leaf_pred in self.leaf_ids:
                leaf_idx = self.leaf_ids[leaf_pred]
            else:
                # Fallback: use the constant term from first leaf
                leaf_idx = 0
            
            # Get weights and features for this leaf
            weights = self.leaf_weights[leaf_idx]
            feat_indices = self.leaf_features[leaf_idx]
            
            # Linear prediction: w0 + sum(w_i * x_i)
            pred = weights[0]  # Bias term
            for j, feat_idx in enumerate(feat_indices):
                if j + 1 < len(weights):
                    pred += weights[j + 1] * X[sample_idx, feat_idx]
            
            predictions[sample_idx] = pred
        
        return predictions
    
    def _get_leaf_predictions(self, X: NDArray) -> NDArray:
        """Get the constant leaf value for each sample (used as leaf ID)."""
        # Bin the data for tree prediction, using training bin edges
        if self.training_binned is not None:
            X_binned = self.training_binned.transform(X)
        else:
            X_binned = array(X)
        preds = self.tree_structure(X_binned)
        if hasattr(preds, 'copy_to_host'):
            preds = preds.copy_to_host()
        return preds


@dataclass
class LinearLeafGBDT(PersistenceMixin):
    """Gradient Boosting with Linear Leaf Trees.
    
    Each tree has linear models in its leaves instead of constant values.
    This enables:
    - Better extrapolation beyond training data range
    - Smoother decision boundaries
    - Can use shallower trees (linear models add complexity)
    
    Recommended settings:
    - Use max_depth=3-4 (shallower than standard GBDT)
    - Use larger min_samples_leaf (need samples to fit linear model)
    
    Args:
        n_trees: Number of boosting rounds
        max_depth: Maximum tree depth (typically 3-4, shallower than standard)
        learning_rate: Shrinkage factor
        loss: Loss function ('mse', 'mae', 'huber', or callable)
        min_samples_leaf: Minimum samples to fit linear model in leaf
        reg_lambda_tree: L2 regularization for tree splits
        reg_lambda_linear: L2 regularization for linear models (ridge)
        max_features_linear: Max features per leaf's linear model
            - None: Use all features
            - 'sqrt': Use sqrt(n_features) features
            - 'log2': Use log2(n_features) features
            - int: Use exactly this many features
        n_bins: Number of bins for histogram building
        
    Example:
        ```python
        model = LinearLeafGBDT(n_trees=100, max_depth=4)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        # Compare extrapolation with standard GBDT
        from openboost import GradientBoosting
        standard = GradientBoosting(n_trees=100, max_depth=6)
        standard.fit(X_train, y_train)
        # LinearLeafGBDT typically extrapolates better on linear trends
        ```
    """
    
    n_trees: int = 100
    max_depth: int = 4  # Typically shallower than standard GBDT
    learning_rate: float = 0.1
    loss: str | LossFunction = 'mse'
    min_samples_leaf: int = 20  # Need enough samples for linear fit
    reg_lambda_tree: float = 1.0
    reg_lambda_linear: float = 0.1  # Ridge regularization for linear models
    max_features_linear: int | str | None = 'sqrt'
    n_bins: int = 256
    
    # Fitted attributes (not init)
    trees_: list[LinearLeafTree] = field(default_factory=list, init=False, repr=False)
    X_binned_: BinnedArray | None = field(default=None, init=False, repr=False)
    _loss_fn: LossFunction | None = field(default=None, init=False, repr=False)
    n_features_in_: int = field(default=0, init=False, repr=False)
    
    def fit(self, X: NDArray, y: NDArray) -> "LinearLeafGBDT":
        """Fit the linear leaf GBDT model.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training targets, shape (n_samples,)
            
        Returns:
            self: Fitted model
        """
        self.trees_ = []
        
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        n_samples, n_features = X.shape
        
        self.n_features_in_ = n_features
        
        # Store raw X for linear fitting (need un-binned values)
        self._X_raw = X
        
        # Get loss function
        self._loss_fn = get_loss_function(self.loss)
        
        # Bin data for tree building
        self.X_binned_ = array(X, n_bins=self.n_bins)
        
        # Determine max features for linear models
        if self.max_features_linear is None:
            n_linear_features = n_features
        elif self.max_features_linear == 'sqrt':
            n_linear_features = max(1, int(np.sqrt(n_features)))
        elif self.max_features_linear == 'log2':
            n_linear_features = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features_linear, int):
            n_linear_features = min(self.max_features_linear, n_features)
        else:
            n_linear_features = n_features
        
        self._n_linear_features = n_linear_features
        
        # Initialize predictions
        pred = np.zeros(n_samples, dtype=np.float32)
        
        for round_idx in range(self.n_trees):
            # Compute gradients
            grad, hess = self._loss_fn(pred, y)
            grad = np.asarray(grad, dtype=np.float32)
            hess = np.asarray(hess, dtype=np.float32)
            
            # Build tree structure (just for routing)
            base_tree = fit_tree(
                self.X_binned_,
                grad,
                hess,
                max_depth=self.max_depth,
                min_child_weight=float(self.min_samples_leaf),
                reg_lambda=self.reg_lambda_tree,
            )
            
            # Fit linear models in each leaf
            linear_tree = self._fit_linear_leaves(
                base_tree, X, y, grad, hess, pred, n_linear_features
            )
            
            self.trees_.append(linear_tree)
            
            # Update predictions
            tree_pred = linear_tree.predict(X)
            pred = pred + self.learning_rate * tree_pred
        
        return self
    
    def _fit_linear_leaves(
        self,
        base_tree: TreeStructure,
        X: NDArray,
        y: NDArray,
        grad: NDArray,
        hess: NDArray,
        current_pred: NDArray,
        n_linear_features: int,
    ) -> LinearLeafTree:
        """Fit linear models in each leaf of the tree.
        
        Uses weighted least squares with hessian as weights.
        Target is the negative gradient divided by hessian (Newton step).
        """
        n_samples, n_features = X.shape
        
        # Get leaf predictions (these serve as leaf identifiers)
        leaf_preds = base_tree(self.X_binned_)
        if hasattr(leaf_preds, 'copy_to_host'):
            leaf_preds = leaf_preds.copy_to_host()
        
        # Find unique leaves and their indices
        unique_leaves = np.unique(leaf_preds)
        n_leaves = len(unique_leaves)
        
        # Map leaf prediction value -> our leaf index
        leaf_ids = {float(v): i for i, v in enumerate(unique_leaves)}
        
        # Storage for linear models
        leaf_weights = np.zeros((n_leaves, n_linear_features + 1), dtype=np.float32)
        leaf_features = []
        
        for leaf_idx, leaf_val in enumerate(unique_leaves):
            # Get samples in this leaf
            mask = np.isclose(leaf_preds, leaf_val)
            n_leaf = np.sum(mask)
            
            if n_leaf < self.min_samples_leaf:
                # Not enough samples: use constant (weighted mean of target)
                w = hess[mask]
                target = -grad[mask] / (hess[mask] + 1e-6)
                if np.sum(w) > 0:
                    leaf_weights[leaf_idx, 0] = np.average(target, weights=w)
                leaf_features.append([])
                continue
            
            # Get data for this leaf
            X_leaf = X[mask]
            grad_leaf = grad[mask]
            hess_leaf = hess[mask]
            
            # Target for regression: Newton step = -grad/hess
            target = -grad_leaf / (hess_leaf + 1e-6)
            
            # Select features based on correlation with target
            if n_linear_features < n_features:
                selected_features = self._select_features(
                    X_leaf, target, n_linear_features
                )
            else:
                selected_features = list(range(n_features))
            
            leaf_features.append(selected_features)
            
            # Fit ridge regression with hessian weights
            weights = self._fit_weighted_ridge(
                X_leaf[:, selected_features],
                target,
                hess_leaf,
                self.reg_lambda_linear,
            )
            
            # Store weights (bias first, then feature weights)
            leaf_weights[leaf_idx, :len(weights)] = weights
        
        return LinearLeafTree(
            tree_structure=base_tree,
            leaf_weights=leaf_weights,
            leaf_features=leaf_features,
            leaf_ids=leaf_ids,
            n_features=n_features,
            training_binned=self.X_binned_,  # For transform on new data
        )
    
    def _select_features(
        self,
        X: NDArray,
        target: NDArray,
        n_select: int,
    ) -> list[int]:
        """Select features based on correlation with target.
        
        Uses absolute correlation to select most relevant features.
        """
        n_features = X.shape[1]
        
        correlations = np.zeros(n_features)
        for j in range(n_features):
            if np.std(X[:, j]) > 1e-8:
                corr = np.corrcoef(X[:, j], target)[0, 1]
                if not np.isnan(corr):
                    correlations[j] = abs(corr)
        
        # Select top features by correlation
        selected = np.argsort(correlations)[-n_select:]
        return sorted(selected.tolist())
    
    def _fit_weighted_ridge(
        self,
        X: NDArray,
        y: NDArray,
        weights: NDArray,
        reg_lambda: float,
    ) -> NDArray:
        """Fit weighted ridge regression.
        
        Minimizes: sum(w_i * (y_i - X_i @ beta)^2) + lambda * ||beta[1:]||^2
        
        Solution: (X'WX + lambda*I)^{-1} X'Wy
        (Don't regularize the bias term)
        """
        n_samples, n_features = X.shape
        
        # Add bias column
        X_aug = np.column_stack([np.ones(n_samples), X])
        
        # Diagonal weight matrix
        W = np.diag(weights)
        
        # Regularization (don't regularize bias)
        reg_matrix = reg_lambda * np.eye(n_features + 1)
        reg_matrix[0, 0] = 0
        
        try:
            # Solve normal equations
            XtWX = X_aug.T @ W @ X_aug + reg_matrix
            XtWy = X_aug.T @ (weights * y)
            beta = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            # Fallback: just use weighted mean
            beta = np.zeros(n_features + 1)
            beta[0] = np.average(y, weights=weights) if np.sum(weights) > 0 else 0
        
        return beta.astype(np.float32)
    
    def predict(self, X: NDArray) -> NDArray:
        """Generate predictions.
        
        Args:
            X: Features, shape (n_samples, n_features)
            
        Returns:
            predictions: Shape (n_samples,)
        """
        if not self.trees_:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        
        pred = np.zeros(n_samples, dtype=np.float32)
        for tree in self.trees_:
            pred = pred + self.learning_rate * tree.predict(X)
        
        return pred
    
    def score(self, X: NDArray, y: NDArray) -> float:
        """R² score (coefficient of determination).
        
        Args:
            X: Features
            y: True target values
            
        Returns:
            R² score (1.0 is perfect, 0.0 is baseline)
        """
        y = np.asarray(y, dtype=np.float32)
        y_pred = self.predict(X)
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1.0 - ss_res / ss_tot
    
    def _post_load(self) -> None:
        """Post-load hook to restore tree references.
        
        After model is loaded, update all LinearLeafTree instances
        with the restored X_binned_ reference for correct transform behavior.
        """
        if hasattr(self, 'X_binned_') and self.X_binned_ is not None:
            for tree in self.trees_:
                tree.training_binned = self.X_binned_
