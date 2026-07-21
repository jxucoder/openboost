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
from .._callbacks import (
    Callback,
    CallbackManager,
    EarlyStopping,
    TrainingState,
    warn_if_early_stopping_without_eval_set,
)
from .._core._growth import TreeStructure
from .._core._tree import fit_tree
from .._loss import LossFunction, get_loss_function
from .._persistence import PersistenceMixin
from .._validation import validate_eval_set

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _binned_matrix(x_binned) -> NDArray:
    """Extract a host (n_features, n_samples) uint8 matrix from binned data."""
    data = x_binned.data if isinstance(x_binned, BinnedArray) else x_binned
    if hasattr(data, 'copy_to_host'):
        data = data.copy_to_host()
    return np.asarray(data)


def _route_to_leaves(tree: TreeStructure, binned: NDArray) -> NDArray:
    """Route all samples to their leaf node index, vectorized.

    Descends every sample in lockstep: each iteration advances all samples
    still at internal nodes one level down, so the loop runs over tree depth,
    never over samples.

    Args:
        tree: TreeStructure with left/right children routing arrays
        binned: Binned features, shape (n_features, n_samples)

    Returns:
        node_ids: (n_samples,) int32 tree node index of each sample's leaf
    """
    left = np.asarray(tree.left_children)
    right = np.asarray(tree.right_children)
    features = np.asarray(tree.features)
    thresholds = np.asarray(tree.thresholds)

    n_samples = binned.shape[1]
    node = np.zeros(n_samples, dtype=np.int32)
    active = np.nonzero(left[node] != -1)[0]
    while active.size:
        idx = node[active]
        go_left = binned[features[idx], active] <= thresholds[idx]
        node[active] = np.where(go_left, left[idx], right[idx])
        active = active[left[node[active]] != -1]
    return node


@dataclass
class LinearLeafTree:
    """A tree with linear models in leaves.

    Instead of constant leaf values, each leaf has a linear model:
        prediction = w0 + w1*x[f1] + w2*x[f2] + ...

    Attributes:
        tree_structure: Base tree for routing samples to leaves
        leaf_weights: (n_leaves, max_features + 1) linear model weights
        leaf_features: List of feature indices used in each leaf
        leaf_ids: Mapping from integer tree node index to our leaf index
        n_features: Total number of features in the dataset
        training_binned: Reference to training BinnedArray for transform
    """
    tree_structure: TreeStructure
    leaf_weights: NDArray  # (n_leaves, max_features_linear + 1)
    leaf_features: list[list[int]]  # Features used per leaf
    leaf_ids: dict[int, int]  # Map tree node index -> our leaf index
    n_features: int = 0
    training_binned: BinnedArray | None = None  # For transform

    def __call__(self, X: NDArray) -> NDArray:
        """Predict using linear leaf tree."""
        return self.predict(X)

    def predict(self, X: NDArray, binned: NDArray | None = None) -> NDArray:
        """Generate predictions using linear models in leaves.

        Args:
            X: Features, shape (n_samples, n_features)
            binned: Optional precomputed binned matrix (n_features, n_samples)
                from this tree's training bin edges; avoids re-binning when
                predicting through many trees that share bin edges.

        Returns:
            predictions: Shape (n_samples,)
        """
        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]

        # Get leaf node indices from tree structure
        leaf_node_ids = self._get_leaf_node_indices(X, binned=binned)

        # Map tree node index -> leaf index. Unknown node ids fall back to
        # leaf 0 (same behavior as the original per-sample lookup).
        node_to_leaf = np.zeros(self.tree_structure.n_nodes, dtype=np.int32)
        for node_id, leaf_idx in self.leaf_ids.items():
            node_to_leaf[node_id] = leaf_idx
        sample_leaf = node_to_leaf[leaf_node_ids]

        predictions = np.zeros(n_samples, dtype=np.float32)

        for leaf_idx, feat_indices in enumerate(self.leaf_features):
            rows = np.nonzero(sample_leaf == leaf_idx)[0]
            if rows.size == 0:
                continue

            weights = self.leaf_weights[leaf_idx]

            # Linear prediction: w0 + sum(w_i * x_i), batched over the leaf's
            # samples. Accumulate feature-by-feature in float32 to stay
            # bit-identical to the original scalar accumulation order.
            leaf_pred = np.full(rows.size, weights[0], dtype=np.float32)
            n_terms = min(len(feat_indices), len(weights) - 1)
            for j in range(n_terms):
                leaf_pred += weights[j + 1] * X[rows, feat_indices[j]]

            predictions[rows] = leaf_pred

        return predictions

    def _get_leaf_node_indices(self, X: NDArray, binned: NDArray | None = None) -> NDArray:
        """Get the tree node index of the leaf each sample falls into."""
        if binned is None:
            # Bin the data for tree prediction, using training bin edges
            if self.training_binned is not None:
                X_binned = self.training_binned.transform(X)
            else:
                X_binned = array(X)
            binned = _binned_matrix(X_binned)

        return _route_to_leaves(self.tree_structure, binned)


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

    Attributes (after fit with observability args):
        evals_result_: Per-round eval-set MSE history recorded during
            training, e.g. ``{'eval_0': {'mse': [...]}}``. Empty dict when
            no ``eval_set`` was passed to ``fit()``.
        best_iteration_: Best round index (set when early stopping is used).
        best_score_: Best monitored metric value (set with best_iteration_).


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
    n_bins: int = 254
    
    # Fitted attributes (not init)
    trees_: list[LinearLeafTree] = field(default_factory=list, init=False, repr=False)
    X_binned_: BinnedArray | None = field(default=None, init=False, repr=False)
    _loss_fn: LossFunction | None = field(default=None, init=False, repr=False)
    n_features_in_: int = field(default=0, init=False, repr=False)
    evals_result_: dict[str, dict[str, list[float]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        callbacks: list[Callback] | None = None,
        eval_set: list[tuple[NDArray, NDArray]] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> LinearLeafGBDT:
        """Fit the linear leaf GBDT model.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training targets, shape (n_samples,)
            callbacks: List of Callback instances (e.g. EarlyStopping,
                Logger) invoked each boosting round with a TrainingState
                carrying the round index, train MSE, and val MSE.
            eval_set: Validation set(s) as a list of ``(X_val, y_val)``
                tuples (a single bare tuple is also accepted). Every eval
                set is scored with MSE each round and the per-round history
                is stored in ``evals_result_``. The LAST eval set's MSE is
                reported to callbacks as ``val_loss``.
            early_stopping_rounds: Stop training when the monitored eval-set
                MSE has not improved for this many consecutive rounds
                (sugar for an ``EarlyStopping`` callback with
                ``restore_best=True``). The model is restored to (truncated
                at) the best iteration and ``best_iteration_`` /
                ``best_score_`` are set. Requires ``eval_set``.

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
        
        # Initialize predictions with base score
        if self.loss in ('logloss', 'binary_crossentropy'):
            p = np.clip(np.mean(y), 1e-7, 1 - 1e-7)
            self.base_score_ = np.float32(np.log(p / (1 - p)))
        else:
            self.base_score_ = np.float32(np.mean(y))
        pred = np.full(n_samples, self.base_score_, dtype=np.float32)

        # Setup callbacks (early_stopping_rounds is sugar for EarlyStopping).
        # EarlyStopping's restore_best snapshots/restores self.trees_; each
        # LinearLeafTree bundles its routing tree AND its per-leaf linear
        # models, so restoring trees_ truncates both consistently.
        cb_list = list(callbacks) if callbacks else []
        if early_stopping_rounds is not None:
            cb_list.append(
                EarlyStopping(patience=early_stopping_rounds, restore_best=True)
            )
        cb_manager = CallbackManager(cb_list)
        state = TrainingState(model=self, n_rounds=self.n_trees)

        eval_set = validate_eval_set(eval_set, n_features)
        warn_if_early_stopping_without_eval_set(cb_list, eval_set)

        # Per-eval-set state: raw features (linear leaves need un-binned
        # values), targets, binned matrix (binned once with the training bin
        # edges), and incrementally maintained predictions.
        eval_data = []
        if eval_set:
            for X_e, y_e in eval_set:
                X_e = np.asarray(X_e, dtype=np.float32)
                y_e = np.asarray(y_e, dtype=np.float32).ravel()
                binned_e = _binned_matrix(self.X_binned_.transform(X_e))
                pred_e = np.full(X_e.shape[0], self.base_score_, dtype=np.float32)
                eval_data.append((X_e, y_e, binned_e, pred_e))

        self.evals_result_ = {
            f'eval_{i}': {'mse': []} for i in range(len(eval_data))
        }

        cb_manager.on_train_begin(state)

        for round_idx in range(self.n_trees):
            state.round_idx = round_idx
            cb_manager.on_round_begin(state)

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

            # Observability: only computed when requested, so the default
            # path (no callbacks, no eval_set) is byte-identical to before.
            if cb_manager.callbacks or eval_data:
                # Train loss: MSE on current predictions (already tracked)
                state.train_loss = float(np.mean((pred - y) ** 2))

                # Score ALL eval sets, record history; callbacks monitor the
                # LAST eval set's MSE (matches DistributionalGBDT semantics)
                val_mse = None
                for i, (X_e, y_e, binned_e, pred_e) in enumerate(eval_data):
                    pred_e += self.learning_rate * linear_tree.predict(
                        X_e, binned=binned_e
                    )
                    val_mse = float(np.mean((pred_e - y_e) ** 2))
                    self.evals_result_[f'eval_{i}']['mse'].append(val_mse)
                if val_mse is not None:
                    state.val_loss = val_mse

                # Check if callbacks want to stop
                if not cb_manager.on_round_end(state):
                    break

        cb_manager.on_train_end(state)

        # MED-21: Release raw data reference to free memory
        self._X_raw = None

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

        # Get leaf node indices (integer IDs) for each sample
        binned_data = _binned_matrix(self.X_binned_)
        leaf_node_ids = _route_to_leaves(base_tree, binned_data)

        # Find unique leaf node IDs
        unique_node_ids = np.unique(leaf_node_ids)
        n_leaves = len(unique_node_ids)

        # Map integer node index -> our leaf index
        leaf_ids = {int(nid): i for i, nid in enumerate(unique_node_ids)}

        # Storage for linear models
        leaf_weights = np.zeros((n_leaves, n_linear_features + 1), dtype=np.float32)
        leaf_features = []

        for leaf_idx, node_id in enumerate(unique_node_ids):
            # Get samples in this leaf
            mask = leaf_node_ids == node_id
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
        
        # Regularization (don't regularize bias)
        reg_matrix = reg_lambda * np.eye(n_features + 1)
        reg_matrix[0, 0] = 0

        try:
            # Solve normal equations — O(n*p^2) instead of O(n^2) via np.diag
            XtWX = X_aug.T @ (weights[:, None] * X_aug) + reg_matrix
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

        # All trees share the training bin edges, so bin X once instead of
        # once per tree.
        shared_binned = None
        first_binned = self.trees_[0].training_binned
        if first_binned is not None and all(
            t.training_binned is first_binned for t in self.trees_
        ):
            shared_binned = _binned_matrix(first_binned.transform(X))

        base = getattr(self, 'base_score_', np.float32(0.0))
        pred = np.full(n_samples, base, dtype=np.float32)
        for tree in self.trees_:
            pred = pred + self.learning_rate * tree.predict(X, binned=shared_binned)

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
