"""Gradient Boosting ensemble model for OpenBoost.

Provides a scikit-learn-like API for training gradient boosting models
with both built-in and custom loss functions.

This module implements batched training that keeps computation on the GPU
without returning to Python between trees, achieving performance competitive
with XGBoost.

Phase 13: Added callback support for early stopping, logging, etc.
Phase 17: Added GOSS sampling and mini-batch training for large-scale datasets.
Phase 18: Added multi-GPU support via Ray for data-parallel training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np

from .._array import BinnedArray, array
from .._backends import is_cuda
from .._loss import get_loss_function, LossFunction
from .._core._tree import fit_tree
from .._core._growth import TreeStructure
from .._callbacks import Callback, CallbackManager, TrainingState
from .._sampling import (
    SamplingStrategy,
    goss_sample,
    random_sample,
    apply_sampling,
    MiniBatchIterator,
    accumulate_histograms_minibatch,
)
from .._persistence import PersistenceMixin
from .._validation import (
    validate_X,
    validate_y,
    validate_sample_weight,
    validate_eval_set,
    validate_hyperparameters,
    validate_predict_input,
)

try:
    from .._distributed._ray import RayDistributedContext
    from .._distributed._tree import fit_tree_distributed
except ImportError:
    RayDistributedContext = None
    fit_tree_distributed = None

try:
    from .._distributed._multigpu import MultiGPUContext, fit_tree_multigpu
except ImportError:
    MultiGPUContext = None
    fit_tree_multigpu = None

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class GradientBoosting(PersistenceMixin):
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
        subsample_strategy: Sampling strategy for large-scale training (Phase 17):
            - 'none': No sampling (default)
            - 'random': Random subsampling
            - 'goss': Gradient-based One-Side Sampling (LightGBM-style)
        goss_top_rate: Fraction of top-gradient samples to keep (for GOSS).
        goss_other_rate: Fraction of remaining samples to sample (for GOSS).
        batch_size: Mini-batch size for large datasets. If None, process all at once.
        
    Examples:
        Basic regression:
        
        ```python
        import openboost as ob
        
        model = ob.GradientBoosting(n_trees=100, loss='mse')
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        ```
        
        Quantile regression (90th percentile):
        
        ```python
        model = ob.GradientBoosting(loss='quantile', quantile_alpha=0.9)
        model.fit(X_train, y_train)
        ```
        
        GOSS for faster training:
        
        ```python
        model = ob.GradientBoosting(
            n_trees=100,
            subsample_strategy='goss',
            goss_top_rate=0.2,
            goss_other_rate=0.1,
        )
        ```
        
        Multi-GPU training:
        
        ```python
        model = ob.GradientBoosting(n_trees=100, n_gpus=4)
        model.fit(X, y)  # Data parallel across 4 GPUs
        ```
    """
    
    n_trees: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    loss: str | LossFunction | Callable[..., tuple] = 'mse'
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    gamma: float = 0.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    n_bins: int = 256
    quantile_alpha: float = 0.5
    tweedie_rho: float = 1.5
    distributed: bool = False
    n_workers: int | None = None
    subsample_strategy: Literal['none', 'random', 'goss'] = 'none'
    goss_top_rate: float = 0.2
    goss_other_rate: float = 0.1
    batch_size: int | None = None
    n_gpus: int | None = None
    devices: list[int] | None = None
    
    # Fitted attributes (not init)
    trees_: list[TreeStructure] = field(default_factory=list, init=False, repr=False)
    X_binned_: BinnedArray | None = field(default=None, init=False, repr=False)
    _loss_fn: LossFunction | None = field(default=None, init=False, repr=False)
    n_features_in_: int = field(default=0, init=False, repr=False)  # Phase 20.3: store for validation
    
    def fit(
        self,
        X: NDArray,
        y: NDArray,
        callbacks: list[Callback] | None = None,
        eval_set: list[tuple[NDArray, NDArray]] | None = None,
        sample_weight: NDArray | None = None,
    ) -> GradientBoosting:
        """Fit the gradient boosting model.
        
        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training targets, shape (n_samples,).
            callbacks: List of Callback instances for training hooks.
                       Use EarlyStopping for early stopping, Logger for progress.
            eval_set: List of (X, y) tuples for validation (used with callbacks).
            sample_weight: Sample weights, shape (n_samples,).
            
        Returns:
            self: The fitted model.
            
        Example:
            ```python
            from openboost import GradientBoosting, EarlyStopping, Logger
            
            model = GradientBoosting(n_trees=1000)
            model.fit(
                X, y,
                callbacks=[EarlyStopping(patience=50), Logger(period=10)],
                eval_set=[(X_val, y_val)]
            )
            ```
        """
        # Clear any previous fit
        self.trees_ = []
        
        # Validate inputs (Phase 20.3)
        X = validate_X(X, allow_nan=True, context="fit")
        y = validate_y(y, n_samples=X.shape[0] if hasattr(X, 'shape') else None, context="fit")
        sample_weight = validate_sample_weight(sample_weight, len(y))
        
        n_samples = len(y)
        
        # Validate hyperparameters
        validate_hyperparameters(
            n_trees=self.n_trees,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            subsample=self.subsample,
            n_samples=n_samples,
        )
        
        # Get loss function (pass parameters for parameterized losses)
        self._loss_fn = get_loss_function(
            self.loss, 
            quantile_alpha=self.quantile_alpha,
            tweedie_rho=self.tweedie_rho,
        )
        
        # Validate eval_set
        if eval_set is not None:
            n_features = X.shape[1] if hasattr(X, 'shape') else X.n_features
            eval_set = validate_eval_set(eval_set, n_features)
        
        # Bin the data (this is the expensive step, but only done once)
        if isinstance(X, BinnedArray):
            self.X_binned_ = X
        else:
            self.X_binned_ = array(X, n_bins=self.n_bins)
        
        # Store for feature importance
        self.n_features_in_ = self.X_binned_.n_features
        
        # Choose training path based on backend
        # Phase 18: Check for multi-GPU first
        use_multigpu = (self.n_gpus is not None and self.n_gpus > 1) or (self.devices is not None and len(self.devices) > 1)
        
        if use_multigpu:
            self._fit_multigpu(X, y, n_samples, callbacks, eval_set)
        elif self.distributed:
            self._fit_distributed(y, n_samples)
        elif is_cuda():
            self._fit_gpu(y, n_samples, callbacks, eval_set, sample_weight)
        else:
            self._fit_cpu(y, n_samples, callbacks, eval_set, sample_weight)
        
        return self
    
    def _fit_distributed(self, y: NDArray, n_samples: int):
        """Distributed training using Ray."""
        if RayDistributedContext is None:
            raise ImportError("Distributed training requires 'ray'. Install with 'pip install ray'.")
            
        ctx = RayDistributedContext(self.n_workers)
        
        X_data = self.X_binned_.data
        if hasattr(X_data, 'copy_to_host'):
            X_data = X_data.copy_to_host()
        
        ctx.setup(X_data, y, self.n_bins)
        
        import ray
        
        for i in range(self.n_trees):
            # Compute gradients on each worker
            grad_hess_refs = [
                w.compute_gradients.options(num_returns=2).remote(self._loss_fn) 
                for w in ctx.workers
            ]
            
            grad_refs = [pair[0] for pair in grad_hess_refs]
            hess_refs = [pair[1] for pair in grad_hess_refs]
            
            # Distributed tree fitting
            tree = fit_tree_distributed(
                ctx, 
                ctx.workers, 
                grad_refs, 
                hess_refs,
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                min_gain=self.gamma,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
            )
            
            # Update predictions on each worker
            for w in ctx.workers:
                w.update_predictions.remote(tree, self.learning_rate)
            
            self.trees_.append(tree)
    
    def _fit_multigpu(
        self,
        X: NDArray,
        y: NDArray,
        n_samples: int,
        callbacks: list[Callback] | None = None,
        eval_set: list[tuple[NDArray, NDArray]] | None = None,
    ):
        """Multi-GPU training using Ray actors (Phase 18).
        
        Each GPU holds a shard of the data and computes local histograms,
        which are aggregated on the driver to build global trees.
        
        This approach provides near-linear scaling for large datasets.
        """
        if MultiGPUContext is None:
            raise ImportError(
                "Multi-GPU training requires Ray. "
                "Install with: pip install 'openboost[distributed]'"
            )
        
        # Setup callbacks
        cb_manager = CallbackManager(callbacks)
        state = TrainingState(model=self, n_rounds=self.n_trees)
        cb_manager.on_train_begin(state)
        
        # Create multi-GPU context and setup workers
        ctx = MultiGPUContext(n_gpus=self.n_gpus, devices=self.devices)
        
        # Get raw data for sharding
        X_data = X if not isinstance(X, BinnedArray) else None
        if X_data is None:
            # Need to get unbinned data for multi-GPU
            # For now, require raw data input
            raise ValueError(
                "Multi-GPU training requires raw (unbinned) data input. "
                "Pass X as a numpy array, not BinnedArray."
            )
        
        ctx.setup(X_data, y, n_bins=self.n_bins)
        
        try:
            import ray
            
            # Training loop
            for i in range(self.n_trees):
                state.round_idx = i
                cb_manager.on_round_begin(state)
                
                # 1. Compute gradients on each GPU (parallel)
                grads_hess = ctx.compute_all_gradients(self._loss_fn)
                
                # 2. Build local histograms on each GPU (parallel)
                local_histograms = ctx.build_all_histograms(grads_hess)
                
                # 3. Aggregate histograms on driver
                global_hist_grad, global_hist_hess = ctx.aggregate_histograms(local_histograms)
                
                # 4. Build tree from global histogram
                # Concatenate gradients for full tree fitting
                all_grad = np.concatenate([g for g, h in grads_hess])
                all_hess = np.concatenate([h for g, h in grads_hess])
                
                # For proper tree building, we need the full binned data
                # Use a simplified approach: fit tree on driver with full histogram info
                from .._core._growth import TreeStructure, GrowthConfig
                from .._core._split import find_best_split, compute_leaf_value
                
                tree = self._build_tree_from_histogram(
                    global_hist_grad,
                    global_hist_hess,
                    all_grad,
                    all_hess,
                    ctx.n_features,
                )
                
                self.trees_.append(tree)
                
                # 5. Update predictions on each GPU (parallel)
                ctx.update_all_predictions(tree, self.learning_rate)
                
                # Compute losses for callbacks (requires collecting predictions)
                if callbacks or eval_set:
                    all_pred = ctx.get_all_predictions()
                    state.train_loss = float(np.mean((all_pred - y) ** 2))
                    
                    if eval_set:
                        X_val, y_val = eval_set[0]
                        val_pred = self.predict(X_val)
                        state.val_loss = float(np.mean((val_pred - y_val) ** 2))
                
                # Check if callbacks want to stop
                if not cb_manager.on_round_end(state):
                    break
            
            cb_manager.on_train_end(state)
            
        finally:
            # Cleanup
            ctx.shutdown()
    
    def _build_tree_from_histogram(
        self,
        hist_grad: NDArray,
        hist_hess: NDArray,
        all_grad: NDArray,
        all_hess: NDArray,
        n_features: int,
    ) -> TreeStructure:
        """Build a tree from aggregated histogram for multi-GPU training.
        
        Uses recursive histogram-based tree building similar to LightGBM.
        """
        from .._core._growth import TreeStructure
        from .._core._split import find_best_split, compute_leaf_value
        
        max_nodes = 2**(self.max_depth + 1) - 1
        features = np.full(max_nodes, -1, dtype=np.int32)
        thresholds = np.zeros(max_nodes, dtype=np.int32)
        values = np.zeros(max_nodes, dtype=np.float32)
        left_children = np.full(max_nodes, -1, dtype=np.int32)
        right_children = np.full(max_nodes, -1, dtype=np.int32)
        
        # Build tree level by level using histogram
        # Start with root node (all samples)
        node_hist_grad = {0: hist_grad.copy()}
        node_hist_hess = {0: hist_hess.copy()}
        node_sum_grad = {0: float(np.sum(all_grad))}
        node_sum_hess = {0: float(np.sum(all_hess))}
        
        active_nodes = [0]
        next_node_id = 1
        actual_depth = 0
        
        for depth in range(self.max_depth):
            if not active_nodes:
                break
            
            actual_depth = depth + 1
            new_active_nodes = []
            
            for node_id in active_nodes:
                h_grad = node_hist_grad.get(node_id)
                h_hess = node_hist_hess.get(node_id)
                s_grad = node_sum_grad.get(node_id, 0.0)
                s_hess = node_sum_hess.get(node_id, 0.0)
                
                if h_grad is None or s_hess < self.min_child_weight:
                    # Make leaf
                    values[node_id] = compute_leaf_value(s_grad, s_hess, self.reg_lambda, self.reg_alpha)
                    continue
                
                # Find best split
                split = find_best_split(
                    h_grad, h_hess,
                    s_grad, s_hess,
                    reg_lambda=self.reg_lambda,
                    min_child_weight=self.min_child_weight,
                    min_gain=self.gamma,
                )
                
                if not split.is_valid:
                    # Make leaf
                    values[node_id] = compute_leaf_value(s_grad, s_hess, self.reg_lambda, self.reg_alpha)
                    continue
                
                # Apply split
                features[node_id] = split.feature
                thresholds[node_id] = split.threshold
                left_children[node_id] = next_node_id
                right_children[node_id] = next_node_id + 1
                
                # Compute left/right histogram sums
                left_grad = float(np.sum(h_grad[split.feature, :split.threshold + 1]))
                left_hess = float(np.sum(h_hess[split.feature, :split.threshold + 1]))
                right_grad = s_grad - left_grad
                right_hess = s_hess - left_hess
                
                # Create child histograms via subtraction
                # Left histogram: cumsum up to threshold
                left_hist_grad = np.zeros_like(h_grad)
                left_hist_hess = np.zeros_like(h_hess)
                for f in range(n_features):
                    # For split feature, take bins <= threshold
                    left_hist_grad[f, :split.threshold + 1] = h_grad[f, :split.threshold + 1]
                    left_hist_hess[f, :split.threshold + 1] = h_hess[f, :split.threshold + 1]
                
                right_hist_grad = h_grad - left_hist_grad
                right_hist_hess = h_hess - left_hist_hess
                
                # Store child info
                left_id = next_node_id
                right_id = next_node_id + 1
                
                node_hist_grad[left_id] = left_hist_grad
                node_hist_hess[left_id] = left_hist_hess
                node_sum_grad[left_id] = left_grad
                node_sum_hess[left_id] = left_hess
                
                node_hist_grad[right_id] = right_hist_grad
                node_hist_hess[right_id] = right_hist_hess
                node_sum_grad[right_id] = right_grad
                node_sum_hess[right_id] = right_hess
                
                new_active_nodes.extend([left_id, right_id])
                next_node_id += 2
            
            active_nodes = new_active_nodes
        
        # Compute leaf values for remaining active nodes
        for node_id in active_nodes:
            s_grad = node_sum_grad.get(node_id, 0.0)
            s_hess = node_sum_hess.get(node_id, 0.0)
            values[node_id] = compute_leaf_value(s_grad, s_hess, self.reg_lambda, self.reg_alpha)
        
        # Trim arrays
        n_nodes = next_node_id
        
        return TreeStructure(
            features=features[:n_nodes],
            thresholds=thresholds[:n_nodes],
            values=values[:n_nodes],
            left_children=left_children[:n_nodes],
            right_children=right_children[:n_nodes],
            n_nodes=n_nodes,
            depth=actual_depth,
            n_features=n_features,
        )
    
    def _fit_gpu(
        self,
        y: NDArray,
        n_samples: int,
        callbacks: list[Callback] | None = None,
        eval_set: list[tuple[NDArray, NDArray]] | None = None,
        sample_weight: NDArray | None = None,
    ):
        """GPU-optimized training using growth strategies with callback support.
        
        Phase 17: Added GOSS sampling for faster training on large datasets.
        """
        from numba import cuda
        
        # Setup callbacks
        cb_manager = CallbackManager(callbacks)
        state = TrainingState(model=self, n_rounds=self.n_trees)
        cb_manager.on_train_begin(state)
        
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
        
        # Determine sampling strategy
        use_goss = self.subsample_strategy == 'goss'
        use_random_sampling = self.subsample_strategy == 'random' and self.subsample < 1.0
        
        # Train trees
        for i in range(self.n_trees):
            state.round_idx = i
            cb_manager.on_round_begin(state)
            
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
            
            # Known Limitation (1.0.0rc1): sample_weight not fully supported on GPU
            # GPU training uses histograms built from all samples; weighting would
            # require weighted histogram accumulation which is not yet implemented.
            # Workaround: Use CPU backend with ob.set_backend('cpu') for weighted training
            # TODO(v1.1): Implement GPU sample weighting via weighted histogram kernels
            
            # Apply sampling strategy (Phase 17)
            if use_goss:
                # GOSS: Compute sampling on CPU (requires sorting), then apply weights
                grad_cpu = grad_gpu.copy_to_host() if hasattr(grad_gpu, 'copy_to_host') else grad_gpu
                hess_cpu = hess_gpu.copy_to_host() if hasattr(hess_gpu, 'copy_to_host') else hess_gpu
                
                sample_result = goss_sample(
                    grad_cpu, hess_cpu,
                    top_rate=self.goss_top_rate,
                    other_rate=self.goss_other_rate,
                    seed=i,
                )
                
                # Create weighted gradient/hessian arrays
                # Zero out non-sampled samples, apply weights to sampled samples
                grad_goss = np.zeros_like(grad_cpu)
                hess_goss = np.zeros_like(hess_cpu)
                grad_goss[sample_result.indices] = grad_cpu[sample_result.indices] * sample_result.weights
                hess_goss[sample_result.indices] = hess_cpu[sample_result.indices] * sample_result.weights
                
                # Transfer to GPU
                grad_goss_gpu = cuda.to_device(grad_goss.astype(np.float32))
                hess_goss_gpu = cuda.to_device(hess_goss.astype(np.float32))
                
                # Build tree with GOSS-weighted gradients
                tree = fit_tree(
                    self.X_binned_,
                    grad_goss_gpu,
                    hess_goss_gpu,
                    max_depth=self.max_depth,
                    min_child_weight=self.min_child_weight,
                    reg_lambda=self.reg_lambda,
                    reg_alpha=self.reg_alpha,
                    gamma=self.gamma,
                    subsample=1.0,
                    colsample_bytree=self.colsample_bytree,
                )
                
                state.extra['goss_sample_rate'] = sample_result.sample_rate
            elif use_random_sampling:
                # Random subsampling (use built-in subsample parameter)
                tree = fit_tree(
                    self.X_binned_,
                    grad_gpu,
                    hess_gpu,
                    max_depth=self.max_depth,
                    min_child_weight=self.min_child_weight,
                    reg_lambda=self.reg_lambda,
                    reg_alpha=self.reg_alpha,
                    gamma=self.gamma,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                )
            else:
                # Standard training (with optional row/col subsampling in fit_tree)
                tree = fit_tree(
                    self.X_binned_,
                    grad_gpu,
                    hess_gpu,
                    max_depth=self.max_depth,
                    min_child_weight=self.min_child_weight,
                    reg_lambda=self.reg_lambda,
                    reg_alpha=self.reg_alpha,
                    gamma=self.gamma,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
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
            
            # Compute losses for callbacks
            state.train_loss = float(np.mean((pred_cpu - y) ** 2))
            
            if eval_set:
                X_val, y_val = eval_set[0]
                val_pred = self.predict(X_val)
                state.val_loss = float(np.mean((val_pred - y_val) ** 2))
            
            # Check if callbacks want to stop
            if not cb_manager.on_round_end(state):
                break
        
        cb_manager.on_train_end(state)
    
    def _fit_cpu(
        self,
        y: NDArray,
        n_samples: int,
        callbacks: list[Callback] | None = None,
        eval_set: list[tuple[NDArray, NDArray]] | None = None,
        sample_weight: NDArray | None = None,
    ):
        """CPU training path with callback support.
        
        Phase 17: Added GOSS sampling and mini-batch training.
        """
        # Setup callbacks
        cb_manager = CallbackManager(callbacks)
        state = TrainingState(model=self, n_rounds=self.n_trees)
        cb_manager.on_train_begin(state)
        
        # Initialize predictions
        pred = np.zeros(n_samples, dtype=np.float32)
        
        # Determine sampling strategy
        use_goss = self.subsample_strategy == 'goss'
        use_random_sampling = self.subsample_strategy == 'random' and self.subsample < 1.0
        use_minibatch = self.batch_size is not None and self.batch_size < n_samples
        
        # Train trees
        for i in range(self.n_trees):
            state.round_idx = i
            cb_manager.on_round_begin(state)
            
            # Compute gradients
            grad, hess = self._loss_fn(pred, y)
            grad = grad.astype(np.float32)
            hess = hess.astype(np.float32)
            
            # Apply sample weights if provided
            if sample_weight is not None:
                weights = np.asarray(sample_weight, dtype=np.float32)
                grad = grad * weights
                hess = hess * weights
            
            # Apply sampling strategy (Phase 17)
            if use_goss:
                # GOSS: Keep high-gradient samples, subsample rest
                sample_result = goss_sample(
                    grad, hess,
                    top_rate=self.goss_top_rate,
                    other_rate=self.goss_other_rate,
                    seed=i,  # Different seed per round
                )
                
                # Create weighted gradient/hessian arrays
                # Zero out non-sampled samples, apply weights to sampled samples
                grad_goss = np.zeros_like(grad)
                hess_goss = np.zeros_like(hess)
                grad_goss[sample_result.indices] = grad[sample_result.indices] * sample_result.weights
                hess_goss[sample_result.indices] = hess[sample_result.indices] * sample_result.weights
                
                # Build tree with GOSS-weighted gradients
                tree = fit_tree(
                    self.X_binned_,
                    grad_goss,
                    hess_goss,
                    max_depth=self.max_depth,
                    min_child_weight=self.min_child_weight,
                    reg_lambda=self.reg_lambda,
                    reg_alpha=self.reg_alpha,
                    gamma=self.gamma,
                    subsample=1.0,  # Already sampled via GOSS
                    colsample_bytree=self.colsample_bytree,
                )
            elif use_random_sampling:
                # Random subsampling (use built-in subsample parameter)
                tree = fit_tree(
                    self.X_binned_,
                    grad,
                    hess,
                    max_depth=self.max_depth,
                    min_child_weight=self.min_child_weight,
                    reg_lambda=self.reg_lambda,
                    reg_alpha=self.reg_alpha,
                    gamma=self.gamma,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                )
            else:
                # Standard training (with optional row/col subsampling in fit_tree)
                tree = fit_tree(
                    self.X_binned_,
                    grad,
                    hess,
                    max_depth=self.max_depth,
                    min_child_weight=self.min_child_weight,
                    reg_lambda=self.reg_lambda,
                    reg_alpha=self.reg_alpha,
                    gamma=self.gamma,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                )
            
            # Update predictions
            tree_pred = tree(self.X_binned_)
            pred += self.learning_rate * tree_pred
            
            self.trees_.append(tree)
            
            # Compute losses for callbacks
            state.train_loss = float(np.mean((pred - y) ** 2))
            
            if eval_set:
                X_val, y_val = eval_set[0]
                val_pred = self.predict(X_val)
                state.val_loss = float(np.mean((val_pred - y_val) ** 2))
            
            # Store extra info for research callbacks
            if use_goss:
                state.extra['goss_sample_rate'] = sample_result.sample_rate
            
            # Check if callbacks want to stop
            if not cb_manager.on_round_end(state):
                break
        
        cb_manager.on_train_end(state)
    
    def predict(self, X: NDArray | BinnedArray) -> NDArray:
        """Generate predictions for X.
        
        Args:
            X: Features to predict on, shape (n_samples, n_features).
               Can be raw numpy array or pre-binned BinnedArray.
               
        Returns:
            predictions: Shape (n_samples,).
            
        Raises:
            ValueError: If model is not fitted or X has wrong shape.
        """
        # Check if fitted first (Phase 20.3)
        if not self.trees_:
            raise ValueError(
                f"This {type(self).__name__} instance is not fitted yet. "
                f"Call 'fit' with appropriate arguments before using 'predict'."
            )
        
        # Validate (Phase 20.3)
        n_features = getattr(self, 'n_features_in_', None) or (self.X_binned_.n_features if self.X_binned_ else None)
        if n_features is None:
            raise ValueError("Model is not properly fitted. Missing feature count.")
        X = validate_predict_input(self, X, n_features)
        
        # Bin the data if needed, using training bin edges for consistency
        if isinstance(X, BinnedArray):
            X_binned = X
        elif self.X_binned_ is not None:
            # Use transform to apply training bin edges to new data
            X_binned = self.X_binned_.transform(X)
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
class MultiClassGradientBoosting(PersistenceMixin):
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
        subsample_strategy: Sampling strategy (Phase 17): 'none', 'random', 'goss'.
        goss_top_rate: Fraction of top-gradient samples to keep (for GOSS).
        goss_other_rate: Fraction of remaining samples to sample (for GOSS).
        
    Example:
        ```python
        import openboost as ob
        
        model = ob.MultiClassGradientBoosting(n_classes=10, n_trees=100)
        model.fit(X_train, y_train)  # y_train: 0 to 9
        predictions = model.predict(X_test)  # Returns class labels
        proba = model.predict_proba(X_test)  # Returns probabilities
        ```
        
        With GOSS sampling:
        
        ```python
        model = ob.MultiClassGradientBoosting(
            n_classes=10, n_trees=100,
            subsample_strategy='goss',
            goss_top_rate=0.2,
            goss_other_rate=0.1
        )
        ```
    """
    
    n_classes: int
    n_trees: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    gamma: float = 0.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    n_bins: int = 256
    subsample_strategy: Literal['none', 'random', 'goss'] = 'none'
    goss_top_rate: float = 0.2
    goss_other_rate: float = 0.1
    
    # Fitted attributes
    trees_: list[list[TreeStructure]] = field(default_factory=list, init=False, repr=False)
    X_binned_: BinnedArray | None = field(default=None, init=False, repr=False)
    n_features_in_: int = field(default=0, init=False, repr=False)
    
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
        
        # Determine sampling strategy (Phase 17)
        use_goss = self.subsample_strategy == 'goss'
        
        # Train trees
        for round_idx in range(self.n_trees):
            # Compute softmax gradients for all classes
            grad, hess = softmax_gradient(pred, y, self.n_classes)
            
            # Apply GOSS if enabled (Phase 17)
            if use_goss:
                # Use sum of absolute gradients across classes for sampling
                grad_magnitude = np.sum(np.abs(grad), axis=1)
                sample_result = goss_sample(
                    grad_magnitude, None,
                    top_rate=self.goss_top_rate,
                    other_rate=self.goss_other_rate,
                    seed=round_idx,
                )
                sample_indices = sample_result.indices
                sample_weights = sample_result.weights
            else:
                sample_indices = None
                sample_weights = None
            
            # Train one tree per class
            round_trees = []
            for k in range(self.n_classes):
                grad_k = grad[:, k].astype(np.float32)
                hess_k = hess[:, k].astype(np.float32)
                
                if use_goss:
                    # Apply GOSS sampling and weighting
                    # Create weighted gradient/hessian arrays
                    grad_k_goss = np.zeros_like(grad_k)
                    hess_k_goss = np.zeros_like(hess_k)
                    grad_k_goss[sample_indices] = grad_k[sample_indices] * sample_weights
                    hess_k_goss[sample_indices] = hess_k[sample_indices] * sample_weights
                    
                    tree = fit_tree(
                        self.X_binned_,
                        grad_k_goss,
                        hess_k_goss,
                        max_depth=self.max_depth,
                        min_child_weight=self.min_child_weight,
                        reg_lambda=self.reg_lambda,
                        reg_alpha=self.reg_alpha,
                        gamma=self.gamma,
                        subsample=1.0,
                        colsample_bytree=self.colsample_bytree,
                    )
                else:
                    tree = fit_tree(
                        self.X_binned_,
                        grad_k,
                        hess_k,
                        max_depth=self.max_depth,
                        min_child_weight=self.min_child_weight,
                        reg_lambda=self.reg_lambda,
                        reg_alpha=self.reg_alpha,
                        gamma=self.gamma,
                        subsample=self.subsample,
                        colsample_bytree=self.colsample_bytree,
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
        
        # Bin the data if needed, using training bin edges for consistency
        if isinstance(X, BinnedArray):
            X_binned = X
        elif self.X_binned_ is not None:
            # Use transform to apply training bin edges to new data
            X_binned = self.X_binned_.transform(X)
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

