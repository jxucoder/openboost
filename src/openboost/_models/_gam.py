"""GPU-accelerated Generalized Additive Model (GAM).

This implements an EBM-style model that's fully GPU-parallelized:
- Shape functions learned via gradient boosting
- All features updated in parallel each round
- Inherently interpretable (each feature has a 1D lookup table)

Unlike InterpretML's EBM which trains features sequentially (CPU-bound),
this trains all feature shape functions simultaneously on GPU.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .._array import BinnedArray, array
from .._backends import is_cuda
from .._loss import LossFunction, get_loss_function
from .._persistence import PersistenceMixin

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class OpenBoostGAM(PersistenceMixin):
    """GPU-accelerated Generalized Additive Model.
    
    An interpretable model where:
        prediction = sum(shape_function[i](feature[i]) for all features)
    
    Each shape function is a lookup table mapping binned feature values
    to contribution scores. Trained via parallel gradient boosting.
    
    Args:
        n_rounds: Number of boosting rounds.
        learning_rate: Shrinkage factor (smaller = more stable, needs more rounds).
        reg_lambda: L2 regularization on leaf values.
        loss: Loss function ('mse', 'logloss', or callable).
        n_bins: Number of bins for histogram building (2-256). Binned data is
            uint8 with bin 255 reserved for missing values, so at most 254
            usable bins; 255/256 are clamped to 254 by ``ob.array``. Shape
            function tables are always allocated 256-wide so every uint8 bin
            index (including the missing-value bin) stays in bounds.

    Example:
        ```python
        import openboost as ob
        
        gam = ob.OpenBoostGAM(n_rounds=1000, learning_rate=0.01)
        gam.fit(X_train, y_train)
        predictions = gam.predict(X_test)
        
        # Interpret: plot shape function for feature 0
        gam.plot_shape_function(0, feature_name="age")
        ```
    """
    
    n_rounds: int = 1000
    learning_rate: float = 0.01
    reg_lambda: float = 1.0
    loss: str | LossFunction = 'mse'
    n_bins: int = 256
    n_trees: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # uint8 binning contract: at most 254 usable bins, bin 255 reserved for
        # missing values. 256 stays accepted as "maximum resolution" (the
        # historical default; ob.array clamps it to 254); anything outside
        # [2, 256] cannot be represented and fails fast here.
        if not 2 <= self.n_bins <= 256:
            raise ValueError(
                f"n_bins must be in [2, 256] (uint8 binning: at most 254 usable bins, "
                f"bin 255 reserved for missing values); got {self.n_bins}"
            )
        if self.n_trees is not None:
            if self.n_rounds != 1000:
                warnings.warn(
                    "Both n_trees and n_rounds specified. Using n_trees.",
                    UserWarning,
                    stacklevel=2,
                )
            self.n_rounds = self.n_trees

    # Fitted attributes
    shape_values_: NDArray | None = field(default=None, init=False, repr=False)
    X_binned_: BinnedArray | None = field(default=None, init=False, repr=False)
    _loss_fn: LossFunction | None = field(default=None, init=False, repr=False)
    
    def fit(self, X: NDArray, y: NDArray) -> OpenBoostGAM:
        """Fit the GAM model.
        
        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training targets, shape (n_samples,).
            
        Returns:
            self: The fitted model.
        """
        y = np.asarray(y, dtype=np.float32).ravel()

        # LOW-12: Initialize base score
        loss_name = self.loss if isinstance(self.loss, str) else ''
        if loss_name in ('logloss', 'binary_crossentropy'):
            p = np.clip(np.mean(y), 1e-7, 1 - 1e-7)
            self.base_score_ = np.float32(np.log(p / (1 - p)))
        else:
            self.base_score_ = np.float32(np.mean(y))

        # Get loss function
        self._loss_fn = get_loss_function(self.loss)
        
        # Bin the data
        if isinstance(X, BinnedArray):
            self.X_binned_ = X
        else:
            self.X_binned_ = array(X, n_bins=self.n_bins)
        
        # Choose training path
        if is_cuda():
            self._fit_gpu(y)
        else:
            self._fit_cpu(y)
        
        return self
    
    def _fit_gpu(self, y: NDArray):
        """GPU training path - all features in parallel."""
        from numba import cuda

        from .._backends._cuda import build_histogram_cuda
        
        n_features = self.X_binned_.n_features
        n_samples = self.X_binned_.n_samples
        binned_gpu = self.X_binned_.data  # (n_features, n_samples) on GPU
        
        # Initialize shape functions to zero.
        # Always 256-wide regardless of n_bins: histogram kernels emit
        # (n_features, 256) and uint8 bin indices (incl. MISSING_BIN=255) must
        # never index out of bounds.
        shape_values_gpu = cuda.device_array((n_features, 256), dtype=np.float32)
        _fill_zeros_2d_gpu(shape_values_gpu)
        
        # Initialize predictions with base score
        base = getattr(self, 'base_score_', np.float32(0.0))
        pred_host = np.full(n_samples, base, dtype=np.float32)
        pred_gpu = cuda.to_device(pred_host)
        
        y_gpu = cuda.to_device(y)
        
        # Boosting loop
        for _ in range(self.n_rounds):
            # Compute gradients on GPU
            grad_gpu, hess_gpu = self._loss_fn(pred_gpu, y_gpu)
            
            # Build histograms for ALL features (your existing kernel!)
            hist_grad, hist_hess = build_histogram_cuda(binned_gpu, grad_gpu, hess_gpu)
            
            # Update ALL shape functions in parallel
            threads = 256
            blocks = (n_features * 256 + threads - 1) // threads
            _update_shape_functions_kernel[blocks, threads](
                hist_grad, hist_hess, shape_values_gpu,
                np.float32(self.learning_rate), np.float32(self.reg_lambda)
            )
            
            # Update predictions using shape functions
            threads = 256
            blocks = (n_samples + threads - 1) // threads
            _predict_gam_kernel[blocks, threads](binned_gpu, shape_values_gpu, pred_gpu)
        
        self.shape_values_ = shape_values_gpu.copy_to_host()
    
    def _fit_cpu(self, y: NDArray):
        """CPU training path."""
        from .._backends._cpu import build_histogram_cpu

        n_features = self.X_binned_.n_features
        n_samples = self.X_binned_.n_samples

        # Get binned data on CPU
        if hasattr(self.X_binned_.data, 'copy_to_host'):
            binned = self.X_binned_.data.copy_to_host()
        else:
            binned = np.asarray(self.X_binned_.data)

        # Always 256-wide regardless of n_bins: build_histogram_cpu returns
        # (n_features, 256) and uint8 bin indices (incl. MISSING_BIN=255) must
        # never index out of bounds.
        shape_values = np.zeros((n_features, 256), dtype=np.float32)
        base = getattr(self, 'base_score_', np.float32(0.0))
        pred = np.full(n_samples, base, dtype=np.float32)
        
        for _ in range(self.n_rounds):
            # Compute gradients
            grad, hess = self._loss_fn(pred, y)
            grad = grad.astype(np.float32)
            hess = hess.astype(np.float32)
            
            # Build histograms
            hist_grad, hist_hess = build_histogram_cpu(binned, grad, hess)
            
            # Update shape functions
            mask = hist_hess > 0
            updates = np.zeros_like(hist_grad)
            updates[mask] = -hist_grad[mask] / (hist_hess[mask] + self.reg_lambda)
            shape_values += self.learning_rate * updates
            
            # Update predictions
            pred = self._predict_from_shape(binned, shape_values)
        
        self.shape_values_ = shape_values
    
    def _predict_from_shape(self, binned: NDArray, shape_values: NDArray) -> NDArray:
        """CPU prediction using shape functions."""
        n_samples = binned.shape[1]
        n_features = binned.shape[0]
        base = getattr(self, 'base_score_', np.float32(0.0))
        pred = np.full(n_samples, base, dtype=np.float32)
        for f in range(n_features):
            pred += shape_values[f, binned[f, :]]
        return pred
    
    def predict(self, X: NDArray | BinnedArray) -> NDArray:
        """Generate predictions.
        
        Args:
            X: Features, shape (n_samples, n_features).
            
        Returns:
            predictions: Shape (n_samples,).
        """
        if self.shape_values_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Bin the data if needed, using training bin edges for consistency
        if isinstance(X, BinnedArray):
            X_binned = X
        elif self.X_binned_ is not None:
            X_binned = self.X_binned_.transform(X)
        else:
            X_binned = array(X, n_bins=self.n_bins)
        
        n_samples = X_binned.n_samples
        
        if is_cuda():
            from numba import cuda
            
            binned_gpu = X_binned.data
            shape_gpu = cuda.to_device(self.shape_values_)
            pred_gpu = cuda.device_array(n_samples, dtype=np.float32)
            
            threads = 256
            blocks = (n_samples + threads - 1) // threads
            _predict_gam_kernel[blocks, threads](binned_gpu, shape_gpu, pred_gpu)

            result = pred_gpu.copy_to_host()
            base = getattr(self, 'base_score_', np.float32(0.0))
            if base != 0.0:
                result += base
            return result
        else:
            if hasattr(X_binned.data, 'copy_to_host'):
                binned = X_binned.data.copy_to_host()
            else:
                binned = np.asarray(X_binned.data)
            return self._predict_from_shape(binned, self.shape_values_)
    
    def predict_proba(self, X: NDArray | BinnedArray) -> NDArray:
        """Predict class probabilities (for classification).
        
        Only valid when loss='logloss'.
        """
        if self.loss not in ('logloss', 'binary_crossentropy'):
            raise ValueError("predict_proba only available for classification losses")
        
        raw_pred = self.predict(X)
        prob_1 = 1 / (1 + np.exp(-raw_pred))
        prob_0 = 1 - prob_1
        return np.column_stack([prob_0, prob_1])
    
    def get_feature_importance(self) -> NDArray:
        """Get feature importance based on shape function variance.
        
        Returns:
            importance: Shape (n_features,), higher = more important.
        """
        if self.shape_values_ is None:
            raise RuntimeError("Model not fitted.")
        
        # Importance = variance of shape function (how much it varies)
        return np.var(self.shape_values_, axis=1)
    
    def get_shape_function(self, feature_idx: int) -> NDArray:
        """Get the shape function values for a feature.
        
        Args:
            feature_idx: Index of the feature.
            
        Returns:
            values: Shape (256,), contribution for each bin.
        """
        if self.shape_values_ is None:
            raise RuntimeError("Model not fitted.")
        return self.shape_values_[feature_idx].copy()
    
    def plot_shape_function(self, feature_idx: int, feature_name: str | None = None, ax=None):
        """Plot the shape function for a feature.

        The x-axis shows original feature values, recovered from the bin
        edges learned at fit time. Categorical or constant features (which
        have no numeric bin edges) fall back to raw bin indices. The
        contribution learned for the missing-value bin (255) is not shown.

        Args:
            feature_idx: Index of the feature to plot.
            feature_name: Optional name for the x-axis label.
            ax: Optional matplotlib Axes to draw on. When None, a new
                figure and axes are created.

        Returns:
            The matplotlib Axes containing the plot.

        Raises:
            RuntimeError: If the model is not fitted.
            ValueError: If feature_idx is out of range.
        """
        if self.shape_values_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        n_features = self.shape_values_.shape[0]
        if not 0 <= feature_idx < n_features:
            raise ValueError(
                f"feature_idx={feature_idx} is out of range for a model "
                f"fitted with {n_features} features"
            )

        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib") from err

        values = self.shape_values_[feature_idx]
        label = feature_name if feature_name is not None else f"Feature {feature_idx}"

        edges = np.array([], dtype=np.float64)
        if self.X_binned_ is not None and feature_idx < len(self.X_binned_.bin_edges):
            edges = np.asarray(self.X_binned_.bin_edges[feature_idx], dtype=np.float64)

        if edges.size > 0:
            # ob.array bins via searchsorted + clip, so occupied bins are
            # 0..len(edges)-1: bin 0 lies below edges[0], bin b between
            # edges[b-1] and edges[b], and the last bin extends past
            # edges[-1]. Use edge midpoints as representative x positions,
            # anchored at the outermost edges.
            n_used = edges.size
            x = np.empty(n_used, dtype=np.float64)
            x[0] = edges[0]
            x[-1] = edges[-1]
            if n_used > 2:
                x[1:-1] = 0.5 * (edges[:-2] + edges[1:-1])
            y_vals = values[:n_used]
            xlabel = label
        else:
            x = np.arange(values.shape[0])
            y_vals = values
            xlabel = f"{label} (binned value)"

        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=(10, 4))

        ax.step(x, y_vals, where='mid')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Contribution to prediction")
        ax.set_title(f"Shape Function: {label}")
        if created_fig:
            fig.tight_layout()
        return ax


# =============================================================================
# CUDA Kernels
# =============================================================================

_update_shape_functions_kernel = None
_predict_gam_kernel = None
_fill_zeros_2d_kernel = None
_fill_zeros_1d_kernel = None


def _init_cuda_kernels():
    """Initialize CUDA kernels lazily."""
    global _update_shape_functions_kernel, _predict_gam_kernel
    global _fill_zeros_2d_kernel, _fill_zeros_1d_kernel
    
    if _update_shape_functions_kernel is not None:
        return
    
    from numba import cuda, float32, int32
    
    @cuda.jit
    def update_shape_kernel(hist_grad, hist_hess, shape_values, learning_rate, reg_lambda):
        """Update all shape functions in parallel from histograms."""
        idx = cuda.grid(1)
        n_features = hist_grad.shape[0]
        total_elements = n_features * 256
        
        if idx < total_elements:
            feature = idx // 256
            bin_idx = idx % 256
            
            g = hist_grad[feature, bin_idx]
            h = hist_hess[feature, bin_idx]
            
            if h > float32(0.0):
                update = -g / (h + reg_lambda)
                shape_values[feature, bin_idx] += learning_rate * update
    
    @cuda.jit
    def predict_kernel(binned, shape_values, predictions):
        """Predict by summing shape function lookups."""
        sample_idx = cuda.grid(1)
        n_features = binned.shape[0]
        n_samples = binned.shape[1]
        
        if sample_idx < n_samples:
            total = float32(0.0)
            for f in range(n_features):
                bin_idx = binned[f, sample_idx]
                total += shape_values[f, int32(bin_idx)]
            predictions[sample_idx] = total
    
    @cuda.jit
    def fill_zeros_2d(arr):
        """Fill 2D array with zeros."""
        idx = cuda.grid(1)
        rows, cols = arr.shape
        total = rows * cols
        if idx < total:
            r = idx // cols
            c = idx % cols
            arr[r, c] = float32(0.0)
    
    @cuda.jit
    def fill_zeros_1d(arr):
        """Fill 1D array with zeros."""
        idx = cuda.grid(1)
        if idx < arr.shape[0]:
            arr[idx] = float32(0.0)
    
    _update_shape_functions_kernel = update_shape_kernel
    _predict_gam_kernel = predict_kernel
    _fill_zeros_2d_kernel = fill_zeros_2d
    _fill_zeros_1d_kernel = fill_zeros_1d


def _fill_zeros_2d_gpu(arr):
    """Fill 2D GPU array with zeros."""
    _init_cuda_kernels()
    rows, cols = arr.shape
    total = rows * cols
    threads = 256
    blocks = (total + threads - 1) // threads
    _fill_zeros_2d_kernel[blocks, threads](arr)


def _fill_zeros_1d_gpu(arr):
    """Fill 1D GPU array with zeros."""
    _init_cuda_kernels()
    n = arr.shape[0]
    threads = 256
    blocks = (n + threads - 1) // threads
    _fill_zeros_1d_kernel[blocks, threads](arr)


# Initialize kernels if CUDA available
if is_cuda():
    try:
        _init_cuda_kernels()
    except Exception:
        warnings.warn("Failed to compile GAM CUDA kernels; will retry on first use", stacklevel=1)

