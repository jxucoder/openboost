"""Array handling and binning for OpenBoost.

Provides `ob.array()` for converting data to the internal binned format.

Phase 14: Added missing value handling (NaN -> bin 255).
Phase 14.3: Added native categorical feature support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import numpy as np

from ._backends import get_backend, is_cuda

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


# Reserved bin index for missing values (NaN)
MISSING_BIN: int = 255


@dataclass
class BinnedArray:
    """Binned feature matrix ready for tree building.
    
    Attributes:
        data: Binned data, shape (n_features, n_samples), dtype uint8
              NaN values are encoded as bin 255 (MISSING_BIN)
        bin_edges: List of bin edges per feature, for inverse transform
        n_features: Number of features
        n_samples: Number of samples
        device: "cuda" or "cpu"
        has_missing: Boolean array (n_features,) indicating which features have NaN
        is_categorical: Boolean array (n_features,) indicating categorical features
        category_maps: List of dicts mapping original values -> bin indices (None for numeric)
        n_categories: Number of categories per feature (0 for numeric)
    """
    data: NDArray[np.uint8]  # Or DeviceNDArray for CUDA
    bin_edges: list[NDArray[np.float64]]
    n_features: int
    n_samples: int
    device: str
    has_missing: NDArray[np.bool_] = field(default_factory=lambda: np.array([], dtype=np.bool_))
    # Phase 14.3: Categorical support
    is_categorical: NDArray[np.bool_] = field(default_factory=lambda: np.array([], dtype=np.bool_))
    category_maps: list[dict | None] = field(default_factory=list)
    n_categories: NDArray[np.int32] = field(default_factory=lambda: np.array([], dtype=np.int32))
    
    def __repr__(self) -> str:
        n_missing = int(np.sum(self.has_missing)) if len(self.has_missing) > 0 else 0
        n_cat = int(np.sum(self.is_categorical)) if len(self.is_categorical) > 0 else 0
        return (
            f"BinnedArray(n_features={self.n_features}, n_samples={self.n_samples}, "
            f"device={self.device!r}, features_with_missing={n_missing}, "
            f"categorical_features={n_cat})"
        )
    
    @property
    def any_missing(self) -> bool:
        """Check if any feature has missing values."""
        return len(self.has_missing) > 0 and np.any(self.has_missing)
    
    @property
    def any_categorical(self) -> bool:
        """Check if any feature is categorical."""
        return len(self.is_categorical) > 0 and np.any(self.is_categorical)
    
    def transform(self, X: ArrayLike) -> "BinnedArray":
        """Transform new data using the bin edges from this BinnedArray.
        
        Use this method to transform test/validation data using the same
        binning learned from training data. This ensures tree splits work
        correctly across train and test sets.
        
        Args:
            X: New input features, shape (n_samples_new, n_features).
               Must have the same number of features as the training data.
               
        Returns:
            BinnedArray with new data binned using training bin edges.
            
        Example:
            >>> X_train_binned = ob.array(X_train)
            >>> model.fit(X_train_binned, y_train)
            >>> X_test_binned = X_train_binned.transform(X_test)
            >>> predictions = model.predict(X_test_binned)
        """
        # Convert to numpy
        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        
        n_samples_new, n_features_new = X_np.shape
        
        if n_features_new != self.n_features:
            raise ValueError(
                f"X has {n_features_new} features, but BinnedArray was fitted with "
                f"{self.n_features} features"
            )
        
        # Bin each feature using existing bin edges
        binned = np.zeros((n_samples_new, self.n_features), dtype=np.uint8)
        
        for j in range(self.n_features):
            col = X_np[:, j].astype(np.float64)
            nan_mask = np.isnan(col)
            
            if self.is_categorical[j] if len(self.is_categorical) > j else False:
                # Categorical feature: use category map
                cat_map = self.category_maps[j] if j < len(self.category_maps) else None
                if cat_map is not None:
                    for i, val in enumerate(col):
                        if np.isnan(val):
                            binned[i, j] = MISSING_BIN
                        else:
                            # Convert to hashable type for dict lookup
                            key = int(val) if np.isfinite(val) else val
                            binned[i, j] = cat_map.get(key, 0)  # Default to 0 for unseen
                else:
                    binned[:, j] = 0
            else:
                # Numeric feature: use bin edges with searchsorted
                edges = self.bin_edges[j]
                if len(edges) == 0:
                    # No bin edges (constant feature)
                    binned[:, j] = 0
                else:
                    # searchsorted finds the bin index
                    bin_idx = np.searchsorted(edges, col[~nan_mask], side='right')
                    # Clip to valid range (in case test values exceed training range)
                    bin_idx = np.clip(bin_idx, 0, len(edges))
                    binned[~nan_mask, j] = bin_idx.astype(np.uint8)
                
                # Handle missing values
                if np.any(nan_mask):
                    binned[nan_mask, j] = MISSING_BIN
        
        # Transpose to feature-major layout
        binned = np.ascontiguousarray(binned.T)
        
        # Move to device if needed
        if self.device == "cuda":
            from numba import cuda
            binned = cuda.to_device(binned)
        
        return BinnedArray(
            data=binned,
            bin_edges=self.bin_edges,  # Keep same bin edges
            n_features=self.n_features,
            n_samples=n_samples_new,
            device=self.device,
            has_missing=self.has_missing,
            is_categorical=self.is_categorical,
            category_maps=self.category_maps,
            n_categories=self.n_categories,
        )


def array(
    X: ArrayLike,
    n_bins: int = 256,
    *,
    categorical_features: Sequence[int] | None = None,
    device: str | None = None,
) -> BinnedArray:
    """Convert input data to binned format for tree building.
    
    This is the primary entry point for data. Binning is done once,
    then the binned data can be used for training many models.
    
    Missing values (NaN) are automatically detected and encoded as bin 255.
    The model learns the optimal direction for missing values at each split.
    
    Categorical features use native category encoding instead of quantile binning,
    enabling the model to learn optimal category groupings.
    
    Args:
        X: Input features, shape (n_samples, n_features)
           Accepts numpy arrays, PyTorch tensors, JAX arrays, CuPy arrays.
           NaN values are handled automatically.
        n_bins: Maximum number of bins for numeric features (max 255).
        categorical_features: List of column indices that are categorical.
                             These use category encoding instead of quantile binning.
                             Max 254 unique categories per feature (255 reserved for NaN).
        device: Target device ("cuda" or "cpu"). Auto-detected if None.
        
    Returns:
        BinnedArray with binned data in feature-major layout (n_features, n_samples).
        NaN values are encoded as MISSING_BIN (255).
        
    Example:
        >>> import openboost as ob
        >>> import numpy as np
        >>> 
        >>> # Numeric features with missing values
        >>> X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])
        >>> X_binned = ob.array(X)
        >>> print(X_binned.has_missing)  # [True, True]
        >>> 
        >>> # Mixed numeric and categorical
        >>> X = np.array([[25, 0, 50000], [30, 1, 60000], [35, 2, 70000]])
        >>> X_binned = ob.array(X, categorical_features=[1])  # Feature 1 is categorical
        >>> print(X_binned.is_categorical)  # [False, True, False]
    """
    # Reserve bin 255 for missing values
    if n_bins > 255:
        n_bins = 255  # Cap at 255, leaving 255 for MISSING_BIN
    
    # Convert to numpy for binning computation
    X_np = _to_numpy(X)
    
    if X_np.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features), got shape {X_np.shape}")
    
    n_samples, n_features = X_np.shape
    
    # Handle categorical features
    categorical_set = set(categorical_features) if categorical_features else set()
    
    # Validate categorical indices
    for idx in categorical_set:
        if idx < 0 or idx >= n_features:
            raise ValueError(f"categorical_features index {idx} out of range [0, {n_features})")
    
    # Process features (numeric vs categorical)
    binned, bin_edges, has_missing, is_categorical, category_maps, n_categories = _bin_features(
        X_np, n_bins, categorical_set
    )
    
    # Transpose to feature-major layout: (n_features, n_samples)
    binned = np.ascontiguousarray(binned.T)
    
    # Determine device
    if device is None:
        device = get_backend()
    
    # Transfer to GPU if needed
    if device == "cuda" and is_cuda():
        from ._backends._cuda import to_device
        binned = to_device(binned)
    
    return BinnedArray(
        data=binned,
        bin_edges=bin_edges,
        n_features=n_features,
        n_samples=n_samples,
        device=device,
        has_missing=has_missing,
        is_categorical=is_categorical,
        category_maps=category_maps,
        n_categories=n_categories,
    )


def _to_numpy(arr: ArrayLike) -> NDArray:
    """Convert various array types to numpy.
    
    Handles: numpy, PyTorch, JAX, CuPy
    """
    # Already numpy
    if isinstance(arr, np.ndarray):
        return arr
    
    # PyTorch
    if hasattr(arr, 'cpu') and hasattr(arr, 'numpy'):
        return arr.cpu().numpy()
    
    # JAX (has __array__ protocol)
    if hasattr(arr, '__array__'):
        return np.asarray(arr)
    
    # CuPy
    if hasattr(arr, 'get'):
        return arr.get()
    
    # Fallback
    return np.asarray(arr)


def _bin_features(
    X: NDArray,
    n_bins: int,
    categorical_set: set[int],
) -> tuple[NDArray[np.uint8], list[NDArray[np.float64]], NDArray[np.bool_], 
           NDArray[np.bool_], list[dict | None], NDArray[np.int32]]:
    """Bin features (numeric and categorical) with missing value handling.
    
    Args:
        X: Input data, shape (n_samples, n_features)
        n_bins: Number of bins for numeric features
        categorical_set: Set of indices for categorical features
        
    Returns:
        binned: Binned data, shape (n_samples, n_features), uint8
        bin_edges: List of bin edges per feature (empty for categorical)
        has_missing: Boolean array (n_features,) for missing values
        is_categorical: Boolean array (n_features,) for categorical features
        category_maps: List of dicts (None for numeric)
        n_categories: Number of categories per feature (0 for numeric)
    """
    from joblib import Parallel, delayed
    
    n_samples, n_features = X.shape
    
    # Pre-compute percentiles for numeric features
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    
    def bin_single_feature(f: int):
        """Bin a single feature column."""
        col = X[:, f]
        is_cat = f in categorical_set
        
        if is_cat:
            return _bin_categorical_feature(col)
        else:
            return _bin_numeric_feature(col, percentiles)
    
    # Parallel processing across features
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(bin_single_feature)(f) for f in range(n_features)
    )
    
    # Combine results
    binned = np.column_stack([r[0] for r in results])
    bin_edges = [r[1] for r in results]
    has_missing = np.array([r[2] for r in results], dtype=np.bool_)
    is_categorical = np.array([r[3] for r in results], dtype=np.bool_)
    category_maps = [r[4] for r in results]
    n_categories = np.array([r[5] for r in results], dtype=np.int32)
    
    return binned, bin_edges, has_missing, is_categorical, category_maps, n_categories


def _bin_numeric_feature(
    col: NDArray,
    percentiles: NDArray,
) -> tuple[NDArray[np.uint8], NDArray[np.float64], bool, bool, dict | None, int]:
    """Bin a numeric feature using quantiles.
    
    Returns:
        binned_col, bin_edges, has_nan, is_categorical, category_map, n_categories
    """
    col = col.astype(np.float64)
    
    # Detect missing values
    nan_mask = np.isnan(col)
    has_nan = bool(np.any(nan_mask))
    
    # Initialize output
    binned_col = np.zeros(len(col), dtype=np.uint8)
    
    if has_nan:
        valid_col = col[~nan_mask]
        
        if len(valid_col) == 0:
            edges = np.array([], dtype=np.float64)
            binned_col[:] = MISSING_BIN
        else:
            edges = np.nanpercentile(valid_col, percentiles)
            edges = np.unique(edges)
            binned_col[~nan_mask] = np.digitize(valid_col, edges).astype(np.uint8)
            binned_col[nan_mask] = MISSING_BIN
    else:
        edges = np.percentile(col, percentiles)
        edges = np.unique(edges)
        binned_col = np.digitize(col, edges).astype(np.uint8)
    
    return binned_col, edges, has_nan, False, None, 0


def _bin_categorical_feature(
    col: NDArray,
) -> tuple[NDArray[np.uint8], NDArray[np.float64], bool, bool, dict, int]:
    """Bin a categorical feature by mapping unique values to bin indices.
    
    Each unique value maps to a bin index 0, 1, 2, ...
    NaN/None values map to MISSING_BIN (255).
    
    Returns:
        binned_col, bin_edges (empty), has_nan, is_categorical, category_map, n_categories
    """
    binned_col = np.zeros(len(col), dtype=np.uint8)
    
    # Handle different types (numeric with NaN, object with None/NaN)
    try:
        # Try numeric - NaN is missing
        col_float = col.astype(np.float64)
        nan_mask = np.isnan(col_float)
    except (ValueError, TypeError):
        # Object dtype - check for None, NaN, 'nan', etc.
        nan_mask = np.array([
            v is None or (isinstance(v, float) and np.isnan(v)) or v == 'nan' or v == ''
            for v in col
        ], dtype=np.bool_)
    
    has_nan = bool(np.any(nan_mask))
    
    # Get unique non-missing values
    if has_nan:
        valid_values = col[~nan_mask]
    else:
        valid_values = col
    
    unique_vals = np.unique(valid_values)
    n_categories = len(unique_vals)
    
    if n_categories > 254:
        raise ValueError(
            f"Categorical feature has {n_categories} unique values, max is 254 "
            "(bin 255 reserved for missing values)"
        )
    
    # Create mapping: value -> bin index
    category_map = {v: i for i, v in enumerate(unique_vals)}
    
    # Encode values
    for i, v in enumerate(col):
        if nan_mask[i]:
            binned_col[i] = MISSING_BIN
        else:
            binned_col[i] = category_map[v]
    
    # Empty edges for categorical (not used in splits)
    edges = np.array([], dtype=np.float64)
    
    return binned_col, edges, has_nan, True, category_map, n_categories


def _quantile_bin(
    X: NDArray[np.floating],
    n_bins: int,
) -> tuple[NDArray[np.uint8], list[NDArray[np.float64]], NDArray[np.bool_]]:
    """Bin features using quantiles (parallelized across features).
    
    Handles missing values (NaN) by encoding them as MISSING_BIN (255).
    Bin edges are computed only on non-missing values.
    
    Note: This is a legacy function. Use _bin_features for new code.
    
    Args:
        X: Input data, shape (n_samples, n_features), may contain NaN
        n_bins: Number of bins (max 255, leaving 255 for missing)
        
    Returns:
        binned: Binned data, shape (n_samples, n_features), uint8
                NaN values are encoded as 255
        bin_edges: List of bin edges per feature
        has_missing: Boolean array (n_features,) indicating which features have NaN
    """
    # Use new function with no categorical features
    binned, bin_edges, has_missing, _, _, _ = _bin_features(X, n_bins, set())
    return binned, bin_edges, has_missing


def as_numba_array(arr):
    """Convert GPU array to Numba device array (zero-copy where possible).
    
    Supports PyTorch, JAX, CuPy arrays via __cuda_array_interface__.
    
    Args:
        arr: Array with __cuda_array_interface__ or numpy array
        
    Returns:
        Numba device array (CUDA) or numpy array (CPU)
    """
    # CUDA arrays (PyTorch .cuda(), JAX GPU, CuPy)
    if hasattr(arr, '__cuda_array_interface__'):
        if is_cuda():
            from ._backends._cuda import as_cuda_array
            return as_cuda_array(arr)
        else:
            raise RuntimeError(
                "Received CUDA array but CUDA backend not available. "
                "Call arr.cpu() first or set OPENBOOST_BACKEND=cpu"
            )
    
    # CPU arrays
    if hasattr(arr, '__array_interface__'):
        return np.asarray(arr)
    
    # Already numpy
    if isinstance(arr, np.ndarray):
        return arr
    
    raise TypeError(
        f"Cannot convert {type(arr).__name__} to Numba array. "
        "Expected numpy, PyTorch, JAX, or CuPy array."
    )


def ensure_contiguous_float32(arr) -> np.ndarray:
    """Ensure array is contiguous float32 (for grad/hess)."""
    arr = _to_numpy(arr) if not isinstance(arr, np.ndarray) else arr
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    return arr
