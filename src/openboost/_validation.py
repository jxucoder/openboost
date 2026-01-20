"""Input validation utilities for OpenBoost.

Phase 20.3: Input Validation & Error Messages

Provides clear, helpful error messages for common user mistakes.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ._array import BinnedArray


class ValidationError(ValueError):
    """Custom exception for validation errors with helpful messages."""

    pass


def validate_X(
    X: Any,
    *,
    allow_binned: bool = True,
    require_2d: bool = True,
    ensure_float32: bool = True,
    allow_nan: bool = False,
    context: str = "fit",
) -> NDArray:
    """Validate feature array X.

    Args:
        X: Input feature array
        allow_binned: Whether to accept BinnedArray
        require_2d: Whether X must be 2D
        ensure_float32: Whether to convert to float32
        allow_nan: Whether NaN values are allowed
        context: Context for error messages ('fit' or 'predict')

    Returns:
        Validated (and possibly converted) array

    Raises:
        TypeError: If X has wrong type
        ValueError: If X has wrong shape or contains invalid values
    """
    from ._array import BinnedArray

    # Check type
    if isinstance(X, BinnedArray):
        if not allow_binned:
            raise TypeError(
                f"Expected numpy array for {context}, got BinnedArray. "
                f"Pass the original numpy array instead."
            )
        return X

    if not isinstance(X, np.ndarray):
        # Try to convert
        try:
            X = np.asarray(X)
        except Exception as e:
            raise TypeError(
                f"X must be a numpy array or array-like, got {type(X).__name__}. "
                f"Conversion failed: {e}"
            ) from e

    # Check dimensions
    if require_2d:
        if X.ndim == 1:
            raise ValueError(
                f"X must be 2D with shape (n_samples, n_features), "
                f"got 1D array with shape {X.shape}. "
                f"Reshape your data using X.reshape(-1, 1) for single feature."
            )
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D with shape (n_samples, n_features), "
                f"got {X.ndim}D array with shape {X.shape}."
            )

    # Check for empty array
    if X.size == 0:
        raise ValueError(
            f"Cannot {context} on empty array. X has shape {X.shape}."
        )

    # Check dtype and convert
    if ensure_float32 and X.dtype != np.float32:
        original_dtype = X.dtype
        X = X.astype(np.float32)
        if original_dtype not in (np.float64, np.float32, np.float16):
            warnings.warn(
                f"X has dtype {original_dtype}, converting to float32. "
                f"For best performance, pass float32 arrays directly.",
                UserWarning,
                stacklevel=3,
            )

    # Check for NaN
    if not allow_nan and np.any(np.isnan(X)):
        nan_count = np.sum(np.isnan(X))
        nan_cols = np.where(np.any(np.isnan(X), axis=0))[0]
        raise ValueError(
            f"X contains {nan_count} NaN values in columns {nan_cols.tolist()}. "
            f"Options:\n"
            f"  1. Impute missing values before training\n"
            f"  2. Use ob.array(X) which handles missing values natively\n"
            f"  3. Drop rows with missing values"
        )

    # Check for infinity
    if np.any(np.isinf(X)):
        inf_count = np.sum(np.isinf(X))
        raise ValueError(
            f"X contains {inf_count} infinite values. "
            f"Replace with finite values or clip to a reasonable range."
        )

    return X


def validate_y(
    y: Any,
    n_samples: int | None = None,
    *,
    task: str = "regression",
    ensure_float32: bool = True,
    context: str = "fit",
) -> NDArray:
    """Validate target array y.

    Args:
        y: Input target array
        n_samples: Expected number of samples (must match X)
        task: Task type ('regression', 'binary', 'multiclass')
        ensure_float32: Whether to convert to float32
        context: Context for error messages

    Returns:
        Validated (and possibly converted) array

    Raises:
        TypeError: If y has wrong type
        ValueError: If y has wrong shape or values
    """
    if not isinstance(y, np.ndarray):
        try:
            y = np.asarray(y)
        except Exception as e:
            raise TypeError(
                f"y must be a numpy array or array-like, got {type(y).__name__}. "
                f"Conversion failed: {e}"
            ) from e

    # Flatten if needed
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    elif y.ndim != 1:
        raise ValueError(
            f"y must be 1D array with shape (n_samples,), "
            f"got shape {y.shape}. "
            f"For multi-output regression, use sklearn's MultiOutputRegressor wrapper."
        )

    # Check for empty
    if y.size == 0:
        raise ValueError(f"Cannot {context} with empty target array y.")

    # Check sample count matches
    if n_samples is not None and len(y) != n_samples:
        raise ValueError(
            f"X and y have inconsistent number of samples: "
            f"X has {n_samples} samples, y has {len(y)} samples."
        )

    # Check for NaN
    if np.any(np.isnan(y)):
        nan_count = np.sum(np.isnan(y))
        nan_indices = np.where(np.isnan(y))[0][:5]  # Show first 5
        raise ValueError(
            f"y contains {nan_count} NaN values at indices {nan_indices.tolist()}{'...' if nan_count > 5 else ''}. "
            f"Remove or impute missing target values."
        )

    # Check for infinity
    if np.any(np.isinf(y)):
        raise ValueError(
            f"y contains infinite values. Replace with finite values."
        )

    # Task-specific validation
    if task == "binary":
        unique_values = np.unique(y)
        if len(unique_values) > 2:
            raise ValueError(
                f"Binary classification expects 2 classes, "
                f"got {len(unique_values)} unique values: {unique_values[:10].tolist()}{'...' if len(unique_values) > 10 else ''}. "
                f"Use MultiClassGradientBoosting for multi-class problems."
            )
        if not np.all(np.isin(y, [0, 1])):
            warnings.warn(
                f"Binary classification expects y in {{0, 1}}, "
                f"got values {unique_values.tolist()}. Converting.",
                UserWarning,
                stacklevel=3,
            )

    elif task == "multiclass":
        unique_values = np.unique(y)
        if not np.issubdtype(y.dtype, np.integer):
            if not np.allclose(y, y.astype(int)):
                raise ValueError(
                    f"Multi-class classification expects integer class labels, "
                    f"got non-integer values. Convert y to integers."
                )
        if np.min(y) != 0:
            warnings.warn(
                f"Multi-class labels should start from 0, "
                f"got min={np.min(y)}. Labels will be remapped.",
                UserWarning,
                stacklevel=3,
            )

    # Convert dtype
    if ensure_float32 and y.dtype != np.float32:
        y = y.astype(np.float32)

    return y


def validate_sample_weight(
    sample_weight: Any | None,
    n_samples: int,
) -> NDArray | None:
    """Validate sample weights.

    Args:
        sample_weight: Sample weights or None
        n_samples: Expected number of samples

    Returns:
        Validated weights or None

    Raises:
        ValueError: If weights have wrong shape or invalid values
    """
    if sample_weight is None:
        return None

    if not isinstance(sample_weight, np.ndarray):
        sample_weight = np.asarray(sample_weight, dtype=np.float32)

    if sample_weight.ndim != 1:
        raise ValueError(
            f"sample_weight must be 1D array, got shape {sample_weight.shape}."
        )

    if len(sample_weight) != n_samples:
        raise ValueError(
            f"sample_weight has {len(sample_weight)} elements, "
            f"but X has {n_samples} samples."
        )

    if np.any(sample_weight < 0):
        raise ValueError(
            f"sample_weight cannot contain negative values. "
            f"Min value: {np.min(sample_weight)}"
        )

    if np.any(np.isnan(sample_weight)):
        raise ValueError("sample_weight contains NaN values.")

    return sample_weight.astype(np.float32)


def validate_eval_set(
    eval_set: list[tuple] | tuple | None,
    n_features: int,
    task: str = "regression",
) -> list[tuple[NDArray, NDArray]] | None:
    """Validate evaluation set for early stopping.

    Args:
        eval_set: List of (X, y) tuples, or a single (X, y) tuple.
                  Both formats are accepted for convenience:
                  - eval_set=[(X_val, y_val)]  # List format
                  - eval_set=(X_val, y_val)    # Single tuple (auto-wrapped)
        n_features: Expected number of features
        task: Task type for y validation

    Returns:
        Validated eval_set as list of tuples, or None
    """
    if eval_set is None:
        return None

    # Handle single tuple: eval_set=(X_val, y_val) -> [(X_val, y_val)]
    # Detect by checking if it's a tuple of 2 arrays (not a list of tuples)
    if isinstance(eval_set, tuple) and len(eval_set) == 2:
        first_elem = eval_set[0]
        # If first element looks like an array (has shape), it's a single (X, y) tuple
        if hasattr(first_elem, 'shape') or hasattr(first_elem, '__len__'):
            eval_set = [eval_set]

    if not isinstance(eval_set, list):
        raise TypeError(
            f"eval_set must be a list of (X, y) tuples or a single (X, y) tuple, "
            f"got {type(eval_set).__name__}."
        )

    validated = []
    for i, item in enumerate(eval_set):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(
                f"eval_set[{i}] must be a tuple of (X, y), got {type(item).__name__}."
            )

        X_val, y_val = item
        X_val = validate_X(X_val, allow_nan=True, context="eval")
        
        if X_val.shape[1] != n_features:
            raise ValueError(
                f"eval_set[{i}] X has {X_val.shape[1]} features, "
                f"but training data has {n_features} features."
            )

        y_val = validate_y(y_val, n_samples=X_val.shape[0], task=task, context="eval")
        validated.append((X_val, y_val))

    return validated


def validate_hyperparameters(
    n_trees: int,
    max_depth: int,
    learning_rate: float,
    min_child_weight: float,
    reg_lambda: float,
    subsample: float,
    n_samples: int | None = None,
) -> None:
    """Validate hyperparameters and warn about suspicious values.

    Args:
        n_trees: Number of trees
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        min_child_weight: Minimum child weight
        reg_lambda: L2 regularization
        subsample: Row sampling rate
        n_samples: Number of samples (for warnings)

    Raises:
        ValueError: For invalid hyperparameters
    """
    # Hard errors
    if n_trees < 1:
        raise ValueError(f"n_trees must be >= 1, got {n_trees}.")

    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}.")

    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}.")

    if min_child_weight < 0:
        raise ValueError(f"min_child_weight must be >= 0, got {min_child_weight}.")

    if reg_lambda < 0:
        raise ValueError(f"reg_lambda must be >= 0, got {reg_lambda}.")

    if not 0 < subsample <= 1:
        raise ValueError(f"subsample must be in (0, 1], got {subsample}.")

    # Warnings for suspicious values
    if learning_rate > 1:
        warnings.warn(
            f"learning_rate={learning_rate} is very high. "
            f"Typical values are 0.01-0.3. High values may cause instability.",
            UserWarning,
            stacklevel=3,
        )

    if max_depth > 15:
        warnings.warn(
            f"max_depth={max_depth} is very deep. "
            f"Deep trees may overfit. Consider max_depth=6-10.",
            UserWarning,
            stacklevel=3,
        )

    if n_samples is not None:
        if n_samples < 100 and n_trees > 50:
            warnings.warn(
                f"Only {n_samples} samples but {n_trees} trees. "
                f"High risk of overfitting. Consider reducing n_trees.",
                UserWarning,
                stacklevel=3,
            )

        if n_samples < min_child_weight * 10:
            warnings.warn(
                f"min_child_weight={min_child_weight} is large relative to "
                f"{n_samples} samples. Trees may be very shallow.",
                UserWarning,
                stacklevel=3,
            )


def check_is_fitted(model: Any, attributes: list[str]) -> None:
    """Check if model is fitted by checking for required attributes.

    Args:
        model: Model instance
        attributes: List of attribute names that should exist after fitting

    Raises:
        ValueError: If model is not fitted
    """
    missing = [attr for attr in attributes if not hasattr(model, attr) or getattr(model, attr) is None]
    
    if missing:
        raise ValueError(
            f"This {type(model).__name__} instance is not fitted yet. "
            f"Call 'fit' with appropriate arguments before using this method."
        )


def validate_predict_input(
    model: Any,
    X: Any,
    n_features_expected: int,
) -> NDArray:
    """Validate input for predict methods.

    Args:
        model: Fitted model
        X: Input features
        n_features_expected: Expected number of features from training

    Returns:
        Validated X array
    """
    from ._array import BinnedArray

    # Check if fitted
    check_is_fitted(model, ['trees_'])

    # Handle BinnedArray
    if isinstance(X, BinnedArray):
        if X.n_features != n_features_expected:
            raise ValueError(
                f"X has {X.n_features} features, but model was trained "
                f"with {n_features_expected} features."
            )
        return X

    # Validate regular array
    X = validate_X(X, allow_nan=True, context="predict")

    if X.shape[1] != n_features_expected:
        raise ValueError(
            f"X has {X.shape[1]} features, but model was trained "
            f"with {n_features_expected} features."
        )

    return X
