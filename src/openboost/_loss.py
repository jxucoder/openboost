"""Loss functions and GPU gradient computation for OpenBoost.

Provides efficient GPU kernels for computing gradients and hessians
of common loss functions, enabling fully batched training.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from ._backends import is_cuda

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Type alias for loss functions
LossFunction = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]

# =============================================================================
# Loss Registry (Extensibility)
# =============================================================================

# name -> gradient fn, or a factory marked with ``__openboost_factory__ = True``
# for parameterized losses (called as ``factory(**kwargs)`` to build the fn).
# Seeded with the built-in losses at the bottom of this module; extended at
# runtime via ``register_loss``.
_LOSS_REGISTRY: dict[str, LossFunction] = {}

# name -> true scalar loss fn registered via ``register_loss(loss_value_fn=...)``
_LOSS_VALUE_REGISTRY: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {}

# Snapshot of the built-in seed, used to detect overridden built-ins.
_BUILTIN_LOSS_SEED: dict[str, LossFunction] = {}


def register_loss(
    name: str,
    fn: LossFunction,
    *,
    loss_value_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    override: bool = False,
) -> LossFunction:
    """Register a custom loss function under a string name.

    After registration the name works everywhere a built-in loss name does,
    e.g. ``GradientBoosting(loss='myloss')``.

    Args:
        name: Name to register the loss under.
        fn: Loss callable with signature ``fn(pred, y) -> (grad, hess)``.
        loss_value_fn: Optional callable ``(pred, y) -> float`` returning the
            TRUE scalar loss value. When provided, training history and early
            stopping report this value instead of the second-order Taylor
            proxy ``mean(grad^2 / (2*hess))``.
        override: Pass True to replace an existing registration (including
            shadowing a built-in). Without it a duplicate name raises
            ValueError.

    Returns:
        ``fn`` unchanged.

    Precedence for the reported train/val loss value of a loss:
        1. ``loss_value_fn`` registered here for the name.
        2. A ``loss_value`` attribute on the loss callable
           (``fn.loss_value = lambda pred, y: ...``).
        3. The built-in formula (built-in names only).
        4. The second-order Taylor proxy ``mean(grad^2 / (2*hess))``.

    Example:
        >>> def my_mse(pred, y):
        ...     return (pred - y).astype(np.float32), np.ones_like(pred, dtype=np.float32)
        >>> register_loss('my_mse', my_mse,
        ...               loss_value_fn=lambda pred, y: float(np.mean((pred - y) ** 2)))
        >>> model = GradientBoosting(loss='my_mse')
    """
    if not isinstance(name, str) or not name:
        raise TypeError(f"Loss name must be a non-empty string, got {name!r}")
    if not callable(fn):
        raise TypeError(f"Loss function must be callable, got {type(fn).__name__}")
    if loss_value_fn is not None and not callable(loss_value_fn):
        raise TypeError(
            f"loss_value_fn must be callable, got {type(loss_value_fn).__name__}"
        )
    if not override and name in _LOSS_REGISTRY:
        raise ValueError(
            f"Loss '{name}' is already registered. Pass override=True to replace it."
        )
    _LOSS_REGISTRY[name] = fn
    if loss_value_fn is not None:
        _LOSS_VALUE_REGISTRY[name] = loss_value_fn
    else:
        # Don't leave a stale value fn behind when overriding.
        _LOSS_VALUE_REGISTRY.pop(name, None)
    return fn


def is_builtin_loss(loss) -> bool:
    """True if *loss* names a built-in loss that has not been overridden."""
    if not isinstance(loss, str):
        return False
    entry = _LOSS_REGISTRY.get(loss)
    return entry is not None and _BUILTIN_LOSS_SEED.get(loss) is entry


def device_loss(fn: LossFunction) -> LossFunction:
    """Mark a custom loss as device-native (opt-in GPU contract).

    On the CUDA backend an unmarked custom loss triggers a host round-trip
    every boosting round: predictions are copied to the host, the callable is
    invoked with numpy arrays, and the returned (grad, hess) are copied back
    to the device.

    A callable decorated with ``@openboost.device_loss`` instead receives the
    DEVICE prediction array and the device-resident targets as-is (targets are
    moved to the device once per fit and cached), and MUST return device
    (grad, hess) arrays of dtype float32 with the same length as ``pred``.

    On the CPU backend the marker is a no-op: the callable simply receives
    numpy arrays like any other custom loss.

    Example:
        >>> @openboost.device_loss
        ... def my_gpu_mse(pred_dev, y_dev):
        ...     # pred_dev / y_dev are device arrays; return device arrays.
        ...     ...
    """
    if not callable(fn):
        raise TypeError(f"device_loss expects a callable, got {type(fn).__name__}")
    fn.__openboost_device__ = True
    return fn


def get_loss_function(loss: str | LossFunction, **kwargs) -> LossFunction:
    """Get a loss function by name or return custom callable.

    Args:
        loss: Loss function name or callable. Available:
            - 'mse': Mean Squared Error (regression)
            - 'mae': Mean Absolute Error (L1 regression)
            - 'huber': Huber loss (robust regression)
            - 'logloss': Binary cross-entropy (classification)
            - 'quantile': Quantile regression (percentile prediction)
            - 'poisson': Poisson deviance (count data)
            - 'gamma': Gamma deviance (positive continuous)
            - 'tweedie': Tweedie deviance (compound Poisson-Gamma)
            - any name registered via ``register_loss``
        **kwargs: Additional parameters for specific losses:
            - quantile_alpha: Quantile level for 'quantile' loss (default 0.5)
            - tweedie_rho: Variance power for 'tweedie' loss (default 1.5)

    Returns:
        Loss function callable.

    Examples:
        >>> loss_fn = get_loss_function('mse')
        >>> loss_fn = get_loss_function('quantile', quantile_alpha=0.9)
        >>> loss_fn = get_loss_function('tweedie', tweedie_rho=1.5)
    """
    if callable(loss):
        return loss

    entry = _LOSS_REGISTRY.get(loss)
    if entry is None:
        available = ', '.join(sorted(_LOSS_REGISTRY))
        raise ValueError(f"Unknown loss '{loss}'. Available: {available}")

    # Parameterized built-ins are stored as factories that close over kwargs.
    if getattr(entry, '__openboost_factory__', False):
        return entry(**kwargs)
    return entry


def compute_loss_value(loss_name: str, pred: np.ndarray, y: np.ndarray, **kwargs) -> float:
    """Compute the actual scalar loss value (not the grad/hess proxy).

    For built-in objectives this uses the true loss formula.  For custom
    (registered) losses, precedence is:

    1. ``loss_value_fn`` passed to :func:`register_loss` for this name.
    2. A ``loss_value`` attribute on the registered gradient callable
       (``fn.loss_value = lambda pred, y: ...``).
    3. The second-order Taylor approximation ``mean(grad^2 / (2 * hess))``.
    """
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Custom true-loss hooks take precedence (also covers overridden built-ins).
    value_fn = _LOSS_VALUE_REGISTRY.get(loss_name)
    if value_fn is None:
        entry = _LOSS_REGISTRY.get(loss_name)
        if entry is not None:
            value_fn = getattr(entry, 'loss_value', None)
    if value_fn is not None:
        return float(value_fn(pred, y))

    if not is_builtin_loss(loss_name):
        # Unknown or custom loss without a value hook: grad/hess proxy.
        return _grad_hess_proxy(loss_name, pred, y, **kwargs)

    if loss_name == 'mse' or loss_name == 'squared_error':
        return float(np.mean((pred - y) ** 2))

    if loss_name == 'mae' or loss_name == 'l1' or loss_name == 'absolute_error':
        return float(np.mean(np.abs(pred - y)))

    if loss_name == 'huber':
        delta = kwargs.get('huber_delta', 1.0)
        diff = np.abs(pred - y)
        loss = np.where(diff <= delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta))
        return float(np.mean(loss))

    if loss_name == 'quantile':
        alpha = kwargs.get('quantile_alpha', 0.5)
        residual = y - pred
        return float(np.mean(np.where(residual >= 0, alpha * residual, (alpha - 1) * residual)))

    if loss_name == 'logloss' or loss_name == 'binary_crossentropy':
        p = 1.0 / (1.0 + np.exp(-np.clip(pred, -500, 500)))
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    if loss_name == 'poisson':
        return float(np.mean(np.exp(np.clip(pred, -20, 20)) - y * pred))

    if loss_name == 'gamma':
        return float(np.mean(pred + y * np.exp(-np.clip(pred, -20, 20))))

    if loss_name == 'tweedie':
        rho = kwargs.get('tweedie_rho', 1.5)
        mu = np.exp(np.clip(pred, -20, 20))
        return float(np.mean(-y * mu ** (1 - rho) / (1 - rho) + mu ** (2 - rho) / (2 - rho)))

    # Unknown/custom loss: fall back to grad/hess proxy
    return _grad_hess_proxy(loss_name, pred, y, **kwargs)


def _grad_hess_proxy(loss_name, pred, y, **kwargs):
    """Fallback: second-order Taylor approximation ``mean(grad^2 / (2*hess))``."""
    loss_fn = get_loss_function(loss_name, **kwargs)
    grad, hess = loss_fn(np.asarray(pred, dtype=np.float32),
                         np.asarray(y, dtype=np.float32))
    if hasattr(grad, 'copy_to_host'):
        grad = grad.copy_to_host()
    if hasattr(hess, 'copy_to_host'):
        hess = hess.copy_to_host()
    grad = np.asarray(grad, dtype=np.float64)
    hess = np.maximum(np.asarray(hess, dtype=np.float64), 1e-10)
    return float(np.mean(grad ** 2 / (2.0 * hess)))


# =============================================================================
# MSE Loss (Regression)
# =============================================================================

def mse_gradient(pred: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """Compute MSE gradient and hessian.

    Loss: L = 0.5 * (pred - y)^2
    Gradient: dL/dpred = (pred - y)
    Hessian: d²L/dpred² = 1

    Uses the 0.5 * MSE convention (matching XGBoost) so that
    reg_lambda has equivalent effect across libraries.
    """
    if is_cuda():
        return _mse_gradient_gpu(pred, y)
    return _mse_gradient_cpu(pred, y)


def _mse_gradient_cpu(pred: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """CPU implementation of MSE gradient."""
    grad = (pred - y).astype(np.float32)
    hess = np.full_like(pred, 1.0, dtype=np.float32)
    return grad, hess


def _mse_gradient_gpu(pred, y):
    """GPU implementation of MSE gradient."""
    from numba import cuda
    _ensure_mse_kernel()

    # Handle device arrays
    if hasattr(pred, 'copy_to_host'):
        n = pred.shape[0]
    else:
        n = len(pred)
        pred = cuda.to_device(np.asarray(pred, dtype=np.float32))

    if not hasattr(y, 'copy_to_host'):
        y = cuda.to_device(np.asarray(y, dtype=np.float32))

    grad = cuda.device_array(n, dtype=np.float32)
    hess = cuda.device_array(n, dtype=np.float32)

    threads = 256
    blocks = (n + threads - 1) // threads
    _mse_gradient_kernel[blocks, threads](pred, y, grad, hess, n)

    return grad, hess


def _get_mse_kernel():
    """Lazily compile MSE gradient kernel."""
    from numba import cuda
    
    @cuda.jit
    def kernel(pred, y, grad, hess, n):
        idx = cuda.grid(1)
        if idx < n:
            grad[idx] = pred[idx] - y[idx]
            hess[idx] = 1.0
    
    return kernel


_mse_gradient_kernel = None


def _ensure_mse_kernel():
    global _mse_gradient_kernel
    if _mse_gradient_kernel is None:
        _mse_gradient_kernel = _get_mse_kernel()
    return _mse_gradient_kernel


# Eager initialization on module load if CUDA available
if is_cuda():
    try:
        _mse_gradient_kernel = _get_mse_kernel()
    except Exception:
        warnings.warn("Failed to compile MSE CUDA kernel; will retry on first use", stacklevel=1)


# =============================================================================
# LogLoss (Binary Classification)
# =============================================================================

def logloss_gradient(pred: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """Compute LogLoss gradient and hessian.
    
    Loss: L = -y*log(p) - (1-y)*log(1-p), where p = sigmoid(pred)
    Gradient: dL/dpred = p - y
    Hessian: d²L/dpred² = p * (1 - p)
    """
    if is_cuda():
        return _logloss_gradient_gpu(pred, y)
    return _logloss_gradient_cpu(pred, y)


def _sigmoid(x: NDArray) -> NDArray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def _logloss_gradient_cpu(pred: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """CPU implementation of LogLoss gradient."""
    p = _sigmoid(pred)
    grad = (p - y).astype(np.float32)
    hess = (p * (1 - p)).astype(np.float32)
    # Clip hessian to avoid numerical issues
    hess = np.clip(hess, 1e-6, 1.0 - 1e-6)
    return grad, hess


def _ensure_logloss_kernel():
    global _logloss_gradient_kernel
    if _logloss_gradient_kernel is None:
        _logloss_gradient_kernel = _get_logloss_kernel()
    return _logloss_gradient_kernel


def _logloss_gradient_gpu(pred, y):
    """GPU implementation of LogLoss gradient."""
    from numba import cuda
    _ensure_logloss_kernel()

    if hasattr(pred, 'copy_to_host'):
        n = pred.shape[0]
    else:
        n = len(pred)
        pred = cuda.to_device(np.asarray(pred, dtype=np.float32))
    
    if not hasattr(y, 'copy_to_host'):
        y = cuda.to_device(np.asarray(y, dtype=np.float32))
    
    grad = cuda.device_array(n, dtype=np.float32)
    hess = cuda.device_array(n, dtype=np.float32)
    
    threads = 256
    blocks = (n + threads - 1) // threads
    _logloss_gradient_kernel[blocks, threads](pred, y, grad, hess, n)
    
    return grad, hess


def _get_logloss_kernel():
    """Lazily compile LogLoss gradient kernel."""
    import math

    from numba import cuda
    
    @cuda.jit
    def kernel(pred, y, grad, hess, n):
        idx = cuda.grid(1)
        if idx < n:
            # Numerically stable sigmoid
            x = pred[idx]
            if x >= 0:
                p = 1.0 / (1.0 + math.exp(-x))
            else:
                exp_x = math.exp(x)
                p = exp_x / (1.0 + exp_x)
            
            grad[idx] = p - y[idx]
            h = p * (1.0 - p)
            # Clip hessian
            hess[idx] = max(1e-6, min(h, 1.0 - 1e-6))
    
    return kernel


_logloss_gradient_kernel = None

if is_cuda():
    try:
        _logloss_gradient_kernel = _get_logloss_kernel()
    except Exception:
        warnings.warn("Failed to compile LogLoss CUDA kernel; will retry on first use", stacklevel=1)


# =============================================================================
# Huber Loss (Robust Regression)
# =============================================================================

def huber_gradient(pred: NDArray, y: NDArray, delta: float = 1.0) -> tuple[NDArray, NDArray]:
    """Compute Huber loss gradient and hessian.
    
    Loss: L = 0.5 * (pred - y)^2           if |pred - y| <= delta
              delta * |pred - y| - 0.5 * delta^2  otherwise
    """
    if is_cuda():
        return _huber_gradient_gpu(pred, y, delta)
    return _huber_gradient_cpu(pred, y, delta)


def _huber_gradient_cpu(pred: NDArray, y: NDArray, delta: float = 1.0) -> tuple[NDArray, NDArray]:
    """CPU implementation of Huber gradient."""
    diff = pred - y
    abs_diff = np.abs(diff)
    
    # Gradient
    grad = np.where(abs_diff <= delta, diff, delta * np.sign(diff))
    
    # Hessian (second derivative)
    hess = np.where(abs_diff <= delta, 1.0, 0.0)
    # Add small constant for stability
    hess = np.maximum(hess, 1e-6)
    
    return grad.astype(np.float32), hess.astype(np.float32)


def _ensure_huber_kernel():
    global _huber_gradient_kernel
    if _huber_gradient_kernel is None:
        _huber_gradient_kernel = _get_huber_kernel()
    return _huber_gradient_kernel


def _huber_gradient_gpu(pred, y, delta: float = 1.0):
    """GPU implementation of Huber gradient."""
    from numba import cuda
    _ensure_huber_kernel()

    if hasattr(pred, 'copy_to_host'):
        n = pred.shape[0]
    else:
        n = len(pred)
        pred = cuda.to_device(np.asarray(pred, dtype=np.float32))
    
    if not hasattr(y, 'copy_to_host'):
        y = cuda.to_device(np.asarray(y, dtype=np.float32))
    
    grad = cuda.device_array(n, dtype=np.float32)
    hess = cuda.device_array(n, dtype=np.float32)
    
    threads = 256
    blocks = (n + threads - 1) // threads
    _huber_gradient_kernel[blocks, threads](pred, y, grad, hess, n, delta)
    
    return grad, hess


def _get_huber_kernel():
    """Lazily compile Huber gradient kernel."""
    from numba import cuda
    
    @cuda.jit
    def kernel(pred, y, grad, hess, n, delta):
        idx = cuda.grid(1)
        if idx < n:
            diff = pred[idx] - y[idx]
            abs_diff = abs(diff)
            
            if abs_diff <= delta:
                grad[idx] = diff
                hess[idx] = 1.0
            else:
                if diff > 0:
                    grad[idx] = delta
                else:
                    grad[idx] = -delta
                hess[idx] = 1e-6  # Small constant for stability
    
    return kernel


_huber_gradient_kernel = None

if is_cuda():
    try:
        _huber_gradient_kernel = _get_huber_kernel()
    except Exception:
        warnings.warn("Failed to compile Huber CUDA kernel; will retry on first use", stacklevel=1)


# =============================================================================
# MAE Loss (L1 Regression) - Phase 9.1
# =============================================================================

def mae_gradient(pred: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """Compute MAE (L1) gradient and hessian.
    
    Loss: L = |pred - y|
    Gradient: sign(pred - y)
    Hessian: 0 (use small constant for GBDT stability)
    
    Note: MAE is not twice-differentiable at pred=y, so we use a small
    constant hessian. This is the standard approach in XGBoost/LightGBM.
    """
    if is_cuda():
        return _mae_gradient_gpu(pred, y)
    return _mae_gradient_cpu(pred, y)


def _mae_gradient_cpu(pred: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """CPU implementation of MAE gradient."""
    diff = pred - y
    grad = np.sign(diff).astype(np.float32)
    # Use small constant hessian for stability (standard practice)
    hess = np.ones_like(pred, dtype=np.float32) * 1.0
    return grad, hess


def _ensure_mae_kernel():
    global _mae_gradient_kernel
    if _mae_gradient_kernel is None:
        _mae_gradient_kernel = _get_mae_kernel()
    return _mae_gradient_kernel


def _mae_gradient_gpu(pred, y):
    """GPU implementation of MAE gradient."""
    from numba import cuda
    _ensure_mae_kernel()

    if hasattr(pred, 'copy_to_host'):
        n = pred.shape[0]
    else:
        n = len(pred)
        pred = cuda.to_device(np.asarray(pred, dtype=np.float32))
    
    if not hasattr(y, 'copy_to_host'):
        y = cuda.to_device(np.asarray(y, dtype=np.float32))
    
    grad = cuda.device_array(n, dtype=np.float32)
    hess = cuda.device_array(n, dtype=np.float32)
    
    threads = 256
    blocks = (n + threads - 1) // threads
    _mae_gradient_kernel[blocks, threads](pred, y, grad, hess, n)
    
    return grad, hess


def _get_mae_kernel():
    """Lazily compile MAE gradient kernel."""
    from numba import cuda
    
    @cuda.jit
    def kernel(pred, y, grad, hess, n):
        idx = cuda.grid(1)
        if idx < n:
            diff = pred[idx] - y[idx]
            if diff > 0:
                grad[idx] = 1.0
            elif diff < 0:
                grad[idx] = -1.0
            else:
                grad[idx] = 0.0
            hess[idx] = 1.0
    
    return kernel


_mae_gradient_kernel = None

if is_cuda():
    try:
        _mae_gradient_kernel = _get_mae_kernel()
    except Exception:
        warnings.warn("Failed to compile MAE CUDA kernel; will retry on first use", stacklevel=1)


# =============================================================================
# Quantile Loss (Pinball Loss) - Phase 9.1
# =============================================================================

def quantile_gradient(pred: NDArray, y: NDArray, alpha: float = 0.5) -> tuple[NDArray, NDArray]:
    """Compute Quantile (Pinball) loss gradient and hessian.
    
    Loss: L = alpha * max(y - pred, 0) + (1 - alpha) * max(pred - y, 0)
    
    This is the standard quantile regression loss:
    - alpha=0.5: Median regression (equivalent to MAE)
    - alpha=0.9: 90th percentile
    - alpha=0.1: 10th percentile
    
    Gradient:
        alpha - 1  if pred > y  (over-prediction)
        alpha      if pred < y  (under-prediction)
        
    Hessian: Use constant (not twice-differentiable)
    
    Args:
        pred: Predictions
        y: Targets
        alpha: Quantile level (0 < alpha < 1)
    """
    if is_cuda():
        return _quantile_gradient_gpu(pred, y, alpha)
    return _quantile_gradient_cpu(pred, y, alpha)


def _quantile_gradient_cpu(pred: NDArray, y: NDArray, alpha: float = 0.5) -> tuple[NDArray, NDArray]:
    """CPU implementation of Quantile gradient.
    
    Quantile loss: L = alpha * max(y - pred, 0) + (1 - alpha) * max(pred - y, 0)
    
    Gradient:
        dL/dpred = (1 - alpha)  if pred > y  (over-prediction)
        dL/dpred = -alpha       if pred < y  (under-prediction)
    """
    diff = pred - y
    # Gradient: (1 - alpha) if pred > y, -alpha if pred <= y
    grad = np.where(diff > 0, 1.0 - alpha, -alpha).astype(np.float32)
    # Use constant hessian
    hess = np.ones_like(pred, dtype=np.float32)
    return grad, hess


def _ensure_quantile_kernel():
    global _quantile_gradient_kernel
    if _quantile_gradient_kernel is None:
        _quantile_gradient_kernel = _get_quantile_kernel()
    return _quantile_gradient_kernel


def _quantile_gradient_gpu(pred, y, alpha: float = 0.5):
    """GPU implementation of Quantile gradient."""
    from numba import cuda
    _ensure_quantile_kernel()

    if hasattr(pred, 'copy_to_host'):
        n = pred.shape[0]
    else:
        n = len(pred)
        pred = cuda.to_device(np.asarray(pred, dtype=np.float32))
    
    if not hasattr(y, 'copy_to_host'):
        y = cuda.to_device(np.asarray(y, dtype=np.float32))
    
    grad = cuda.device_array(n, dtype=np.float32)
    hess = cuda.device_array(n, dtype=np.float32)
    
    threads = 256
    blocks = (n + threads - 1) // threads
    _quantile_gradient_kernel[blocks, threads](pred, y, grad, hess, n, alpha)
    
    return grad, hess


def _get_quantile_kernel():
    """Lazily compile Quantile gradient kernel."""
    from numba import cuda
    
    @cuda.jit
    def kernel(pred, y, grad, hess, n, alpha):
        idx = cuda.grid(1)
        if idx < n:
            diff = pred[idx] - y[idx]
            if diff > 0:
                grad[idx] = 1.0 - alpha  # Over-prediction
            else:
                grad[idx] = -alpha  # Under-prediction
            hess[idx] = 1.0
    
    return kernel


_quantile_gradient_kernel = None

if is_cuda():
    try:
        _quantile_gradient_kernel = _get_quantile_kernel()
    except Exception:
        warnings.warn("Failed to compile Quantile CUDA kernel; will retry on first use", stacklevel=1)


# =============================================================================
# Poisson Loss (Count Data) - Phase 9.3
# =============================================================================

def poisson_gradient(pred: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """Compute Poisson deviance gradient and hessian.
    
    For count data (clicks, purchases, etc.). Predictions are in log-space.
    
    Loss: L = exp(pred) - y * pred  (negative log-likelihood)
    Gradient: dL/dpred = exp(pred) - y
    Hessian: d²L/dpred² = exp(pred)
    
    Note: y must be non-negative integers (counts).
    """
    if is_cuda():
        return _poisson_gradient_gpu(pred, y)
    return _poisson_gradient_cpu(pred, y)


def _poisson_gradient_cpu(pred: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """CPU implementation of Poisson gradient."""
    exp_pred = np.exp(np.clip(pred, -20, 20))  # Clip for numerical stability
    grad = (exp_pred - y).astype(np.float32)
    hess = np.maximum(exp_pred, 1e-6).astype(np.float32)  # Hessian = exp(pred)
    return grad, hess


def _ensure_poisson_kernel():
    global _poisson_gradient_kernel
    if _poisson_gradient_kernel is None:
        _poisson_gradient_kernel = _get_poisson_kernel()
    return _poisson_gradient_kernel


def _poisson_gradient_gpu(pred, y):
    """GPU implementation of Poisson gradient."""
    from numba import cuda
    _ensure_poisson_kernel()

    if hasattr(pred, 'copy_to_host'):
        n = pred.shape[0]
    else:
        n = len(pred)
        pred = cuda.to_device(np.asarray(pred, dtype=np.float32))
    
    if not hasattr(y, 'copy_to_host'):
        y = cuda.to_device(np.asarray(y, dtype=np.float32))
    
    grad = cuda.device_array(n, dtype=np.float32)
    hess = cuda.device_array(n, dtype=np.float32)
    
    threads = 256
    blocks = (n + threads - 1) // threads
    _poisson_gradient_kernel[blocks, threads](pred, y, grad, hess, n)
    
    return grad, hess


def _get_poisson_kernel():
    """Lazily compile Poisson gradient kernel."""
    import math

    from numba import cuda
    
    @cuda.jit
    def kernel(pred, y, grad, hess, n):
        idx = cuda.grid(1)
        if idx < n:
            # Clip for stability
            p = pred[idx]
            if p > 20:
                p = 20.0
            elif p < -20:
                p = -20.0
            exp_p = math.exp(p)
            grad[idx] = exp_p - y[idx]
            hess[idx] = max(exp_p, 1e-6)
    
    return kernel


_poisson_gradient_kernel = None

if is_cuda():
    try:
        _poisson_gradient_kernel = _get_poisson_kernel()
    except Exception:
        warnings.warn("Failed to compile Poisson CUDA kernel; will retry on first use", stacklevel=1)


# =============================================================================
# Gamma Loss (Positive Continuous) - Phase 9.3
# =============================================================================

def gamma_gradient(pred: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """Compute Gamma deviance gradient and hessian.
    
    For positive continuous data (insurance claims, etc.). Predictions are in log-space.
    
    Loss: L = y * exp(-pred) + pred  (negative log-likelihood, ignoring constants)
    Gradient: dL/dpred = 1 - y * exp(-pred)
    Hessian: d²L/dpred² = y * exp(-pred)
    
    Note: y must be strictly positive.
    """
    if is_cuda():
        return _gamma_gradient_gpu(pred, y)
    return _gamma_gradient_cpu(pred, y)


def _gamma_gradient_cpu(pred: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """CPU implementation of Gamma gradient."""
    exp_neg_pred = np.exp(np.clip(-pred, -20, 20))
    grad = (1.0 - y * exp_neg_pred).astype(np.float32)
    hess = np.maximum(y * exp_neg_pred, 1e-6).astype(np.float32)
    return grad, hess


def _ensure_gamma_kernel():
    global _gamma_gradient_kernel
    if _gamma_gradient_kernel is None:
        _gamma_gradient_kernel = _get_gamma_kernel()
    return _gamma_gradient_kernel


def _gamma_gradient_gpu(pred, y):
    """GPU implementation of Gamma gradient."""
    from numba import cuda
    _ensure_gamma_kernel()

    if hasattr(pred, 'copy_to_host'):
        n = pred.shape[0]
    else:
        n = len(pred)
        pred = cuda.to_device(np.asarray(pred, dtype=np.float32))
    
    if not hasattr(y, 'copy_to_host'):
        y = cuda.to_device(np.asarray(y, dtype=np.float32))
    
    grad = cuda.device_array(n, dtype=np.float32)
    hess = cuda.device_array(n, dtype=np.float32)
    
    threads = 256
    blocks = (n + threads - 1) // threads
    _gamma_gradient_kernel[blocks, threads](pred, y, grad, hess, n)
    
    return grad, hess


def _get_gamma_kernel():
    """Lazily compile Gamma gradient kernel."""
    import math

    from numba import cuda
    
    @cuda.jit
    def kernel(pred, y, grad, hess, n):
        idx = cuda.grid(1)
        if idx < n:
            neg_p = -pred[idx]
            if neg_p > 20:
                neg_p = 20.0
            elif neg_p < -20:
                neg_p = -20.0
            exp_neg_p = math.exp(neg_p)
            y_exp = y[idx] * exp_neg_p
            grad[idx] = 1.0 - y_exp
            hess[idx] = max(y_exp, 1e-6)
    
    return kernel


_gamma_gradient_kernel = None

if is_cuda():
    try:
        _gamma_gradient_kernel = _get_gamma_kernel()
    except Exception:
        warnings.warn("Failed to compile Gamma CUDA kernel; will retry on first use", stacklevel=1)


# =============================================================================
# Tweedie Loss (Compound Poisson-Gamma) - Phase 9.3
# =============================================================================

def tweedie_gradient(pred: NDArray, y: NDArray, rho: float = 1.5) -> tuple[NDArray, NDArray]:
    """Compute Tweedie deviance gradient and hessian.
    
    Tweedie distribution interpolates between Poisson (rho=1) and Gamma (rho=2).
    Commonly used for insurance claims with many zeros.
    
    For rho in (1, 2), predictions are in log-space:
    Loss: L = -y * exp(pred * (1-rho)) / (1-rho) + exp(pred * (2-rho)) / (2-rho)
    
    Args:
        pred: Predictions (in log-space)
        y: Targets (non-negative, can have zeros)
        rho: Variance power (1 < rho < 2 for compound Poisson-Gamma)
    
    Note: rho=1.5 is a common default for insurance data.
    """
    if is_cuda():
        return _tweedie_gradient_gpu(pred, y, rho)
    return _tweedie_gradient_cpu(pred, y, rho)


def _tweedie_gradient_cpu(pred: NDArray, y: NDArray, rho: float = 1.5) -> tuple[NDArray, NDArray]:
    """CPU implementation of Tweedie gradient."""
    # Clip predictions for numerical stability
    pred_clipped = np.clip(pred, -20, 20)
    
    # mu = exp(pred)
    mu = np.exp(pred_clipped)
    
    # Gradient: mu^(1-rho) * (mu - y) = exp(pred*(2-rho)) - y*exp(pred*(1-rho))
    grad = (np.power(mu, 2 - rho) - y * np.power(mu, 1 - rho)).astype(np.float32)
    
    # Hessian: (2-rho) * mu^(2-rho)
    hess = np.maximum((2 - rho) * np.power(mu, 2 - rho), 1e-6).astype(np.float32)
    
    return grad, hess


def _ensure_tweedie_kernel():
    global _tweedie_gradient_kernel
    if _tweedie_gradient_kernel is None:
        _tweedie_gradient_kernel = _get_tweedie_kernel()
    return _tweedie_gradient_kernel


def _tweedie_gradient_gpu(pred, y, rho: float = 1.5):
    """GPU implementation of Tweedie gradient."""
    from numba import cuda
    _ensure_tweedie_kernel()

    if hasattr(pred, 'copy_to_host'):
        n = pred.shape[0]
    else:
        n = len(pred)
        pred = cuda.to_device(np.asarray(pred, dtype=np.float32))
    
    if not hasattr(y, 'copy_to_host'):
        y = cuda.to_device(np.asarray(y, dtype=np.float32))
    
    grad = cuda.device_array(n, dtype=np.float32)
    hess = cuda.device_array(n, dtype=np.float32)
    
    threads = 256
    blocks = (n + threads - 1) // threads
    _tweedie_gradient_kernel[blocks, threads](pred, y, grad, hess, n, rho)
    
    return grad, hess


def _get_tweedie_kernel():
    """Lazily compile Tweedie gradient kernel."""
    import math

    from numba import cuda
    
    @cuda.jit
    def kernel(pred, y, grad, hess, n, rho):
        idx = cuda.grid(1)
        if idx < n:
            p = pred[idx]
            if p > 20:
                p = 20.0
            elif p < -20:
                p = -20.0
            
            # mu^(2-rho) and mu^(1-rho) via exp
            mu_2_rho = math.exp(p * (2.0 - rho))
            mu_1_rho = math.exp(p * (1.0 - rho))
            
            grad[idx] = mu_2_rho - y[idx] * mu_1_rho
            hess[idx] = max((2.0 - rho) * mu_2_rho, 1e-6)
    
    return kernel


_tweedie_gradient_kernel = None

if is_cuda():
    try:
        _tweedie_gradient_kernel = _get_tweedie_kernel()
    except Exception:
        warnings.warn("Failed to compile Tweedie CUDA kernel; will retry on first use", stacklevel=1)


# =============================================================================
# Softmax Loss (Multi-class Classification) - Phase 9.2
# =============================================================================

def softmax_gradient(pred: NDArray, y: NDArray, n_classes: int) -> tuple[NDArray, NDArray]:
    """Compute Softmax cross-entropy gradient and hessian for multi-class.
    
    This returns gradients for ALL classes at once. For GBDT, you typically
    train K trees per round (one per class).
    
    Args:
        pred: Predictions, shape (n_samples, n_classes) - raw logits
        y: Labels, shape (n_samples,) - integer class labels (0 to n_classes-1)
        n_classes: Number of classes
        
    Returns:
        grad: Gradients, shape (n_samples, n_classes)
        hess: Hessians, shape (n_samples, n_classes)
        
    Note: For binary classification, use logloss instead (more efficient).
    """
    if is_cuda():
        return _softmax_gradient_gpu(pred, y, n_classes)
    return _softmax_gradient_cpu(pred, y, n_classes)


def _softmax_gradient_cpu(pred: NDArray, y: NDArray, n_classes: int) -> tuple[NDArray, NDArray]:
    """CPU implementation of Softmax gradient."""
    n_samples = pred.shape[0]
    
    # Compute softmax probabilities (with numerical stability)
    pred_max = np.max(pred, axis=1, keepdims=True)
    exp_pred = np.exp(pred - pred_max)
    probs = exp_pred / (np.sum(exp_pred, axis=1, keepdims=True) + 1e-10)
    
    # One-hot encode y
    y_onehot = np.zeros((n_samples, n_classes), dtype=np.float32)
    y_onehot[np.arange(n_samples), y.astype(np.int32)] = 1.0
    
    # Gradient: prob - y_onehot
    grad = (probs - y_onehot).astype(np.float32)
    
    # Hessian: prob * (1 - prob) for diagonal approximation
    hess = (probs * (1 - probs)).astype(np.float32)
    hess = np.maximum(hess, 1e-6)  # Stability
    
    return grad, hess


def _softmax_gradient_gpu(pred, y, n_classes: int):
    """GPU implementation of Softmax gradient."""
    # For simplicity, use CPU implementation and transfer
    # TODO: Implement proper CUDA kernel for large-scale
    if hasattr(pred, 'copy_to_host'):
        pred_cpu = pred.copy_to_host()
    else:
        pred_cpu = np.asarray(pred, dtype=np.float32)

    y_cpu = y.copy_to_host() if hasattr(y, 'copy_to_host') else np.asarray(y)

    return _softmax_gradient_cpu(pred_cpu, y_cpu, n_classes)


# =============================================================================
# Built-in Loss Registry Seed
# =============================================================================
# Parameterized built-ins are stored as factories (marked with
# ``__openboost_factory__``) so that get_loss_function can pass kwargs
# (quantile_alpha, tweedie_rho, huber_delta) through to them.

def _quantile_factory(**kwargs):
    alpha = kwargs.get('quantile_alpha', 0.5)
    return lambda pred, y: quantile_gradient(pred, y, alpha=alpha)


_quantile_factory.__openboost_factory__ = True


def _tweedie_factory(**kwargs):
    rho = kwargs.get('tweedie_rho', 1.5)
    return lambda pred, y: tweedie_gradient(pred, y, rho=rho)


_tweedie_factory.__openboost_factory__ = True


def _huber_factory(**kwargs):
    delta = kwargs.get('huber_delta', kwargs.get('delta', 1.0))
    return lambda pred, y: huber_gradient(pred, y, delta=delta)


_huber_factory.__openboost_factory__ = True


_LOSS_REGISTRY.update({
    'mse': mse_gradient,
    'squared_error': mse_gradient,
    'logloss': logloss_gradient,
    'binary_crossentropy': logloss_gradient,
    'huber': _huber_factory,
    'mae': mae_gradient,
    'l1': mae_gradient,
    'absolute_error': mae_gradient,
    'poisson': poisson_gradient,
    'gamma': gamma_gradient,
    'quantile': _quantile_factory,
    'tweedie': _tweedie_factory,
})
_BUILTIN_LOSS_SEED.update(_LOSS_REGISTRY)

