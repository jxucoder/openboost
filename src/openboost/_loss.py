"""Loss functions and GPU gradient computation for OpenBoost.

Provides efficient GPU kernels for computing gradients and hessians
of common loss functions, enabling fully batched training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from ._backends import is_cuda

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Type alias for loss functions
LossFunction = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def get_loss_function(loss: str | LossFunction) -> LossFunction:
    """Get a loss function by name or return custom callable.
    
    Args:
        loss: Either a string name ('mse', 'logloss', 'huber') or a callable
              that takes (pred, y) and returns (grad, hess).
              
    Returns:
        Loss function callable.
    """
    if callable(loss):
        return loss
    
    loss_map = {
        'mse': mse_gradient,
        'squared_error': mse_gradient,
        'logloss': logloss_gradient,
        'binary_crossentropy': logloss_gradient,
        'huber': huber_gradient,
    }
    
    if loss not in loss_map:
        available = ', '.join(loss_map.keys())
        raise ValueError(f"Unknown loss '{loss}'. Available: {available}")
    
    return loss_map[loss]


# =============================================================================
# MSE Loss (Regression)
# =============================================================================

def mse_gradient(pred: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """Compute MSE gradient and hessian.
    
    Loss: L = (pred - y)^2
    Gradient: dL/dpred = 2 * (pred - y)
    Hessian: d²L/dpred² = 2
    """
    if is_cuda():
        return _mse_gradient_gpu(pred, y)
    return _mse_gradient_cpu(pred, y)


def _mse_gradient_cpu(pred: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """CPU implementation of MSE gradient."""
    grad = 2.0 * (pred - y)
    hess = np.full_like(pred, 2.0, dtype=np.float32)
    return grad.astype(np.float32), hess


def _mse_gradient_gpu(pred, y):
    """GPU implementation of MSE gradient."""
    from numba import cuda
    
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
            grad[idx] = 2.0 * (pred[idx] - y[idx])
            hess[idx] = 2.0
    
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
        pass  # Will be compiled on first use


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


def _logloss_gradient_gpu(pred, y):
    """GPU implementation of LogLoss gradient."""
    from numba import cuda
    
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
    from numba import cuda
    import math
    
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
        pass


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


def _huber_gradient_gpu(pred, y, delta: float = 1.0):
    """GPU implementation of Huber gradient."""
    from numba import cuda
    
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
        pass

