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
    
    # Handle parameterized losses
    if loss == 'quantile':
        alpha = kwargs.get('quantile_alpha', 0.5)
        return lambda pred, y: quantile_gradient(pred, y, alpha=alpha)
    
    if loss == 'tweedie':
        rho = kwargs.get('tweedie_rho', 1.5)
        return lambda pred, y: tweedie_gradient(pred, y, rho=rho)
    
    loss_map = {
        'mse': mse_gradient,
        'squared_error': mse_gradient,
        'logloss': logloss_gradient,
        'binary_crossentropy': logloss_gradient,
        'huber': huber_gradient,
        'mae': mae_gradient,
        'l1': mae_gradient,
        'absolute_error': mae_gradient,
        'poisson': poisson_gradient,
        'gamma': gamma_gradient,
    }
    
    if loss not in loss_map:
        available = ', '.join(sorted(set(loss_map.keys()) | {'quantile', 'tweedie'}))
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


def _mae_gradient_gpu(pred, y):
    """GPU implementation of MAE gradient."""
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
        pass


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
        alpha - 1  if pred > y  (under-prediction)
        alpha      if pred < y  (over-prediction)
        
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


def _quantile_gradient_gpu(pred, y, alpha: float = 0.5):
    """GPU implementation of Quantile gradient."""
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
        pass


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


def _poisson_gradient_gpu(pred, y):
    """GPU implementation of Poisson gradient."""
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
    _poisson_gradient_kernel[blocks, threads](pred, y, grad, hess, n)
    
    return grad, hess


def _get_poisson_kernel():
    """Lazily compile Poisson gradient kernel."""
    from numba import cuda
    import math
    
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
        pass


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


def _gamma_gradient_gpu(pred, y):
    """GPU implementation of Gamma gradient."""
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
    _gamma_gradient_kernel[blocks, threads](pred, y, grad, hess, n)
    
    return grad, hess


def _get_gamma_kernel():
    """Lazily compile Gamma gradient kernel."""
    from numba import cuda
    import math
    
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
        pass


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


def _tweedie_gradient_gpu(pred, y, rho: float = 1.5):
    """GPU implementation of Tweedie gradient."""
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
    _tweedie_gradient_kernel[blocks, threads](pred, y, grad, hess, n, rho)
    
    return grad, hess


def _get_tweedie_kernel():
    """Lazily compile Tweedie gradient kernel."""
    from numba import cuda
    import math
    
    @cuda.jit
    def kernel(pred, y, grad, hess, n, rho):
        idx = cuda.grid(1)
        if idx < n:
            p = pred[idx]
            if p > 20:
                p = 20.0
            elif p < -20:
                p = -20.0
            
            mu = math.exp(p)
            
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
        pass


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
    probs = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
    
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
    
    if hasattr(y, 'copy_to_host'):
        y_cpu = y.copy_to_host()
    else:
        y_cpu = np.asarray(y)
    
    return _softmax_gradient_cpu(pred_cpu, y_cpu, n_classes)

