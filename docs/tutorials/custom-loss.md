# Custom Loss Functions

OpenBoost lets you define any loss function in Python. No C++, no recompilation.

## How Gradient Boosting Works

Gradient boosting minimizes a loss function by iteratively fitting trees to negative gradients. For each iteration:

1. Compute gradient: `grad = ∂L/∂pred`
2. Compute Hessian: `hess = ∂²L/∂pred²`
3. Fit tree to `(grad, hess)` weighted samples
4. Update predictions: `pred += learning_rate * tree(X)`

To use a custom loss, you just need to provide the gradient and Hessian.

## Basic Custom Loss

A custom loss function takes predictions and targets, returns gradients and Hessians:

```python
import numpy as np
import openboost as ob

def my_custom_loss(pred, y):
    """Custom loss function.
    
    Args:
        pred: Current predictions, shape (n_samples,)
        y: True targets, shape (n_samples,)
        
    Returns:
        grad: Gradient of loss w.r.t. predictions, shape (n_samples,)
        hess: Hessian (second derivative), shape (n_samples,)
    """
    # Example: Mean Squared Error
    # L = 0.5 * (pred - y)²
    # grad = pred - y
    # hess = 1
    
    grad = (pred - y).astype(np.float32)
    hess = np.ones_like(pred, dtype=np.float32)
    
    return grad, hess

# Use with GradientBoosting
model = ob.GradientBoosting(n_trees=100, loss=my_custom_loss)
model.fit(X_train, y_train)
```

## Example: Asymmetric Loss

Penalize under-predictions more than over-predictions:

```python
def asymmetric_loss(pred, y, alpha=0.7):
    """Asymmetric loss: heavier penalty for under-prediction.
    
    L = alpha * |error| if error > 0 (under-prediction)
        (1-alpha) * |error| if error < 0 (over-prediction)
    
    Args:
        alpha: Weight for under-prediction penalty (0.5 = symmetric)
    """
    error = y - pred
    
    # Gradient
    grad = np.where(error > 0, -alpha, 1 - alpha).astype(np.float32)
    
    # Hessian (constant for linear loss)
    hess = np.ones_like(pred, dtype=np.float32)
    
    return grad, hess

# Conservative model (prefers over-prediction)
model = ob.GradientBoosting(
    n_trees=100,
    loss=lambda p, y: asymmetric_loss(p, y, alpha=0.8),
)
model.fit(X_train, y_train)
```

## Example: Quantile Regression

Predict any quantile (not just the mean):

```python
def quantile_loss(pred, y, tau=0.5):
    """Quantile loss for any percentile.
    
    tau=0.5: Median (robust to outliers)
    tau=0.9: 90th percentile
    tau=0.1: 10th percentile
    """
    error = y - pred
    
    grad = np.where(error > 0, -tau, 1 - tau).astype(np.float32)
    hess = np.ones_like(pred, dtype=np.float32)
    
    return grad, hess

# Predict 90th percentile
model = ob.GradientBoosting(
    n_trees=100,
    loss=lambda p, y: quantile_loss(p, y, tau=0.9),
)
model.fit(X_train, y_train)
```

Note: OpenBoost also has built-in quantile loss via `loss='quantile'` and `quantile_alpha=0.9`.

## Example: Huber Loss

Robust to outliers (L2 near zero, L1 far from zero):

```python
def huber_loss(pred, y, delta=1.0):
    """Huber loss: smooth transition between L2 and L1.
    
    L = 0.5 * error² if |error| < delta
        delta * (|error| - 0.5*delta) if |error| >= delta
    """
    error = pred - y
    abs_error = np.abs(error)
    
    # Gradient
    grad = np.where(
        abs_error < delta,
        error,  # L2 region: gradient = error
        delta * np.sign(error)  # L1 region: gradient = ±delta
    ).astype(np.float32)
    
    # Hessian
    hess = np.where(
        abs_error < delta,
        1.0,  # L2 region: constant Hessian
        1e-6,  # L1 region: small constant for stability
    ).astype(np.float32)
    
    return grad, hess

model = ob.GradientBoosting(n_trees=100, loss=huber_loss)
```

## Example: Focal Loss (Classification)

For imbalanced classification, down-weight easy examples:

```python
def focal_loss(pred, y, gamma=2.0):
    """Focal loss for imbalanced classification.
    
    Focuses learning on hard misclassified examples.
    gamma=0: Standard log loss
    gamma=2: Strong focus on hard examples
    """
    # Sigmoid to get probabilities
    p = 1 / (1 + np.exp(-pred))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    
    # Focal weight: (1-p_t)^gamma
    p_t = np.where(y == 1, p, 1 - p)
    focal_weight = (1 - p_t) ** gamma
    
    # Gradient (includes focal weight)
    grad = focal_weight * (p - y)
    
    # Hessian (approximation)
    hess = np.maximum(
        focal_weight * p * (1 - p),
        1e-6
    ).astype(np.float32)
    
    return grad.astype(np.float32), hess

model = ob.GradientBoosting(
    n_trees=100,
    loss=lambda p, y: focal_loss(p, y, gamma=2.0),
)
model.fit(X_train, y_train)
```

## Example: Log-Cosh Loss

Smooth approximation to MAE:

```python
def log_cosh_loss(pred, y):
    """Log-cosh loss: smooth approximation to L1.
    
    L = log(cosh(error))
    Behaves like L2 for small errors, L1 for large errors.
    """
    error = pred - y
    
    # Gradient: tanh(error)
    grad = np.tanh(error).astype(np.float32)
    
    # Hessian: sech²(error) = 1 - tanh²(error)
    hess = (1 - np.tanh(error) ** 2).astype(np.float32)
    hess = np.maximum(hess, 1e-6)  # Numerical stability
    
    return grad, hess

model = ob.GradientBoosting(n_trees=100, loss=log_cosh_loss)
```

## Low-Level API: Full Control

For complete control over the training loop:

```python
import openboost as ob
import numpy as np

# Bin data once
X_binned = ob.array(X_train)

# Initialize predictions
pred = np.zeros(len(y_train), dtype=np.float32)
trees = []
learning_rate = 0.1

for i in range(100):
    # YOUR loss function
    error = pred - y_train
    grad = error  # MSE gradient
    hess = np.ones_like(grad)
    
    # Fit tree to gradients
    tree = ob.fit_tree(
        X_binned, grad, hess,
        max_depth=6,
        min_child_weight=1.0,
        reg_lambda=1.0,
    )
    
    # Update predictions
    pred = pred + learning_rate * tree(X_binned)
    trees.append(tree)
    
    # YOUR early stopping logic
    if i % 10 == 0:
        loss = 0.5 * np.mean(error ** 2)
        print(f"Round {i}: Loss = {loss:.4f}")
```

## Using with PyTorch/JAX

Get gradients from your deep learning framework:

```python
import torch
import openboost as ob

# Your custom PyTorch loss
def my_torch_loss(pred, y):
    return torch.mean((pred - y) ** 2)

# Training loop
X_binned = ob.array(X_train)
pred = torch.zeros(len(y_train), requires_grad=True)
y = torch.from_numpy(y_train)

for i in range(100):
    # Compute gradients with PyTorch
    loss = my_torch_loss(pred, y)
    grad = torch.autograd.grad(loss, pred, create_graph=True)[0]
    
    # For Hessian, use autograd again or approximate
    hess = torch.ones_like(grad)  # Approximation
    
    # Convert to numpy for OpenBoost
    grad_np = grad.detach().numpy().astype(np.float32)
    hess_np = hess.detach().numpy().astype(np.float32)
    
    # Fit tree
    tree = ob.fit_tree(X_binned, grad_np, hess_np, max_depth=6)
    
    # Update predictions
    tree_pred = torch.from_numpy(tree(X_binned))
    pred = pred + 0.1 * tree_pred
```

## Tips for Custom Losses

### 1. Ensure Numerical Stability

```python
def stable_loss(pred, y):
    # Clip values to avoid overflow
    pred = np.clip(pred, -100, 100)
    
    # Avoid division by zero in Hessian
    hess = np.maximum(hess, 1e-6)
    
    return grad.astype(np.float32), hess.astype(np.float32)
```

### 2. Always Return float32

OpenBoost uses float32 internally:

```python
return grad.astype(np.float32), hess.astype(np.float32)
```

### 3. Hessian Must Be Positive

The Hessian should always be positive for the optimization to work properly:

```python
hess = np.maximum(hess, 1e-6)
```

### 4. Test Your Loss

Verify gradients numerically:

```python
def check_gradient(loss_fn, pred, y, eps=1e-5):
    """Check gradient with finite differences."""
    grad, _ = loss_fn(pred, y)
    
    # Numerical gradient
    numerical_grad = np.zeros_like(pred)
    for i in range(len(pred)):
        pred_plus = pred.copy()
        pred_plus[i] += eps
        pred_minus = pred.copy()
        pred_minus[i] -= eps
        
        loss_plus = np.mean((pred_plus - y) ** 2)  # Your loss
        loss_minus = np.mean((pred_minus - y) ** 2)
        numerical_grad[i] = (loss_plus - loss_minus) / (2 * eps)
    
    print(f"Max gradient error: {np.max(np.abs(grad - numerical_grad))}")
```

## Built-in Loss Functions

OpenBoost includes these losses out of the box:

| Loss | String | Use Case |
|------|--------|----------|
| MSE | `'mse'` | Regression |
| MAE | `'mae'` | Robust regression |
| Huber | `'huber'` | Outlier-robust |
| Quantile | `'quantile'` | Quantile regression |
| LogLoss | `'logloss'` | Binary classification |
| Softmax | `'softmax'` | Multi-class (use MultiClassGradientBoosting) |
| Poisson | `'poisson'` | Count data |
| Gamma | `'gamma'` | Positive continuous |
| Tweedie | `'tweedie'` | Zero-inflated positive |

```python
# Using built-in losses
model = ob.GradientBoosting(n_trees=100, loss='huber')
model = ob.GradientBoosting(n_trees=100, loss='quantile', quantile_alpha=0.9)
model = ob.GradientBoosting(n_trees=100, loss='tweedie', tweedie_rho=1.5)
```

## Next Steps

- [Uncertainty Quantification](uncertainty.md) - Probabilistic predictions
- [Migration from XGBoost](../migration/from-xgboost.md) - Switching from XGBoost
