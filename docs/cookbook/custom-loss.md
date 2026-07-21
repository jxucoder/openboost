# Recipe: Custom Loss with a True Loss Value

Register a custom objective under a string name so it works everywhere a
built-in loss name does — including correct train/validation loss reporting.

**You will use:** `ob.register_loss`, the `loss_value_fn` hook,
`GradientBoosting(loss='<your name>')`.

## The problem

A callable passed as `loss=` only tells OpenBoost the *gradient* and *hessian*.
When training needs a scalar loss (for `Logger`, `EarlyStopping`, or history),
OpenBoost falls back to a second-order Taylor proxy `mean(grad² / (2·hess))` —
which is usually *not* your actual loss. Registering the loss with a
`loss_value_fn` fixes that, and gives the loss a reusable name.

## Complete script

Train with an asymmetric squared error that penalizes under-prediction 4x more
than over-prediction (useful when running out of stock costs more than
overstocking):

```python
import numpy as np
import openboost as ob

# --- Data -------------------------------------------------------------------
rng = np.random.default_rng(0)
X = rng.standard_normal((2000, 8)).astype(np.float32)
y = (2.0 * X[:, 0] + np.sin(2.0 * X[:, 1]) + 0.2 * rng.standard_normal(2000)).astype(np.float32)
X_train, X_val = X[:1600], X[1600:]
y_train, y_val = y[:1600], y[1600:]

# --- Asymmetric squared loss ------------------------------------------------
# L(pred, y) = w * (pred - y)^2,  w = 4 when pred < y (under), else 1
UNDER_WEIGHT = 4.0

def asymmetric_mse(pred, y):
    """Gradient/hessian pair. Always return float32 arrays."""
    residual = pred - y
    w = np.where(residual < 0, UNDER_WEIGHT, 1.0)
    grad = (2.0 * w * residual).astype(np.float32)
    hess = (2.0 * w).astype(np.float32)
    return grad, hess

def asymmetric_mse_value(pred, y):
    """The TRUE scalar loss, reported during training instead of the proxy."""
    residual = pred - y
    w = np.where(residual < 0, UNDER_WEIGHT, 1.0)
    return float(np.mean(w * residual**2))

# --- Register and train by name --------------------------------------------
ob.register_loss('asymmetric_mse', asymmetric_mse, loss_value_fn=asymmetric_mse_value)

history = ob.HistoryCallback()
model = ob.GradientBoosting(n_trees=200, max_depth=4, learning_rate=0.1, loss='asymmetric_mse')
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[history])

# history now contains the true asymmetric loss, not the Taylor proxy
print(f"final train loss: {history.history['train_loss'][-1]:.4f}")
print(f"final val loss:   {history.history['val_loss'][-1]:.4f}")

# The asymmetric penalty pushes predictions upward: under-prediction gets rarer
frac_under = float(np.mean(model.predict(X_val) < y_val))
baseline = ob.GradientBoosting(n_trees=200, max_depth=4, learning_rate=0.1, loss='mse')
baseline.fit(X_train, y_train)
frac_under_mse = float(np.mean(baseline.predict(X_val) < y_val))
print(f"under-predicted: {frac_under:.1%} (asymmetric) vs {frac_under_mse:.1%} (mse)")
assert frac_under < frac_under_mse
```

## How the reported loss value is chosen

For a loss used by name, the train/val loss reported to callbacks follows this
precedence:

1. `loss_value_fn` registered via `ob.register_loss` for that name.
2. A `loss_value` attribute on the loss callable itself
   (`fn.loss_value = lambda pred, y: ...`).
3. The built-in formula, for non-overridden built-in names.
4. The second-order Taylor proxy `mean(grad² / (2·hess))`.

The attribute hook (2) is handy when you want the callable to carry its own
value function:

```python
def pinball90(pred, y):
    """Quantile (pinball) loss for the 90th percentile."""
    residual = y - pred
    grad = np.where(residual > 0, -0.9, 0.1).astype(np.float32)
    hess = np.ones_like(pred, dtype=np.float32)
    return grad, hess

pinball90.loss_value = lambda pred, y: float(
    np.mean(np.where(y - pred >= 0, 0.9 * (y - pred), -0.1 * (y - pred)))
)
ob.register_loss('pinball90', pinball90)

q_model = ob.GradientBoosting(n_trees=150, max_depth=4, loss='pinball90')
q_model.fit(X_train, y_train)
frac_below = float(np.mean(y_val <= q_model.predict(X_val)))
print(f"targets at or below the predicted 90th percentile: {frac_below:.1%}")
```

## Notes

- **Duplicate names raise.** `register_loss` refuses to silently replace an
  existing name (including built-ins like `'mse'`). Pass `override=True` to
  replace one deliberately.
- **Registration is process-wide** and lives for the lifetime of the Python
  process; call `register_loss` at import time of your own module.
- **float32 in, float32 out.** Gradients and hessians should be float32 and the
  hessian must stay positive (clip with `np.maximum(hess, 1e-6)` if needed).
- **Running on GPU?** A registered custom loss works on the CUDA backend too,
  at the cost of a host round-trip per round. See
  [Device-Native Loss](device-loss.md) to eliminate it.
