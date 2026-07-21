# Recipe: Device-Native Loss (GPU)

Mark a custom loss as device-native so the CUDA backend computes gradients
entirely on the GPU, skipping the per-round host round-trip.

**You will use:** `ob.device_loss`, `GradientBoosting(loss=<callable>)`.

!!! warning "This recipe needs CUDA to show a benefit"
    `ob.device_loss` is a **no-op on the CPU backend** — the script below runs
    anywhere (and is what CI runs), but the round-trip it eliminates only
    exists on the CUDA backend. The honest claim is: same results everywhere,
    faster only on GPU.

## The problem

On the CUDA backend, an *unmarked* custom loss forces a host round-trip every
boosting round: device predictions are copied to the host, your callable runs
on NumPy arrays, and the returned `(grad, hess)` are copied back to the
device. For big datasets that copy can dominate the round.

## The contract

Decorating a loss with `@ob.device_loss` sets `fn.__openboost_device__ = True`
and changes what the CUDA path hands you:

- `pred` is the **device** (CuPy) prediction array — no copy.
- `y` is the training target **already resident on the device** (moved once
  per `fit` and cached).
- You **must return device `(grad, hess)` float32 arrays** with the same
  length as `pred`.
- On the CPU backend the marker is ignored: your callable receives plain NumPy
  arrays like any other custom loss.

## Complete script (runs on CPU; identical code runs on GPU)

Written with operations that both NumPy and CuPy arrays support, so one
function serves both backends:

```python
import numpy as np
import openboost as ob


@ob.device_loss
def device_asymmetric_mse(pred, y):
    """Asymmetric squared error, backend-agnostic.

    CUDA backend: `pred` and `y` are CuPy device arrays; the arithmetic below
    stays on the GPU and returns device arrays. CPU backend: plain NumPy.
    """
    residual = pred - y
    # (residual < 0) * 3.0 + 1.0  ==  4 where under-predicting, else 1,
    # written without np./cp. so it works on either array type.
    w = (residual < 0) * 3.0 + 1.0
    grad = (2.0 * w * residual).astype('float32')
    hess = (2.0 * w).astype('float32')
    return grad, hess


rng = np.random.default_rng(0)
X = rng.standard_normal((1500, 6)).astype(np.float32)
y = (X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.1 * rng.standard_normal(1500)).astype(np.float32)

model = ob.GradientBoosting(n_trees=80, max_depth=4, loss=device_asymmetric_mse)
model.fit(X[:1200], y[:1200])

pred = model.predict(X[1200:])
print(f"marked device-native: {device_asymmetric_mse.__openboost_device__}")
print(f"fraction under-predicted: {float(np.mean(pred < y[1200:])):.1%}")
```

## Explicit CuPy version

If you prefer explicit device code (or need CuPy-only operations), import
`cupy` inside the function. This variant requires a CUDA GPU:

<!-- docs-ci: skip -->
```python
import cupy as cp
import openboost as ob

@ob.device_loss
def gpu_logcosh(pred_dev, y_dev):
    residual = pred_dev - y_dev
    grad = cp.tanh(residual).astype(cp.float32)
    hess = cp.maximum(1.0 - grad * grad, 1e-6).astype(cp.float32)
    return grad, hess

ob.set_backend('cuda')
model = ob.GradientBoosting(n_trees=500, loss=gpu_logcosh)
model.fit(X_train, y_train)  # gradients never leave the GPU
```

## Notes

- **Correctness first**: returning host arrays (or non-float32) from a marked
  loss on the CUDA path violates the contract — if you cannot keep the math on
  the device, simply leave the loss unmarked and accept the round-trip.
- Works with named registration too: `ob.register_loss('gpu_logcosh',
  gpu_logcosh)` keeps the device marker, since the callable itself carries it.
- The built-in string losses (`'mse'`, `'logloss'`, ...) already run
  device-native on CUDA; `device_loss` is only for *custom* callables.
- See [Custom Loss](custom-loss.md) for loss-value reporting hooks that make
  logging/early-stopping show your true loss.
