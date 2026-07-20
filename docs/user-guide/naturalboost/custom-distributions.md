# Custom Distributions

Define your own probability distribution for NaturalBoost.

## Using `create_custom_distribution`

`ob.create_custom_distribution()` returns a ready-to-use `CustomDistribution`
**instance** (not a class). Pass it directly as `distribution=`:

```python
import numpy as np
import openboost as ob

# Gaussian with learned mean and scale
dist = ob.create_custom_distribution(
    param_names=['loc', 'scale'],
    link_functions={'loc': 'identity', 'scale': 'softplus'},  # scale > 0
    nll_fn=lambda y, params: (
        0.5 * np.log(2 * np.pi * params['scale'] ** 2)
        + 0.5 * ((y - params['loc']) / params['scale']) ** 2
    ),
    mean_fn=lambda params: params['loc'],
    variance_fn=lambda params: params['scale'] ** 2,
)

# Use with NaturalBoost
model = ob.NaturalBoost(distribution=dist, n_trees=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

The signature is:

```python
ob.create_custom_distribution(
    param_names,          # list[str]: names of the distribution parameters
    link_functions,       # dict[str, str]: param name -> link type
    nll_fn,               # callable(y, params) -> per-sample NLL array
    mean_fn=None,         # optional callable(params) -> mean prediction
    variance_fn=None,     # optional callable(params) -> variance
)
```

Note the argument order of `nll_fn`: **`y` first, then the params dict**. It
must return the per-sample negative log-likelihood as an array, computed from
the *constrained* (post-link) parameter values.

## Link Functions

Links map the unconstrained raw scores that the trees predict into valid
parameter ranges. Gradients are taken with respect to the raw scores, with
the chain rule through the link applied for you:

| Link | Range | Use For |
|-----------|-------|---------|
| `'identity'` | (-∞, ∞) | Location parameters |
| `'exp'` | (0, ∞) | Rate/scale parameters |
| `'softplus'` | (0, ∞) | Scale, variance (smoother than exp) |
| `'sigmoid'` | (0, 1) | Probabilities |
| `'square'` | [0, ∞) | Non-negative parameters |

## Full Example: Laplace Distribution

```python
import numpy as np
import openboost as ob

def laplace_nll(y, params):
    """Negative log-likelihood for the Laplace distribution."""
    loc = params['loc']
    scale = params['scale']
    return np.log(2 * scale) + np.abs(y - loc) / scale

laplace = ob.create_custom_distribution(
    param_names=['loc', 'scale'],
    link_functions={'loc': 'identity', 'scale': 'softplus'},
    nll_fn=laplace_nll,
    mean_fn=lambda params: params['loc'],
    variance_fn=lambda params: 2.0 * params['scale'] ** 2,
)

# Train
model = ob.NaturalBoost(distribution=laplace, n_trees=100)
model.fit(X_train, y_train)

# Predict
mean = model.predict(X_test)
lower, upper = model.predict_interval(X_test, alpha=0.1)
```

## Gradient Computation: JAX or Numerical

You do not provide gradients — they are computed automatically from `nll_fn`:

- **JAX** (used automatically when `jax` is installed): exact autodiff of the
  NLL composed with the link functions, vectorized with `jax.vmap`.
- **Numerical fallback** (no extra dependency): central finite differences on
  the constrained parameters with the exact chain rule through the link.

For more control, construct `CustomDistribution` directly. It accepts the
same arguments plus `init_fn` (initial raw parameter values from `y`),
`use_jax` (set `False` to force the numerical path), and `eps` (finite
difference step):

```python
import numpy as np
from openboost import CustomDistribution

dist = CustomDistribution(
    param_names=['loc', 'scale'],
    link_functions={'loc': 'identity', 'scale': 'softplus'},
    nll_fn=laplace_nll,
    mean_fn=lambda params: params['loc'],
    init_fn=lambda y: {'loc': float(np.median(y)), 'scale': 0.0},
    use_jax=False,  # force the numerical fallback
)
```

Note: if you use JAX, write `nll_fn` with operations that JAX can trace
(plain arithmetic and `numpy`-style ufuncs work; avoid in-place mutation).

## Fisher Information

NaturalBoost preconditions gradients with the Fisher information matrix.
Custom distributions use an **empirical diagonal approximation** built from
per-sample gradients (`F[j,j] = mean(g_j^2)`), so natural-gradient training
works out of the box without deriving the Fisher matrix by hand. Built-in
distributions use exact closed-form Fisher matrices.

## Available Built-in Distributions

```python
import openboost as ob

# List all available distributions
print(ob.list_distributions())

# Get a distribution by name
normal = ob.get_distribution('normal')
gamma = ob.get_distribution('gamma')
```
