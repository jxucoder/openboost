# Custom Distributions

Define your own probability distribution for NaturalBoost.

## Using `create_custom_distribution`

```python
import openboost as ob
import numpy as np

# Define a custom distribution
MyDist = ob.create_custom_distribution(
    name='MyDist',
    param_names=['loc', 'scale'],
    nll_fn=lambda params, y: (
        0.5 * np.log(2 * np.pi * params['scale']**2) +
        0.5 * ((y - params['loc']) / params['scale'])**2
    ),
    mean_fn=lambda params: params['loc'],
    param_transforms={'scale': 'softplus'},  # Ensure scale > 0
)

# Use with NaturalBoost
model = ob.NaturalBoost(distribution=MyDist(), n_trees=100)
model.fit(X_train, y_train)
```

## Parameter Transforms

Transforms ensure parameters stay in valid ranges:

| Transform | Range | Use For |
|-----------|-------|---------|
| `'softplus'` | (0, ∞) | Scale, variance |
| `'exp'` | (0, ∞) | Rate parameters |
| `'sigmoid'` | (0, 1) | Probabilities |
| `None` | (-∞, ∞) | Location parameters |

## Full Example: Laplace Distribution

```python
import openboost as ob
import numpy as np

def laplace_nll(params, y):
    """Negative log-likelihood for Laplace distribution."""
    loc = params['loc']
    scale = params['scale']
    return np.log(2 * scale) + np.abs(y - loc) / scale

def laplace_mean(params):
    return params['loc']

def laplace_std(params):
    return params['scale'] * np.sqrt(2)

LaplaceDist = ob.create_custom_distribution(
    name='Laplace',
    param_names=['loc', 'scale'],
    nll_fn=laplace_nll,
    mean_fn=laplace_mean,
    std_fn=laplace_std,
    param_transforms={'scale': 'softplus'},
)

# Train
model = ob.NaturalBoost(distribution=LaplaceDist(), n_trees=100)
model.fit(X_train, y_train)

# Predict
mean = model.predict(X_test)
lower, upper = model.predict_interval(X_test, alpha=0.1)
```

## With JAX Autodiff

For automatic gradient computation:

```python
import jax.numpy as jnp

def custom_nll_jax(params, y):
    """JAX-compatible NLL for autodiff."""
    loc = params['loc']
    scale = params['scale']
    return 0.5 * jnp.log(2 * jnp.pi * scale**2) + 0.5 * ((y - loc) / scale)**2

# Gradients computed automatically via JAX
MyDist = ob.create_custom_distribution(
    name='MyDist',
    param_names=['loc', 'scale'],
    nll_fn=custom_nll_jax,
    autodiff='jax',  # Use JAX for gradients
)
```

## Available Built-in Distributions

```python
import openboost as ob

# List all available distributions
print(ob.list_distributions())

# Get a distribution by name
normal = ob.get_distribution('normal')
gamma = ob.get_distribution('gamma')
```
