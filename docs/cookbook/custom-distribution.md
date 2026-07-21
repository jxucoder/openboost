# Recipe: Custom Distribution for NaturalBoost

Define a probability distribution that OpenBoost does not ship, train
`NaturalBoost` on it, and register it under a string name so
`NaturalBoost(distribution='<your name>')` just works.

**You will use:** `ob.create_custom_distribution`, `ob.CustomDistribution`,
`ob.register_distribution`, `ob.NaturalBoost`.

## Complete script

The [Gumbel distribution](https://en.wikipedia.org/wiki/Gumbel_distribution)
models block maxima (floods, peak load, worst-case latency). Its NLL is

$$\mathrm{NLL}(y; \mu, \beta) = \log \beta + z + e^{-z}, \qquad z = \frac{y - \mu}{\beta}$$

You only write the NLL — gradients and hessians are derived automatically
(JAX autodiff when `jax` is installed, otherwise exact-chain-rule numerical
differentiation; no extra dependency required).

```python
import numpy as np
import openboost as ob

# --- Data: heteroscedastic Gumbel noise -------------------------------------
rng = np.random.default_rng(0)
n = 3000
X = rng.standard_normal((n, 4)).astype(np.float32)
true_loc = 2.0 * X[:, 0] + 1.0
true_scale = np.exp(0.5 * X[:, 1])
y = (true_loc + rng.gumbel(0.0, true_scale, size=n)).astype(np.float32)
X_train, X_test = X[:2400], X[2400:]
y_train, y_test = y[:2400], y[2400:]

# --- Define the distribution -------------------------------------------------
EULER_GAMMA = 0.5772156649015329

def gumbel_nll(y, params):
    """Per-sample negative log-likelihood. Signature: (y, params) -> array."""
    z = (y - params['loc']) / params['scale']
    return np.log(params['scale']) + z + np.exp(-z)

gumbel = ob.create_custom_distribution(
    param_names=['loc', 'scale'],
    link_functions={'loc': 'identity', 'scale': 'softplus'},  # scale > 0
    nll_fn=gumbel_nll,
    mean_fn=lambda p: p['loc'] + EULER_GAMMA * p['scale'],
    variance_fn=lambda p: (np.pi**2 / 6.0) * p['scale'] ** 2,
)

# --- Train and predict distributions ----------------------------------------
model = ob.NaturalBoost(distribution=gumbel, n_trees=500, max_depth=3, learning_rate=0.1)
model.fit(X_train, y_train)

mean = model.predict(X_test)                              # E[y | x]
lower, upper = model.predict_interval(X_test, alpha=0.2)  # 80% interval
coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
print(f"80% interval coverage: {coverage:.1%}")
assert 0.6 < coverage < 0.95
```

## Registering it under a name

`create_custom_distribution` returns an **instance**, which is convenient for
one-off use. To make the distribution available by name (in `NaturalBoost`,
`DistributionalGBDT`, and the `OpenBoostDistributionalRegressor` sklearn
wrapper), register a **class** that constructs with no arguments — the class is
instantiated fresh each time the name is resolved:

```python
class Gumbel(ob.CustomDistribution):
    """Same distribution, packaged as a no-argument class."""

    def __init__(self):
        super().__init__(
            param_names=['loc', 'scale'],
            link_functions={'loc': 'identity', 'scale': 'softplus'},
            nll_fn=gumbel_nll,
            mean_fn=lambda p: p['loc'] + EULER_GAMMA * p['scale'],
            variance_fn=lambda p: (np.pi**2 / 6.0) * p['scale'] ** 2,
        )

ob.register_distribution('gumbel', Gumbel)

# The name now works like 'normal', 'gamma', 'poisson', ...
model_by_name = ob.NaturalBoost(distribution='gumbel', n_trees=200, max_depth=3, learning_rate=0.2)
model_by_name.fit(X_train, y_train)
print("predictions by name:", model_by_name.predict(X_test)[:3])
print("'gumbel' listed:", 'gumbel' in ob.list_distributions())
```

## Notes

- **Names are case-insensitive** — they are stored lowercased, matching
  `ob.get_distribution`. Duplicate names raise `ValueError` unless you pass
  `override=True`.
- **Fisher information**: custom distributions use an empirical diagonal
  approximation built from per-sample gradients, so natural-gradient training
  works without deriving the Fisher matrix by hand. Built-ins use exact
  closed-form Fisher matrices.
- **Quantiles/sampling**: `CustomDistribution.quantile()` and `.sample()` use a
  Normal approximation from `mean_fn`/`variance_fn`. If you need exact
  quantiles (the Gumbel is skewed), override `quantile()` in your subclass.
- **JAX users**: write `nll_fn` with traceable operations (plain arithmetic and
  `numpy`-style ufuncs; no in-place mutation). Force the numerical path with
  `CustomDistribution(..., use_jax=False)`.
- Background on links, `init_fn`, and the numerical fallback:
  [Custom Distributions guide](../user-guide/naturalboost/custom-distributions.md).
