# Recipe: Custom Growth Strategy

Plug your own tree-growth policy into `GradientBoosting` under a string name,
next to the built-in `'levelwise'`, `'leafwise'`, and `'symmetric'` strategies.

**You will use:** `ob.GrowthStrategy` / `ob.LevelWiseGrowth`,
`ob.register_growth_strategy`, `GradientBoosting(growth='<your name>')`.

## Complete script

The easiest custom strategy wraps a built-in one and rewrites its
`GrowthConfig`. Here: **depth-capped random-feature growth** — every tree is
level-wise but capped at depth 3 and sees only a random 70% of the features,
regardless of what the model was configured with. (Think of it as an
extra-randomized, heavily regularized forest layer.)

```python
import dataclasses

import numpy as np
import openboost as ob


class ShallowRandomGrowth(ob.LevelWiseGrowth):
    """Level-wise growth, capped at depth 3, random 70% feature subset per tree."""

    depth_cap = 3
    feature_fraction = 0.7

    def grow(self, binned, grad, hess, config,
             has_missing=None, is_categorical=None, n_categories=None):
        config = dataclasses.replace(
            config,
            max_depth=min(config.max_depth, self.depth_cap),
            colsample_bytree=min(config.colsample_bytree, self.feature_fraction),
        )
        return super().grow(
            binned, grad, hess, config,
            has_missing=has_missing,
            is_categorical=is_categorical,
            n_categories=n_categories,
        )


# Register once (process-wide); the class must construct with no arguments —
# it is instantiated fresh each time the name is resolved.
ob.register_growth_strategy('shallow_random', ShallowRandomGrowth)

# --- Use it by name ----------------------------------------------------------
rng = np.random.default_rng(0)
X = rng.standard_normal((2000, 10)).astype(np.float32)
y = (X[:, 0] - 2.0 * X[:, 1] + X[:, 2] * X[:, 3] + 0.1 * rng.standard_normal(2000))
y = y.astype(np.float32)
X_train, X_test = X[:1600], X[1600:]
y_train, y_test = y[:1600], y[1600:]

model = ob.GradientBoosting(n_trees=100, max_depth=6, growth='shallow_random', random_state=0)
model.fit(X_train, y_train)

# Every tree honored the cap even though the model asked for max_depth=6
depths = {tree.depth for tree in model.trees_}
print("tree depths:", depths)
assert max(depths) <= 3

rmse = float(np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2)))
print(f"test RMSE: {rmse:.4f}")
```

## The `GrowthStrategy` contract

To write a strategy from scratch, subclass `ob.GrowthStrategy` and implement
one method:

```python
class MyGrowth(ob.GrowthStrategy):
    def grow(self, binned, grad, hess, config,
             has_missing=None, is_categorical=None, n_categories=None):
        ...
        return tree  # an ob.TreeStructure
```

- `binned` — binned features, shape `(n_features, n_samples)`, uint8 (bin 255
  is reserved for missing values).
- `grad`, `hess` — float32 arrays of shape `(n_samples,)`.
- `config` — an `ob.GrowthConfig` dataclass carrying `max_depth`,
  `max_leaves`, `min_child_weight`, `reg_lambda`, `reg_alpha`, `min_gain`,
  `subsample`, and `colsample_bytree`.
- `has_missing` / `is_categorical` / `n_categories` — optional per-feature
  metadata arrays; a minimal strategy may ignore them (numeric, fully-observed
  data), but then must not be used with missing values or categoricals.
- Return an `ob.TreeStructure` (routing arrays + leaf values), which handles
  prediction on both CPU and GPU.

The building blocks used by the built-in strategies are exported for reuse:
`ob.build_node_histograms`, `ob.find_node_splits`, `ob.partition_samples`,
`ob.compute_leaf_values`, and `ob.get_nodes_at_depth`. Reading
`LevelWiseGrowth.grow` in `openboost/_core/_growth.py` (~120 lines) is the
best template for a from-scratch implementation.

## Notes

- **Names are case-insensitive** and stored lowercased. Registering an existing
  name (including built-in aliases like `'oblivious'`) raises `ValueError`
  unless you pass `override=True`.
- The name works in `GradientBoosting(growth=...)` and in the low-level
  `ob.fit_tree(X_binned, grad, hess, growth=...)`. You can also bypass the
  registry entirely and pass a strategy *instance*:
  `fit_tree(..., growth=MyGrowth())`.
- Hyperparameters on a registered strategy must be class attributes (the
  registry instantiates with no arguments). If you need per-model
  configuration, register several named variants or pass configured instances
  to `fit_tree`.
