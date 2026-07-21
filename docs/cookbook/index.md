# Extending OpenBoost

OpenBoost is all-Python, so every extension point is a plain Python object plus
a registry call — no C++ plugins, no recompilation. Three registries make
custom components usable **by name**, exactly like the built-ins, and one
decorator opts a custom loss into GPU-native execution:

| Extension point | Register with | Then use as |
|---|---|---|
| Loss function | `ob.register_loss(name, fn, loss_value_fn=...)` | `GradientBoosting(loss='name')` |
| Distribution | `ob.register_distribution(name, cls)` | `NaturalBoost(distribution='name')` |
| Growth strategy | `ob.register_growth_strategy(name, cls)` | `GradientBoosting(growth='name')` |
| Device-native loss | `@ob.device_loss` (marker, not a registry) | `GradientBoosting(loss=fn)` on CUDA |

Shared registry rules:

- **Names, not instances**: `register_distribution` and
  `register_growth_strategy` take a *class* that must construct with no
  arguments; it is instantiated fresh each time the name is resolved. Names
  are stored lowercased (case-insensitive lookup).
- **No silent replacement**: registering an existing name — including a
  built-in like `'mse'` or `'levelwise'` — raises `ValueError` unless you pass
  `override=True`.
- **Process-wide and import-time**: registrations live for the lifetime of the
  Python process. Put them at import time of your own module so saved models
  that reference the name can be loaded and used anywhere the module is
  imported.

## The recipes

Each recipe is a complete, copy-pasteable script (they are executed in CI, so
they stay runnable):

- **[Custom Loss](custom-loss.md)** — an asymmetric objective registered by
  name, with a `loss_value_fn` so logging and early stopping report the true
  loss instead of a Taylor proxy.
- **[Custom Distribution](custom-distribution.md)** — a Gumbel distribution
  for NaturalBoost from just its NLL (autodiff or numerical gradients),
  registered so `distribution='gumbel'` works.
- **[Custom Growth Strategy](custom-growth-strategy.md)** — depth-capped
  random-feature growth in ~20 lines, plus the full `GrowthStrategy` contract
  for from-scratch strategies.
- **[Device-Native Loss (GPU)](device-loss.md)** — the `@ob.device_loss`
  contract for computing gradients entirely on the GPU (honest note: it needs
  CUDA to show any benefit; on CPU it is a no-op).

## Related pages

- [Custom Loss Functions tutorial](../tutorials/custom-loss.md) — gradient/
  hessian derivations for many classic losses.
- [Custom Distributions guide](../user-guide/naturalboost/custom-distributions.md)
  — link functions, `init_fn`, JAX vs numerical gradients.
- [Callbacks](../user-guide/training/callbacks.md) — the training hooks that
  registered losses report into.
