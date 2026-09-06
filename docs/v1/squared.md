# Complete scalar squared boosting

`openboost.recipes.squared` composes `Squared` geometry, once-weighted row fields,
train-fitted binning, depthwise trees, mapped terms and immutable run transactions.
It returns final accepted state and per-round evidence. It is a CPU correctness
path, with no measured quality or speed parity claim.

```python
import numpy as np
from openboost import NumericData, Problem, RunContext
from openboost.recipes import squared

train_x = NumericData([[0], [1], [2], [3]], [10, 11, 12, 13], ("x",))
valid_x = NumericData([[0.5], [2.5]], [20, 21], ("x",))
train = Problem(train_x, [[-3], [-1], [1], [3]], train_x.row_ids)
valid = Problem(valid_x, [[-2], [2]], valid_x.row_ids)
fit = squared(train, valid, context=RunContext("demo", seed=7),
              rounds=3, learning_rate=0.5, bins=4)
assert fit.state.version == 3
assert fit.steps[-1].loss_after < fit.steps[0].loss_before
prediction = fit.state.best_model.predict(valid_x)
assert prediction.shape == (2, 1) and np.isfinite(prediction).all()
```

The objective uses weighted mean half-square loss and initializes the raw base
with the training weighted mean of `target - offset`. Derivatives include offsets;
raw state never stores observation offsets. Statistics apply original training
weights once. Binning uses training features only. Validation selects an immutable
`best_model`; the final accepted `model` can differ. Both are inference artifacts.

`step="fixed"` commits each finite candidate even if loss rises. With
`step="backtracking"`, at most `max_trials` (1–6) coefficients are tried, halving
`learning_rate` each time. Strict training-loss improvement accepts a trial; full
rejection leaves terms, caches, best model and version unchanged. A learner is
fitted once per round. The initial CPU implementation recomputes ensemble
predictions during trial validation; prediction caching remains future work.
Nonfinite fixed-step arithmetic raises; backtracking rejects an invalid numerical
trial and continues at a smaller coefficient. Structural errors raise immediately.

The defaults expose `max_depth`, `max_leaves`, `reg_lambda`, `min_child_h` and
`split_penalty`. `learner(binned_data, weighted_fields)` can replace the default
learner; configure custom growth in that callable and leave recipe growth options
at defaults. The recipe rejects conflicting nondefault growth options. Low-level
code can instead use `Squared.fields`, `depthwise`, `TreeTerm`, `propose_terms`,
`preview` and `resolve` directly, including explicit output mappings.

This recipe accepts scalar `[N, 1]` targets and raw_width=1. The
[Normal recipe](normal.md) provides two-parameter distributional geometry.
Classification, specialized targets, vector learners, callbacks/early stopping,
CUDA remains a future slice. [Categorical inputs](categorical.md) are supported. Best-first and symmetric
growers can be substituted through the learner argument. Unsupported arguments fail.
The trace retains per-round arrays for correctness inspection, and is not a
memory-efficient large-workload implementation or a training-resume checkpoint.
