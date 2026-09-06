# Gamma positive-target means

The CPU Gamma recipe accepts strictly positive scalar targets and predicts
exp(raw + offset). It fits a mean with the unit-dispersion objective
target/mean + log(mean); dispersion is not estimated.

```python
import numpy as np
from openboost import NumericData, Problem, RunContext
from openboost.recipes import gamma
from openboost.outputs import positive_mean

x = NumericData([[0], [1], [2], [3]], [0, 1, 2, 3], ("x",))
v = NumericData([[0.5], [2.5]], [10, 11], ("x",))
train = Problem(x, [[0.5], [2], [3], [8]], x.row_ids, weight=[1, 2, 1, 3])
valid = Problem(v, [[1], [4]], v.row_ids)
fit = gamma(train, valid, context=RunContext("gamma-demo", 7), rounds=3)
mean = positive_mean(fit.state.best_model.predict(v))
assert np.isfinite(mean).all() and (mean > 0).all()
```

Gamma.geometry exposes weighted mean loss and unweighted derivatives with
respect to log mean: gradient=1-target/mean, curvature=target/mean.
The shared Newton adapter applies original weights once. Initialization is
log(weighted_mean(target*exp(-offset))), evaluated with log sums. Offsets enter
geometry once and remain outside raw caches.

Fixed/backtracking steps and validation selection use this Gamma objective.
For fixed targets it orders predictions like Gamma deviance, but the returned
loss is not a fitted-dispersion likelihood or a reported deviance statistic.
Zero/negative targets, unknown structure roles and unrepresentable geometry
are rejected. Exposure has no automatic meaning in this recipe.

For severity applications, define eligible positive claims and their selection
rules outside training. Claim-level targets and policy-average targets have
different weight meanings. When predictors/offsets are identical within a policy,
a positive claim average weighted by claim count reproduces the summed
claim-level gradients and curvature. Unit policy weights generally do not.

Generic artifacts retain raw base/trees. Call positive_mean after model.predict,
supplying inference offsets explicitly. This transform produces one mean per row,
without training targets or a fitted distribution. Fresh-process tests cover
numeric/categorical/missing/unseen input composition.

Independent tests verify derivatives, offset-aware base, three rounds, weight
semantics, rejected updates and persistence. Real A8 quality/selection evaluation,
dispersion/calibration, Tweedie/composition/A9, AFT/A10 and CUDA remain open.
