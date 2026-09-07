# Poisson counts and exposure

The CPU Poisson recipe models integer counts with mean
exposure * exp(raw + offset). Raw model output is a log rate per unit exposure.
Declare strictly positive exposure as an aligned [N,1] structure role; use ones
for unit exposure. Original sample weights are independent of exposure.

```python
import numpy as np
from openboost import NumericData, Problem, RunContext
from openboost.recipes import poisson
from openboost.outputs import poisson_mean

x = NumericData([[0], [1], [2], [3]], [0, 1, 2, 3], ("x",))
v = NumericData([[0.5], [2.5]], [10, 11], ("x",))
train = Problem(x, [[0], [1], [3], [5]], x.row_ids,
                structure={"exposure": [[0.5], [1], [2], [1]]})
valid = Problem(v, [[1], [2]], v.row_ids,
                structure={"exposure": [[1], [0.5]]})
fit = poisson(train, valid, context=RunContext("poisson-demo", 7), rounds=3)
means = poisson_mean(fit.state.best_model.predict(v), [1, 0.5])
assert np.isfinite(means["rate"]).all()
np.testing.assert_allclose(means["count_mean"], means["rate"] * [1, 0.5])
```

Poisson.geometry returns weighted mean negative log likelihood (including the
log-factorial constant) and unweighted gradients/curvatures. Curvature equals
the count mean. The standard Newton adapter applies original weights once.
There is no implicit curvature floor or additional Poisson step constraint.

Initialization maximizes the constant-rate likelihood with the declared offsets:
log(sum(weight*count)) - log(sum(weight*exposure*exp(offset))).
The sums are evaluated in log space. When all positive-weight counts are zero,
minimum_rate supplies an explicit positive raw-rate initializer (default 1e-6);
it is not a prediction floor. Fixed/backtracking steps and validation best-model
selection use Poisson likelihood.

Offsets are additional log-rate offsets. Do not also put log exposure in offset
when supplying exposure separately. Accepted raw caches exclude both roles.
Supply inference offsets explicitly through model.predict, then call
poisson_mean(raw, exposure) to obtain named rate and count_mean arrays. Doubling
exposure at fixed raw output doubles count mean and leaves the unit rate unchanged.

Generic model artifacts store raw trees/base, not exposure, training offsets or
an output-family tag. The caller must retain the declared Poisson inference
transform and supply new exposure. Fresh-process tests verify that composition
on numeric/categorical/missing/unseen inputs.

Fractional or negative counts, nonpositive/missing exposure, unknown structure,
nonfinite geometry and means outside positive float64 range fail explicitly.
Tests verify independent geometry, offset-aware initialization, three-round
trees, zero counts and exposure scaling. Real A7 deviance/calibration comparisons,
Gamma/A8, Tweedie/composition/A9, AFT/A10 and CUDA remain required work.

The current A7 evaluation worker binds frozen exposure vectors to the public
structure role and persists a period-count output tag with the raw model.
Prediction packets must supply new exposure; extra offsets and exposure on
other tasks are rejected. Five frozen frequency folds pass bounded validation
and exact fresh inference in Sprint 054. Full A7 quality/search gates stay open.
