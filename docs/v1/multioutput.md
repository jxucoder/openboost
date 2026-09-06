# Multi-output squared regression

multi_squared fits numeric targets [N,K] with matching raw width. It supports
independent scalar trees or one shared vector tree per round. All K updates
use the same accepted snapshot and commit or reject together.

```python
import numpy as np
from openboost import NumericData, Problem, RunContext
from openboost.recipes import multi_squared
from openboost.multioutput import TargetScale, MultiOutputModel

x = NumericData([[0], [1], [2], [3]], [0, 1, 2, 3], ("x",))
v = NumericData([[0.5], [2.5]], [10, 11], ("x",))
train = Problem(x, [[0, 3], [2, 8], [1, 2], [6, 1]], x.row_ids)
valid = Problem(v, [[1, 4], [4, 2]], v.row_ids)
scaling = TargetScale.fit(train)
fit = multi_squared(scaling.transform(train), scaling.transform(valid),
                     context=RunContext("multioutput-demo", 7), mode="shared", rounds=3)
model = MultiOutputModel(fit.state.best_model, scaling)
prediction = model.predict(v)
rmse = np.sqrt(np.average((prediction-valid.target)**2, axis=0, weights=valid.weight))
assert prediction.shape == (2, 2) and rmse.shape == (2,)
```

MultiSquared uses the sum of output half-squared errors, averaged by original
row weights. Its mse method reports weighted error for every target separately;
take square roots for per-target RMSE. Scalar K=1 agrees with the ordinary squared
recipe. Censored target kinds and classification schemas are rejected explicitly.

mode="independent" fits K scalar trees with separate topology. mode="shared"
sums vector gains and fits full-dimensional leaves on common topology. grower
selects depthwise, best_first or symmetric. Optional projection [K,S] in shared
mode transforms split gradients by P and diagonal curvature by P**2; leaves still
use full K statistics. The caller owns the projection and its scientific meaning.
There is no automatic sketch selection or speed claim.

TargetScale.fit uses training-only weighted means and population standard
deviations. Targets constant over positive-weight rows use scale one and a
persisted constant flag. No validation statistics are fitted. transform scales
targets and offsets consistently; absent offsets give zero weighted base up to
roundoff. With offsets, the base corrects the weighted residual mean.

The raw recipe performs no implicit target standardization. For the A6 workflow,
fit the scaler on training, transform training/validation, and wrap the selected
model in MultiOutputModel. The wrapper restores original units and accepts
original-unit inference offsets. Its versioned artifact embeds model, means,
scales and constant flags. Target columns retain their declared positional order.

Tests cover three rounds under all policies/modes, weights/offsets, target
permutation, K=1, constant targets, foreign target semantics, atomic rejection,
and fresh-process mixed/missing/unseen prediction in both modes. Real Parkinsons
subject-split evaluation, standardized aggregate plus per-target quality,
shared preparation/stopping and CUDA remain required work.
