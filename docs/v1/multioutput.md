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

## Experimental resident geometry

Sprint 118 constructs `openboost.device_multi_squared.objective()` and public
`prepare`, `base`, `geometry`, `loss` and `compare` operations. Targets and offsets
are explicitly uploaded as float32 with matching K columns. Geometry returns
independent owned `[N,K]` gradients and unit curvature. Base and reporting loss
use float64 reductions; loss sums output half-squares before averaging by row
weight. Every row must have finite float32 geometry, including weight-zero rows.

`compare` encloses the exact cancelled polynomial change with directed doubles;
only validation flags and a 32-byte summary leave the device. Identical raw gives
an exact unchanged result only after complete domain validation. Reporting ties
do not establish unchanged loss. Finite enclosures that include zero remain
unresolved and cannot establish strict improvement.

`device_objectives.vector_fields(ops, data, gradient, curvature, projection=None)`
applies weights once to diagonal geometry. Optional `[K,S]` float32 projections
use `g @ P` and `h @ (P**2)` for split statistics; callers separately retain full
leaf fields. Projection columns must remain nonzero and finite after conversion.
`channel_fields(..., channel=k)` selects scalar fields for independent trees.
The geometry inputs remain caller-owned. Historical objective checks and the
pending candidate validation are scoped in the [checkpoint](checkpoint.md).

## Experimental resident recipes and general mappings

`device_recipes.multi_squared` composes these operations with the resident grower
and joint transactions. `mode="independent"` fits K scalar trees from one geometry
snapshot; `mode="shared"` fits one K-output tree. Both propose all outputs atomically.
A shared projection changes split statistics while full K fields solve the leaves.
Fixed steps commit valid proposals; backtracking requires a certified strict training
improvement. Best validation and patience each own a separate comparison anchor.
An accepted no-op increments the round but does not extend the best model.

```python
# Experimental resident API; requires real CUDA and matching numeric Problems.
from openboost import device_recipes
from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext

# train and validation are matching numeric multi-output Problems.
with ExecutionContext("cuda:0") as execution:
    fit = device_recipes.multi_squared(
        DeviceOperations(execution), train, validation,
        run_id="multi-output", seed=17, mode="shared",
        rounds=20, bins=16, max_depth=2, patience=4,
    )
    model = fit.run.export(fit.state, best=True)
    fit.run.close()
```

An optional `learner(ops, data, split_fields, leaf_fields, output_width)` supplies
tree policy through public operations and owns its tree parameters. For more
general compositions, `device_recipes.mapped(..., objective=..., learner=...)`
calls `learner(run, raw)` once per outer round; return a nonempty tuple of
`DeviceTerm` objects. The driver snapshots terms, releases new learner workspace
and preserves preexisting caller-owned trees. Result history retains scalar
trial/decision information and the run owns final/best trees, with no per-round
raw-array history. The caller closes the run after exporting required artifacts.

`DeviceTerm(tree, mapping)` now accepts an immutable real float32 `[L,K]` map,
with L matching the tree and K matching the run. `map_update` rounds each product
and accumulates in increasing leaf-channel order before coefficient multiplication
and raw addition. L=1 retains its earlier kernel path. Exported CPU Model inference
uses float64 arithmetic; numerical agreement and exact stored-device arithmetic
are separate checks. Do not claim universal bitwise identity between those paths.

Target scaling remains explicit: fit `TargetScale` on training, transform both
problems, then wrap the exported model with `MultiOutputModel(model, scale)`.
This preserves constant-target flags and original-unit offsets in saved inference.
New cases include fresh scaled-model replay, separate anchors, reporting ties,
partial learner failures, zero rounds and retained storage over 24 rounds.

Historical checks cover vector objectives, comparisons, final/best models and
scaled inference. Their newer raw archives are outside this PR; see the
[checkpoint](checkpoint.md). Full R8, full-budget train-many, performance and formal
quality/cost acceptance remain open, as does validation of the curated candidate.
