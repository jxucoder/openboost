# Joint Normal distributional boosting

A scalar observation can have multiple predicted parameters. Declare
`Problem(..., raw_width=2)` for Normal's raw mean and log-scale columns. Targets
remain `[N, 1]`; optional offsets must be `[N, 2]`. The same runtime and mapped
scalar tree terms used by squared boosting update these parameters jointly.

```python
import numpy as np
from openboost import NumericData, Problem, RunContext
from openboost.recipes import normal
from openboost.objectives import Normal

x = NumericData([[0], [1], [2], [3], [4], [5]], np.arange(6), ("x",))
v = NumericData([[0.5], [2.5], [4.5]], [10, 11, 12], ("x",))
train = Problem(x, [[-3], [-1], [0], [1], [3], [6]], x.row_ids, raw_width=2)
valid = Problem(v, [[-2], [0.5], [4]], v.row_ids, raw_width=2)
fit = normal(train, valid, context=RunContext("normal-demo", seed=7),
             rounds=3, bins=6, mode="natural")
raw = fit.state.best_model.predict(v)
parameters = Normal.parameters(raw)  # columns: mean, positive scale
assert parameters.shape == (3, 2) and np.all(parameters[:, 1] > 0)
assert np.isfinite(Normal.loss(valid, raw))
```

`Normal.geometry(problem, raw)` exposes weighted mean NLL, unweighted ordinary
likelihood gradient and Fisher diagonal `[N, 2]`. For residual r=mean-target and
precision p=exp(-2 log-scale), the gradient is `(r*p, 1-r*r*p)` and Fisher diagonal
is `(p, 2)`. It is not the observed Hessian. `diagonal_direction` returns `-g` in
ordinary mode or `-g/(F+damping)` in natural mode. Ordinary mode rejects nonzero
damping. No dense per-row metric is needed for Normal's diagonal Fisher.

`least_squares(problem, direction_column)` creates G=-w*z and H=w from an
unweighted direction. These regression curvatures are distinct from Fisher
entries. Both parameter trees fit the same accepted snapshot, and one coefficient
commits or rejects both terms. The default uses six-trial backtracking with strict
training NLL decrease. `step="fixed"` commits finite candidates. Geometry, trees,
weights, offsets, best snapshots and persistence use public shared components.

With no offsets, initialization uses training weighted mean and log standard
deviation, floored by `minimum_scale` (default 1e-6). With offsets, the mean uses
weights multiplied by exp(-2 log-scale-offset); the variance uses those residual
squares normalized by original weight mass. The floor applies to initial base
scale only. Later scale/precision underflow or overflow is rejected without
clipping. Backtracking continues after invalid numerical trials, recording error
types in `NormalStep.failures`; wrong schemas fail before any trials.

`Model.save/load` stores raw ensemble state and mappings. `Normal.parameters`
converts raw mean/log-scale to mean/scale without training data. For observation
offsets, pass them to `Model.predict(..., offset=...)` before converting; objective
loss receives unoffset raw caches and applies Problem offsets once. The model is
a raw predictor and does not persist a distribution tag or calibrated intervals.

This is a joint-update CPU recipe, not full NGBoost parity. Ordered parameter
updates, additional distributions and CUDA remain required later work.
[Formula](formula-runs.md) now probes full GGN geometry through shared components. No real-dataset quality or speed advantage is claimed.
Per-round traces retain arrays, and trial validation currently recomputes ensemble
predictions. Formula and heterogeneous sequential runs now provide the next construction probe.

## Comparing stored predictions

`Normal.compare(problem, before_raw, after_raw)` is a separate public CPU operation.
It returns an immutable `LossChange`, bounding the weighted mean NLL change at the
exact stored raw values in the problem's row order. It applies offsets and weights
once and validates both snapshots, including zero-weight rows. It does not call
the reporting loss or gradient callbacks. Absolute `Normal.loss` values retain
their existing meaning.

`change.lower` and `change.upper` are authoritative bounds. `estimate` and
`uncertainty` are derived diagnostics; `method` identifies the numerical method.
`status` distinguishes improvement, worsening, identical stored raw, and unresolved
sign. `change.improves(min_delta=0)` requires `upper < -min_delta`. An unchanged
stored pair has exactly zero bounds; a different pair with equal loss is unresolved.
Unresolved support carries a `reason` and `None` bounds/diagnostics.

The current method uses bounded Taylor/interval arithmetic with separately rounded
binary64 operations and gradual underflow. Evaluated exponent intervals must stay
within `[-64,64]`; finite inputs outside that comparison support are unresolved.
Invalid Normal domains raise, and there is no clipping or implicit fallback. The
bound does not rest on an assumed universal error for platform `exp`/`expm1`.
CPU correctness checks include recorded false improvements and an analytic
`-2^-61` improvement lost by subtraction of full losses. No speed claim is made.

This operation is available for explicit algorithm composition. Current recipe
backtracking, best-model selection and patience still compare reported losses;
their migration is the separate Sprint 092-C work. A comparison component alone
does not establish corrected boosting conformance.
