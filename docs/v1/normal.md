# Normal distributional boosting

A scalar observation can have multiple predicted parameters. Declare
`Problem(..., raw_width=2)` for Normal's raw mean and log-scale columns. Targets
remain `[N, 1]`; optional offsets must be `[N, 2]`. The same runtime and mapped
scalar tree terms used by squared boosting update these parameters jointly or
in an explicit parameter order.

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
entries. With `update="joint"` (default), both parameter trees fit the same accepted
snapshot, and one coefficient commits or rejects both terms. `update="forward"`
fits mean then log-scale; `update="reverse"` fits log-scale then mean. Each ordered
substep recomputes geometry from the latest accepted state. A rejected substep
leaves that state unchanged and does not prevent the next parameter attempt. The default uses six-trial backtracking with strict
training NLL decrease proved by `Normal.compare`. `step="fixed"` commits finite candidates. Geometry, trees,
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

The default CPU learner uses [exact Newton ordering and original-row leaves](newton-order.md)
for its once-weighted scalar direction fields. It sums stored binary64 G/H exactly,
compares rational split gains and rounds the final leaf solution once. This fixes
cancellation in leaf reductions as well as split ties; it does not make the
floating Normal geometry exact. Regularization, curvature minima and split penalty
configure the default learner. A custom `learner` continues to own all tree settings.

```python
from functools import partial
from openboost.newton_order import rank, leaf
from openboost.tree import best_first

ordered = normal(
    train, valid, context=RunContext("normal-ordered", seed=7),
    rounds=3, bins=6, update="reverse", damping=.25,
    learner=partial(best_first, max_depth=3, max_leaves=4,
                    ordering=rank, field_leaf=leaf),
)
assert len(ordered.steps) == ordered.stop.completed_rounds
assert all(len(substeps) == 2 for substeps in ordered.steps)
```

These CPU recipes do not establish full NGBoost parity. The experimental
[CUDA Normal path](execution.md) supports joint and ordered updates. Exact
resident split/leaf choices and [default/installed integration](normal-exact-default.md)
have source-specific historical checks. The [checkpoint](checkpoint.md) describes
the omitted replay material and pending candidate validation. Real selected Normal
quality and full cost acceptance remain open; no speed or complete v1 claim follows.
[Formula](formula-runs.md) now probes full GGN geometry through shared components. No real-dataset quality or speed advantage is claimed.
Full per-round traces retain arrays; summary retention omits sample arrays.
CPU best selection replays the current best validation model for its comparison
anchor. This extra replay is a correctness-first reference cost.

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

All three CPU update orders use this operation for three separate decisions. Backtracking
compares each candidate with accepted training raw. Validation best compares with
the previous best model's predictions at zero threshold. Patience compares with
its last qualifying validation snapshot using `min_delta`, once per outer round.
Best may advance between ordered substeps; patience waits for the complete sweep.
An improvement can advance best and patience even when reporting floats are equal.
Unresolved or unchanged evidence cannot prove improvement. Fixed acceptance can
still commit a finite worsening or unresolved candidate.

`steps` retains one entry per completed outer round. Joint entries are NormalStep
records; ordered entries are tuples of two NormalStep records. Each has
`round_index`, `channels`, `before_version` and `after_version`.
`NormalStep.comparisons` retains each trial's `LossChange` (or None for a failure
before comparison); `validation_change` records the patience comparison on the
last substep only. Summary retention keeps this grouping and scalar evidence,
while omitting sample arrays. Zero rounds create no learners or substep records.
The shared recipe-result validator and scheduler accept both forms. Absolute
losses remain truthful; exact CPU consumers do not supply missing CUDA evidence.
