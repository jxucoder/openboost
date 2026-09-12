# Event/right-censored log-normal AFT

AFT models log time as raw + offset + sigma*Z with standard Normal Z and fixed
positive sigma. Bind [lower, upper] targets using target_kind="event_right":
exact events have equal positive bounds; right-censored observations have a
finite positive lower bound and upper=+infinity. Left/interval censoring and
truncation are unsupported and rejected. Ordinary numeric targets remain finite.

```python
import numpy as np
from openboost import NumericData, Problem, RunContext
from openboost.recipes import aft
from openboost.survival import AFTModel

x = NumericData([[0], [1], [2], [3]], [0, 1, 2, 3], ("x",))
v = NumericData([[0.5], [2.5]], [10, 11], ("x",))
train = Problem(x, [[1, 1], [2, np.inf], [4, 4], [8, np.inf]], x.row_ids,
                target_kind="event_right")
valid = Problem(v, [[2, 2], [6, np.inf]], v.row_ids, target_kind="event_right")
sigma = 0.7
fit = aft(train, valid, context=RunContext("aft-demo", 7), sigma=sigma, rounds=3)
model = AFTModel(fit.state.best_model, sigma)
output = model.predict(v, times=[1, 3, 10], probabilities=[0.1, 0.5, 0.9])
assert output["survival"].shape == (2, 3)
assert np.all(np.diff(output["survival"], axis=1) <= 0)
```

LogNormalAFT.geometry returns weighted censored negative log likelihood,
unweighted gradients and diagonal curvature. Exact-event density includes its
time Jacobian and Normal constant; right censoring uses negative log survival,
not event density. Original weights enter the shared Newton fields once.
Offsets enter geometry once and remain outside accepted raw caches.

Normal upper tails use erfc centrally and Gauss–Laguerre quadrature above z=8,
computing the small curvature term without subtracting nearly equal values.
Tests compare this with an independent continued fraction, including extreme
positive standardized times. Unrepresentable geometry fails explicitly.

Initialization is the weighted average of log lower bounds minus offset.
It is a finite starting location, not a censoring-adjusted intercept optimum;
even an all-censored sample can initialize. Fixed/backtracking steps and
validation selection use censored likelihood. Censored lower bounds are not
treated as observed deaths for evaluation.

AFTModel stores the raw model and declared sigma in openboost-aft-lognormal-v1.
Use the same sigma used for fitting. Loading validates both and needs no
training objective or observations. predict requires a nonempty positive times
grid and accepts probabilities strictly between zero and one. Outputs are median,
mean, survival [N,times] and quantile [N,probabilities]. Mean is exp(F+sigma²/2),
median exp(F); supply inference offsets explicitly.

Target kind participates in problem identity, and valid +infinity survives
binding. Tests cover bounds rejection, three-round geometry/tree parity,
finite differences, censoring effects, monotone outputs and fresh-process
mixed/missing/unseen inference. Real A10 NLL/IPCW comparisons, calibration,
learned scale, other censoring forms and CUDA remain open.

The current A10 worker converts frozen time/event arrays to event_right bounds,
fixes sigma=1 and persists it with a log-normal output tag. Five frozen Veteran
folds pass bounded validation and exact fresh location/scale replay in Sprint 059.
Independent censored NLL checks pass. This is not full A10 quality acceptance.

## Experimental device components

`device_aft.objective(sigma=1.0)` supplies resident preparation, initialization,
loss, gradient and named Newton fields for the scalar `DeviceRun` operations.
`device_aft.geometry(ops, problem, raw)` returns an owned float32 `[N,2]`
gradient/curvature buffer. The prepared record retains the fitting sigma as
float64 and binds it to the objective. Different objective scales or families
cannot reuse that record. Censoring uses a separate bool indicator derived from
the original bounds; lower times and offsets use float32 resident storage.

Row likelihood/derivative arithmetic and base reductions use float64 at those
stored inputs. All rows must have finite representable geometry, including rows
with zero weight. Curvature must remain positive in float32; a censored gradient
must remain nonzero. Unsupported time/geometry domains fail explicitly. These
operations contain no host geometry fallback, learned scale or broader censoring
support. Categorical CUDA remains unsupported.

`device_aft.export(run, state, best=False)` produces an `AFTModel` using the scale
from the prepared training record. The caller need not repeat sigma. After export,
the existing CPU format supports raw location, median, mean, survival and quantile
inference with explicit offsets, including after the device run is closed.

The objective also supplies `device_aft.compare(ops, problem, before, after)`.
This encloses the likelihood change at exact stored inputs, independently of the
reporting loss. Event density constants cancel in a quadratic difference. Censored
changes use bounds on the integral of the Normal Mills ratio, with enclosing
logarithm, density and continued-fraction arithmetic. All rows pass the existing
geometry checks first; the two snapshots remain caller-owned. Only a 32-byte
summary and validation flags leave CUDA, and comparison scratch is released.

Censored comparison supports standardized segment endpoints in [-16,1e12]; a
changed finite segment outside that range returns `tail_range` unresolved. Unavailable
arithmetic bounds return `arithmetic_range`. Bounds containing zero are also
unresolved. The integral bounds can be conservative for large or competing steps.
Only an upper bound below `-min_delta` proves a qualifying decrease; rounded
reporting ties and empirical tolerances do not determine the decision.

`device_recipes.aft(..., sigma=1.0)` composes the existing scalar recipe with this
objective. It accepts the scalar recipe's explicit run_id/seed, tree configuration,
fixed/backtracking step, rounds and validation patience. Backtracking tries at most
six coefficients and accepts only proved improvement. Fixed steps explicitly allow
finite worsening or unresolved candidates. Accepted state, best validation state
and last-qualifying patience state have distinct anchors. The returned run/state
is caller-owned; use `device_aft.export` for final/best scale-aware inference and
close the run when finished.

Historical construction covers independent tail/domain and loss-change
mathematics, prescribed CPU compositions, recipe trajectories and fresh inference.
A subsequent CUDA run exercised the declared component, comparison and recipe
controls. Its newer raw archive is outside this PR; see the [checkpoint](checkpoint.md)
for candidate validation. Full R5 conformance, real-data survival quality and
execution-cost acceptance remain open.
