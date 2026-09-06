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
