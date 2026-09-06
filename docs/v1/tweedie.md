# Tweedie nonnegative means

Tweedie mean regression accepts zero and positive scalar targets, with variance
power fixed per fit strictly between one and two. Raw predictions are log means;
positive_mean converts them to means. Dispersion and the full compound
distribution are not fitted.

```python
import numpy as np
from openboost import NumericData, Problem, RunContext
from openboost.recipes import tweedie
from openboost.outputs import positive_mean

x = NumericData([[0], [1], [2], [3]], [0, 1, 2, 3], ("x",))
v = NumericData([[0.5], [2.5]], [10, 11], ("x",))
exposure = np.array([0.5, 1, 2, 1])
period_total = np.array([0, 2, 4, 8])
train = Problem(x, (period_total/exposure)[:, None], x.row_ids, weight=exposure)
valid = Problem(v, [[0], [4]], v.row_ids, weight=[1, 0.5])
fit = tweedie(train, valid, context=RunContext("tweedie-demo", 7),
               power=1.5, rounds=3)
annualized_mean = positive_mean(fit.state.best_model.predict(v))
period_mean = annualized_mean * [1, 0.5]
assert np.isfinite(period_mean).all()
```

For power p and total log mean f=raw+offset, the objective is
y*exp((1-p)*f)/(p-1) + exp((2-p)*f)/(2-p).
Its gradient is exp((2-p)*f)-y*exp((1-p)*f), and its curvature is
(2-p)*exp((2-p)*f)+(p-1)*y*exp((1-p)*f).
The zero-target term is exactly zero. Original weights are applied once by
the shared Newton adapter.

The constant initializer accounts for offsets using the log ratio of
sum(w*y*exp((1-p)*offset)) to sum(w*exp((2-p)*offset)).
All-zero positive-weight targets use the explicit minimum_mean initializer
(default 1e-6), not a prediction floor. Fixed/backtracking steps and validation
selection use weighted objective values. This objective has the same prediction
ordering as Tweedie deviance at fixed targets/power, but is not a reported
deviance or full normalized distribution likelihood.

For A9 annualized loss, divide period amounts by exposure and use exposure
weights, including explicit extra business weights if required. Do not also
add log exposure to the offset. Convert annualized means to period means by
multiplying by exposure at inference. Other units require an explicit contract.
Unknown structure roles, negative targets, invalid powers and unrepresentable
positive terms are rejected.

Generic inference artifacts store raw trees/base; callers retain the output
transform and unit conversion. Fresh-process tests cover mixed, missing and
unseen features plus offsets. Power affects fitting and evaluation but not the
exp(raw) inference transform.

Independent tests cover three powers, zero-target finite differences, intercepts,
three rounds, annualized weights, rejected steps and persistence. Frequency–
severity composition, real A9 quality, calibrated tails, AFT/A10 and CUDA remain
separate required work.
