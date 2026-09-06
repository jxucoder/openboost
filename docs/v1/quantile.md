# Quantile and penalized residual leaves

Quantile boosting uses pinball gradients and unit pseudo-curvature to choose
splits, then fits each leaf from its routed current residuals and original
weights. Pseudo-curvature is not the Hessian of pinball loss.

```python
import numpy as np
from openboost import NumericData, Problem, RunContext
from openboost.recipes import quantile
from openboost.tree import best_first

x = NumericData([[0], [1], [2], [3]], [0, 1, 2, 3], ("x",))
v = NumericData([[0.5], [2.5]], [10, 11], ("x",))
train = Problem(x, [[0], [2], [1], [6]], x.row_ids, weight=[1, 2, 1, 3])
valid = Problem(v, [[1], [4]], v.row_ids)
fit = quantile(train, valid, context=RunContext("quantile-demo", 7),
               q=0.8, rounds=3, grower=best_first, penalty=2, anchor=0)
assert np.isfinite(fit.state.best_model.predict(v)).all()
```

Initialization is the weighted quantile of target minus offset. Each round
recomputes residuals from the accepted model, including the offset once.
Fixed or backtracking steps and validation selection use weighted mean pinball
loss. The gradient at an exact target/prediction tie is -q, matching the declared
reference subgradient convention.

All three growth policies accept row_leaf(view, total, names) together with
leaf_context=ResidualContext(problem, residual). The context owns aligned
unweighted residuals. Each immutable ResidualView carries global row IDs,
original weights and residuals for exactly the routed rows. total and names
describe the additive leaf statistics. Context identity must match the split
problem; custom additive and routed leaf solvers cannot be supplied together.

quantile_leaf(view, q=...) selects the leftmost weighted quantile when penalty=0.
With penalty>0 it minimizes the sum of original-weight pinball losses plus
penalty*(value-anchor)**2/2. A monotone subgradient scan finds the unique
minimizer at a residual breakpoint or between breakpoints. Weight mass is not
normalized away from this penalty. Zero-mass leaves return the anchor only for
positive penalty; unpenalized zero-mass leaves fail explicitly.

The recipe's reg_lambda regularizes split scoring. Its separate penalty and
anchor configure leaf fitting; they do not change the validation metric or
add a model-wide training penalty. An anchor with zero penalty is rejected.
Depthwise, best-first and symmetric policies share this contract through grower.

Inference stores the solved scalar leaf values in the existing tree format,
without residuals, training rows or a training objective. These tests establish
CPU mechanics and D3 solver correctness, not real A5 quality, quantile coverage,
noncrossing guarantees, agent-author effort, CUDA or performance results.
