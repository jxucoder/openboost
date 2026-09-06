# Initial CPU state components

B03 provides public numeric inputs, run identity, immutable state transitions and
constant-term artifacts. It is an architecture slice, not a boosting trainer.

```python
import numpy as np
from openboost import NumericData, Problem, RunContext
from openboost.runtime import initialize, propose, resolve

x = NumericData([[1.0], [2.0]], [10, 20], ("feature",))
problem = Problem(x, [[5.0], [7.0]], [10, 20],
                  weight=[1.0, 3.0], offset=[[2.0], [4.0]])

def score(problem, raw):
    error = problem.with_offset(raw) - problem.target
    return float(np.dot(problem.weight, error[:, 0] ** 2) / problem.weight.sum())

state = initialize(RunContext("demo", seed=7), problem, problem, [0.0], score=score)
trial = propose(state, [6.0], coefficient=0.5)
assert resolve(state, trial, accept=False, score=score) is state
accepted = resolve(state, trial, accept=True, score=score)
np.testing.assert_array_equal(accepted.train_raw, [[3.0], [3.0]])
np.testing.assert_array_equal(
    accepted.model.predict(x, offset=problem.offset), [[5.0], [7.0]]
)
assert accepted.best_score == 0.0
```

The example deliberately uses the same problem for train and validation to expose
state arithmetic; real evaluation must use the prescribed separate partitions.

`NumericData` owns float64 CPU features and unique integer row IDs. Feature names
are ordered. NaN represents numeric missingness; infinity is rejected. No binning
or category transformer is fitted yet. Owned array values cannot be made writable;
callers must not alter array metadata. Identity includes feature content/order,
row IDs and schema, and is computed once on construction.

`Problem` requires targets and offsets shaped `[N, K]` and row weights `[N]`.
One explicit row-ID vector declares the order of all role arrays; binding rejects
a mismatch instead of sorting. The caller is responsible for aligning each role
before binding. Targets are finite numeric values; specialized target types,
queries, structure and objective support arrive in later slices. Unknown arguments
are rejected. Original weights are retained, never automatically applied here.

`RunContext.rng(round_index, component, purpose)` returns a new deterministic
stream for that logical key. Reusing the key reproduces the stream; different
run IDs distinguish streams. It neither mutates global NumPy RNG nor advances a
shared run cursor. CUDA devices are explicitly rejected in B03.

Accepted raw caches contain the base and accepted term sum, without offsets.
Algorithm code owns `score(problem, raw)`, including exactly-once weights and
offsets. Smaller validation scores are better. `resolve(..., accept=True)` need
not improve validation; it commits the term while preserving an earlier best
model when appropriate. A rejected proposal returns the identical state. Stale
parents, other runs and divergent parent histories are rejected. Nonfinite scoring
fails before a new state is returned. Vector terms commit jointly.

`ConstantModel.save(path)` / `ConstantModel.load(path)` use the explicit
`openboost-constant-v1` JSON format. It records feature names, vector base and
constant terms with one coefficient each. Offsets are supplied at inference and
are never embedded as training-row offsets. Loading needs no training objective.
This is an inference artifact, not a training-resume checkpoint; it does not store
run RNG, best/stop history or input datasets. Unknown versions/fields, duplicate
fields, nonfinite payloads and inconsistent output widths fail.

B04 adds numeric preparation and shared tree operations. B05 adds complete
squared-error and Normal recipes; B06 must exercise Formula and heterogeneous
sequential runs before interfaces stabilize. The constant-only artifact format
makes no promise of representing those future learners or mappings.
