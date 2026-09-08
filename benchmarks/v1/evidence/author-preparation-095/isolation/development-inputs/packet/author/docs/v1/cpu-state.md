# Initial CPU state components

B03 provides public numeric inputs, run identity, immutable state transitions and
ensemble artifacts. The [squared recipe](squared.md) now composes these components.

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

`resolve(..., compare=callback)` explicitly chooses objective-based validation
best selection. The callback receives `(problem, before_raw, after_raw)` and must
return `LossChange`; only proved improvement replaces best. The before snapshot
is an owned read-only replay of the current best model. Current acceptance remains
the caller's explicit boolean. Omitting compare retains reported-score selection.
Comparison errors leave the immutable parent unchanged.

`NumericData` owns float64 CPU features and unique integer row IDs. Feature names
are ordered. NaN represents numeric missingness; infinity is rejected. No binning
is fitted by this record; use Binning separately. Use MixedData (outside this packet)
for explicit numeric/categorical columns. Owned array values cannot be made writable;
callers must not alter array metadata. Identity includes feature content/order,
row IDs and schema, and is computed once on construction.

`Problem` requires targets shaped `[N, T]`, offsets `[N, raw_width]` and row
weights `[N]`. `raw_width` defaults to T; set it explicitly when parameters differ
from observations, such as scalar Normal targets with two raw parameters.
One explicit row-ID vector declares the order of all role arrays; binding rejects
a mismatch instead of sorting. The caller is responsible for aligning each role
before binding. Ordinary targets are finite numeric values. AFT (outside this packet) declares
target_kind="event_right" for validated [lower,upper] bounds, retaining only
valid upper +infinity and defaulting raw_width to one. ClassSchema (outside this packet)
binds typed training-label order to encoded classification targets. Named `structure`
arrays are separately owned `[N, S]` roles in the declared row order; they affect
problem identity but are never automatically appended to features. Recipes must
consume or reject them; squared and Normal reject supplied structure. Unknown arguments
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

`Model.save(path)` / `Model.load(path)` use the explicit
`openboost-ensemble-v2` JSON format. This replaces the earlier constant-only format
and rejects earlier ensemble versions. Model replaces ConstantModel without a
compatibility shim. Optional class order is bound to state and persisted. It records feature
names, vector base, constant terms and vector tree terms with explicit `[L, K]`
output matrices and one coefficient each. `propose_terms` submits multiple terms
as one atomic update. Offsets are supplied at inference and
are never embedded as training-row offsets. Loading needs no training objective.
This is an inference artifact, not a training-resume checkpoint; it does not store
run RNG, best/stop history or input datasets. Unknown versions/fields, duplicate
fields, nonfinite payloads and inconsistent output widths fail.

B04 adds numeric preparation and shared tree operations. The squared recipe is
available alongside joint Normal updates with ordinary/Fisher directions. Initial B06 now exercises Formula and
heterogeneous sequential runs; interfaces remain provisional. Model construction
uses a conservative absolute-value envelope to reject possible prediction
overflow, which can reject extremely large terms even when they would cancel.

## Incremental proposals

`preview_raw(state, proposal)` returns immutable training and validation candidate
raw arrays without observation offsets. A proposal evaluates only its new terms
in their declared order. `resolve` transfers these internally derived values on
acceptance; rejection leaves the original state and its encodings unchanged.
Public state construction and dataclass replacement recompute predictions and do
not accept raw cache arguments. `preview(state, proposal).predict(data)` remains
an independent full-ensemble replay for verification and export.

The runtime reuses `BinnedData` by immutable feature-data and fitted-binning
identities within a run. Trees also accept `predict(data, binned=encoding)` after
checking both identities, including row order, feature values and categorical
schema. Neither caller-provided code arrays nor prediction caches are trusted.
This reduces repeated tree evaluation; it does not claim all training work is
linear, and does not change full trace retention or implement CUDA.
