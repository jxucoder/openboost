# Formula and heterogeneous sequential runs

The saturation Formula models `a * (1 - exp(-b*x))`, where a and b are softplus
transforms of two raw parameters. Structural x is a positive `[N, 1]` role supplied
separately from feature columns. Trees predict parameters from features; the
formula combines those parameters with structural x.

```python
import numpy as np
from openboost import NumericData, Problem, RunContext
from openboost.objectives import Formula
from openboost.recipes import formula, squared
from openboost.runs import RunSpec, run_many

x = NumericData([[0], [1], [2], [3]], [10, 11, 12, 13], ("feature",))
y = [[0.5], [1], [2], [3]]
structured = Problem(x, y, x.row_ids, raw_width=2,
                     structure={"x": [[0.2], [0.5], [1], [2]]})
scalar = Problem(x, y, x.row_ids)
specs = [RunSpec(RunContext("formula", 7), structured, structured, formula,
                 {"rounds": 2, "bins": 4}),
         RunSpec(RunContext("squared", 7), scalar, scalar, squared,
                 {"rounds": 1, "bins": 4})]
results = run_many(specs)
assert all(item.error_type is None for item in results)
raw = results[0].result.state.model.predict(x)
prediction = Formula.predict(raw, structured.structure["x"])
assert prediction.shape == (4, 1) and np.isfinite(prediction).all()
```

The example shares train/validation inputs only to demonstrate composition. Use
separate prescribed partitions for real evaluation.

`Formula.geometry` returns weighted half-square loss, unweighted gradient and
per-row GGN `J.T @ J`. It is rank one; `full_direction(..., damping=...)` uses a
symmetric positive-definite Cholesky solve and rejects numerically singular
systems. The recipe defaults to damping=0.1. There is no implicit pseudoinverse
or diagonal fallback. Learners fit each direction with the existing once-weighted
least-squares adapter, and mapped terms commit jointly through shared backtracking.
The initial base uses inverse-softplus of max(weighted target mean, 1e-6) and 1;
it is an initializer, not the coupled optimum. Offsets apply to raw parameters
before geometry and output. A fixed two-parameter dense GGN is used here; this
is not a memory guarantee for arbitrary high-dimensional models.

`Model.save/load` persists the raw ensemble. At inference, provide structural x
again to `Formula.predict`, and apply any parameter offsets through Model.predict.
No training-row structure or formula tag is embedded in the raw model. This is
one explicit formula probe, not a general symbolic-expression engine.

`RunSpec` declares context, train/validation problems, recipe and copied immutable
scalar options. Problems can share NumericData without sharing accepted state.
`run_many` validates unique IDs before running, returns every outcome in requested
order, and records recipe exceptions while continuing other jobs. Result context
and problem identities must match the spec. Completed results follow the
[shared result contract](results.md); external recipes retain their own result
types and per-round payloads. KeyboardInterrupt/SystemExit propagate.
An expected injected failure tests isolation; actual required failures still fail
evaluation. Reusing a run ID in a separate invocation replays its logical identity.

Only sequential execution is supported. This provides independent recipe results,
per-run round budgets, best snapshots and errors, not process isolation, fused
training or a resource scheduler. [Validation patience](stopping.md) is a public
operation also used by the built-in recipes. Arbitrary recipe code
must respect the immutable input contract. M=1/2/8 comparisons verify deterministic
same-ID independent and reordered execution; they establish no speed benefit.

[Shared preparation](preparation.md) now supports explicit training-code reuse
across independent runs.
