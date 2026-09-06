# Independent validation stopping

Every CPU recipe accepts `patience=None` and `min_delta=0.0`. Positive integer
patience enables early stopping; None uses the complete round budget. A nonzero
min_delta requires enabled patience. Invalid configuration fails even at zero rounds.

```python
from openboost import NumericData, Problem, RunContext
from openboost.recipes import squared
from openboost.stopping import StopState

x = NumericData([[0], [1], [2], [3]], [0, 1, 2, 3], ("x",))
train = Problem(x, [[0], [1], [2], [3]], x.row_ids)
valid = Problem(x, [[3], [2], [1], [0]], x.row_ids)
fit = squared(train, valid, context=RunContext("stop-demo", 7),
              rounds=20, patience=2)
assert fit.stop.reason == "patience"
assert fit.stop.completed_rounds == len(fit.steps) == 2
assert fit.state.version == 2
assert fit.state.best_model.terms == ()

# External algorithm loops use the same public operation.
stop = StopState.start(10.0, rounds=8, patience=2, min_delta=1.0)
stop = stop.observe(9.0)  # Exact threshold does not reset patience.
stop = stop.observe(8.5)  # Improvement from 10 exceeds the threshold.
assert stop.stale_rounds == 0 and stop.reason is None
```

The small example deliberately uses opposite targets on the same feature rows
to expose stopping behavior; it is not a real evaluation split.

StopState is immutable and separate from AcceptedState. Initial validation is
the baseline and consumes no round. Once per completed outer round, observe the
current finite validation score, with smaller scores better (ranking uses negative
NDCG). Strict improvement must exceed min_delta relative to the last qualifying
improvement. Ties and insufficient improvements increment stale_rounds; qualifying
improvement resets it. Stop at patience consecutive stale rounds or the round
budget. A simultaneous limit reports `patience`; zero rounds reports `budget`.
Further observations after termination raise an error.

Backtracking still uses training loss for step acceptance. Individual search
trials and ordered substeps must not advance the stopping clock. A fully rejected
outer round observes the unchanged model once and consumes patience. Accepted-state
version counts commits, not completed rounds. The existing per-step coefficients
record attempted step sizes; they do not determine patience.

`fit.state.model` is the final accepted model; `fit.state.best_model` is the strict
validation minimum, including the initial model. Best-model selection is independent
of min_delta, so small improvements can update best_model without resetting patience.
`fit.stop` reports completed_rounds, stale_rounds, reference_score, last_score and
reason. It is an in-memory progress record, not a training-resume checkpoint.

RunSpec passes scalar patience/min_delta options unchanged. Each run owns its stop
record even when it shares PreparedData. Reordering or retrying stable run IDs
preserves results; invalid configuration or a nonfinite initial/observed metric
fails that run and run_many retains its error. Candidate numerical failures during
backtracking retain the recipe's existing rejection behavior.

Tests cover all twelve recipes, hand-calculated threshold sequences and M=1/8/32
heterogeneous scalar/Normal runs with different actual validation stop rounds,
failed runs, retries and regrouping. This establishes CPU state semantics, not
real model-selection quality, GPU batching, fusion or a speed improvement.
