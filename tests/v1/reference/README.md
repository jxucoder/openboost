# Independent v1 references

These are small, deliberately slow NumPy oracles for [Sprint 001](../../../v1-sprints/001-scalar-tree-reference.md)
and [Sprint 003](../../../v1-sprints/003-data-classification-reference.md),
not a production OpenBoost implementation or a performance baseline.

`scalar.py` implements weighted half-square loss, unweighted derivatives, explicit
weight application, row sums, Newton leaves and half-scaled quadratic improvement.
`tree.py` exhaustively routes original rows for every candidate. It does **not**
call histogram/prefix-sum kernels, production objectives or the existing trainer.
Best-first rescans all leaves; symmetric intersects feasible candidate conditions
and sums layer gains before selecting. This differs structurally from optimized code.

`tree.py` inputs are fixed numeric bin matrices `[N,F]`, with nonnegative integer-valued codes
and NaN for missing. This tree reference still consumes numeric bins only.
The maximum observed code remains a candidate so observed-vs-missing splits are
possible. Exact ties order feature, threshold, then missing direction (right before
left). Children require positive weighted curvature and the configured minimum;
cohort information is independent of training weights. Empty/illegal candidates
do not split. Default lambda=1, depth=2, eta=.1 and two rounds follow the task cards.

The tests use hand calculations, fixed counterexamples, integer-weight replication,
two-round traces and an isolated process that blocks every `openboost` import.

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache OPENBOOST_BACKEND=cpu uv run --no-sync pytest tests/v1 --confcutdir=tests/v1 -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check tests/v1
```

`data.py` independently fits numeric cuts by direct order-statistic interpolation
and counts crossed cuts by row. Fitted column states are tuples. Missing has a
separate mask; its placeholder code is never a regular value. Category dictionaries
use sorted homogeneous strings or integers (no bool/mixed/floating tokens); unknown
values route as missing. One-vs-rest uses equality, never ordinal thresholds.
An all-missing fitted column has no cuts; validation cannot add cuts. These are
column semantics, not a PreparedData implementation, content hash or row-ID binder.

`classification.py` keeps class schema separate from feature dictionaries: unknown
or missing labels fail, and training schema needs at least two classes. Binary
base requires both classes and an explicit representable clipping probability.
Signed-margin binary loss avoids cancellation; derivatives remain unweighted.
Softmax returns both the exact full Hessian and the explicitly named diagonal
upper bound. Tests check calculus, class permutations, integer-weight replication,
and two-round joint-snapshot updates through the independent tree reference.

Remaining F0.2 work includes row alignment/identity, complete categorical grow,
other typed targets/objective families, quantile/vector leaves and transaction/run
fixtures with their independent tests. New production parity, persistence, CUDA and real task evaluation are pending.
