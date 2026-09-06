# Independent v1 references

These are small, deliberately slow NumPy oracles for [Sprint 001](../../../v1-sprints/001-scalar-tree-reference.md)
[Sprint 003](../../../v1-sprints/003-data-classification-reference.md),
[Sprint 004](../../../v1-sprints/004-ranking-quantile-vector-reference.md),
and [Sprint 005](../../../v1-sprints/005-positive-aft-reference.md),
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

`ranking.py` enumerates all strict-relevance pairs within each query. Query loss
is the mean over eligible pair count, scaled by query weight; explicit pair weights
scale the numerator, not the count. Row weights fail. Lambda weights are frozen
absolute swap deltas from the current NDCG ranking, with stable row-ID ties; the
returned loss is a surrogate, not NDCG. Zero-IDCG query NDCG is one.

`quantile.py` supplies pinball loss and pseudo split fields, then replaces terminal
leaves using their original routed residuals/weights. Its weighted quantile uses
the left endpoint convention and ignores zero weights. A ties fixture shows that
a correct leaf solver can coexist with no profitable pseudo split.

`vector.py` fits a shared stump by summing output split gains; optional projected
gradients and diagonal projected curvatures select topology while leaves always
use full original output statistics. It is a stump probe, not full vector growth.
Target scaling uses train-only unweighted population mean/std and constant flags.

`positive.py` defines Poisson exposure offsets, Gamma log-mean and fixed-power
Tweedie geometry with explicit support checks. Returned derivatives are unweighted.
Poisson's all-zero base needs an explicit minimum rate; its prediction requires
exposure and distinguishes rate/count. A tiny policy join separates positive-paid
count/mean from raw ClaimNb and reports inconsistent or excluded records.

`survival.py` accepts exact events and right-censored log-normal intervals only.
It separates event density from survival probability, keeps a stable log-tail and
inverse Mills curvature, and exposes median/mean/survival/quantile mathematics.
The far-tail continued fraction is checked against independent quadrature. These
are numerical fixtures, not clinical evaluation or persistence implementation.

Remaining F0.2 work includes row alignment/identity, complete categorical/vector grow,
other typed targets/objective families and transaction/run
fixtures with their independent tests. New production parity, persistence, CUDA and real task evaluation are pending.
