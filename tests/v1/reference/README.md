# Independent v1 references

These are small, deliberately slow NumPy oracles for [Sprint 001](../../../v1-sprints/001-scalar-tree-reference.md),
not a production OpenBoost implementation or a performance baseline.

`scalar.py` implements weighted half-square loss, unweighted derivatives, explicit
weight application, row sums, Newton leaves and half-scaled quadratic improvement.
`tree.py` exhaustively routes original rows for every candidate. It does **not**
call histogram/prefix-sum kernels, production objectives or the existing trainer.
Best-first rescans all leaves; symmetric intersects feasible candidate conditions
and sums layer gains before selecting. This differs structurally from optimized code.

Inputs are fixed numeric bin matrices `[N,F]`, with nonnegative integer-valued codes
and NaN for missing. No raw-feature bin fitting or categorical encoding is claimed.
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

Remaining F0.2 work includes typed/class/categorical data, the other objective
families, quantile/vector leaves, transaction/run fixtures and their independent
tests. New production parity, persistence, CUDA and real task evaluation are pending.
