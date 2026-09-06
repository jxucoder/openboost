# Sprint 008: Exact D1/D3/D4 author-task references

Starting revision: `6ae1960`. Status: sprint complete; F0.2 ongoing.
B01/F0.2 author development references; not an E5 author evaluation.

## Plan and acceptance

1. D1: tau=.8 expectile, weighted base, positive/negative/zero residuals, tau=.5 reduction and two rounds.
2. D3: minimize sum(w*pinball)+lambda*(v-anchor)^2/2 by residual breakpoints and interval stationary
   points; compare independent subgradients/optimality, lambda/anchor and two-round routed leaves.
   Do not normalize the weight sum.
3. D4: ordered Normal parameters; alpha=.1*.5^j, j=0..5. Commit only finite strict decrease.
   Reverse/NaN trials reject all six; parameter2 reads accepted or unchanged state. Reuse immutable Update.
4. Full regression/lint, acceptance ledger, reflection and commit.

Public API extensions, real author time, persistence and full runtime/best/RNG integration remain
later work. Solving development tasks provides oracles, not E5 success.

## Results and verification

Added author.py, 26 tests and import isolation. **255 passed, no skips**; Ruff passed.
Local macOS CPU/Python 3.12.12/NumPy 2.3.5.

- D1: residual-sign gradient/Hessian; at r=0 choose h=2*tau explicitly, not a unique classical
  second derivative. tau=.5 reduces to half-square. For[0,2], tau=.8 base=1.6; weights[3,1] give8/7.
  Independently enumerate fixed-weight stationary points between targets. Check both-sign finite
  differences, zero-weight outliers, replication, constants, two-round predictions and hand leaves.
- D3: sum(w*pinball) is unnormalized. Enumerate all residual breakpoints and stationary points of
  every open interval, including tails. [0,2,10]/[1,3,1]/q=.5/anchor0 gives v=1.5 at lambda1,
  .15 at10, and 2 at.1. Independent subgradients contain0; nine q/anchor grids check global minima.
  Replication matches weighting; scaling only weights changes the solution, scaling lambda too preserves it.
- Routed D3 trees replace terminal leaves only, rereading original residuals/weights. High leaves
  are8/7.2, low leaves-2/-1.8; raw2=[1.62,1.62,3.52]. Test distinction from ordinary quantiles separately.
- D4 updates0→1, each with at most six steps(.1,.05,.025,.0125,.00625,.003125). Reversed/NaN
  directions reject all six without terms/raw/loss residue. Parameter2 reads accepted state after
  success or original state after rejection, across two rounds. Input mutation cannot alter old
  snapshots; deterministic fixtures do not consume global RNG. Full best/per-run RNG remains integration work.

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py
```

Initial missing-module failure; first15 cases passed after implementation, then boundary/weight
checks completed. No public author extensions, competitor-cost experiments, real tasks, CUDA or
persistence ran. These are not E5 results.

## Reflection: Converging after three implementation commits

D1/D3/D4 now have falsifiable references rather than vague custom objective/leaf/update claims.
Expectile changes geometry, penalized quantiles can optimize between breakpoints, and ordered
policies depend on accepted state. They reuse trees without sharing incorrect mathematical assumptions.
F1 should expose objective, leaf solver and update-policy replacement; F2 measures actual change cost.

The last three sprints filled direction/run/development-task mathematics, not production.
Only tests/v1/reference changed; F0.3 data/budgets/judge are not frozen. Converge on the ledger,
without expanding development tasks or adding ungated references. Next: full categorical/multilevel
vector growth, raw transforms, offset/two-stage compositions, then F0.2 exit and F0.3. F1–F5 and
all A1–A13 remain required; do not mark E-gates green early.

## Commits

- This slice: `test: add exact author mutation references for v1`.
