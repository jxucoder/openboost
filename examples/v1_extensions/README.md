# Current v1 development extension wheels

Three repository-authored packages exercise the installed public CPU foundation:

- `ob-cohort-splits`: D2 adds cohort information independent of training weights
  and rejects candidates unless each child has at least one unit per cohort.
  Cohorts are not input features. Existing histograms, growth and scoring are reused.
- `ob-penalized-leaves`: D3 replaces routed leaf solving with an independent
  bisection implementation of weighted pinball plus an anchored quadratic penalty.
  The built-in leaf solver uses a different breakpoint algorithm.
- `ob-ordered-updates`: D4 recomputes Normal/Formula geometry after each accepted
  parameter and applies bounded backtracking through public transactions. See
  [ordered update semantics and limits](ordered_updates/README.md).

These are exploratory development examples, not independent authors, timed agent
comparisons, held-out tasks or adoption evidence. All use public imports and
require no core edits. The historical `examples/extensions/` packages remain separate.

## Reproduce

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python examples/v1_extensions/verify.py /tmp/openboost-extension-check
```

The verifier builds four wheels offline, creates a fresh environment, installs
the wheels and NumPy 2.3.5, and executes copied checks from outside the repository
using Python `-I`. Installed module paths must be under site-packages. It then
uninstalls all three extensions and starts another interpreter to load eight saved
models and verify exact predictions with none of the extensions importable.
The report records source/wheel hashes, revision/dirty state, environment,
commands, three-round outputs and failures. Offline dependencies must already be
cached. Nothing is uploaded or published. Run with Python 3.12 for the recorded
environment; broader platform support has not been checked.

## Public composition

Pass `CohortLearner(problem, information)` as `squared(..., learner=...)`.
Information has shape `[N, cohorts]` and is bound to that problem's identity.
The learner owns tree settings; the recipe rejects conflicting growth settings.

Pass `PenalizedLeaves(q=0.7, penalty=5, anchor=4)` as
`quantile(..., q=0.7, grower=...)`. Keep the recipe and solver quantile aligned
for the D3 task. The plugin owns leaf penalty/anchor and replaces the default
row_leaf callback; recipe penalty/anchor options should remain at their defaults.
The scoring and prediction loss remain unpenalized pinball as declared in D3.

This highlights a usability cost: leaf replacement currently uses a grower
adapter, and the quantile value is supplied in two places. Neither requires loop
copying or private access. Record this friction for later author trials before
adding another core configuration abstraction.

## Evidence and limits

`checks.py` independently enumerates D2 feasible cuts, checks the unconstrained
optimum is rejected, tests all three growth policies and a no-feasible split,
and verifies information survives zero objective weights. D3 uses exhaustive
breakpoint/stationary-point minimization, subgradient containment and penalty
contraction over deterministic weighted fixtures, including duplicate residuals
and zero weights. Three-round execution verifies changed subsequent raw values
and each externally solved leaf against the oracle. Mixed/missing feature models
round-trip after plugin removal.

Source development tests in `tests/v1/test_public_extensions.py` are distinct
from the installed-wheel checks. These results are partial E2/E6 evidence;
D1, complete D5 author evaluation, formal E5, CUDA, real-data quality/cost
and external adoption remain open. Structural result integration now permits
ordered recipes in run_many; installed M=1/8/32 checks exercise mixed recipes,
independent stopping, failure isolation and reorder/regroup/retry equivalence.
D4 reference traces and installed results are
recorded separately from the earlier D2/D3 evidence.
