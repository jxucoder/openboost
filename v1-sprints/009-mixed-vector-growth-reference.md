# Sprint 009: Mixed features and multilevel vector trees

Starting revision: `a6b3eeb`. Status: sprint complete; F0.2 ongoing.
B01/F0.2, A2/A6/C1–C3 reference integration gaps.

## Plan and acceptance

1. Fit numeric/category transformers on training; trees retain them for raw validation prediction.
2. Independently enumerate original rows for multilevel scalar/vector depthwise/best-first/symmetric
   trees. Categories use one-vs-rest equality, both missing directions; projected splits retain full leaf statistics.
3. Two-round mixed binary and multioutput tasks; compare K=1 against scalar oracle, check row conservation,
   unknown/missing, output permutations, replication, no legal split and immutable snapshots.
4. Full regression/lint, ledger and reflection; offset/two-stage models belong to the next finite slice.

No production API/GPU path. Reference outputs are[N, K]; internal layout is not a production constraint.
Keep the existing structurally different scalar oracle rather than replacing it to obtain self-consistency.

## Results and verification

Added mixed.py, 24 tests and import isolation. **279 passed, no skips**. Training transforms,
full scalar/vector trees and raw prediction connect in the same fixtures.

- Transformer fits training X only, retaining names/kinds/cuts/category dictionaries. Predict reuses
  it; unknown follows missing. Identity binds original contents, row IDs and fitted transform state.
- Growth reroutes/reduces original rows for every candidate. All three policies support multiple
  levels. Category equality and numeric thresholds require no histogram/backend. No leaf-budget
  option exists: this reference stops at max_depth, not a claim that all growth options are covered.
- Candidate gain sums split channels; explicit gP/diag(PᵀHP) projection retains full K leaves.
  Three-policy two-round four-leaf fixtures map one row per leaf: predictions y/2,.95*y/2,
  raw2=.0975*y. K=1 matches scalar on numeric/missing; permutations and replication pass.
- Mixed numeric/category three-level tree yields six single-row leaves and exact vectors at lambda0.
  Symmetric checks common per-level conditions. Original scalar/stump references remain unchanged comparisons.
- Two-round mixed binary joins gradients/trees/raw predictions. Out-of-range numeric, unseen categories
  and missing values do not refit cuts/dictionaries. Middle category m is isolated; unknown follows missing.
- Both missing directions, zero-weight/all-missing/zero-curvature rows, invalid schema/projection/depth,
  input mutation, row conservation and no duplicate routing are covered.

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py
```

Local macOS CPU/Python 3.12.12/NumPy 2.3.5. Missing-module failure first; initial8/11 passed.
Three failures reflected the incorrect expectation below. Preserve the counterexample and use
positive-gain multilevel fixtures. No CUDA, real quality, production API or persistence results.
Transformer is an immutable reference record, not a production layout/serialization format.

## Reflection: Regularization cost of vector splits

Initially first-axis±1/second-axis±3 targets were expected to split again, but all policies stopped
at one level. Second-level first-axis gain is+.5; repeated regularization of constant second-axis
leaves costs-1.5, totaling-1. This follows summed-output gain, not a grow bug. Keep the no-more-split
test and use first-axis±2 for positive second-level gain and full two-round checks; do not lower thresholds.

Categorical/vector references now connect training transforms to raw prediction with independent
scalar/stump comparisons. Those F0.2 gaps are covered, not C1/C2/C3 production conformance.
Next complete offset/two-stage and best/per-run RNG compositions, audit remaining F0.2, then F0.3.
Avoid endlessly redoing completed mathematics in isolated fixtures. All A1–A13 remain required.

## Commits

- This slice: `test: add mixed feature and full vector growth references for v1`.

Precommit review: finite node scores do not guarantee a finite sum of child gains. Add an overflow
counterexample and explicit candidate/layer-total checks so infinity cannot win. Final count: 24 new, 279 passed.
