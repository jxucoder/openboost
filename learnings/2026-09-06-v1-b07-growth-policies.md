# 2026-09-06: Three public numeric growth policies

## Context

B07 requires distinct depthwise, best-first and symmetric growth semantics through
shared operations. The independent reference rescans leaves, allowing a structurally
different heap implementation to be checked without using production as its oracle.

## Decision or Result

Ordinary public policy functions share only local construction scratch and the
existing histogram/candidate/scoring/legality/routing/leaf operations. Best-first
caches unaffected leaf candidates in a heap. Symmetric growth intersects legal
conditions across the layer and sums all associated gains, including negative
individual gains. It never substitutes each node's independently best split.

## Changes

- tree: shared construction scratch, retained depthwise policy, new best_first
  and symmetric functions using the same validated NumericTree artifact.
- Tests: exact independent topology/leaf/prediction comparisons, symmetric negative
  local-gain counterexample, callback checks and complete recipe substitution.
- Public docs explain pure callbacks, common-layer semantics and full-layer budgets.

## Verification

- Initial comparison test failed before implementation with missing public growers.
- `OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run
  --no-sync pytest tests/ -m 'not gpu and not benchmark' --tb=short -q`:
  603 passed, including 22 new growth cases. Python 3.12.12, NumPy 2.3.5, macOS.
- Three policies times five depth/leaf budgets match independent exhaustive
  topology, leaf values and predictions with weighted rows and missing features.
- A symmetric layer accepts gains -1/+3 jointly and refuses a partial-layer
  budget. Replacement feasibility restricts features; doubled custom leaf values
  double output. Candidate scoring occurs once per node/condition.
- Best-first and symmetric three-round squared recipes match independent reference
  predictions. Their ensemble round trips preserve unseen/missing predictions.
- Invalid depth/capacity/nonfinite callback values fail. All previous tree,
  squared, Normal, Formula and run-state regression cases still pass.
- Ruff, strict MkDocs and offline build pass. All code examples across six public
  documentation pages pass under Python -I from an isolated installed wheel.
  Wheel SHA256: 50d76d90710f9945aded1e47f9aa7113e6a4735af33f2f8a9ea96b4497030645.

## Failed Attempts

No mathematical fixture or reference adjustment was required. The initial missing
imports were the intended pre-implementation failure. Formatting was normalized
before regression verification.

## Risks and Follow-ups

Callbacks must be pure with respect to candidate statistics and immutable config;
heap caching intentionally does not support call-count-dependent scoring. Shared
scratch is local per grow invocation, not shared run state. No performance claim
follows from using a heap. All policies remain numeric/scalar CPU implementations.

Next B07 slice: categorical preparation, candidate conditions, routing and artifact
validation through these same operations. Vector/specialized leaves, ordered
Normal, stopping policies, CUDA and complete evaluation remain required. F0.3
and full F1/v1 acceptance remain incomplete; nothing was pushed.

## Commits

- This slice: feat: add best-first and symmetric numeric growth policies.
