# Sprint 004: Ranking, quantile and vector-leaf references

Starting revision: `e3d99ad`. Status: sprint complete; F0.2 ongoing.
Scope: B01/F0.2, A4/A5/A6 independent mathematics and two-round probes.

## Plan and acceptance

1. Write hand calculations/counterexamples for query pairs/normalization, left weighted
   quantiles and shared split/vector leaves.
2. Implement NumPy references; verify finite differences, query shifts, round-two residuals,
   K=1 reduction and output permutations.
3. Run regression/lint, reflect and commit while retaining all unfinished F0.2/v1 scope.

- Ranking enumerates strict-relevance pairs within each query, averages by eligible pair count,
  then applies query weight. Pair weight changes the numerator only; generic row weight is rejected.
  Lambda weights freeze current ranking; ties use stable row IDs. NDCG=1 when IDCG=0.
- Quantile leaves use the left weighted quantile of routed residuals, not Newton values.
  Pseudo h=1 is topology-only. Hand, nonsmooth optimality and two-round checks are required.
- A minimal shared vector stump tests summed output gain, full vector leaves and optional linear
  split projection. Compare independent scalar trees. Train-only target scaling uses scale=1 for constants.
- A stump does not establish full vector growth, ranking sampling, real quality, persistence,
  CUDA or a production foundation. E-gates remain unpassed.

## Results and verification

**Bounded sprint complete; F0.2 ongoing.** Three independent modules and 27 tests added,
including subprocess production-import blocking. **122 passed, no skips** in default v1 regression.

- Ranking: one pair with query weight3 has g=(-1.5,1.5), h=(.75,.75); three pairs still average
  to log(2). Ordinary-pair and frozen-lambda-surrogate finite differences pass. Cases cover
  query shifts, gradient conservation, row-ID ties/permutations, zero IDCG, extreme scores,
  no pairs/zero-weight queries, unknown weight keys and row-weight rejection. First tree leaves
  are ±.4; round two recomputes from sigmoid(-.08). A new-ranking lambda-change case is retained.
  No sampling or ordinary NDCG-gradient claim is made.
- Quantile: q=.1/.5/.9 hand values, left ties, positive-weight filtering, replication and pinball
  optimality. Pseudo h=1 is not an exact Hessian. Scalar reference selects topology, then each
  routed leaf solves residual/weight values. High leaves are8 then7.2; raw2=3.52.
- Vector: shared candidate gains sum across outputs, with independently solved K-dimensional
  leaves. Projection uses gP and diag(PᵀHP), summing explicit sketch channels, not a full projected
  Hessian. Identity projection is default. Projection changes topology only, never leaf width.
  Shared fixture chooses feature1; projection to output1 chooses feature0; separate trees choose
  different features. K=1 matches scalar depth1. Permutations, replication, root-only and two-round checks pass.
- TargetScale: unweighted per-column training population mean/std (ddof=0), constant scale=1,
  tuple state. Mutating training inputs does not alter validation transform/inverse. Not a full output artifact.

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py
```

Ruff passed. Local macOS CPU/Python 3.12.12/NumPy 2.3.5/pytest9.0.2. Initial collection lacked the
ranking module. Initial implementation passed11/12; the remaining fixture assumption was wrong.
After retaining that counterexample, correcting the acceptance fixture and adding boundaries, all pass.
CUDA, real data, external methods, serialization, full-depth vector growth and E-gates are unverified.

## Reflection: Counterexamples and closure

Initial quantile fixture y=[0,2,10], w=[1,3,1], base=2 expected a high leaf8; it actually had no
split and update0. Under equality convention 1[y<F]-q, both child pseudo-gradients share a sign,
so default regularization gives no positive gain. Correct residual quantiles do not guarantee
improvement of the pseudo split objective. Retain the no-split case; use w=[2,1,2] for positive-gain
two-round leaf verification. Do not change thresholds or fabricate trees. Real A5 quality needs evaluation.

These cases need different inputs but reuse original-row routing and scalar mathematics:
ranking reduces pairs to rows; quantile leaves revisit residuals; vector split/leaf statistics
can have different dimensions. Preserve objective→stats→topology/leaf boundaries. A shared stump
is only a minimal probe, not full growth/artifacts. Next: positive/count and survival/AFT, then
Normal/Formula, typed identity, state/run and remaining growth checks. F0.3 must freeze evaluation
before F1. All A1–A13 remain required.

## Commits

- This slice: `test: add ranking quantile and vector leaf references for v1`.
