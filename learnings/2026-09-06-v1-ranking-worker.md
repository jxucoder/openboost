# 2026-09-06: Query-aware ranking comparator worker

## Context

The synthetic builtin capability probe did not provide an A4 worker with explicit
query identity, group weighting and validation/persistence semantics.

## Decision or Result

Require contiguous query IDs and disjoint training/validation queries. Translate
explicit query weights into each library's native representation; reject ordinary
row weights. Preserve query IDs outside features and score row identity unchanged.

## Changes

- [Ranking validation](../benchmarks/v1/ranking.py) and A4 branches in the
  [baseline worker](../benchmarks/v1/baseline_worker.py).
- [Smoke](../benchmarks/v1/ranking_smoke.py), raw CPU histories and initial failure.
- Five focused group/weight/partition counterexamples.

## Verification

- `uv run --no-sync pytest tests/v1/test_ranking_worker.py -n 0 -q`: five passed.
- `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 build/v1-env/bin/python -m benchmarks.v1.ranking_smoke`:
  XGBoost, LightGBM and CatBoost passed fit, named-NDCG stopping-minimum/maximum
  checks and saved-score replay; see [evidence](../benchmarks/v1/evidence/ranking-cpu.json).
- Existing 30 fixed-round plus 30 stopping CPU cells still passed.
- Full suite after adapter additions: 460 passed, no skips.

## Failed Attempts

The initial test used the last native metric, assuming it was NDCG. CatBoost
also reports PairLogit, invalidating that assumption. The corrected test explicitly
selects the named NDCG series. Original source and error remain committed.

## Risks and Follow-ups

Native ranking optimizers, weighting and gain conventions differ; independent
v1 NDCG determines cross-method quality. Synthetic queries do not replace the
unavailable MSLR dataset/agreement. GPU and complete matrix integration remain open.

## Commits

- This slice: `eval: add query-aware ranking comparator workers`.
