# Sprint 028: B09 query-local ranking

Parent: 08e323a. Status: complete for the bounded slice.

## Plan and acceptance

1. Expose query-local pairwise logistic and NDCG-weighted lambda geometry with
   deterministic row-ID ties, explicit query weights and per-query pair normalization.
2. Compose a CPU ranking recipe with existing scalar trees and atomic state.
   Recompute lambda ranks each round; evaluate with query-weighted NDCG.
3. Verify independent geometry, query isolation, multiple rounds, offsets,
   rejected unsupported row weights and score persistence.
4. Run regressions/lint/docs/build and record reflection before a local commit.

This bounded B09 slice does not complete quantile/penalized leaves, real A4
evaluation, pair sampling or CUDA. Explicit pair weights are deferred and not
accepted by this API. The first failing test imports the missing Ranking objective.

## Results and reflection

Delivered Ranking geometry and the fixed-step ranking recipe. Query roles are
bound to Problem identity; non-unit row weights and unknown structure fail.
The recipe recomputes geometry per accepted round and selects best_model by
query-weighted NDCG. Pair dependencies require no new tree or transaction engine.

Twelve focused tests pass: independent pairwise/lambda gradients and curvature,
three-round tree/prediction parity, logistic finite differences, query isolation,
row-ID tie invariance under permutation, offsets, invalid roles, degenerate
relevance and fresh-process score persistence. CPU regression: 642 passed.
Ruff, strict MkDocs, offline sdist/wheel builds and all eleven installed-wheel
documentation examples pass (macOS, Python 3.12.12, NumPy 2.3.5).
Local wheel SHA256:
2c2809558bfe558cf12b8d7476f381e6e02fdaadd0888184fcb8a586ad1ee8e5.

Observation: the scalar foundation accepts pair-reduced fields unchanged.
Evidence: multi-round results agree with independent pair loops and tree growth.
Decision: keep query-specific preparation in a public objective module, and
avoid interpreting query weights as row weights. Lambda pair loss uses moving
weights and is not the validation metric. Fixed steps keep that distinction
explicit; future line search needs a separately declared frozen-trial objective.

This is a CPU correctness slice with quadratic per-query enumeration. Pair weights,
sampling, real A4 folds, scalability and CUDA remain unverified. Quantile and
penalized leaves are the next B09 construction slice and must expose routed
residuals/original weights, not just additive histograms. No application scope
or evaluation gate was removed; F0.3 and F1–F5 remain incomplete.

Status: complete for this bounded slice. Detailed verification:
[learning record](../learnings/2026-09-06-v1-b09-ranking.md).
