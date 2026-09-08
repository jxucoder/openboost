# Sprint 074: A4 ranking and A5 quantile quality

Status: planned. Mapping: N4–N5 / B09–B14 / A4–A5 / R2–R3 / E1, E3.
Depends on: the verified [071](071-real-multioutput-selection.md) selection pipeline.
A4 additionally needs the source/terms freeze from 070; A5 can proceed independently.
Shared evidence/closure rules: [roadmap](roadmap-after-063.md).

## Outcome and first checks

Verify selected models for structured row dependencies and routed residual leaves.
First reject a ranking split that crosses query IDs and a Bike schema containing
target-derived features; independently verify a weighted pinball/NDCG fixture.

## Work

- A4: close MSLR source/access and official folds before evaluation; connect both
  pairwise and NDCG-weighted lambda recipes to real data and full search/release.
  Preserve query-local pairs, ties, weighting and the declared pair-resource limit.
  Profile an actual limit failure before changing pair construction or sampling.
- A5: all five frozen Bike rolling origins and declared quantiles 0.1/0.5/0.9.
  Exclude casual/registered and unavailable future features; evaluate weighted
  pinball, crossings, calibration and interval coverage with width.
- Use semantically supported baseline paths and full frozen searches. Do not sort
  quantiles after prediction or introduce an unregistered pair sampler to improve
  quality/timing. Distinct methods need separate preregistration and evidence.

## Acceptance and reflection

A4 meets the NDCG@10 median/worst difference gate and all query/pair semantics;
A5 independently meets its pinball gate and reports crossings/calibration/width.
Both retain selected prediction replay and all search failures. An unavailable
A4 source remains open; it does not invalidate completed A5 work or pass ranking.

Reflect on whether public routing/leaf and problem records express these cases
without task-name branches or a separate trainer. Substantial adapter/correctness
fixes become their own verified slices before real evaluation.

## Results

Not run. A4 is currently synthetic adapter evidence; A5 has bounded real validation.
