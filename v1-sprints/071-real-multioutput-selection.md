# Sprint 071: Real A6/A13 selection and release

Status: planned. Mapping: N4 / B11–B14 / A6, A13 / R8–R9 / E3 and CPU selection cost.
Depends on: [068](068-trace-retention.md), [070](070-coverage-and-judging.md).
Shared evidence/closure rules: [roadmap](roadmap-after-063.md).

## Outcome and first check

Complete one real, full-budget train → validation selection → sealed test release
workflow. First attempt to bind a selected model to another fold's target scale
or selection receipt; reject it before evaluating the real selected models.

## Work

- Use all five frozen Parkinsons subject-grouped folds, train-only preprocessing
  and verified target scales. Include independent trees and shared vector topology
  with the frozen, semantically appropriate comparison methods.
- Preflight the exact 16 configurations per method and existing 300/1000-round
  search spaces. Respect the frozen per-trial 1,800-second, 8,192-MiB, two-thread
  limits and no retries. This is not the 066 diagnostic or four-round adapter smoke.
- Estimate all jobs and maximum compute before launch; preserve failed trials and
  stop blind expansion after a resource preflight failure. No posthoc budget shrink.
- Select configuration and comparator on validation, then release selected test
  predictions in original units. Independently recompute each target's errors and
  standardized aggregate, retaining scale and model hashes and paired fold results.
- Exercise real CPU M=1/8/32 schedules against independent same-ID execution and
  retain selection, stopping, RNG, failure and full-set cost. CPU scheduling remains
  sequential; it does not establish CUDA batching or E4 speed.

## Acceptance and reflection

Selected inference replays in a fresh process with correct inverse scaling, no
subject/target leakage and a complete selection receipt. A6 must pass every
per-target gate and the declared standardized report; A13 inherits selected-task
quality and leakage checks. Apply E3 median/worst-fold thresholds individually,
not only an aggregate across targets. Include all search failures in cost.

The sprint may record a valid quality/resource failure; that leaves the relevant
gate open. Once the pipeline is proved, reuse it in 072–076, retaining each schema's
semantics. Reflect on total selected-model cost and which foundation component
was reused. Do not turn another short fit into a selected-quality claim.

## Results

Not run. Existing synthetic search and real four-round validation remain earlier evidence.
