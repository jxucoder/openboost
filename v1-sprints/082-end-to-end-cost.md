# Sprint 082: Formal end-to-end cost evaluation

Status: planned. Mapping: B12–B14 / F3 / A13 / E4.
Depends on: 080–081 correctness and selected E3-qualified workloads from 071–076.
Shared evidence/closure rules: [roadmap](roadmap-after-063.md).

## Outcome and first check

Determine whether the verified foundation meets the frozen GPU, batching and CPU
inference cost gates at matched quality. First verify the judge rejects a speed
result with regressed quality, missing failed trials, an unqualified baseline,
profiled official timing, changed workload or skipped GPU evidence.

## Work

- Freeze the exact E4 matrix/environment before timing: small startup/single-row,
  two medium/large real sources with one at least 100k rows, K>1, M=1/8/32.
  Use the preregistered T4 environment; other feasibility hardware stays separate.
- Measure one first fit and three warm fits over at least three seeds, matching
  hardware/threads/cache policy. No profiler or memory-sampling thread during
  official timing. Record memory in separate instrumented runs and label its scope.
- Compare public composition, optimized same algorithm and qualifying external
  GPU baselines. Include preparation, objective work, transfers, JIT, validation,
  rejected trials, export and prediction. Retain current P7's separate original
  workload/result and 1.2 threshold; the new gate cannot overwrite it.
- Measure fresh CPU import/load/predict plus warm single-row and batch inference,
  declared custom dependencies, repeats and timer correction. Include total real
  model-selection and full-set costs, not only the fastest kernel or fit.

## Acceptance and reflection

Apply the unchanged E4 conditions individually:

- Both medium/large standard warm-fit medians are at most 2x the fastest qualifying
  external GPU baseline; public/same-algorithm optimized ratio is at most 1.25.
- M=1 overhead is at most 10%. At least one of M=8/32 reduces full-set time by
  at least 20%; the other and every remaining required ensemble are at most 10%
  slower than the sequential shared-preparation reference. Quality/state and caps pass.
- Standard CPU inference median ratio is at most 2x the fastest qualifying CPU
  baseline, with all startup/load and warm measurements reported.

Commit raw repeated observations, independent judgments and failures. If a gate
fails, distinguish quality, residency, algorithm work and redundant overhead;
retain experimental status and create a focused response sprint. Do not relax
budgets or claim an end-to-end win from a microbenchmark. Passing E4 is not E7.

## Results

Not run. No current v1 GPU cost claim exists.
