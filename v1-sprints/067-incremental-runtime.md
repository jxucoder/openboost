# Sprint 067: Incremental transaction execution

Status: ready after partial-profile diagnosis; implementation not started. Mapping: N2b / B05 / C4–C5 / E1.
Depends on: [066](066-practical-cpu-profile.md) diagnosis justifying the change.
Shared evidence/closure rules: [roadmap](roadmap-after-063.md).

## Outcome and first failing check

Training evaluates new terms without replaying the entire accepted ensemble on
every proposal. First turn Sprint 063's fixed-step counting counterexample into
a focused test for linear new-term work, alongside independently computed full
model predictions. Reproduce the old quadratic behavior before editing.

## Work

- Bind cached training/validation raw state to immutable model, data, binning and
  layout identity. Reuse fitted encodings without allowing stale prepared inputs.
- Compute candidate deltas from new terms, keep rejection residue out of accepted
  state, and advance caches atomically on acceptance. Preserve best-model snapshots.
- Keep full-model replay independent as a verification/export path; external
  callers cannot provide arbitrary trusted prediction caches.
- Make the smallest runtime change supported by 066. Do not add a general cache
  framework, second trainer or trace-retention redesign in this sprint.

## Acceptance and reflection

- Fixed-step tree replay grows linearly with newly added terms; final raw state
  agrees with full replay. Rerun the same bounded practical cases with matched
  environment; report improvements, regressions and resource failures separately.
- Forced rejection, failed and stale proposals, zero/rejected rounds, ordered and
  joint updates, offsets, best/stop and independent run ownership remain correct.
- Numeric, missing, categorical and vector predictions round-trip through fresh
  inference. Changed data/binning/layout cannot reuse invalid cached state.
- No tolerance or acceptance-rule change may conceal an optimization difference.
  If exact CPU replay changes through summation order, resolve it explicitly against
  the existing correctness protocol before accepting the change.

Reflect on whether the new state boundary remains usable by external recipes.
If the fix requires a broad architecture change, stop expansion and revise this
card with the counterexample. Next: [068](068-trace-retention.md).

## Results

Not implemented. Sprint 066's partial 60-second Housing profile finds tree
prediction at 75.3% of fit time, with nested re-encoding at 67.5%. Address accepted
ensemble replay and repeated encoding together through the smallest justified
state boundary. Preserve exact term-addition order; avoid summing a combined delta
that changes floating-point results. The baseline sweep has six passing cases,
one preempted case and one not_run case. Comparisons must retain those gaps;
no complete long-round baseline or speed claim exists.
