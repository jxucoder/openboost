# Sprint 067: Incremental transaction execution

Status: complete; exact paired replay, installed isolation and bounded practical sweep pass. Mapping: N2b / B05 / C4–C5 / E1.
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

Initial design basis (implementation/evidence below supersede this status): Sprint 066's partial 60-second Housing profile finds tree
prediction at 75.3% of fit time, with nested re-encoding at 67.5%. Address accepted
ensemble replay and repeated encoding together through the smallest justified
state boundary. Preserve exact term-addition order; avoid summing a combined delta
that changes floating-point results. The baseline sweep has six passing cases,
one preempted case and one not_run case. Comparisons must retain those gaps;
no complete long-round baseline or speed claim exists.

### First implementation slice

Plan: reproduce the count failure; implement owned proposal evaluation and verified
encoding reuse; run transaction/full-regression checks; commit; then repeat the
frozen practical diagnostic and reflect on measured improvement and remaining costs.
The first test failed at 60/120 and 216/432 tree calls for squared/Normal at 4/8
rounds. New-term evaluation reduces these to 8/16 and 16/32 with exact independent
full replay, preserving individual term-addition order. Public `preview_raw`
serves external loops as well as built-in recipes. Runtime metadata/model-envelope
work may still grow with ensemble size; this is not a claim of linear total fit time.

### Same-container comparison amendment

All eight optimized fits and both profiles pass at `e99e89c`. Cross-run inspection
finds a changed OpenBLAS architecture (Haswell baseline, SkylakeX optimized).
Squared models/predictions match exactly; Normal differs by at most 8.9e-16.
Do not attribute this difference or the timing ratio to the runtime alone.

Before interpreting timings, run six paired baseline/candidate cases in one
container, in the original order, baseline first for each pair. Use the retained
hash-verified baseline wheel and current wheel, identical input and dependencies,
two numerical threads, 8-GiB address ceiling and 120-second per-child cap. Maximum
12 uninstrumented fits, no profiles, one 1500-second function, stop at first failure.
This is a separate diagnostic amendment, not a repeated realization of the original
eight-fit budget. Do not run baseline 128-round cases. Verify exact predictions and
model bytes between paired wheels, and report one observation per pair without
statistical or formal gate claims. Preserve the cross-host results as such.

### Closure and reflection

Implementation `e99e89c` passes 957 CPU tests, the installed D1–D4/custom-policy/
ordered M=1/8/32 checks and ten exact fresh inference models. The count artifact
verifies 2*K*T tree evaluations at 4/8/16/32 rounds. All eight frozen practical
fits and both 128-round profiles pass with exact full replay. The profiles use
256/512 tree predictions and three encodings each; tree prediction is below 0.3%
of fit time. Tree construction is now about 77–78% of instrumented fit time.

The same-container comparison at `da0fede` passes all twelve fits. Actual loaded
source hashes distinguish the baseline/candidate wheels. All six paired raw
predictions and model JSON files are exact, resolving the numerical ambiguity
without changing tolerances. At 8192/32, squared fit is 8.136 → 1.783 seconds and
Normal 11.929 → 3.644 seconds (observed 4.56x/3.27x fit ratios, 4.37x/3.17x
end-to-end ratios). One ordered observation per case is not a stable speed claim.
The cross-host results and missing 128-round baseline are retained explicitly.

All evidence and exact commands are in
[incremental-067](../benchmarks/v1/evidence/incremental-067/README.md).
Source/artifact hashes, per-worker loaded sources, independent validation metrics
and paired predictions/model bytes were checked. Ruff, docs and offline packaging
pass; later harness-only amendments pass ten focused profile tests. Raw pstats
text preserves its original trailing blank line despite whitespace lint.

Reflection: the measured repeated-work cause is removed through the public
transaction boundary, which remains usable by external recipes via `preview_raw`.
Full replay stays independent, acceptance/rejection ownership stays explicit and
no broad cache framework or second trainer was introduced. CPU architecture was
an important comparison confound; same-container wheel pairing is preferable for
future correctness/performance attribution. Full trace retention still grows with
rounds and samples, so follow 068 before expanding full searches. Tree construction
is the remaining measured CPU hotspot, not an automatic license for another
optimization sprint. Formal phase/quality/author/CUDA/adoption gates remain open.

This is a retrospective checkpoint after implementation, installed/count evidence,
a comparison amendment and practical evidence. Next: Sprint 068's explicit
summary/full diagnostic retention, preserving all accepted/best/stop semantics.
