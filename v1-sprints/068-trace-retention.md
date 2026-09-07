# Sprint 068: Bounded diagnostic retention

Status: implementation in progress; source equivalence passes, installed/practical evidence pending. Mapping: N2b / C4–C5 / E1 and practical resource evidence.
Depends on: [067](067-incremental-runtime.md).
Shared evidence/closure rules: [roadmap](roadmap-after-063.md).

## Outcome and first failing check

Long runs can retain useful decisions and diagnostics without retaining every
round's full training arrays. First measure unique retained arrays on the pinned
counting fixture and fail a summary-mode bound that the old full trace exceeds.
Do not mistake logical retained bytes for process RSS or a memory leak.

## Work

- Define explicit summary and full diagnostic retention. Summary keeps round
  decisions/scalars plus current run state; full arrays remain available for oracles.
- Preserve one outer-round trace observation and the structural result contract
  from 064. Avoid requiring ordered commit count to equal outer-round count.
- Document retained data, external diagnostic payload ownership and selected/best
  snapshots. Do not obtain savings by disabling validation or removing evidence.
- Run the matched 066 diagnostic after the change, with identical caps, inputs,
  environment and separate instrumentation. Keep old failure records intact.

## Acceptance and reflection

Summary mode retains O(N*K) run arrays plus O(T) scalar history and the model's
own storage, without O(T*N*K) built-in diagnostic arrays. Full and summary modes
agree on accepted/best state, stop reason, rejection behavior and final inference
for scalar, vector, joint and ordered paths. Test external result compatibility
and installed scheduling after any trace-contract change.

Record both logical retention and measured peak RSS, time and replay; missing
measurements remain missing. Decide whether the CPU is ready for the exact full
search preflight, or record the next measured blocker. Do not claim formal E4.

## Results

Not run. This card proposes a retention boundary, not a measured memory reduction.

### Construction plan and first counterexample

Measure full logical arrays, implement immediate per-round summary retention,
verify all recipe/state/stop contracts and installed author-owned ordered payloads,
commit, then rerun the fixed CPU diagnostic with summary retention and reflect.
The pinned eight-round squared/Normal counterexample initially fails because
`retention` is unsupported. Full traces exceed the summary array bound.

All twelve built-ins now have explicit opt-in summary mode; full remains default.
Summary retains scalar decisions/losses/trials and per-output MSE, dropping sample
arrays immediately. External ordered recipes explicitly summarize their own state
payloads; the structural result contract is unchanged. Tests cover all twelve
families at normal, stopped and zero budgets, ordered updates, failed backtracking,
immutable payload restrictions and final/best/stop identity equality.

### Preregistered retention comparison

Use the unchanged eight frozen cases in one CPU container with the same wheel,
full then summary for each case. This explicit amendment permits at most sixteen
uninstrumented fits and two summary profiles after all pairs pass. Keep two threads,
8-GiB per-worker address ceiling, 120-second per-worker deadline, 60-second soft
profile deadline and 1500-second aggregate function ceiling. Stop after a failure;
retain missing measurements. No tolerance or algorithm settings change.

Compare exact final predictions and model files within each pair, plus retained
array bytes, guest peak RSS and timing. Summary still records every completed
round. One ordered pair per case supports scoped observations, not a statistical
speed or memory guarantee. Same-container pairing avoids Sprint 067's architecture
confound. No old baseline wheel or unmeasured search budget expansion is involved.
