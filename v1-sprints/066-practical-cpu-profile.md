# Sprint 066: Practical CPU cost profile

Status: in progress; numerical-worker resource checks pass, profiling fits not started.
Mapping: N2a / F1 / C4 / E1 and diagnostic cost evidence.
Depends on: [065](065-installed-run-isolation.md), completed at `e2b6c4c`.
Shared evidence/closure rules: [roadmap](roadmap-after-063.md).

## Outcome and first check

Determine which repeated work and retained state limit realistic CPU execution
before changing the runtime. First verify that the harness actually terminates
an over-time and over-memory child and preserves its partial/error records.
A declared RAM field is not enforcement. Use a supported enforced environment
if the local host cannot provide the limit; do not label an advisory limit enforced.

## Work

- Freeze Housing fold zero without test labels: first 8,192 training and 1,024
  validation rows in frozen order, retaining and hashing their IDs. Reject absent
  subset sizes rather than silently shrinking them.
- Squared and Normal: 4/32/128 rounds, 32 bins, depth two, learning rate 0.1,
  regularization 1. Also run each at 32 rounds with 2,048 training rows: eight
  fixed-step cases. Add separate tiny deterministic rejection/backtracking fixtures.
- Enforce 120 seconds per case, two CPU threads and 8 GiB RAM. Stop further
  expensive cases after the first resource failure pending diagnosis; retain it.
- Separate uninstrumented end-to-end/peak-RSS measurements from instrumented
  stage and call-count profiles. Include preparation, prediction and diagnostics.
  Compare the small counting fixture from Sprint 063 without treating it as timing.

## Acceptance and reflection

Every attempted case has hashes, environment, cap enforcement, timing scope,
status and full-replay checks. Unattempted cases remain not_run with the stop
reason. A diagnosed resource failure is a valid diagnostic outcome, not a speed
pass. Record whether tree replay, encoding, hashing, histogram work or trace
retention dominates, and which measured cause [067](067-incremental-runtime.md)
will address. Do not change formal search budgets or implement an optimization
in this measurement sprint.

## Results

The [resource preflight](../benchmarks/v1/evidence/resource-preflight-066/README.md)
ran on one Modal CPU container. It verified child allocation rejection under
128-MiB RLIMIT_AS, process-group timeout killing, retained partial logs and successful
normal execution. Both cgroup inspection paths are unavailable in the gVisor
environment, so the complete preflight remains failed/incomplete. Requested
container caps were not disproven; this mechanism could not verify them.

The local macOS RLIMIT_AS attempt was rejected before allocation, so local advisory
configuration was not used as an enforced-budget result. No Housing case ran.
All eight diagnostic cases remain not_run and no runtime optimization has started.

### Reflection and next bounded slice

This is the requested retrospective checkpoint after three verified implementation/
evidence commits: 064 resolves a concrete public-boundary restriction; 065 verifies
installed semantics without more core edits; 066 exposes an instrumentation gap.
The foundation hypothesis now has stronger internal conformance evidence, not
author-cost, practical CPU, CUDA or adoption evidence.

Keep the profile-first plan. Next verify the actual 8-GiB per-worker address-space
ceiling and peak-RSS accounting on the same Modal environment, explicitly separating
address space, resident memory and requested container limits. Preserve this failed
inspection record. A more conservative process ceiling must be declared, not
silently equated with an RSS measurement or changed formal search budget. Confirm
two-thread numerical execution before freezing the eight cases and launching fits.

Only after resource checks pass should the diagnostic profile determine the remedy
in 067. No additional objective family, GPU expansion or second runtime is justified
by the current findings. See the [learning](../learnings/2026-09-06-v1-resource-preflight.md).

### Numerical-worker follow-up

The [new worker preflight](../benchmarks/v1/evidence/worker-resources-066/README.md)
passes nine checks. Exact 8-GiB RLIMIT_AS limits reject a 9-GiB virtual mapping;
OpenBLAS reports two threads; guest ru_maxrss responds to a touched 64-MiB
allocation and retains the peak after release. VmHWM and host cgroups remain
unavailable. Prior failed inspection evidence is preserved.

The diagnostic will use this conservative per-process address ceiling, explicitly
separate from requested container RAM and measured resident memory. It runs no
spawned training workers; numerical threads share the bounded address space.
This adds a stricter declared constraint without changing any formal search gate.
Next freeze input prefixes/case order and rerun these checks in the profiling
image before fits. See the [follow-up learning](../learnings/2026-09-06-v1-worker-resources.md).
