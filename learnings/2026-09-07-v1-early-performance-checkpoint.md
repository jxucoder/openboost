# 2026-09-07: Measure practical execution before extending CUDA coverage

## Context

The user asks whether OpenBoost is fast after the successful correctness rerun.
Existing tiny GPU fits use hundreds of dispatches/synchronizations, and the prior
real CPU A6 probe is expensive. Neither establishes competitive GPU speed. The
user explicitly approves an early performance checkpoint on squared and Normal.

## Decision or Result

[Sprint 104](../v1-sprints/104-early-performance-checkpoint.md) moves a bounded
same-host CPU/CUDA measurement ahead of new recipe ports. Use synthetic workloads
at 10,000/100,000 training rows, plus a 1,000-row Normal control. Require complete
measurements and comparable independent held-out metrics before reporting a ratio.
Retain timeouts instead of replacing the CPU reference or reducing its work after
execution. Formal E4, real application value and external baselines remain open.

## Changes

- The measurement child records dataset creation, context/preparation, training,
  export/cleanup, model loading and supported CPU inference. It writes progress
  after each fit so a later timeout preserves completed repetitions.
- CPU uses two fits, CUDA four, with fixed inputs/settings and fresh per-case
  processes. A separate first/warm profile records inclusive compiler, dispatch,
  synchronization and blocking-transfer costs; these overlap and are not official
  timing ratios. GPU children receive separate disk-cache paths.
- A failure-closed quality judge independently recomputes scores from saved raw
  predictions, verifies state/repetition/source identity and withholds ratios
  on incomplete work, profiling, leaked buffers or quality changes beyond the
  declared thresholds. Constructed judge controls are not GPU evidence.
- The existing bounded Modal runner and isolated CPU builder are reused without
  modifying previous frozen sources. Run 10 uploads 46 exact files, requests one
  T4/two CPUs/8192 MiB with 900/600-second function/test limits and zero retries.

## Verification

- Sixteen focused local tests cover actual CPU model replay, independent Normal
  scores, workload identity, malformed/quality-failing judge inputs and a real
  child timeout with retained partial work.
- The exact source closure builds as an offline wheel and collects six GPU test
  IDs in isolation. Local collection executes no CUDA case.
- Full CPU regression passes 1965 tests with one Linux-only skip in 13.96 seconds.
  The sixteen focused checks pass in 0.80 seconds. Production/support Ruff and
  MkDocs pass, with the existing 090 evidence-link warning. Source-freeze and
  approved dispatch guards pass with the fixed output absent.

## Failed Attempts

The [unqualified development calibration](../v1-sprints/104-local-calibration.json)
needs 25.59–26.14 seconds
per twenty-round Normal fit on 1,000 rows (four repetitions). Source inspection
shows per-row Python interval comparisons; this is a hypothesis for cost, not a
profiled attribution. It motivates explicit deadlines and a small Normal control.
No performance ratio or formal benchmark claim uses this development observation.
Its 31 production-source hashes match `c84d566`; the evolving benchmark script
was not frozen, and that provenance limitation is explicit in the retained record.

Call-path inspection catches explicit GPU binning being combined with a nondefault
bin-count argument, which DeviceRun rejects. The harness now lets the supplied
binning own its configuration. This is corrected before source freeze or upload.

## Risks and Follow-ups

Larger public CPU Normal fits may time out, leaving their speed ratios unavailable.
The profile may also exceed its separate cap; preserve that failure. First/warm
differences include more than compilation. CPU prediction of a CUDA-trained model
does not establish general GPU inference throughput. Tiny/missing/weighted fixtures
and internal CPU/CUDA quality are not real-task or external-library evidence.

## Commits

- `c84d566` closes the preceding bounded CUDA comparison coverage.
- This slice freezes the user-approved early performance checkpoint for one run.

## Automatic approval review after the freeze

`8677bba` commits the locally verified packet. The attempted dispatch is rejected
before process creation by automatic approval review. Its reason distinguishes
approval of the performance study from authorization to upload the exact private
repository payload to Modal. No output directory or remote execution exists and
the tenth allowance is not consumed. No alternate upload or indirect retry occurs.

Restore only upload authorization to pending; the study/compute approval remains.
Preserve all 45 frozen source hashes, cases, budgets and the original local
collection report. The next user approval must explicitly cover the 46-file,
approximately 358 kB Modal upload for the bounded single T4 invocation. This is an
automatic-review requirement, not a new mathematical or benchmark gate.
The local recheck confirms all 45 prefrozen hashes remain unchanged, the output
is absent and the dispatch guard rejects pending upload authorization.
