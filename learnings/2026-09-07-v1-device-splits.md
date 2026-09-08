# 2026-09-07: Public device scores and feasibility masks

## Context

The user continued after 078-A's real T4 acceptance. Both prior GPU allocations
are consumed; complete local construction and a concrete run package before
requesting more hardware verification.

## Decision or Result

Separate candidate statistics, scalar scores, feasibility masks, choice, routes
and leaves. D2 composes named independent minima and ordinary legality using public
device operations. Preserve the CPU active-bin universe through an explicit padded
batch mask. The operations do not imply arbitrary custom-kernel registration.

## Changes

- [087](../v1-sprints/087-cuda-split-operations.md) records the retrospective decision,
  public contract, fixed cases, acceptance and proposed unapproved run bounds.
- DeviceCandidates, DeviceScores, DeviceMask and DeviceSplit retain exact registered
  identities. Operations compose resident named statistics through scoring and
  routing. Choice/child-size transfers have separate decision-byte accounting.
- Scalar validation rejects unsupported schemas, negative original information or
  curvature, nonfinite results and invalid parameters, including overflowing Python
  integers and complex values. Partial output allocations roll back on failure.
- Fifty-five real-device tests are written, including full candidate intermediates
  and D2's changed winner. They are not executed on hardware yet.

## Verification

Independent original-row references and public CPU agreement precede implementation.
Before the approved run, GPU validation remained not_run until a concrete package
and new allowance existed. The completed real-device result is recorded below.

The frozen reference suite passes 31 cases, covering ten fixtures and minima
0/1/2 plus hand-checked D2 and tie winners. CPU routing preserves unsorted row
order. The D2 best unconstrained gain is 12 at threshold 0; its best feasible
gain is 20/3 at threshold 1. Lint passes. Accessing DeviceOperations.candidates
fails with AttributeError before implementation. No device run is consumed.

After implementation, full CPU regression passes 1185 cases with one Linux-only
skip. This includes the 31 new reference cases and eight earlier run-2 judge cases.
Production and changed-test lint pass; all 55 new GPU tests collect locally.
Collection is not kernel compilation or correctness validation. There is no new
GPU artifact or author-independence evidence.

The run-3 package freezes 88 real-device cases and 38 source files plus the
protocol's dispatch hash. Seventeen local judge/manifest tests pass (nine new,
eight existing), including actual collection equality, source drift, missing
installed/snapshot/version evidence, pending/consumed authorization and output
reuse. Documentation, lint and offline wheel/sdist builds pass; the wheel's 23
production modules match the frozen source bytes. The pending CLI exits before
Modal import or output creation. No GPU allowance is consumed by these checks.

The user replied "Continue" to the concrete additional T4 approval request. Record
that allowance in the protocol and commit before dispatch. All 17 local manifest
checks pass again, verifying unchanged frozen sources and all 88 cases; diff
whitespace checks pass. The limits and zero-retry policy remain fixed.

## Failed Attempts

The candidate API was absent at the starting revision; its first local probe failed.

At `043861a`, automatic approval review rejected creation of the Modal launcher
process because source/test/metadata transfer to Modal needed explicit user
authorization beyond the compute request. Do not retry through another route.
The launcher log and fixed output directory are absent; the 39-file source closure
still matches its freeze. No upload or remote invocation occurred and the approved
single-run compute allowance remains unused. This supplies no GPU evidence.
The documentation build and whitespace checks pass for the block record.

The user then explicitly approved uploading the frozen 39-file private source,
test and metadata package to Modal for the approved single T4 run. This resolves
the block without changing the payload scope, cases, limits or launcher route.
All 17 local manifest/judge checks and whitespace checks pass with this recorded
approval; the 38 prefrozen hashes still match and the case matrix remains 88.

## Real-device result and reflection

All 88 frozen T4 cases pass at clean `9ce790e`, with no retry or skipped/missing
case. The [evidence](../benchmarks/v1/evidence/cuda-splits-078/README.md) retains
raw JUnit/logs and the complete source, installed-module, version and artifact
manifest. Re-judging and checking every snapshot hash against its Git revision
passes: 39 files, 23 production modules, 17 versions and three artifact hashes.

D2's independent cohort minima change the winner on CUDA through public mask
composition; names/order/minima vary without private dispatch. Every candidate's
sums/counts/gain/masks, routed rows and Newton leaves meet the frozen oracle.
The one approved invocation is consumed. No additional hardware is authorized.

The loaded runtime remains 12090 and driver API 13000 on T4 driver 580.95.05,
distinct from the CUDA 12.6.3 image tag. Pytest takes 8.26 seconds; the worker takes
9.75 seconds. These are validation durations, with 98 occupancy warnings retained.
Small split fixtures peak at 19968 bytes in the sampled private pool; neither
that measurement nor compact decision transfers establishes training cost/speed.

Updating the README after acceptance exposed a local harness-test assumption:
it compared a consumed freeze to the current README and failed. Completed-run
tests now audit the retained source/case evidence; active freezes still compare
the live tree and actual collection. No frozen hashes, cases or launcher guards
were relaxed. Stored artifact integrity has a regression check. All 18 local
manifest/judge checks and full CPU regression (1195 passed, one Linux-only skip)
pass after this change. Production modules are unchanged from the T4 snapshot.
Production/changed-support lint, whitespace checks, documentation build and offline
wheel/sdist build pass at closure. No extra CUDA run was used for these checks.

The product boundary has advanced from aggregation to a composable split decision.
Resident raw updates, trees, training transactions and saved CPU inference are
the next falsifiable result. Keep 065 isolation and 068 retention/ownership in
that construction contract. Primitive rollback does not imply accepted-state
immutability, and known D2 development work is not independent author evidence.
Pause for the planned retrospective before 078-C construction.

## Risks and Follow-ups

Distinguish padded slots from actual candidates; preserve original row order and
identity. Validate schemas and buffer/batch ownership before dispatch. Counts and
routes are exact; the original E1 tolerances and mathematical failures remain.
No CPU search expansion, author-cost claim or training implementation is included.
The new run request is one T4 invocation, 900-second function/600-second tests,
16-MiB private pools, two requested CPU cores/8192-MiB host capacity, zero retries.
The declared primitive behavior is verified; training and end-to-end device
behavior remain open. Preserve the distinction between runtime versions, driver
support, image labels and package metadata.

## Commits

`0ba39a3` freezes the oracle before implementation; `0bcb52f` adds the operations
and pending real-device cases. `161c02e` freezes the shared executor guards and
run-3 package. `043861a` records compute approval, `a29749b` the initial transfer
block and `9ce790e` the explicit upload approval and executed source revision.
Commit the verified raw evidence, archival checks and retrospective together.
Nothing is pushed.
