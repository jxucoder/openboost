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
GPU validation remains not_run until a concrete package and new allowance exist.

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

The candidate API is absent at the starting revision; no remote experiment is run.

## Risks and Follow-ups

Distinguish padded slots from actual candidates; preserve original row order and
identity. Validate schemas and buffer/batch ownership before dispatch. Counts and
routes are exact; the original E1 tolerances and mathematical failures remain.
No CPU search expansion, author-cost claim or training implementation is included.
The new run request is one T4 invocation, 900-second function/600-second tests,
16-MiB private pools, two requested CPU cores/8192-MiB host capacity, zero retries.
Actual hardware behavior remains unknown. Preserve the distinction between
runtime versions, driver support, image labels and package metadata.

## Commits

`0ba39a3` freezes the oracle before implementation; `0bcb52f` adds the operations
and pending real-device cases. `161c02e` freezes the shared executor guards and
run-3 package. The approval record is a separate commit. Nothing is pushed.
