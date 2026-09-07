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

## Verification

Independent original-row references and public CPU agreement precede implementation.
GPU validation remains not_run until a concrete package and new allowance exist.

The frozen reference suite passes 31 cases, covering ten fixtures and minima
0/1/2 plus hand-checked D2 and tie winners. CPU routing preserves unsorted row
order. The D2 best unconstrained gain is 12 at threshold 0; its best feasible
gain is 20/3 at threshold 1. Lint passes. Accessing DeviceOperations.candidates
fails with AttributeError before implementation. No device run is consumed.

## Failed Attempts

The candidate API is absent at the starting revision; no remote experiment is run.

## Risks and Follow-ups

Distinguish padded slots from actual candidates; preserve original row order and
identity. Validate schemas and buffer/batch ownership before dispatch. Counts and
routes are exact; the original E1 tolerances and mathematical failures remain.
No CPU search expansion, author-cost claim or training implementation is included.

## Commits

Commit verified local slices separately. Do not push.
