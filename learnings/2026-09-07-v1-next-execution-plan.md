# 2026-09-07: Plan the path from CUDA storage to programmable training

## Context

The user requested planning ahead after verified device storage. The current code
has no device fields, histogram kernels, tree training or accepted-state integration.
Some sprint headers still said no CUDA implementation/run despite the appended
storage evidence. One of the two approved device runs has been consumed.

## Decision or Result

The next product milestone is two-round scalar CUDA training plus an installed
D2 public composition with CPU-readable inference. First verify named fields and
routed histograms, then candidate/route/leaf operations and resident transactions.
The one remaining 900-second allocation verifies the bounded aggregation slice;
it is not an adequate basis for promising all training/device gates will close.
Stop after that run for a retrospective and a concrete further compute decision.

Author accounting/isolation preparation continues independently. The built author
view is not a sandbox or an independent attempt. Preserve cohorts and expose actual
measurement blockers rather than substituting more synthetic accounting artifacts.

## Changes

- [086 plan](../v1-sprints/086-next-execution-plan.md) defines construction slices,
  frozen aggregation fixtures, acceptance, budget and retrospective boundaries.
- Current sprint/main-plan/agent pointers distinguish verified storage from missing
  GPU training and point to the next bounded slice.
- The previous seam audit is explicitly historical; original evidence remains intact.

## Verification

Read production execution/storage tests, public documentation, CPU field/histogram
semantics, the existing device evidence, E1/E4/E5 and the remaining sprint cards.
All 115 local file-link targets across nine changed Markdown files resolve;
anchor fragments were not checked. English-prose and `git diff --check` checks
pass. No source or runtime behavior changes; no new CPU/GPU result, author attempt
or budget use. Runtime tests were not rerun for this documentation-only change.

## Failed Attempts

No implementation or experiment attempted in this planning turn. Documentation
inspection found stale current-status statements; correct the summaries while
preserving the dated historical observations and their raw artifacts.

## Risks and Follow-ups

The public device extension contract must reconcile opaque ownership with real
custom batch operations. Decide that using the D2 consumer before introducing a
private trainer or mutable accepted-array aliases. Pin the actual numba-cuda/CuPy
environment before running kernels; the storage run's loaded CUDA runtime differed
from the image tag. Training, independent benefit, quality, cost and adoption remain
open. Formal scope and thresholds are unchanged.

## Commits

One cohesive planning/status commit follows verification; no push.
