# 2026-09-06: Separate protocol readiness from evaluation completion

## Context

Repeated F0.3 slices added working real-data comparator bindings while the public
foundation remained unimplemented. The user requested an audit of prerequisites.

## Decision or Result

The current written F0.3 gate remains incomplete. Full E0–E7 results are later
phase exits, but adapters, matrices, judges and cohort/environment freezes are
explicit early requirements. An early B03 start is a proposed sequencing amendment,
not an interpretation that F0.3 has passed. Preserve all application scope.

## Changes

- [Sprint 017 audit](../v1-sprints/017-f0-sequencing-audit.md): evidence-backed
  dependency ledger, implementation gaps and bounded B03–B06 overlap proposal.
- Sprint index/current execution record link the audit without changing phase gates.

## Verification

Inspected active phase/build/evaluation plans, actual integrity/quality/selection
paths, source freezes, safe held-out metadata, CPU/CUDA artifacts and the empty
production namespace. Strict MkDocs and whitespace checks passed for this slice.
No tests or benchmarks were rerun because this is a documentation audit.

## Failed Attempts

The earlier verbal recommendation understated explicit F0.3 requirements. The
audit corrects it: later result execution and earlier protocol implementation
are different requirements; a sequencing change must be recorded explicitly.

## Risks and Follow-ups

A6 standardized score/selection-scale binding, global coverage aggregation and
formal runtime access/resource enforcement remain substantive gaps. They cannot
be dismissed as documentation work. The proposed overlap does not authorize a
quality claim or CPU interface freeze, and is not adopted by this audit.

## Commits

- This slice: `docs: audit F0 prerequisites and proposed foundation sequencing`.
