# Sprint 053: Current A5 temporal quantile worker

Parent: f9bf01f. Status: complete; all five bounded real-data origins pass.

## Plan and acceptance

1. Reproduce the missing A5 worker contract with weighted direct-recipe parity.
2. Compose independent public scalar quantile recipes at frozen levels 0.1/0.5/0.9,
   retaining per-level stopping and final/best selection, explicit persistence
   metadata and ordered raw output. Reject malformed level/model schemas.
3. Verify fresh-process persistence, weighted pinball scores, unsupported input
   rejection and all five frozen calendar-only Bike rolling origins at unchanged
   90-second fit/30-second replay caps. Retain failures and raw artifacts.
4. Run regression/lint/docs, record reflection and learning, commit locally.

No quantile sorting, joint stopping, test scoring or quality acceptance claim.
Calendar availability, source identity and whole-date split checks remain those
of the frozen exporter. A5 quality/search and all other required rows remain open.

## Results and reflection

Six new A5 tests initially failed because the current worker rejected A5. The
adapter now composes the existing public scalar recipes, without core changes.
Weighted direct-recipe parity covers final and best-validation selection; fresh
processes reproduce all columns and source IDs exactly. Malformed level order,
model count, output width and feature schema are rejected. A crossing fixture
confirms raw predictions are never sorted.

All five frozen calendar-only Bike origins pass bounded integration and replay.
Independent recomputation matches each selected validation pinball score; no
crossings occurred in these four-round runs. Source/output hashes match the
[raw evidence](../benchmarks/v1/evidence/quantile-053/README.md).

This adds evidence that routed scalar leaves compose into a real multi-quantile
workflow without a new foundation primitive. Independent stopping is explicit;
a future coupled/noncrossing algorithm must declare different semantics rather
than silently changing this recipe. The result does not establish A5 quality,
calibration, search acceptance, CUDA or reduced external author effort.

Next connect remaining real-data adapters, starting with A7 count/exposure
semantics, then A8/A9/A10/A12 and unresolved A4 integration, while retaining real
search and D5 checks in the execution map. All required applications remain in
scope. See [learning](../learnings/2026-09-06-v1-quantile-worker.md).

Closure: 858 CPU tests passed; Ruff, strict MkDocs and diff whitespace checks
passed. No foundation production files changed. Nothing pushed.
