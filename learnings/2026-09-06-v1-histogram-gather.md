# 2026-09-06: Reuse selected statistics across histogram features

## Context

Parent 215d85f. Sprint 050 identified histogram aggregation as a major full-input
CPU cost. The existing loop repeatedly selected identical statistic rows for each
feature. Sprint 051 removed a separate repeated row-hashing cost.

## Decision or Result

Gather selected rows once, preserve the original C-order parent sum, then retain
contiguous statistic columns for all feature bincount calls. Each bin receives
identical weights in identical row order. This is local scratch, not a persistent
cache, new public abstraction or shared-run fusion.

## Changes

- `ops.histogram`: reuse selected columns without changing statistic semantics.
- Exact legacy-formula tests cover mixed/missing features, vector weighted fields,
  independent fields, cancellation, and full/reordered/empty row selections.
- Frozen-packet replay harness verifies input hashes and fresh model inference.
- [Sprint 052](../v1-sprints/052-histogram-gather.md) records full-input outcomes.

## Verification

Use `UV_CACHE_DIR=/tmp/openboost-research-uv-cache`.

- All three new cases failed the buffer-reuse assertion before the change.
- `uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' -q --tb=short`:
  851 passed, including independent candidate/routing/tree and recipe checks.
- Fold zero has byte-identical model persistence and exact prediction arrays
  against Sprint 051. Source hashes match the evaluated implementation.
- Full frozen-fold evidence: [histogram-052](../benchmarks/v1/evidence/histogram-052/README.md).

## Failed Attempts

No mathematical failure was reproduced. Repeated gathering was redundant work.
Preserving the parent reduction layout avoids introducing floating-point changes
while optimizing the per-feature aggregation path.

## Risks and Follow-ups

The contiguous buffer retains approximately rows times fields times eight bytes;
transpose construction can transiently retain two such arrays. Peak process
memory was not measured. No memory improvement or stable speed ratio is claimed.
Full quality searches, remaining adapters, author probes and CUDA remain open.

## Commits

- This histogram-gather slice; parent 215d85f.

Final verification: all five folds pass (67.8–70.3 seconds), exact fresh replay on
each, and all source/output hashes match. Focused tests pass again (3), Ruff and
strict MkDocs pass. These outcomes close the bounded integration failure only.
