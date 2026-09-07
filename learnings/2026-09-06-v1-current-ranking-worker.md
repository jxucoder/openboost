# 2026-09-06: Preserve query weights and source-row ties in current ranking

## Context

Parent ca73a2d. The public CPU ranking recipe and baseline query validator existed,
but the current worker had no A4 path. Real MSLR access/binding remains unresolved.

## Decision or Result

Reuse strict contiguous/disjoint query validation. Map groups to integer public
structure roles and repeat explicit per-query weights across their rows. Preserve
integer source row IDs for NDCG tie behavior; reject ordinary row weights and
ambiguous/overlapping identities. Both pairwise and lambda modes are exposed.

## Changes

- Current worker and inference support A4 raw scores and query-weighted selection.
- Source manifests include the imported query validator.
- Four direct/fresh cases plus seven invalid-input cases; no core changes.
- [Sprint 061](../v1-sprints/061-ranking-worker.md).

## Verification

- Current worker suite: 83 passed, including four new fresh-process replays.
- Full CPU regression: 923 passed. Ruff, strict MkDocs and whitespace pass.
  Commands use uv run --no-sync and UV_CACHE_DIR=/tmp/openboost-research-uv-cache;
  macOS/Python 3.12.12/NumPy 2.3.5.
- Configured build/v1-data has no MSLR/ranking-fold files. No real-data fit claimed.

## Failed Attempts

No data substitution or modified query population was used. The missing real-data
prerequisite remains explicit, rather than counting synthetic parity as A4 quality.

## Risks and Follow-ups

Per-query pair enumeration is quadratic. Real source agreement, data binding,
quality/search, D5 and CPU phase acceptance are open. Next review those finite
prerequisites and the GPU transition; CUDA remains unimplemented.

## Commits

- This current A4 adapter slice; parent ca73a2d. Local only, no push.
