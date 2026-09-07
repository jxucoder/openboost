# 2026-09-06: Hoist immutable row identity out of candidate enumeration

## Context

Parent d1dede6. Sprint 050 measured repeated row hashing as a major cost within
candidate enumeration on the full Covertype input. Each candidate in one call
uses the same immutable Histogram.rows array.

## Decision or Result

Compute the existing digest once per candidates invocation and reuse the identical
string. No global cache, hash format, candidate ordering, statistics, feasibility
or routing changes. Histogram aggregation is intentionally a separate optimization.

## Changes

- ops.candidates: one local invariant digest replaces repeated hashing.
- Two tests with numeric/categorical/missing features, full/subset rows and zero
  weights verify exact digest identity, aggregate conservation and one hash call.
- [Sprint 051](../v1-sprints/051-candidate-row-hash.md).

## Verification

Use UV_CACHE_DIR=/tmp/openboost-research-uv-cache.

- `uv run --no-sync pytest tests/v1/test_candidate_hash.py -q -o addopts=''`:
  both initially failed with 16 calls rather than one; both pass after hoisting.
- `uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' -q --tb=short`:
  848 passed, including independent histogram/tree references and recipe tests.
- Uninstrumented Sprint 049 fold-zero job rerun under the unchanged 90-second
  process cap and one thread, after the regression suite completed: passes in
  87.4 seconds. Exact fresh seven-class predictions/source IDs and normalized
  probabilities verified under a 30-second replay cap.
- Source/raw artifact hashes match [row-hash-051](../benchmarks/v1/evidence/row-hash-051/README.md).
  Ruff and strict MkDocs pass on macOS/Python 3.12.12/NumPy 2.3.5.

## Failed Attempts

The focused test reproduced redundant work without a mathematical failure.
The same row bytes now produce the same digest once; no digest weakening or
identity bypass was used.

## Risks and Follow-ups

One local rerun is not a fair speed comparison or full five-fold A3 acceptance.
Histogram aggregation remains the other measured cost. Full required quality,
D5, CUDA and adoption gates remain open. No push or publication.

## Commits

- This invariant-hash slice; parent d1dede6.
