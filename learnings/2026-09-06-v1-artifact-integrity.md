# 2026-09-06: Artifact integrity before evaluation gates

## Context

F0.2 references are complete. F0.3 needs to reject incomplete or inconsistent
benchmark evidence before interpreting model quality or algorithm authoring cost.

## Decision or Result

Implement an offline integrity judge with a deliberately separate `integrity_pass`
field and empty `gate_results`. It checks the declared matrix, cache identities,
worker/backend status and artifact bytes; it does not trust a producer's pass claim
as an independently evaluated quality result.

## Changes

- [Sprint 011](../v1-sprints/011-artifact-integrity-judge.md): plan, tests and reflection.
- [Integrity schema and CLI](../benchmarks/v1/README.md): strict JSON, complete cell
  records, file hashes, finite numeric predictions, visible optional failures.
- Full manifest/cell cache identity includes code, environment, data/split,
  preprocessing, config, seed and protocol; dirty code explicitly unsupported
  until a patch digest can identify uncommitted contents.

## Verification

- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q`:
  336 passed, no skips; macOS, Python 3.12.12.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost benchmarks/v1 tests/v1 tests/conftest.py`: pass.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync mkdocs build --strict`: pass.
- 48 adversarial checks, including a missing second fold with every application
  still represented, stale cache inputs, nonzero workers and malformed predictions.

## Failed Attempts

- Preimplementation collection failed because the judge module did not exist.
- Review strengthened the missing-fold fixture: A-ID coverage alone cannot detect
  a missing repeat/fold when the application still has another successful record.

## Risks and Follow-ups

- Declared provenance is not authenticated. Target alignment, metric recomputation,
  actual protocol design, package hashes and budgets remain outside this slice.
- Real dataset acquisition/hashes and baseline capability smoke are next; no frozen
  real manifest, runner, held-out tasks or E-gate evaluation is claimed yet.

## Commits

- This slice: `test: add v1 artifact integrity judge` (parent `bc68ab9`).
