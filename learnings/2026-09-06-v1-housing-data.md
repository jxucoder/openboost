# 2026-09-06: Housing legacy identity and five v1 splits

## Context

A1/A11 require the same California Housing inputs, but the historical record only
contains seeds 0–2. F0.3 must verify those and add seeds 3–4.

## Decision or Result

The independent v1 adapter reproduces the archived X/y byte hash and all nine old
split hashes. Five seeds now have committed hashes. Keep little-endian float32 and
the historical raw-byte hash convention, explicitly distinguished from Bike hashes.

## Changes

- [Sprint 013](../v1-sprints/013-housing-five-splits.md) records the plan/reflection.
- [Housing adapter](../benchmarks/v1/housing.py) validates raw shape, finite values,
  positive household denominators and float32 overflow; uses isolated seeded RNG.
- [Freeze](../benchmarks/v1/datasets/housing.json) binds all five splits and source.
  A1/A11 remain distinct quality tasks but one data source.

## Verification

- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.housing build/foundation_data/cal_housing.tgz --verify benchmarks/v1/datasets/housing.json`:
  real-data replay matched; corrupted seed4 hash caused nonzero exit.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q`:
  380 passed, no skips, including 20 new adapter tests.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost benchmarks/v1 tests/v1 tests/conftest.py`: pass.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync mkdocs build --strict`: pass.
- macOS/Python3.12.12/NumPy2.3.5; no training or test metrics.

## Failed Attempts

- Preimplementation collection failed on the absent module; early lint rejected
  a semicolon-separated test line, subsequently split by formatting.
- Original source webpage timed out. Archive contains no license file; license
  stays unresolved rather than inferred from download availability.

## Risks and Follow-ups

- Random splits do not establish geographic generalization. Data identity does not
  establish predictive quality, and unresolved licensing remains an F0.3 item.
- Next Adult official-test/stratified training splits; remaining required datasets,
  baseline capabilities, budgets, held-out tasks and runner still pending.

## Commits

- This slice: `data: freeze A1 A11 housing with five splits` (parent `dd2e9ad`).
