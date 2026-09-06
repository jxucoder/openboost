# 2026-09-06: Connect the current foundation to evaluation workers

## Context

Parent 03ff34e. Sprint 038 M3 called for current OpenBoost integration; the existing
baseline worker only executed incumbents. Further isolated objectives would not
resolve this evidence gap.

## Decision or Result

Add a separate current worker and inference entry point rather than importing
current recipe policy into incumbent branches. A1 and A11 consume frozen numeric
packets with explicit validation targets. Fixed budgets return final models;
patience returns the strict best validation snapshot. Persist output semantics
with the raw model: scalar means or Normal means/standard deviations. The bundle
is benchmark-specific JSON, not a stable new public API.

## Changes

- openboost_worker.py: schema-checked current CPU trial adapter.
- openboost_predict.py: declared output decoding without recipe imports.
- openboost_worker_smoke.py: bounded five-fold real validation replay.
- Fifteen focused tests; [Sprint 044](../v1-sprints/044-current-worker.md) and
  [raw evidence](../benchmarks/v1/evidence/current-worker-044/README.md).

## Verification

Use UV_CACHE_DIR=/tmp/openboost-research-uv-cache.

- Initial `uv run --no-sync pytest tests/v1/test_current_worker.py -q -o addopts=''`
  failed on the missing worker. After implementation fifteen focused cases pass.
- `uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' -q --tb=short`:
  814 passed. Changed-file Ruff and strict MkDocs pass.
- `uv run --no-sync python -m benchmarks.v1.openboost_worker_smoke /tmp/openboost-current-worker-044`:
  10/10 A1/A11 five-fold validation cells pass, each with exact fresh-process replay.
- Recorded source and raw worker artifact hashes match committed files.
- macOS x86_64/Python 3.12.12/NumPy 2.3.5; one thread per real worker,
  90-second fit limits and 30-second inference limits. No memory cap or CUDA.

## Failed Attempts

The initial absence was reproduced. No core code change was needed. Validation
packets are supplied explicitly; the adapter does not invent dummy labels or
silently skip best-model/stopping semantics.

## Risks and Follow-ups

Four rounds/32 bins validate integration only. All A1–A13 remain required; A6/A13
scaling/selection and other adapters are next. Full search, held-out test scoring,
quality comparison and OS test-label isolation are still absent. Packet files
are an interface boundary, not a security boundary. Formal F0.3/E3/E6 remain open.
Remaining D5 development also remains open. Nothing pushed or published.

## Commits

- This current-worker slice; parent 03ff34e.
