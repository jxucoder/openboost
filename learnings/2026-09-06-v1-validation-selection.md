# 2026-09-06: Independent validation selection and sealed test release

## Context

F0.3 had numeric trial workers and metrics, but no connection from complete
validation search to a fixed winning model. A producer's claimed score, omitted
trial or replaced model must not determine the final test evaluation.

## Decision or Result

The trusted orchestrator pins the full selection protocol, then the independent
selector checks all 16 configurations per declared method and recomputes scores
from row-aligned validation arrays. It selects across methods and seals all scores
and artifact hashes. Release re-audits before opening test features. Hashes are
consistency checks whose custody matters; they are not proof of access chronology.

## Changes

- [Selection](../benchmarks/v1/selection.py): independent audit, exclusive receipt
  creation, pinned receipt verification and test feature release.
- [Integration smoke](../benchmarks/v1/selection_smoke.py): actual isolated CPU
  training jobs, audit/seal/release and selected-model inference in a new process.
- [Tests](../tests/v1/test_selection.py): missing/failed/duplicate trials, changed
  config/protocol/model, fabricated scores/winner, partition overlap and test labels.

## Verification

- `uv run --no-sync pytest tests/v1/test_selection.py -n 0 -q`: 12 passed.
- `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 build/v1-env/bin/python -m benchmarks.v1.selection_smoke build/v1-selection-smoke-001`:
  16 actual XGBoost worker processes completed, independent selection sealed and
  selected-model inference completed in a new process. No real task was evaluated.
  [Summary](../benchmarks/v1/evidence/selection-cpu.json) identifies code/environment;
  the full generated packet remains in ignored build output and is reproducible.

## Failed Attempts

The initial focused test failed at import because the selection module did not
exist. API design explicitly excludes claimed scores rather than comparing them
with independently recomputed values that a caller might accidentally ignore.
A forged receipt hash alone cannot authorize a different winner: release re-audits.

## Risks and Follow-ups

The caller must pin protocol and receipt hashes independently of the producer.
Filesystem isolation, authenticated execution provenance, full method/task/device
matrix, early stopping and the full E3 judge remain unfinished. Row disjointness
does not replace query/entity/time split contracts. This is progress within F0.3,
not its exit. No OpenBoost production implementation or formal gate was added.

## Commits

- This slice: `eval: seal independent validation selection before test release`.

Final verification: 443 tests passed with no skips; Ruff and strict MkDocs build
passed. The work remains on the existing design branch with no external publication.
