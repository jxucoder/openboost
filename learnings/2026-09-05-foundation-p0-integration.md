# 2026-09-05: Foundation P0 Integration

## Context

The user authorized execution of the foundation checklist starting at P0.
The clean design branch at `67ee05b` needed the pinned remote revision
`6ebe3a8ced0e621b17e3cf63e31721af58471053`, while preserving the local
categorical persistence/cardinality, batch fail-fast, ScoringBench and CI fixes.

## Decision or Result

Merge the histories on `codex/gpu-python-foundation-design`; do not move main.
Keep CLAUDE.md as a pointer to canonical AGENTS.md. Reconcile GPU setup and
installation documentation with actual backend eligibility and experimental
scaling boundaries.

The new unified models were absent from the generic `ob.load()` class map.
Four new numeric/mixed-feature round-trip tests reproduced `Unknown model
class` for FormulaBoost and WeibullAFT. Add the two classes to that map without
changing the serialization schema or the preserved categorical state handling.

The remote benchmark page claimed committed results but identified its source
as transcription from ignored task notes. Remove unsupported result tables
and their guide/migration summaries; retain reproduction scripts and the
older committed CPU artifact. Historical design notes are explicitly marked
unverified, not promoted to current passed gates.

## Changes

- Merge the remote trainer, objectives, FormulaBoost, WeibullAFT, tests,
  benchmarks and documentation into the existing local history.
- [Generic loader](../src/openboost/_persistence.py): register the new model
  classes for generic loading.
- [Integration regression](../tests/test_unified_persistence.py): exact
  prediction and parameter round trips for both classes with numeric features
  and with categorical/missing features.
- [Benchmark evidence](../docs/benchmarks.md), GPU setup, installation and
  related model guides: preserve capability boundaries and require raw evidence.

## Verification

Commands use `UV_CACHE_DIR=/tmp/openboost-research-uv-cache` because the default
uv cache is not writable in the sandbox. `OPENBOOST_BACKEND=cpu` selects CPU
for all local model tests. Python 3.12.12 on Intel macOS.

- `uv sync --locked --extra dev`: passed after retrying outside the sandbox
  because the first attempt could not resolve PyPI DNS.
- Red test: `uv run --no-sync pytest tests/test_unified_persistence.py -n 0 -q`:
  **4 failed**, each because the corresponding class was absent from ob.load.
- Targeted merge regression: `uv run --no-sync pytest tests/test_categorical.py
  tests/test_persistence.py tests/test_batch.py tests/test_extensibility.py
  tests/test_formula.py tests/test_survival.py -n 0 -q`: **93 passed, 1 GPU
  skipped**, 313.77 seconds.
- After the loader fix: `uv run --no-sync pytest tests/test_unified_persistence.py
  tests/test_persistence.py -n 0 -q`: **21 passed**, 25.12 seconds.
- Full CPU regression: `uv run --no-sync pytest tests/ -m "not gpu and not
  benchmark" --tb=short`: **749 passed, 34 skipped**, 545.68 seconds. This run
  collected before the four new loader cases were added; those and the final
  loader change are covered by the 21-test focused run above. Skips include
  unavailable CUDA/multi-GPU, JAX and plotting dependencies, not successful
  validation of those capabilities.
- `uv run --no-sync ruff check src/openboost/ tests/test_unified_persistence.py`:
  passed. `git diff --cached --check`: passed.
- `uv run --no-sync mkdocs build`: passed, with the same 29 existing griffe
  docstring warnings recorded in the earlier scaling-boundaries learning.
- `uv build`: passed, producing both sdist and wheel after the loader fix.
- Source comparison confirms `_array.py`, `_core/_tree.py`, `_models/_boosting.py`,
  `.github/` and `benchmarks/scoringbench/` retain the local pre-merge content.
  The only additional persistence change is the two loader registrations.

## Failed Attempts

- The merge produced the three expected documentation/instruction conflicts;
  there were no production-code text conflicts. Clean merges still required
  semantic review, which found the generic-loader integration gap.
- Sandboxed dependency sync failed to fetch hatchling due to DNS restrictions;
  the authorized network retry succeeded with the same lockfile.

## Risks and Follow-ups

- Local CPU checks do not validate CUDA or the planned device-resident
  extension path. P1 establishes the stricter Modal test harness.
- The possible weighted constant-Hessian mismatch, device dispatch and
  fallback behavior remain P2 work. They are not claimed fixed here.
- Benchmark result tables need committed raw artifacts before reinstatement.
- Existing API-docstring warnings remain separate from merge correctness.

## Commits

- `67ee05b` — local parent containing the approved design.
- `6ebe3a8` — pinned remote parent containing the unified trainer.
- This entry accompanies the verified P0 merge commit.
