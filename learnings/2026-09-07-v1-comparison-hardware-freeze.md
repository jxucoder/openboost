# 2026-09-07: Separate comparison cohorts and a bounded hardware freeze

## Context

The user approved completing the historical bindings and preparing a reviewable
run request after `2c5927c`. The binding slice is committed at `c881259`. All seven
prior device allowances are consumed; preparation does not authorize upload or
an eighth invocation. See [092-C/D](../v1-sprints/092-cohort-bindings.md).

## Decision or Result

Freeze two separate pytest processes in one bounded invocation. Historical cases
keep their original sources, full-loss predicates, tolerances and literal verdict.
The two original coefficient failures and 24 old recipe byte assertions are
preregistered disagreements: the latter omit the new owned best validation buffer.
They remain failures. A different assertion, unexpected pass, missing case, skip,
error or duplicate prevents validation completion.

The revised 529-case gate requires every case and the installed-source/replay
artifacts to pass. Its 383 bindings retain all original requirements and settings;
117 comparison cases, 27 consumer cases and two lowering/cost checks add explicit
distinctions. The 236 unchanged operation cases run in both processes and must not
be counted as new independent requirements. No CUDA execution has occurred.

## Changes

- `benchmarks/v1/cuda_comparison_preflight.py` preserves separate logs, JUnit and
  verdicts; enforces both allowances, frozen sources, clean checkout, resources,
  one output directory and a shared 600-second test deadline. Reuses immutable
  installed-source checks and bounded artifact retention from the prior harness.
- `benchmarks/v1/freeze_comparison_run8.py` derives exact cases and 409 artifact
  names; copies only the 86-file closure and builds/extracts its wheel locally.
  It refuses to regenerate an approved/consumed or attempted allowance.
- `tests/v1/test_comparison_lowering_cost_cuda.py` collects actual directed-double
  PTX lowering and four uninstrumented bounded fits (weighted and installed D2,
  twice each), with independent prediction/NLL/CRPS checks and ownership counters.
  Compilation may be warm from prior tests; no speed or real-data claim follows.
- [Protocol](../v1-sprints/092-comparison-run8.json) and
  [isolated collection](../v1-sprints/092-isolated-collection.json) retain all hashes
  and the exact 385 historical / 529 revised case lists. Both authorizations remain
  pending. The 32 MiB artifact cap bounds JSON return, not model memory or GPU cost.

## Verification

- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/v1/test_cuda_comparison_manifest.py -n 0 -q` — 37 pass. Fault injection covers missing/duplicate/skipped/error/extra cases, wrong assertion/type, unexpected passes, provenance, pending/reused allowance, dirty source, changed freeze and budget drift. Actual run-7 JUnit also preserves its two literal failures.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.freeze_comparison_run8 --collect-snapshot` — all 914 executions collect from only the copied closure and extracted wheel under Python `-I`; core source hashes match. No CUDA case executed.
- `OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' --tb=short` — 1794 pass, one Linux-only skip (Python 3.12.12, macOS).
- Production and changed-file Ruff, `uv run --no-sync mkdocs build`, `uv build --offline`, and `git diff --check` pass. MkDocs retains the existing execution-page evidence-link warning.

## Failed Attempts

- The initial focused test could not import the absent new cohort harness. It
  passes after implementation; this is a local accounting test, not CUDA evidence.
- Archived pytest JUnit omits the optional exception `type` attribute. Requiring
  it would falsely reject the two known failures. The judge now verifies the
  actual `>` assertion line and terminal `AssertionError`, while rejecting any
  supplied contradictory type. Merely finding an assertion in source context
  cannot qualify a different failure.
- Initial isolated collection used macOS's symlinked temporary path and emitted
  empty file prefixes. Resolving the snapshot path restores complete pytest node
  IDs. No file, case, tolerance or oracle was relaxed to repair collection.

## Risks and Follow-ups

Real CUDA lowering, revised trajectories, ownership, fresh replay and actual cost
remain unverified. The two old numerical failures and separate split near-tie
remain visible. After a separately approved upload/invocation, retain partial
failures, consume the allowance and stop for retrospective. No retry or push is
included. Formal P7/E4, other required CUDA families, independent author/accounting
and application gates remain open.

## Commits

- `c881259` — complete collected bindings and independent reference trajectories.
- This entry accompanies the run-8 harness and exact pending freeze.
