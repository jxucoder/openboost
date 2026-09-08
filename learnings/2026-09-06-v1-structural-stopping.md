# 2026-09-06: Separate stopping completion from the default policy

## Context

Sprint 064, implementation parent `8c9f3de`. The user resumed execution through
the sprint plan, requesting a later evidence-driven retrospective. An external
completed record was rejected only because it was not a concrete StopState.

## Decision or Result

Expose structural StoppingStatus with rounds, completed_rounds and reason.
Keep default StopState and all built-in policy behavior. Validate completion and
trace/context identities without converting external results or interpreting their
policy-specific fields. Read-only typing does not enforce arbitrary payload ownership.

## Changes

- Public stopping/result contracts and explicit malformed-count/reason failures.
- Source tests for preserved external object identity, every missing field, invalid
  metadata, isolated failures, and a real public tree/transaction threshold loop.
- The independent recurrence predicts losses 0.125 and 0.03125 and raw outputs
  plus/minus 0.75 after two rounds; validation-best remains the initial model under
  opposite validation targets. Zero budget and coincident custom/budget termination
  also pass. This is a development policy, not a statistical stopping method.
- Public result/stopping docs, extension example and Sprint 064 reflection.

## Verification

Environment: macOS, Python 3.12.12, pytest 9.0.2; CPU only. Commands use
`UV_CACHE_DIR=/tmp/openboost-research-uv-cache`.

- Before core edits: `uv run --no-sync pytest tests/v1/test_public_results.py -k structural_stopping -n 0 -q`
  failed with `recipe result requires AcceptedState and StopState`.
- `uv run --no-sync pytest tests/v1/test_public_results.py tests/v1/test_public_stopping.py tests/v1/test_public_ordered.py -n 0 -q`: 77 passed.
- `uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' -q --tb=short`: 943 passed.
- `uv run --no-sync ruff check src/openboost tests/v1/test_public_results.py examples/v1_extensions/custom_stopping.py`: passed.
- `uv run --no-sync mkdocs build --strict` and `uv build --offline`: passed.
- Planning/documentation targets and staged diff checked before local commit.

## Failed Attempts

The expected first failure exposed concrete-type coupling. Ruff initially found
one test import-order issue; corrected before final verification. No model-quality,
CUDA or formal author experiment was performed.

## Risks and Follow-ups

Structural records can still misreport policy mathematics; independent policy
verifiers remain necessary. The scheduler validates common metadata and preserves
diagnostics, not arbitrary mutation/process isolation. Sprint 065 will verify
installed custom completion and independent RNG/preparation/failure cases.

## Commits

- Structural stopping implementation; parent `8c9f3de`.
