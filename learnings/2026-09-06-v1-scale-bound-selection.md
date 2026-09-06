# 2026-09-06: Verify A6 normalization before selecting current models

## Context

Parent e83e7b5. Sprint 017 identified arbitrary positive A6 selection weights as
unenforced despite correct worker scaling. Current A13 integration must not build
on that gap.

## Decision or Result

Bind hashed training targets and scale into A6 protocols. The audit checks exact
row alignment, recomputes population means/stds, and requires inverse-std weights.
Report mean standardized RMSE. Preserve the strict all-trials-success receipt rule
and trusted orchestrator digest requirements. Do not modify other applications.

## Changes

- selection.py: enforce scale/target/weight consistency and score definition.
- Eight selection tests, including the reproduced missing-binding acceptance and
  a fixture where normalization changes the correct winner.
- current_selection_smoke.py: 16 current A6 configurations, retained outcomes,
  independent audit, seal/re-audit/release and fresh-process selected inference.
- [Sprint 046](../v1-sprints/046-scale-bound-selection.md) and
  [raw evidence](../benchmarks/v1/evidence/current-selection-046/README.md).

## Verification

Use UV_CACHE_DIR=/tmp/openboost-research-uv-cache.

- `uv run --no-sync pytest tests/v1/test_selection.py::test_a6_requires_scale_binding -q -o addopts=''`:
  failed before implementation because no ValueError was raised.
- `uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' -q --tb=short`:
  827 passed. Changed-file Ruff and strict MkDocs pass.
- `uv run --no-sync python -m benchmarks.v1.current_selection_smoke /tmp/openboost-selection-046`:
  16 successful trials; winner openboost:15; sealed selected prediction passes.
- Source hashes verified. macOS/Python 3.12.12/NumPy 2.3.5, one thread per trial,
  60-second fit and 30-second prediction caps; no RAM cap or CUDA.

## Failed Attempts

The counterexample accepted an unbound A6 protocol. The fix rejects it rather
than treating the protocol's arbitrary coefficients as evidence of train scaling.
The old denominator also did not report mean standardized RMSE, although fixed
coefficients gave the same within-fold ordering; the new score is explicit.

## Risks and Follow-ups

Trusted input provenance and pinned digests remain required. Packet separation
is not OS isolation. The synthetic grid is not the frozen real quality search;
all application adapters/real searches, A6 final quality aggregation, D5 and GPU
remain open. No test scores, speed or independent author/adoption claims.
Nothing pushed or published.

## Commits

- This scale-bound selection slice; parent e83e7b5.
