# 2026-09-06: Execute matched frequency-severity through public components

## Context

Parent e603f1f. Sprint 057 verified matched positive-payment inputs; composition
training and persisted real-data replay remained untested in the current workflow.

## Decision or Result

Use paid_loss_problems for Poisson paid-event counts and Gamma positive policy
averages weighted by paid count. Select each component independently using its
own validation objective; persist both roles with FrequencySeverity. Emit named
rate/count/severity/annualized/period outputs. Do not claim joint selection.

## Changes

- Dedicated strict composition worker and inference-only replay command.
- Five-fold hash-bound smoke harness, combined fit budget and exact named replay.
- Direct public parity and invalid contract tests; no foundation code changes.
- [Sprint 058](../v1-sprints/058-composition-worker.md).

## Verification

- Focused tests: 9 passed. Full CPU regression: 901 passed.
- Tests compare component model identities, stopping and fresh-process outputs;
  reject test/offset/weight injections, overlapping policies and invalid counts.
- All five real fits and named-array replays pass; products and exposure units
  pass. Binding/source/output hashes match
  [evidence](../benchmarks/v1/evidence/composition-058/README.md).
- Ruff, strict MkDocs and whitespace pass. Commands use uv run --no-sync with
  UV_CACHE_DIR=/tmp/openboost-research-uv-cache; macOS/Python 3.12.12/NumPy 2.3.5.

## Failed Attempts

Lint corrected test import ordering. No source eligibility, units or performance
caps were changed to obtain passing tests.

## Risks and Follow-ups

Independent component selection is not aggregate-loss joint selection. Full A9
search/quality, remaining applications, D5 and CUDA remain open. The paired mean
is not a calibrated compound distribution. Hashes are not OS access isolation.

## Commits

- This composition execution slice; parent e603f1f. Local only, no push.
