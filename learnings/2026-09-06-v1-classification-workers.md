# 2026-09-06: Preserve classification output and external packet identity

## Context

Parent b430df0. Current evaluation bindings still lacked A2/A3 despite public
classification recipes. Explicit class order and packet identity are required
for meaningful log-loss/calibration and saved prediction.

## Decision or Result

Add binary/multiclass adapter branches using canonical encoded labels, explicit
class counts and persisted class order. Return binary positive-class probability
or all multiclass columns. Apply existing patience/best-validation selection.
At the packet boundary preserve unique external integer/string IDs and map to
local integer rows inside NumericData. Never silently replace emitted source IDs.

## Changes

- Current worker/inference: class schema, probability outputs, encoded-label
  validation, weighted selection and external/local row separation.
- Twelve tests plus explicit A2/A3 smoke options. Default smoke set unchanged.
- [Sprint 048](../v1-sprints/048-classification-workers.md) and
  [failed/passing evidence](../benchmarks/v1/evidence/classification-048/README.md).

## Verification

Use UV_CACHE_DIR=/tmp/openboost-research-uv-cache.

- `uv run --no-sync pytest tests/v1/test_current_worker.py::test_binary_worker_probabilities -q -o addopts=''`:
  initially failed on unsupported job. The final worker file has 32 passing cases.
- `uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' -q --tb=short`:
  844 passed after the external-ID fix. Changed-file Ruff and strict docs pass.
- `uv run --no-sync python -m benchmarks.v1.openboost_worker_smoke /tmp/openboost-classification-048 --applications A2`:
  0/5: string source IDs rejected by NumericData.
- Rerun with `/tmp/openboost-classification-048-fixed`: 5/5 pass, exact fresh
  probability replay and source row identity retained.
- macOS/Python 3.12.12/NumPy 2.3.5; one thread, 90-second fit and 30-second replay
  caps. No memory cap or CUDA. A3 has synthetic verification only in this slice.

## Failed Attempts

All original Adult failures remain recorded. The problem was an adapter assumption,
not incompatible training mathematics. Fix at the external packet boundary instead
of expanding the core row-ID API or casting strings into invented source IDs.
String-ID tests now verify the actual consumer path.

## Risks and Follow-ups

Complete Covertype runs, calibration/quality search and other application adapters
remain required. D5, test isolation, real search, CUDA and author/adoption gates
are not closed by these integrations. No quality/speed or E3 claim. Nothing pushed.

## Commits

- This classification adapter slice; parent b430df0.
