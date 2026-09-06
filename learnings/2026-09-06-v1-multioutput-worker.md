# 2026-09-06: Preserve the frozen scale convention in A6 workers

## Context

Parent 374ab87. A6 remained unsupported in the current evaluation worker. The
public TargetScale.fit uses weights; the frozen A6/comparator convention uses
unweighted training-population means and standard deviations.

## Decision or Result

Use the frozen preprocessing operation on training targets only and construct a
public TargetScale from its output. Train/validation objectives use the same
standardized units. Persist the scale with the model and inverse-transform once
through MultiOutputModel. Weights still affect objective derivatives and losses;
they do not silently redefine the frozen normalization convention.

## Changes

- Current worker and inference: matrix target schema, shared/independent recipes,
  validated scale metadata and original-unit output.
- Smoke: selectable applications, A6 freeze equality, five grouped folds.
- [Sprint 045](../v1-sprints/045-multioutput-worker.md), five new tests and
  [raw evidence](../benchmarks/v1/evidence/multi-worker-045/README.md).

## Verification

Use UV_CACHE_DIR=/tmp/openboost-research-uv-cache.

- Before editing, `uv run --no-sync pytest tests/v1/test_current_worker.py::test_a6_train_scale_and_original_units -q -o addopts=''`
  failed with unsupported job. After changes the full worker file has 20 passes.
- `uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' -q --tb=short`:
  819 passed. Changed-file Ruff and strict MkDocs pass.
- `uv run --no-sync python -m benchmarks.v1.openboost_worker_smoke /tmp/openboost-multi-worker-045-final --applications A6`:
  5/5 real shared-tree cells pass, exact freeze-scale equality and fresh replay.
- Source and raw worker artifact hashes match final files. macOS x86_64,
  Python 3.12.12, NumPy 2.3.5, one thread, 90-second fit/30-second replay caps.

## Failed Attempts

A6 rejection was reproduced before the adapter change. No core changes or frozen
metadata revisions were required. Final review corrected a metric label: the
recipe averages over rows and sums over channels. The real artifact run was
repeated after correcting that label; selected models did not change. Reusing weighted fit directly would have changed
the declared evaluation convention; the adapter explicitly avoids that substitution.

## Risks and Follow-ups

Real runs use shared mode; independent mode has synthetic direct/fresh parity.
Four-round/32-bin validation checks are not quality or cost measurements. A13
cross-configuration selection, all remaining adapters, D5, formal label isolation,
full search/test scoring and F0.3 remain open. Nothing pushed or published.

## Commits

- This A6 worker slice; parent 374ab87.
