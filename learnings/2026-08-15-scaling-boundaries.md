# 2026-08-15: Scaling Boundaries

## Context

The high-level `GradientBoosting` and `MultiClassGradientBoosting` APIs exposed
`batch_size`, but neither training loop read it. The large-scale guide also
passed a feature-major binned memmap directly to a high-level `fit` method that
validates sample-major input. GPU and multi-GPU pages quoted speedups without a
checked-in reproducible artifact.

## Decision or Result

Unsupported scale features now fail or read as experimental instead of looking
production-ready:

- a non-`None` high-level `batch_size` raises `NotImplementedError` before fit;
- memmap and mini-batch histogram utilities remain available as low-level
  building blocks, not an out-of-core model API;
- GOSS is described by its sampling behavior, without a universal speed/quality
  promise;
- multi-GPU is explicitly experimental until parity and repeated two-/four-GPU
  measurements exist;
- numeric GPU speedup tables were removed in favor of an evidence checklist.

## Changes

- `src/openboost/_models/_boosting.py`: validate the reserved batch parameter in
  single-output and multiclass fits; remove unsupported performance wording.
- `tests/test_large_scale.py`: cover both high-level model families.
- `docs/user-guide/training/large-scale.md`: replace the broken out-of-core
  recipe with a capability/status matrix and evidence gate.
- GPU setup, installation, sklearn docstrings, model guide, and GPU example:
  align public wording with the actual support boundaries.

## Verification

- Before the fix, both new high-level tests failed because no exception was
  raised and training proceeded while ignoring `batch_size`.
- Focused batch-size tests: 2 passed.
- `uv run pytest -q tests/test_large_scale.py tests/test_core.py
  tests/test_losses.py`: 77 passed, 3 expected bin-count warnings.
- `uv run mkdocs build`: passed with the repository's 29 existing griffe
  warnings.
- Focused source Ruff checks, example compilation, and `git diff --check`:
  passed.

## Failed Attempts

- `uv run mkdocs build --strict` stopped on 29 existing griffe warnings in API
  docstrings across callbacks, distributions, losses, models, arrays, trees,
  and importance helpers. The current docs workflow is non-strict, so the
  matching build was used for this change. Strict docs cleanliness remains a
  separate maintenance task.

## Risks and Follow-ups

- The low-level memmap and mini-batch helpers are not proof of end-to-end
  out-of-core training. Implement and test a real loop before reintroducing that
  claim.
- Multi-GPU correctness is not established by documentation. Require exact
  single-device parity and real two-/four-GPU artifacts before promotion.
- Run single-GPU scale-extension benchmarks on Linux/CUDA with full provenance;
  this Intel macOS environment cannot provide those results.

## Commits

- `ee555cb` — `fix: fail fast for unsupported model batching`
- `6440e31` — `docs: mark experimental scaling boundaries`
