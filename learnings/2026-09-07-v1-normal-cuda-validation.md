# 2026-09-07: Resume Normal CUDA validation after deferring agent evaluation

## Context

The user instructed "finish" after the Sprint 101 engineering-priority amendment.
The next checkpoint is the already constructed Normal objective-comparison path.
Its source and acceptance packet was frozen before the intervening authoring work.

## Decision or Result

Execute that one bounded GPU validation under the existing run-8 request. Agent
studies remain paused. The instruction is applied to this concrete checkpoint,
without expanding the upload, resources, tests or retry allowance.

## Changes

- [Sprint 102](../v1-sprints/102-normal-cuda-validation.md) records execution,
  acceptance and the required retrospective.
- Only `authorization` and `upload_authorization` change to approved in the
  run-8 protocol. Every prefrozen source, case, tolerance and resource bound stays
  unchanged. The resulting protocol hash will be captured at clean dispatch.

## Verification

- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/v1/test_cuda_comparison_manifest.py -n 0 -q`:
  37 pass in 0.14 seconds on Python 3.12.12 / macOS.
- Source closure check passes: 86 files, 85 unchanged prefrozen source hashes,
  385 historical and 529 revised cases, 409 declared JSON artifacts. Fixed output
  directory does not exist before dispatch. Local Modal package is 1.3.0.post1.
- Real CUDA results are pending. Local harness checks are not device validation.

## Failed Attempts

None in this slice before dispatch. Earlier runs and their failures stay intact.

## Risks and Follow-ups

Actual CUDA lowering and revised comparison decisions remain unverified until
the run returns complete evidence. Preserve partial output on failure and stop
at retrospective without retry. Other required CUDA recipes, train-many and
practical quality/cost remain subsequent engineering work.

## Commits

- `f7f3c60` — defer author evaluation and prioritize foundation execution.
- This commit records the execution instruction and bounded run-8 authorization.
