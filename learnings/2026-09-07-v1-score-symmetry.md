# 2026-09-07: Scalar CUDA score symmetry

## Context

The user continued after the run-4 retrospective. The weighted/missing resident
fixture fails split and prediction parity despite all earlier primitive cases
passing. See [089](../v1-sprints/089-cuda-score-symmetry.md) and the
[unchanged run-4 result](../benchmarks/v1/evidence/cuda-resident-078/README.md).

## Decision or Result

Keep the public scoring boundary and exact tie rule. Independently round the
existing child and parent score products. This addresses a demonstrated arithmetic
symmetry counterexample; the exact cause of run 4 remains unproven without device
candidate buffers and generated code.

## Changes

Prepare direct swapped-summary and adjacent-ULP device verifiers plus weighted-root
diagnostics. Preserve the run-4 scoring function as test-only historical code so
the next bounded run can compare both implementations on identical device inputs.
The archived function is not an independent mathematical oracle or production
fallback. The original-row references and all original 202 cases remain unchanged.

## Verification

`uv run --no-sync pytest tests/v1/test_score_symmetry_diagnostic.py
tests/v1/test_score_diagnostic_source.py tests/v1/test_device_api.py -o addopts= -q`
passes 21 cases. The historical function's source-segment SHA256 matches the exact
run-4 function. Ten new real-device checks collect without importing CUDA packages;
collection is not execution. Changed-file Ruff passes. Commands use
`UV_CACHE_DIR=/tmp/openboost-research-uv-cache`. No new GPU invocation or source
upload is authorized by this continuation.

## Failed Attempts

Run 4 failed acceptance; do not reclassify its 14 failures or widen tolerances.

## Risks and Follow-ups

Real-device compilation, exact scores, selected splits, leaves, predictions and
metrics must pass the new frozen matrix. PTX and device-buffer observations must
distinguish the contraction hypothesis from histogram/candidate accumulation.
All broader v1 acceptance and author/adoption questions remain open.

## Commits

`f5922fb` records the failed device run and retrospective. Source correction and
the next run freeze will be separate local commits; no push is requested.
