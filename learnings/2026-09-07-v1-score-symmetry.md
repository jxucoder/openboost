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

## Source correction and reflection

`13fdd83` records diagnostics before production changes. The existing CUDA scalar
scorer now independently rounds its left, right and parent products using
`fmul_rn`; no public signature, scoring factorization, validity check or tie policy
changes. This prevents asymmetric contraction at the score-sum boundary by the
documented operation contract; real-device acceptance remains pending.

`OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync
pytest tests/ -m 'not gpu and not benchmark' --tb=short -n 0 -q` passes 1265 cases
with one Linux-only skip and 212 GPU cases deselected. Production/changed-test Ruff
and documentation build pass. The failed run-4 artifacts and its exact original
case/source records are unchanged. No CPU test is counted as a kernel compilation
or GPU parity result. Freeze the complete package next, preserving both hardware
and private-upload authorization guards.

## Risks and Follow-ups

Real-device compilation, exact scores, selected splits, leaves, predictions and
metrics must pass the new frozen matrix. PTX and device-buffer observations must
distinguish the contraction hypothesis from histogram/candidate accumulation.
All broader v1 acceptance and author/adoption questions remain open.

## Frozen package

The [run-5 protocol](../v1-sprints/089-symmetry-run5.json) keeps all 202 original
cases and their unchanged test/reference bytes, and adds ten diagnostics. Its
49-file closure includes 27 production modules and no sealed cards or unbounded
directory upload. Both compute and private-upload authorizations are pending;
the existing launcher guards are reused. The actual pending CLI rejects before
Modal import/output creation, and no new invocation or transfer occurs.

All 40 focused manifest/judging/arithmetic/source checks pass. Offline `uv build
--offline` produces a wheel and sdist; all 27 wheel modules match the freeze.
All 48 prefrozen hashes match and the proposed output directory is absent.
The protocol requests one invocation within the previous 900/600-second bounds,
with zero retries. Existing run-4 raw artifacts and provenance remain immutable.

Final package regression uses the full CPU command above: 1269 passed, one
Linux-only skip and 212 GPU cases deselected. Production/changed-support Ruff,
documentation build and whitespace checks pass. No environment mutation or remote
execution was needed. The package is ready for a concrete approval decision, while
the numerical fix remains unaccepted on real CUDA.

## Commits

`f5922fb` records the failed device run and retrospective; `13fdd83` freezes
diagnostic cases; `d3ee1c3` commits the candidate score correction. The next run
freeze is a separate local commit; no push is requested.

## Explicit run-5 approval

The user replied "sure" to the concrete 49-file private-upload request and one
Modal T4 invocation of all 212 checks, with 900/600-second limits and zero retries.
Both approvals are recorded for the unchanged package. Verify and commit before
dispatch; retain the actual result separately from authorization. The freeze is
`aab6882`; no new package or environment change is needed.
All 40 local integrity/judging/arithmetic checks pass with approvals recorded.
The 49-file closure and 48 prefrozen hashes match; output and launcher log are
absent before dispatch. Existing full CPU and packaging checks apply to the
unchanged source payload.
