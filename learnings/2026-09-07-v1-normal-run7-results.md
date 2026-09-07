# 2026-09-07: Measured false improvement in Normal backtracking

## Context

The user approved the concrete 70-file upload and single T4 diagnostic invocation
frozen at `1e0acfb`. Approval commit `80740f2` was clean at dispatch. All previous
source, test and tolerance hashes were retained. Run 7 is now consumed; no retry.

## Decision or Result

The [raw run](../benchmarks/v1/evidence/cuda-acceptance-091/README.md) has **383 passes
and two failures**. All 383 original outcomes repeat run 6; both new diagnostic
cases pass. The overall verdict remains false. All 70 sources, eighteen package
versions, installed core/D2 sources and 81 raw artifact hashes verify. All nineteen
saved-model CPU replays pass; model bytes equal run 6.

Both original assertions fail at round zero's mean update. The GPU accepts rate 4
after rejecting rate 8. Measured loss improves by `8.881784197001252e-16`, while
independent 60/100-digit math at the exact stored inputs worsens by
`5.904866222924252e-18`. Ordinary float64 original-row math also reports a false
improvement. This resolves the missing-state hypothesis: the actual failing
candidate is a numerical false improvement, not merely different starting precision.

Float32-rounded gradients/Fisher match the independent geometry exactly. Ordered
float32 accumulation changes the exact sum of the stored mean-gradient field from
zero to `4.470348358154297e-08`, yielding a nonzero leaf. The loss decision then
accepts its one-float32-value mean decrease. Reverse first rejects six log-scale
trials and reaches the same mean candidate. State/term/best updates and cleanup
follow the measured decisions correctly; validation truly improves on that step.

## Changes

- Retain all raw outputs and mark the live protocol consumed. Preserve approved
  protocol/source hashes in the immutable dispatch manifest.
- Archive checks verify the exact failure set, traces, gradients, false-improvement
  sign, state/ownership evidence and nineteen saved-model CPU replays.
- Derive separate forward/reverse analyses using the prefrozen CPU oracle. Raw
  test outputs, production code, tolerances and the strict judge remain unchanged.

## Verification

- SHA256 of every dispatch source matches `git show 80740f27c07c9d676531edf2467bd82a7ef97da6:<path>`;
  all 81 raw hashes verify and the original strict judge reconstructs the verdict.
- `OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync
  pytest tests/v1/test_cuda_acceptance_manifest.py tests/v1/test_cuda_normal_manifest.py
  -n 0 -q` — **23 passed**, including both histories and nineteen run-7 saved-model
  CPU replays. Ruff passes for the changed verifier.
- Preserve the raw JUnit trailing whitespace and force-stage the exact ignored
  `pytest.log`. Check staged raw bytes against the manifest; apply whitespace
  checks to changed support/docs, without editing immutable raw evidence.

## Failed Attempts

One hardware invocation reproduces two conformance failures by design. No additional
test, source/installation or observation failure occurred. No retry was launched.
The tiny NLL difference does not justify an application-quality or speed claim.

## Risks and Follow-ups

Specify a stable loss-difference/comparison boundary with explicit numerical
resolution and independent adversarial evidence. A fixed epsilon cannot preserve
both tiny true improvements and false-improvement rejection. Higher precision in
aggregation alone cannot establish reliability of full-loss subtraction.
Record the semantic-cohort implications before implementation; retain the two
original tests and failed runs. Stop for the planned retrospective before any
production correction or further hardware. All R/C/A, P7/E4 and author gates remain.

## Commits

- `1e0acfb` — concrete run-7 freeze.
- `80740f2` — explicit upload/compute approval and clean dispatch revision.
