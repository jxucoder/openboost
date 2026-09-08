# Sprint 102: Execute the frozen Normal CUDA correction

Status: local readiness verified; external dispatch blocked by automatic approval
review before process creation. The user's "finish" was interpreted as directing
the next foundation checkpoint, but review requires specific authorization for
the source transfer and paid GPU invocation. The
[run-8 request](092-comparison-run8-request.md) remains pending and unchanged in
scope. Agent evaluation and the Sprint 100 model test remain deferred.

## Plan

1. Verify the frozen source closure and local dispatch/judging checks. Obtain
   specific approval for the existing upload/compute request, then commit the
   two authorization fields before dispatch.
2. Upload the exact 86-file closure to Modal and execute one T4 invocation with
   two requested CPUs, 8192 MiB requested memory, 900 function seconds, a shared
   600-second test deadline and zero retries. Use the existing fixed output path.
3. Preserve all returned logs, literal verdicts and declared artifacts. Reconcile
   source identity, every historical/revised case, numerical decisions, ownership,
   fresh CPU inference and recorded costs. Mark the allowance consumed and commit
   the evidence. Complete the planned retrospective before another hardware run.

The local pre-dispatch check passes all 37 harness cases. All 85 prefrozen source
hashes match; the 86-file pending packet is 1,459,822 bytes before the two
authorization values change. Only those values change in the protocol. The
original isolated collection and earlier failed runs remain unchanged.

## Acceptance

Apply the original [run-8 criteria](092-comparison-run8-request.md): all 529 revised
cases pass, all 385 historical cases appear with exactly their 26 preregistered
disagreements, and all 409 declared JSON artifacts are retained. Historical
failures remain failures. Missing evidence, unexpected outcomes or skipped CUDA
cases fail this checkpoint. Keep the known split near-tie limitation explicit.

This is bounded Normal/D2 correctness validation. It does not establish all CUDA
recipes, matched-quality speed, real-task value, agent benefit or complete v1.

## Result and reflection

The local checks pass, but no GPU run occurred. Commit `b58b168` recorded the
interpretation that "finish" authorized the pending checkpoint. Automatic review
then rejected process creation because that instruction did not specifically
authorize uploading 86 repository files and launching paid Modal T4 compute under
the repository's pending-run rule. No workaround or retry was attempted.

Restore both authorization fields to pending. The fixed output directory is
absent: no manifest or remote result was produced, and no GPU allowance is
consumed. The protocol returns byte-for-byte to its preauthorization contents.
All 85 frozen source hashes still match. Actual CUDA validation is the remaining
blocked step; it cannot be replaced by local CPU results or skipped tests.

The concrete approval request remains one upload of the frozen 86-file closure
and one T4 invocation with two CPUs, 8192 MiB, 900 function seconds, a shared
600-second test deadline and zero retries. After that run, preserve its evidence
and stop for retrospective. Any device failure must be classified before changing
production code, oracles or tolerances.
