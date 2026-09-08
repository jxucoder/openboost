# Sprint 102: Execute the frozen Normal CUDA correction

Status: the user's "finish" after Sprint 101 directs execution of the next
foundation checkpoint. Use the existing bounded
[run-8 request](092-comparison-run8-request.md) without changing its scope.
Agent evaluation and the Sprint 100 model test remain deferred.

## Plan

1. Verify the frozen source closure and local dispatch/judging checks. Record the
   execution instruction and commit the two authorization fields before dispatch.
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

Pending the single dispatch. No automatic retry or additional source upload is
included. Any failure must be classified from retained evidence before changing
production code, oracles or tolerances.
