# Sprint 096 CPU smoke authorization

The user replied "continue" to the concrete request to upload thirteen frozen
files and run one Modal CPU Sandbox for 90 seconds with 2 CPUs, 2048 MiB RAM,
an image build if needed, no retries and no model calls. This authorizes that
specific smoke. It does not authorize GPU run 8 or an independent author attempt.

The reviewed [freeze](096-linux-worker-smoke.json) changes only `authorization`
from `pending` to `approved`. Its 22 file hashes, thirteen uploads, nineteen cases,
image, CLI and resource limits remain unchanged from `cda1947` / `251edb4`.
The frozen [construction plan](096-linux-author-worker.md) and earlier local
evidence retain their original pending status as historical records.

## Execution plan

1. Verify the authorization-only change, all CPU smoke inputs, and the unchanged
   run-8 source hashes. Commit before the first service request.
2. Execute the exact frozen CLI once from that clean revision. Retain build output,
   worker stdout/stderr, provider completion/timeout and evaluator integrity hashes.
   A build/startup/probe failure is retained; do not silently retry or widen scope.
3. Archive the raw result and reflect. Report the observed isolation boundary and
   any failures separately from token accounting, author benefit and GPU conformance.

Smallest failing check: compare the reviewed and approved freeze after removing
the authorization field; they must be identical. The normal local input checker
must still pass before any network action.

## Outcome

The single invocation executed at clean `7985645`. It passed 14/19 declared checks
and failed worker identity/core protection; no timeout marker was reached. The
allowance is consumed, with no retry. The [result](096-linux-worker-result.md)
retains the exact approved freeze; the active freeze now uses `consumed` solely
to prevent normal execution reuse. A corrected workload needs a separate freeze
and allowance.
