# Sprint 097 result and retrospective

Status: the single corrected CPU smoke passes at clean `518eccf`; its allowance
is consumed, with no retry. [Raw evidence](../benchmarks/v1/evidence/author-linux-identity-097/README.md)
and the [construction/freeze design](097-explicit-worker-identity.md) retain exact
inputs, checks and results. This record supersedes the design's pending-run status.

## Observation and evidence

All nineteen original checks pass. The two additional records verify UID/GID
triples of 1000, empty supplementary groups and `no_new_privs=1`, before and after
a fresh interpreter exec. The worker then runs all six public examples and writes
its workspace successfully. Core/material writes fail with `EACCES`; root
restoration fails with `EPERM`. Core hashes and the four controller evaluator
hashes are unchanged. All private-path and child/symlink probes remain denied.

After those checks, a separate-session child starts and the provider actually
expires the Sandbox: return code 124, 90.949-second wait phase. Image/creation work
takes a separate 22.540 seconds. The final sleep's `KeyboardInterrupt` is retained.
Cleanup returns without error. This validates the declared timeout gate; it does
not independently observe child liveness after expiry or test the entire author
budget. No model request or independent attempt ran, and generated tokens remain
unmeasured (`null`), not an invented zero.

All thirteen original uploaded files, including the old probe, are byte-identical
to 096. Only the trusted launcher is added. The same five checks that failed under
UID 0 now pass. All 24 frozen source hashes match; all 85 pending GPU run-8 source
hashes are unchanged. The 096 failed archive and its consumed freeze are untouched.

## What this changes

The known-code worker now provides a usable, scoped Linux boundary for preparing
the author experiment. This removes the concrete root-identity blocker. It does
not establish correctness or lower authoring cost for a researcher-designed
algorithm. The CPU/CUDA foundation implementation did not change in this sprint.

The important correction was enforcing process identity on the actual command
path and measuring it after exec. Local API acceptance and mocked syscalls only
established construction and control flow. The real run is the evidence for this
boundary; preserve both failure and correction instead of replacing the old result.

## Next steps and stopping point

Stop here for the planned retrospective. Return to [069](069-authoring-pilot.md)
and [093's author-readiness sequence](093-foundation-progress-and-next-steps.md):

1. Design a trusted controller that owns the actual model usage ledger, enforces
   remaining generated-token allowance before requests and the total wall deadline,
   and fails closed when usage is unavailable. Every worker command must use the
   verified launcher. Keep credentials and the evaluator outside the worker.
2. Freeze a small real exhaustion smoke before any model invocation: exact model,
   counting rules, request/output/tool limits, spend ceiling and failure retention.
   Synthetic token events cannot close this gate. Any allowance must be concrete;
   this CPU smoke does not authorize model use or another remote run.
3. Complete fair incumbent paths, task acceptance and model/tools/settings, then
   conduct authorized independent D1/D2 attempts. Measure time to first correct
   result, failed work, tokens, hints and core/private edits. Keep formal E5 and
   sealed-task handling intact; designer development work is not independent work.

GPU run 8 remains separately pending. Full required recipes, batching, real A1–A13
quality, matched-quality cost and adoption are still open. Do not build a general
sandbox platform or broaden CPU optimization to avoid these product questions.

## Verification and commits

- `518eccf`: correction, 24-input freeze and fourteen-file upload boundary.
- 52 focused launcher/controller checks pass; full CPU suite: 1880 pass and one
  Linux-only skip. Ruff passes. MkDocs builds with the existing evidence-link warning.
- All eighteen raw artifact hashes and all 21 indexed archive files match. Local
  replay passes only with actual expiry; original 096 failure remains rejected.
- Active authorization changes only to `consumed`; normal execute reuse stops
  locally before creating output or reaching Modal. No push occurs.
