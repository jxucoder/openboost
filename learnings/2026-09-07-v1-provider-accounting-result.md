# 2026-09-07: Real cap exhaustion passes; the cancellation probe finishes early

## Context

The user approved the concrete [099 packet](../v1-sprints/099-accounting-smoke.json)
after local accounting/cancellation construction. The allowance was at most three
generation requests, 4288 output tokens and $0.05, with zero retries. This is
preparation for fair independent algorithm-authoring attempts under 069.

## Decision or Result

At clean `5c0f31ab1ec1c93936396bd40a2f5be6e02f5d77`, the exact frozen command
executes once and returns exit code 1. Actual exhaustion passes: output caps are
128 then the remaining 64; both responses are incomplete at `max_output_tokens`,
and all 192 generated tokens are reasoning tokens. Final usage reconciles and a
subsequent request is rejected before HTTP. These are actual provider counts.

The cancellation probe completes before its five-second work deadline with a
short explanation that its requested integer list cannot fit in one reply. Its
104 output tokens include 83 reasoning tokens. The controller returns that timely
answer, reconciles usage and rejects another request at the one-request limit.
No cancellation occurs. The frozen overall failure is retained; complete usage
does not imply a successful cancellation test.

Three creates and seven retrievals yield ten complete HTTP 200 replies. The
run's output total is 296, including 275 reasoning tokens, with 107 input tokens.
Runtime is 12.713916 seconds including local archival. The frozen conservative
rates imply $0.00038195, a usage-based estimate rather than a verified invoice.
The allowance is consumed; no retry, worker, GPU, independent author or push occurs.

## Changes

- [Evidence](../benchmarks/v1/evidence/author-accounting-099/README.md): all 81
  original files, comprising the original 80-file index and everything it lists.
  Preserve original replies, request bytes, ledgers, source hashes and failed verdict.
- A standalone verifier derives usage and classifications from raw response
  bodies and checks the original hashes without importing the request controller
  or using HTTP. Post-run analysis and command metadata have a separate index.
- [099 retrospective](../v1-sprints/099-accounting-result.md): distinguishes the
  passed bounded cap gate from unexercised cancellation and remaining author work.
- Active packet authorization becomes `consumed`; the archived approved packet,
  all twelve frozen sources and all earlier evidence remain unchanged.

## Verification

- The no-network packet preflight passes before dispatch and after consumption.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache OPENBOOST_BACKEND=cpu uv run --no-sync pytest tests/v1/test_author_accounting_smoke.py -n 0 -q`:
  ten checks pass before the authorization commit and after the consumed result.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python benchmarks/v1/evidence/author-accounting-099/verify.py --check`:
  verifies all 80 original indexed files, twelve source copies, exact request
  prompts, raw/canonical response equality, actual usage and the failed verdict.
- Original/output copies agree byte for byte. Credential-pattern inspection
  finds no retained key; headers and exception text were not logged. No private
  task or dataset content is included in any request.
- All 85 GPU run-8 sources, 24 frozen 097 sources and original 096/097 indexed
  artifacts remain unchanged. Ruff/format checks pass for the added verifier;
  MkDocs builds with the existing `execution.md` evidence-link warning.
- No production code or dependency changes. The prior full CPU suite remains
  1933 passed with one Linux-only skip; no redundant full rerun for archival.
- Staged whitespace checking flags the existing final blank line in the frozen
  `source/benchmarks/__init__.py` copy. Preserve that exact hashed source; the
  authored-file whitespace check excludes only this unchanged archival file.

## Failed Attempts

A long-output instruction does not guarantee an active response at a chosen
deadline. The model may complete with a short capacity explanation. The recorded
sequence contains `queued -> in_progress -> completed`, so an active state was
observable. It does not prove that a later cancel would win the completion race.

## Risks and Follow-ups

Stop at this retrospective. Next local design should request cancellation as soon
as an active response is observed and retain completion races as unexercised.
Freeze a new bounded live allowance before dispatch. Keep this direct-cancellation
observation distinct from total author-deadline enforcement. If final cancelled
usage is null, a provider-supported reconciliation path remains necessary; do not
invent zero usage or count estimates as measurements.

The result advances the real cap/usage gate only. Actual interruption accounting,
full 20k/1800-second enforcement, worker integration, attempt authority, artifact
collection, fair D1/D2 arms and independent author benefit remain open. GPU run 8
is unchanged and pending. Do not expand CPU search or provider fixture catalogs
as a substitute for the missing observation.

## Commits

- `68951ef`: frozen local packet and acceptance criteria.
- `5c0f31a`: explicit one-run authorization and clean execution revision.
- This commit archives the measured failed overall result and retrospective.
