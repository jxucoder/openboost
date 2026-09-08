# 2026-09-07: Cancellation and final usage are separate observations

## Context

[098](../v1-sprints/098-author-request-accounting.md) reserves output before one
supervised synchronous request. Client expiry alone cannot establish provider
stop or final generated usage. The next step toward 069 is a small real accounting
observation, preceded by a concrete cancellation path and request freeze.

## Decision or Result

Add optional background transport to the trusted controller. Permit only create,
retrieve and cancel on the fixed Responses host. A known response ID allows one
cancel plus one final retrieval within a shared 15-second cleanup interval after
interruption. Cleanup grants no author work. Valid terminal usage reconciles the
reservation, but the interrupted answer is never returned, even if completion
races cancellation. Unknown creation IDs or usage remain unresolved and closed.

The current official cancellation example has null usage. Therefore confirmed
cancellation and complete accounting are separate acceptance conditions; neither
local fixtures nor cancellation acknowledgement close the real usage gate.

## Changes

- `responses_transport.py`: fixed retrieve/cancel routes with validated response
  identifiers; the existing HTTPS subprocess, byte bound and no-retry behavior
  apply to every operation.
- `responses_background.py`: sequential polling and bounded cleanup, retaining
  each operation, raw reply and interruption. No new generation in cleanup.
- `accounting.py`: optional background mode, terminal reconciliation after a
  stopped transport and unconditional withholding of the stopped answer.
- [099 plan](../v1-sprints/099-background-accounting-smoke.md): construction,
  provider decisions and a separately reviewable live smoke to follow.

## Verification

- First new test fails to import the absent background implementation. It then
  verifies deadline-triggered cancellation, final retrieval and known usage while
  the caller receives a timeout, not a model answer.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache OPENBOOST_BACKEND=cpu uv run --no-sync pytest tests/v1/test_author_background.py tests/v1/test_author_accounting.py -n 0 -q`:
  43 pass in 5.41 seconds. Eleven new cases cover normal polling, cancellation,
  unknown usage/identity, completion races, cleanup failure and shared expiry,
  and HTTPS retrieve/cancel endpoint confinement. Counts are explicit fixtures.
- Ruff and formatting pass for the changed support files. No production code,
  dependency, frozen worker source or GPU run-8 source changes in this slice.

## Failed Attempts

An interruption with a cancelled response but null usage cannot reconcile the
reservation. The tests preserve this as unknown rather than zero-token success.
The real API has not been invoked; account access and actual cancellation behavior
remain unverified.

## Risks and Follow-ups

Freeze exact public text inputs, model alias/settings, maximum requests, output
caps, work/cleanup windows, cost estimate and verdict before requesting a live
allowance. Model generation, independent authors, worker integration and GPU run 8
remain pending. Abrupt controller/host loss can still prevent cleanup; a bounded
cancel attempt is not a guarantee of immediate provider cancellation.

## Commits

- `71ef69e`: preceding local request accounting.
- This slice adds local cancellation construction, not a provider result.

## Frozen smoke construction and reflection

The second slice constructs `accounting_smoke.py` and the pending
[099 packet](../v1-sprints/099-accounting-smoke.json). Twelve exact source hashes
cover the implementation, tests, plan and dependency lock. Three fixed ASCII
integer-list prompts exercise two cap requests (128 then 64, sharing 192 tokens)
and one cancellation request (4096 tokens, five seconds). Total possible output
reservation is 4288. The proposed allowance is $0.05; the conservative token
estimate is $0.0082176, not an account-level hard dollar limit or a measured bill.

Preflight performs no HTTP or credential access. Execution requires the separately
approved packet, a clean committed source tree and a fixed fresh output directory.
The runner archives raw operations, accounting, source bytes and hashes, stops at
the first failed case and blocks reuse of that directory. This is a smoke-specific
one-use guard, not the full authority for independently authored attempts.

Review found that `service_tier=default` can still return a different actual tier.
The first check for this failed: the harness would issue a second generation after
an unexpected priority-tier response. The correction checks the retained returned
tier and stops before another generation. Actual counts stay known even when this
separate pricing/dispatch condition fails. The fixed default tier is now explicit
in every controller request; all original protocol cases still pass.

Verification:

- All 53 focused accounting/background/smoke checks pass in 5.50 seconds. These
  include the two-request exhaustion call path, cancellation with/without final
  usage, early completion, no speculative extra generation, source integrity,
  pending/consumed authorization, stop-on-failure and one-use output handling.
- Full CPU suite: 1933 pass, one Linux-only skip in 13.78 seconds. Ruff and format
  checks pass for production and changed support code. The source freeze is
  separately checked with the no-network CLI after final source preparation.
- All 85 run-8 source hashes and 24 source hashes in the consumed 097 freeze are
  unchanged. All 20 original 096 and 21 original 097 indexed artifacts match.
- No real model call, Modal invocation, GPU execution or push occurs in either
  local construction slice. No private/sealed task content is used.

The next useful observation is the one frozen live smoke, not more protocol
fixtures. Its acceptance distinguishes actual cancellation from complete usage;
the documented null-usage example makes failure plausible. Preserve that result
without adaptive retries. Even a pass would not establish a full author runner,
20k/1800-second enforcement, independent author benefit, or formal E5 completion.

- `67ecc94`: locally verified background cancellation and usage reconciliation.
- This second commit prepares the live request; authorization remains pending.

Final packet preflight passes with `network_used=false`, prompt sizes 151/89/163
bytes and all twelve sources verified. The focused ten-case smoke suite passes
again after adding exact prompt-hash and archived-config checks. MkDocs builds
with its existing `execution.md` evidence-link warning. These checks use no live
model and do not consume the proposed allowance.
