# 2026-09-07: Stop on an observed active response

## Context

The [099 live result](../v1-sprints/099-accounting-result.md) establishes actual
bounded output-cap exhaustion. Its cancellation probe finished early, so the
overall frozen criterion failed without a cancel operation. The measured sequence
contained an in-progress state. The user's next "continue" authorizes the local
construction proposed in that retrospective; the old live allowance is consumed.

## Decision or Result

Introduce a trusted `stop_on_in_progress` policy for background requests. After
validating identity, model, cap and status, record the active source operation,
response ID and timing before entering bounded cleanup. An already expired
observation cannot count as the active trigger. Use the existing one-cancel /
one-retrieval sequence without another polling delay. Withhold all stopped
answers, including completion races, while reconciling valid terminal usage.

The live classifier records active trigger, cancellation, cleanup and final
usage separately. A deadline-cleanup cancellation can be observed without
satisfying the direct active-stop criterion. A cancelled response with missing
usage remains unresolved and fails. Neither case can become a false pass.

## Changes

- `accounting.py` and `responses_background.py`: opt-in active stop with explicit
  trigger provenance and existing cancellation/usage handling. Synchronous use
  is rejected before creating an attempt; default behavior is unchanged.
- `cancellation_smoke.py`: one-request runner, source/prompt preflight, separate
  verdict fields, exact fixed output directory and pending/consumed rejection.
  Retains all raw evidence and never generates a speculative follow-up request.
- [100 plan and acceptance](../v1-sprints/100-active-cancellation.md) and
  [pending packet](../v1-sprints/100-cancellation-smoke.json): same model, prompt,
  cap and work window as 099's third request. Seventeen frozen inputs include
  predecessor artifacts used by the local replay. Proposed live allowance is
  $0.01, one request, 4096 output tokens, five seconds plus fifteen for cleanup,
  and zero retries. No new live call occurs during construction.

## Verification

- First failing check: the background transport rejects the absent active-stop
  argument. After construction, replaying the actual 099 queued/in-progress
  prefix yields `create, retrieve, cancel, retrieve`, with no extra poll. The
  subsequent cancellation replies are explicitly synthetic protocol fixtures.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache OPENBOOST_BACKEND=cpu uv run --no-sync pytest tests/v1/test_author_active_cancellation.py tests/v1/test_author_background.py tests/v1/test_author_accounting.py tests/v1/test_author_accounting_smoke.py -n 0 -q`:
  66 pass in 5.65 seconds. Thirteen new cases cover the recorded prefix, normal
  cancellation, null usage, completion races, early completion, expired/invalid
  active observations, retrieval failure, tier mismatch, source/config integrity,
  pending/consumed rejection and one-use output handling. All original cases pass.
- Full CPU regression: 1946 pass, one Linux-only skip in 13.78 seconds. Ruff and
  formatting pass for production/changed support code. No boosting production
  code or dependency changes; the new source closure is separately preflighted.
- Original 099 evidence verifies through its standalone archive checker. Do not
  rehash the consumed old freeze against the changed active controller; its
  historical source snapshots and results remain byte-identical. GPU run-8 and
  earlier Linux evidence are preserved.
- The seventeen-file no-network preflight passes with a $0.0059392 conservative
  estimate below the proposed $0.01 allowance. All 85 run-8 source hashes, 24
  frozen 097 source hashes, 20/21 original Linux archive entries and 84 indexed
  099 original/analysis artifacts match. The consumed 099 packet is unchanged.
  MkDocs builds with its existing `execution.md` evidence-link warning.

## Failed Attempts

The original long-answer/deadline hypothesis is not retried. Only the local stop
trigger changes. Review distinguishes provider cancellation from exercising
that trigger: counting any deadline cancellation as an active-stop success would
misstate the observation. These fields are kept separate in the new classifier.

## Risks and Follow-ups

Local replay proves controller behavior, not actual provider cancellation. A
future response may complete before observation or race the cancel request; keep
either as unexercised, without retry. A cancelled response may lack final usage;
retain its reservation instead of inventing zero. Any live request needs the new
concrete allowance and must end at retrospective. This direct stop does not
establish 20k/1800-second author enforcement, worker integration, attempt authority,
independent author benefit or formal E5. GPU run 8 remains pending.

## Commits

- `421ea54`: original cap-exhaustion result and unexercised cancellation archive.
- This commit contains local active-stop construction and a pending live packet.
