# Sprint 100: Cancel an observed active response

Status: local construction under the user's "continue" after the
[099 retrospective](099-accounting-result.md). No live generation is authorized
by this construction step. The previous allowance is consumed. This remains
069 accounting preparation, with no author attempt, worker upload or GPU run.

## Plan and acceptance

1. Add an explicit trusted stop policy to the background transport and controller.
   After a validated `in_progress` response arrives before the work deadline,
   persist the triggering observation and immediately enter existing bounded
   cancellation/retrieval. Do not poll again before requesting cancellation.
2. Verify the call path with 099's actual queued/in-progress prefix and explicitly
   synthetic cancellation replies. Keep unknown final usage unresolved, withhold
   stopped answers, distinguish completion races and preserve default behavior.
3. Freeze a separate one-request live smoke, including sources, exact request,
   allowance and verdict. Run local regression/lint and commit before requesting
   live approval. Preserve all original 099 evidence and its failed verdict.

First failing check: replay 099's queued/in-progress prefix through the trusted
controller and assert the next operation is cancel, followed by one final GET.
No active-stop policy currently exists. Replay plus a constructed cancel reply
is a local protocol check, not observed provider cancellation.

## Boundary

Use `stop_on_in_progress=True` only with background requests. Default behavior
is unchanged. Validate response identity/model/cap/status before triggering stop;
an expired observation may enter deadline cleanup but cannot count as the active
trigger. Record the response ID, source operation, transport elapsed time and remaining
work allowance before cancellation. Cleanup still permits one cancel and one
final GET within fifteen seconds; it does not create another response.

The controller reconciles valid terminal usage and withholds every stopped
answer, including completed responses racing cancellation. A completed response
before any active observation remains an unexercised cancellation outcome.
Missing cancelled usage stays unknown, with the original reservation retained.
This direct-stop operation does not establish the full author wall limit.

The new live packet will retain 099's third request unchanged: `gpt-5.6-luna`
alias, medium reasoning, default tier, the same 163-byte integer-list prompt,
4096 output tokens and a five-second work window. Only the trusted stop trigger
changes. One request plus at most fifteen seconds of cleanup is proposed under
a $0.01 allowance, with no retries. Model identity is still an alias and provider
cancelled usage may remain unavailable. The live verdict must keep those limits.

## Concrete live packet: pending approval

[100-cancellation-smoke.json](100-cancellation-smoke.json) freezes seventeen files:
the controller/transport/runner, tests, this plan, dependency lock and selected
099 predecessor artifacts used by the replay. The old approved request's model,
prompt and output cap are checked against the new configuration. The original
099 archive and consumed freeze are not rewritten as the active code advances.

| Bound | Value |
|---|---|
| Generation | One create, at most 4096 output tokens including reasoning |
| Model/settings | `gpt-5.6-luna` alias, medium reasoning, default service tier |
| Input | Same 163-byte ASCII prompt as 099's third request; no tools/history/files |
| Trigger | First valid in-progress observation received within five seconds |
| Polling | One-second intervals, at most sixty polls, clipped to the work deadline |
| HTTP | Ten seconds per operation, clipped to work or cleanup deadline |
| Cleanup | One cancel and one final GET within a shared fifteen seconds |
| Time | Five-second work window plus at most fifteen seconds cleanup; local setup/archive separate |
| Input review | Stop/fail if reported input exceeds 4096 tokens or returned tier differs |
| Cost request | $0.01 allowance; conservative estimate $0.0059392 at 099's frozen rates |
| Reuse | Zero retries; fixed fresh `/tmp/openboost-author-cancellation-100` directory |
| Other work | No worker, GPU, independent author, tool action or publication |

The conservative estimate reserves 4096 input tokens at $0.25 per million and
4096 output tokens at $1.20 per million, retaining 099's pricing assumptions.
It is not an invoice or provider account-level dollar limit. The exact small
prompt is checked before dispatch; actual input and returned tier are checked
afterward. The model alias limitation and `store=false` behavior remain as in
099. No claim about actual server retention follows from this packet.

No-network preflight:

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.authoring.cancellation_smoke
```

After explicit approval, change only the packet's authorization to `approved`,
record the approval and commit before running once from the clean revision:

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.authoring.cancellation_smoke --execute
```

The CLI has one fixed output location; an existing directory blocks reuse.
Preserve it after failure. Do not change the prompt, model, cap or deadline,
silently retry, remove that guard or interpret unspent money as another run.
This is a local smoke guard, not the future authority for independent authors.

## Verdict and retrospective

Report the active trigger, provider's terminal cancellation status, cleanup
completion and final usage separately. Pass only when the recorded active
observation references its validated source response before the deadline, the
trigger causes one successful cancel and final retrieval, the final state is
cancelled, terminal usage reconciles within the frozen bounds, the answer is
withheld and a further request is rejected locally. Known usage returned during
cleanup is valid for reconciliation even if the original work deadline passes;
it must not become usable author output.

A cancelled response with null usage reports observed cancellation and unknown
usage, retains the full reservation and fails. Early completion or a completion
race is an unexercised cancellation outcome and fails. A queued request cancelled
only by deadline cleanup can report cancellation but fails this active-trigger
criterion. Invalid active observations, mismatched tier/input, failed control
operations or unresolved creation identity also fail. No estimated token count
may replace missing provider usage.

Retain exact requests, raw HTTP replies, request IDs, operation sequence, trigger
record, final ledger, model/settings/environment, source snapshot and hashes.
No credentials, headers or exception messages are added to logs. Execute once
and stop for retrospective regardless of outcome. If final cancelled usage is
unavailable, investigate supported reconciliation before any author dispatch.
Do not conflate this direct stop with full 20k-token/1800-second enforcement.

## Local result and reflection

The first test fails because the active policy is absent. After construction,
099's measured queued/in-progress prefix causes immediate cancellation in the
local replay, followed by exactly one retrieval. The cancellation responses in
that replay are explicitly synthetic. Default behavior and original 099 outcome
checks continue to pass. The classifier separates the stop trigger from provider
cancellation: deadline cleanup can cancel a response without exercising this
trigger, and a cancelled response can still have unknown usage.

The next useful observation is the single frozen live test. This construction
does not supply a real cancellation result, independent author evidence or a new
GPU allowance. Original 099 evidence is checked through its archive verifier;
the consumed old packet continues to identify its historical source revision,
not the subsequently changed active controller files.

All 66 focused accounting/background/harness checks pass in 5.65 seconds. Full
CPU regression passes 1946 tests with one Linux-only skip in 13.78 seconds.
Ruff and formatting pass. No production boosting code or dependency changes.
The seventeen-file packet preflight passes without network use. MkDocs builds
with the existing `execution.md` evidence-link warning. The one-request live
observation and its result remain pending.
