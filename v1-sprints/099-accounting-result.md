# Sprint 099 result and retrospective

Status: the user's "continue" approved the exact frozen live smoke, committed at
clean `5c0f31ab1ec1c93936396bd40a2f5be6e02f5d77`. Its one allowance is consumed.
The [original evidence](../benchmarks/v1/evidence/author-accounting-099/README.md)
retains the failed overall verdict. No retry, additional model request, worker,
GPU run, independent author or push occurs. The twelve frozen sources and prompts
are unchanged; only the active authorization advances to `consumed`.

## Measured result

| Case | Requested output cap | Actual input / output / reasoning tokens | Result |
|---|---:|---:|---|
| Exhaustion request 1 | 128 | 39 / 128 / 128 | Incomplete at `max_output_tokens` |
| Exhaustion request 2 | 64 | 28 / 64 / 64 | Incomplete at `max_output_tokens` |
| Cancellation probe | 4096 | 40 / 104 / 83 | Completed before the deadline; no cancellation |

The exhaustion case passes its frozen criterion. Both actual outputs consist
entirely of reasoning tokens, with no visible answer. The second cap is clamped
to the remaining 64 tokens; final usage totals 192, reservations clear, and a
subsequent request is rejected before transport. This is real bounded cap/usage
evidence, including reasoning-only output, on the frozen model/account path.

The cancellation probe fails its criterion. It follows `queued -> in_progress ->
completed`, completing within about 3.55 seconds of controller start, before the
five-second work deadline. Its short answer states that the entire requested list
cannot fit in one response. The controller correctly returns that timely answer,
counts its 104 output tokens, reaches its one-request limit and rejects another
request. No cancel operation or cleanup interval occurs. This result neither
passes nor falsifies provider cancellation/final usage after cancellation.

Across both cases there are three creates, seven retrievals and zero cancels,
all ten HTTP responses complete with status 200. The run takes 12.713916 seconds
including local source archival, with 107 input tokens and 296 output tokens;
275 reasoning tokens are already included in that output total. All returned
model IDs and service tiers match the requested alias and `default` tier. The
frozen conservative rates yield a token-cost estimate of $0.00038195. This is a
calculation from recorded usage, not a verified invoice.

The exact model identifier remains an alias, not an immutable snapshot. Python
3.12.12 and OpenSSL 3.5.4 run on macOS 26.3 with sixteen reported logical CPUs;
there is no GPU or dataset. Raw returned defaults, including reasoning settings
and prompt-cache retention, are preserved. This run does not audit actual server
retention or establish stable model-version control for an author cohort.

## What changed in our evidence

The repository now has a real observation of bounded output-cap exhaustion and
complete final usage through the trusted background transport. That replaces one
protocol-only assumption with a measured result. It does not finish 069: the
live interruption/cancellation gate is still unexercised, the full 20k/1800-second
budget is untested, and worker commands, global attempt authority, edit/artifact
collection and fair D1/D2 arms remain unconnected. Independent author benefit and
formal E5 remain open. The required foundation/application/device scope is intact.

The known-code Linux identity result and this token observation are separate
pieces of preparation; neither establishes isolation/accounting of a real author
attempt. GPU run 8 and all its 85 frozen source hashes remain unchanged and pending.
CPU search and speculative CPU optimization remain paused.

## Counterexample and next decision

The failed hypothesis was that requesting a very long answer would keep a cheap
model generating until the chosen deadline. The model can instead finish with a
short capacity explanation. More requested output or higher reasoning would be
an adaptive attempt to force the outcome and is not authorized by this freeze.
The failed overall verdict is correct and must not be relabeled as a pass merely
because all final usage was known.

Next local design should make the cancellation trigger explicit: after a valid
response is observed in progress, request stop immediately and use the bounded
cancel/retrieve path. Retain terminal-completion races as unexercised outcomes.
The current raw sequence shows an in-progress observation was available; it does
not prove the provider would still be active when a future cancel arrives.
Keep that direct-cancellation observation separate from deadline enforcement.
Do not claim that an immediate stop verifies the full author wall limit.

Before another live invocation, construct and locally verify that bounded trigger,
freeze a new concrete allowance and preserve this run as its predecessor. If
cancelled responses have null final usage, identify a provider-supported usage
reconciliation path instead of substituting estimates or zeros. There is no new
model or hardware allowance in this retrospective. Stop here for the planned
review before further construction.

## Verification and preservation

The original `archive.json` verifies 80 original files, including twelve source
copies. It is itself preserved, giving 81 original files. The standalone evidence
verifier reads raw HTTP bodies, canonical responses, request bytes, ledgers and
the frozen packet; it reproduces usage totals and the failed overall verdict
without HTTP or model execution. An additional index covers the original archive
and the verifier, command record and derived analysis. Model output and original
source snapshots are unchanged.

The ten focused smoke-harness tests pass before dispatch and after consumption;
the no-network preflight verifies the consumed packet. Lint and formatting pass
for the verifier. No production code or dependency changes in this result slice.
The prior full CPU regression remains 1933 passed with one Linux-only skip; that
suite is not rerun merely to archive a result. Documentation builds with its
existing `execution.md` evidence-link warning.
