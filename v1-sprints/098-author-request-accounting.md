# Sprint 098: Accounted model requests

Status: local construction follows the user's "continue" after the 097
retrospective. No model call, new worker upload, independent author, GPU run or
external publication is authorized by this slice. Mapping: 069 / exploratory
F2 preparation. The 20,000-generated-token / 1800-second formal budget is unchanged.

## Plan and acceptance

1. Implement a trusted, sequential Responses request boundary. Persist each exact
   request and its reserved output allowance before network dispatch. Clamp its
   output cap to the smaller of the frozen per-request cap and remaining budget.
   Count `usage.output_tokens` once; reasoning is a subset. Missing, inconsistent,
   duplicate or over-cap usage closes the controller and stays unknown.
2. Give the actual HTTPS transport a supervised process deadline. Disable retries,
   redirects, background generation and model tools in this initial text-only
   slice. Keep credentials in the trusted process environment and out of logs.
   Retain raw response bytes, request IDs, failure state and unused reservations.
   A local deadline kills the transport and stops dispatch; it does not by itself
   prove that the provider has stopped generation or finalized usage.
3. Verify budget exhaustion, reasoning-only usage, malformed/missing usage,
   interrupted requests, persistence failure and actual local process timeout.
   Unit fixtures are protocol tests only, never measured model-token evidence.
   Run relevant regression/lint, document remaining integration and commit.

First failing check: after a response uses part of the allowance, a second request
must carry exactly the remaining cap, and exhaustion must prevent a third network
call. No implementation currently owns this pre-dispatch accounting boundary.

## Provider contract and limits

The [Responses create reference](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
defines `max_output_tokens` as a cap including reasoning and visible output. The
[reasoning guide](https://developers.openai.com/api/docs/guides/reasoning) places
reasoning under output usage and documents incomplete responses at the output
limit, including exhaustion before visible text. These support the controller
rule; they are not evidence of enforcement on our actual account/model.

The [background guide](https://developers.openai.com/api/docs/guides/background)
documents cancellation for background responses. This first synchronous path does
not claim confirmed remote cancellation merely because its client is killed.
An interrupted request retains its full reservation as an upper bound, records
unknown generated usage and permanently prevents continuation or retry. A fresh
output directory cannot resume an old attempt; the future dispatch authority must
own attempt identity and prevent reissuing it under a new directory.

The existing core environment has neither `openai` nor `httpx` installed. Use
Python's HTTPS client and process supervision for this narrow protocol without
changing the production dependency closure or GPU run-8 freeze. No API-key tool
is available in this session. Credentials are not inspected during construction;
actual service access remains a separate, concrete smoke request.

## Next real gate

Before requesting model use, freeze a model snapshot and reasoning setting,
input text/hashes, request count and output caps, input/spend ceiling, wall limit,
failure retention and verdict. Require real output-cap exhaustion, actual usage,
no subsequent generation after exhaustion, and an interrupted/deadline outcome
that never becomes zero-token success. Unresolved cancellation/final usage must
remain a failed accounting gate. Separately integrate the verified 097 launcher
for every author tool command, with the total attempt deadline and evaluator
outside the worker. This text-request slice is not the complete author runner.

Then complete fair D1/D2 incumbent paths and model/tool/settings freezes before
authorized independent attempts. Do not expand numerical fixtures, CPU searches,
the isolation probe catalog or sealed-task access as a substitute for that result.

## Local result and reflection

The text-request boundary is implemented and all 32 focused protocol/transport
checks pass. A real local Python process is killed at its deadline; the default
transport also fails before HTTP in a credential-free test environment. Partial
response prefixes remain on disk with an incomplete-body marker after interruption.
These are local code checks. No real generated-token measurement or model run is
archived, and the controller never marks independent dispatch ready.

Review found two accounting counterexamples in the first local implementation:
an unexpected echoed cap was accepted, and nonterminal usage was treated as final.
Both tests failed before correction. Final-state/model/cap validation now precedes
reconciliation; wrong or unfinished responses keep the full pending reservation.
The ledger also distinguishes known usage from failure to persist the final record.

This is the first accounting construction slice, not closure of 069. The next
provider decision is how to obtain confirmed cancellation/final usage for an
interrupted request, then freeze a bounded real smoke with actual model access.
Do not add more synthetic usage cohorts as a substitute for that observation.
The verified 097 launcher is not yet connected to author tool requests; total
attempt authority, fair arms, artifact/edit collection and model settings remain
required. GPU run 8, all its 85 source hashes and both Linux archives are unchanged.

Full CPU regression passes 1912 tests with one Linux-only skip in 13.88 seconds.
Ruff/formatting pass for the changed support code, and MkDocs builds with the
existing `execution.md` evidence-link warning. No push, model generation or remote
worker invocation occurs in this construction slice.
