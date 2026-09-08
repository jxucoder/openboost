# Sprint 099: Background cancellation and a bounded accounting smoke

Status: local construction under the user's "continue". Real model generation
requires the separate, concrete allowance below; no worker upload or GPU run is
part of this slice. Mapping: 069 / exploratory F2 preparation. The formal author
budget remains 20,000 generated tokens or 1800 seconds, whichever is reached first.

## Construction plan

1. Extend the trusted HTTPS path to the Responses create, retrieve and cancel
   routes only. Retain each operation and raw reply. Add bounded background
   polling, then at most one cancel and one final retrieval on interruption.
   A cleanup interval reconciles usage; it grants no further author work.
2. Connect cancellation to the existing reservation ledger. Count only verified
   terminal usage; keep unknown usage and its reservation after any unresolved
   interruption. A completion racing cancellation must never release late work.
3. Build a fixed-input smoke command with source hashes, request/output/wall
   bounds, conservative cost estimate and distinct exhaustion/cancellation
   verdicts. Local protocol checks precede a separately approved live run.
   Commit verified slices and stop at a reviewable run request.

First failing check: a background response still running at the work deadline
must receive cancellation and final retrieval; its final usage must reconcile
without returning a usable model answer. A transport disconnect alone cannot
pass this check.

## Provider decisions

The [background guide](https://developers.openai.com/api/docs/guides/background)
documents asynchronous creation, retrieval and idempotent cancellation. This
path uses `background=true`, `stream=false`, `store=false`; the current guide
describes temporary retention for polling even with `store=false`. Account
behavior is unverified. A terminal cancel acknowledgement with missing usage
does not close the accounting gate. One final GET records whether usage becomes
available; unknown usage remains unknown, with no new generation or automatic
retry. If creation never yields a valid response ID, cancellation is unresolved.

The proposed [GPT-5.6 Luna model](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
is an economical protocol-smoke choice, not the author-comparison model. The
current page lists only the `gpt-5.6-luna` alias; do not invent an immutable dated
snapshot. Freeze that exact identifier, require the returned identity to match,
and retain the alias limitation. Fair D1/D2 arms and model/version control for
independent author attempts remain separate work.

## Local cancellation result

Eleven new protocol checks plus the 32 original accounting checks pass. They
exercise actual trusted HTTPS method/path construction through a local connection
fixture, completed background polling, cancellation, a completion race, unknown
creation identity, final retrieval failure and bounded cleanup. These are local
protocol results, not measured provider usage. The existing real local transport
process deadline test remains part of the passing suite.

The cancellation reference itself gives a cancelled response with `usage=null`.
This is a concrete reason to keep the live verdict open: a confirmed stop may
still leave generated usage unresolved. A 15-second cleanup interval permits at
most one cancel and one final GET, with no new generation. All returned answers
from an interrupted transport are withheld, including a completion race that
reports valid usage. Missing usage retains the reservation and closes dispatch.
