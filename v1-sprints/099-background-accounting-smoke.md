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

## Frozen live request: pending approval

The runnable packet is [099-accounting-smoke.json](099-accounting-smoke.json).
Its twelve source hashes include the runner, transport, accounting, tests, this
plan and dependency lock. Its three exact ASCII prompts request lists of integers;
no repository, task card, evaluator, dataset or author material is sent to the API.
The prompts are frozen in the packet and checked against the runner before use.

| Field | Frozen bound |
|---|---|
| Model | `gpt-5.6-luna` alias, medium reasoning; no immutable snapshot claim |
| Processing | Responses HTTPS; explicit `service_tier=default`; no tools/history |
| Generation count | At most three creates; zero retries |
| Exhaustion | Two requests: 128 then 64 output tokens if the first exhausts its cap |
| Shared exhaustion allowance | 192 output tokens, 60 seconds for both requests |
| Cancellation | One request, 4096 output tokens, five-second work deadline |
| Total output reservation | At most 4288 tokens, including reasoning |
| Input | Each exact prompt is ASCII and below 1024 bytes |
| Input review bound | 4096 reported input tokens per response; stop if exceeded |
| Polls | At most sixty per create, one-second intervals |
| HTTP | Ten seconds per operation, clipped to work/cleanup deadline |
| Cleanup | At most one cancel and one final GET in fifteen seconds; no generation |
| Time | At most 80 seconds of work/cleanup windows, plus local setup/archive time |
| Proposed cost allowance | $0.05; conservative token estimate $0.0082176 |
| Output | `/tmp/openboost-author-accounting-099`, fresh and fixed for one use |
| Other resources | Local trusted controller; no Modal, worker, GPU or independent author |

The estimate uses the current model page's $0.20 input / $1.20 output per million
tokens, conservatively raising all input to $0.25 for its stated cache-write
multiplier. It assumes 4096 input tokens per response, well above these short
prompts. The byte bound is enforced before dispatch; the reported-input check is
after dispatch. This is a conservative estimate and requested allowance, not a
provider account-level hard dollar limit or an actual invoice. An unexpected
returned service tier or input count stops before another generation. Preserve
actual usage and returned tier in the archive. No adaptive prompt enlargement,
automatic retry or model substitution is permitted if access or an API option fails.

No-network preflight:

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.authoring.accounting_smoke
```

After explicit approval, change only the packet's authorization to `approved`,
commit, verify the clean source tree and execute once:

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.authoring.accounting_smoke --execute /tmp/openboost-author-accounting-099
```

The existing output directory blocks reuse. Do not delete it or issue a new copy
of this smoke to bypass the allowance. This local one-use guard is not the future
global authority for independent author attempts. Mark the allowance consumed
and archive the original outcome before any later proposal.

## Live acceptance and retrospective boundary

Exhaustion passes only with two real incomplete responses specifically reporting
`max_output_tokens`, exact requested caps 128/64, valid final usage summing to
192, and rejection of a subsequent request before transport. Early completion,
unknown usage, changed model/tier or excess input is a failed gate. Do not launch
the cancellation case unless exhaustion passes.

Cancellation passes only with observed provider cancellation, valid terminal
usage, a reconciled reservation, no returned answer after the five-second work
deadline and a subsequent request rejected locally. Preserve cancellation status
separately from accounting: `cancelled` with null usage is a failure, with unknown
generated tokens. Completion before the deadline or a completion race does not
establish cancellation; retain it as a failed smoke outcome and stop without retry.
If the initial response ID is lost or cleanup cannot finish, retain the raw prefix,
operation records and unresolved reservation. A client timeout is not a provider
stop guarantee. Both passing cases are required for this bounded real smoke.

Retain the clean revision, source snapshot, exact request bytes, raw response
bodies, HTTP request IDs, final ledgers, operation sequence, Python/OpenSSL/OS,
timestamps and artifact hashes. Credential headers and exception messages are
excluded. The provider may temporarily retain background data with `store=false`;
the account's actual retention configuration is not verified. Original raw results
remain local until reviewed for the committed evidence archive.

Stop for reflection after this one live allowance, including failure. Do not
replace a null usage field with an estimate, upgrade models to force a pass, or
launch authors. If cancellation lacks final usage, design a provider-supported
reconciliation path before full author execution. If both pass, next connect
trusted worker commands, total attempt authority and frozen fair D1/D2 arms. Formal
E5, full 20k/1800-second enforcement and author benefit remain unverified. GPU run 8
and all existing CPU/GPU evidence are unchanged.

Local construction is complete: all 53 focused accounting/background/harness
checks pass; full CPU regression passes 1933 with one Linux-only skip. Review
exposed a returned-tier counterexample before correction, and the harness now
stops before another generation on that mismatch. Ruff and formatting pass.
The frozen-source preflight passes without network use. The three exact prompts
are 151, 89 and 163 ASCII bytes. MkDocs builds with the existing `execution.md`
evidence-link warning. Actual provider accounting
and the live retrospective remain pending; no model dispatch has occurred.
