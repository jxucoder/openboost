# Bounded provider accounting smoke: cap passes, cancellation unexercised

At clean `5c0f31ab1ec1c93936396bd40a2f5be6e02f5d77`, the separately approved live
smoke returns exit code 1 and an overall failed verdict. The one allowance is
consumed, with no retry. This directory retains all original run files; it is
provider protocol evidence, not an independent author attempt or boosting result.

| Observation | Actual result |
|---|---|
| Model and tier | `gpt-5.6-luna` alias, medium reasoning, `default` |
| First exhaustion request | Cap 128, actual output 128; all reasoning |
| Second exhaustion request | Cap clamped to 64 remaining, actual output 64; all reasoning |
| Exhaustion decision | Both incomplete at `max_output_tokens`; 192 total; next request blocked |
| Cancellation probe | Completed within approximately 3.55 seconds, before five-second deadline |
| Cancellation observation | Zero cancel operations; the gate is unexercised and fails the freeze |
| Total usage | 107 input, 296 output, including 275 reasoning tokens |
| HTTP operations | Three creates, seven retrievals, all complete HTTP 200 |
| Wall time | 12.713916 seconds including local archival |

The cancellation probe returns a short explanation that the requested 100,000
integers cannot fit in one response. Its raw sequence contains `queued`,
`in_progress`, then `completed`. The controller returns this timely answer and
rejects the next request at its one-request limit. There is no interrupted
response, cancellation acknowledgement or final cancelled usage to judge.
Complete known usage does not turn the failed cancellation gate into a pass.

The frozen conservative rates applied to actual usage yield $0.00038195. This is
an estimate, not a verified invoice. All returned service tiers are `default`,
all input counts are below the review bound, and cached/cache-write input counts
are zero. The model is an alias; no immutable version or wider author-budget
claim follows. See the [retrospective](../../../../v1-sprints/099-accounting-result.md).

## Reproduction and immutable originals

The original command is recorded in `controller-command.json`:

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.authoring.accounting_smoke --execute /tmp/openboost-author-accounting-099
```

This is historical; the active packet is consumed and its fixed output directory
already exists. Do not rerun, remove that guard or change prompts/model to force
the missing cancellation outcome.

`archive.json` is the untouched runner index for 80 files: approved freeze, clean
revision/environment/result, twelve exact sources, three requests and their
ledgers, canonical responses, HTTP bodies/metadata and transport logs. Preserving
that index gives 81 original files. `archive-index.json` additionally hashes that
original index and the post-run command record, verifier and analysis. Neither
the README nor the additional index hashes itself. Credential headers are absent;
exception-message logging is disabled. Only the three frozen public prompts were
sent to the model; archived source files remain local.

Verify original hashes, request/response consistency, actual usage and the failed
verdict without network access:

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python benchmarks/v1/evidence/author-accounting-099/verify.py --check
```

The verifier derives `analysis.json` from the original files. It does not import
the production controller or invoke its request path. The original 099 source
freeze and all previous GPU/Linux evidence remain unchanged.
