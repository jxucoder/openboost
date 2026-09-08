# 2026-09-07: Reserve model output before dispatch

## Context

The [097 corrected worker smoke](../v1-sprints/097-worker-identity-result.md) passes
its declared Linux identity/isolation gate. Independent authoring still lacks
actual generated-token enforcement and complete dispatch settings. The earlier
Responses proposal supplies a request-level output cap including reasoning, but
no trusted code previously owned its reservation and reconciliation.

## Decision or Result

Construct one sequential text-request boundary outside the author image. Persist
the request and cap before dispatch; reserve the smaller of the per-request limit
and remaining attempt allowance. Validate final response identity, cap and usage
before releasing the reservation. Reasoning is included in output usage, not an
additional charge against the generated-token count. An unfinished response's
usage is not a final total. Missing/interrupted usage stops continuation and remains
unknown, with confirmed prior usage and the outstanding reservation kept separate.

Use a supervised trusted Python HTTPS process without SDK retries or redirects.
This verifies a local wall boundary independently of socket inactivity timeouts.
It does not prove provider cancellation or final usage after disconnect. Keep that
remaining gate explicit instead of reporting an interrupted response as zero cost.

## Changes

- `accounting.py`: explicit bounded limits, durable ledger/request writes, serialized
  dispatch, response reconciliation and closed failure states. Known usage survives
  final-ledger failure; existing attempt directories cannot reset their budget.
- `responses_transport.py`: fixed HTTPS endpoint, raw body and HTTP/request metadata,
  isolated trusted subprocess, bounded response size and local deadline supervision.
  Only the trusted child reads the key; auth headers and exception text are omitted.
- [098 plan](../v1-sprints/098-author-request-accounting.md): records provider sources,
  local acceptance and remaining real smoke/worker/attempt-authority integration.
  Existing worker freezes, production code and pending GPU run 8 are unchanged.

## Verification

- First failing check: the remaining-budget protocol test cannot import an
  accounting implementation. After construction it observes caps 8 then 3 for a
  ten-token fixture allowance, counts reasoning once and prevents a third request.
- Review then exposes two local counterexamples: a different returned cap was
  accepted, and in-progress usage was recorded as final before rejecting its status.
  Both checks fail before the correction, then require unknown usage with the full
  reservation retained. No remote response is claimed for these protocol fixtures.
- Focused command: `UV_CACHE_DIR=/tmp/openboost-research-uv-cache OPENBOOST_BACKEND=cpu uv run --no-sync pytest tests/v1/test_author_accounting.py -n 0 -q`.
  All 32 pass in 1.28 seconds. Covers malformed/missing/duplicate usage, reasoning-only exhaustion, interrupted
  and late requests, persistence failure, default credential-free startup rejection,
  request-count exhaustion, redirects/HTTP failures and secret-safe failure records.
- The actual local deadline test starts a sleeping Python process, expires its
  one-second allowance and confirms the PID no longer exists. It runs no provider
  request and does not establish remote cancellation or author-workspace isolation.
- Interrupted HTTP protocol tests retain the received response prefix and an
  incomplete-body marker without logging a fixture credential. All 85 pending
  GPU source hashes, all 24 frozen 097 source hashes and the 20/21 indexed files
  in the original/corrected Linux archives remain unchanged. Production code and
  package dependencies are untouched.
- Full CPU regression: `UV_CACHE_DIR=/tmp/openboost-research-uv-cache OPENBOOST_BACKEND=cpu uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' --tb=short -q`:
  1912 pass, one Linux-only skip in 13.88 seconds. Ruff and formatting checks pass
  for production/changed support files. MkDocs builds with the pre-existing
  `execution.md` evidence-link warning. Authored whitespace checks pass.

## Failed Attempts

Local protocol counters are useful for testing code, not for closing the project's
real token gate. The initial implementation confused nonterminal usage with final
usage; the correction moves final-state/cap validation before reconciliation.
The existing Codex event-interface audit remains unresolved. No actual API key
or external service was used in these tests; credential values are explicit fixtures.

## Risks and Follow-ups

Pin model/input/settings and a concrete spend/request/output allowance before an
actual exhaustion smoke. Resolve or measure cancellation and final usage at the
provider boundary; killing the local client is insufficient evidence. Integrate
the 097 launcher for all author tools under the full attempt deadline, retain
edit/artifact accounting and prevent repeated attempts under new directories.
Then use fair frozen D1/D2 arms for authorized independent attempts. This slice
does not change the 20k/1800-second formal budget or establish author benefit.

## Commits

- Parent `aa1664d`: passing known-code Linux identity smoke and retrospective.
- This commit contains locally verified request accounting, not a model-run result.
