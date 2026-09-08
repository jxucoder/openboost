# Standalone D1/D2 development verification

These commands prepare Sprint 069's numerical verifier boundary. They do not
dispatch an agent or establish independent author benefit, complete task-card
acceptance, token enforcement or filesystem isolation. The existing cfca092
author packet and all sealed tasks remain unchanged.

## Execution

From the repository, export public fixtures and the trusted judge:

```bash
uv run python -m benchmarks.v1.authoring.export /tmp/d1-d2-verifier
```

The resulting bundle contains `inputs.json`, `expected.json`, `judge.py` and
`manifest.json`. The manifest hashes the three runtime files and the explicit
public mathematical source closure. Reference code is used only during export;
the standalone judge needs Python, NumPy and the installed OpenBoost core wheel.
The installed smoke records versions and hashes of every installed distribution
file, including NumPy's binary dependencies, and retains the three built wheels.

An observation producer receives only `inputs.json` and writes `D1.json` or
`D2.json` plus one `<case-id>.model.json` for each case. The included
`development.py` adapts the known installed `ob_expectile` and `ob_cohort_splits`
extensions. It imports no oracle and receives no expected-answer argument.
Its class names and result layout are development adapter choices; they do not
freeze an author API or restrict an incumbent to the same implementation path.

After collection, invoke each judge separately in a fresh core-only environment:

```bash
python -I /tmp/d1-d2-verifier/judge.py /tmp/d1-d2-verifier /tmp/observations D1
python -I /tmp/d1-d2-verifier/judge.py /tmp/d1-d2-verifier /tmp/observations D2
```

Any missing input, mismatch or load failure exits nonzero. Success prints a
machine-readable result with exact observation/model/manifest hashes. D1 does
not require D2 or vice versa; neither requires D3/D4 extensions. To reproduce
the entire installed development check with separate collector/judge environments:

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache OPENBOOST_BACKEND=cpu \
  uv run --no-sync python -m benchmarks.v1.authoring.verify /tmp/d1-d2-installed
```

The destination must be new. Builds and environment installations use uv offline;
a missing cached dependency fails visibly. Raw commands, outputs, package/file
identities, all models, observations and the deliberate wrong-model failure are
retained. This command runs only designer-authored Python extensions, not an agent.

## Checked numerical scope

- D1: weighted initialization, offsets, tau 0.5 and 0.8, zero-residual curvature,
  missing numeric data, unweighted derivatives, and two exhaustive-tree rounds.
- D2: depthwise, best-first and symmetric growers, constrained root choice and
  predictions, independent information surviving zero objective weight, no legal
  split, two constrained squared rounds and saved scalar inference.
- Verifier schema v2 adds D1 invalid tau/rate observations and D2 wrong shape,
  negative/nonfinite information and foreign problem/row identity rejection.
  Known implementations that accept invalid tau or foreign problem bindings
  fail despite correct numerical training. The v1 archive remains unchanged.
- Judge: exact fields/array lengths, finite numbers, duplicate-key rejection,
  complete rounds, saved output shape, independent model replay and changed-file
  detection against the trusted manifest. Numerical tolerance is fixed at
  `rtol=1e-10, atol=1e-12`; these tiny CPU cases do not set GPU tolerances.

This stage does not yet collect private/core edits, arbitrary algorithm variants,
independent validation, CUDA
attempts or time to first correct result. Existing development tests cover several
of these separately; they are not silently counted as standalone pilot acceptance.

## Trust and remaining preparation

The evaluator bundle and its manifest must be protected outside an author's write
and read scope. Hash checks detect divergence from a trusted manifest; an author
able to rewrite that manifest can defeat them. Separate venvs and `python -I`
demonstrate dependency separation, not host isolation. Collector observations are
not trusted proof of execution in an unisolated environment. This smoke only runs
known local code. Protecting the actual judge, preventing expected-answer access,
and enforcing real generated-token/wall budgets remain explicit prerequisites.
The [runner audit](../evidence/author-runner-audit-094/README.md) records concrete gaps.

Before any independent attempt, finish that actual isolation/accounting smoke,
freeze task adapters and missing checks, select fair incumbent arms and pin the
model/tools/settings. Refresh the author view as a new named revision; do not
reinterpret this development artifact as an author result or an E5 pass.

## Current author view and experimental local isolation

`uv run python -m benchmarks.v1.prepare_author_packet /tmp/author-view` requires a
clean checkout and a new output directory. It copies an explicit set of six public
CPU pages plus the D1/D2 cards, renders omitted links as labeled plain text, and
records original and delivered file hashes. It never follows links to expand the
allowlist. Offline builds retain failures, and every Python wheel entry must match
the source core; extra files outside core/metadata are rejected.

`isolation.py` is an experimental native macOS worker probe, not an author runner.
It accepts that packet, an exported evaluator bundle and a fresh output outside
the repository. It copies the real evaluator inputs, installs a core-only runtime
with independent file copies, and retains all outcomes in `manifest.json`, including
startup failures. It changes no host settings. Each probe runs in a short-lived
subprocess with network denied and writes limited to its work directory.

The first, narrower policy passed six installed public-document checks and denied
the declared evaluator/core accesses, child/symlink/hardlink attempts and a simple
wall timeout. It nevertheless allowed another answer copy outside the denied
locations. The stronger read-allowlist policy aborts the current macOS/x86_64 Python
runtime before Python starts. Do not count that as successful isolation or fall
back silently to the weaker policy. [Sprint 095](../../../v1-sprints/095-author-packet-and-local-isolation.md)
retains these outcomes. The native probe does not establish protection from all IPC
channels, unrelated processes, resource exhaustion or an escaping process session.
Its wall timeout is a worker smoke, not enforcement of the full author budget.

The next runner decision needs a supported execution environment plus real token
accounting. Independent dispatch, incumbent/model/settings freeze and the pending
GPU run remain separate gates.

## Linux worker smoke and explicit identity correction

[Sprint 096](../../../v1-sprints/096-linux-author-worker.md) chooses a Modal CPU
Sandbox for the next bounded isolation smoke. Its original controller verified the
frozen 095 packet and constructed an image from thirteen individually selected
files: twelve author materials and the generic `linux_probe.py`. The evaluator
stays on the controller host. No directory mount, author solution, credential or
model runner is delivered. The image requested UID 1000 with a writable workspace,
root-owned core/materials and outbound networking blocked. The actual approved
smoke runs as root because Modal ignores `USER`, so core/material protection fails.
The [raw result](../evidence/author-linux-isolation-096/README.md) retains all 14/19
passing and five failing checks. The timeout stage is not reached.

Check the proposed run locally without starting Modal:

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache \
  uv run --no-sync python -m benchmarks.v1.authoring.modal_worker
```

The original source/upload closure and one 90-second CPU Sandbox are recorded in
`v1-sprints/096-linux-worker-smoke.json`. Its allowance is consumed and normal
`--execute` reuse is blocked. The original approved freeze, raw output, versions,
core hashes, controller-side evaluator hashes and provider status are archived.
Failed checks remain failures; no retry occurred. Passing
requires all six public examples, all thirteen isolation/integrity observations,
and a provider timeout after a child starts in another process session. A startup
abort or an empty timed-out worker cannot pass.

Local image construction and classifier tests are not remote isolation evidence.
One failed external TCP connection is a limited observation, coupled with the
provider's network policy; it is not a test of every possible network channel.
The full author time/token boundary, fair arms and independent dispatch remain
open. This smoke runs no model and makes no author-cost claim. Follow the
[096 retrospective](../../../v1-sprints/096-linux-worker-result.md) for the next
explicit privilege-drop correction; the original case definitions stay fixed.

[Sprint 097](../../../v1-sprints/097-explicit-worker-identity.md) replaces the ignored
image directive with `linux_launcher.py`, delivered as the fourteenth selected
file. The trusted root entrypoint sets `no_new_privs`, clears supplementary groups,
sets all three UID/GID values to 1000, and checks those values. A fresh interpreter
checks inheritance before executing the unchanged probe. The controller requires
both identity records, the original nineteen checks and actual provider expiry.
The launcher fails before worker code on an unsupported host or failed syscall.
Every future author command must use this entrypoint; direct root command access
must not be exposed to the author. This does not implement the full author runner.

The local-check command above now uses `v1-sprints/097-worker-identity-smoke.json`.
The user's "sure" approves one corrected 90-second / 2-CPU / 2048-MiB smoke, with
no retries, GPU or model calls. Exact inputs are frozen and committed before
service access. Local syscall-order and classifier tests do not establish actual
Linux enforcement; retain the remote outcome separately from the original failure.

The [real 097 result](../evidence/author-linux-identity-097/README.md) now passes all
nineteen original checks, both identity guards and actual provider expiry at clean
`518eccf`. The one allowance is consumed; `--execute` reuse is blocked. All original
thirteen uploads are unchanged, and the five originally failed checks now pass.
Core/material writes and root restoration are denied; evaluator hashes stay intact.
This removes the observed worker-identity blocker. The
[retrospective](../../../v1-sprints/097-worker-identity-result.md) returns to actual
token-budget enforcement and the fair-arm/model/settings packet. No model or
independent author attempt ran, and no broader isolation or author-cost claim follows.

## Trusted request accounting under construction

[Sprint 098](../../../v1-sprints/098-author-request-accounting.md) adds a text-only
Responses boundary in `accounting.py` and `responses_transport.py`. It is evaluator
support code, outside the installed OpenBoost package and author worker image.
Each controller requires an explicit model/setting, token/request limits and wall
deadline. Requests are serialized; the trusted ledger and exact request are saved
before sending a cap no larger than the remaining allowance.

The ledger distinguishes confirmed output totals, a pending reservation and the
complete generated total. The latter stays `null` while a request is unresolved.
Returned output usage includes reasoning; its reasoning breakdown is not added
again. Only terminal responses matching the model and requested cap can supply
final usage. Inconsistent/missing usage or duplicate IDs stop all further requests.
Late complete usage is retained, but the response is not returned to the caller.
Known usage survives a failure to write the final ledger. Failures never trigger
automatic request retries or a reset of the directory's budget.

The actual transport starts an isolated trusted Python process for one HTTPS POST
to the Responses endpoint. It reads `OPENAI_API_KEY` only there, stores no auth
headers or exception messages, follows no redirects and uses no SDK retry loop.
The parent supervises the process with the remaining total deadline. Raw response
bytes, HTTP/request identifiers and failures are retained in the trusted directory;
the body is bounded to 4 MiB plus one overflow-detection byte. No model tools or
worker commands are exposed by this text-only slice.

Local tests verify the protocol and actually kill a sleeping transport process.
The default transport also fails before HTTP when its test environment has no
credential. Injected transports are labeled `injected_protocol_test`; their counts
are fixtures, not model evidence. The original 098 slice used no live model.
Client termination does not confirm provider cancellation or final usage. On an
interruption without final usage, the full reservation remains, generated usage is
unknown and continuation is blocked. A future authority must prevent reissuing the
same attempt under another directory and supervise worker lifetime independently.

The later 099 observation below tests real cap/exhaustion; cancellation remains
open before integrating the verified worker boundary and fair D1/D2 arms. Neither
this controller nor passing local tests closes 069 preparation or formal E5.

[099](../../../v1-sprints/099-background-accounting-smoke.md) adds optional
`background=True` to `Controller`. The trusted transport retains each create,
retrieve and cancel operation. It polls at most sixty times, with one-second
intervals and the original work deadline. On interruption, a known response ID
permits one cancel and one final GET within fifteen additional seconds reserved
for cleanup. Each HTTP child is limited to ten seconds or the remaining relevant
deadline, whichever is smaller. No retry creates another response.

Terminal usage after cancellation can reconcile the ledger, but no stopped
answer is returned, including a completion racing cancellation. Unknown response
identity or final usage leaves its reservation unresolved and closes the
controller. A cancel acknowledgement alone does not prove final usage; current
provider documentation includes a cancelled response with null usage. Abrupt
controller loss can still prevent cleanup. Actual cancellation and complete
author integration remain unverified.

The [099 live packet](../../../v1-sprints/099-accounting-smoke.json) now fixes
three integer-list prompts, the model alias/settings, twelve source hashes and
all request/token/time bounds. Its allowance is consumed and its frozen hashes
identify the historical execution revision. The active controller advances in
100; verify 099's unchanged archived evidence without network use:

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python benchmarks/v1/evidence/author-accounting-099/verify.py --check
```

The [run request and acceptance](../../../v1-sprints/099-background-accounting-smoke.md)
explain the proposed $0.05 allowance, conservative $0.0082176 token estimate,
one-use output directory, failure retention and retrospective boundary. The
runner rejects pending/consumed authorization, dirty or changed sources and output
reuse. Exhaustion must pass before the cancellation case starts. A different
returned service tier or excess reported input stops further generation. No
author code, evaluator, repository source or dataset enters these model requests.

The [actual 099 result](../evidence/author-accounting-099/README.md) at clean
`5c0f31a` passes cap exhaustion: 128 then 64 actual reasoning tokens, reconciled
usage and further dispatch blocked. The cancellation probe finishes before the
deadline with 104 output tokens, so cancellation is unexercised and the frozen
overall verdict fails. All three responses are retained, totaling 296 output
tokens. There is no retry or new allowance. The
[retrospective](../../../v1-sprints/099-accounting-result.md) calls for an explicit
stop trigger after observing an active response, followed by a separately frozen
live test. Full author-budget enforcement and independent author benefit remain open.

[100](../../../v1-sprints/100-active-cancellation.md) adds
`Controller(..., background=True, stop_on_in_progress=True)`. After a valid
in-progress observation arrives before the work deadline, the trusted transport
persists its source operation/response ID and timing, then calls cancel and a
final retrieval without another polling interval. Cleanup uses the same bounded
fifteen seconds. This policy is rejected for synchronous requests; the default
background policy is unchanged.

Cancelled status, a recorded active trigger and known final usage are separate
observations. Stopped answers are withheld even if completion races cancellation.
An expired observation cannot satisfy the active trigger. Unknown cancelled
usage retains its reservation. This direct stop does not establish full author
wall-budget enforcement.

The [pending packet](../../../v1-sprints/100-cancellation-smoke.json) proposes one
request with 099's exact third prompt, model, 4096-output-token cap and five-second
window, under a $0.01 allowance. Current no-network preflight:

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.authoring.cancellation_smoke
```

The [run request](../../../v1-sprints/100-active-cancellation.md) fixes acceptance,
seventeen source files, cleanup and the one-use directory. No new model call or
allowance is implied by the local protocol results. Retain 099's unexercised
cancellation failure and stop after the next separately approved observation.
