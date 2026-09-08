# 2026-09-07: Explicit Linux author-worker delivery

## Context

[095](../v1-sprints/095-author-packet-and-local-isolation.md) disproved a native
path-denial boundary and retained the stronger profile's interpreter abort. The
next step is a supported worker environment, not more macOS policy exceptions.

## Decision or Result

Use a Modal CPU Sandbox with an explicit thirteen-file image closure, non-root
execution and no evaluator in its filesystem. The installed `modal==1.3.0.post1`
can construct the planned image using its public SDK. Local construction does not
build or upload it. A real remote isolation result remains unmeasured.

The current Codex token audit remains blocked. The official Responses interface
provides a concrete output cap including reasoning; a separate actual controller
smoke is needed before selecting that path for independent attempts. Do not add
simulated token events or reinterpret host-generated activity as author evidence.

## Changes

- `authoring/modal_worker.py`: verifies packet/source hashes, rejects unlisted or
  symlink inputs, stages bytes individually, constructs the pinned SDK image and
  retains provider outcomes. Pending authorization fails before service access.
- `authoring/linux_probe.py`: six public examples and thirteen declared isolation/
  integrity checks, then a child in a separate session and forced Sandbox expiry.
  No task-specific solution or evaluator byte is supplied to the worker.
- [096 plan](../v1-sprints/096-linux-author-worker.md): CPU resource bounds and
  acceptance distinguish local preparation, measured isolation and full author
  readiness. Original numerical cohorts and the GPU run-8 freeze are unchanged.

## Verification

- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache OPENBOOST_BACKEND=cpu uv run --no-sync pytest tests/v1/test_author_linux_worker.py -n 0 -q`:
  25 pass. Real current packet staging and installed-SDK image construction work;
  tests reject unlisted/remapped/changed/missing/symlink files, traversal, staging
  mutation, pending execution and incomplete/false result streams. A missing
  protected file still retains a failed gate and the raw log hashes.
- All 59 focused author-preparation checks pass. Full CPU regression:
  `UV_CACHE_DIR=/tmp/openboost-research-uv-cache OPENBOOST_BACKEND=cpu uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' --tb=short -q`:
  1853 passed, one Linux-only skip in 13.35 seconds. This duration is a test record,
  not a product performance claim.
- Ruff and `git diff --check` pass. MkDocs builds with the existing
  `execution.md` external-evidence link warning.
- At clean `cda1947`, the [local preparation archive](../benchmarks/v1/evidence/author-linux-preparation-096/README.md)
  retains the successful CLI check and installed SDK signatures/image construction.
  All 22 CPU smoke inputs and all 85 run-8 sources match their hashes. No image
  build, upload or remote run occurs. After adding build-log retention, the 25
  focused worker tests and Ruff also pass; remote behavior remains unverified.

## Failed Attempts

No new native-policy or remote attempt. The prior failed isolation artifacts stay
immutable. Synthetic classifier records exercise rejection logic only, not real
timeouts, token accounting or provider isolation.

## Risks and Follow-ups

The Dockerfile user, file permissions, usable Linux execution and provider lifetime
must pass on the actual service before any author attempt. Image provisioning is
separate from the 90-second Sandbox lifetime; record its latency/logs. Credential
checks cover the explicitly named model/service variables, not every hypothetical
secret source. No host environment or credentials are copied into the image.

After the one frozen smoke, reflect on its result; do not expand the probe catalog
as a substitute for independent authoring. Actual generated-token exhaustion,
model/settings, fair incumbent paths and edit accounting remain the next author
gates. During local preparation, no model call, remote upload, GPU run, independent
attempt or push occurred. The subsequent separately approved smoke is recorded below.

## Commits

- `cda1947` — verified Linux worker preparation and pending CPU smoke freeze.
- Subsequent evidence commit — clean-source local audit; no remote result.

## Subsequent bounded execution approval

The user replied "continue" to the explicit thirteen-file/90-second CPU smoke
request. [Authorization record](../v1-sprints/096-linux-worker-authorization.md)
binds this to the original payload and one invocation. Only the freeze's
authorization value changes; the GPU run-8 allowance remains pending. The local
CLI check passes, comparison against `251edb4` confirms that sole change, and
all 22 CPU smoke / 85 GPU run-8 source hashes match before dispatch. The declared
temporary output does not already exist. `git diff --check` passes.

## Real Linux result and reflection

At clean `7985645`, the [single approved smoke](../benchmarks/v1/evidence/author-linux-isolation-096/README.md)
runs once: 14/19 checks pass. All six public examples work. Five failures share
the incorrect worker identity: UID/GID 0, writable core/materials, successful root
restoration and a changed core file. The changed `__init__.py` hash matches the
probe's deliberate `b"tamper"` write. The controller's four evaluator hashes are
unchanged; expected-answer paths and child/symlink reads are unavailable.

The build log explicitly skips `USER`. The [provider's documented contract](https://modal.com/docs/guide/existing-images#user)
requires process-level privilege reduction. Relying on a Dockerfile directive
was our harness mistake; the earlier SDK-construction check never verified that
behavior. Check command semantics, not just accepted method parameters, before
spending a remote allowance. Preserve this failure rather than remove its gates.

The worker exits 1 before the timeout marker, so lifetime enforcement and child
cleanup remain unverified. The application creates one Sandbox and does not retry.
All seventeen raw artifacts match their manifest hashes; the exact approved
freeze is retained. Active authorization becomes `consumed`, with no payload
changes. All 22 original CPU source hashes and 85 run-8 hashes still match.

The [retrospective](../v1-sprints/096-linux-worker-result.md) specifies explicit
supplementary-group/UID/GID reduction and `no_new_privs` before every author command.
Stop at this checkpoint: no patched worker or additional remote attempt is claimed.
The latest CPU regression remains 1853 passed/one Linux-only skip; this turn changes
authorization, evidence and documentation only. Artifact replay and documentation
checks pass at closure: all twenty indexed files match; replay rejects the actual
failure both with its real non-timeout status and with a deliberately false timeout
claim. Comparing the approved and consumed freezes confirms no payload change.
The consumed CLI exits locally before creating output or accessing Modal. MkDocs
builds with the existing external-evidence link warning. Authored-file whitespace
checks pass. The exact provider `build.log` retains two trailing-space lines and
its final blank line; those raw-byte whitespace warnings are intentional and the
staged log hash matches the original. All twenty indexed artifacts are present
in the staged Git blobs with their original hashes, including the ignored log,
wheel and wheel-directory `.gitignore`, which required explicit staging.
No model, independent author, GPU or push occurred.
