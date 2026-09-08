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
gates. No model call, remote upload, GPU run, independent attempt or push occurred.

## Commits

- Pending verified preparation commit.
