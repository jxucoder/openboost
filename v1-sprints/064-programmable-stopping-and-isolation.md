# Sprint 064: Programmable stopping completion

Baseline: `df23796`. Status: planned; implementation has not started.
Mapping: N1 / F1 / B11 / C4 / E1–E2 development conformance.
Entry: current CPU result and stopping contracts. Ready to implement.

This card now contains the first slice of the earlier Sprint 064 plan. Installed
isolation moves to [Sprint 065](065-installed-run-isolation.md); no completed work
or evidence is moved. See the [sprint roadmap](roadmap-after-063.md) for dependencies,
shared acceptance and all subsequent cards. The [retrospective](063-retrospective-and-next-plan.md)
and [landscape addendum](063-landscape-feedback.md) retain the rationale.

## Outcome and scope

An external public recipe can finish using its own stopping policy and preserve
its true reason and diagnostics through run_many. This is development conformance,
not formal E5 or statistical validation of a new stopping method. The default
patience policy remains available. No full ScoreStop implementation is required.

## Implementation: Separate completion metadata from the default stopping policy

### First failing test

Extend tests/v1/test_public_results.py with a frozen external record that has
rounds=5, completed_rounds=2 and reason="external_rule", along with an otherwise
valid result containing two outer-round trace records. The current validator
rejects it because it is not a concrete StopState. Observe this failure before
editing production code. The addendum's source probe already establishes the
concrete-type restriction; the implementation test must use the intended full
completion contract rather than accept an arbitrary object.

### Implementation design

- In src/openboost/stopping.py, expose a structural `StoppingStatus` protocol:
  read-only rounds, completed_rounds and reason. A reason is None while running
  and a nonempty string after termination. The default StopState implements
  this structure naturally and retains its patience/min_delta logic.
- In src/openboost/results.py, declare RecipeResult.stop using that protocol.
  Validate common completion metadata without requiring inheritance from
  StopState or inspecting policy-specific statistical payloads.
- Common validation checks exact integer counts (not booleans), nonnegative
  budget, 0 <= completed_rounds <= rounds, and a nonempty terminal reason.
  A "budget" reason requires completed_rounds == rounds. A None reason is an
  unfinished result and remains rejected. Custom reasons are not a closed enum.
- Keep one trace entry per completed outer round and all context/problem identity
  checks. Ordered parameter commits may exceed outer-round count; do not equate
  state.version with completed_rounds.
- Preserve the original result, stopping object and diagnostic payload. Do not
  convert external records into the built-in class or mislabel custom termination
  as patience/budget. Protocol conformance does not prove callback immutability
  or arbitrary algorithm correctness; those remain author contracts/verifier work.
- Built-in recipes continue using StopState. No universal stopping callback graph,
  recipe rewrite or new configuration system is required for this counterexample.

Acceptance and best-model selection remain separate from stopping. This change
must not alter training-loss acceptance or validation-best snapshots. Inference
artifacts are unchanged; this completion record is not a resumable checkpoint.

### Acceptance

- The external record is accepted with object identity and custom fields preserved.
- Missing fields, malformed counts, booleans, negative/over-budget counts, empty
  reasons, unfinished status and premature "budget" termination fail explicitly.
- Built-in patience, coincident patience/budget, zero rounds and rejected-round
  behavior continue to pass. Malformed results fail only their own run.
- A tiny external public loop makes an independently checkable termination decision
  before its maximum budget. Its reason must describe that rule; wrapping a shorter
  built-in fit and relabeling its budget termination is not sufficient evidence.
- Update docs/v1/stopping.md and extension documentation to distinguish the common
  result contract from the default policy. Do not claim ScoreStop support.

Suggested focused commands, with UV_CACHE_DIR=/tmp/openboost-research-uv-cache:

```sh
uv run --no-sync pytest tests/v1/test_public_results.py tests/v1/test_public_stopping.py tests/v1/test_public_ordered.py -n 0 -q
uv run --no-sync ruff check src/openboost tests/v1/test_public_results.py
```

## Exit and reflection

- Pass the focused result, stopping and ordered checks, relevant CPU regression,
  production/changed-support lint, strict documentation and package build.
- Record the independently checked external loop, negative outcomes, source revision,
  exact commands and capability boundary in this card and a learning entry.
- Inspect the staged diff and commit the verified contract change locally.
- Reflect on whether completion metadata is sufficient without policy-specific
  dependencies. Next execute Sprint 065 against the resulting installed wheel.

## Results

Not run. No implementation, new test pass or gate completion is claimed.
The earlier planning verification remains in the
[original learning](../learnings/2026-09-06-v1-sprint064-plan.md).
