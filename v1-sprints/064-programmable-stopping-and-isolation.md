# Sprint 064: Programmable stopping and installed run isolation

Baseline: `e15704d`. Status: planned; implementation has not started.
This is the execution card for Sprint 063 N1. The [retrospective](063-retrospective-and-next-plan.md)
and [landscape addendum](063-landscape-feedback.md) explain the evidence and remaining
N2–N5 work. All R1–R9/C1–C7/A1–A13 and E0–E7 requirements remain unchanged.

## Outcome and scope

An installed external recipe can finish using its own stopping policy and report
its true reason through run_many. Independent runs preserve their own RNG streams,
preparation identity, stopping, results and failures under reorder/regroup/retry.
This sprint establishes development conformance, not formal E5 or statistical
validity of a new stopping method.

Deliver two cohesive implementation commits, each with focused checks and a
learning update. Record actual results here as execution proceeds. No full
ScoreStop implementation, objective expansion, GPU work, source-data retrieval,
formal agent cohort or external publication belongs to this sprint.

## Slice 1: Separate completion metadata from the default stopping policy

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

## Slice 2: Installed stopping, RNG and preparation conformance

Extend examples/v1_extensions/scheduler_checks.py and its existing verify.py
workflow. Reuse current public wheels, mixed K=1/2 fixtures and M=1/8/32 schedules.
This slice adds checks; edit the core only if a new failing case demonstrates a
separate defect, with its own narrow verification and commit.

### Required cases

| Case | Expected result |
|---|---|
| Same seed, distinct stable run IDs | Sampled keyed streams differ; do not infer different predictions from a deterministic recipe |
| Same run ID under retry/reorder/regroup | RNG draws, accepted state, stop state and predictions match independent execution exactly |
| Changed feature values, unchanged source row IDs | Old PreparedData is rejected even when shapes/names match |
| Changed source row IDs, unchanged feature values | Old PreparedData is rejected |
| Fresh preparation for either changed dataset | Shared-prepared execution matches fresh independent direct execution |
| Changed target/weights with identical features and row identity | Valid preparation reuse continues to work; do not reject safe sharing |
| External stopping policy mixed with built-in and ordered recipes | True reason/count/payload survives; other runs retain their independent outcomes |
| Failed preparation or malformed stopping status | Explicit retained failure; neighboring valid runs and same-ID retry are unaffected |

Compute independent direct results before forbidding Binning.fit during shared
execution. Restore any instrumentation after the check. Use newly constructed
owned data for mutation cases, not illegal in-place mutation of read-only input.
Retain existing different-patience, invalid-result and plugin-free inference checks.

### Installed acceptance and evidence

Run copied checks from outside the repository with Python -I and imports from
site-packages. Build/install the wheel and extension packages in the existing
isolated uv workflow, then remove training plugins and verify saved inference.

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python examples/v1_extensions/verify.py /tmp/openboost-v1-sprint064
```

Use a fresh output directory. If cached dependencies are unavailable, report the
installation failure separately from semantic failures. Evidence belongs under
benchmarks/v1/evidence/scheduling-064/ after verification; record source/wheel hashes,
revision/dirty state, environment, exact commands and all expected error outcomes.
Expected fault-injection failures must be labeled as such, never hidden or treated
as successful training. Do not count this internal verifier as an independent author.

## Sprint exit and reflection

- Both slices pass focused checks and installed verification.
- Run the relevant CPU regression once after the final implementation, production/
  changed-support lint, strict docs and package build. Repeat only after a new change
  or unresolved failure justifies it. Inspect staged diffs before each commit.
- Verify raw artifact/source hashes and document capability boundaries.
- Record what failed, whether the public boundary changed, and whether any workaround
  or private import remained. N1 completion does not declare full E2/E5/E6.
- Update current navigation with the actual result and next slice. Commit locally;
  future push/PR/merge needs an explicit request for that action.

## Following milestones and decision points

| Order | Deliverable | Acceptance / decision |
|---|---|---|
| N2a | Bounded practical CPU profile using Sprint 063's eight Housing cases and separate rejection fixtures | Freeze subset hashes; 120 s/case, two threads, enforced 8-GiB cap; retain resource failures; distinguish profiled and uninstrumented measurements |
| N2b | Incremental candidate/accepted raw state and explicit summary/full diagnostics, justified by N2a | Linear tree replay in new terms on the counting fixture; summary array storage bounded by run state; exact CPU semantics and full/summary equivalence |
| N3 preparation | One control and one deep-change exploratory author task with appropriate incumbent paths | Verifier and accounting/isolation work before attempts; preserve original D/H scope; additional independent execution requires authorization |
| N4 first closure | Current coverage ledger and one complete real A6/A13 selected-model workflow | Frozen searches, validation-only selection, sealed test release, per-target/standardized quality and complete cost/failure records |
| N5 | Formal E5, all application E3 and installed E6; authorized external E7 trials | Existing thresholds and every required case; separate cohorts after semantic changes |

N2 should split diagnosis, state changes and diagnostic-retention changes into
independently verified commits as needed. Do not combine an execution redesign,
new algorithm and large benchmark in one change. Coupled matrix leaves and learned
learner mappings remain N3 development options; implementing all cited papers is
not a new prerequisite.

### Proposed GPU decision after N1 and relevant N2 checks

Recommended: permit bounded B12 feasibility alongside unfinished author/quality
work. First resident squared-error training, immediately followed by K=2 Normal
and actual backtracking; then a nondefault component, remaining required CUDA
subsets and compatible M=1/8/32 execution. Full costs, CPU/CUDA intermediate parity,
missingness, failure/state isolation and CPU-readable inference are required.

This sequencing amendment remains proposed. This plan does not approve or start
it. Until explicitly adopted in the active plan, follow existing formal F2→F3
ordering while continuing independent CPU/evaluation work. No E0–E7 threshold or
A1–A13 obligation changes under either route.

## Planning verification

Read current stopping/result/scheduler contracts, preparation tests, installed-wheel
verifier, public stopping docs and Sprint 063's evidence. No implementation, model
fit, new test pass or phase completion is claimed by this planning slice.
Documentation/link checks are recorded in the [planning learning](../learnings/2026-09-06-v1-sprint064-plan.md).
