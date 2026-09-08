# 2026-09-07: Separate objective comparison consumers

## Context

The approved [092-C ownership plan](../v1-sprints/092-comparison-consumers.md)
requires three independently anchored consumers. Correct comparison mathematics
alone had not changed acceptance, best selection or patience.

## Decision or Result

CPU Normal now uses objective evidence for all three decisions. The run-7 stored
worsening candidates are rejected, while tiny genuine improvements lost in total
NLL rounding advance acceptance, best and patience. The current/best/patience
anchors follow the prescribed five-transition sequence independently. Reporting
scores retain their original values.

## Changes

- CPU resolve accepts explicit compare and replays an owned best validation anchor.
- StopState.observe_change consumes proved improvement against min_delta; the
  Normal recipe owns and replaces its immutable patience snapshot on that decision.
- Normal trial and stopping comparison records survive full and summary retention.
  TraceSummary explicitly permits LossChange, an immutable scalar record; arbitrary
  author records and arrays remain rejected.

## Verification

- Before implementation, all ten new consumer cases failed, including a real
  recipe that rejected equal-score improvements and stopped prematurely.
- Focused command: `OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/v1/test_comparison_consumers.py tests/v1/test_trace_retention.py tests/v1/test_public_normal.py tests/v1/test_public_stopping.py tests/v1/test_incremental_runtime.py -n 0 -q --tb=short` — 100 pass.
- Full CPU regression: `OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' -q --tb=short` — 1646 pass, one Linux-only skip.
- Ruff passes for production and the new consumer test. MkDocs builds with the
  existing execution-page link warning for evidence outside the documentation tree.
- No CUDA execution, upload, frozen-oracle change or tolerance adjustment.

## Failed Attempts

The first consumer implementation attempted to put LossChange into the existing
scalar-only summary validator. It correctly rejected the record. The summary
contract now names that single supported immutable scalar record explicitly;
generic dataclass traversal would risk retaining author arrays or state.

## Risks and Follow-ups

CPU best selection adds a full validation-model replay. This is a correctness
reference, not an optimization claim. Resident best ownership, device consumers,
383-case requirement bindings and the 092-D freeze remain open. All seven device
allowances are consumed; no hardware execution is authorized by this local slice.

## Commits

- `5b11828` — CPU Normal comparison consumers and 1646 passing CPU checks.
- `b016b24` — Named resident comparison policy, owned best anchors and twelve
  collected/unrun CUDA ownership distinctions.

## Resident transaction construction

The next slice adds named run policies, parent-bound comparison and one owned
best validation raw per accepted objective-mode state. The reported policy keeps
the old low-level/scalar behavior and storage. Every fallible comparison/copy
precedes serial advancement and term-reference publication. Copies also isolate
unchanged best anchors; sharing is deliberately deferred. Public raw(best=True,
validation=True) returns a copy. Release/close account for the new storage.

Six CPU policy-preflight cases failed before editing; all 41 comparison/Normal
API checks then pass. Twelve CUDA consumer cases collect, but no device execution
has occurred. They target copied-anchor lifetimes, failure cleanup, parent/RNG
invariants, captured false improvements, hidden true improvements and unresolved
policies. The full CPU regression command above passes 1652 checks with one
Linux-only skip; Ruff passes production and the two changed comparison test files.

The shared try_terms consumer uses the named policy and retains each comparison.
Normal recipe policy selection, patience ownership and complete historical-case
bindings are still open. This is construction evidence, not a CUDA acceptance gate.

## Resident recipe construction and reflection

Normal now explicitly selects objective comparison and independently owns its
patience validation anchor. It observes only complete outer sweeps, replaces an
anchor on proved improvement even when reported scores remain equal, and retains
the scalar comparison on the last substep. Its anchor and temporary observation
copy are released on success/failure; best storage remains owned by the run.

The missing-comparison recipe preflight failed before editing, then passed.
Fifteen real-CUDA recipe cases collect but are unrun. The prescribed means sequence
distinguishes all three anchors; a second tiny improvement must have the exact
change from the replaced anchor, so stale-anchor reuse cannot pass by producing
another negative sign. Failure tests target validation-view copies by their public
selector, avoiding brittle counts of unrelated tree-copy internals.

After three implementation commits, the foundation exposes objective comparison
at each required decision. This remains a correctness construction, without CPU
speed/author-benefit claims. Hardware, complete historical-requirement bindings,
installed D2 and fresh inference under the new semantics remain open. Historical
recipe memory expectations exclude the new best snapshot and must retain their
old meaning; revised tests explicitly include its `4*N_validation*K` bytes.

Full CPU regression passes 1653 checks with one Linux-only skip. Ruff passes
production and changed support tests. MkDocs builds with the same existing
external-evidence link warning, and `uv build --offline` builds both wheel and
source archive. These checks do not execute CUDA kernels.
