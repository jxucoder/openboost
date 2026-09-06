# Sprint 018: B03 CPU ownership, state and minimal artifacts

Starting revision: `de3015d`. Status: initial B03 slice complete. User approved B03–B06 overlap
with unfinished F0.3; all v1 requirements and thresholds remain unchanged.

## Plan and acceptance

1. Public numeric prepared data and Problem: own immutable arrays, content/order/
   feature identity, aligned target/weight/offset roles; CPU only, reject unsupported
   devices/fields. Numeric bin fitting remains B04.
2. Explicit RunContext and accepted/proposed state: deterministic keyed randomness,
   separate train/validation raw caches, offsets applied at output only, atomic
   accept/reject, stale/cross-run rejection and immutable best snapshots.
3. Minimal constant-term inference artifact to exercise B03 without a tree grower:
   explicit vector base/coefficients, shape/version/corruption checks and fresh-process
   prediction roundtrip. This is not a trained boosting recipe.
4. Verify independent hand/reference cases, full CPU regression, lint, docs and
   package build; update public capability claims and commit verified slices.

First counterexamples: mutating caller arrays cannot alter owned problem/model
state; two rejected trials cannot change raw predictions, best state or keyed RNG;
repeated updates cannot accumulate an input offset a second time.

F0.3 remains open. B04/B05 implement trees and the first complete recipes. B06
must test Formula and heterogeneous run state before interface stabilization.

## Result and reflection

Initial B03 slice delivered: public NumericData/Problem, RunContext, immutable
accepted/proposed state and constant-term inference. All 509 tests pass, including
21 public cases. Ruff, strict docs, runnable documentation, sdist/wheel build and
isolated wheel inference checks pass. See the
[learning record](../learnings/2026-09-06-v1-b03-cpu-state.md).

A content-bound parent is necessary: version alone cannot distinguish divergent
accepted histories. Offsets must stay outside cached raw to avoid compounding
exposure across commits. These are shared state boundaries, not task-name branches.

This is an initial CPU architecture milestone, not all F1.1 semantics or CPU v1
acceptance. No tree training or GPU implementation exists. Next: B04 numeric
binning/statistics/candidates/routing/leaves, then B05 squared/Normal compositions.
B06 must exercise Formula and heterogeneous runs before interface stabilization.
F0.3 stays open under the approved overlap, with every requirement retained.
