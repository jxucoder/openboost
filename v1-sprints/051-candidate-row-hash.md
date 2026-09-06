# Sprint 051: Hoist invariant candidate row hashing

Parent: d1dede6. Status: invariant-hash optimization complete; one full fold passes.

## Plan and acceptance

1. Demonstrate repeated identical row hashing within one candidate enumeration.
2. Compute that digest once per invocation without changing any candidate field,
   routing, statistics or immutable identity semantics.
3. Run focused and full correctness tests; rerun the unchanged full Covertype
   fold-zero worker at its original 90-second cap, retain outcomes and replay.
4. Record evidence and next step, commit locally. No histogram or cache rewrite.

One fold is a bounded diagnostic rerun, not full A3 acceptance or a speed claim.

## Results and reflection

The focused mixed-feature/full/subset tests reproduced sixteen identical row
hash calls per enumeration. The implementation now computes the same digest once
per call. Candidate order, immutable identities and all statistics are unchanged;
independent histogram/tree and recipe tests pass. Full CPU regression: 848 passed.
Ruff and strict docs pass.

The unchanged full Covertype fold-zero job completes in 87.4 seconds under its
original 90-second cap, with four rounds and a saved seven-class model. Fresh
inference reproduces probabilities and source row IDs exactly, and probability
rows normalize to one. Source/input/output hashes and raw artifacts are retained
in [row-hash-051](../benchmarks/v1/evidence/row-hash-051/README.md).

The previous uninstrumented fold-zero run timed out at 90 seconds. This single
successful rerun demonstrates completion under the cap, not a stable speed ratio;
there is little margin and the other four folds were not rerun. Full A3 validation
remains incomplete. Next address the separately profiled histogram cost, retaining
exact weighted/missing/categorical/vector statistics, before repeating full-fold
validation. No general cache redesign or GPU claim is justified by this slice.
See [learning](../learnings/2026-09-06-v1-candidate-row-hash.md).
