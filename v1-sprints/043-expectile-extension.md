# Sprint 043: External expectile objective

Parent: b7fdb60. Status: complete for D1 development. Mapping: Sprint 038 M2, D1.

## Plan and acceptance

1. Add a separate installable expectile package using only public CPU operations.
   Preserve weighted initialization, offsets, analytic derivatives and tau=.8.
2. Compare initialization to stationary-interval enumeration and two rounds to
   independent exhaustive tree references; cover zero weights and residual signs.
3. Run isolated installed checks and fresh-process raw prediction after removing
   the training plugin. Record source hashes and reproducible artifacts.
4. Run regression/lint/docs, record friction and limitations, commit locally.

The first focused test is the missing external expectile implementation against
the existing D1 oracle. No core objective or task-specific runner branches are
planned. This is internal development evidence, not a timed E5 attempt or an
advantage over incumbent custom-objective hooks. Real-data integration remains next.

## Results and reflection

Completed as an external package with no core edits/private imports. Expectile
geometry validates scalar problems and honors weights once through newton. Base
initialization bisects the weighted derivative over active target-minus-offset
values; an independent stationary-interval oracle checks 32 asymmetric weighted
fixtures plus explicit offset/zero-mass cases. Derivative checks include positive,
negative and zero residuals and finite differences away from the kink.

Independent exhaustive trees match both boosting rounds with nonzero offsets,
missing values and zero weights. The installed result also passes run_many's
structural contract. Five wheels install in an isolated environment; all earlier
D2/D3/D4 and scheduling checks rerun. Nine raw models preserve exact predictions
in a fresh interpreter after removal of all four training plugins.

Verification: 799 CPU tests pass; Ruff and strict MkDocs pass. Installed offline
wheel verification passes on macOS x86_64, Python 3.12.12, NumPy 2.3.5 with one
BLAS/OpenMP thread. See [raw evidence](../benchmarks/v1/evidence/expectile-043/README.md)
and [learning](../learnings/2026-09-06-v1-expectile-extension.md).

The author still writes a small loop to wire geometry, statistics, transactions
and stopping. This is documented friction, not a measured author-time failure.
D1 remains a control for incumbent-friendly custom objectives; this internal
implementation does not imply a comparative win. No new builtin objective or
trainer abstraction was needed. Remaining D5 probes and current real worker
integration are next; full E2/E5/E6, GPU, real quality and adoption remain open.
