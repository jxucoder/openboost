# Sprint 036: A6 multi-output squared CPU recipes

Parent: 83f6a9a. Status: complete for the bounded slice.

## Plan and acceptance

1. Add explicit numeric multi-output squared geometry and per-target error metrics.
2. Compose independent scalar trees and shared vector trees, including optional
   projected split statistics with full-dimensional leaves, through one transaction.
3. Verify multiple rounds, weights/offsets, all growth policies, rejection, wrong
   target kinds and fresh-process persistence against independent tree references.
4. Run regression/lint/docs/build, reflect and commit locally.

This closes a bounded A6 construction gap, not real subject-split quality evaluation. Shared preparation/stopping/M32 is next.
First failing check imports the missing MultiSquared objective.

## Results and reflection

Delivered MultiSquared and multi_squared with independent/shared trees and
optional shared split projection. All outputs commit together; full-dimensional
leaves survive projected splits. K=1 matches ordinary squared boosting.
Added TargetScale and MultiOutputModel for training-only weighted scaling,
constant-target flags and original-unit prediction/offset persistence.

Fourteen focused tests pass: three rounds across three policies and three modes,
per-output metrics, target permutation, K=1 equivalence, scaling/constant targets,
invalid target/projection rejection, atomic rejection and fresh-process mixed
inference for independent/shared models. CPU regression: 730 passed. Ruff,
strict MkDocs, offline sdist/wheel and all eighteen installed-wheel documentation
examples pass (macOS/Python 3.12.12/NumPy 2.3.5).
Local wheel SHA256:
8934f48ad1a71421ce9f0573a0973a16f6990e13e514fb2d0be2e2b5517d36ed.

Counterexample: testing variance == 0 misidentified a constant target because
the weighted mean rounded slightly away from its value. Fix: detect identical
values over positive-weight training rows directly, retain that exact mean,
and assign scale one. This is covered by the focused regression.

Observation: independent and shared regression now use the same objective,
state and three existing growers. Evidence: every intermediate agrees with
independent tree/reference calculations, including projection and output
permutation. Decision: keep raw recipe/scaling separate and persist the inverse
explicitly. Weighted scaling gives a zero weighted base without offsets, up
to roundoff; offsets require a residual-mean correction.

This closes Sprint 035 item 1's CPU mechanics, including the scaling boundary.
It does not close real A6 subject splits, per-target quality or E gates. Next:
shared prepared inputs and independent validation-driven stopping/M32. Other
audit items and F0.3/F1–F5 remain open.

Status: complete for the bounded slice. See
[learning record](../learnings/2026-09-06-v1-a6-multioutput.md).
