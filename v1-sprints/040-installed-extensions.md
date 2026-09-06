# Sprint 040: Installed public D2/D3 development extensions

Parent: 2f6c77a. Status: complete for the D2/D3 development slice. Mapping: Sprint 038 M2, C2/C3/C6,
exploratory F2.1 and partial E2/E6. This is repository-authored development work,
not independent authorship, E5 comparative cost, held-out evidence or adoption.

## Plan and acceptance

1. Build two separate extension wheels using installed public interfaces only:
   independent cohort feasibility and a separately implemented penalized leaf solver.
2. Verify D2 against exhaustive feasible split enumeration and no-feasible cases;
   verify D3 against breakpoint/stationary-point minimization and next-round effects.
3. Run multi-round recipes from an isolated installed environment, save mixed/
   missing-feature models, remove both extensions and verify fresh-process predictions.
4. Record artifacts, source/wheel hashes, failures and usability observations;
   run regression/lint/docs and commit locally. No core modification is presumed.

First failing check: import the nonexistent extension modules. Acceptance requires
both callbacks actually execute, zero private OpenBoost imports/core changes,
independent math checks and inference without training plugins. D1/D4/D5 and the
remaining E2/E6 requirements remain separate.

## Results and reflection

Two separately built packages implement custom cohort feasibility and an external
penalized leaf solver. They run through public learner/grower callbacks with zero
core changes or private imports. Core wheel hash remains identical to Sprint 039.

D2's exhaustive oracle selects cut 1 while the unconstrained optimum is different;
depthwise, best-first and symmetric growth all choose the constrained cut. A
cohort-separated feature has no feasible split. Zero training weights do not
erase independent information and a foreign problem fails binding.

D3's bisection solver matches independent breakpoint/stationary enumeration on
30 deterministic weighted fixtures (maximum absolute difference
8.881784197001252e-16), including repeated residuals and zero weights. Subgradient
containment and movement toward the anchor under stronger penalty pass. Three
rounds with routed mixed/missing-feature leaves match the original-row oracle;
the custom leaves change subsequent raw values. Both saved models preserve exact
predictions in a fresh interpreter after both extension packages are uninstalled.

Verification: 767 CPU regression tests pass; the strengthened routed-leaf check
also passes its two focused tests and a fresh full installation verifier. Ruff
and strict MkDocs pass. Installed environment: macOS/Python 3.12.12/NumPy 2.3.5,
OpenBoost 1.0.0.dev0 and extension versions 0.1.0. Raw source/wheel/environment/
command/output evidence is in [the artifact directory](../benchmarks/v1/evidence/installed-extensions-040/README.md).

Observation: these two deep changes reuse existing semantic boundaries without
duplicating recipe loops. Friction: the leaf callback is supplied through a grower
adapter and the quantile must be aligned in recipe and solver configuration.
Decision: retain this observable usability cost; do not infer a need for a new
generic trainer from one development attempt. Neither package is an independent
author, and these results do not measure comparative task effort or adoption.

Next: D4 ordered Normal/Formula update composition, remaining D1/D5 installed
evidence and current OpenBoost real-data integration. M2 and formal E2/E6 remain
incomplete. See [learning entry](../learnings/2026-09-06-v1-installed-extensions.md).
