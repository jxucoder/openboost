# Sprint 083: Engineering v1 acceptance and delivery

Status: planned. Mapping: B14 / F5 / C7 / complete required E0–E6.
Depends on: all required engineering evidence, including 071–077 and 080–082.
Shared evidence/closure rules: [roadmap](roadmap-after-063.md).

## Outcome and first check

Produce a locally reviewable release candidate whose documentation and independently
reconstructed gates match implemented behavior. First remove or corrupt a required
artifact, omit an application, and try source-checkout imports from an installation
check. Each must fail rather than leave a green acceptance report.

## Work

- Reconstruct E0–E6 from committed raw records: every R/C/A cell, all thirteen
  applications and at least six independent sources, correct revisions and parent
  hashes. Preserve not_run/fail/unsupported/error/timeout; no manual override to pass.
- Build and clean-install CPU and one real CUDA wheel environment for the declared
  Python/OS matrix. CPU installation requires no CUDA. Exercise every standard
  recipe and at least two independent extension packages using public interfaces.
- Run install → baseline → algorithm change → verify → save/load → CPU inference.
  Remove training plugins for core tree/raw inference. Declare dependencies for
  custom formulas/links; do not serialize arbitrary executable closures silently.
- Align docs, signatures, capability tables, errors, versions/licenses and release
  notes with results. Stabilize only contracts justified by use; preserve historical
  correctness evidence. A necessary semantic fix triggers affected re-evaluation.
- Assemble local wheels, reproducibility instructions and an acceptance report.
  Push, PR, merge, release publication and leaderboard updates need a new explicit
  request; this delivery card does not perform an external action by itself.

## Acceptance and reflection

Engineering v1 passes only when every required E0–E6 cell and each A1–A13 passes.
All installed checks run outside the source tree, runtime failures return nonzero,
and gate reconstruction agrees with independent artifacts. If any cell remains
open, publish the local report as a partial milestone, not completed v1.

Reflect on the measured benefit, remaining limits and repeatedly used public
boundaries. Report the separate E7 status even if engineering passes. Green CI,
test totals and objective counts cannot replace this decision.

## Results

Not run. This is a release-candidate plan, not a release or claim of v1 completion.
