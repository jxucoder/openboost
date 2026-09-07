# Sprint 080: Complete required CUDA recipe correctness

Status: planned. Mapping: B12 / F3 / required R1/R4/R5/R6/R8 CUDA subsets / E1.
Depends on: [079](079-cuda-distribution-and-extension.md).
Shared evidence/closure rules: [roadmap](roadmap-after-063.md).

## Outcome and first check

Fill every required device recipe cell without generalizing the first two successes.
First derive the expected CUDA matrix from the canonical R table and make the
independent verifier reject a missing recipe, skipped GPU or hidden CPU fallback.

## Work

- Reuse resident operations for binary and multiclass mappings/probabilities,
  Poisson with exposure, fixed-scale event/right-censored AFT, and independent/
  shared squared-error vector topology. Preserve squared and Normal coverage.
- For each subset verify loss, gradients/effective curvature, statistics, split
  optimum/ties, routing, leaf solve, two-round state propagation and inference.
  Exercise missingness, weights/offsets, zero-weight rows and scoped seeds.
- Retain Normal ordinary/Fisher and adaptive step evidence from 079 with validated
  parent hashes; rerun affected cases after any shared operation changes.
- Reject unsupported category/device or broader censoring requests explicitly.
  Optional CUDA quantile/ranking/Gamma/Tweedie/Formula cells stay distinct from
  required subsets; CPU support is neither removed nor relabeled as device support.

## Acceptance and reflection

Every required CUDA R1/R4/R5/R6/R8 cell passes intermediate and final task-metric
E1 parity on real hardware, plus specialized fresh CPU inference and failure tests.
Correct shapes or similar final loss cannot excuse a wrong non-tied split or
misapplied exposure. Record complete costs without claiming E4 performance yet.

Commit verified recipe ports separately and reflect after every three implementation
commits. A substantial new kernel/state abstraction needs a follow-up card rather
than expanding this coverage sprint into another foundation rewrite. Update the
public device table from actual results, then enter 081.

## Results

Not run. All required device cells remain open at this planning baseline.
