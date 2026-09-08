# Sprint 080: Complete required CUDA recipe correctness

Status: in progress; bounded squared/Normal and binary/Poisson evidence retained,
with remaining required cells open. Mapping: B12 / F3 / required
R1/R4/R5/R6/R8 CUDA subsets / E1.
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

At the original planning baseline this sprint had no results. Subsequent scalar,
Normal and comparison runs retain their individual boundaries and failures.
[108](108-glm-validation-result.md) now passes all 153 binary/Poisson objective,
comparison and recipe cases on T4, alongside 418 regressions. This is bounded
device evidence, not full R1/R4 or complete 080 acceptance. Multiclass, fixed-scale
event/right-censored AFT and independent/shared squared vector topology remain
required; start multiclass construction next. All formal quality/cost and R9 gates
remain separate. No further hardware allowance follows automatically.
