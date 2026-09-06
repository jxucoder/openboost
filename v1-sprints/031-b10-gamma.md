# Sprint 031: B10 Gamma positive targets

Parent: dc03382. Status: complete for the bounded slice.

## Plan and acceptance

1. Add strictly positive Gamma mean geometry, offset-aware initialization and
   explicit positive-mean inference.
2. Compose scalar CPU Gamma rounds with original weights and existing state.
3. Verify independent derivatives, three rounds, claim-level versus weighted
   averages, invalid support, offsets and fresh-process mixed inference.
4. Run regression/lint/docs/build, reflect and commit locally.

A8 CPU mechanics only; no fitted dispersion or calibrated distribution claim.
A9 Tweedie/composition, A10 AFT, real data and CUDA remain required.
First failing check imports the absent Gamma objective.

## Results and reflection

Delivered Gamma positive-target geometry, offset-aware intercept, CPU
fixed/backtracking recipe and positive_mean inference. Nine focused tests pass:
independent derivatives/base, three-round tree parity, claim-average/count-weight
equivalence, support/structure rejection, backtracking rejection, output support
and fresh-process mixed-feature persistence.

CPU regression: 674 passed. Ruff, strict MkDocs, offline sdist/wheel and all
fourteen installed-wheel documentation examples pass on macOS/Python 3.12.12/
NumPy 2.3.5. Local wheel SHA256:
5301e2f6c554163ddada1e32e3f1ca01d7540b5c339eac7d74cabb8752cd4dda.

Observation: Gamma adds target/weight semantics without new tree/state code.
Evidence: independent multi-round results and claim-average geometry equivalence.
Decision: expose the mean-only objective and require users to specify their
observation unit and eligibility; do not imply estimated dispersion/calibration.

Across the last three slices, routed residual leaves required one new explicit
context boundary, while Poisson and Gamma reused the existing scalar path.
Readable recipe loops currently repeat configuration/transaction orchestration;
keep objective distinctions visible and consider a small shared scalar loop only
when its tests preserve these per-family semantics. No refactor or performance
claim is justified merely by line count.

Next: Tweedie/A9 and frequency-severity composition, then AFT/A10. Real A8,
full A6 workflows, application output metadata and F0.3/F1–F5 remain open.
Status: complete for the bounded slice. See
[learning record](../learnings/2026-09-06-v1-b10-gamma.md).
