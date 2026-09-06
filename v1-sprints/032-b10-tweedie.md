# Sprint 032: B10 Tweedie nonnegative means

Parent: 373e16d. Status: complete for the bounded slice.

## Plan and acceptance

1. Add fixed-power Tweedie geometry and offset-aware mean initialization,
   retaining exact zero-target handling and explicit finite-range failures.
2. Compose CPU scalar rounds using original weights and positive_mean output.
3. Verify independent formulas, finite differences, three-round trees across
   powers, all-zero initialization, weights/offsets and loaded inference.
4. Run regressions/lint/docs/build, reflect and commit locally.

This is A9 mean mechanics, not frequency-severity composition, real quality,
dispersion fitting or compound distribution inference. AFT and CUDA remain open.
First failing check imports the missing Tweedie objective.

## Results and reflection

Delivered fixed-power Tweedie geometry, offset-aware intercept, explicit
all-zero initializer and scalar fixed/backtracking CPU recipe. Zero targets
avoid evaluating log(0); positive terms fail on underflow/overflow. Inference
reuses positive_mean with caller-owned unit conversion.

Ten focused tests pass: independent three-round trees/geometry at powers
1.1/1.5/1.9, finite differences including zero targets, annualized weights,
all-zero initialization, invalid support/power/roles, rejected updates and
fresh-process mixed-feature mean composition. CPU regression: 684 passed.
Ruff, strict MkDocs, offline sdist/wheel and all fifteen installed-wheel examples
pass on macOS/Python 3.12.12/NumPy 2.3.5. Local wheel SHA256:
f9c4c0604b6a026d503e71ba3d4d540279f60a1b12fae909ff5914bb79223de2.

Observation: the same scalar Newton path supports zeros and positive noninteger
targets without a new grower. Evidence: multi-power/multi-round independent
parity and zero-target derivatives. Decision: keep fixed power explicit in
fitting/evaluation, and keep annualized loss units separate from Poisson
count/exposure semantics. No automatic exposure structure is accepted.

This implements mean mechanics, not normalized compound-distribution inference
or estimated dispersion. The generic model still needs external output/unit
metadata. Next is A9 frequency-severity composition, including persisted
two-model dependencies and aligned policy semantics, then AFT/A10. Real A9,
full A6 and F0.3/F1–F5 remain open.

Status: complete for the bounded slice. See
[learning record](../learnings/2026-09-06-v1-b10-tweedie.md).
