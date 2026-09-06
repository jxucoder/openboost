# Sprint 030: B10 Poisson counts and exposure

Parent: dd42967. Status: complete for the bounded slice.

## Plan and acceptance

1. Add count support validation, explicit positive exposure, Poisson likelihood/
   derivatives and offset-aware rate initialization.
2. Compose a scalar CPU recipe and explicit rate/count inference transform.
3. Verify independent geometry and multiple rounds, exactly-once weights/offsets,
   exposure scaling, all-zero counts, invalid support and loaded inference.
4. Run regressions/lint/docs/build, reflect and commit locally.

A7 mechanics only: Gamma/A8, Tweedie/composition/A9, AFT/A10, real data and CUDA
remain separate required work. First failing check imports missing Poisson.

## Results and reflection

Delivered Poisson count likelihood/geometry, exposure validation, offset-aware
constant-rate initialization, fixed/backtracking CPU rounds and explicit
rate/count inference transforms. The count/exposure problem uses the same scalar
Newton fields, growers and state; no new training engine or tree format.

Ten focused tests pass: independent geometry/intercept and finite differences,
three-round tree composition, zero-count initialization, exposure doubling,
invalid count/exposure/unknown-role rejection, overflow/underflow rejection,
atomic rejection and fresh-process mixed-feature output composition.
CPU regression: 665 passed. Ruff, strict MkDocs, offline sdist/wheel and all
thirteen installed-wheel documentation examples pass (macOS, Python 3.12.12,
NumPy 2.3.5). Local wheel SHA256:
3ea7a840b94395907474a1ef4bc2041c91f9ef78311010dc6469278a9fb24419.

Observation: exposure belongs in the count likelihood, not in automatic sample
weighting. Evidence: offset-aware base has zero weighted intercept gradient,
original weights match independent multi-round geometry, and fixed raw rates
give doubled counts under doubled exposure. Decision: keep exposure as a
required explicit role and rate/count output as a named external transform.

The artifact remains a generic raw model: consumers retain the Poisson transform
and supply inference exposure/offsets. This does not yet establish a packaged
application artifact with output-family metadata. All-zero positive-weight counts
use an explicit raw-rate initializer, not a hidden prediction/curvature floor.
Real A7 results and E gates remain open. Gamma/A8 is the next B10 slice, followed
by Tweedie/composition/A9 and AFT/A10; none are substituted by Poisson success.

Status: complete for this bounded slice. See
[learning record](../learnings/2026-09-06-v1-b10-poisson.md).
