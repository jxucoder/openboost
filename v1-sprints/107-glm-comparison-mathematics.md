# Sprint 107-A: Convex scalar loss-change bounds

Status: numerical contract before production comparison kernels. Inputs are the
exact stored float32 values from 106. No CUDA execution or upload is included.

## Identity and decision

For a scalar twice-differentiable convex row loss, raw step d and curvature bounds
`h_min <= loss''(z) <= h_max` along the closed old/new segment, Taylor's integral
remainder gives

```text
loss(new) - loss(old) in gradient(old)*d + (d*d/2)*[h_min, h_max].
```

This is valid for either sign of d. Every sum, product, division and segment endpoint
uses enclosing binary64 arithmetic. Offsets enter the old/new effective raw once
and cancel algebraically from d. Weight multiplies each row interval once; the
ordered weighted sum and positive weight sum are enclosed before division.
No rounded reporting likelihood, stored gradient or empirical epsilon supplies
the sign. `upper < -min_delta` proves the requested decrease; zero-containing or
unavailable bounds are unresolved. Identical stored raw is unchanged only after
both snapshots pass every row's 106 geometry domain. Changes only on zero-weight
rows are unresolved at zero, not identical raw.

## Binary and Poisson curvature

For binary encoded y, use signed margin `s=(1-2*y)*(raw+offset)` and signed step
`d=(1-2*y)*(new_raw-old_raw)`. The derivative of softplus(s) is sigmoid(s).
Compute sigmoid with the small exponential in either tail, preserving its small
positive value for correctly classified extreme observations. Curvature is
`q/(1+q)^2`, with `q=exp(-abs(s))`. It decreases as abs(s) grows. The maximum
absolute margin on the segment determines the minimum curvature; the nearest
margin to zero determines the maximum, with the exact global upper bound 1/4.
Sigmoid(0)=1/2 and curvature(0)=1/4 may use exact point identities.

For Poisson use `mu=exposure*exp(raw+offset)`, gradient `mu-y`, and curvature mu.
The minimum/maximum effective raw on the segment bounds curvature monotonically.
Exposure remains a separate positive stored input; log-factorial and the fixed
log-exposure term cancel. The independent Decimal oracle evaluates old/new
likelihoods directly, rather than this Taylor remainder.

## Arithmetic and support

Reuse 092's separately rounded IEEE binary64 interval +/*/division contract.
The production CUDA arithmetic must use directed double intrinsics. Exponential
enclosures use an 18-term Taylor polynomial after halving into [-1/16,1/16],
with remainder less than `2*abs(t)^19/19!`, followed by interval squaring. The
same derivation applies to endpoints in [-256,256]; more halvings are needed than
092's range. Standard libm accuracy does not establish an enclosure. The independent
prototype reuses the old range-64 polynomial through three extra halvings/squarings,
leaving room for outward rounding at the +/-256 endpoints.
Production may implement the full range directly. Old Normal sources are unchanged.

All-row loss/float32-gradient/positive-curvature domain checks remain prerequisites,
including zero-weight rows. Nonfinite inputs or invalid geometry raise. Finite
interval endpoints beyond [-256,256], overflowing interval operations or a weight
sum enclosure containing zero return an explicit unresolved result. Domain errors
take precedence over range failures. No CPU computation substitutes for CUDA.

The convex bound can be wider than the exact change for a finite step. Backtracking
may need a smaller step; best-model selection and patience may conservatively retain
their anchors. Tiny changes near cancellation can remain unresolved. These are
declared resolution limits, not evidence that the objectives improved. The 160/220
digit oracle tests bound containment and required signs; agreement alone is not
a proof of compiler arithmetic. Real PTX and hardware behavior remain a later gate.

## First counterexamples and downstream contracts

Freeze stationary Poisson steps whose full binary64 NLL difference is zero but
whose mathematical change is positive, tiny binary improvements, both logistic
tails, exact no-ops, equal losses at different raw, weight-zero changes, nonuniform
weights/exposure/offsets, crossing-zero margins and deterministic random rows.
Reverse directions and row permutations must retain valid bounds. Original 106
geometry-domain failures remain required. Accept/best/patience use separate anchors;
fixed steps may commit a valid worsening or unresolved candidate explicitly.
