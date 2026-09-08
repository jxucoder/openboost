# 2026-09-07: Avoid immutable artifacts inside candidate scoring

## Context

The installed profile still spends 29.092 cumulative seconds constructing vector
leaves during candidate scoring. Those temporary values are consumed by a dot
product; they are not model artifacts.

## Decision or Result

Keep public/persisted leaves immutable, but use local float64 scratch values during
vector scoring. Validate the shared regularizer once per candidate, while every
node/channel still checks curvature, denominator, gradient and resulting value.
Extract the original scalar checks/division into a shared private helper.

## Changes

- No immutable bytes-backed leaf copy for each scored left/right/parent node.
- Identical scalar division, dot operation and gain evaluation order.
- Public callbacks, feasibility and actual leaf construction remain unchanged.

## Verification

The initial ownership-path test fails before implementation. Forty-two focused
checks pass, including scores, near ties, invalid inputs and exact model bytes.
Full CPU suite: 1077 passed, one Linux-only skip; lint passes.

## Failed Attempts

None beyond the deliberately failing ownership-path regression.

## Risks and Follow-ups

No full-fit speedup claim. Verify installed execution with the same approved
profile; keep paired full-fit correctness/cost evidence as a separate requirement.
Parent-score caching and further parameter preparation are not part of this slice.

## Commits

- Scratch vector scoring; parent `a3b78a3`.

### Installed scratch-path evidence and reflection

At clean `21f76b1`, the [same approved diagnostic](../benchmarks/v1/evidence/a6-scratch-profile-070/README.md)
completes. All 32 source hashes and six artifacts verify. Raw pstats records just
77 vector_leaf calls (previous layout profile: 904838); temporary candidate leaves
now take the scratch path. This prefix scores 352465 candidates, compared with
301590 previously, but separate instrumented prefixes are not full-fit evidence.

Reflection: the two narrow changes remove measured redundant work while retaining
exact tested semantics. Stop stacking micro-optimizations now. Next prepare one
paired real fit against the original scoring baseline in a common environment,
requiring identical selected model bytes/predictions/stopping plus full cost and
replay accounting. Keep the remaining search matrix pending until that evidence
is reviewed. No end-to-end speedup is claimed.

Validation: 1077 CPU tests passed, one Linux-only skip; lint/docs pass. The installed
profile completes through the intended soft deadline, not a successful model fit.
