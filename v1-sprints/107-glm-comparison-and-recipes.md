# Sprint 107: GLM comparison and scalar recipe integration

Status: active after the user's "continue" following 106. Mapping: 080 / B12 /
R1 binary / R4 Poisson. Local construction can continue; no hardware/upload allowance.

## Purpose

Turn the new objective components into usable scalar recipes without repeating
Normal's rounded-loss acceptance failure. Preserve shared fields/tree/runtime
operations, explicit exposure and class-aware CPU inference. Required multiclass,
AFT, vector topology, R9 compatible M=1/8/32 and formal E4 remain subsequent work.

## Order and acceptance

1. Freeze independent high-precision loss-change cases and a declared resolution
   contract for binary and Poisson. Cover exact no-ops, resolvable improvement and
   worsening, unresolved changes, extreme logistic tails, zero counts, offsets,
   exposure, zero/nonuniform weights and rejected geometry domains. Include changes
   that rounded reported losses cannot classify. Verify sign/bounds against an
   independently evaluated objective; never use reporting-loss subtraction as its
   oracle. Record derivation and support limits before implementing CUDA arithmetic.
2. Implement objective-owned resident comparison through the existing `LossChange`
   callback. Use cancellation-aware expressions with justified error bounds; return
   unresolved where support is insufficient. Retain all-row validation, exactly-once
   weights/exposure, owned-stream scratch and scalar-only summaries. Add allocation,
   dispatch and callback-failure recovery tests without changing old Normal evidence.
3. Add binary/Poisson recipe entry points through the shared scalar grower and
   transaction runtime. Acceptance, best-validation selection and independent
   patience must use the declared comparison. Verify fixed and bounded backtracking
   updates, full/partial rejection, zero rounds, retained best state, retries and
   release behavior. Export retains class labels and explicit Poisson inference
   inputs. Use independent trajectories and saved CPU inference, not shape checks.
4. Reflect, then construct one reviewable hardware request: immutable source and
   environment inventory, exact collected test IDs, numerical tolerances, bounded
   timeout/cost/retry policy and durable raw outputs. Include relevant existing
   storage/objective/runtime/recipe regressions and all new GLM cases. No dispatch
   or new file upload until that concrete packet has an allowance. Collection and
   CPU passes cannot substitute for real-device results.

Commit after each verified slice; reflect after three implementation commits or
any numerical/architectural counterexample. No performance optimization, CPU speed
campaign, external benchmark or author/model study enters this sprint.

## Exit evidence

- Independent numerical controls and relevant CPU regression pass unchanged limits.
- Public recipe behavior, ownership, metadata and persistence match documented scope.
- New real-device cases are enumerated; until they run successfully, the new recipes
  remain experimental and no R1/R4 device conformance or speed claim is made.
- A later passing bounded device packet still does not close every required 080
  recipe, R9, application quality, author/adoption benefit or formal E4.

## Slice A result

The [convex-bound derivation](107-glm-comparison-mathematics.md), 59 frozen cases
and independent direct 160/220-digit likelihood oracle precede production code.
All 73 local controls pass in 1.28 seconds, including reversed directions,
permutations, extreme exponent bounds and invalid zero-weight no-ops. The
stationary Poisson reporting tie is correctly bounded as worsening. Ruff passes.
No new device result or tolerance change is claimed.

## Slice B construction

The production scalar factory and resident callbacks implement the declared
convex intervals, positive-domain guards, ordered weighting and owned scratch.
Binary/Poisson factories now supply `compare`; old Normal arithmetic and consumers
remain unchanged. The Python scalar expression passes all 59 independent cases
plus two range checks; together with the independent study and host configuration,
160 tests pass in 1.95 seconds. No CPU host computation is used on the device path.

Seventy-seven separate CUDA cases collect: all 59 frozen inputs, invalid geometry
and identity, allocation/dispatch cleanup, forbidden host/reporting callbacks and
actual directed-double PTX inspection. Each numerical case will retain inputs,
enclosure, direct Decimal result and transfer counters when run. Hardware execution
is pending. Next construction connects distinct acceptance/best/patience consumers.
