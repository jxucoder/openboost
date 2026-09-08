# Sprint 106: Binary and Poisson device components

Status: active after the user's "continue" following the completed 105 retrospective.
Mapping: 080 / B12 / R1 binary / R4 Poisson / E1. Construction only; all eleven
hardware allowances are consumed. No new upload, model call or device run is included.

## Purpose and order

Exercise the existing programmable objective/fields/tree/runtime boundary with
classification and exposure-aware counts. Keep the measured validation optimization.
No CPU speed work, kernel optimization campaign or author study enters this slice.

1. Freeze the complete required device-cell inventory and independent binary/
   Poisson base, loss, gradient and curvature oracles. First tests reject a missing
   cell, skipped/fallback evidence, doubled exposure/weights and vanished binary
   tails. Bind consumed run-11 checks to its committed execution sources so new
   construction cannot overwrite historical evidence or freeze current APIs.
2. Implement resident binary and Poisson objective operations with explicit
   preparation, initialization, geometry, named fields and scalar loss. Reuse
   ownership/workspace transactions and field weighting. No training-time raw,
   target or gradient download and no CPU objective fallback.
3. Preserve classification schema through DeviceRun validation/export; test new
   objectives through two prescribed public-operation rounds, exhaustive split/
   leaf references, validation predictions/metrics and saved CPU inference. Record
   the local results and reflect after these three independently verified commits.

## Numerical contract frozen before kernels

Inputs and resident gradient/curvature use float32 storage. Scalar row math and
loss/base reductions use float64 at the stored values. Validate every row, including
weight zero. Reject nonfinite float32 storage and nonpositive representable
curvature; do not clip raw state or silently erase unsupported tiny curvature.
Binary initialization is the CPU contract's clipped weighted class prior minus
weighted offset, explicitly not an offset maximum-likelihood fit. Both observed
classes are required for initialization, including classes whose total weight is
zero. Preserve signed-margin logistic tails (use complementary probabilities).

Poisson keeps exposure as separate owned data. Counts must survive conversion to
float32 exactly; exposure must stay positive. Compute log-rate initialization by
log-sum-exp of weighted counts and weighted exposure times offset, using the
configured minimum rate only when all positively weighted counts are zero. The
loss includes log-factorial, and exposure/offset/weight each enter once. Mean and
curvature must remain positive finite float32, including zero-weight rows.

Freeze well-conditioned geometry rtol=1e-6/atol=1e-8 and default device stored
intermediate rtol=1e-4/atol=1e-5, task metric difference at most
1e-3*max(1, abs(reference)), as in the canonical E1 contract. Positive tiny binary
derivatives receive relative checks so a loose absolute tolerance cannot hide zero.
Extreme precision fixtures compare to stored-input mathematics with separately
declared limits. Do not relax tolerances after device observations.

## Acceptance and boundary

- Independent CPU oracle checks and finite differences pass before kernel work.
- Real-CUDA tests cover base/geometry, weights/exposure, exact routing and a unique
  split, leaf values, two-round state, predictions, task scores, ownership/recovery,
  same-stream operation and class-aware saved CPU inference. Collection is not a pass.
- Validation class order must match training before device work; saved models retain
  it after the run closes. Reuse the existing Model format. AFT metadata remains
  a separate required application slice, not a generic metadata placeholder.
- The required inventory retains squared, binary, multiclass, Poisson, AFT, Normal
  ordinary/Fisher with fixed/backtracking, and independent/shared squared topology.
  R9 compatible M=1/8/32 remains required in 081; optional device cells do not replace it.
- This slice implements composable objectives and verifies prescribed updates.
  Public binary/Poisson recipe adapters and their acceptance/best/stopping comparison
  semantics require the next integration slice; neither reported loss subtraction
  nor two manually accepted rounds proves stable model-selection decisions.
- Keep all consumed source freezes and raw artifacts immutable. A complete hardware
  packet and allowance must precede real-device validation; no skipped test counts.

After construction, reflect on objective-specific support and export semantics,
then integrate scalar recipes and explicit numerical decision checks before a
bounded device request. Multiclass, AFT, vector topology, batching and formal E4
remain required follow-ups. All R1–R9/C1–C7/A1–A13 remain in scope.

## Slice A verification

Thirty-eight local oracle/scope controls pass: CPU formulas and finite differences,
relative binary tail checks, twelve stored-domain cases, Decimal Poisson intercepts
at offsets shifted by 800, exactly-once exposure/weight semantics, and every required
cell's removal/skip/fallback counterexample. Five run-11 freeze checks also pass;
consumed packets now read their committed execution source bytes, preserving the
archive while allowing current production to evolve. Total: 43 passed in 7.83 s;
changed-file Ruff passes. No new production kernel or CUDA case has run.
