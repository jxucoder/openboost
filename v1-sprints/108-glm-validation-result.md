# Sprint 108: GLM hardware result and retrospective

Status: complete. The one approved run executes at clean `fe12beb`; all 571
cases pass. [Raw evidence and offline audit](../benchmarks/v1/evidence/cuda-glm-108/README.md)
retain all 77 declared JSON artifacts. All twelve GPU allowances are consumed.

## Result and evidence

All 153 GLM cases pass: 38 objective operations, six prescribed-round controls,
77 comparison/ownership/lowering checks and 32 recipe cases. All 418 selected
scalar, Normal and field-validation regressions also pass, with no skips or
changed assertions. The installed 35-file core, 85-file snapshot and eighteen
pinned packages match the approved freeze. No production or original test fix
is needed after hardware observation.

The numerical audit recomputes 59 standalone and 187 trajectory loss differences
at 220 digits from their stored inputs. All 246 lie inside the returned bounds.
Both families retain actual directed-double PTX. All sixteen recipe reports
include lossless fixture bytes, models, trials, stopping state and comparison
inputs; all 32 final/best models replay locally against independent trajectories.
The remote final-model replays also pass in the installed CPU-only environment.

The run uses one T4, two requested CPU cores and 8,192 MiB, with zero retries.
Pytest takes 52.93 s, worker execution 55.324 s and total dispatch 382.977 s.
These instrumented correctness durations do not answer GLM fit speed. Retained
JSON totals 1,013,949 bytes; all eighty raw hashes and the manifest binding pass.

Local closure: 56 audit/freeze/retention controls pass in 12.99 s; full CPU
regression passes 2,277 tests with one Linux-only skip in 18.37 s. Production and
changed-file Ruff and the documentation build pass. The previous historical Normal
evidence-link warning remains. No production code, frozen assertion or tolerance
change follows the device result.

## Reflection against the foundation goal

The objective boundary now supports native binary classification and exposure-
aware counts through one scalar recipe, existing growers and owned transactions.
The additions required objective-specific preparation, geometry and comparisons,
while fields, routing, leaves, tree construction and run-state ownership remain
shared. That is concrete reuse of the foundation under two application families.
It does not by itself establish lower authoring cost or adoption.

Separating reported likelihood from numerical improvement decisions survives real
device execution. Tiny logistic tails and Poisson stationary changes retain their
frozen signs; unresolved changes remain explicit. The large-step Poisson recipes
retain earlier best prefixes while accepted state advances, and patience stops
independently. This is stronger evidence than matching only final array shapes
or rounded loss values.

Retaining exact input bytes resolves the earlier cross-host reconstruction concern
for these new recipe artifacts. The shared frozen launcher also completed this
expanded correctness matrix comfortably within its deadline. Most total elapsed
time lies outside worker execution, but this observation does not motivate a new
infrastructure or performance sprint by itself.

## Next bounded construction

Stop at this retrospective as the approved packet requires. On the next execution
continuation, return to [080 required CUDA recipes](080-cuda-required-recipes.md):

1. Start multiclass with an independent numeric/shape contract covering weighted
   offsets, stable softmax probabilities, zero-sum score geometry and class order.
   Determine the required shared versus separate topology from the canonical R1
   contract before designing mapped device operations.
2. Reuse mapped fields/tree/runtime boundaries and preserve all-row validation,
   device ownership and persisted CPU inference. A new objective comparison must
   have independent loss-change mathematics before acceptance/best/patience use;
   rounded reporting-loss subtraction is not a sufficient decision rule.
3. Verify local oracle/configuration/construction slices in small commits, then
   freeze their exact real-device packet. A further upload/hardware run needs its
   own concrete allowance; no automatic run 13 follows this success.

Fixed-scale event/right-censored AFT and independent/shared squared vector topology
remain required next cells, followed by [081 compatible train-many](081-cuda-train-many.md)
and [082 real-workload quality/cost](082-end-to-end-cost.md). Optional device
quantile/ranking/Gamma/Tweedie/Formula support does not replace any required cell.
All R1–R9/C1–C7/A1–A13 requirements remain; author studies stay deferred under 101.
