# Sprint 090: Normal device construction and independent fixtures

Status: 090-A complete; 090-B next. No Normal device implementation or new device run.
Mapping: [079](079-cuda-distribution-and-extension.md) / B12 / F3.2 / R6 /
C2–C5 / E1–E2 development conformance. Parent: `48a1386`.

## Purpose and execution boundary

Use Normal K=2 to test whether the scalar CUDA components form a programmable
foundation. Both raw parameters, joint proposals and author-controlled ordered
updates must compose through public operations. A second private trainer would
not answer that question. D2's independent cohort information must reach the
same tree operations through an installed external package.

Run 5 satisfies 079's bounded scalar correctness/residency prerequisite. Its
212 passing cases remain tied to `af026ef`; they do not verify future changes.
The latest user continuation starts this local tranche after the recorded
retrospective. All five hardware allowances are consumed. Local construction is
authorized; another private upload or hardware invocation needs a concrete frozen
package and a new allowance. Do not launch independent authors or inspect H1/H2.
069's accounting/isolation work and all formal R/C/A/E obligations remain open.

## Sequence and first failing checks

| Slice | Work and smallest distinguishing check | Exit |
|---|---|---|
| 090-A | Freeze original-row Normal base, geometry, directions and joint/ordered transactions before kernels. Offset/weight and actual rejected-then-accepted trials must differ from scalar fixtures. | Independent imports blocked, analytic checks and public CPU agreement; record exact fixture behavior. |
| 090-B | Resident K=2 Normal operations: initialization, geometry, direction and least-squares fields. | Collectable hardware cases for offsets, weights, ordinary/Fisher/damped directions, invalid numerical domains and ownership. No device-pass claim from CPU tests. |
| 090-C | Generalize raw buffers and atomic mapped-term proposals in the existing runtime. | Joint rejection preserves both columns and all ownership; ordered commits recompute from the latest accepted state. Scalar regressions remain unchanged. |
| 090-D | Compose joint and both ordered Normal loops, bounded search and stopping. | Geometry, trees, trial decisions, validation/best, raw states and final scores match the frozen reference. Bound retention and count rejected work. |
| 090-E | Install an external D2 device learner; export Normal models for fresh CPU inference without CUDA/training plugins. | The external package changes an observed legal split using independent fields, with no private imports/core edits. Numeric/missing inference and mappings survive persistence. |
| 090-F | Freeze one bounded correctness package and request its upload/run allowance. | Every selected test/file/hash, environment, CLI, resource ceiling and expected count are reviewable before approval. Archive every result and stop for reflection. |

Commit after each independently verified slice. Reflect after each slice, every
three implementation commits and every correctness/architecture counterexample.
090-A is a reference/design deliverable, not implementation of Normal CUDA.

## Public component design

Keep `DeviceOperations` responsible for owned buffers, named fields, routed
histograms, scores, feasibility, routing and scalar leaf/tree operations. Normal
does not require changing scalar leaf semantics: two scalar learners can map into
a two-column raw state. Full vector leaves and full GGN remain separate required
scope, not implied by this construction.

1. Separate prepared target/offset storage from the squared-only validator. A
   prepared problem records target and raw widths, matching data/problem identity,
   and owns copied float32 targets/offsets. Supported objective operations validate
   their schemas explicitly; preparation must not silently accept unsupported
   classes, structure or other inputs through an objective that ignores them.
2. Expose Normal base, geometry, diagonal direction and per-column least-squares
   field construction as ordinary public device functions. Geometry produces
   unweighted `[N,2]` gradient and Fisher diagonal buffers. The direction operation
   takes those buffers, mode and damping, independent of the Normal recipe. Field
   construction selects a direction column and applies objective weight once.
3. Add an explicit objective operation bundle to `DeviceRun`: validation,
   preparation, base and loss callables. The squared bundle remains the default
   for existing verifiers. The runtime calls these operations without selecting
   behavior by objective/task name. Geometry/direction/learner choices remain
   algorithm code. Normal training uses the same runtime with the Normal bundle.
4. Generalize base broadcast and raw addition from `[1]`/`[N,1]` to `[K]`/`[N,K]`.
   Introduce an immutable public mapped scalar term with finite `[1,K]` mapping.
   `propose_terms` accepts a nonempty ordered tuple and a common trial coefficient.
   Each term's prediction is mapped and added in declared order, matching saved
   model replay. The scalar `propose` convenience delegates to this transaction.
   Reject incompatible widths, forged/foreign/released trees and nonfinite maps
   before entering numerical step search.
5. `propose_terms` snapshots every caller-owned tree/mapping, computes both train
   and validation candidate raw buffers, and publishes only a complete proposal.
   A joint commit increments version once and appends both terms. Rejection returns
   the identical accepted state. Best selection occurs independently after each
   accepted transaction; an ordered outer sweep may commit zero, one or two times.

The exact function names may be refined during construction, but ownership,
observable semantics and the absence of task-name branches are acceptance rules.
An operation bundle is a small explicit dependency record, not a universal booster
base class. Author loops receive owned raw snapshots through the existing public
copy boundary; count those device copies. Do not add unsafe borrowed mutable raw
views to avoid a measured copy. Revisit copies only after correctness and a
specific cost observation justify it.

## Normal mathematics and numerical contract

Raw columns are mean and log scale. State/model raw excludes offsets; the
objective adds the two-column offset exactly once. With `r = mean - target` and
`p = exp(-2 * log_scale)`, unweighted per-row quantities are:

- NLL: `log_scale + r*r*p/2 + log(2*pi)/2`.
- Gradient: `(r*p, 1-r*r*p)`.
- Fisher diagonal: `(p, 2)`, not the observed Hessian.
- Ordinary direction: `-gradient`, with zero damping required.
- Natural direction: `-gradient / (Fisher + damping)`, damping finite/nonnegative.
- Direction-regression fields: `G = -weight * direction`, `H = weight`.
  Fisher is used in the direction, never substituted for the regression weight.
  D2 cohort fields remain independent of objective weight, including zero-weight
  rows. Loss is normalized by total objective weight, not row count.

Initialization uses normalized weights `m`, centered targets `z = y-offset_mean`
and relative precision `q = exp(-2*offset_log_scale)`. Mean is
`sum(m*q*z)/sum(m*q)`; variance is `sum(m*q*(z-mean)^2)` with no further division
by `sum(m*q)`. The minimum scale is an initialization floor only. Runtime scale,
precision, gradients and losses must remain finite/in domain; no clipping or
silent row dropping, even for zero-weight rows.

Device raw/fields/directions are float32; ordered base and metric reductions use
float64 accumulators. The independent oracle is original-row float64 mathematics,
not a CUDA emulator. Before 090-B kernels, freeze float32 comparison tolerances
and explicit representability cases. Distinguish finite non-improvement from
invalid numerical trials, and do not demand that float64 and float32 fail at the
same overflow boundary. Ordinary in-domain fixtures must agree on selected
conditions, acceptance, term counts and best/stop decisions.

Backtracking tries at most six `learning_rate * 0.5**j` coefficients, fits each
learner once per accepted geometry snapshot, and requires strict training-NLL
decrease. Fixed mode accepts a finite candidate explicitly. Joint mode fits both
columns from the same snapshot and resolves one proposal. Ordered modes `(0,1)`
and `(1,0)` recompute after each resolution and observe stopping once per outer
sweep. A rejected first parameter must still allow a second-parameter attempt.
Schema errors fail outside search; recoverable numerical errors retain their
attempt records and cannot alter accepted raw, best, version or keyed RNG.

## Verification and ownership matrix

Freeze deterministic small weighted/zero-weight/missing, validation-conflict and
D2 original-row fixtures. Cover ordinary/natural/damped, joint/forward/reverse,
fixed/backtracking, depths 0/1/2, zero rounds, zero-direction full rejection,
numeric invalid-trial recovery, nontrivial output maps and order-dependent states.
Do not label designer-authored fixtures as independent author attempts.

Compare base, gradient/Fisher/directions, named fields, candidate legality/gain,
selected conditions, leaves, both raw columns, train/validation NLL, decoded scale,
independent CRPS, versions, accepted/best term prefixes, trials and stopping.
Use distinct validation offsets/targets so a training-only check cannot pass best
selection. Preserve exact score tie behavior established in 089.

Audit 065/068 on this actual path: caller tree/mapping mutation, stale/foreign
records, released fields/raw/tree, failure after the second term or validation
evaluation, rejected proposal release, accepted/best survival after parent release,
run close with caller context still usable, same-ID retry and keyed RNG. Summary
retention must stay `O(N*K)` raw state plus model storage and `O(T)` scalar history;
no retained per-round sample arrays. Explicit diagnostic copies may be requested,
but are owned by their caller and counted separately. All 212 existing hardware
tests stay in the next regression selection unless an explicit semantic change
requires a separate cohort; never rewrite old raw artifacts.

## Cost and later obligations

Record metric synchronization count/bytes, trial losses and failures, rejected
prediction/geometry work, host decisions, uploads/downloads/device copies,
compilation policy, logical retention and measured workspace separately. Report
whole fit and prediction wall time including preparation and model export with
clearly separated cold/warm scopes. Suite duration is not a performance claim.

Original P7 remains a separate current-revision reproduction of the
[historical protocol](../benchmarks/results/foundation/20260905T193308Z-5ebd75ab/README.md):
P2 Housing hashes/splits, seeds 0/1/2, 30 rounds, depth 3, rate .05, 64 bins;
legacy CPU, legacy CUDA, default composition and independent A+B+C CUDA methods.
Its paired historical baseline and 1.2 warm-fit gate remain unchanged. Preserve
the original proper-score/coverage thresholds, repeated fits, profiling and memory
scopes. 090's tiny Normal/D2 checks do not reproduce P7 or establish E4. Preparing
the exact reproducible current package remains open; the historical README is
provenance, not evidence that a current protocol package already exists.

079 cannot close without its installed extension, fresh inference, ownership,
bounded device evidence and separate original-P7 accounting. Then 080 carries the
remaining required CUDA recipe matrix. No quality, author benefit or adoption
claim follows from any number of these development tests.

## Results and reflection

Full CPU regression passes **1371 tests**, with one Linux-only skip. Ruff and
documentation build pass; see the [verification record](../learnings/2026-09-07-v1-normal-device-design.md).

090-A adds [original-row Normal math](../tests/v1/reference/device_normal.py) and
[101 local checks](../tests/v1/test_device_normal_reference.py). The first focused
check failed at import because the new oracle did not yet exist. After construction,
the 90 three-round public-transaction cases span five fixture/depth/minimum tuples,
three direction/damping combinations, three update orders and fixed/backtracking.
The other eleven checks cover analytic/finite-difference geometry, stationarity,
CRPS, initialization floor, zero-weight domain errors, decisions, partial/full
rejection, zero rounds, a retained numerical diagnostic and blocked imports.
No production code changed. This is development conformance, not an E5 attempt.

The final weighted fixture uses one feature, depth 1, unequal weights with row 4
at zero and row 0 at 0.5, and missing values. At rate 8, ordinary and natural joint
updates reject 8 and 4, then accept 2. Both keep the initial validation-best model
despite the accepted training update. D2 has depth-1/default and depth-2/constrained
cases: the first root changes from `(0,0,False)` to `(0,1,False)` for both columns
when each child needs one unit of each independent cohort field. Conflict fixtures
cover depths 0/2. Distinct train/validation offsets are used throughout.

All three ordinary update orders produce distinct final raw arrays. For undamped
natural directions, reverse order and joint updates can agree mathematically:
the mean direction simplifies to `target - mean`, independent of scale. The
forward order still recomputes the scale direction after updating mean. The test
requires the distinguishing ordinary outcomes and permits only float64 rounding
in the natural equality; it does not require different answers from equivalent
algorithms. Versions distinguish three joint from six ordered commits.

A separate off-optimum state with target 100, raw `(0,0)`, direction `(100,5000)`
rejects numerical trials at .2 and .1, then accepts .05 under actual float64 Normal
NLL. Scale overflows first; precision underflows second. A zero mean direction
rejects all six trials before a scale update succeeds in the same ordered sweep.
These fixtures establish mathematical transaction examples. Float32 numerical
limits and CUDA retry/cleanup remain work for 090-B/C; they are not verified here.

### Numerical counterexample retained before freezing kernels

The inherited two-feature weighted fixture had a zero-weight finite extreme row.
At the second mean update (natural, damping .25, forward order), CPU histogram
scores for `(0,0,True)` and `(0,3,False)` were `1.1156968959413` and
`1.1156968959413003`. Their right/left gradient sums differ by one float64 rounding
step from accumulation order. Their full-row partitions differ on zero-weight
row 0. Observed depth-2 predictions differ by about .00925. This is not merely a
harmless child-label permutation, and no raw tolerance was widened to hide it.

The original inputs remain as `weighted_ties`, with captured G/H and a public
histogram/original-row diagnostic. It records the near-equal scores and differing
routes/leaf predictions without claiming a cross-backend winner. The primary
Normal fixture was refined before freeze: one feature, positive extreme-row mass
and depth 1 remove this ambiguity from the ordered-state comparison. Giving the
extreme row mass alone did not remove equivalent split conditions deeper in the
two-feature tree; that unsuccessful draft is the reason for the bounded fixture.
Depth-2, zero-weight and missing coverage are not dropped from the project: D2
covers depth 2 here, the numerical diagnostic is retained, and all old 212 device
regressions stay unchanged. The diagnostic is not counted as repaired conformance.

This differs from 089's identical resident G/H child swap with a contracted score
product: here the aggregates themselves differ. Before 090-B/C device acceptance,
freeze how ambiguous mathematical ties are diagnosed separately from exact ties
on identical stored statistics and well-separated winners. Do not silently add
an epsilon to `choose`, change 089 fixtures, or declare the new case GPU-verified.
Resolving this broader numerical policy remains open.

Reflection: the second algorithm justifies K-column raw state and multi-term
transactions while reusing scalar learners. It also shows why validation/best,
order semantics and numerical-domain decisions must be frozen before kernels.
Proceed with 090-B's float32-domain cases and public Normal operations; carry the
tie-policy question into that slice. Installed D2, resident transactions, cleanup,
cost, P7 and the broader 079 acceptance remain open.
