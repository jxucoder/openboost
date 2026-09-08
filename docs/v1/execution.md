# Experimental CUDA storage ownership

`openboost.execution.ExecutionContext` owns a CUDA stream and private CuPy memory
pool. `upload`, `copy`, `export` and `release` operate on opaque `DeviceBuffer`
handles. A handle exposes shape/dtype/size, never a mutable device array. Cross-
context, forged, released or closed handles fail. Contexts are single-thread owned.

Host upload makes an independent snapshot and synchronizes before returning;
export is an explicit blocking copy. Device copies use the owning stream and
independent allocations. Release/close synchronize before invalidating handles.
Transfer bytes, synchronization counts, live logical bytes and sampled pool peaks
are exposed through a detached read-only metrics record. `max_bytes` limits the
private CuPy pool, not driver/context memory or other GPU allocations.

```python
import numpy as np
from openboost.execution import ExecutionContext

with ExecutionContext("cuda:0", max_bytes=16 * 1024**2) as execution:
    data = execution.upload(np.arange(8, dtype=np.float32))
    independent = execution.copy(data)
    execution.release(data)
    restored = execution.export(independent)
```

Requires real CUDA and CuPy; an unavailable device never selects CPU fallback.
Supported nonempty native-endian storage types are float32/64, int32/64, uint8 and
bool. Missing values are copied unchanged; semantic validation belongs to later
field/operation records. There is no external-stream or raw-array adoption API yet.

## Experimental named fields and routed histograms

`openboost.device.DeviceOperations(execution)` provides resident additive
operations. The installed implementation at `ad2f4e6` passes 33 real T4 tests,
including 21 aggregation checks and twelve storage regressions. The committed
record is `benchmarks/v1/evidence/cuda-aggregation-078/README.md`; this verifies
the declared statistics/row operations, not GPU training or speed.

```python
from openboost.device import DeviceOperations

# binned and problem are matching public CPU BinnedData and Problem objects.
# gradient_curvature is a finite float32 [N, 2] host array in problem row order.
with ExecutionContext("cuda:0") as execution:
    ops = DeviceOperations(execution)
    data = ops.prepare(binned, problem)  # explicit codes/missing/weight upload
    values = execution.upload(gradient_curvature)
    fields = ops.fields(data, values, names=("gradient", "curvature"),
                        roles=("unweighted", "unweighted"))
    weighted = ops.apply_weight(fields)
    rows = ops.rows(data)  # or explicit host positions / an owned int32 buffer
    histogram = ops.histogram(data, weighted, rows)
    host_total = execution.export(histogram.total)  # explicit diagnostic export
```

Records belong to one operations instance. Prepared data retains problem, data
and fitted-binning identities. Records cannot be constructed/replaced externally
and then passed as valid owned records. Foreign, released, closed and misaligned
inputs fail. Field schemas have unique names and explicit `unweighted`, `training`
or `independent` roles. `apply_weight` transforms only unweighted fields and rejects
reapplication. Generic additive fields can be signed; a nonnegative information
contract is explicit in `add_independent(fields, name, column, nonnegative=True)`.
That method appends a resident float32 column without changing objective weights.

Rows preserve original positional order and uniqueness, including unsorted and
empty views. Host row lists are a declared upload; resident int32 row buffers are
validated on device. Empty views use a real zero-length allocation, never a sentinel
sample. This does not add empty host uploads to the storage API.

Histogram sums have shape `[features, max(bin_counts)+1, fields]`, counts have the
first two dimensions and dtype int64, and total has one entry per field. Each
feature's missing slot is at its own bin count; higher padded slots are zero.
The kernel reduces the actual selected rows with float32 accumulation and no
floating atomic updates. Objective weight is not reapplied during aggregation.
The initial kernels are a correctness implementation, with no speed claim.

Caller-supplied field and row buffers are borrowed read-only through opaque handles.
Releasing one invalidates records that use it. Operation outputs own independent
allocations. `ops.release(record)` releases only allocations owned by that record;
closing the execution context releases all remaining device storage. A failed
allocation/validation discards the operation's partial outputs and preserves inputs.
This cleanup is not boosting acceptance/rejection or accepted-state conformance.

The owned CuPy stream is used for numba-cuda kernels. Only internally owned arrays
are adapted with `as_cuda_array(sync=False)`, as all producer/consumer work uses
that same stream; no global synchronization option is changed. See the
[Numba memory/stream contract](https://nvidia.github.io/numba-cuda/user/memory.html).
All explicit device arrays, including scratch/validation flags, use the context
pool. Driver, module/JIT and context allocations are outside that pool limit.

Execution metrics add kernel launch attempts, host kernel dispatch seconds
(including lazy JIT, not GPU elapsed time), and validation-export bytes. Finite
validation exports int32 flags per field; resident-row validation exports one flag.
Histogram execution exports two flags per field and no bulk arrays. Flag export
and scratch release synchronize explicitly and are included in storage counters.
These diagnostics are not an end-to-end fit-cost measurement.

## Experimental candidate, feasibility, route and leaf operations

Sprint 087 adds Python CUDA kernels and public operations for scalar numeric
splits. All 88 real T4 tests pass at clean `9ce790e`, including 55 new split cases
and the earlier 33 storage/aggregation cases. See the committed evidence at
`benchmarks/v1/evidence/cuda-splits-078/README.md`. The frozen comparison uses an
independent float64 original-row oracle and checks every candidate's sums, counts,
gain and feasibility, then winner, routed rows and leaves. Float32 tolerances are
rtol=1e-4/atol=1e-5; counts, row order and exact-tie winners must match exactly.

`candidates(histogram)` returns named left/right sums `[slots, 2, fields]`, int64
counts `[slots, 2]` and a bool active mask `[slots]`. Slots are padded to
`features * max(bin_counts) * 2`, ordered by feature, threshold and missing-left
False/True. `batch.key(index)` decodes metadata. Only bins observed in the full
prepared data are active, including when the histogram uses a subset of rows.

`newton_scores(batch, reg_lambda=1, split_penalty=0)` and
`feasible(batch, min_child_h=0)` are separate operations. Both require named,
once-weighted scalar gradient/curvature fields and nonnegative original row
curvature. Ordinary feasibility requires two nonempty children with positive
curvature meeting the minimum. Structurally illegal scores are zero. Legal scores,
regularization and penalties must remain finite in float32.

`child_minimum(batch, name, minimum)` tests a nonnegative independent-information
column on both children. For example, with independently appended cohort columns,
the following composes D2's constraint through the same public device boundary:

```python
batch = ops.candidates(histogram)
scores = ops.newton_scores(batch)
allowed = ops.feasible(batch)
for name in ("cohort:red", "cohort:blue"):
    allowed = ops.mask_and(allowed, ops.child_minimum(batch, name, 1))
split = ops.choose(batch, scores, allowed)
if split is not None:
    left_rows, right_rows = ops.partition(rows, split)
    left_histogram = ops.histogram(data, weighted, left_rows)
    left_leaf = ops.leaf(left_histogram)
```

This sketch belongs inside the live context above, using fields that include the
named cohorts. This public composition passes renamed/reordered cohort checks on
the installed T4 package; it does not establish independent author productivity.
`scores(batch, buffer)` and `mask(batch, buffer)` also bind explicitly supplied
resident float32/bool vectors. These bindings validate shape, finite scores and
exact batch identity. Custom masks replace ordinary feasibility; callers compose
it explicitly when needed. `choose` always excludes inactive padding and requires
strictly positive scores. Exact ties select the first lexicographic slot.

`partition` requires the exact row record used by the split's histogram. Its
children contain actual original positions in input order, including empty views.
`leaf(histogram, reg_lambda=1)` returns an owned float32 `[1]` buffer containing
`-G / (H + lambda)` for those rows; a nonpositive/nonfinite denominator fails.
Release the leaf through `execution.release`, and other records through
`ops.release`. Candidate records retain histogram, field and row dependencies;
keep those records and their borrowed buffers alive through scoring and routing.
Records and their originating batch cannot be forged or substituted.

All computation stays on the context stream. Validation flags remain explicit;
choice exports one int32 winner index and partition exports two int32 child sizes.
These are counted as `decision_export_bytes`, a subset of total export bytes.
Diagnostic exports made by tests are separate from measured operation transfers.
Allocation/validation cleanup preserves inputs but is not a boosting transaction.
Initial choice and stable routing use ordered kernels; no speed claim is made.

## Resident squared geometry and trees: bounded T4 evidence

Sprint 088 adds `openboost.device_objectives` and `openboost.device_tree`.
The real T4 run at clean `af026ef` passes all 212 checks, including all 202
original cases and ten score diagnostics. This passes the bounded scalar
correctness/residency matrix; full device scope and quality/cost gates remain open.
See `benchmarks/v1/evidence/cuda-score-symmetry-089/README.md` for raw evidence.
The earlier 14 failures at `c415755` remain recorded separately. The existing
`openboost.recipes` and NumericData remain CPU interfaces. Experimental resident
training uses the separate explicit device interfaces below.

`device_objectives.prepare(ops, data, problem)` explicitly uploads scalar targets
and offsets after checking the prepared problem identity. `base`, `broadcast`,
`gradient`, `fields` and `loss` operate on resident raw values. Base and loss use
ordered float64 reductions; raw, gradients and fields use float32. `fields` applies
weights exactly once. Unsupported target schemas and nonfinite/overflowing results
fail. `loss` exports one float64 scalar, counted in `metric_export_bytes`.

`device_tree.depthwise(ops, data, fields, binning=..., max_depth=...)` composes the
public histogram, candidate, score, mask, routing and leaf operations. Supplied
`scoring(ops, batch)`, `legality(ops, batch)` and `leaf(ops, histogram)` callbacks
replace those policies and own their parameters. Callbacks may compose named
cohort minima on device. Structural nonempty-child masking always applies;
`ops.nonempty(batch)` exposes that condition independently of curvature policy.
No maximum-leaf budget, categorical growth or custom-kernel registry is added.

New callback workspace is temporary and released within growth. Preexisting
buffers are borrowed; returned leaf values are copied into private tree storage.
The opaque DeviceTree retains immutable host topology and fitted binning, and owns
packed device topology/values independently of training records. Root positions
are generated on device, without a repeated host row-index upload. Growth uploads
only the final int32 topology; it does not download leaves or bulk training arrays.

`device_tree.predict(ops, tree, data)` returns owned float32 `[N, 1]` predictions,
excluding base and observation offsets. It checks exact fitted-binning identity,
including for validation data with different row identities. `copy` makes an
independent tree snapshot; release trees through `ops.release`. `export` explicitly
downloads node values into the existing validated CPU Tree artifact, which can be
saved and loaded without CUDA packages. All 42 objective/tree checks pass, including
the previously failing weighted/missing splits and their later leaf/prediction
assertions. Exact topology and float32 tolerances are unchanged.

Sprint 089 changes scalar score products to explicit nearest-even multiplication
through libdevice, preserving symmetry when left/right child summaries are swapped.
This correction passes real T4 validation. Ten additional device checks cover
the original weighted root, swapped summaries and adjacent-ULP ordering. They also
capture the archived and corrected kernels on identical inputs and their PTX.
The archived scorer reproduces the one-ULP difference and wrong winner; the
corrected scorer has equal scores and selects the original-row oracle's winner.
The public score/mask/choice boundaries and strict lexicographic tie rule remain
unchanged; no epsilon tie band is applied.

## Resident transactions and recipe: bounded scalar gate passed

`openboost.device_runtime.DeviceRun(ops, train, validation, run_id=..., seed=...)`
binds explicit CPU preparation and private resident problem/raw storage. Pass fitted
`binning` to reuse chosen cuts, or `bins` to fit training data once and transform
validation with those cuts. The default objective supports numeric scalar squared targets.
`initialize`, `gradient`, `fields`, `propose`, `resolve`, `raw`, `export`, `release`
and `close` are separate public operations. `add_raw` is a separate resident scalar
update operation with a finite signed coefficient and independent output storage.

Accepted/proposal records expose immutable diagnostics, with no raw buffer handles.
`run.raw(record, validation=False)` returns an independently releasable device copy.
`propose(state, tree, coefficient=...)` snapshots the tree and adds only its new
contribution to both accepted raw buffers. `resolve(..., accept=True/False)` requires
an explicit boolean decision. Rejection returns the same state. Acceptance copies
proposal raw buffers and shares only immutable private tree terms. Releasing the
caller tree, proposal or earlier state cannot invalidate a retained newer state.
Foreign/forged/released records and wrong-parent proposals fail.

Strict validation improvement selects the best prefix of accepted terms, independently
of acceptance. `run.export(state, best=True)` explicitly builds a CPU Model artifact.
The model predicts raw values; observation offsets stay separate. Each raw snapshot
costs storage until released. `run.close()` releases run-owned storage, preserving
caller-owned buffers and the execution context. Closing the context invalidates all
its buffers. These ownership contracts do not isolate hostile Python processes.

`openboost.device_recipes.squared(ops, train, validation, run_id=..., seed=...)`
composes that runtime with the public objective/tree operations. Its optional
`learner(ops, data, fields)` can supply a tree from resident custom score, feasibility
or leaf operations. It uses fixed acceptance or up to six backtracking trials with
strict training-loss improvement. StopState observes accepted validation once per
outer round, including full rejection. Keyed RNG retains run/seed/round/component/
purpose semantics; the default deterministic recipe makes no sampling claim.

The result contains its live run, final state, StopState and scalar step diagnostics.
The recipe releases superseded raw states, proposals and callback workspace, leaving
O(N) raw storage and O(T) tree storage. This is a separate experimental result,
without implicit CPU run_many integration. All 72 runtime cases pass, including
weighted/missing two-round and recipe comparisons, dedicated ownership/rollback,
policy, preparation, 24-round retention and fresh CPU-process inference checks.
The fixtures establish this bounded correctness/residency gate, not real application
quality or performance. All five approved runs are consumed; no retry is authorized.
Other required device recipes, train-many, quality, cost and author/adoption gates
remain open.

## Normal components: bounded evidence and preserved historical failures

`openboost.device_normal` now provides resident `prepare`, `base`, `geometry` and
`loss` operations for scalar targets with two raw columns (mean/log scale).
Preparation checks Normal schema and owns float32 target/offset copies. Geometry
returns two independently owned float32 `[N,2]` buffers: unweighted gradient and
Fisher diagonal. Offset-aware initialization uses normalized weights; its scale
floor is used only at initialization. Loss exports one weighted mean float64 NLL.
Row expressions and reductions use float64 intermediates on stored float32 inputs.
Scale/Fisher must be positive finite float32 and gradients finite float32, including
zero-weight rows. Numerical violations fail explicitly without clipping.

`device_objectives.broadcast` supports a `[K]` base. The separate public
`diagonal_direction(ops, gradient, metric, mode=..., damping=...)` and
`least_squares(ops, data, direction, channel)` functions compose across objectives.
The latter produces once-weighted `G=-w*z, H=w`; Fisher belongs in the direction
solve. Independent cohort information can be appended using the existing field
operations. Returned buffers and fields own their outputs; input buffers stay
caller-owned. Release buffers with the context and field records with operations.

`device_normal.objective(minimum_scale=...)` constructs an explicit
`ObjectiveOperations` bundle (validation/preparation/base/loss). Pass it as
`DeviceRun(..., objective=device_normal.objective())` to use the shared runtime.
The bundle does not itself train a model. [Run 6](../../benchmarks/v1/evidence/cuda-normal-090/README.md)
passes all 23 Normal geometry checks and reruns all 212 scalar cases successfully
at `4143d18`. The overall 383-case run fails two ordered acceptance checks, so it
does not establish full Normal conformance or quality/cost parity.

`DeviceTerm(tree, mapping)` owns an immutable float32 `[1,K]` mapping. A
`run.propose_terms(state, terms, coefficient=...)` call snapshots the entire tuple
of trees and evaluates their mapped contributions in term order. All terms share
one trial coefficient and commit atomically. A joint two-term proposal increments
version once; an ordered loop may propose one parameter, resolve it, then recompute
geometry from `run.raw(updated_state)` and `run.problem`. `validate_terms` exposes
structural checks separately so schema errors fail before numerical step search.

`map_update(ops, raw, scalar_prediction, mapping, coefficient)` is a separate public
operation. Small mappings travel as kernel scalar metadata, with no bulk upload.
Products are rounded separately before raw addition. The `[1]` identity mapping
delegates to the existing scalar addition path. Generalized `add_raw` also accepts
aligned `[N,K]` deltas. All outputs own their device storage. Runtime best-prefix
selection, proposal rejection, keyed RNG and explicit record release remain the
same contracts. Mapped model export uses the existing CPU TreeTerm format.

Ninety frozen three-round Normal transaction cases and six ownership/rollback
cases execute on T4: 94 pass and two fail. They include both update orders,
independent cohort fields and failures after partially copying/predicting terms.
The failures concern exact backtracking decisions in ordinary ordered depth-zero
conflict fixtures near a stationary base. Run 6 did not retain enough state to
identify the cause; run 7's observations below close that gap. Strict acceptance
remains implemented as a decrease in the reported training NLL;
no tolerance was changed to turn these failures into passes.

`device_recipes.normal(ops, train, validation, run_id=..., seed=...)` now composes
these components. `mode` chooses ordinary/natural direction; `damping` belongs to
the natural solve. `update` is `joint`, `forward` (mean then scale), or `reverse`.
Joint learners share one geometry snapshot and one commit. Ordered substeps
recompute geometry from the latest accepted raw state, including after rejection.
`step` is fixed or backtracking; the default is at most six strict-NLL-decrease
trials. Stopping observes accepted validation once per outer sweep.

`try_terms(run, state, terms, ...)` exposes bounded search independently of the
Normal recipe. It validates state/term structure before search, retains numerical
failures and all finite candidate losses, and releases every temporary proposal.
The caller owns the previous and returned states. Recipe history contains flat
DeviceNormalStep records, each with channels, outer round, before/after versions
and a tuple of DeviceTrial records. It keeps scalar diagnostics only; the recipe
releases superseded states, geometry and working trees. Optional `learner` uses
the same public device signature as squared and owns its tree configuration.

The 31 additional recipe checks cover frozen trajectories, zero budgets, numeric
sixth-trial recovery, partial/full rejection, patience, invalid-learner cleanup,
24 ordered rounds and fresh CPU model loading with CUDA imports disabled. All 31
pass in run 6. Twenty additional installed-D2/fresh-inference checks pass, including
nineteen saved models replayed in a separate CPU environment without CUDA or the
training extension. Timing/copy/synchronization diagnostics are retained for tiny
fits; original P7, full Normal acceptance and real-data quality/cost remain open.

Run 7 at `80740f2` repeats all 383 original outcomes and passes two added observation
cases. Both failures occur on round zero's mean update. At the accepted candidate,
reported device NLL decreases by `8.88e-16` while high-precision original-row math
at the exact stored inputs increases by `5.90e-18`. Ordinary float64 original-row
loss subtraction also reports the wrong sign. The captured gradients/Fisher match
their float32-rounded reference; float32 reduction cancellation creates a small
leaf, and rounding of total losses misclassifies its update. Transaction ownership,
version/best updates and saved inference remain correct for the measured decisions.

At run 7, Normal backtracking remained experimental: independently rounded total
losses are an unreliable comparison near stationarity. An objective-owned
loss-change operation with an explicit numerical-resolution contract was
initially implemented as a separate experimental component. The run-8 consumer
results below supersede that construction-only status.
The repository archive `benchmarks/v1/evidence/cuda-acceptance-091/`
retains both failures and the complete traces. No tolerance or original test was
changed, and successful diagnostics do not pass full Normal conformance.

`device_normal.compare(ops, problem, before, after)` compares two resident float32
`[N,2]` raw buffers in the prepared problem's row order. It returns the public
`LossChange` record: bounds, declared numerical method/reason, derived status and
an explicit `improves(min_delta)` query. The private Normal expression is shared
with CPU and uses directed double arithmetic on CUDA for its bounded Taylor and
interval evaluation. There is no host computation fallback. CPU passes do not
verify the device lowering or arithmetic.

The operation checks the same float32 scale/gradient/Fisher domain as Normal's
existing geometry on every row, including zero weights. Invalid domains raise;
finite inputs outside comparison support return unresolved. Both input buffers
remain caller-owned. Scratch is `32*N + 32` bytes beyond existing validation flags
and is released on success/failure. Only one 32-byte scalar summary plus validation
flags is exported. `comparison_calls` and `comparison_export_bytes` supplement
the existing allocation, launch and transfer counters. The run-8 operation and
ownership checks verify these transfers on a real T4.

`ObjectiveOperations(..., compare=callback)` exposes the programmable dependency.
`objective.loss_change(ops, problem, before, after)` validates the result type and
raises `NotImplementedError` when no callback is supplied; it never substitutes
subtracted reporting losses. The Normal factory supplies this operation, while
the scalar objective retains its existing contract. Agents can supply another
comparison callback without modifying the grower or runtime.

The separate 117-case GPU operation cohort covers all 106 frozen numerical inputs,
invalid domains/identities, callback independence and failures during allocation
or kernel dispatch. All 117 pass on a real T4 at `469ca0e`. Actual PTX contains all
six required directed-double operations, and both stored run-7 false improvements
are classified as worsening. See the run-8 evidence in
`benchmarks/v1/evidence/cuda-comparison-092/README.md`.

## Objective comparison consumers: bounded device evidence

`DeviceRun(..., comparison="objective")` explicitly requests objective comparison;
missing support fails before data preparation. The default `"reported"` policy
keeps historical low-level/scalar callers identifiable. `run.compare(state, proposal)`
obtains the objective change at the exact bound parent/candidate training raw.
It rejects stale, foreign, forged and released records before callback execution.
`try_terms` uses this evidence in objective mode, retains it in each finite
`DeviceTrial.comparison`, and accepts backtracking only on proved improvement.
Fixed steps can still commit finite worsening/unresolved candidates.

In objective mode, every accepted state owns an independent best validation raw
snapshot. Resolve compares with the old best and copies the selected anchor before
publishing a new state or incrementing tree references. Each snapshot costs
`4*N_validation*K` bytes. Rejection creates none; reported mode creates none.
`raw(state, validation=True, best=True)` returns an independent copy of that
anchor. `validation_problem` exposes the prepared validation record for public
comparison composition. State release and run closure release best storage.

The twelve-case consumer/ownership CUDA cohort passes in run 8. It covers
equal-score improvements, both stored run-7 false improvements, retained earlier
states, all three resolve copies, initialization/callback failure cleanup, and
explicit unresolved policies. Local preflight tests verify missing/invalid policy
rejection without CUDA.

The resident Normal recipe now selects objective comparison explicitly for joint,
forward and reverse updates. It owns a separate last-qualifying validation raw
snapshot for patience, and observes once per completed outer sweep. The last
substep in each sweep records `validation_change`; other substeps leave it None.
The recipe replaces its patience anchor on proved `min_delta` improvement even
if reporting floats are equal. It releases old anchors, observation scratch and
the final anchor on completion/failure. During training this adds one retained
`4*N_validation*K` snapshot, plus a temporary snapshot during observation.

All fifteen separate recipe CUDA tests pass in run 9 at `a7173d9`: three-anchor sequences,
equal-score anchor replacement, full rejection across every update order, zero
rounds, and failures during initial/observation copy, comparison, invalid callback
result or a later learner. Together with the twelve transaction cases, these
distinguish the new consumers; they do not replace the original 383 requirements.
Historical Normal recipe byte assertions exclude the new best snapshot and will
disagree by `4*N_validation*K`; that planned semantic difference is retained in
the historical cohort. All 383 historical requirements now have collected
counterparts, including 147 revised transaction/recipe/installed-D2 checks.
They retain original settings and tolerances and audit actual stored-input
comparisons independently. All ninety float64 reference trajectories retain their
original summaries. All 383 bound requirements pass in run 8, and all 2,548
recorded trajectory comparisons pass their independent stored-input audit.

Run 8's full revised result remains 528/529. Its forward three-anchor fixture
expects a best prefix of ten after a zero-valued tenth term; strict improvement
retains prefix nine. The test-only correction has independent CPU no-op coverage.
Run 9 then passes the corrected case and all later best-score/raw, comparison and
ownership assertions, along with its fourteen recipe siblings. All uploaded and
installed sources and pinned versions match the separate run-9 freeze.

The combined audit establishes 514 earlier passes plus fifteen new recipe passes
with identical production, completing the bounded revised coverage across two
runs. It is not a single 529/529 invocation. Historical results retain all 26
expected disagreements, and run 8's overall raw verdict remains failed. See
`benchmarks/v1/evidence/cuda-recipe-103/README.md` for the new result and source
bindings. No full Normal/application conformance, matched-quality speed or broader
CUDA recipe claim follows from these results; the split near-tie limitation remains.
