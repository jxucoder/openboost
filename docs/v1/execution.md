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

## Candidate, feasibility, route and leaf operations: awaiting hardware validation

Sprint 087 adds Python CUDA kernels and public operations for scalar numeric
splits. These additions have not run on a GPU. The earlier 33 passing tests verify
the implementation at `ad2f4e6`, not this new slice. The frozen comparison uses an
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
named cohorts. It is a development contract pending real-device acceptance.
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

Trees, device gradients, custom-kernel registration, transactions and GPU training
remain unimplemented. Current recipes and NumericData remain CPU-only.
