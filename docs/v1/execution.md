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
operations. This implementation is awaiting the declared real-device aggregation
run; storage's earlier T4 result does not validate these kernels.

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

Trees, candidates, device gradients, custom-kernel registration, transactions and
GPU training remain unimplemented. Current recipes and NumericData remain CPU-only.
