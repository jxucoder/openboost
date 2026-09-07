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

This module does **not** provide device fields, histograms, trees, transactions or
GPU training. Current recipes and NumericData remain CPU-only. Buffer ownership
is a prerequisite for future accepted/proposal state, not proof of its correctness.
