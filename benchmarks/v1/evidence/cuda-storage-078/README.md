# Sprint 078: First real CUDA storage-ownership result

Clean source `77aa105`. Twelve installed-wheel tests pass on a Tesla T4; all 26
source hashes and the raw pytest.log hash verify. No skipped test or CPU array
emulation is counted as device evidence.

Verified behavior: owned host snapshots, independent device copies, detached
exports, dtype and NaN/sign-bit preservation, same-stream copy ordering, caller
stream restoration, cross-thread rejection, foreign/forged/released/closed handle
rejection, allocation-budget failure and subsequent usable state. Public metrics
are detached read-only records. Current CPU data/recipes remain unchanged.

The largest fixture is 8192x32 float32. Its logged counters are:

| Counter | Bytes/count |
| --- | ---: |
| Host-to-device upload | 1048576 |
| Device-to-device copy | 1048576 |
| Device-to-host exports | 2097152 |
| Peak live logical storage | 2097152 |
| Sampled private-pool peak | 2097152 |
| Synchronizations at logged checkpoint | 4 |

The context pool limit is 16 MiB. Its peak is not total process/driver/device
memory, and the counters cover this storage API only. The 0.96-second pytest
report is a test duration, not a training or transfer-performance benchmark.

## Environment and budget

Python 3.12.1, CuPy 13.6.0, NumPy 2.3.5; Tesla T4 with 15360 MiB and driver
580.95.05. The image is based on CUDA 12.6.3-devel Ubuntu 22.04, but observed
runtimeGetVersion is **12090** and driverGetVersion is **13000**. Record actual
loaded runtime separately from the image tag; do not claim runtime 12.6 parity.
The resolved image identity and exact CLI are in manifest.json.

One T4 function requested two CPUs and 8192 MiB, capped at 900 seconds; its tests
had 600 seconds and zero retries. This is **run 1 of the two-run 085 allowance**.
No failed allocation or retry occurred. One further run capped at 900 seconds
remains; preregister that run's concrete fixtures before dispatch. Image build,
container startup and CUDA context overhead are not a measured cost result here.

```bash
uv run --no-sync python -m benchmarks.v1.cuda_storage_preflight /tmp/cuda-storage
```

The harness itself is not a global budget scheduler; the sprint record owns the
remaining-run allowance. Do not repeat this command as an uncounted retry.

## Reflection and next boundary

We now have a verified device storage boundary rather than only a CPU design.
Opaque handles and explicit copy/export avoid treating mutable CuPy arrays as
immutable accepted state. The existing public CPU path continues to reject CUDA
training requests; no CPU fallback has been introduced.

Next implement named device fields and routed histogram reductions, carrying
independent cohort information for D2. Accepted/proposal state integration,
candidate selection, trees, two-round training, CPU artifact export, D2 device
execution and practical cost remain open. This storage result is not scalar
boosting, full ownership/transaction conformance, E1/E4, P7 or authoring evidence.
