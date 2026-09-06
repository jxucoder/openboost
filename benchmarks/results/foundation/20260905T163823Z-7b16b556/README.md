# P4.4 level-wise builder: real T4 validation

Clean source: `e02403bb0e571436e7a631f8bd5c562cac911424`.
Wheel SHA256: `46dec4c697a5dfa21d8c6bca7e17a26019fd3c3373f0dd2ad539eed3ff819450`.

**6 passed / 0 skipped**: two smoke cases, histogram, split/routing, leaf
reduction/rule and whole-tree builder. Pytest 25.22 s; remote function 29.88 s.
These are validation durations including JIT, excluding image/startup, not
training performance or billed-runtime claims.

The small whole-tree test compares all compact tree arrays and sample predictions
against an independent recursive row-mask oracle. Device views are released and
the memory pool cleared before cached prediction is checked. Default-stream,
numeric/nonmissing, L2/full-sampling and histogram-budget boundaries are tested.

Four synthetic cells use 16/4097 rows, seed 127, two Normal parameters and two
rounds with nonconstant channel coefficients, with/without bounded leaves.
Maximum CPU/CUDA raw error is 1.7881393432617188e-7; NLL and CRPS agree within
2e-5 absolute/relative tolerance. Clipping changes final predictions. Each cell
saves standard trees and reproduces raw prediction after CPU load. Exact data
hashes and individual metrics are in results.json; top-level manifest data
identifies the inherited smoke fixture.

During 17 GPU builds, the named copy spy observes exactly five compact arrays
per tree: 85 calls, 2,380 bytes total. Sample-sized and histogram-shaped arrays
are rejected by that spy and Numba copy_to_host is blocked. Scalar validation
synchronization is allowed. This is not a profiler/whole-process transfer audit.

Scope: direct GPU builder composition; experimental GPU Booster.fit remains P5.
CPU Booster can opt into this builder; its broader existing default is retained.
These training-fixture metrics establish numerical agreement, not held-out
quality, calibration, speed, cost savings or external adoption.

Environment: Tesla T4, driver 580.95.05, 15360 MiB, CUDA runtime 12090,
CUDA driver API 13000, Python 3.12.1, CuPy 13.6.0, NumPy 2.3.5,
Numba 0.63.1 / numba-cuda 0.27.0. Requested CPU=2, RAM=8192 MiB,
thread settings=2; CPU model unknown. Complete environment/image/package
provenance is retained in manifest/results. All 12 uploaded source hashes and
uv.lock match the source commit; embedded and separate JUnit reports match.

Reproduce from the clean source commit:

```sh
uv run --no-sync python -m benchmarks.foundation.prepare --suite builder
uv run --no-sync modal run benchmarks/foundation/modal_app.py::foundation_builder
```

Validate saved evidence offline:

```sh
uv run --no-sync python -m benchmarks.foundation.runner benchmarks/results/foundation/20260905T163823Z-7b16b556
```
