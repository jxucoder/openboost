# P4.3 leaf reduction/rule: real T4 validation

Clean source: `4da7f4bdef0d981a141a326ed00d9baa684056d8`.
Wheel SHA256: `0b6eeb2b8e047f8763ad3d4999adda50527af66c7bd9cf94dcce8c3103d43d4c`.

**5 passed / 0 skipped**: existing two smoke cases, histogram, split/routing,
and leaf rule/reduction. Pytest 23.90 s; remote function 28.17 s,
including checks/JIT, excluding image/startup. These are validation durations,
not a training benchmark or billed runtime.

Direct row sums independently verify weighted G/H and physical sample counts
for a 6-row fixture, a 4,097-row fixture (seed 109) and empty input. All sums
and counts match exactly for these dyadic inputs. Float32 Newton/clipped values
match the float64 mathematical reference at rtol=atol=1e-6. Input hashes,
dimensions, sums/counts and actual values are retained in results.json. The
top-level manifest dataset describes the pre-existing smoke fixture.

Two GPU-composed rounds with y=[2,4], weights=[1,3], lambda=1 and coefficient=1
produce raw≈3.36 with Newton leaves versus raw=1 with bound=0.5. The second
weighted gradient is approximately [0.8,-3.6] versus exactly [-1.5,-10.5].
Thus the rule changes both leaf output and the next objective input. This GPU
check composes primitives; a separate local CPU test exercises the actual
experimental Booster via a custom root builder. Neither proves the still
unassembled experimental GPU Booster.

The suite checks empty/zero curvature, nonzero-gradient/zero-denominator error,
wrong output device/dtype/finiteness, nonzero inactive values and GPU mutation
of rule inputs. Reduction/rule arrays remain CuPy arrays; non-default stream
execution passes. Named cupy.asnumpy, Numba copy_to_host and legacy leaf wrapper
calls are blocked during relevant operations. Scalar validation synchronization
is allowed; no profiler trace or whole-process zero-transfer claim. Float32
atomic order is not generally deterministic beyond these exact fixtures.

Environment: Tesla T4, Tesla T4, 580.95.05, 15360 MiB, CUDA runtime 12090,
CUDA driver API 13000, Python 3.12.1,
CuPy 13.6.0, NumPy 2.3.5,
Numba 0.63.1 / numba-cuda 0.27.0.
Requested CPU=2, RAM=8192 MiB, thread settings=2; CPU model=unknown.
Full package/image/environment provenance is in manifest/results.

Reproduce from the clean source commit:

```sh
uv run --no-sync python -m benchmarks.foundation.prepare --suite leaves
uv run --no-sync modal run benchmarks/foundation/modal_app.py::foundation_leaves
```

Validate the saved artifact offline:

```sh
uv run --no-sync python -m benchmarks.foundation.runner benchmarks/results/foundation/20260905T152639Z-62d96727
```

Boundary: default Newton rule supports L2, and clipping keeps the existing
split criterion. Builder assembly, independent installation/use and complete
GPU training/quality/cost remain later gates. No external adoption is claimed.
