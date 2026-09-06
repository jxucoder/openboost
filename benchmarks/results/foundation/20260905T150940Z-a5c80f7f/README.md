# P4.1 batch histogram: real T4 validation

Clean source: `cf61611a5fc9c4d701de65aa97dae082fbc89003`.
Wheel SHA256: `19b63c3c0ea7da16bbf34334c04ca15217bab70ce6cf9fd646e87c2aad3d13a4`.

Result: **3 passed / 0 skipped** (two existing smoke tests plus the batch
histogram device/oracle test). Test execution 19.47 s; remote function 23.49 s,
including checks/JIT, excluding image build/startup. These are validation
runtimes, not training performance measurements or billed durations.

The histogram test checks a 6-row weighted/missing/constant-feature fixture,
a 4,097-row random fixture (seed 103), and an empty input. G/H are compared with
independent float64 direct sample sums and the CPU implementation. Counts match
exactly. Maximum absolute G error is 9.835e-7, H error 1.252e-6; small and empty
fixtures are exact. Inputs are hashed in `results.json` alongside dimensions,
slot counts and allocated bytes. The top-level manifest dataset describes the
existing smoke fixture; histogram fixture metadata lives in these case records.

Outputs remain CuPy arrays, with non-default stream synchronization verified.
The test blocks `cupy.asnumpy`, Numba `copy_to_host` and the legacy dictionary
histogram wrapper during aggregation. This checks named boundaries only;
scalar validity checks intentionally synchronize. No profiler/PCIe trace or
whole-process zero-transfer claim. Invalid devices/H/IDs and insufficient
returned-buffer budgets are rejected. Atomic float summation is not bitwise
deterministic across runs. No downstream GPU split/tree/quality claim follows.

Environment: Tesla T4 15 GiB, driver 580.95.05, CUDA runtime reported 12.9, Python
3.12.1, CuPy 13.6.0, NumPy 2.3.5, Numba 0.63.1 / numba-cuda 0.27.0. The pinned
base image identifies CUDA 12.4; the separately recorded runtime is authoritative
for the loaded Python stack. Requested CPU=2, RAM=8192 MiB, thread settings=2.
The host CPU model is unavailable (`unknown`); full environment is in results.

Reproduce from the source commit with a clean checkout:

```sh
uv run --no-sync python -m benchmarks.foundation.prepare --suite histograms
uv run --no-sync modal run benchmarks/foundation/modal_app.py::foundation_histograms
```

The prepare step builds a fresh wheel and hashes the precise upload allowlist.
The runner requires all three cases, wheel/source identity, installed-file
verification, actual CUDA execution and the histogram device checks. Validate
this saved result offline with:

```sh
uv run --no-sync python -m benchmarks.foundation.runner benchmarks/results/foundation/20260905T150940Z-a5c80f7f
```
