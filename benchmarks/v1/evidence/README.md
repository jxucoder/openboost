# Comparator capability evidence

These are tiny synthetic capability probes, not real-data quality or performance
results, and not OpenBoost production implementation or E0–E7 acceptance.

## Current result

`modal-capabilities-isolated.json` records a real Tesla T4 (15,360 MiB), driver
580.95.05, Python/package inventory, CUDA runtime/driver and source/lock hashes.
Each task/library/device executes in a fresh process with a 90-second wall limit.
The Modal container has two CPUs and 8,192 MiB host memory. Exact invocation:

```bash
uv run --no-sync modal run benchmarks/v1/modal_preflight.py::main
```

Across A1–A11, CPU has 29 passing and four unsupported cells; CUDA has 28 passing
and five unsupported cells. Unsupported built-ins: XGBoost A11; LightGBM A10/A11;
CatBoost A8, plus CatBoost A10 on CUDA. They do not count as quality failures or
OpenBoost support. No real ranking dataset is implied by the synthetic A4 probe.

The CUDA LightGBM build uses the hash-locked 4.7.0 source distribution, GCC and
`USE_CUDA=ON`, targeting T4 architecture 75. The standard wheel did not support
CUDA. Container base digest and build commands are in `../modal_preflight.py`.
Build-isolation dependencies/native library hashes are not fully frozen yet;
this evidence cannot stand in for a final reproducible performance environment.
The raw isolated artifact records the exact probe hash but not its harness hash.

`cpu-capabilities.json` is the corrected macOS CPU probe. `adapter-cpu.json`
checks native categories, missing values and nonunit-weight response in the
three libraries, plus external rate/exposure and XGBoost base-margin persistence.
`secondary-cpu.json` checks weighted NGBoost Normal scoring/persistence, three
GLMs, a fixed-scale linear AFT fit and synthetic global-formula identification.
Their package/source metadata is embedded. Reproduce with the hash-locked CPU
interpreter and two numerical threads:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 build/v1-env/bin/python -m benchmarks.v1.capability_smoke --device cpu --output /tmp/cpu-capabilities-replay.json
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 build/v1-env/bin/python -m benchmarks.v1.adapter_smoke
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 build/v1-env/bin/python -m benchmarks.v1.secondary_smoke
```

## Retained failures

- `cpu-capabilities-initial.json`: an invalid CatBoostRanker prediction keyword;
  the corresponding initial source snapshot is retained.
- `modal-capabilities-initial.json`: stock LightGBM wheel CUDA rejection,
  CatBoost GPU AFT rejection, and XGBoost CPU/GPU inference differences under
  an incorrectly shared strict same-device tolerance. Original probe/harness
  snapshots are retained. The corrected check reloads on the same device first,
  then separately applies the existing E1 float32 cross-device tolerance.
- `lightgbm-cuda-build-initial.log`: missing Clang in the initial native build.
- `lightgbm-cuda-native-abort.log`: CUDA build succeeded, but sequential probes
  in one process caused a native illegal-memory-access abort. Probe/harness
  snapshots preserve that attempt. Fresh-process isolation passed all supported
  cells; it does not establish the root cause or safe mixed-library context reuse.

Private Modal run links are omitted from retained build logs. Source snapshots
are historical evidence, not imported code. Failure records are never overwritten
with successful reruns. Timings inside preflights are diagnostic only.

## Py-Boost

`pyboost-cuda.json` records Py-Boost 0.5.2 weighted scalar and two-output MSE
training on the same T4 environment; JSON dump/load prediction errors were zero.
Reproduce with `uv run --no-sync modal run benchmarks/v1/modal_preflight.py::pyboost`.
This does not verify Py-Boost classification or custom-objective task arms.
`pyboost-cuda-initial.json` and its harness snapshot preserve the first attempt:
`verbose=0` caused integer modulo by zero because the option is an interval.
The corrected probe uses the documented default interval of 10.
