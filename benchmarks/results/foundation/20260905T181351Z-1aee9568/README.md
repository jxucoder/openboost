# P6 independent GPU wheels: installation and conformance

Clean source: `43fcda31b70a5970b0e588c1d0e4b836707af90e`. **3 passed / 0 skipped** on real T4:
two existing smoke cases plus the independent GPU package test, covering eight
composed CPU/CUDA cells and the public GPU demo. After pytest exits, both example
packages are uninstalled and a new interpreter verifies **nine exact CPU raw
prediction roundtrips**, with neither plugin importable.

Wheel SHA256:

- OpenBoost: `5dde197f412ffa9a1b0d59e445b043359c7dfef2a6e2170b9d1a849a0969e4ed`
- normal_fisher 0.2.0: `47ddff3a87c85b3f4bb187d7b53b650180a6072983a20bc48f0fb8d379da4e1b`
- bounded_leaves 0.2.0: `dc236c0f207f7fdf0a52e6bdb8e7330f1ffbb6f63e082bb5226c49294a0f02c3`

Only three wheels, test/demo files and a manifest were uploaded; repository
source was not mounted. Both installed extension modules resolve in site-packages
and match their wheel's Python file byte-for-byte. Package source hashes and
public-import checks accompany the wheel hashes. The core wheel is unchanged
from P5: these methods required no core edit or private OpenBoost import.

Independent float64 finite-difference weighted NLL and analytic Fisher verify
GPU gradients/curvature on a small nonuniform/zero-weight fixture. Device loss
and constrained parameters agree with reference values. Host inputs in a CUDA
step and extreme-scale underflow/overflow are rejected. The larger experiment
uses 16/4097 rows, seed 149, two channels, two rounds and nonuniform/zero weights.
Each size runs objective only, objective+schedule, objective+bounded leaf, and
all three combined, against CPU fits with the same configuration.

Maximum CPU/CUDA raw error: 3.5762786865234375e-7.
Maximum NLL difference: 2.3084373301784922e-8.
Maximum CRPS difference: 2.8600352086627367e-8.
Per-cell values and exact generated-data hashes are retained in results.json.
Clipping bounds actual stored leaves and changes next-round mean gradients;
the nonconstant schedule changes predictions. The 64-row public demo runs in a
separate process with actual_device=cuda and coefficients mu=[0.2,0.1],
log_sigma=[0.1,0.05]. All nine GPU-trained saved models survive plugin removal.

The named cupy.asnumpy spy permits only five compact tree arrays per tree:
160 calls / 4,480 bytes across eight instrumented GPU fits (32 trees). It excludes
the separate demo process and explicit reference conversions. P5 defensive device
copies, compact verification uploads and scalar synchronization remain. This is
not a profiler trace or a whole-process transfer audit.

The first attempt failed in the public-demo subprocess because top-level training
could be re-entered by spawned CUDA discovery workers. The corrected demo uses
a main guard; a CPU import test checks __mp_main__ causes no training/files.
[The original failure](../20260905T180943Z-e145df4a/README.md) is retained. No data,
seed or parity tolerance was changed to obtain the passing result.

Environment: Tesla T4, nvidia-smi `Tesla T4, 580.95.05, 15360 MiB`,
CUDA runtime 12090, driver API 13000, Python 3.12.1,
CuPy 13.6.0, NumPy 2.3.5,
Numba 0.63.1 / numba-cuda 0.27.0.
Requested CPU=2, RAM=8192 MiB, threads=2; CPU model unknown. Pytest 36.81 s;
remote function including uninstall/inference 43.29 s. These validation/JIT
wall times are not training benchmarks or billed runtimes.

Reproduce from the clean source revision:

```sh
uv run --no-sync python -m benchmarks.foundation.prepare --suite extensions
uv run --no-sync modal run benchmarks/foundation/modal_app.py::foundation_extensions
```

Validate saved evidence offline:

```sh
uv run --no-sync python -m benchmarks.foundation.runner benchmarks/results/foundation/20260905T181351Z-1aee9568
```

All source/wheel/lock hashes and JUnit copies were verified; private URL/local-path
scan passed. This establishes technical extension installation/conformance (G3),
not third-party adoption (G5), held-out quality or engineering cost advantage (G4).
Next is P7's preregistered matched-quality timing/memory/value evaluation.
