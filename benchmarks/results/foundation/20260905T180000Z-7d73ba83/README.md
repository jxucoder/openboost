# P5 strict CUDA trainer: complete declared adapter coverage

Clean test source: `42998587ff9e2461d8f1935643f18d8dc20864cb`.
Implementation: `8c34b26` (unchanged by the test coverage commit).
Wheel SHA256: `5dde197f412ffa9a1b0d59e445b043359c7dfef2a6e2170b9d1a849a0969e4ed`.

**7 passed / 0 skipped** on real T4: existing smoke and P4 primitives/builder,
plus actual strict experimental Booster.fit. Pytest 44.20 s, remote function
49.67 s. These include validation/JIT work, exclude image/startup, and are not
training benchmark or billed-runtime claims.

The actual trainer runs CPU initialization/binning, then CuPy objective inputs,
statistics and raw updates. Two rounds, two parameters, nonuniform/zero weights
and a nonconstant schedule exercise both the default builder and a duck-typed
external builder delegating through the public LevelWiseBuilder API. The reports
identify LevelWiseBuilder versus ExternalBuilder; legacy native dispatch and
named sample-download wrappers are blocked during those fits.

Maximum raw CPU/CUDA error across both scheduled fits and four adapter cells is
1.1920928955078125e-7. All Normal/Poisson × ordinary/natural gradient modes pass
at rtol=atol=2e-5. The scheduled Normal fixture matches weighted NLL exactly and
weighted CRPS within 3e-9. Exact target/data hashes, each error and metric are in
results.json. These are training-fixture numerical checks, not held-out quality.

Saved GPU-trained models predict exactly after CPU loading. Broken kernels,
wrong-device outputs and input mutation preserve the previous fitted state;
inconsistent cached predictions fail and preserve predictions. Additional
negative cases reject float64 statistics, negative Hessians and aliased outputs.
CPU preflight checks cover unsupported features/parameters/evaluation and full
CPU fallback before device execution.

The named spy records 40 compact downloads / 1,120 bytes for the two scheduled
fits (eight trees). This count excludes the additional adapter/error fixtures.
The session deliberately makes device copies of plugin inputs, including binned
values per tree, and uploads compact finalized trees for independent cache
verification. Scalar synchronization is allowed. **nsys is unavailable; no
profiler trace was collected.** This is not a whole-process zero-transfer or
performance claim. Optimization and matched-quality cost remain later work.

Environment: Tesla T4, nvidia-smi `Tesla T4, 580.95.05, 15360 MiB`,
CUDA runtime 12090, driver API 13000, Python 3.12.1,
CuPy 13.6.0, NumPy 2.3.5,
Numba 0.63.1 / numba-cuda 0.27.0.
Requested CPU=2, RAM=8192 MiB, thread settings=2; CPU model unknown. Full image,
package and environment provenance is retained in manifest/results.

Reproduce from the clean test source:

```sh
uv run --no-sync python -m benchmarks.foundation.prepare --suite trainer
uv run --no-sync modal run benchmarks/foundation/modal_app.py::foundation_trainer
```

Validate saved evidence offline:

```sh
uv run --no-sync python -m benchmarks.foundation.runner benchmarks/results/foundation/20260905T180000Z-7d73ba83
```

All uploaded source hashes and lock hash matched git objects; embedded and separate
JUnit matched; private URL/local-path scans passed. Initial narrower evidence is
preserved alongside this artifact. Next: real installed P6 GPU extension packages.
Strict CUDA eval/callbacks/early stopping remain unsupported; no external adoption
or speed advantage is claimed.
