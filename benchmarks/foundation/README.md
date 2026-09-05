# Foundation GPU evidence

This entrypoint is isolated from `tests/modal_gpu_tests.py`: the legacy app
registers source-mounted jobs and broad dependencies. Loading it would defeat
the installed-wheel boundary and build unrelated images. Existing jobs remain
available; this app uploads only its allowlisted bundle with automatic source
inclusion disabled.

From a clean committed checkout:

```bash
uv run python -m benchmarks.foundation.prepare
uv run modal run benchmarks/foundation/modal_app.py::foundation_smoke
```

The bundle is generated in ignored `build/foundation/`. Stale/dirty source,
modified bundle files, failing/missing/skipped/duplicate required tests,
timeouts, incorrect wheel provenance and silent objective fallback fail the
command. Results are written before validation so failed jobs retain evidence.
An image-build failure before the local entrypoint starts is reported by Modal
itself and must be recorded separately; it is not a completed GPU test.

The smoke uses one T4, two CPU cores, 8 GiB requested memory, one container,
no application retries, a 300-second remote function timeout and a 240-second
pytest subprocess limit. Platform startup/build/restarts are outside this
execution timing. No cost or scaling claim follows from the smoke.

The two mandatory cases check installed wheel contents, CuPy/Numba pointer
sharing and owner lifetime with a real kernel, and a two-round Normal fit
whose gradients and native tree calls actually run on device. Private imports
and spies here are test instrumentation, not the public extension examples
planned for P6. The small dataset is training-only; NLL is a finite-value
sanity check, not evidence of held-out quality or complete CPU/CUDA parity.

Offline revalidation:

```bash
uv run python -m benchmarks.foundation.runner benchmarks/results/foundation/RUN_ID
```

Refresh the hash-locked requirements only when changing the environment:

```bash
uv export --locked --extra cuda --extra test --no-dev --no-emit-project --prune jax --prune jaxlib --no-annotate --no-header -o benchmarks/foundation/requirements.txt
```

Markers remain in the export and are evaluated on Linux. JAX is not required
for these NumPy/CuPy/Numba smoke cases. Record the exact installed versions in
each run. The CUDA base digest is Linux amd64 CUDA 12.4.0 devel Ubuntu 22.04,
resolved from NVIDIA's registry; the image's Python patch version is recorded
at runtime. The uv image installer is pinned to 0.12.1.

P2 weighted correctness regression (includes the smoke cases):

```bash
uv run python -m benchmarks.foundation.prepare --suite correctness
uv run modal run benchmarks/foundation/modal_app.py::foundation_correctness
```

This currently covers fixed-bin weighted histograms/Newton predictions and
three-round weighted Normal/Poisson CPU/CUDA comparisons. It does not yet
constitute the complete P2 baseline gate.

## Remaining P2 execution boundaries and baseline

The `boundaries` suite adds eight real-device tests to the five correctness
cases: custom/exposure/generic fallback, device-error rollback, row/column
sampling preflight and Normal/Poisson callback/eval/cross-device persistence.

```bash
uv run python -m benchmarks.foundation.prepare --suite boundaries
uv run modal run benchmarks/foundation/modal_app.py::foundation_boundaries
```

The `baseline` suite includes all 13 boundary/correctness cases and stops on
any failure before entering its real-data matrix. Download the public
California Housing archive once into the ignored data directory:

```bash
uv run python -c 'from benchmarks.foundation.dataset import fetch; fetch("build/foundation_data/cal_housing.tgz")'
uv run python -m benchmarks.foundation.prepare --suite baseline
uv run modal run benchmarks/foundation/modal_app.py::foundation_baseline
```

The archive hash is sklearn 1.8.0's published hash, and the transformation was
checked against that installed sklearn loader. `housing.json` freezes the
archive/array/split hashes. Float32 conversion follows the original per-row
ratio transformations; no learned scaling is applied. Each seed 0/1/2 uses a
60/20/20 train/validation/test permutation. Only training data fits bin edges.

The predefined Normal model uses 30 rounds, depth 3, learning rate .05 and 64
bins. For every CPU/CUDA × seed × no-eval/eval cell, a fresh Python subprocess
and empty NUMBA_CACHE_DIR measure first and repeated fit. Fits include binning,
objective math, copies and compilation; imports, dataset loading and container
startup are excluded. Prediction includes test binning. Small path-counting
wrappers are included in timings. Repeated CUDA predictions use rtol=2e-5 /
atol=2e-6 because floating-point atomic reductions need not be bit-identical.

CPU/CUDA quality gates are frozen before collection: per seed/mode NLL absolute
difference <= .01 * max(1, abs(CPU NLL)), CRPS regression <= 1%, and coverage90
absolute difference <= .01. These real-data gates cannot override the strict
micro-oracle tests. Three seeds do not establish statistical significance.

The baseline function is limited to one T4, two CPU cores, 8 GiB requested
memory, no retries and 1800 seconds. Its pytest subprocess is capped at 1740
seconds and each matrix worker at 150 seconds. Failure output and partial
completed cells are retained. Exact CPU model is recorded if /proc exposes it.
The trainer transfer counter is partial, and is not total PCIe traffic or a
zero-transfer assertion. GPU memory peaks and scaling remain later gates.

Status: locally validated harness; P2.2/P2.3 GPU execution remains pending.
