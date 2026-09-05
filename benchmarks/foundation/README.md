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
