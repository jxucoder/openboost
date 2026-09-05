# P5.1 initial strict CUDA trainer evidence

Clean source `8c34b266958a4975e3d6770f407e019a4f20ef87`, wheel SHA256
`5dde197f412ffa9a1b0d59e445b043359c7dfef2a6e2170b9d1a849a0969e4ed`.

7 passed / 0 skipped on real T4. Actual experimental Booster.fit runs two rounds,
two parameters and nonuniform/zero weights with nonconstant coefficients.
Default/explicit LevelWiseBuilder fits match CPU raw predictions within maximum
1.1920928955078125e-7, NLL exactly for this fixture, and CRPS within 3e-9.
CPU load predictions are exact; runtime failure, wrong device output, input
mutation and invalid cached predictions exercise rollback. Legacy native dispatch
and named sample-download wrappers are blocked during measured strict fits.
40 compact array downloads / 1,120 bytes across those eight trees are recorded;
this does not audit all transfer APIs or scalar synchronization. The session also
performs device defensive input copies and compact tree verification uploads.

Pytest 38.48 s, remote function 42.96 s (validation/JIT duration, not benchmark
or billed runtime). Full environment, input hash, source and package provenance
are in manifest/results. nsys is unavailable; no profiler trace was collected.
Source hashes match the clean git revision and separate/embedded JUnit match.

This initial slice verifies Normal natural-gradient shared trainer execution.
A following test slice expands adapter-mode and external-builder coverage.
To reproduce this exact initial suite use the clean source revision:

```sh
uv run --no-sync python -m benchmarks.foundation.prepare --suite trainer
uv run --no-sync modal run benchmarks/foundation/modal_app.py::foundation_trainer
```

Later runner versions require the expanded evidence. For offline validation of
this initial artifact, use `python -m benchmarks.foundation.runner` from its
source revision. No GPU speed, held-out quality or external adoption is claimed.
