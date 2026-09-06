# Fixed-slot growth: real CUDA correctness

Clean source `81acf0bf3b3e2ff5da437cdafe1b1c94169acfac`; core wheel
`8c94b62773f6541a983280b5970642de46c60a35e2bed0fcdf62e9d4a9cb3077`.
Real T4: **7 passed / 0 skipped**. The implementation changes only fixed-slot
split application/frontier construction in LevelWiseBuilder, avoiding boolean
compaction while retaining all validation and numerical primitives.

The suite checks histogram/split/leaf original-row oracles, complete trees,
weighted two-channel CPU/CUDA composition, bounded leaves, saved CPU predictions,
and actual strict GPU Normal/Poisson fits in ordinary/natural modes. It also
checks explicit external builder dispatch, mutation/invalid-statistics/cached
prediction failures and rollback. Maximum adapter raw CPU/CUDA error 1.19e-7;
actual Normal NLL agrees and CRPS differences stay below 7e-9. No fallback.

Named transfers remain at the existing expected counts (85 compact arrays in
the builder test, 40 in the strict trainer test). This does not constitute a
complete transfer trace. nsys is unavailable. No performance claim follows from
correctness duration; the frozen P7 value matrix is the next separate check.

T4, driver 580.95.05; Python 3.12.1, CUDA runtime/package/CPU/RAM/thread details
are in results.json. Pytest 38.78 s, function 42.88 s; 300-second limit, no retries.
Only wheel and allowlisted tests/oracles/manifest uploaded; no source mount.

Reproduce from the source revision:

```sh
uv run --no-sync python -m benchmarks.foundation.prepare --suite trainer
uv run --no-sync modal run benchmarks/foundation/modal_app.py::foundation_trainer
```

Validate offline:

```sh
uv run --no-sync python -m benchmarks.foundation.runner benchmarks/results/foundation/20260905T193001Z-cece3e8f
```
