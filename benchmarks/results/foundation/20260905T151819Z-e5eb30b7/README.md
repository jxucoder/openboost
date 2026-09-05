# P4.2 numeric split/routing: real T4 validation

Clean source: `b75b95a372ed87c36141cdf626115f74bc454989`.
Wheel SHA256: `7ab786cab2f23b462571725ea0f11cc73988f24d0eb01b11cf929a9eaa7f5ed6`.

**4 passed / 0 skipped**: two existing smoke cases, histogram regression, and
split/routing oracle. Pytest 20.23 s; remote function 24.79 s, including checks
and JIT, excluding image/startup. These are verification durations, not a
training speed measurement or billed runtime.

Split expected values come from exhaustive direct row masks and float64 sample
sums, independently of production histograms. Weighted and zero-weight samples,
L2/unregularized settings and excessive child/gain thresholds are covered.
Features, thresholds and routed IDs match exactly; gain tolerance is
rtol=atol=1e-10 on these exactly representable inputs. Child histograms are
rebuilt from routed rows and their G/H totals checked against the rows; next
level split topology is also independently verified. This is not parent
histogram scaling. The saved input hashes/parameters/topology/gains/IDs are in
`results.json`; the top-level manifest dataset describes the existing smoke
fixture, not these split cases.

Separate exact tie and min_gain equality checks pass, including inclusive child
weight equality. Negative/zero/no-legal gains, constant features, inactive and
terminal slots, empty routing, invalid IDs/child indices and missing-bin
rejections pass. Arrays remain CuPy arrays; named full-array download wrappers
are blocked around histogram/split/routing. Scalar validation syncs are allowed;
there is no profiler trace or whole-process zero-transfer claim. The histogram
regression retains its non-default stream test; this is not a stream guarantee
for an assembled trainer.

Boundary: numeric L2 only; positive curvature required in each child, including
when min_child_weight=0. Missing/categorical builder support and end-to-end
experimental GPU training remain unimplemented. No adoption or cost advantage
can be inferred from these primitive tests.

Environment: Tesla T4 15 GiB, driver 580.95.05, reported CUDA runtime 12.9,
Python 3.12.1, CuPy 13.6.0, NumPy 2.3.5, Numba 0.63.1 / numba-cuda 0.27.0.
Requested CPU=2, RAM=8192 MiB, thread settings=2. CPU model is unknown.
Full package/image/environment provenance is retained in manifest/results.

Reproduce from the clean source commit:

```sh
uv run --no-sync python -m benchmarks.foundation.prepare --suite splits
uv run --no-sync modal run benchmarks/foundation/modal_app.py::foundation_splits
```

Validate this saved evidence offline:

```sh
uv run --no-sync python -m benchmarks.foundation.runner benchmarks/results/foundation/20260905T151819Z-e5eb30b7
```
