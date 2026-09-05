# P1 single-T4 smoke

Source: `3101a4486034ff1848e764413ef50751a6b29678` (clean).
Wheel SHA256: `83f41ea79ef6f5af2c61b662570deba29292f667788d81776423371fae4a6284`.

Result: **2 passed**, no skipped tests. Local Modal command exited 0; offline
evidence validation also passed. This run used one Tesla T4 (15,360 MiB),
driver 580.95.05, Python 3.12.1, NumPy 2.3.5, Numba 0.63.1, numba-cuda 0.27.0,
and CuPy 13.6.0. CuPy reported runtime 12.9 and driver API 13.0; the pinned
base image contains the CUDA 12.4.0 toolkit. These are distinct version fields.

The wheel's 37 Python files matched the installed site-packages contents.
CuPy/Numba shared the same device pointer, and a real CUDA kernel succeeded
after releasing the original CuPy reference. The two-round Normal fit made
2 device objective calls and 4 native GPU tree calls; host fallback would
fail the test. Dataset hash and full environment are in results.json.

Pytest reported 9.16 seconds; remote function wall time was 14.66 seconds,
including environment checks and first-use compilation. Neither is billed GPU
duration or a performance benchmark. Image build/startup are excluded; one
smoke job was submitted, with no application retries. Actual billing was not
queried. Small grids generated expected under-utilization warnings.

Requested resources were 2 CPU cores and 8 GiB RAM. The container separately
reported 18 visible CPUs and host memory; these are not the requested limits.
CPU architecture was reported as x86_64; an exact physical CPU model was not
collected. P2 performance provenance should include it when exposed.

Reproduce from this source commit in a clean checkout:

```bash
uv run python -m benchmarks.foundation.prepare
uv run modal run benchmarks/foundation/modal_app.py::foundation_smoke
```

Validate the saved report without allocating a GPU:

```bash
uv run python -m benchmarks.foundation.runner benchmarks/results/foundation/20260905T080803Z-c5ced00e
```

This establishes the test/deployment boundary and this small GPU execution
path. It does not establish full gradient/split/leaf CPU parity, weighted
correctness, transfer-free training, held-out quality, or a speed advantage.
