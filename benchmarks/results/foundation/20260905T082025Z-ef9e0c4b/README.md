# P2 weighted CUDA correctness: fixed source

- Source: `8609f70dd5f3cfb030d76f1358e6b53f966e0e0d`, clean.
- Wheel SHA-256: `40a5ec42661eba6f72bf340147f29d0fd2ffed37877e77b6981d78cd1d0f888b`.
- Real Modal T4: **5 passed, 0 skipped**, pytest 9.97 seconds.
- Remote function wall time: 15.65 seconds, excluding image/startup; not billed
  duration or a performance comparison.

Fixed-bin native weighted Hessian sums are `[7, 10]`, matching CPU. Analytic
Newton leaves are `2.625` and `-1.81818187`; CPU and GPU one-round raw scores
match exactly. Three-round weighted Normal and Poisson raw scores also match
exactly on these fixtures, with matching NLL (2.24100208 and 2.19981337).
Gradient comparisons and root split assertions pass. Tests include zero and
nonuniform positive sample weights.

The suite also verifies 37 installed wheel Python files, device interop,
two device objective calls and four native tree calls in the smoke fixture.
See `results.json` for environment, test output and numeric checks; `manifest.json`
for input hashes and exact CLI; `junit.xml` for mandatory-case completion.
Weighted fixtures are fully specified in the hash-pinned `test_correctness.py`;
the manifest's dataset field describes the additional smoke fixture only.

Reproduce from this source commit in a clean checkout:

```bash
uv run --no-sync python -m benchmarks.foundation.prepare --suite correctness
uv run --no-sync modal run benchmarks/foundation/modal_app.py::foundation_correctness
```

Offline validation:

```bash
uv run --no-sync python -m benchmarks.foundation.runner benchmarks/results/foundation/20260905T082025Z-ef9e0c4b
```

This establishes the narrow P2.1 weighted regression gate. It does not establish
real-data quality, general deep-tree parity, callback/evaluation transfer
boundaries, cross-device persistence or a cold/warm performance baseline.
