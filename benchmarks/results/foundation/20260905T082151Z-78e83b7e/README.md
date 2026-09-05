# P2 weighted CUDA correctness: pre-fix failure

- Source: `502f37fd0cc38e215a2e888ed98167191a461a04`, clean detached checkout.
- Wheel SHA-256: `83f41ea79ef6f5af2c61b662570deba29292f667788d81776423371fae4a6284`.
- Real Modal T4: **3 failed, 2 passed, 0 skipped**, pytest 9.55 seconds.
- The CLI exited 1 and retained the failure artifacts, as required.

This wheel is byte-identical to the earlier P1 wheel. Every test/config input
hash matches the explicitly authorized fixed-source bundle. The same suite
passes on [fixed source 8609f70](../20260905T082025Z-ef9e0c4b/README.md).

The native constant-Hessian hint incorrectly produces histogram sums `[4, 4]`
instead of CPU's weighted `[7, 10]`. One-round GPU raw values become `0.42` and
`-0.4`, versus the analytic/CPU `0.2625` and `-0.18181819`. Weighted Normal and
Poisson CPU/CUDA raw maximum errors are 0.55151367 and 0.09349906 respectively.
All three weighted regression tests fail; both P1 smoke cases still pass.

Lower training NLL in the incorrect path is not evidence of improved quality:
it used different Newton denominators and therefore different updates. This
is an algorithmic parity regression, not a held-out comparison.

Reproduce in a clean checkout at this source commit:

```bash
uv run --no-sync python -m benchmarks.foundation.prepare --suite correctness
uv run --no-sync modal run benchmarks/foundation/modal_app.py::foundation_correctness
```

The offline runner must reject this report. `manifest.json`, `results.json`
and `junit.xml` retain provenance, environment and the actual assertions.
