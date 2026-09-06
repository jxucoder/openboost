# Installed structural-result and scheduling evidence

Parent b99376d plus the dirty source changes recorded by the wheel/source hashes.
This foundation revision resolves Sprint 041's rejected OrderedResult; the older
failure artifact remains intact. No extension algorithm source change was needed.

- scheduler-checks.json: M=1/8/32 built-in/ordered runs, expected state identities,
  stop rounds, predictions, and failure/reorder/regroup/retry checks.
- ordered-checks.json: six ordered cases now match direct and scheduled execution.
- ordered-expected.json: independent mathematical reference traces.
- d2.json, d3.json, ordered-0.json through ordered-5.json: saved raw inference models.
- checks.json: rerun D2/D3 mathematical checks and predictions.
- manifest.json: source/reference/wheel/artifact hashes, environment, exact
  commands/cwds and installed verification status.

Reproduce with `uv run --no-sync python examples/v1_extensions/verify.py OUTPUT_DIR`
using cached offline dependencies, Python 3.12 and NumPy 2.3.5. The disposable
environment is removed; absolute command-log paths identify this particular run.
These are deterministic sequential development checks, not formal E5, full D5
author evidence, batching performance, real selection quality or adoption.
