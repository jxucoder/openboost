# Fixed-slot growth: independent CPU wheel conformance

Clean source `a5bf26d3d54374e4c524375a02294f0c64abbc1c`. Fresh wheel installation
outside the repository: **7 passed**, standalone weighted demo passed, six exact
CPU predictions after uninstalling both plugins and starting a new interpreter.
The core wheel is identical to the T4 correctness and P7 value bundles:
`8c94b62773f6541a983280b5970642de46c60a35e2bed0fcdf62e9d4a9cb3077`.
Both 0.2.0 extension wheels remain unchanged from P6.

Tests cover independent Normal NLL/Fisher mathematics, bounded leaf/schedule
composition, subsequent gradients, early stopping and persistence. Public-only
imports, source hashes, actual installed paths and environment are in results.
Darwin x86_64 / Python 3.12.12 / NumPy 2.2.6 / Numba .61.2 / one thread; CPU only.
This is conformance, not independent adoption or a speed benchmark.

```sh
UV_OFFLINE=1 uv run --no-sync python examples/extensions/verify_wheels.py OUTPUT
```

Offline execution used already-cached dependencies; no package publishing.
